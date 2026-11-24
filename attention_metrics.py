"""Interpretability analysis of trained INTR runs.  Written by our team.

For every test image classified correctly by ALL given models (the paper's protocol for
insertion/deletion), using the last decoder layer and the true class's query:

1. Insertion / deletion (Petsiuk et al. 2018; paper Table 6). Grid cells are ranked by an
   explanation map; cells are deleted from the image (replaced by the mean colour) or
   inserted into a blurred copy, 1/STEPS of the cells at a time; AUC of the judge's
   probability for the true class. Insertion: higher = better, deletion: lower = better.
   With --judge (R3 ResNet-50) the judge is shared, as in the paper, and Grad-CAM on R3 is
   compared too. Without --judge each INTR model judges its own map (the original RISE
   definition). Maps: each INTR run (average over heads, as in the paper) and a random
   ranking (reference).
2. Attention on the pet: share of attention mass on pet pixels (Oxford trimaps; the 14
   empty trimaps are skipped), vs the share a uniform map would give.
3. Head diversity (multi-head runs): mean pairwise IoU of the heads' top-10% cells.
4. Class distinctiveness: IoU of the top-10% cells of the true class's map and the
   runner-up class's map (paper's claim: classes must look in different places).
5. Optional test-resolution check (--resolutions): top-1 of each INTR run at other test sizes.

  python attention_metrics.py --intr outputs/R1/intr_pet_R1.pth outputs/R2/intr_pet_R2.pth \
      outputs/FC/intr_pet_FC.pth --judge outputs/R3/resnet50_pet_R3.pth --data_root data \
      --out_dir analysis --resolutions 320 384 448 512
"""
import argparse
import json
import random
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

from intr_pet.baseline import MaskedResNet50  # noqa: E402
from intr_pet.data import load_split, make_transforms  # noqa: E402
from intr_pet.inference import heatmap_overlay  # noqa: E402
from intr_pet.model import load_weights  # noqa: E402
from util.misc import nested_tensor_from_tensor_list  # noqa: E402  (official)

STRIDE = 32


def topk_mask(m, frac=0.10):
    k = max(1, int(round(frac * m.size)))
    thr = np.sort(m.ravel())[-k]
    return m >= thr


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def auc(curve):
    y = np.asarray(curve)
    return float(((y[:-1] + y[1:]) / 2).mean())


@torch.no_grad()
def intr_forward(model, x, device):
    out, enc, hs, att, _ = model(nested_tensor_from_tensor_list([x.to(device)]))
    h, w = enc.shape[-2:]
    a = att[-1, 0].float()                                     # [heads, classes, h*w]
    return out["query_logits"][0].float().cpu(), a.reshape(a.shape[0], a.shape[1], h, w).cpu().numpy()


def gradcam(judge, x, label, device):
    x = x.to(device)[None]
    mask = torch.zeros(1, x.shape[2], x.shape[3], dtype=torch.bool, device=device)
    with torch.enable_grad():
        feat = judge.features(x)
        feat.retain_grad()
        logit = judge.pool_classify(feat, mask)[0, label]
        judge.zero_grad(set_to_none=True)
        logit.backward()
    w = feat.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * feat).sum(1))[0]
    return cam.detach().cpu().numpy()


def cell_mask_to_pixels(cells, H, W):
    """[h, w] float mask of grid cells -> [H, W] pixel mask (each cell = 32x32 pixels)."""
    t = torch.from_numpy(cells.astype(np.float32))[None, None]
    up = F.interpolate(t, scale_factor=STRIDE, mode="nearest")[0, 0]
    up = F.pad(up, (0, max(0, W - up.shape[1]), 0, max(0, H - up.shape[0])))
    return up[:H, :W]


def resnet_judge(model, device):
    """R3 ResNet-50 as the shared judge (paper's protocol). batch [N,3,H,W] -> probs [N,C]."""
    @torch.no_grad()
    def judge_fn(batch):
        mask = torch.zeros(batch.shape[0], *batch.shape[2:], dtype=torch.bool, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            logits = model.pool_classify(model.features(batch), mask).float()
        return torch.softmax(logits, 1)
    return judge_fn


def intr_judge(model, device, chunk=12):
    """An INTR model judging its own maps (original RISE definition: the explained model)."""
    @torch.no_grad()
    def judge_fn(batch):
        out = []
        for part in batch.split(chunk):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                lg = model(nested_tensor_from_tensor_list(list(part)))[0]["query_logits"].float()
            out.append(torch.softmax(lg, 1))
        return torch.cat(out)
    return judge_fn


@torch.no_grad()
def insertion_deletion(judge_fn, x, label, rank_map, steps, device, blurred):
    """Probability curves of the judge while cells are inserted / deleted in rank order."""
    H, W = x.shape[1:]
    order = np.argsort(-rank_map.ravel(), kind="stable")
    n = order.size
    ins, dele = [], []
    for s in range(steps + 1):
        keep = np.zeros(n, dtype=np.float32)
        keep[order[: int(round(s / steps * n))]] = 1.0
        m = cell_mask_to_pixels(keep.reshape(rank_map.shape), H, W)
        ins.append(blurred * (1 - m) + x * m)                  # reveal top cells
        dele.append(x * (1 - m))                               # 0 = mean colour after normalisation
    batch = torch.stack(ins + dele).to(device)
    p = judge_fn(batch)[:, label].float().cpu().numpy()
    return p[: steps + 1], p[steps + 1:]


def pet_cells(trimap_path, H, W, grid):
    t = np.asarray(Image.open(trimap_path).resize((W, H), Image.NEAREST))
    pet = torch.from_numpy(np.isin(t, (1, 3)).astype(np.float32))[None, None]
    gh, gw = grid
    pet = F.pad(pet, (0, gw * STRIDE - W, 0, gh * STRIDE - H)) if gh * STRIDE >= H and gw * STRIDE >= W else pet
    frac = F.adaptive_avg_pool2d(pet, (gh, gw))[0, 0].numpy()
    return frac, float(pet.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intr", nargs="+", required=True, help="final INTR weight files")
    ap.add_argument("--judge", default=None,
                    help="R3 ResNet-50 weight file (shared judge, as in the paper). "
                         "Without it each INTR model judges its own maps.")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_dir", default="analysis")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--max_images", type=int, default=None, help="cap on evaluated images")
    ap.add_argument("--resolutions", type=int, nargs="*", default=[])
    ap.add_argument("--fig_breeds", nargs="*", default=["Chihuahua", "British Shorthair", "Russian Blue"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include_wrong", action="store_true", help="debug only: skip the all-correct filter")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = args.device
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)
    df, classes = load_split(root)
    names = [c["breed"] for c in classes]
    test = df[df.split == "test"].reset_index(drop=True)

    runs = {}
    for p in args.intr:
        m, a, ck = load_weights(p, dev)
        name = Path(p).stem.replace("intr_pet_", "")
        runs[name] = dict(model=m, heads=a.dec_heads, preset=ck.get("preset", "colab"))
    judge = None
    if args.judge:
        jck = torch.load(args.judge, map_location="cpu", weights_only=False)
        judge = MaskedResNet50(len(classes), imagenet=False)
        judge.load_state_dict(jck["model"]); judge.to(dev).eval()
        shared_fn = resnet_judge(judge, dev)
        print("insertion/deletion judge: R3 ResNet-50 (shared, as in the paper)")
    else:
        self_fns = {r: intr_judge(R["model"], dev) for r, R in runs.items()}
        print("insertion/deletion judge: each INTR model judges its own maps (no R3 given)")
    preset = next(iter(runs.values()))["preset"]
    tf = make_transforms("test", preset)

    # ---------------- optional: accuracy at other test resolutions (whole test set)
    res_rows = []
    for size in args.resolutions:
        tfs = make_transforms("test", preset, test_size=size)
        correct = {r: 0 for r in runs}
        for _, row in test.iterrows():
            x, _ = tfs(Image.open(root / row.rel_path).convert("RGB"), None)
            for r, R in runs.items():
                logits, _ = intr_forward(R["model"], x, dev)
                correct[r] += int(logits.argmax() == row.class_id)
        for r in runs:
            res_rows.append(dict(run=r, test_short_side=size, top1=round(100 * correct[r] / len(test), 2)))
        print("resolution", size, {r: res_rows[-len(runs) + i]["top1"] for i, r in enumerate(runs)}, flush=True)
    if res_rows:
        pd.DataFrame(res_rows).to_csv(out / "resolution_check.csv", index=False)

    # ---------------- per-image interpretability metrics
    order = list(range(len(test)))
    random.shuffle(order)
    rows, curves = [], {}
    kept, t0 = 0, time.time()
    fig_cache = {}                                    # (breed) -> list of (img, {run: attn})
    for i in order:
        if args.max_images and kept >= args.max_images:
            break
        row = test.iloc[i]
        img = Image.open(root / row.rel_path).convert("RGB")
        x, _ = tf(img, None)
        y = int(row.class_id)
        outs = {r: intr_forward(R["model"], x, dev) for r, R in runs.items()}
        judge_wrong = False
        if judge is not None:
            with torch.no_grad():
                jlog = judge(nested_tensor_from_tensor_list([x.to(dev)]))[0].float().cpu()
            judge_wrong = jlog.argmax() != y
        if not args.include_wrong and (judge_wrong or any(o[0].argmax() != y for o in outs.values())):
            continue                                   # paper: images all models get right
        kept += 1
        H, W = x.shape[1:]
        blurred = TF.gaussian_blur(x, kernel_size=51, sigma=15.0)
        rec = dict(image=row.image, breed=row.breed, species=row.species)

        grid = next(iter(outs.values()))[1].shape[-2:]
        rand_map = np.random.rand(*grid)
        cam = gradcam(judge, x, y, dev) if judge is not None else None
        if judge is not None:                         # (method name, map, judge function)
            tasks = [(f"INTR-{r}", o[1][:, y].mean(0), shared_fn) for r, o in outs.items()]
            tasks += [("Grad-CAM (R3)", cam, shared_fn), ("Random", rand_map, shared_fn)]
        else:
            tasks = [(f"INTR-{r}", o[1][:, y].mean(0), self_fns[r]) for r, o in outs.items()]
            tasks += [(f"Random (judge {r})", rand_map, self_fns[r]) for r in outs]
        method_names = [t[0] for t in tasks]
        for name, mp, jfn in tasks:
            if mp.shape != tuple(grid):               # safety: align Grad-CAM grid
                mp = F.interpolate(torch.from_numpy(mp)[None, None].float(), size=tuple(grid))[0, 0].numpy()
            ci, cd = insertion_deletion(jfn, x, y, mp, args.steps, dev, blurred)
            rec[f"{name}|insertion"], rec[f"{name}|deletion"] = auc(ci), auc(cd)
            c = curves.setdefault(name, [np.zeros(args.steps + 1), np.zeros(args.steps + 1)])
            c[0] += ci; c[1] += cd

        tri = root / "oxford-iiit-pet" / "annotations" / "trimaps" / f"{row.image}.png"
        if tri.exists():
            petf, pet_share = pet_cells(tri, H, W, grid)
            if pet_share > 0:
                rec["uniform|pet_attention"] = float(petf.mean())
                for r, (lg, at) in outs.items():
                    a = at[:, y]                                    # [heads, h, w]
                    a = a / a.sum((1, 2), keepdims=True)
                    rec[f"INTR-{r}|pet_attention"] = float((a.mean(0) * petf).sum())
                    rec[f"INTR-{r}|pet_attention_min_head"] = float((a * petf).sum((1, 2)).min())
        for r, (lg, at) in outs.items():
            heads = at[:, y]
            if heads.shape[0] > 1:
                masks = [topk_mask(h) for h in heads]
                rec[f"INTR-{r}|head_iou"] = float(np.mean([iou(a, b) for a, b in combinations(masks, 2)]))
            second = int(torch.topk(lg, 2).indices[1])
            rec[f"INTR-{r}|class_iou"] = iou(topk_mask(at[:, y].mean(0)), topk_mask(at[:, second].mean(0)))
        rows.append(rec)
        if row.breed in args.fig_breeds and len(fig_cache.setdefault(row.breed, [])) < 4:
            fig_cache[row.breed].append((img, {r: o[1][:, y] for r, o in outs.items()}, cam))
        if kept % 200 == 0:
            print(f"{kept} images analysed ({time.time() - t0:.0f}s)", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(out / "attention_per_image.csv", index=False)
    summary = {"n_images": len(per), "steps": args.steps,
               "judge": "R3 ResNet-50 (shared)" if judge is not None else "self (each INTR model)",
               "metrics": {}}
    for col in per.columns[3:]:
        summary["metrics"][col] = round(float(per[col].mean()), 4)
    (out / "attention_summary.json").write_text(json.dumps(summary, indent=2))
    tab = []
    for name in method_names:
        tab.append(dict(method=name, insertion=summary["metrics"][f"{name}|insertion"],
                        deletion=summary["metrics"][f"{name}|deletion"]))
    for r in runs:
        d = next(t for t in tab if t["method"] == f"INTR-{r}")
        for k in ("pet_attention", "pet_attention_min_head", "head_iou", "class_iou"):
            d[k] = summary["metrics"].get(f"INTR-{r}|{k}", "")
    tab = pd.DataFrame(tab)
    tab.to_csv(out / "attention_summary.csv", index=False)
    print(f"\nJudge: {summary['judge']}. {len(per)} images (correct for all models). Uniform-map pet share: "
          f"{summary['metrics'].get('uniform|pet_attention', float('nan')):.3f}")
    print(tab.to_markdown(index=False))

    # ---------------- figures
    xs = np.linspace(0, 1, args.steps + 1)
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for name, (ci, cd) in curves.items():
        axs[0].plot(xs, ci / len(per), label=name)
        axs[1].plot(xs, cd / len(per), label=name)
    axs[0].set_title("Insertion (higher = better)", loc="left", fontsize=9)
    axs[1].set_title("Deletion (lower = better)", loc="left", fontsize=9)
    for a in axs:
        a.set_xlabel("Fraction of image cells inserted / deleted")
        a.set_ylabel("Judge probability of true class")
        a.spines[["top", "right"]].set_visible(False)
    axs[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_insertion_deletion.png", dpi=200); plt.close(fig)

    for breed, items in fig_cache.items():
        for r in runs:
            nh = items[0][1][r].shape[0]
            fig, axs = plt.subplots(len(items), nh + 1, figsize=(1.25 * (nh + 1), 1.3 * len(items)),
                                    squeeze=False)
            for ri, (img, att, _) in enumerate(items):
                axs[ri, 0].imshow(img)
                for h in range(nh):
                    axs[ri, h + 1].imshow(heatmap_overlay(img, att[r][h], 0.5))
                    if ri == 0:
                        axs[ri, h + 1].set_title(f"Head {h + 1}" if nh > 1 else "Attention", fontsize=7)
            axs[0, 0].set_title("Image", fontsize=7)
            for a in axs.flat:
                a.axis("off")
            fig.suptitle(f"INTR-{r}: {breed}, true-class query, last decoder layer", fontsize=8)
            fig.tight_layout()
            fig.savefig(out / f"fig_heads_{r}_{breed.replace(' ', '_')}.png", dpi=200)
            plt.close(fig)
        img, att, cam = items[0]
        cols = [("Image", None)] + [(f"INTR-{r} (avg)", att[r].mean(0)) for r in runs]
        if cam is not None:
            cols.append(("Grad-CAM (R3)", cam))
        fig, axs = plt.subplots(1, len(cols), figsize=(1.6 * len(cols), 1.8))
        for a, (t, mp) in zip(np.atleast_1d(axs), cols):
            a.imshow(img if mp is None else heatmap_overlay(img, mp, 0.5))
            a.set_title(t, fontsize=7); a.axis("off")
        fig.tight_layout(); fig.savefig(out / f"fig_methods_{breed.replace(' ', '_')}.png", dpi=200)
        plt.close(fig)
    print("figures and tables written to", out)


if __name__ == "__main__":
    main()
