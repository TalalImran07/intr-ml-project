"""Train and evaluate INTR on Oxford-IIIT Pet (R1 = reproduction; R2 / INTR-FC = ablations).

Attribution
-----------
ADAPTED from the official INTR main.py + engine.py (third_party/INTR):
  same model, loss (cross-entropy), optimizer (AdamW, separate backbone lr),
  StepLR schedule, gradient clipping (0.1), drop_last training batches, DETR init.
Our changes (feasibility / engineering, all documented in the report):
  * our Pet data pipeline at reduced image sizes (intr_pet/data.py)
  * mixed precision (AMP, fp16) with a gradient scaler
  * test-set evaluation every --eval_every epochs instead of every epoch
  * resume support + a time guard that saves and stops before Kaggle's 12 h limit
  * CSV logging, final predictions file, small final-weights file with SHA-256
  * --timing mode to measure speed/memory before launching a full run
  * hooks for the ablations (--dec_heads, --head fc)

Examples
--------
  python train.py --run_name R1 --timing 50             # speed test, prints suggested epochs
  python train.py --run_name R1 --epochs 100             # R1 (lr drop at 80% of epochs)
  python train.py --run_name R2 --epochs 100 --dec_heads 1
  python train.py --run_name FC --epochs 100 --head fc
"""
import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from intr_pet.data import build_loaders, load_split
from intr_pet.model import DETR_R50_URL, build_intr, load_detr, save_weights


def get_args():
    ap = argparse.ArgumentParser("INTR on Oxford-IIIT Pet")
    ap.add_argument("--run_name", default="R1")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_root", default="outputs")
    # optimisation (paper Appendix E for Pet: lr 1e-4, batch 12, 100 epochs, drop at 80)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr_drop", type=int, default=None, help="default: round(0.8 * epochs)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--clip_max_norm", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    # model / ablations
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--dec_heads", type=int, default=None, help="decoder cross-attention heads (R2: 1)")
    ap.add_argument("--head", choices=["shared", "fc"], default="shared", help="fc = INTR-FC")
    ap.add_argument("--finetune", default=DETR_R50_URL, help="DETR-R50 checkpoint (URL or path)")
    # data / system
    ap.add_argument("--preset", choices=["colab", "paper"], default="colab")
    ap.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_amp", action="store_true", help="train in full fp32")
    ap.add_argument("--eval_every", type=int, default=10)
    ap.add_argument("--time_limit_hours", type=float, default=11.0,
                    help="save and stop before this much wall time (Kaggle limit is 12 h)")
    ap.add_argument("--resume_dir", default=None,
                    help="folder with a previous checkpoint.pth/log.csv (e.g. Kaggle input)")
    ap.add_argument("--timing", type=int, default=0, help="N>0: time N iterations and exit")
    ap.add_argument("--budget_hours", type=float, default=9.0, help="for --timing suggestion")
    ap.add_argument("--limit_train", type=int, default=None, help="debug: subset size")
    ap.add_argument("--limit_test", type=int, default=None, help="debug: subset size")
    a = ap.parse_args()
    if a.lr_drop is None:
        a.lr_drop = max(1, round(0.8 * a.epochs))
    a.amp = (not a.no_amp) and a.device.startswith("cuda")
    return a


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(samples, targets, device):
    labels = torch.cat([t["image_label"] for t in targets]).to(device, non_blocking=True)
    return samples.to(device), labels


def train_one_epoch(model, criterion, loader, optimizer, scaler, device, amp, clip, max_iters=None):
    """Adapted from official engine.train_one_epoch (+ AMP, + skip non-finite batches)."""
    model.train()
    n, loss_sum, correct, skipped = 0, 0.0, 0, 0
    for it, (samples, targets) in enumerate(loader):
        if max_iters is not None and it >= max_iters:
            break
        samples, labels = to_device(samples, targets, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            outputs = model(samples)[0]
            loss = criterion(outputs, [{"image_label": labels}], model)["CE_loss"]
        if not math.isfinite(loss.item()):
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            if skipped > 20:
                sys.exit("More than 20 non-finite losses in one epoch - stopping. "
                         "Re-run with --no_amp.")
            continue
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        bs = labels.numel()
        n += bs
        loss_sum += loss.item() * bs
        correct += (outputs["query_logits"].argmax(1) == labels).sum().item()
    return dict(train_loss=loss_sum / max(n, 1), train_acc1=100.0 * correct / max(n, 1),
                skipped=skipped)


@torch.no_grad()
def evaluate(model, criterion, loader, device, amp, keep_logits=False, max_iters=None):
    """Adapted from official engine.evaluate; optionally returns all logits."""
    model.eval()
    n, loss_sum, c1, c5, all_logits, all_labels = 0, 0.0, 0, 0, [], []
    for it, (samples, targets) in enumerate(loader):
        if max_iters is not None and it >= max_iters:
            break
        samples, labels = to_device(samples, targets, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            outputs = model(samples)[0]
            loss = criterion(outputs, [{"image_label": labels}], model)["CE_loss"]
        logits = outputs["query_logits"].float()
        top5 = logits.topk(5, dim=1).indices
        bs = labels.numel()
        n += bs
        loss_sum += loss.item() * bs
        c1 += (top5[:, 0] == labels).sum().item()
        c5 += (top5 == labels[:, None]).any(1).sum().item()
        if keep_logits:
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    stats = dict(test_loss=loss_sum / max(n, 1), test_acc1=100.0 * c1 / max(n, 1),
                 test_acc5=100.0 * c5 / max(n, 1), n_test=n)
    if keep_logits:
        return stats, torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()
    return stats


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, args):
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "args": vars(args)}, tmp)
    os.replace(tmp, path)                     # atomic: never leaves a half-written file


def peak_mem_gb(device):
    return torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0


def run_timing(args, model, criterion, train_loader, test_loader, optimizer, scaler, out_dir):
    dev = args.device
    print(f"\n== Timing test: {args.timing} training iterations (after 5 warm-up) ==")
    train_one_epoch(model, criterion, train_loader, optimizer, scaler, dev, args.amp,
                    args.clip_max_norm, max_iters=5)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    train_one_epoch(model, criterion, train_loader, optimizer, scaler, dev, args.amp,
                    args.clip_max_norm, max_iters=args.timing)
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    t_it = (time.time() - t0) / args.timing
    n_eval = min(30, len(test_loader))
    t0 = time.time()
    evaluate(model, criterion, test_loader, dev, args.amp, max_iters=n_eval)
    e_it = (time.time() - t0) / n_eval

    train_min = t_it * len(train_loader) / 60
    eval_min = e_it * len(test_loader) / 60

    def total_hours(e):
        return (e * train_min + (e // args.eval_every + 1) * eval_min) / 60

    best = max([e for e in range(1, 101) if total_hours(e) <= args.budget_hours] or [1])
    res = dict(sec_per_train_iter=round(t_it, 3), train_iters_per_epoch=len(train_loader),
               minutes_per_epoch=round(train_min, 2), minutes_per_full_eval=round(eval_min, 2),
               peak_gpu_mem_gb=round(peak_mem_gb(dev), 2), amp=args.amp,
               hours_for_100_epochs=round(total_hours(100), 2),
               budget_hours=args.budget_hours, suggested_epochs=best,
               suggested_lr_drop=max(1, round(0.8 * best)))
    print(json.dumps(res, indent=2))
    (out_dir / "timing.json").write_text(json.dumps(res, indent=2))
    print(f"\n-> Suggested: --epochs {best} (lr drop at {res['suggested_lr_drop']}), "
          f"about {total_hours(best):.1f} h on this GPU.")


def main():
    args = get_args()
    out_dir = Path(args.out_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    dev = args.device
    print("Config:", json.dumps(vars(args), indent=1))
    if dev.startswith("cuda"):
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, test_loader, test_set, classes = build_loaders(
        args.data_root, args.batch_size, args.num_workers, args.preset, args.seed,
        args.limit_train, args.limit_test)
    print(f"train {len(train_loader.dataset)} images / {len(train_loader)} batches | "
          f"test {len(test_set)} images / {len(test_loader)} batches")

    model, criterion, margs = build_intr(len(classes), device=dev, dec_heads=args.dec_heads,
                                         head=args.head, nheads=args.nheads)
    # Official main.py builds the optimizer BEFORE loading DETR, while the ResNet stem and
    # layer1 are still frozen (backbone.py), and only then sets requires_grad=True on
    # everything. Net effect in the official code: stem/layer1 get gradients (which count
    # in the gradient-clipping norm) but are never updated. We reproduce that exactly.
    trainable_in_official = {n for n, p in model.named_parameters() if p.requires_grad}

    # --- resume or start from DETR ---
    if args.resume_dir:                        # bring a previous session's files over
        for f in ("checkpoint.pth", "log.csv"):
            src = Path(args.resume_dir) / f
            if src.exists() and not (out_dir / f).exists():
                shutil.copy(src, out_dir / f)
    ckpt_path = out_dir / "checkpoint.pth"
    start_epoch = 0
    if ckpt_path.exists() and not args.timing:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from {ckpt_path} at epoch {start_epoch}")
    else:
        ckpt = None
        load_detr(model, margs, args.finetune)
        print("Initialised from DETR-R50:", args.finetune)

    for p in model.parameters():               # as official main.py after loading DETR
        p.requires_grad = True
    named = list(model.named_parameters())
    param_groups = [
        {"params": [p for n, p in named if "backbone" not in n]},
        {"params": [p for n, p in named if "backbone" in n and n in trainable_in_official],
         "lr": args.lr_backbone},
        # stem + layer1: lr 0 and no weight decay = never updated (official behaviour).
        # They sit in the optimizer only so the AMP scaler un-scales their gradients
        # before clipping; otherwise their scaled gradients would distort the clip norm.
        {"params": [p for n, p in named if "backbone" in n and n not in trainable_in_official],
         "lr": 0.0, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)   # gamma 0.1
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["lr_scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        del ckpt
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_params:,}")

    if args.timing:
        run_timing(args, model, criterion, train_loader, test_loader, optimizer, scaler, out_dir)
        return

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=1))
    log_path = out_dir / "log.csv"
    fields = ["epoch", "lr", "train_loss", "train_acc1", "test_loss", "test_acc1", "test_acc5",
              "epoch_minutes", "peak_gpu_mem_gb", "skipped_batches"]
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    session_start, epoch_times = time.time(), []
    for epoch in range(start_epoch, args.epochs):
        # time guard: stop cleanly if the next epoch might not finish in this session
        if epoch_times:
            projected = time.time() - session_start + 1.3 * max(epoch_times[-3:])
            if projected > args.time_limit_hours * 3600:
                (out_dir / "STATUS.txt").write_text(
                    f"PAUSED before epoch {epoch}. Re-run with --resume_dir pointing at this folder.\n")
                print(f"Time limit reached - saved. Resume from epoch {epoch} in a new session.")
                return
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        stats = train_one_epoch(model, criterion, train_loader, optimizer, scaler, dev,
                                args.amp, args.clip_max_norm)
        scheduler.step()
        save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, epoch, args)
        last = epoch + 1 == args.epochs
        if (epoch + 1) % args.eval_every == 0 and not last:
            stats.update(evaluate(model, criterion, test_loader, dev, args.amp))
        epoch_times.append(time.time() - t0)
        row = dict(epoch=epoch, lr=lr_now, epoch_minutes=round(epoch_times[-1] / 60, 2),
                   peak_gpu_mem_gb=round(peak_mem_gb(dev), 2),
                   skipped_batches=stats.pop("skipped"), **stats)
        row.pop("n_test", None)
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow({k: row.get(k, "") for k in fields})
        print(f"epoch {epoch:3d} | lr {lr_now:.1e} | loss {row['train_loss']:.4f} | "
              f"train acc {row['train_acc1']:.2f}"
              + (f" | TEST acc1 {row['test_acc1']:.2f} acc5 {row['test_acc5']:.2f}"
                 if "test_acc1" in row else "")
              + f" | {row['epoch_minutes']} min", flush=True)

    # --- final evaluation on the full test set (last epoch, no model selection) ---
    stats, logits, labels = evaluate(model, criterion, test_loader, dev, args.amp, keep_logits=True)
    with open(log_path, "a", newline="") as f:
        csv.DictWriter(f, fields).writerow(
            {"epoch": args.epochs - 1, "test_loss": stats["test_loss"],
             "test_acc1": stats["test_acc1"], "test_acc5": stats["test_acc5"]})
    df, _ = load_split(args.data_root)
    test_df = df[df.split == "test"]
    if args.limit_test:
        test_df = test_df.iloc[:args.limit_test]
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    top5 = np.argsort(-logits, 1)[:, :5]
    pred = pd.DataFrame({"image": test_df.image.values, "label": labels,
                         "cross_split_dup": test_df.cross_split_dup.values})
    for k in range(5):
        pred[f"pred{k + 1}"] = top5[:, k]
        pred[f"prob{k + 1}"] = probs[np.arange(len(probs)), top5[:, k]].round(5)
    pred["correct"] = pred.pred1 == pred.label
    pred.to_csv(out_dir / "predictions.csv", index=False)
    np.save(out_dir / "test_logits.npy", logits)
    no_dup = pred[~pred.cross_split_dup]

    metrics = dict(run=args.run_name, epochs=args.epochs, top1=round(stats["test_acc1"], 2),
                   top5=round(stats["test_acc5"], 2), n_test=int(stats["n_test"]),
                   top1_without_cross_split_dups=round(100 * no_dup.correct.mean(), 2),
                   n_test_without_dups=len(no_dup),
                   paper_top1_pet=90.4, total_train_hours_this_session=round(
                       (time.time() - session_start) / 3600, 2))
    weights_path = out_dir / f"intr_pet_{args.run_name}.pth"
    metrics["weights_sha256"] = save_weights(
        model, margs, weights_path,
        extra={"classes": classes, "metrics": metrics, "preset": args.preset})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "STATUS.txt").write_text("DONE\n")
    print("\nFINAL (last epoch, full test set):", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
