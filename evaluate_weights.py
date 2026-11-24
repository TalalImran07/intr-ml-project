"""Re-run the final test evaluation from a saved INTR weights file.  Written by our team.

Recreates predictions.csv, test_logits.npy and metrics.json exactly as train.py writes
them at the end of training (same test transform, same order), e.g. when the run's
output files were lost but intr_pet_<run>.pth was kept.

  python evaluate_weights.py --weights weights/intr_pet_R1.pth --data_root data --out_dir outputs/R1
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from intr_pet.data import build_loaders
from intr_pet.model import load_weights, sha256
from intr_pet.results import write_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit_test", type=int, default=None, help="debug only")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    amp = dev.startswith("cuda")

    model, margs, ckpt = load_weights(args.weights, dev)
    digest = sha256(args.weights)
    print("weights:", args.weights, "\nsha256:", digest)
    _, test_loader, _, _ = build_loaders(args.data_root, args.batch_size, args.num_workers,
                                         ckpt.get("preset", "colab"), limit_test=args.limit_test)
    logits, labels = [], []
    with torch.no_grad():
        for i, (samples, targets) in enumerate(test_loader):
            y = torch.cat([t["image_label"] for t in targets])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                lg = model(samples.to(dev))[0]["query_logits"].float().cpu()
            logits.append(lg); labels.append(y)
            if i % 50 == 0:
                print(f"batch {i}/{len(test_loader)}", flush=True)
    logits, labels = torch.cat(logits), torch.cat(labels)
    top5 = logits.topk(5, 1).indices
    top1 = 100 * (top5[:, 0] == labels).float().mean().item()
    acc5 = 100 * (top5 == labels[:, None]).any(1).float().mean().item()
    loss = F.cross_entropy(logits, labels).item()
    _, acc_nodup, n_nodup = write_predictions(out, logits.numpy(), labels.numpy(), args.data_root,
                                              args.limit_test)
    stored = ckpt.get("metrics", {})
    metrics = dict(stored)
    metrics.update(top1=round(top1, 2), top5=round(acc5, 2), n_test=len(labels), test_loss=round(loss, 4),
                   top1_without_cross_split_dups=round(acc_nodup, 2), n_test_without_dups=n_nodup,
                   weights_sha256=digest, recomputed_from_weights=True,
                   top1_at_end_of_training=stored.get("top1"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    if "top1" in stored:
        print(f"\nRecomputed top-1 {top1:.2f}% vs {stored['top1']}% at the end of training "
              f"(tiny differences can come from GPU rounding).")


if __name__ == "__main__":
    main()
