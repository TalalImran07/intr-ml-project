"""R3: fine-tune an ImageNet ResNet-50 on Oxford-IIIT Pet (the paper's baseline, re-run
under our settings).  Written by our team.

Same data, split, augmentation, image sizes, padded batches and final-epoch reporting as
the INTR runs. The paper does not give its ResNet-50 fine-tuning recipe, so we use a
standard one (our choice, documented in the report): SGD with momentum 0.9, weight decay
1e-4, batch 24, lr 0.005 with 1 warm-up epoch then cosine decay, 30 epochs, mixed precision.

  python train_resnet.py --run_name R3            # ~30-60 min on a T4
  python train_resnet.py --run_name R3 --timing 30
"""
import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from intr_pet.baseline import MaskedResNet50
from intr_pet.data import build_loaders
from intr_pet.model import sha256
from intr_pet.results import write_predictions


def get_args():
    ap = argparse.ArgumentParser("ResNet-50 baseline on Oxford-IIIT Pet")
    ap.add_argument("--run_name", default="R3")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_root", default="outputs")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--warmup_epochs", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preset", choices=["colab", "paper"], default="colab")
    ap.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--no_imagenet", action="store_true", help="debug: random init, no download")
    ap.add_argument("--timing", type=int, default=0)
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_test", type=int, default=None)
    a = ap.parse_args()
    a.amp = (not a.no_amp) and a.device.startswith("cuda")
    return a


@torch.no_grad()
def evaluate(model, loader, device, amp, keep=False):
    model.eval()
    n, loss_sum, c1, c5, L, Y = 0, 0.0, 0, 0, [], []
    for samples, targets in loader:
        y = torch.cat([t["image_label"] for t in targets]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(samples.to(device)).float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        top5 = logits.topk(5, 1).indices
        c1 += (top5[:, 0] == y).sum().item()
        c5 += (top5 == y[:, None]).any(1).sum().item()
        n += y.numel()
        if keep:
            L.append(logits.cpu()); Y.append(y.cpu())
    stats = dict(test_loss=loss_sum / n, test_acc1=100 * c1 / n, test_acc5=100 * c5 / n, n_test=n)
    return (stats, torch.cat(L).numpy(), torch.cat(Y).numpy()) if keep else stats


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = args.device
    print("Config:", json.dumps(vars(args), indent=1))

    train_loader, test_loader, test_set, classes = build_loaders(
        args.data_root, args.batch_size, args.num_workers, args.preset, args.seed,
        args.limit_train, args.limit_test)
    model = MaskedResNet50(len(classes), imagenet=not args.no_imagenet).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    warm = int(args.warmup_epochs * len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(warm, 1) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    def train_steps(max_iters=None):
        model.train()
        n, loss_sum, correct = 0, 0.0, 0
        for it, (samples, targets) in enumerate(train_loader):
            if max_iters is not None and it >= max_iters:
                break
            y = torch.cat([t["image_label"] for t in targets]).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                logits = model(samples.to(dev))
                loss = F.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            n += y.numel(); loss_sum += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
        return dict(train_loss=loss_sum / max(n, 1), train_acc1=100 * correct / max(n, 1))

    if args.timing:
        train_steps(3)
        t0 = time.time(); train_steps(args.timing)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t = (time.time() - t0) / args.timing
        print(json.dumps({"sec_per_iter": round(t, 3), "iters_per_epoch": len(train_loader),
                          "minutes_per_epoch": round(t * len(train_loader) / 60, 2),
                          "est_hours_total": round(t * total / 3600 * 1.15, 2)}, indent=2))
        return

    log_path = out_dir / "log.csv"
    fields = ["epoch", "lr", "train_loss", "train_acc1", "test_loss", "test_acc1", "test_acc5",
              "epoch_minutes"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()
    start = time.time()
    for epoch in range(args.epochs):
        t0 = time.time()
        lr_now = opt.param_groups[0]["lr"]
        row = dict(epoch=epoch, lr=lr_now, **train_steps())
        if (epoch + 1) % args.eval_every == 0 and epoch + 1 < args.epochs:
            row.update(evaluate(model, test_loader, dev, args.amp))
        row["epoch_minutes"] = round((time.time() - t0) / 60, 2)
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow({k: row.get(k, "") for k in fields})
        print(f"epoch {epoch:3d} | lr {lr_now:.1e} | loss {row['train_loss']:.4f} | "
              f"train acc {row['train_acc1']:.2f}"
              + (f" | TEST acc1 {row['test_acc1']:.2f}" if "test_acc1" in row else "")
              + f" | {row['epoch_minutes']} min", flush=True)

    stats, logits, labels = evaluate(model, test_loader, dev, args.amp, keep=True)
    with open(log_path, "a", newline="") as f:
        csv.DictWriter(f, fields).writerow({"epoch": args.epochs - 1, "test_loss": stats["test_loss"],
                                            "test_acc1": stats["test_acc1"],
                                            "test_acc5": stats["test_acc5"]})
    _, acc_nodup, n_nodup = write_predictions(out_dir, logits, labels, args.data_root, args.limit_test)
    weights = out_dir / f"resnet50_pet_{args.run_name}.pth"
    torch.save({"model": model.state_dict(), "classes": classes, "preset": args.preset}, weights)
    metrics = dict(run=args.run_name, model="ResNet-50 (ImageNet-1K) fine-tuned",
                   epochs=args.epochs, top1=round(stats["test_acc1"], 2),
                   top5=round(stats["test_acc5"], 2), n_test=stats["n_test"],
                   top1_without_cross_split_dups=round(acc_nodup, 2), n_test_without_dups=n_nodup,
                   paper_top1_pet_resnet50=89.5, train_hours=round((time.time() - start) / 3600, 2),
                   weights_sha256=sha256(weights))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "STATUS.txt").write_text("DONE\n")
    print("\nFINAL (last epoch, full test set):", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
