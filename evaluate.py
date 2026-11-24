"""Accuracy analysis of finished runs (no GPU needed).  Written by our team.

Reads outputs/<run>/predictions.csv (+ log.csv) and produces, in --out_dir:
  summary.json / summary.md        top-1, top-5 (with 95% CI), macro-F1, cat<->dog errors,
                                   accuracy without the 19 cross-split near-duplicates
  per_class_<run>.csv              accuracy / precision / recall / F1 per breed
  confusions_<run>.csv             most frequent (true -> predicted) mistakes
  fig_confusion_<run>.png          37x37 confusion matrix (row-normalised)
  fig_per_class_<run>.png          per-breed accuracy
  fig_training_<run>.png           training curves from log.csv
  mcnemar.csv                      paired significance test between every two runs

  python evaluate.py --runs outputs/R1 outputs/R3 --data_root data --out_dir analysis
"""
import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PAPER = {"INTR": 90.4, "ResNet-50": 89.5}
CAT_C, DOG_C = "#2a78d6", "#eb6834"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on the discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def per_class(pred, classes):
    rows = []
    for c in classes:
        cid = c["class_id"]
        tp = ((pred.label == cid) & (pred.pred1 == cid)).sum()
        n_true = (pred.label == cid).sum()
        n_pred = (pred.pred1 == cid).sum()
        prec = tp / n_pred if n_pred else 0.0
        rec = tp / n_true if n_true else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(dict(class_id=cid, breed=c["breed"], species=c["species"], n_test=int(n_true),
                         accuracy=round(100 * rec, 2), precision=round(100 * prec, 2),
                         f1=round(100 * f1, 2)))
    return pd.DataFrame(rows)


def plot_confusion(pred, classes, path, title):
    order = sorted(classes, key=lambda c: (c["species"], c["breed"]))
    ids = [c["class_id"] for c in order]
    m = pd.crosstab(pred.label, pred.pred1).reindex(index=ids, columns=ids, fill_value=0).values
    m = m / m.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
    names = [c["breed"] for c in order]
    ax.set_xticks(range(len(ids)), names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(ids)), names, fontsize=6)
    for t, c in zip(ax.get_xticklabels() + ax.get_yticklabels(), order + order):
        t.set_color(CAT_C if c["species"] == "cat" else DOG_C)
    n_cat = sum(c["species"] == "cat" for c in order)
    ax.axhline(n_cat - 0.5, color="#52514e", lw=0.6)
    ax.axvline(n_cat - 0.5, color="#52514e", lw=0.6)
    ax.set_xlabel("Predicted breed")
    ax.set_ylabel("True breed")
    ax.set_title(title, loc="left", fontsize=10)
    fig.colorbar(im, fraction=0.04, pad=0.02, label="Fraction of the true breed's test images")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_per_class(pc, path, title):
    pc = pc.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    colors = [CAT_C if s == "cat" else DOG_C for s in pc.species]
    ax.bar(range(len(pc)), pc.accuracy, color=colors, width=0.75)
    ax.set_xticks(range(len(pc)), pc.breed, rotation=70, ha="right", fontsize=6.5)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(min(50, pc.accuracy.min() - 5), 100)
    ax.set_title(title, loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training(log, path, title):
    tr = log[log.train_loss.notna()]
    te = log[log.test_acc1.notna()].drop_duplicates("epoch", keep="last")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.8))
    a1.plot(tr.epoch + 1, tr.train_loss, color="#2a78d6")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Training loss (cross-entropy)")
    a2.plot(tr.epoch + 1, tr.train_acc1, color="#2a78d6", label="Train (augmented)")
    a2.plot(te.epoch + 1, te.test_acc1, "o-", color="#eb6834", ms=3, label="Test")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Top-1 accuracy (%)")
    a2.legend(frameon=False, fontsize=8)
    drops = tr.lr.astype(float).diff().fillna(0) < 0
    for e in tr.epoch[drops]:
        for a in (a1, a2):
            a.axvline(e + 1, color="#8a8a85", ls="--", lw=0.7)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run folders, e.g. outputs/R1")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--out_dir", default="analysis")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    classes = json.loads((Path(args.data_root) / "classes.json").read_text())
    species_of = {c["class_id"]: c["species"] for c in classes}
    name_of = {c["class_id"]: c["breed"] for c in classes}

    summary, preds = [], {}
    for run_dir in map(Path, args.runs):
        run = run_dir.name
        pred = pd.read_csv(run_dir / "predictions.csv")
        preds[run] = pred
        n, k = len(pred), int(pred.correct.sum())
        top5 = int(sum((pred[[f"pred{i}" for i in range(1, 6)]].values == pred.label.values[:, None]).any(1)))
        pc = per_class(pred, classes)
        pc.to_csv(out / f"per_class_{run}.csv", index=False)
        wrong = pred[~pred.correct]
        species_err = (wrong.label.map(species_of) != wrong.pred1.map(species_of)).sum()
        conf = (wrong.groupby(["label", "pred1"]).size().sort_values(ascending=False)
                .head(15).reset_index(name="count"))
        conf["true"] = conf.label.map(name_of)
        conf["predicted"] = conf.pred1.map(name_of)
        conf[["true", "predicted", "count"]].to_csv(out / f"confusions_{run}.csv", index=False)
        nd = pred[~pred.cross_split_dup]
        lo, hi = wilson(k, n)
        cat = pred[pred.label.map(species_of) == "cat"]
        dog = pred[pred.label.map(species_of) == "dog"]
        summary.append(dict(
            run=run, n_test=n, top1=round(100 * k / n, 2), top1_ci95=f"{lo:.1f}-{hi:.1f}",
            top5=round(100 * top5 / n, 2), macro_f1=round(pc.f1.mean(), 2),
            top1_cats=round(100 * cat.correct.mean(), 2), top1_dogs=round(100 * dog.correct.mean(), 2),
            species_errors=int(species_err), species_error_rate=round(100 * species_err / n, 2),
            top1_without_dups=round(100 * nd.correct.mean(), 2), n_without_dups=len(nd),
            worst_breed=pc.sort_values("accuracy").iloc[0].breed,
            worst_breed_acc=float(pc.accuracy.min()),
            top_confusion=(f"{conf.iloc[0].true} -> {conf.iloc[0].predicted} ({conf.iloc[0]['count']})"
                           if len(conf) else "")))
        plot_confusion(pred, classes, out / f"fig_confusion_{run}.png", f"{run}: confusion matrix (test set)")
        plot_per_class(pc, out / f"fig_per_class_{run}.png", f"{run}: accuracy per breed (blue = cat, orange = dog)")
        if (run_dir / "log.csv").exists():
            plot_training(pd.read_csv(run_dir / "log.csv"), out / f"fig_training_{run}.png",
                          f"{run}: training curves (dashed = learning-rate drop)")

    rows = []
    for a, b in itertools.combinations(preds, 2):
        m = preds[a][["image", "correct"]].merge(preds[b][["image", "correct"]], on="image",
                                                 suffixes=("_a", "_b"))
        only_a = int((m.correct_a & ~m.correct_b).sum())
        only_b = int((~m.correct_a & m.correct_b).sum())
        rows.append(dict(run_a=a, run_b=b, n=len(m), only_a_correct=only_a, only_b_correct=only_b,
                         p_value=round(mcnemar_exact(only_a, only_b), 4)))
    if rows:
        pd.DataFrame(rows).to_csv(out / "mcnemar.csv", index=False)

    (out / "summary.json").write_text(json.dumps(dict(runs=summary, mcnemar=rows, paper=PAPER), indent=2))
    md = pd.DataFrame(summary).to_markdown(index=False) if summary else ""
    (out / "summary.md").write_text(md + "\n")
    print(md)
    if rows:
        print("\nMcNemar (paired, exact):\n" + pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
