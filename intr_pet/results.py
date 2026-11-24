"""Shared helper: write per-image test predictions in one format for every run.  Ours."""
import numpy as np
import pandas as pd
import torch

from intr_pet.data import load_split


def write_predictions(out_dir, logits, labels, data_root, limit_test=None):
    """Saves predictions.csv + test_logits.npy; returns (pred_df, acc_without_dups)."""
    df, _ = load_split(data_root)
    test_df = df[df.split == "test"]
    if limit_test:
        test_df = test_df.iloc[:limit_test]
    assert len(test_df) == len(labels) and (test_df.class_id.values == labels).all(), \
        "prediction order does not match the split file"
    probs = torch.softmax(torch.from_numpy(logits).float(), 1).numpy()
    top5 = np.argsort(-logits, 1)[:, :5]
    pred = pd.DataFrame({"image": test_df.image.values, "label": labels,
                         "species": test_df.species.values,
                         "cross_split_dup": test_df.cross_split_dup.values})
    for k in range(5):
        pred[f"pred{k + 1}"] = top5[:, k]
        pred[f"prob{k + 1}"] = probs[np.arange(len(probs)), top5[:, k]].round(5)
    pred["correct"] = pred.pred1 == pred.label
    pred.to_csv(out_dir / "predictions.csv", index=False)
    np.save(out_dir / "test_logits.npy", logits)
    no_dup = pred[~pred.cross_split_dup]
    return pred, 100 * no_dup.correct.mean(), len(no_dup)
