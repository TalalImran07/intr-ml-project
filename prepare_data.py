"""Download Oxford-IIIT Pet and write the split file used by every other script.

Written by our team (logic carried over from our Deliverable 2 notebook,
INTR_Pet_Data_Pipeline.ipynb, Sections 1-2).

Output (inside --root):
    oxford-iiit-pet/            official images + annotations (from torchvision)
    pet_split.csv               one row per labelled image:
                                image, rel_path, class_id (0-36), breed, species,
                                split (train/test), cross_split_dup (bool)
    classes.json                the 37 breed names in class-id order

Usage:
    python prepare_data.py --root data
    python prepare_data.py --root data --no-download   # images already in data/oxford-iiit-pet
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def read_split_file(path):
    """Parse trainval.txt / test.txt: 'Image CLASS-ID SPECIES BREED-ID' per line."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, class_id, species, _ = line.split()
            rows.append(dict(image=name,
                             class_id=int(class_id) - 1,          # 0-based for PyTorch
                             species="cat" if species == "1" else "dog"))
    return pd.DataFrame(rows)


def breed_from_filename(name):
    # 'american_pit_bull_terrier_12' -> 'American Pit Bull Terrier'
    return name.rsplit("_", 1)[0].replace("_", " ").title()


def dhash(img, size=8):
    """64-bit difference hash, robust to resizing/re-compression (near-duplicate check)."""
    g = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return int("".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).flatten()), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="folder for the dataset and split file")
    ap.add_argument("--no-download", action="store_true",
                    help="skip torchvision download (files already in <root>/oxford-iiit-pet)")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    pet_dir = root / "oxford-iiit-pet"

    if not args.no_download:
        # torchvision is used only as a downloader of the two official archives.
        from torchvision.datasets import OxfordIIITPet
        OxfordIIITPet(root=str(root), split="trainval", download=True)
        OxfordIIITPet(root=str(root), split="test", download=True)

    ann = pet_dir / "annotations"
    train = read_split_file(ann / "trainval.txt").assign(split="train")
    test = read_split_file(ann / "test.txt").assign(split="test")
    df = pd.concat([train, test], ignore_index=True)
    df["breed"] = df.image.map(breed_from_filename)
    df["rel_path"] = "oxford-iiit-pet/images/" + df.image + ".jpg"

    # Same consistency checks as the Deliverable 2 audit.
    assert df.groupby("class_id").breed.nunique().max() == 1, "class id -> several breeds"
    assert not (set(train.image) & set(test.image)), "image in both splits"
    assert df.class_id.nunique() == 37, f"expected 37 classes, got {df.class_id.nunique()}"

    # Decode every image (catches corrupt files) and hash it for near-duplicates.
    hashes = []
    for p in df.rel_path:
        with Image.open(root / p) as im:
            im.load()
            hashes.append(dhash(im))
    df["dhash"] = hashes
    groups = df.groupby("dhash").split.agg(lambda s: set(s))
    cross = set(groups[groups.map(len) > 1].index)
    df["cross_split_dup"] = df.dhash.isin(cross)
    n_test_dup = int((df.cross_split_dup & (df.split == "test")).sum())

    classes = (df.drop_duplicates("class_id").sort_values("class_id")
                 [["class_id", "breed", "species"]].to_dict("records"))
    (root / "classes.json").write_text(json.dumps(classes, indent=1))
    df.drop(columns=["dhash"]).to_csv(root / "pet_split.csv", index=False)

    print(f"{len(df):,} images | train {len(train):,} | test {len(test):,} | 37 classes")
    print(f"near-duplicate groups spanning train/test: {len(cross)} "
          f"({n_test_dup} test images flagged)")
    print("wrote", root / "pet_split.csv", "and", root / "classes.json")


if __name__ == "__main__":
    main()
