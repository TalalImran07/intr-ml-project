"""Oxford-IIIT Pet dataset, transforms and loaders for INTR.

Attribution
-----------
* PetDataset, scale presets, loader construction: written by our team
  (carried over from our Deliverable 2 notebook).
* Transform classes (RandomResize, RandomSizeCrop, ...) and collate_fn:
  REUSED unchanged from the official INTR repo (third_party/INTR, copied there
  from DETR). We only pass in smaller image sizes (COLAB scales) - a documented
  feasibility change.
"""
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

import intr_pet  # noqa: F401  (adds third_party/INTR to sys.path)
import datasets.transforms as T          # official INTR / DETR transforms
from util.misc import collate_fn         # official padding + mask batching

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Image-size presets. PAPER = official INTR datasets/build.py.
# COLAB = our reduced sizes (about 0.56x), used for all our runs.
SCALES = {
    "paper": dict(train_scales=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
                  crop_resize=[400, 500, 600], crop_min=384, crop_max=600,
                  test_size=800, max_size=1333),
    "colab": dict(train_scales=[320, 352, 384, 416, 448],
                  crop_resize=[240, 300, 360], crop_min=224, crop_max=360,
                  test_size=384, max_size=746),
}


def make_transforms(split, preset="colab", test_size=None):
    """Same structure as INTR's datasets/build.py make_transforms, sizes from SCALES."""
    s = SCALES[preset]
    normalize = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    if split == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(s["train_scales"], max_size=s["max_size"]),
                T.Compose([
                    T.RandomResize(s["crop_resize"]),
                    T.RandomSizeCrop(s["crop_min"], s["crop_max"]),
                    T.RandomResize(s["train_scales"], max_size=s["max_size"]),
                ]),
            ),
            normalize,
        ])
    size = test_size or s["test_size"]
    # keep the long-side cap proportional if a different test size is requested
    max_size = round(s["max_size"] * size / s["test_size"])
    return T.Compose([T.RandomResize([size], max_size=max_size), normalize])


class PetDataset(torch.utils.data.Dataset):
    """Returns (image, target) in the exact format of INTR's CreateImageFolder:
    target = {"file_name": [path], "image_label": tensor([label])}."""

    def __init__(self, frame, root, transform):
        self.paths = [str(Path(root) / p) for p in frame.rel_path]
        self.labels = frame.class_id.astype(int).tolist()
        self.names = frame.image.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # .convert("RGB"): 4 files are PNG with a .jpg name, 3 of them RGBA (our audit)
        img = Image.open(self.paths[i]).convert("RGB")
        target = {"file_name": [self.paths[i]],
                  "image_label": torch.tensor([self.labels[i]], dtype=torch.int64)}
        return self.transform(img, target)


def load_split(root):
    root = Path(root)
    df = pd.read_csv(root / "pet_split.csv")
    classes = json.loads((root / "classes.json").read_text())
    return df, classes


def build_loaders(root, batch_size=12, num_workers=2, preset="colab", seed=42,
                  limit_train=None, limit_test=None):
    """Train loader: shuffled, drop_last=True (as INTR main.py). Test loader: in order."""
    df, classes = load_split(root)
    tr = df[df.split == "train"]
    te = df[df.split == "test"]
    if limit_train:
        tr = tr.sample(limit_train, random_state=seed)
    if limit_test:
        te = te.iloc[:limit_test]
    train_set = PetDataset(tr, root, make_transforms("train", preset))
    test_set = PetDataset(te, root, make_transforms("test", preset))
    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True, generator=g,
        collate_fn=collate_fn, num_workers=num_workers, persistent_workers=num_workers > 0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=False,
        collate_fn=collate_fn, num_workers=num_workers)
    return train_loader, test_loader, test_set, classes
