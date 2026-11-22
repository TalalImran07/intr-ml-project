# Reproducing INTR on Oxford-IIIT Pet

Machine Learning Paper Reproduction & Deployment — IBA, Fall 2026
Team: Talal Imran (36237) · Warisha Athar (36243) · Muhammad Zain (29403)

Paper: D. Paul *et al.*, "A Simple Interpretable Transformer for Fine-Grained Image
Classification and Analysis", ICLR 2024. Official code: https://github.com/Imageomics/INTR

> **Trained weights:** _link to be added (Hugging Face)_

## What is in this repository

```
intr-oxford-pet/
├── third_party/INTR/     official INTR code, UNMODIFIED (commit 69b8a69, Apache 2.0)
├── intr_pet/
│   ├── data.py           Pet dataset, reduced image sizes, loaders
│   └── model.py          builds INTR from the official code; DETR init; ablation options
├── prepare_data.py       downloads Oxford-IIIT Pet, writes data/pet_split.csv + classes.json
├── train.py              training + evaluation (R1 / R2 / INTR-FC)
├── kaggle_run.ipynb      runs prepare_data.py + train.py on a Kaggle GPU
└── requirements.txt
```
Evaluation scripts, the ResNet-50 baseline and the Streamlit app will be added next.

## Code attribution

| Part | Status | Where |
|---|---|---|
| INTR model: ResNet-50 backbone, Transformer encoder/decoder, class queries, shared presence vector, cross-entropy loss | **Reused, unchanged** | `third_party/INTR/models/` |
| DETR → INTR weight adaptation (drop detection heads, keep first 37 queries, new presence vector) | **Reused, unchanged** (called from our code) | `third_party/INTR/util/misc.py: load_model` |
| Image transforms (flip, multi-scale resize, random crop), padding + mask batching | **Reused, unchanged**; we pass smaller sizes | `third_party/INTR/datasets/transforms.py`, `util/misc.py: collate_fn` |
| Training and evaluation loop | **Adapted** from `main.py` + `engine.py`: mixed precision, evaluation every 10 epochs, resume + time guard, CSV logs, predictions file, timing mode | `train.py` |
| Model builder, 1-head decoder option, INTR-FC head, weight save/load | **Ours** | `intr_pet/model.py` |
| Dataset download, split file, data audit, near-duplicate flags | **Ours** (from our Deliverable 2 notebook) | `prepare_data.py`, `intr_pet/data.py` |
| Kaggle runner | **Ours** | `kaggle_run.ipynb` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py --root data          # ~800 MB download, then audit + split file
```

`data/pet_split.csv` has one row per labelled image with the **official split**
(3,680 `trainval` images for training, 3,669 `test` images for evaluation) and a
`cross_split_dup` flag for test images that have a near-duplicate in the training set.

## Training

A GPU is required (we use a free Kaggle T4). Always run the timing test first:

```bash
python train.py --run_name R1 --timing 50     # prints minutes/epoch, memory, suggested epochs
python train.py --run_name R1 --epochs 100    # R1: reproduction (8 heads, shared vector)
python train.py --run_name R2 --epochs 100 --dec_heads 1   # ablation: 1-head cross-attention
python train.py --run_name FC --epochs 100 --head fc       # ablation: INTR-FC (vector per class)
```

On Kaggle, open `kaggle_run.ipynb`, set `RUN`, `TIMING` and `EPOCHS` in the first cell
and use *Save & Run All (Commit)*; see the notebook's first cell for resuming a paused run.

Outputs in `outputs/<run>/`:

| File | Contents |
|---|---|
| `log.csv` | per epoch: learning rate, train loss/accuracy, test accuracy every 10 epochs, minutes, GPU memory |
| `checkpoint.pth` | full state for resuming (model, optimizer, scheduler, AMP scaler, epoch) |
| `intr_pet_<run>.pth` | final model only (~166 MB), used by the app; its SHA-256 is in `metrics.json` |
| `predictions.csv`, `test_logits.npy` | per test image: label, top-5 classes and probabilities |
| `metrics.json` | final top-1 / top-5, top-1 without the cross-split duplicates |

The reported result is always the **last epoch**: the dataset has no validation split
(the paper also evaluates on the test set), so we do not select epochs by test accuracy.

## Settings: paper vs ours (R1)

| Setting | Paper / official code | Ours | Reason |
|---|---|---|---|
| Initialisation | DETR-R50 (ImageNet + COCO) | same | — |
| Classes / queries | one per class | 37 | Pet has 37 breeds |
| Heads / encoder / decoder layers / width | 8 / 6 / 6 / 256 | same | reproduction |
| Optimizer | AdamW (code), Adam (paper text); weight decay 1e-6 | AdamW, 1e-6 | follows the authors' code |
| Learning rate (transformer / backbone) | 1e-4 for Pet (paper) / 1e-5 | same | — |
| Schedule | StepLR, ×0.1 at epoch 80 of 100 | ×0.1 at 80% of our epochs | keeps the schedule's shape |
| Epochs | 100 | 100 or fewer (set by the timing test) | **feasibility** |
| Batch size | 12 (official command uses 4 GPUs; likely 12 per GPU) | 12 on one GPU | **feasibility** |
| Training image size (short side) | 480–800, long side ≤ 1333 | 320–448, long side ≤ 746 | **feasibility** |
| Test image size | 800 | 384 | **feasibility** |
| Precision | fp32 | mixed precision (fp16) | **feasibility** (speed) |
| Gradient clipping | 0.1 | 0.1 | — |
| Frozen layers | ResNet stem + layer1 never updated | same (reproduced exactly) | — |

Note on the last row: the official `main.py` builds the optimizer while the ResNet stem
and layer1 are frozen and only afterwards sets `requires_grad=True`, so those layers get
gradients (which count in the clipping norm) but are never updated. `train.py`
reproduces this with a learning-rate-0 parameter group.
