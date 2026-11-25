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
│   ├── model.py          builds INTR from the official code; DETR init; ablation options
│   ├── baseline.py       ResNet-50 with masked average pooling (R3)
│   ├── inference.py      single-image pipeline used by the app
│   └── results.py        shared predictions-file writer
├── prepare_data.py       downloads Oxford-IIIT Pet, writes data/pet_split.csv + classes.json
├── train.py              INTR training + evaluation (R1 / R2 / INTR-FC)
├── train_resnet.py       ResNet-50 baseline (R3)
├── evaluate.py           accuracy analysis, figures, significance tests
├── evaluate_weights.py   re-runs the final test evaluation from a saved weights file
├── attention_metrics.py  insertion/deletion, attention on the pet, head diversity, figures
├── app/streamlit_app.py  deployed demo
├── kaggle_run.ipynb      runs prepare_data.py + training on a Kaggle GPU
├── kaggle_analysis.ipynb runs both analysis scripts on finished runs (Kaggle GPU)
└── requirements.txt
```

| Run | Model | Command |
|---|---|---|
| R1 | INTR, the paper's model (reproduction) — **used by the app** | `train.py --run_name R1` |
| R2 | INTR with 1-head decoder cross-attention (ablation) | `train.py --run_name R2 --dec_heads 1` |
| FC | INTR-FC: one presence vector per class (ablation) | `train.py --run_name FC --head fc` |
| R3 | ResNet-50 fine-tuned under the same settings (paper's baseline) | `train_resnet.py --run_name R3` |

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
| ResNet-50 baseline training (masked pooling, SGD recipe) | **Ours**; ImageNet weights from torchvision | `train_resnet.py`, `intr_pet/baseline.py` |
| Evaluation and statistics | **Ours** | `evaluate.py` |
| Interpretability metrics (insertion/deletion protocol from Petsiuk et al. 2018 as used in the paper; Grad-CAM written from its definition) | **Ours** | `attention_metrics.py` |
| Inference pipeline and Streamlit app | **Ours** | `intr_pet/inference.py`, `app/` |

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

## Evaluation

```bash
python evaluate.py --runs outputs/R1 outputs/R3 --data_root data --out_dir analysis
```
Writes `summary.md` (top-1 with 95% interval, top-5, macro-F1, cat/dog accuracy, cat↔dog
errors, accuracy without the 19 near-duplicates), per-breed tables, the most frequent
confusions, confusion-matrix / per-breed / training-curve figures, and an exact McNemar
test between every pair of runs (`mcnemar.csv`).

## Interpretability analysis

```bash
python attention_metrics.py --intr outputs/R1/intr_pet_R1.pth [outputs/R2/intr_pet_R2.pth outputs/FC/intr_pet_FC.pth] \
    --judge outputs/R3/resnet50_pet_R3.pth --data_root data --out_dir analysis --resolutions 320 384 448 512
```
On every test image that all models classify correctly (the paper's protocol), using the
last decoder layer and the true class's query:
* **insertion / deletion** AUC with the R3 ResNet-50 as the shared judge, for each INTR run
  (heads averaged, as in the paper), Grad-CAM on R3 and a random ranking;
* **attention on the pet**: share of attention on trimap pet pixels vs a uniform map;
* **head diversity**: mean pairwise IoU of the heads' top-10% cells (lower = more distinct);
* **class distinctiveness**: IoU of the true and runner-up class maps (lower = more distinct);
* optional **test-resolution check**, and head / method comparison figures.

## Running the app (deployment)

Pipeline: **input photo → preprocessing (short side 384 px, RGB, ImageNet normalisation)
→ trained INTR → 37 breed probabilities → prediction + cross-attention maps**.

```bash
pip install -r requirements.txt
# put the trained weights at weights/intr_pet_R1.pth (download link at the top of this file)
streamlit run app/streamlit_app.py      # opens http://localhost:8501
```

* **Input:** one JPG/PNG/WebP/BMP photo of a cat or dog, uploaded in the browser.
* **Output:** the predicted breed and species with its confidence, the top-5 breeds, a
  low-confidence warning, the attention map of every head for the predicted breed, the same
  maps for any other top-5 breed ("why not X?"), and a collapsible panel with the pipeline
  steps (input size, preprocessed size, feature grid, timing), the raw logits and the SHA-256
  of the weights file.
* Runs on CPU (no GPU needed); about 0.3–1 s per image on a laptop.
* **Ablation view (optional):** put the 1-head model at `weights/intr_pet_R2.pth` and switch on
  *Compare with the 1-head model (R2)* in the sidebar to see R2's prediction and single
  attention map next to R1's maps for the same photo. Without that file the app is unchanged.

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
