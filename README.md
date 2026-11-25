# Reproducing INTR on Oxford-IIIT Pet

Machine Learning Paper Reproduction & Deployment — IBA, Fall 2026
Team: Talal Imran (36237) · Warisha Athar (36243) · Muhammad Zain (29403)

We reproduce **INTR** ("A Simple Interpretable Transformer for Fine-Grained Image
Classification and Analysis", Paul *et al.*, ICLR 2024) on the Oxford-IIIT Pet dataset
(37 cat and dog breeds), run an ablation with one attention head instead of eight, and
deploy the model as a Streamlit app that predicts the breed of a photo and shows where
the model looked. Full details are in our final report.

* **Trained weights:** https://huggingface.co/Talloo/intr-oxford-pet
* **Official INTR code:** https://github.com/Imageomics/INTR

## Results (test set, 3,669 images)

| Run | Model | Top-1 | Paper |
|---|---|---|---|
| R1 | INTR, 8 heads (reproduction, used in the app) | **91.74%** | 90.4% |
| R2 | INTR, 1 head (ablation) | 92.26% | – |
| R3 | ResNet-50 baseline | 93.00% | 89.5% |

The number of heads does not change accuracy, but the 8-head model gives clearly better
explanation maps. All result files are in `results/`.

## Run the app

```bash
pip install -r requirements.txt
mkdir -p weights
wget -P weights https://huggingface.co/Talloo/intr-oxford-pet/resolve/main/intr_pet_R1.pth
wget -P weights https://huggingface.co/Talloo/intr-oxford-pet/resolve/main/intr_pet_R2.pth   # optional: 1-head comparison
streamlit run app/streamlit_app.py     # opens http://localhost:8501
```

Upload a photo of a cat or dog. The app shows the breed, the confidence and top-5 breeds,
and the attention map of each of the 8 heads. Runs on a CPU.

## Reproduce the experiments

A GPU is needed for training (we used a free Kaggle T4; see `kaggle_run.ipynb`).

```bash
python prepare_data.py --root data                          # download dataset + official split
python train.py --run_name R1 --epochs 100                  # reproduction (8 heads)
python train.py --run_name R2 --epochs 100 --dec_heads 1    # ablation (1 head)
python train_resnet.py --run_name R3                        # ResNet-50 baseline
python evaluate.py --runs outputs/R1 outputs/R2 outputs/R3 --data_root data --out_dir analysis
python attention_metrics.py --intr outputs/R1/intr_pet_R1.pth outputs/R2/intr_pet_R2.pth \
    --judge outputs/R3/resnet50_pet_R3.pth --data_root data --out_dir analysis
```

`kaggle_analysis.ipynb` runs the two evaluation scripts on Kaggle using the weights from
Hugging Face.

## Code attribution

| Part | Source |
|---|---|
| INTR model, DETR weight loading, image transforms (`third_party/INTR/`) | Official code (Apache 2.0) |
| Training loop (`train.py`) | **Adapted** from the official `main.py` / `engine.py` |
| Everything else: data pipeline, 1-head option, ResNet-50 baseline, evaluation, explanation metrics, Grad-CAM, app | **Written by our team** |
