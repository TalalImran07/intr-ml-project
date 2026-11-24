"""INTR pet-breed demo: Input -> Preprocessing -> Trained INTR -> Prediction -> Output
(+ the cross-attention maps that explain the prediction).  Written by our team.

Run from the project folder:
    streamlit run app/streamlit_app.py
Weights: weights/intr_pet_R1.pth (or set INTR_WEIGHTS=/path/to/file.pth).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from intr_pet.inference import INTRPredictor, heatmap_overlay  # noqa: E402

WEIGHTS = Path(os.environ.get("INTR_WEIGHTS", ROOT / "weights" / "intr_pet_R1.pth"))
WEIGHTS_URL = os.environ.get("INTR_WEIGHTS_URL", "")      # Hugging Face link, added later
DATA_ROOT = Path(os.environ.get("INTR_DATA", ROOT / "data"))

st.set_page_config(page_title="INTR · Interpretable pet breed classifier", page_icon="🐾",
                   layout="wide")


@st.cache_resource(show_spinner="Loading the trained INTR model…")
def get_predictor(path):
    return INTRPredictor(path, device="cpu")


def ensure_weights():
    if WEIGHTS.exists():
        return True
    if WEIGHTS_URL:
        import urllib.request
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        with st.spinner("Downloading trained weights (one time, ~166 MB)…"):
            urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS)
        return True
    st.error(f"Trained weights not found at `{WEIGHTS}`. Download them (see README) "
             "or set the INTR_WEIGHTS environment variable.")
    return False


def head_grid(img, attn_c, alpha, title_prefix=""):
    """Average map + one map per head for one class. attn_c: [heads, h, w]."""
    n = attn_c.shape[0]
    tiles = [("Average of heads" if n > 1 else "Attention", attn_c.mean(0))]
    tiles += [(f"Head {i + 1}", attn_c[i]) for i in range(n)] if n > 1 else []
    cols_per_row = 5
    for start in range(0, len(tiles), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, (name, amap) in zip(cols, tiles[start:start + cols_per_row]):
            col.image(heatmap_overlay(img, amap, alpha), caption=f"{title_prefix}{name}",
                      width="stretch")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🐾 INTR demo")
if not ensure_weights():
    st.stop()
pred = get_predictor(str(WEIGHTS))
cfg, met = pred.config, pred.metrics
st.sidebar.markdown(
    f"**Model:** INTR (DETR-R50), {cfg['num_queries']} class queries, "
    f"{cfg['dec_heads']} cross-attention heads\n\n"
    f"**Trained on:** Oxford-IIIT Pet, 37 breeds\n\n"
    + (f"**Test accuracy:** top-1 {met['top1']}%, top-5 {met['top5']}% "
       f"(paper: 90.4%)\n\n" if "top1" in met else "")
    + f"**Weights file:** `{WEIGHTS.name}`\n\n**SHA-256:** `{pred.weights_sha256[:16]}…`")
with st.sidebar.expander("Full SHA-256 (matches the report)"):
    st.code(pred.weights_sha256, language=None)
st.sidebar.divider()
layer = st.sidebar.select_slider("Decoder layer shown", options=list(range(1, cfg["dec_layers"] + 1)),
                                 value=cfg["dec_layers"],
                                 help="The paper visualises the last (6th) decoder layer.")
alpha = st.sidebar.slider("Heat-map opacity", 0.2, 0.8, 0.5, 0.05)
threshold = st.sidebar.slider("Low-confidence warning below", 0.1, 0.9, 0.5, 0.05)

# ------------------------------------------------------------------ input
st.title("Which breed is it — and where did the model look?")
st.caption("INTR gives every breed its own query that searches the image. The answer is the "
           "breed whose search found the strongest evidence; its attention maps show where "
           "it found it.")

tab_up, tab_cam, tab_test = st.tabs(["📁 Upload a photo", "📷 Camera", "🧪 Random test image"])
image, true_label = None, None
with tab_up:
    f = st.file_uploader("Cat or dog photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if f:
        image = Image.open(f)
with tab_cam:
    c = st.camera_input("Take a photo")
    if c:
        image = Image.open(c)
with tab_test:
    split_file = DATA_ROOT / "pet_split.csv"
    if split_file.exists():
        if st.button("Pick a random image from the official test set"):
            df = pd.read_csv(split_file)
            row = df[df.split == "test"].sample(1).iloc[0]
            st.session_state["test_pick"] = (str(DATA_ROOT / row.rel_path), row.breed)
        if "test_pick" in st.session_state and image is None:
            path, true_label = st.session_state["test_pick"]
            image = Image.open(path)
    else:
        st.info("Test images not found locally (run `python prepare_data.py --root data`).")

if image is None:
    st.info("Upload a photo, take one with the camera, or pick a test image to start.")
    st.stop()

# ------------------------------------------------------------------ pipeline
t0 = time.perf_counter()
r = pred.predict(image, layer=layer - 1)
total_ms = (time.perf_counter() - t0) * 1000
top = r["order"][0]
p_top = float(r["probs"][top])

steps = st.columns(5)
steps[0].metric("① Input", f"{r['original_size'][1]}×{r['original_size'][0]}")
steps[1].metric("② Preprocessed", f"{r['input_size'][1]}×{r['input_size'][0]}",
                help="Short side resized to 384 px (aspect ratio kept), RGB, ImageNet normalisation")
steps[2].metric("③ INTR model", f"{r['grid'][1]}×{r['grid'][0]} grid",
                help="ResNet-50 + Transformer encoder feature grid searched by 37 class queries")
steps[3].metric("④ Prediction", f"{p_top:.0%}")
steps[4].metric("⑤ Time (CPU)", f"{total_ms:.0f} ms",
                help=f"preprocess {r['timings']['preprocess_ms']:.0f} ms, "
                     f"model {r['timings']['model_ms']:.0f} ms")

left, right = st.columns([1, 1.25])
with left:
    st.image(r["image"], caption="Input image", width="stretch")
with right:
    species = pred.species[top]
    st.subheader(f"{pred.classes[top]}  ·  {species}")
    if true_label:
        ok = true_label == pred.classes[top]
        (st.success if ok else st.error)(f"True label (test set): {true_label}")
    if p_top < threshold:
        st.warning(f"Low confidence ({p_top:.0%}). The model always answers with one of its "
                   "37 breeds, so this photo may show another breed, a mixed breed, or not a "
                   "cat or dog at all.")
    top5 = pd.DataFrame({"Breed": [pred.classes[i] for i in r["order"][:5]],
                         "Probability": [float(r["probs"][i]) for i in r["order"][:5]]})
    st.dataframe(top5, hide_index=True, width="stretch",
                 column_config={"Probability": st.column_config.ProgressColumn(
                     format="%.3f", min_value=0.0, max_value=1.0)})

# ------------------------------------------------------------------ explanations
st.divider()
st.subheader(f"Where did the “{pred.classes[top]}” query look?")
st.caption(f"Cross-attention of decoder layer {layer}. Each head learns to look for a different "
           "trait; red = high attention.")
head_grid(r["image"], r["attention"][:, top], alpha)

st.divider()
st.subheader("Why not another breed?")
others = r["order"][1:5]
choice = st.selectbox("Compare with", others,
                      format_func=lambda i: f"{pred.classes[i]} ({r['probs'][i]:.1%})")
st.caption("Every breed searches the same image. A wrong breed usually finds weaker or "
           "scattered evidence.")
head_grid(r["image"], r["attention"][:, choice], alpha, title_prefix=f"{pred.classes[choice]}: ")

with st.expander("Raw model output (proof the answer comes from the model)"):
    st.dataframe(pd.DataFrame({"Breed": pred.classes, "Logit": r["logits"].round(4),
                               "Probability": r["probs"].round(5)})
                 .sort_values("Logit", ascending=False), hide_index=True,
                 width="stretch")
    st.caption(f"Model loaded in {pred.load_seconds:.1f} s on CPU · weights SHA-256 "
               f"{pred.weights_sha256}")
