"""INTR pet-breed demo: Input -> Preprocessing -> Trained INTR -> Prediction -> Output
(+ the cross-attention maps that explain the prediction).  Written by our team.

Run from the project folder:
    streamlit run app/streamlit_app.py
Weights: weights/intr_pet_R1.pth (or set INTR_WEIGHTS=/path/to/file.pth).
Optional ablation comparison: weights/intr_pet_R2.pth (1-head model, or set INTR_COMPARE_WEIGHTS).
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
COMPARE = Path(os.environ.get("INTR_COMPARE_WEIGHTS", ROOT / "weights" / "intr_pet_R2.pth"))

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
cfg = pred.config
layer = st.sidebar.select_slider("Decoder layer shown", options=list(range(1, cfg["dec_layers"] + 1)),
                                 value=cfg["dec_layers"],
                                 help="The paper visualises the last (6th) decoder layer.")
alpha = st.sidebar.slider("Heat-map opacity", 0.2, 0.8, 0.5, 0.05)
threshold = st.sidebar.slider("Low-confidence warning below", 0.1, 0.9, 0.5, 0.05)
pred2 = None
if COMPARE.exists():
    st.sidebar.divider()
    if st.sidebar.toggle("Compare with the 1-head model (R2)", value=False,
                         help="Our ablation: identical training, but 1 cross-attention head instead of 8."):
        pred2 = get_predictor(str(COMPARE))

# ------------------------------------------------------------------ input
st.title("Which breed is it — and where did the model look?")
st.caption("INTR gives every breed its own query that searches the image. The answer is the "
           "breed whose search found the strongest evidence; its attention maps show where "
           "it found it.")

f = st.file_uploader("Upload a cat or dog photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
if f is None:
    st.info("Upload a photo to start.")
    st.stop()
image = Image.open(f)

# ------------------------------------------------------------------ pipeline
t0 = time.perf_counter()
r = pred.predict(image, layer=layer - 1)
total_ms = (time.perf_counter() - t0) * 1000
top = r["order"][0]
p_top = float(r["probs"][top])

left, right = st.columns([1, 1.25])
with left:
    st.image(r["image"], caption="Input image", width="stretch")
with right:
    species = pred.species[top]
    st.subheader(f"{pred.classes[top]}  ·  {species}")
    st.markdown(f"Confidence: **{p_top:.1%}**")
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

if pred2 is not None:
    st.divider()
    st.subheader("Ablation: 8 heads (R1) vs 1 head (R2)")
    t1 = time.perf_counter()
    r2 = pred2.predict(image, layer=layer - 1)
    ms2 = (time.perf_counter() - t1) * 1000
    top2 = r2["order"][0]
    agree = top2 == top
    st.caption("Both models were trained identically; the only difference is the number of "
               "cross-attention heads in the decoder. R1 splits its search into 8 maps, R2 does "
               "a single search.")
    c1, c2 = st.columns(2)
    c1.markdown(f"**R1 (8 heads) predicts:** {pred.classes[top]} ({p_top:.1%})")
    c2.markdown(f"**R2 (1 head) predicts:** {pred2.classes[top2]} "
                f"({float(r2['probs'][top2]):.1%})")
    if not agree:
        st.info("The two models disagree on this image.")
    cols = st.columns(3)
    cols[0].image(r["image"], caption="Input", width="stretch")
    cols[1].image(heatmap_overlay(r["image"], r["attention"][:, top].mean(0), alpha),
                  caption=f"R1: average of 8 heads ({pred.classes[top]})", width="stretch")
    cols[2].image(heatmap_overlay(r["image"], r2["attention"][0, top], alpha),
                  caption=f"R2: its single head ({pred.classes[top]})", width="stretch")
    if not agree:
        st.image(heatmap_overlay(r["image"], r2["attention"][0, top2], alpha), width=300,
                 caption=f"R2's map for its own prediction ({pred2.classes[top2]})")
    st.caption(f"Same breed query, same decoder layer ({layer}). R1's 8 individual head maps are "
               "shown above.")

st.divider()
st.subheader("Why not another breed?")
others = r["order"][1:5]
choice = st.selectbox("Compare with", others,
                      format_func=lambda i: f"{pred.classes[i]} ({r['probs'][i]:.1%})")
st.caption("Every breed searches the same image. A wrong breed usually finds weaker or "
           "scattered evidence.")
head_grid(r["image"], r["attention"][:, choice], alpha, title_prefix=f"{pred.classes[choice]}: ")

with st.expander("Pipeline details and raw model output (proof the answer comes from the model)"):
    h, w = r["original_size"]
    ih, iw = r["input_size"]
    gh, gw = r["grid"]
    st.markdown(
        f"1. **Input:** {w}×{h} pixels\n"
        f"2. **Preprocessing:** short side resized to 384 px (aspect ratio kept), RGB, ImageNet "
        f"normalisation → {iw}×{ih}\n"
        f"3. **INTR model:** ResNet-50 + Transformer encoder → {gw}×{gh} feature grid, searched by "
        f"37 class queries in 6 decoder layers\n"
        f"4. **Prediction:** softmax over 37 logits → {pred.classes[top]} ({p_top:.1%})\n"
        f"5. **Time on CPU:** {total_ms:.0f} ms (preprocessing {r['timings']['preprocess_ms']:.0f} ms, "
        f"model {r['timings']['model_ms']:.0f} ms)"
        + (f"; R2: {ms2:.0f} ms" if pred2 is not None else ""))
    st.dataframe(pd.DataFrame({"Breed": pred.classes, "Logit": r["logits"].round(4),
                               "Probability": r["probs"].round(5)})
                 .sort_values("Logit", ascending=False), hide_index=True,
                 width="stretch")
    st.caption(f"Model loaded in {pred.load_seconds:.1f} s on CPU · {WEIGHTS.name} SHA-256 "
               f"{pred.weights_sha256}"
               + (f" · {COMPARE.name} SHA-256 {pred2.weights_sha256}" if pred2 is not None else ""))
