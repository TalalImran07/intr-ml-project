"""Inference pipeline used by the app (and reusable in analysis).  Written by our team.

Input image -> preprocessing (same test transform as evaluation) -> trained INTR ->
37 class scores + per-head cross-attention maps of the last decoder layer.
"""
import time

import numpy as np
import torch
from PIL import Image, ImageOps

import intr_pet  # noqa: F401
from intr_pet.data import IMAGENET_MEAN, IMAGENET_STD, make_transforms
from intr_pet.model import load_weights, sha256
from util.misc import nested_tensor_from_tensor_list   # official padding helper


class INTRPredictor:
    def __init__(self, weights_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        self.model, self.args, ckpt = load_weights(weights_path, self.device)
        self.load_seconds = time.time() - t0
        self.classes = [c["breed"] for c in ckpt["classes"]]
        self.species = [c["species"] for c in ckpt["classes"]]
        self.preset = ckpt.get("preset", "colab")
        self.metrics = ckpt.get("metrics", {})
        self.config = ckpt["config"]
        self.weights_sha256 = sha256(weights_path)
        self.transform = make_transforms("test", self.preset)

    def preprocess(self, image):
        """PIL image (any mode / orientation) -> normalised tensor [3, H, W] + the RGB image."""
        img = ImageOps.exif_transpose(image).convert("RGB")   # phone photos carry rotation
        tensor, _ = self.transform(img, None)                  # resize short side, normalise
        return tensor, img

    @torch.no_grad()
    def predict(self, image, layer=-1):
        timings = {}
        t0 = time.perf_counter()
        tensor, rgb = self.preprocess(image)
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        samples = nested_tensor_from_tensor_list([tensor.to(self.device)])
        out, encoder_output, hs, attention, avg_attention = self.model(samples)
        logits = out["query_logits"][0].float()
        timings["model_ms"] = (time.perf_counter() - t0) * 1000

        h, w = encoder_output.shape[-2:]
        # attention: [dec_layers, batch, heads, classes, h*w] -> [heads, classes, h, w]
        attn = attention[layer, 0].float().reshape(attention.shape[2], attention.shape[3], h, w)
        probs = torch.softmax(logits, 0)
        order = torch.argsort(probs, descending=True).tolist()
        return dict(probs=probs.cpu().numpy(), logits=logits.cpu().numpy(), order=order,
                    attention=attn.cpu().numpy(), grid=(int(h), int(w)),
                    input_size=tuple(tensor.shape[1:]), original_size=rgb.size[::-1],
                    image=rgb, timings=timings)


def heatmap_overlay(image, amap, alpha=0.5, cmap="jet"):
    """Upsample an attention map to the image size and blend it over the image (PIL)."""
    import matplotlib
    a = amap - amap.min()
    a = a / (a.max() + 1e-12)
    heat = Image.fromarray(np.uint8(a * 255)).resize(image.size, Image.BILINEAR)
    colored = matplotlib.colormaps[cmap](np.asarray(heat) / 255.0)[..., :3]
    base = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return Image.fromarray(np.uint8(255 * ((1 - alpha) * base + alpha * colored)))


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
