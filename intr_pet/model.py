"""Build INTR from the official code and load DETR / our checkpoints.

Attribution
-----------
* The model itself (backbone, Transformer, INTR class, loss) is REUSED unchanged
  from third_party/INTR.
* load_detr() calls the official util.misc.load_model (reused) to turn a DETR
  checkpoint into an INTR initialisation.
* Written by our team: build_intr() (argument handling, skipping the unneeded
  ImageNet download), the two ablation options below, checkpoint save/load helpers.

Ablation options (default = R1, the unmodified paper model):
* dec_heads: number of heads in the DECODER CROSS-ATTENTION only (R2 uses 1).
  Encoder and decoder self-attention keep `nheads`. nn.MultiheadAttention's
  weights have the same shapes for any head count, so DETR weights load as-is.
* head="fc": INTR-FC - one presence vector per class (w_c) instead of the shared
  w (paper Appendix, Eq. 10).
"""
import argparse
import hashlib

import torch
from torch import nn

import intr_pet  # noqa: F401  (adds third_party/INTR to sys.path)
import models.backbone as official_backbone
from main import get_args_parser            # official defaults
from models import build_model              # official INTR builder
from util.misc import load_model as official_adapt_detr

DETR_R50_URL = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"


class PerClassPresence(nn.Module):
    """INTR-FC head: logit_c = w_c . z_c + b_c (one vector per class)."""

    def __init__(self, num_classes, hidden_dim):
        super().__init__()
        self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_classes, hidden_dim)))
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, hs):                       # hs: [batch, num_classes, hidden_dim]
        logits = (hs * self.weight).sum(-1) + self.bias
        return logits.unsqueeze(-1)              # INTR.forward squeezes the last dim


def intr_args(num_classes=37, **overrides):
    """Official main.py defaults, with our overrides (e.g. lr, nheads)."""
    args = get_args_parser().parse_args([])
    args.num_queries = num_classes
    args.dataset_name = "pet"                 # not 'cub', so build() keeps num_queries
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def build_intr(num_classes=37, device="cpu", dec_heads=None, head="shared",
               imagenet_init=False, **overrides):
    """Returns (model, criterion, args). With imagenet_init=False the ResNet-50 starts
    random (it is overwritten by DETR or our checkpoint anyway), so nothing is downloaded."""
    args = intr_args(num_classes, device=str(device), **overrides)
    original = official_backbone.is_main_process
    if not imagenet_init:
        official_backbone.is_main_process = lambda: False   # -> resnet50(pretrained=False)
    try:
        model, criterion = build_model(args)
    finally:
        official_backbone.is_main_process = original

    if dec_heads and dec_heads != args.nheads:
        for layer in model.transformer.decoder.layers:
            layer.multihead_attn = nn.MultiheadAttention(args.hidden_dim, dec_heads,
                                                         dropout=args.dropout)
            layer.nhead = dec_heads
    if head == "fc":
        model.presence_vector = PerClassPresence(num_classes, args.hidden_dim)
    args.dec_heads = dec_heads or args.nheads
    args.head = head
    return model.to(device), criterion, args


def load_detr(model, args, source=DETR_R50_URL):
    """Initialise INTR from the COCO-trained DETR-R50 checkpoint (as in the paper):
    detection heads dropped, first `num_queries` object queries kept, new presence vector."""
    if str(source).startswith("http"):
        try:
            ckpt = torch.hub.load_state_dict_from_url(source, map_location="cpu",
                                                      check_hash=True, weights_only=False)
        except TypeError:                     # older torch without weights_only
            ckpt = torch.hub.load_state_dict_from_url(source, map_location="cpu", check_hash=True)
    else:
        ckpt = torch.load(source, map_location="cpu", weights_only=False)
    state = official_adapt_detr(args, ckpt["model"])
    if args.head == "fc":                     # replace the shared-vector init with per-class
        state.pop("presence_vector.weight")
        state.pop("presence_vector.bias")
        fresh = model.presence_vector.state_dict()
        state.update({f"presence_vector.{k}": v for k, v in fresh.items()})
    model.load_state_dict(state, strict=True)
    for p in model.parameters():              # same as official main.py after finetune
        p.requires_grad = True
    return model


MODEL_KEYS = ("num_queries", "nheads", "dec_heads", "head", "enc_layers", "dec_layers",
              "hidden_dim", "dim_feedforward", "dropout")


def save_weights(model, args, path, extra=None):
    """Small file for the app / Hugging Face: weights + what is needed to rebuild."""
    cfg = {k: getattr(args, k) for k in MODEL_KEYS}
    torch.save({"model": model.state_dict(), "config": cfg, **(extra or {})}, path)
    return sha256(path)


def load_weights(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["config"])
    num_classes = cfg.pop("num_queries")
    dec_heads, head = cfg.pop("dec_heads"), cfg.pop("head")
    model, _, args = build_intr(num_classes, device="cpu", dec_heads=dec_heads, head=head, **cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), args, ckpt


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
