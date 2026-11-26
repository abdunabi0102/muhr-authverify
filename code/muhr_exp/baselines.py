"""Deep-watermarking baselines for E2 / E4c / U-cross, loaded through Video Seal's own interface
(facebookresearch/videoseal: docs/baselines.md, videoseal/models/baselines.py, model cards).

Every method is wrapped as a Video Seal `Videoseal`/`Wam` object, so embedding (resize to the
method's processing size, residual upscaled to full resolution, method-specific attenuation and
scaling) and detection follow the authors' published inference code, not our re-implementation.

The search verifier for E4c uses exactly the synchronization hypotheses (ny, nx) of OUR verifier
(ImageWatermarker._grid_candidates with the same max_n), with each baseline's detector as the tile
decoder and the tile outputs averaged — the same aggregation our decoder uses.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import torch
import torch.nn.functional as F

TS_BASELINES = ("hidden", "mbrs", "cin", "trustmark", "wam")      # TorchScript files (HF community repo)
CARDS = {"videoseal": "videoseal", "pixelseal": "pixelseal"}      # official model cards
ALL_METHODS = TS_BASELINES + tuple(CARDS)


@contextmanager
def _cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_method(name: str, vs_root: str, device) -> torch.nn.Module:
    """Load one method from a Video Seal checkout at `vs_root` (ckpts/ and configs/ resolved there)."""
    if vs_root not in sys.path:
        sys.path.insert(0, vs_root)
    from videoseal.models.baselines import build_baseline
    from videoseal.utils.cfg import setup_model_from_model_card
    with _cwd(vs_root):
        if name in TS_BASELINES:
            model = build_baseline(name)
        elif name in CARDS:
            model = setup_model_from_model_card(CARDS[name])
        else:
            raise ValueError(f"unknown method {name}; choose from {ALL_METHODS}")
    model = model.to(device).eval()
    model.method_name = name
    return model


def n_bits(model) -> int:
    torch.manual_seed(0)
    return int(model.get_random_msg(1).shape[-1])


def proc_size(model) -> int:
    return int(model.img_size)


def _pool(preds: torch.Tensor) -> torch.Tensor:
    """Detector output -> (B, 1+k): channel 0 = detection score (zeros for bit-only baselines)."""
    if preds.dim() == 4:
        preds = preds.mean(dim=(-2, -1))
    return preds.float()


@torch.no_grad()
def embed(model, x: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
    """x: (1,3,H,W) in [0,1] on device; msg: (1,k) {0,1}. Returns watermarked (1,3,H,W)."""
    out = model.embed(x, msg.to(x.device), is_video=False) if _is_videoseal(model) else model.embed(x, msg.to(x.device))
    return out["imgs_w"].clamp(0, 1)


@torch.no_grad()
def detect(model, x: torch.Tensor, batch: int = 32) -> torch.Tensor:
    """Native single decode (the method's own resize to its processing size). x: (B,3,H,W)."""
    outs = []
    for s in range(0, x.shape[0], batch):
        xs = x[s:s + batch]
        p = model.detect(xs, is_video=False) if _is_videoseal(model) else model.detect(xs)
        outs.append(_pool(p["preds"]))
    return torch.cat(outs, 0)


def _is_videoseal(model) -> bool:
    return type(model).__name__ == "Videoseal"


def tiles_at(x: torch.Tensor, ny: int, nx: int, S: int) -> torch.Tensor:
    """Same tiling as ImageWatermarker._tiles, but at the baseline's processing size S."""
    r = F.interpolate(x, size=(ny * S, nx * S), mode="bilinear", align_corners=False, antialias=True)
    return r.view(1, 3, ny, S, nx, S).permute(0, 2, 4, 1, 3, 5).reshape(ny * nx, 3, S, S)


@torch.no_grad()
def detect_hypotheses(model, x: torch.Tensor, cands, max_hyp: int, batch: int = 32):
    """Soft outputs (max_hyp, 1+k) for each grid hypothesis (tile outputs averaged); rows >= len(cands) are 0."""
    S = proc_size(model)
    tiles, owner = [], []
    for hi, (ny, nx) in enumerate(cands[:max_hyp]):
        t = tiles_at(x, ny, nx, S)
        tiles.append(t)
        owner += [hi] * t.shape[0]
    allt = torch.cat(tiles, 0)
    logits = detect(model, allt, batch)
    key = torch.tensor(owner, device=logits.device)
    sums = torch.zeros(max_hyp, logits.shape[1], device=logits.device).index_add_(0, key, logits)
    cnt = torch.zeros(max_hyp, device=logits.device).index_add_(0, key, torch.ones_like(key, dtype=logits.dtype))
    return sums / cnt.clamp(min=1)[:, None]


__all__ = ["TS_BASELINES", "CARDS", "ALL_METHODS", "load_method", "n_bits", "proc_size", "embed", "detect",
           "tiles_at", "detect_hypotheses"]
