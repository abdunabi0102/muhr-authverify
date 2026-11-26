"""Shared helpers for Paper 1 experiments (E3, E4, E8).

Key idea: the decoder is keyless, so for every (image, attack) we store the hard bits of ALL
synchronization hypotheses once (GPU). Decisions for any codec / key / tag length / search policy
are then computed offline on CPU (analysis scripts).
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True
from PIL import Image

from muhr.crypto import phash16
from muhr.image.metrics import np_psnr, np_ssim
from muhr.image.watermarker import ImageWatermarker, pil_to_tensor, tensor_to_pil

MAX_HYP = 16          # N_geo <= 16 for 100..4096 px, aspect 1:4..4:1 (methodology.md)


def load_rgb(path, max_side=None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)
    return img


@torch.no_grad()
def embed_bits(wm: ImageWatermarker, img: Image.Image, bits: np.ndarray, mode: str = "global",
               beta: float | None = None) -> tuple[Image.Image, tuple]:
    """Same as ImageWatermarker.embed but with an externally encoded 128-bit word (codec v2)."""
    beta = wm.beta if beta is None else beta
    x = pil_to_tensor(img).to(wm.device)
    _, _, h, w = x.shape
    ny, nx = wm._grid(h, w) if mode == "tiled" else (1, 1)
    tiles = wm._tiles(x, ny, nx)
    msgs = torch.from_numpy(np.asarray(bits)).float().to(wm.device)[None].expand(tiles.shape[0], -1)
    res = wm.enc(tiles, msgs)
    b = wm.block
    res = res.view(1, ny, nx, 3, b, b).permute(0, 3, 1, 4, 2, 5).reshape(1, 3, ny * b, nx * b)
    res = F.interpolate(res, size=(h, w), mode="bilinear", align_corners=False)
    return tensor_to_pil((x + beta * res).clamp(0, 1)), (ny, nx)


@torch.no_grad()
def all_hypotheses(wm: ImageWatermarker, img: Image.Image):
    """Hard bits (MAX_HYP, 128) for every grid hypothesis in the verifier's order; grids (MAX_HYP, 2), -1 padded."""
    img = img.convert("RGB")
    x = pil_to_tensor(img).to(wm.device)
    cands = wm._grid_candidates(img.height, img.width)[:MAX_HYP]
    bits = np.zeros((MAX_HYP, wm.msg_len), np.uint8)
    conf = np.zeros(MAX_HYP, np.float32)
    grids = np.full((MAX_HYP, 2), -1, np.int16)
    for i, (ny, nx) in enumerate(cands):
        p = torch.sigmoid(wm.dec(wm._tiles(x, ny, nx)).mean(0)).cpu().numpy()
        bits[i] = (p > 0.5)
        conf[i] = float(np.mean(np.abs(p - 0.5)) * 2)
        grids[i] = (ny, nx)
    return bits, grids, conf, len(cands)


@torch.no_grad()
def all_hypotheses_batch(wm: ImageWatermarker, imgs, max_n: int = 8, amp: bool = False, max_tiles: int = 64):
    """Batched version of all_hypotheses for several images at once (identical outputs: the decoder is
    in eval mode, so every tile is processed independently). Returns a list of (bits, grids, conf, n)."""
    tiles, owner, meta = [], [], []
    for ii, img in enumerate(imgs):
        img = img.convert("RGB")
        x = pil_to_tensor(img).to(wm.device)
        cands = wm._grid_candidates(img.height, img.width, max_n)[:MAX_HYP]
        meta.append(cands)
        for hi, (ny, nx) in enumerate(cands):
            t = wm._tiles(x, ny, nx)
            tiles.append(t)
            owner += [(ii, hi)] * t.shape[0]
    allt = torch.cat(tiles, 0)
    use_amp = amp and allt.is_cuda
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        logits = torch.cat([wm.dec(allt[s:s + max_tiles]).float() for s in range(0, allt.shape[0], max_tiles)], 0)
    key = torch.tensor([ii * MAX_HYP + hi for ii, hi in owner], device=logits.device)
    sums = torch.zeros(len(imgs) * MAX_HYP, logits.shape[1], device=logits.device).index_add_(0, key, logits)
    cnt = torch.zeros(len(imgs) * MAX_HYP, device=logits.device).index_add_(0, key, torch.ones_like(key, dtype=logits.dtype))
    probs = torch.sigmoid(sums / cnt.clamp(min=1)[:, None]).view(len(imgs), MAX_HYP, -1).cpu().numpy()
    out = []
    for ii, cands in enumerate(meta):
        n = len(cands)
        bits = np.zeros((MAX_HYP, wm.msg_len), np.uint8)
        conf = np.zeros(MAX_HYP, np.float32)
        grids = np.full((MAX_HYP, 2), -1, np.int16)
        bits[:n] = probs[ii, :n] > 0.5
        conf[:n] = np.mean(np.abs(probs[ii, :n] - 0.5), axis=1) * 2
        grids[:n] = np.array(cands, dtype=np.int16)
        out.append((bits, grids, conf, n))
    return out


def amp_check(wm: ImageWatermarker, imgs, max_n: int, threshold: float = 0.995) -> dict:
    """Bit agreement between fp32 and fp16 decoding on sample images. AMP is used only if agreement
    >= threshold (recorded in meta.json)."""
    if not torch.cuda.is_available():
        return {"amp": False, "reason": "no cuda"}
    a = all_hypotheses_batch(wm, imgs, max_n, amp=False)
    b = all_hypotheses_batch(wm, imgs, max_n, amp=True)
    tot = sum(int(x[3]) * wm.msg_len for x in a)
    same = sum(int((x[0][: x[3]] == y[0][: y[3]]).sum()) for x, y in zip(a, b))
    agree = same / max(tot, 1)
    return {"amp": agree >= threshold, "bit_agreement_fp16_vs_fp32": agree, "n_images": len(imgs)}


class LPIPSMeter:
    """LPIPS (AlexNet). Images larger than `max_side` are downscaled for LPIPS only (documented)."""

    def __init__(self, device, max_side=1024):
        self.ok, self.max_side, self.device = False, max_side, device
        try:
            import lpips
            self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
            self.ok = True
        except Exception as e:          # keep running without LPIPS; reported as NaN
            print("LPIPS unavailable:", e)

    @torch.no_grad()
    def __call__(self, a: Image.Image, b: Image.Image) -> float:
        if not self.ok:
            return float("nan")
        if max(a.size) > self.max_side:
            s = self.max_side / max(a.size)
            size = (round(a.width * s), round(a.height * s))
            a, b = a.resize(size, Image.Resampling.BICUBIC), b.resize(size, Image.Resampling.BICUBIC)
        ta = pil_to_tensor(a).to(self.device) * 2 - 1
        tb = pil_to_tensor(b).to(self.device) * 2 - 1
        return float(self.net(ta, tb).item())


def quality(a: Image.Image, b: Image.Image, lp: LPIPSMeter) -> dict:
    A, Bn = np.asarray(a), np.asarray(b)
    return {"psnr": float(np_psnr(A, Bn)), "ssim": float(np_ssim(A, Bn)), "lpips": lp(a, b)}


def env_info() -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "python": platform.python_version(),
            "torch": torch.__version__, "device": gpu, "host": platform.node(), "pid": os.getpid()}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


__all__ = ["MAX_HYP", "load_rgb", "embed_bits", "all_hypotheses", "all_hypotheses_batch", "amp_check", "LPIPSMeter", "quality", "env_info",
           "write_json", "phash16"]
