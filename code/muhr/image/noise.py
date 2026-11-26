"""Differensiallanuvchi shovqin qatlami N (2.3-§, 2.3-jadval) va kurrikulum strategiyasi (2.4-jadval).

Har bir hujum ikki usulda ishlatiladi:
  * sample(strength) -> tasodifiy parametrlar (o'qitishda);
  * apply(x, cover, **params) -> aniq parametrlar bilan (validatsiyada).
Tasvirlar [0, 1] oralig'ida, shakli [B, 3, H, W].
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- JPEG (STE)
_Q_LUMA = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)
_Q_CHROMA = torch.full((8, 8), 99.0)
_Q_CHROMA[:4, :4] = torch.tensor([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]],
                                 dtype=torch.float32)


def _dct_matrix(n: int = 8) -> torch.Tensor:
    m = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            a = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
            m[k, i] = a * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    return m


_DCT = _dct_matrix()


def _qtable(base: torch.Tensor, quality: int) -> torch.Tensor:
    quality = int(min(max(quality, 1), 100))
    s = 5000 / quality if quality < 50 else 200 - 2 * quality
    return torch.clamp(torch.floor((base * s + 50) / 100), 1, 255)


def round_ste(x: torch.Tensor) -> torch.Tensor:
    """To'g'ridan-to'g'ri o'tkazuvchi baholagich (2.13): oldinga — round, orqaga — gradiyent 1."""
    return x + (torch.round(x) - x).detach()


def _blockwise(x: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    blk = x.view(b, c, h // 8, 8, w // 8, 8).permute(0, 1, 2, 4, 3, 5)
    out = mat @ blk @ mat.transpose(0, 1)
    return out, (b, c, h, w)


def _unblock(blk: torch.Tensor, shape) -> torch.Tensor:
    b, c, h, w = shape
    return blk.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)


def jpeg_differentiable(x: torch.Tensor, quality: int) -> torch.Tensor:
    b, _, h, w = x.shape
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    r, g, bl = x[:, 0:1] * 255, x[:, 1:2] * 255, x[:, 2:3] * 255
    y = 0.299 * r + 0.587 * g + 0.114 * bl
    cb = -0.168736 * r - 0.331264 * g + 0.5 * bl + 128
    cr = 0.5 * r - 0.418688 * g - 0.081312 * bl + 128
    cb, cr = F.avg_pool2d(cb, 2), F.avg_pool2d(cr, 2)          # 4:2:0 pastki diskretlash

    dct = _DCT.to(x.device, x.dtype)
    ql, qc = _qtable(_Q_LUMA, quality).to(x), _qtable(_Q_CHROMA, quality).to(x)

    def codec(ch, q):
        coef, shape = _blockwise(ch - 128, dct)
        coef = round_ste(coef / q) * q
        rec, _ = _blockwise(_unblock(coef, shape), dct.transpose(0, 1))
        return _unblock(rec, shape) + 128

    y, cb, cr = codec(y, ql), codec(cb, qc), codec(cr, qc)
    cb = F.interpolate(cb, scale_factor=2, mode="nearest")
    cr = F.interpolate(cr, scale_factor=2, mode="nearest")
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    bl = y + 1.772 * (cb - 128)
    out = torch.cat([r, g, bl], 1) / 255
    return out[:, :, :h, :w].clamp(0, 1)


# ----------------------------------------------------------------------------- hujumlar
class Attack:
    name = "attack"

    def sample(self, strength: float) -> dict:
        return {}

    def apply(self, x, cover, **p):
        raise NotImplementedError


class Identity(Attack):
    name = "identity"

    def apply(self, x, cover, **p):
        return x


class Jpeg(Attack):
    name = "jpeg"

    def sample(self, s):
        return {"quality": random.randint(int(90 - 60 * s), 90)}

    def apply(self, x, cover, quality=50):
        return jpeg_differentiable(x, quality)


class GaussianNoise(Attack):
    name = "noise"

    def sample(self, s):
        return {"sigma": random.uniform(0, 0.08 * s)}

    def apply(self, x, cover, sigma=0.05):
        return (x + torch.randn_like(x) * sigma).clamp(0, 1)


class GaussianBlur(Attack):
    name = "blur"

    def sample(self, s):
        k = random.choice([3, 5, 7][: 1 + int(round(2 * s))])
        return {"kernel": k, "sigma": random.uniform(0.5, 0.5 + 1.5 * s)}

    def apply(self, x, cover, kernel=5, sigma=1.0):
        ax = torch.arange(kernel, device=x.device, dtype=x.dtype) - (kernel - 1) / 2
        g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        g = (g / g.sum())
        k2 = (g[:, None] * g[None, :]).expand(3, 1, kernel, kernel)
        xp = F.pad(x, [kernel // 2] * 4, mode="reflect")
        return F.conv2d(xp, k2, groups=3)


class CropResize(Attack):
    """Maydonning bir qismini kesib olib, dastlabki o'lchamga qaytarish (kesish + masshtablash)."""
    name = "crop"

    def sample(self, s):
        return {"area": random.uniform(0.9 - 0.5 * s, 1.0)}

    def apply(self, x, cover, area=0.6):
        b, _, h, w = x.shape
        side = math.sqrt(area)
        ch, cw = max(8, int(h * side)), max(8, int(w * side))
        outs = []
        for i in range(b):
            t, l = random.randint(0, h - ch), random.randint(0, w - cw)
            crop = x[i:i + 1, :, t:t + ch, l:l + cw]
            outs.append(F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False))
        return torch.cat(outs, 0)


class Resize(Attack):
    name = "resize"

    def sample(self, s):
        return {"scale": random.uniform(1 - 0.5 * s, 1 + 1.0 * s)}

    def apply(self, x, cover, scale=0.5):
        _, _, h, w = x.shape
        nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
        y = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False, antialias=scale < 1)
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)


class RotateShift(Attack):
    """Affin o'zgartirish (2.14): burish + siljitish."""
    name = "rotate"

    def sample(self, s):
        return {"angle": random.uniform(-30 * s, 30 * s), "shift": 0.1 * s}

    def apply(self, x, cover, angle=15.0, shift=0.0):
        b = x.shape[0]
        a = math.radians(angle)
        theta = torch.zeros(b, 2, 3, device=x.device, dtype=x.dtype)
        theta[:, 0, 0] = math.cos(a); theta[:, 0, 1] = -math.sin(a)
        theta[:, 1, 0] = math.sin(a); theta[:, 1, 1] = math.cos(a)
        if shift:
            theta[:, :, 2] = (torch.rand(b, 2, device=x.device) * 2 - 1) * shift * 2
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class PixelDropout(Attack):
    name = "dropout"

    def sample(self, s):
        return {"p": random.uniform(0, 0.3 * s)}

    def apply(self, x, cover, p=0.3):
        mask = (torch.rand_like(x[:, :1]) < p).expand_as(x)
        return torch.where(mask, cover, x)


class Gamma(Attack):
    name = "gamma"

    def sample(self, s):
        return {"gamma": random.uniform(1 - 0.3 * s, 1 + 0.4 * s)}

    def apply(self, x, cover, gamma=1.2):
        return x.clamp(1e-6, 1) ** gamma


class Combined(Attack):
    """2–3 ta hujumni tasodifiy tartibda ketma-ket qo'llash."""
    name = "combined"

    def __init__(self, pool):
        self.pool = pool

    def sample(self, s):
        k = random.choice([2, 3])
        ops = random.sample(self.pool, k)
        return {"steps": [(op, op.sample(s)) for op in ops]}

    def apply(self, x, cover, steps=()):
        for op, p in steps:
            x = op.apply(x, cover, **p)
        return x


BASE_ATTACKS = [Jpeg(), GaussianNoise(), GaussianBlur(), CropResize(), Resize(), RotateShift(),
                PixelDropout(), Gamma()]
ATTACKS = {a.name: a for a in [Identity(), *BASE_ATTACKS, Combined([a for a in BASE_ATTACKS
                                                                     if a.name != "dropout"])]}

# Validatsiya uchun qat'iy parametrli hujumlar to'plami
VALIDATION_SUITE = [
    ("identity", {}), ("jpeg", {"quality": 50}), ("noise", {"sigma": 0.05}),
    ("blur", {"kernel": 5, "sigma": 1.0}), ("crop", {"area": 0.6}), ("resize", {"scale": 0.5}),
    ("rotate", {"angle": 15.0}), ("dropout", {"p": 0.3}),
]


# ----------------------------------------------------------------------------- kurrikulum
def curriculum_stage(epoch: int, total: int) -> tuple[dict, float]:
    """2.4-jadval: (hujum turlari ehtimolliklari, kuch darajasi) qaytaradi."""
    f = epoch / max(1, total)
    names = [a.name for a in BASE_ATTACKS]
    if f < 0.1:
        probs = {"identity": 0.8, **{n: 0.2 / len(names) for n in names}}
        return probs, 0.3
    if f < 0.3:
        probs = {"identity": 0.4, **{n: 0.6 / len(names) for n in names}}
        return probs, 0.6
    if f < 0.6:
        allnames = ["identity", *names, "combined"]
        return {n: 1 / len(allnames) for n in allnames}, 1.0
    probs = {"identity": 0.05, "combined": 0.35, **{n: 0.6 / len(names) for n in names}}
    return probs, 1.0


class NoiseLayer(nn.Module):
    def __init__(self, total_epochs: int, curriculum: bool = True):
        super().__init__()
        self.total = total_epochs
        self.curriculum = curriculum
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def forward(self, stego, cover):
        if self.curriculum:
            probs, strength = curriculum_stage(self.epoch, self.total)
        else:
            probs, strength = curriculum_stage(self.total, self.total)
        name = random.choices(list(probs), weights=list(probs.values()))[0]
        op = ATTACKS[name]
        return op.apply(stego, cover, **op.sample(strength)), name
