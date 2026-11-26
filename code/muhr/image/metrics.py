"""Yo'qotish funksiyalari va sifat ko'rsatkichlari: PSNR, SSIM (2.5), BER."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _gauss_window(size=11, sigma=1.5, channels=3, device=None, dtype=None):
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(channels, 1, size, size).contiguous()


def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    """O'rtacha SSIM (Wang va boshq., 2004) — [B,C,H,W] tenzorlar uchun, skalyar qaytaradi."""
    c = x.shape[1]
    w = _gauss_window(channels=c, device=x.device, dtype=x.dtype)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_x, mu_y = F.conv2d(x, w, groups=c), F.conv2d(y, w, groups=c)
    sxx = F.conv2d(x * x, w, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, groups=c) - mu_x * mu_y
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean()


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(x, y).item()
    return float("inf") if mse == 0 else 10 * np.log10(data_range ** 2 / mse)


def bit_accuracy(logits: torch.Tensor, msgs: torch.Tensor) -> float:
    return ((logits > 0).float() == msgs).float().mean().item()


def np_psnr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def np_ssim(a: np.ndarray, b: np.ndarray) -> float:
    ta = torch.from_numpy(a.astype(np.float32) / 255).permute(2, 0, 1)[None]
    tb = torch.from_numpy(b.astype(np.float32) / 255).permute(2, 0, 1)[None]
    return float(ssim(ta, tb))
