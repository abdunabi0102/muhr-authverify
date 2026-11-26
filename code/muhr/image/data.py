"""O'quv to'plami va baholash uchun haqiqiy (PIL) hujumlar."""
from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(root: str | Path) -> list[Path]:
    files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)
    if not files:
        raise FileNotFoundError(f"{root} papkasida tasvir topilmadi")
    return files


class ImageFolderDataset(Dataset):
    """Tasodifiy kesim, aks ettirish va yengil rang o'zgartirish (2.2-§ augmentatsiyasi)."""

    def __init__(self, root, block=256, train=True, limit=None):
        self.files = list_images(root)[:limit] if limit else list_images(root)
        self.block, self.train = block, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        b = self.block
        s = min(img.size)
        if s < b:
            img = img.resize((max(b, round(img.width * b / s)), max(b, round(img.height * b / s))),
                             Image.Resampling.BICUBIC)
        if self.train:
            left, top = random.randint(0, img.width - b), random.randint(0, img.height - b)
        else:
            left, top = (img.width - b) // 2, (img.height - b) // 2
        img = img.crop((left, top, left + b, top + b))
        if self.train:
            if random.random() < 0.5:
                img = ImageOps.mirror(img)
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
        a = np.asarray(img, dtype=np.float32) / 255
        return torch.from_numpy(a).permute(2, 0, 1)


# ----------------------------------------------------------------------------- haqiqiy hujumlar
def _jpeg(img, q):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=int(q)); buf.seek(0)
    return Image.open(buf).convert("RGB")


def _webp(img, q):
    buf = io.BytesIO(); img.save(buf, "WEBP", quality=int(q)); buf.seek(0)
    return Image.open(buf).convert("RGB")


def _noise(img, sigma):
    a = np.asarray(img, dtype=np.float32) / 255
    a = np.clip(a + np.random.normal(0, sigma, a.shape), 0, 1)
    return Image.fromarray((a * 255).round().astype(np.uint8))


def _crop(img, area):
    side = area ** 0.5
    w, h = int(img.width * side), int(img.height * side)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def _scale(img, s):
    return img.resize((max(16, int(img.width * s)), max(16, int(img.height * s))),
                      Image.Resampling.BICUBIC)


def _gamma(img, g):
    a = np.asarray(img, dtype=np.float32) / 255
    return Image.fromarray((np.power(a, g) * 255).round().astype(np.uint8))


def _screenshot(img, _):
    x = _scale(img, 0.8)
    x = _gamma(x, 1.1)
    return _jpeg(x, 90)


REAL_ATTACKS = {
    "yo'q": (lambda img, p: img, [None]),
    "JPEG": (_jpeg, [90, 70, 50, 30]),
    "WebP": (_webp, [80, 50]),
    "Gauss shovqini": (_noise, [0.02, 0.05]),
    "Xiralashtirish": (lambda img, r: img.filter(ImageFilter.GaussianBlur(r)), [1.0, 2.0]),
    "Median filtr": (lambda img, k: img.filter(ImageFilter.MedianFilter(k)), [3, 5]),
    "Kesish": (_crop, [0.8, 0.6, 0.4]),
    "Masshtablash": (_scale, [0.5, 0.75, 1.5]),
    "Burish": (lambda img, a: img.rotate(a, resample=Image.Resampling.BILINEAR), [5, 15, 30]),
    "Gamma": (_gamma, [0.8, 1.3]),
    "Gistogramma tekislash": (lambda img, p: ImageOps.equalize(img), [None]),
    "Ekran surati": (_screenshot, [None]),
    "JPEG50+kesish0.7": (lambda img, p: _jpeg(_crop(img, 0.7), 50), [None]),
}


def apply_real_attack(img: Image.Image, name: str, param) -> Image.Image:
    fn, _ = REAL_ATTACKS[name]
    return fn(img, param)
