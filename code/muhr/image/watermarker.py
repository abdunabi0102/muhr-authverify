"""O'qitilgan model asosida tasvirni muhrlash va tekshirish.

Ixtiyoriy o'lchamdagi tasvir blok o'lchamidagi to'rga keltiriladi, belgi har bir blokka takroriy
joylashtiriladi (fazoviy zaxiralash). Tekshirishda bir necha to'r gipotezasi sinab ko'riladi,
bloklar logitlari o'rtalashtirilib (ko'pchilik ovozi, 2.15) kriptografik dekoderga uzatiladi.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..crypto import CryptoCodec, KeySet, Payload, hamming16, phash16
from .metrics import np_psnr, np_ssim
from .models import Decoder, Encoder

HASH_TOLERANCE = 3   # idrok xeshining ruxsat etilgan Xemming farqi


@dataclass
class ExtractResult:
    ok: bool
    reason: str
    payload: Payload | None = None
    content_match: bool | None = None
    hash_distance: int | None = None
    corrected_bits: int = 0
    confidence: float = 0.0
    grid: tuple = ()
    probs: np.ndarray | None = field(default=None, repr=False)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1)[None]


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    a = (t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a)


class ImageWatermarker:
    def __init__(self, ckpt_path: str, keys: KeySet, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        self.block = cfg["block"]
        self.msg_len = cfg["msg_len"]
        self.beta = cfg.get("beta", 1.0)
        self.enc = Encoder(self.msg_len, cfg["base"]).to(self.device).eval()
        self.dec = Decoder(self.msg_len, cfg["base"], use_stn=cfg.get("stn", True)).to(self.device).eval()
        self.enc.load_state_dict(ck["encoder"])
        self.dec.load_state_dict(ck["decoder"])
        self.codec = CryptoCodec(keys, self.msg_len)
        self.meta = {k: ck.get(k) for k in ("epoch", "val")}

    def _grid(self, h, w):
        return max(1, round(h / self.block)), max(1, round(w / self.block))

    def _grid_candidates(self, h, w, max_n=8):
        """Sinxronizatsiyani qidirish (2.3-§): tasvir masshtablangan yoki o'lchami o'zgargan bo'lsa,
        muhrlashdagi bloklar to'ri (ny, nx) asl o'lchamga bog'liq, joriy o'lchamga emas. Masshtablash
        tomonlar nisbatini saqlagani uchun asl balandlik H0 ning har bir ehtimoliy qiymatiga
        W0 = H0 * w/h mos keladi; shu orqali (ny, nx) gipotezalari to'plami tuziladi."""
        b, ratio = self.block, w / h
        g0 = self._grid(h, w)
        cands = {g0, (1, 1)}
        for ny in range(1, max_n + 1):
            for f in np.linspace(-0.49, 0.49, 7):
                h0 = max(1.0, (ny + f) * b)
                cands.add((ny, max(1, round(h0 * ratio / b))))
        cands = [c for c in cands if c[1] <= max_n]
        return sorted(cands, key=lambda c: (c != g0, c != (1, 1), abs(c[0] - g0[0]) + abs(c[1] - g0[1]), c))

    def _tiles(self, x: torch.Tensor, ny: int, nx: int) -> torch.Tensor:
        b = self.block
        r = F.interpolate(x, size=(ny * b, nx * b), mode="bilinear", align_corners=False,
                          antialias=True)
        t = r.view(1, 3, ny, b, nx, b).permute(0, 2, 4, 1, 3, 5).reshape(ny * nx, 3, b, b)
        return t

    @torch.no_grad()
    def embed(self, img: Image.Image, author_id: int, beta: float | None = None,
              on: date | None = None, mode: str = "tiled") -> tuple[Image.Image, dict]:
        """mode="tiled" — tasvir 256 px li bloklar to'riga bo'linadi, belgi har blokda takrorlanadi;
        mode="global" — butun tasvir bitta blok sifatida (256x256 ga keltirilib) muhrlanadi, qoldiq
        asl o'lchamga kattalashtiriladi. Global rejimda butun tasvirga qo'llangan geometrik
        hujumlar o'qitishdagi hujumlar bilan aynan bir xil ko'rinishga ega bo'ladi."""
        if mode not in ("tiled", "global"):
            raise ValueError("mode 'tiled' yoki 'global' bo'lishi kerak")
        beta = self.beta if beta is None else beta
        img = img.convert("RGB")
        payload = Payload.create(author_id, phash16(img), on)
        bits = self.codec.encode(payload)
        x = pil_to_tensor(img).to(self.device)
        _, _, h, w = x.shape
        ny, nx = self._grid(h, w) if mode == "tiled" else (1, 1)
        tiles = self._tiles(x, ny, nx)
        msgs = torch.from_numpy(bits).float().to(self.device)[None].expand(tiles.shape[0], -1)
        res = self.enc(tiles, msgs)
        b = self.block
        res = res.view(1, ny, nx, 3, b, b).permute(0, 3, 1, 4, 2, 5).reshape(1, 3, ny * b, nx * b)
        res = F.interpolate(res, size=(h, w), mode="bilinear", align_corners=False)
        stego = tensor_to_pil((x + beta * res).clamp(0, 1))
        a, s = np.asarray(img), np.asarray(stego)
        info = {"payload": payload, "psnr": np_psnr(a, s), "ssim": np_ssim(a, s),
                "grid": (ny, nx), "mode": mode, "bits": bits, "codec": self.codec.describe()}
        return stego, info

    @torch.no_grad()
    def soft_bits(self, img: Image.Image, ny: int, nx: int) -> np.ndarray:
        x = pil_to_tensor(img).to(self.device)
        logits = self.dec(self._tiles(x, ny, nx))
        return torch.sigmoid(logits.mean(0)).cpu().numpy()     # bloklar bo'yicha ovoz berish

    @torch.no_grad()
    def extract(self, img: Image.Image) -> ExtractResult:
        img = img.convert("RGB")
        h, w = img.height, img.width
        best = None
        # har bir gipoteza HMAC bilan tekshiriladi: yolg'on tasdiqlash ehtimoli <= gipotezalar soni * 2^-32
        for g in self._grid_candidates(h, w):
            probs = self.soft_bits(img, *g)
            r = self.codec.decode(probs)
            res = ExtractResult(r.ok, r.reason, r.payload, corrected_bits=r.corrected_bits,
                                confidence=r.confidence, grid=g, probs=probs)
            if r.ok:
                d = hamming16(phash16(img), r.payload.phash16)
                res.hash_distance, res.content_match = d, d <= HASH_TOLERANCE
                return res
            if best is None or res.confidence > best.confidence:
                best = res
        return best
