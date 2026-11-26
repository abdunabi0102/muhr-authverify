"""Foydali yuklama (2.5-jadval): ID_m (24) | kun (20) | idrok xeshi (16) | versiya (4) = 64 bit."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
from PIL import Image
from scipy.fft import dctn

EPOCH = date(2020, 1, 1)
ID_BITS, DAY_BITS, HASH_BITS, VER_BITS = 24, 20, 16, 4
PAYLOAD_BYTES = 8
CURRENT_VERSION = 1


@dataclass(frozen=True)
class Payload:
    author_id: int
    day: int
    phash16: int
    version: int = CURRENT_VERSION

    def __post_init__(self):
        if not 0 <= self.author_id < 2 ** ID_BITS:
            raise ValueError(f"author_id 0..{2**ID_BITS - 1} oralig'ida bo'lishi kerak")
        if not 0 <= self.day < 2 ** DAY_BITS:
            raise ValueError("Sana diapazondan tashqarida")
        if not 0 <= self.phash16 < 2 ** HASH_BITS:
            raise ValueError("phash16 16 bitli bo'lishi kerak")
        if not 0 <= self.version < 2 ** VER_BITS:
            raise ValueError("Versiya 4 bitli bo'lishi kerak")

    @staticmethod
    def create(author_id: int, phash16: int, on: date | None = None) -> "Payload":
        on = on or date.today()
        return Payload(author_id=author_id, day=(on - EPOCH).days, phash16=phash16)

    @property
    def date(self) -> date:
        return EPOCH + timedelta(days=self.day)

    def to_bytes(self) -> bytes:
        v = (self.author_id << (DAY_BITS + HASH_BITS + VER_BITS)) \
            | (self.day << (HASH_BITS + VER_BITS)) \
            | (self.phash16 << VER_BITS) | self.version
        return v.to_bytes(PAYLOAD_BYTES, "big")

    @staticmethod
    def from_bytes(b: bytes) -> "Payload":
        v = int.from_bytes(b, "big")
        return Payload(
            author_id=(v >> (DAY_BITS + HASH_BITS + VER_BITS)) & (2 ** ID_BITS - 1),
            day=(v >> (HASH_BITS + VER_BITS)) & (2 ** DAY_BITS - 1),
            phash16=(v >> VER_BITS) & (2 ** HASH_BITS - 1),
            version=v & (2 ** VER_BITS - 1),
        )


# Zigzag tartibida past chastotali koeffitsiyentlar (DC dan tashqari)
_ZIGZAG = sorted(((i, j) for i in range(6) for j in range(6) if (i, j) != (0, 0)),
                 key=lambda p: (p[0] + p[1], p[0]))


def phash16(img: Image.Image) -> int:
    """DCT asosidagi idrok xeshining 16 bitli qisqartmasi.

    Qayta siqish va masshtablashga sezgir emas, mazmunan boshqa tasvir uchun esa keskin farq qiladi.
    """
    g = np.asarray(img.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    c = dctn(g, norm="ortho")
    coeffs = np.array([c[i, j] for i, j in _ZIGZAG[:24]])
    bits = coeffs[:16] > np.median(coeffs)
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming16(a: int, b: int) -> int:
    return bin((a ^ b) & 0xFFFF).count("1")
