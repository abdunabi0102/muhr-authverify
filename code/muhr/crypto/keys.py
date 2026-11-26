"""Kalitlarni boshqarish (2.4-§): yagona bosh kalitdan HKDF orqali vazifaviy kalitlar hosil qilinadi."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_SALT = b"muhr-v1"


@dataclass(frozen=True)
class KeySet:
    enc: bytes   # K1 — AES-128 shifrlash kaliti
    mac: bytes   # K2 — HMAC-SHA256 kaliti
    perm: bytes  # K3 — bitlarni o'rin almashtirish kaliti
    text: bytes  # matnli suv belgisi (III bob) uchun kalit


def generate_master_key() -> str:
    """256 bitli tasodifiy bosh kalit (hex ko'rinishida)."""
    return os.urandom(32).hex()


def _derive(master: bytes, info: bytes, length: int) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=_SALT, info=info).derive(master)


def derive_keys(master_hex: str) -> KeySet:
    try:
        master = bytes.fromhex(master_hex.strip())
    except ValueError as e:
        raise ValueError("Bosh kalit hex ko'rinishida bo'lishi kerak") from e
    if len(master) < 16:
        raise ValueError("Bosh kalit kamida 128 bit (32 hex belgi) bo'lishi kerak")
    return KeySet(
        enc=_derive(master, b"enc", 16),
        mac=_derive(master, b"mac", 32),
        perm=_derive(master, b"perm", 32),
        text=_derive(master, b"text", 32),
    )


def load_or_create_master_key(path: str | Path) -> tuple[str, bool]:
    """Kalit faylini o'qiydi; yo'q bo'lsa yangisini yaratadi. (kalit, yangi_yaratildimi) qaytaradi."""
    path = Path(path)
    if path.exists():
        return path.read_text().strip(), False
    path.parent.mkdir(parents=True, exist_ok=True)
    key = generate_master_key()
    path.write_text(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key, True
