"""Kriptografik mustahkamlash operatori Ω va uning teskarisi (2.4-§, 2.3-rasm).

Joylashtirish:  W -> tau = HMAC_K2(W)[:32 bit] -> W_e = AES-CTR_K1(W, IV = tau)
                -> BCH(W_e || tau) -> pi_K3(...) -> W~ (msg_len bit)
Tekshirish:     teskari tartibda; HMAC mos kelmasa belgi rad etiladi.

Eslatma: tau ham autentifikatsiya kodi, ham CTR rejimi uchun boshlang'ich vektor (SIV sxemasi).
Bu bir xil nonce qayta ishlatilishi xavfini yo'qotadi va qo'shimcha bit talab qilmaydi.
"""
from __future__ import annotations

import hmac
import hashlib
from dataclasses import dataclass, field

import bchlib
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from .keys import KeySet
from .payload import PAYLOAD_BYTES, Payload

TAG_BYTES = 4                                   # 32 bitli qisqartirilgan HMAC
DATA_BYTES = PAYLOAD_BYTES + TAG_BYTES          # 12 bayt = 96 bit himoyalanuvchi blok
DATA_BITS = DATA_BYTES * 8
_MAC_DOMAIN = b"muhr-img-v1|"


def bytes_to_bits(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes(b), dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, np.uint8)])
    return np.packbits(bits).tobytes()


def _choose_bch(msg_len: int) -> tuple[int, int]:
    """msg_len ga sig'adigan eng kuchli BCH(m, t) parametrlarini tanlaydi."""
    best = None
    for m in range(7, 11):
        n = 2 ** m - 1
        cap = min(n, msg_len)
        t = (cap - DATA_BITS) // m
        if t >= 1 and (best is None or t > best[1]):
            best = (m, t)
    if best is None:
        raise ValueError(f"msg_len={msg_len} juda kichik: kamida {DATA_BITS + 7} bit kerak")
    return best


@dataclass
class DecodeResult:
    ok: bool
    reason: str = ""
    payload: Payload | None = None
    corrected_bits: int = 0
    confidence: float = 0.0
    extra: dict = field(default_factory=dict)


class CryptoCodec:
    def __init__(self, keys: KeySet, msg_len: int):
        self.keys = keys
        self.msg_len = msg_len
        self.m, self.t = _choose_bch(msg_len)
        self.bch = bchlib.BCH(self.t, m=self.m)
        self.ecc_bits = self.bch.ecc_bits
        self.code_bits = DATA_BITS + self.ecc_bits
        seed = int.from_bytes(keys.perm[:8], "little")
        self.perm = np.random.default_rng(seed).permutation(msg_len)

    def describe(self) -> str:
        return (f"BCH(n={self.bch.n}, m={self.m}, t={self.t}): {DATA_BITS} ma'lumot + "
                f"{self.ecc_bits} tuzatuvchi bit, {self.msg_len - self.code_bits} to'ldiruvchi bit; "
                f"{self.t} tagacha bit xatolik tuzatiladi")

    # --- kriptografik qatlam ---
    def _tag(self, pb: bytes) -> bytes:
        return hmac.new(self.keys.mac, _MAC_DOMAIN + pb, hashlib.sha256).digest()[:TAG_BYTES]

    def _ctr(self, tag: bytes, data: bytes) -> bytes:
        iv = tag + b"\x00" * (16 - TAG_BYTES)
        c = Cipher(algorithms.AES(self.keys.enc), modes.CTR(iv)).encryptor()
        return c.update(data) + c.finalize()

    # --- Ω ---
    def encode(self, payload: Payload) -> np.ndarray:
        pb = payload.to_bytes()
        tag = self._tag(pb)
        data = self._ctr(tag, pb) + tag
        ecc = self.bch.encode(bytearray(data))
        bits = np.concatenate([bytes_to_bits(data), bytes_to_bits(ecc)[: self.ecc_bits]])
        bits = np.concatenate([bits, np.zeros(self.msg_len - len(bits), np.uint8)])
        return bits[self.perm].astype(np.uint8)

    def decode(self, probs: np.ndarray) -> DecodeResult:
        """probs: dekoderdan olingan har bir bitning 1 bo'lish ehtimoli (uzunligi msg_len)."""
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        if probs.size != self.msg_len:
            return DecodeResult(False, f"kutilgan {self.msg_len} bit, olingan {probs.size}")
        conf = float(np.mean(np.abs(probs - 0.5)) * 2)
        rx = (probs > 0.5).astype(np.uint8)
        bits = np.empty_like(rx)
        bits[self.perm] = rx
        data = bytearray(bits_to_bytes(bits[:DATA_BITS]))
        ecc_rx = np.zeros(self.bch.ecc_bytes * 8, np.uint8)
        ecc_rx[: self.ecc_bits] = bits[DATA_BITS: DATA_BITS + self.ecc_bits]
        ecc = bytearray(bits_to_bytes(ecc_rx))
        nerr = self.bch.decode(data, ecc)
        if nerr < 0:
            return DecodeResult(False, "BCH: xatoliklar tuzatish imkoniyatidan ko'p", confidence=conf)
        self.bch.correct(data, ecc)
        ct, tag = bytes(data[:PAYLOAD_BYTES]), bytes(data[PAYLOAD_BYTES:])
        pb = self._ctr(tag, ct)
        if not hmac.compare_digest(self._tag(pb), tag):
            return DecodeResult(False, "HMAC mos kelmadi: belgi soxta yoki kalit boshqa",
                                corrected_bits=nerr, confidence=conf)
        try:
            payload = Payload.from_bytes(pb)
        except ValueError as e:
            return DecodeResult(False, f"Yuklama noto'g'ri: {e}", confidence=conf)
        return DecodeResult(True, "", payload, corrected_bits=nerr, confidence=conf)
