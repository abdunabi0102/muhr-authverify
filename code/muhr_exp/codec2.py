"""Codec v2 for Paper 1 experiments (C1', Proposition 5).

Differences from muhr.crypto.codec.CryptoCodec (v1, kept unchanged for dissertation reproducibility):
  * bit-level layout: payload fields, tag length and BCH code are configurable;
  * BCH over GF(2^7) via `galois` (bit-level shortening, no byte padding);
  * optional decoding radius t_dec <= t (accept only if <= t_dec errors were corrected);
  * keyed whitening: the codeword is XOR-ed with a key-derived mask before the keyed permutation,
    so that for any decoder output chosen without the key the BCH acceptance probability equals the
    sphere-packing value 2^(k-n) * V(n, t_dec) (Proposition 5);
  * optional truncated tag check (for the scaled-tag validation E4b).

Encode:  P -> tau = HMAC_K2(dom || cfg || P)[:tag_bits]
            C = P xor AES-CTR_K1(IV = tau || 0)[:|P|]
            cw = BCH(C || tau)                       (n_use = |P| + tag_bits + parity bits)
            w  = (cw || 0^pad) xor M_K  (if whiten)  -> permute with pi_K3  -> 128 bits
"""
from __future__ import annotations

import hashlib
import os
import hmac
from dataclasses import dataclass, field
from math import comb

import galois

from .fastbch import FastBCH
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# n = 127 narrow-sense binary BCH codes: k -> designed t
BCH127 = {99: 4, 92: 5, 85: 6, 78: 7, 71: 9, 64: 10}
_DOM = b"muhr-img-v2|"


def _bits(b: bytes, n: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))[:n].astype(np.uint8)


def _pack(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


@dataclass(frozen=True)
class CodecConfig:
    name: str = "A"
    id_bits: int = 24
    day_bits: int = 20
    hash_bits: int = 16
    ver_bits: int = 4
    tag_bits: int = 32
    k_full: int = 99          # BCH(127, k_full)
    t_dec: int | None = None  # decoding radius; None = designed t
    whiten: bool = True
    msg_len: int = 128

    @property
    def payload_bits(self) -> int:
        return self.id_bits + self.day_bits + self.hash_bits + self.ver_bits

    @property
    def k_use(self) -> int:
        return self.payload_bits + self.tag_bits

    @property
    def t(self) -> int:
        return BCH127[self.k_full]

    @property
    def radius(self) -> int:
        return self.t if self.t_dec is None else min(self.t_dec, self.t)

    @property
    def n_use(self) -> int:
        return self.k_use + (127 - self.k_full)

    def p_bch(self) -> float:
        """Probability that a uniformly random n_use-bit word is accepted by bounded-distance decoding."""
        n, k, r = self.n_use, self.k_use, self.radius
        return sum(comb(n, i) for i in range(r + 1)) * 2.0 ** (k - n)

    def fa_bound(self, n_geo: int = 16, n_c: int = 1) -> float:
        return n_geo * n_c * self.p_bch() * 2.0 ** (-self.tag_bits)

    def check(self):
        if self.k_full not in BCH127:
            raise ValueError(f"k_full must be one of {sorted(BCH127)}")
        if self.k_use > self.k_full:
            raise ValueError(f"payload+tag = {self.k_use} > k = {self.k_full}")
        if self.n_use > self.msg_len:
            raise ValueError("codeword longer than channel")
        if self.tag_bits > 128 or self.tag_bits < 1:
            raise ValueError("tag_bits out of range")


# Designs from provenance_report.md section 3 (G uses the (127,71) code, radius 8 or 9).
DESIGNS = {
    "A":  CodecConfig("A", 24, 20, 16, 4, 32, 99),
    "C":  CodecConfig("C", 24, 20, 16, 4, 25, 92),
    "E":  CodecConfig("E", 24, 16, 16, 0, 29, 85),
    "F":  CodecConfig("F", 24, 16, 12, 0, 26, 78),
    "G8": CodecConfig("G8", 24, 16, 8, 0, 23, 71, t_dec=8),
    "G9": CodecConfig("G9", 24, 16, 8, 0, 23, 71, t_dec=9),
}


@dataclass
class Decoded:
    ok: bool
    reason: str = ""
    fields: dict = field(default_factory=dict)
    n_err: int = -1


class CodecV2:
    def __init__(self, keys, cfg: CodecConfig):
        cfg.check()
        self.cfg, self.keys = cfg, keys
        self.bch = galois.BCH(127, cfg.k_full)
        self.s = cfg.k_full - cfg.k_use            # shortening
        self._fast = None
        seed = int.from_bytes(keys.perm[:8], "little")
        self.perm = np.random.default_rng(seed).permutation(cfg.msg_len)   # same pi_K3 as v1
        # layout string (independent of the design name and of the decoding radius, so that e.g. G8 and
        # G9 share one codeword); used for MAC domain separation and for the whitening mask
        self._cfgtag = (f"{cfg.id_bits},{cfg.day_bits},{cfg.hash_bits},{cfg.ver_bits},"
                        f"{cfg.tag_bits},{cfg.k_full}|").encode()
        kw = hmac.new(keys.perm, b"muhr-whiten-key", hashlib.sha256).digest()
        mask = hmac.new(kw, b"mask|" + self._cfgtag, hashlib.sha256).digest()
        self.mask = _bits(mask, cfg.msg_len) if cfg.whiten else np.zeros(cfg.msg_len, np.uint8)

    # ---------------------------------------------------------------- payload <-> bits
    def payload_to_bits(self, author_id: int, day: int, phash: int, version: int = 0) -> np.ndarray:
        c = self.cfg
        out = []
        for v, nb in ((author_id, c.id_bits), (day, c.day_bits), (phash >> (16 - c.hash_bits) if c.hash_bits else 0,
                                                                   c.hash_bits), (version, c.ver_bits)):
            if nb:
                v = int(v) & ((1 << nb) - 1)
                out.append(np.array([(v >> (nb - 1 - i)) & 1 for i in range(nb)], np.uint8))
        return np.concatenate(out) if out else np.zeros(0, np.uint8)

    def bits_to_fields(self, bits: np.ndarray) -> dict:
        c, pos, res = self.cfg, 0, {}
        for name, nb in (("author_id", c.id_bits), ("day", c.day_bits), ("hash", c.hash_bits), ("version", c.ver_bits)):
            v = 0
            for b in bits[pos:pos + nb]:
                v = (v << 1) | int(b)
            res[name], pos = v, pos + nb
        return res

    # ---------------------------------------------------------------- crypto
    def _tag(self, pbits: np.ndarray) -> np.ndarray:
        d = hmac.new(self.keys.mac, _DOM + self._cfgtag + bytes([len(pbits)]) + _pack(pbits), hashlib.sha256).digest()
        return _bits(d, self.cfg.tag_bits)

    def _keystream(self, tag: np.ndarray, n: int) -> np.ndarray:
        iv = _pack(np.concatenate([tag, np.zeros(128 - len(tag), np.uint8)]))
        enc = Cipher(algorithms.AES(self.keys.enc), modes.CTR(iv)).encryptor()
        return _bits(enc.update(b"\x00" * ((n + 7) // 8)) + enc.finalize(), n)

    # ---------------------------------------------------------------- encode / decode
    def encode_bits(self, pbits: np.ndarray) -> np.ndarray:
        c = self.cfg
        tag = self._tag(pbits)
        ct = pbits ^ self._keystream(tag, len(pbits))
        msg = galois.GF2(np.concatenate([ct, tag]).astype(int))
        cw = np.asarray(self.bch.encode(msg), dtype=np.uint8)            # length n_use
        word = np.concatenate([cw, np.zeros(c.msg_len - len(cw), np.uint8)]) ^ self.mask
        return word[self.perm].astype(np.uint8)

    def encode(self, author_id: int, day: int, phash: int, version: int = 0) -> np.ndarray:
        return self.encode_bits(self.payload_to_bits(author_id, day, phash, version))

    def unwhiten(self, hard: np.ndarray) -> np.ndarray:
        """hard: (..., 128) received bits in channel order -> (..., n_use) codeword candidates."""
        hard = np.asarray(hard, dtype=np.uint8)
        inv = np.empty_like(hard)
        inv[..., self.perm] = hard
        return (inv ^ self.mask)[..., : self.cfg.n_use]

    def bch_decode_many(self, words: np.ndarray):
        """Vectorised BCH decode. words: (N, n_use). Returns (msgs (N, k_use), n_err (N,)) with radius applied."""
        words = np.asarray(words, dtype=np.uint8).reshape(-1, self.cfg.n_use)
        if os.environ.get("MUHR_BCH", "fast") == "galois" or len(words) < 8:
            msgs, nerr = self.bch.decode(galois.GF2(words.astype(int)), errors=True)
            msgs, nerr = np.asarray(msgs, dtype=np.uint8), np.asarray(nerr, dtype=int)
        else:          # vectorised BM + Chien, validated word-for-word against galois (experiment_log)
            if self._fast is None:
                self._fast = FastBCH(self.bch, self.cfg.n_use, self.cfg.k_use)
            msgs, nerr = self._fast.decode(words)
        nerr = np.where(nerr > self.cfg.radius, -1, nerr)
        return msgs, nerr

    def verify_msg(self, msg: np.ndarray, check_bits: int | None = None) -> Decoded:
        """msg = decoded C || tau. check_bits < tag_bits truncates the comparison (E4b)."""
        c = self.cfg
        ct, tag = msg[: c.payload_bits], msg[c.payload_bits: c.k_use]
        pbits = ct ^ self._keystream(tag, c.payload_bits)
        exp = self._tag(pbits)
        kb = c.tag_bits if check_bits is None else min(check_bits, c.tag_bits)
        if not hmac.compare_digest(_pack(exp[:kb]), _pack(tag[:kb])):
            return Decoded(False, "mac")
        return Decoded(True, "", self.bits_to_fields(pbits))

    def decode_hard(self, hard: np.ndarray, check_bits: int | None = None) -> Decoded:
        msgs, nerr = self.bch_decode_many(self.unwhiten(hard)[None])
        if nerr[0] < 0:
            return Decoded(False, "bch")
        d = self.verify_msg(msgs[0], check_bits)
        d.n_err = int(nerr[0])
        return d

    def decode(self, probs: np.ndarray, check_bits: int | None = None) -> Decoded:
        return self.decode_hard((np.asarray(probs) > 0.5).astype(np.uint8), check_bits)
