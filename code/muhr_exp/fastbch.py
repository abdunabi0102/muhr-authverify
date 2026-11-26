"""Vectorised bounded-distance BCH decoder (Berlekamp-Massey + Chien) for many words at once.

Drop-in replacement for `galois.BCH.decode(..., errors=True)` on shortened binary BCH codes over
GF(2^m), used only for large offline FA analyses (hundreds of thousands of words). It takes the
field, the roots and the code layout from the `galois.BCH` object itself and is validated against
galois word by word (tools: test_fastbch in scripts/analyze_fa.py --selftest).

Layout (galois convention): word index 0 is the highest-degree coefficient; a shortened word of
length n_use is the full codeword with n - n_use leading zeros removed; systematic, message first.
"""
from __future__ import annotations

import numpy as np


class FastBCH:
    def __init__(self, bch, n_use: int, k_use: int):
        self.n, self.t, self.n_use, self.k_use = bch.n, bch.t, n_use, k_use
        GF = bch.extension_field
        order = GF.order - 1                                    # 2^m - 1
        alpha = GF.primitive_element
        self.order = order
        self.exp = np.array([int(alpha ** i) for i in range(order)] * 2 + [1], dtype=np.int32)
        self.log = np.full(order + 1, -1, dtype=np.int32)
        for i in range(order):
            self.log[self.exp[i]] = i
        roots = [int(r) for r in bch.roots][: 2 * self.t]       # alpha^c ... alpha^(c+2t-1)
        self.root_logs = np.array([self.log[r] for r in roots], dtype=np.int64)
        # syndrome matrix over GF(2): S_j = sum_i r_i * root_j^(deg_i), deg_i = n_use-1-i
        deg = np.arange(self.n_use - 1, -1, -1, dtype=np.int64)
        m = int(np.log2(GF.order))
        pw = self.exp[(self.root_logs[None, :] * deg[:, None]) % order]          # (n_use, 2t) elements
        self.m = m
        self.synd_bits = ((pw[..., None] >> np.arange(m)) & 1).reshape(self.n_use, -1).astype(np.uint8)

    # GF helpers on int arrays
    def _mul(self, a, b):
        la, lb = self.log[a], self.log[b]
        r = self.exp[(la + lb) % self.order]
        return np.where((a == 0) | (b == 0), 0, r)

    def _div(self, a, b):                                       # b != 0 assumed
        la, lb = self.log[a], self.log[b]
        r = self.exp[(la - lb) % self.order]
        return np.where(a == 0, 0, r)

    def decode(self, words: np.ndarray, chunk: int = 20000):
        words = np.asarray(words, dtype=np.uint8).reshape(-1, self.n_use)
        msgs = np.empty((len(words), self.k_use), np.uint8)
        nerr = np.empty(len(words), np.int64)
        for s in range(0, len(words), chunk):
            msgs[s:s + chunk], nerr[s:s + chunk] = self._decode(words[s:s + chunk])
        return msgs, nerr

    def _decode(self, w: np.ndarray):
        N, t2 = len(w), 2 * self.t
        sb = (w.astype(np.int64) @ self.synd_bits.astype(np.int64)) & 1          # (N, 2t*m)
        S = (sb.reshape(N, t2, self.m) << np.arange(self.m)).sum(-1).astype(np.int32)   # (N, 2t)
        L = np.zeros(N, np.int64)
        C = np.zeros((N, t2 + 1), np.int32); C[:, 0] = 1
        Bp = np.zeros((N, t2 + 1), np.int32); Bp[:, 0] = 1
        mshift = np.ones(N, np.int64)
        b = np.ones(N, np.int32)
        cols = np.arange(t2 + 1)
        for r in range(t2):
            d = S[:, r].copy()
            for i in range(1, r + 1):
                d ^= self._mul(C[:, i], S[:, r - i])
            nz = d != 0
            coef = np.where(nz, self._div(d, b), 0)
            src = cols[None, :] - mshift[:, None]
            Bs = np.where(src >= 0, np.take_along_axis(Bp, np.clip(src, 0, t2), 1), 0)
            Cn = C ^ self._mul(coef[:, None], Bs)
            upd = nz & (2 * L <= r)
            Bp = np.where(upd[:, None], C, Bp)
            b = np.where(upd, d, b)
            L = np.where(upd, r + 1 - L, L)
            mshift = np.where(upd, 1, mshift + 1)
            C = np.where(nz[:, None], Cn, C)
        # Chien search over all n degrees (roots in the shortened region -> failure)
        p = np.arange(self.n, dtype=np.int64)
        val = np.zeros((N, self.n), np.int32)
        for i in range(t2 + 1):
            ai = self.exp[(-p * i) % self.order]                              # alpha^(-p i)
            val ^= self._mul(C[:, i:i + 1], ai[None, :])
        roots = val == 0                                                      # error at degree p
        nroots = roots.sum(1)
        inside = roots[:, : self.n_use].sum(1)
        ok = (L <= self.t) & (nroots == L) & (inside == L)
        flip = roots[:, : self.n_use][:, ::-1]                                # degree p -> index n_use-1-p
        corr = np.where(ok[:, None], w ^ flip.astype(np.uint8), w)
        return corr[:, : self.k_use], np.where(ok, L, -1)
