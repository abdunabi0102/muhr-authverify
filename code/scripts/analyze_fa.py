"""Offline FA analysis (E4a, E4b) from e4_unmarked.py outputs. CPU only.

For each fresh key, design and whitening setting:
  * BCH pass rate of all hypothesis words (empirical p_BCH) vs the sphere-packing value;
  * image-level accepts with the full tag (expected 0) and with truncated tags (8, 10, 12, 16 bits),
    compared with the prediction sum_over_images n_hyp * p_BCH * 2^-k and an exact Poisson interval.
Results are written to <out>/fa_results.csv and fa_results.md; every number comes from this run.

Example: python scripts/analyze_fa.py --e4 e4_unmarked --keys 5 --designs A,G9 --out results/fa
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muhr.crypto import derive_keys, generate_master_key        # noqa: E402
from muhr_exp.codec2 import DESIGNS, CodecV2                    # noqa: E402
from muhr_exp.stats import clopper_pearson, poisson_interval    # noqa: E402

TRUNC = (8, 10, 12, 16)


def load(e4: Path):
    words, owner, natural = [], [], []
    base = 0
    for fn in sorted(e4.glob("chunk_*.npz")):
        z = np.load(fn, allow_pickle=False)
        hyp, nhyp, names = z["hyp"], z["nhyp"], z["names"]
        N, T = nhyp.shape
        for i in range(N):
            for t in range(T):
                for h in range(int(nhyp[i, t])):
                    words.append(hyp[i, t, h]); owner.append((base + i) * T + t)
            natural.append(not str(names[i]).startswith("degenerate_"))
        base += N
    words = np.unpackbits(np.stack(words), axis=1)
    return words, np.array(owner), np.array(natural), T


_CACHE = {}


def _codec(key_hex, design, whiten):
    k = (key_hex, design, whiten)
    if k not in _CACHE:
        _CACHE[k] = CodecV2(derive_keys(key_hex), replace(DESIGNS[design], whiten=whiten))
    return _CACHE[k]


def _work(args):
    key_hex, design, whiten, words = args
    codec = _codec(key_hex, design, whiten)
    msgs, nerr = codec.bch_decode_many(codec.unwhiten(words))
    passed = np.nonzero(nerr >= 0)[0]
    acc = {k: [] for k in ("full", *TRUNC)}
    for i in passed:
        if codec.verify_msg(msgs[i]).ok:
            acc["full"].append(i)
        for k in TRUNC:
            if k < codec.cfg.tag_bits and codec.verify_msg(msgs[i], check_bits=k).ok:
                acc[k].append(i)
    return passed, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e4", required=True)
    p.add_argument("--keys", type=int, default=5)
    p.add_argument("--designs", default="A,G8,G9")
    p.add_argument("--whiten", default="1,0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    words, owner, natural, T = load(Path(a.e4))
    img_of_owner = owner // T
    nat_word = natural[img_of_owner]
    n_it = {"natural": int(natural.sum()) * T, "degenerate": int((~natural).sum()) * T}
    print(f"{len(words)} hypothesis words, {len(natural)} images x {T} transforms ({time.time()-t0:.0f}s)")
    keys = [generate_master_key() for _ in range(a.keys)]
    rows = []
    pool = Pool(a.workers)
    for design in [d.strip() for d in a.designs.split(",")]:
        cfg = DESIGNS[design]
        for wh in [bool(int(x)) for x in a.whiten.split(",")]:
            splits = np.array_split(np.arange(len(words)), a.workers)
            for ki, key in enumerate(keys):
                res = pool.map(_work, [(key, design, wh, words[s]) for s in splits])
                passed = np.concatenate([s[r[0]] for s, r in zip(splits, res)])
                acc = {k: np.concatenate([s[np.array(r[1][k], dtype=int)] for s, r in zip(splits, res)])
                       for k in ("full", *TRUNC)}
                for subset, mask in (("natural", nat_word), ("degenerate", ~nat_word)):
                    nw = int(mask.sum())
                    pb = float(np.isin(np.arange(len(words))[mask], passed).mean()) if nw else float("nan")
                    row = {"design": design, "whiten": int(wh), "key": ki, "subset": subset,
                           "image_transforms": n_it[subset], "hyp_words": nw,
                           "bch_pass": int(np.isin(passed, np.nonzero(mask)[0]).sum()),
                           "p_bch_emp": pb, "p_bch_theory": cfg.p_bch()}
                    for k in ("full", *TRUNC):
                        idx = acc[k][mask[acc[k]]] if len(acc[k]) else np.array([], int)
                        n_acc = len(np.unique(owner[idx]))           # image-level (any hypothesis)
                        bits = cfg.tag_bits if k == "full" else k
                        if k != "full" and k >= cfg.tag_bits:
                            row[f"acc_{k}"] = ""
                            continue
                        pred = nw * cfg.p_bch() * 2.0 ** (-bits)
                        lo, hi = poisson_interval(n_acc)
                        cp = clopper_pearson(n_acc, n_it[subset])
                        row[f"acc_{k}"] = n_acc
                        row[f"pred_{k}"] = pred
                        row[f"poisson95_{k}"] = f"[{lo:.2f}, {hi:.2f}]"
                        row[f"fa_rate_ub95_{k}"] = cp[1]
                    rows.append(row)
                    print({kk: row[kk] for kk in ("design", "whiten", "key", "subset", "bch_pass", "p_bch_emp",
                                                   "acc_full", "acc_8", "pred_8")}, f"{time.time()-t0:.0f}s",
                          flush=True)
    pool.close()
    fields = sorted({k for r in rows for k in r}, key=lambda x: (x not in rows[0], list(rows[0]).index(x)
                                                                 if x in rows[0] else 0))
    with open(out / "fa_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("written", out / "fa_results.csv")


if __name__ == "__main__":
    main()
