"""Offline decisions for E8 (scripts/e8_games.py): which payload the keyed verifier accepts, if any.

For each game and decode variant: accepted (any valid tag), accepted with the victim's / group's author
(copy, collage, averaging: a "successful forgery of provenance" on a foreign image), accepted with the
attacker's author (overwrite), and the advisory content-hash check (payload hash vs pHash of the seen image,
top hash_bits compared, tolerance --tol) that should flag copied marks.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muhr.crypto import derive_keys                     # noqa: E402
from muhr_exp.codec2 import DESIGNS, CodecV2            # noqa: E402
from muhr_exp.stats import wilson                       # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e8", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tol", type=int, default=1, help="Hamming tolerance on the top hash_bits")
    a = p.parse_args()
    d = Path(a.e8)
    meta = json.loads((d / "meta.json").read_text())
    au = np.load(d / "authors.npz")
    zs = [np.load(f) for f in sorted(d.glob("chunk_*.npz"))]
    cat = lambda k: np.concatenate([z[k] for z in zs])          # noqa: E731
    hyp, nhyp, ahash, idx = cat("hyp"), cat("nhyp"), cat("ahash"), cat("index")
    keys = derive_keys((d / "exp_master.key").read_text().strip())
    games = meta["games"]
    emb = meta["design"]
    variants = ["G8", "G9"] if emb in ("G8", "G9") else [emb]
    rows = []
    for v in variants:
        c = CodecV2(keys, replace(DESIGNS[v], whiten=True))
        hb = c.cfg.hash_bits
        id_mask = (1 << c.cfg.id_bits) - 1
        for gi, g in enumerate(games):
            N = len(idx)
            acc_any = acc_victim = acc_attacker = hash_ok = 0
            for n in range(N):
                i = int(idx[n])
                words = np.unpackbits(hyp[n, gi, : nhyp[n, gi]], axis=1)
                msgs, ne = c.bch_decode_many(c.unwhiten(words))
                got = None
                for h in np.nonzero(ne >= 0)[0]:
                    r = c.verify_msg(msgs[h])
                    if r.ok:
                        got = r.fields
                        break                                   # verifier stops at the first valid hypothesis
                if got is None:
                    continue
                acc_any += 1
                victim = au["group_author"][au["group"][i]] if g == "copy_avg" else au["authors"][i]
                if got["author_id"] == int(victim) & id_mask:
                    acc_victim += 1
                if g.startswith("overwrite") and got["author_id"] == int(au["attackers"][i]) & id_mask:
                    acc_attacker += 1
                if hb:
                    seen_top = int(ahash[n, gi]) >> (16 - hb)
                    if bin(seen_top ^ got["hash"]).count("1") <= a.tol:
                        hash_ok += 1
            row = {"game": g, "decode": v, "n": N}
            for name, x in (("accepted_any", acc_any), ("accepted_victim_author", acc_victim),
                            ("accepted_attacker_author", acc_attacker), ("accepted_and_hash_consistent", hash_ok)):
                lo, hi = wilson(x, N)
                row[name] = x / N
                row[name + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
            rows.append(row)
            print(row, flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "e8_games.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("written", out / "e8_games.csv")


if __name__ == "__main__":
    main()
