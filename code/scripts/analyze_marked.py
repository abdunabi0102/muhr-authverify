"""Offline analysis of e3_marked.py outputs (E3, E-OP, E6 search/voting/MAC ablations). CPU only.

Policies (verifier variants), all evaluated on the SAME stored decoder outputs:
  search      — all hypotheses in the verifier's order, first MAC-valid one is accepted (deployed rule)
  nosync      — only hypothesis 0 (grid computed from the presented size)
  global_only — only the (1,1) hypothesis
Decode variants per embedded design: A -> A; E -> E; G -> G8, G9 (same codeword, different radius).
Also reports: errors of the best hypothesis on the n_use code positions (for TPR(t) curves), wrong-author
accepts, content-hash drift, quality (PSNR/SSIM/LPIPS) with bootstrap CIs, McNemar vs design A.

Example: python scripts/analyze_marked.py --e3 e3_coco --out results/e3_coco
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muhr.crypto import derive_keys                                           # noqa: E402
from muhr_exp.codec2 import DESIGNS, CodecV2                                  # noqa: E402
from muhr_exp.stats import bootstrap_mean_ci, holm, mcnemar_exact, wilson     # noqa: E402

DECODE_VARIANTS = {"A": ["A"], "C": ["C"], "E": ["E"], "F": ["F"], "G": ["G8", "G9"]}


def hamming16(a, b):
    return bin(int(a) ^ int(b)).count("1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--e3", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    e3, out = Path(a.e3), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import json
    meta = json.loads((e3 / "meta.json").read_text())
    designs = [d.strip() for d in meta["args"]["designs"].split(",")]
    modes, conds = meta["modes"], [tuple(c) for c in meta["conditions"]]
    keys = derive_keys((e3 / "exp_master.key").read_text().strip())
    codecs = {v: CodecV2(keys, DESIGNS[v]) for d in designs for v in DECODE_VARIANTS[d]}
    chunks = [np.load(f, allow_pickle=False) for f in sorted(e3.glob("chunk_*.npz"))]
    if not chunks:
        sys.exit("no chunks")
    cat = lambda k: np.concatenate([c[k] for c in chunks])
    hyp, grids, nhyp, tx, q = cat("hyp"), cat("grids"), cat("nhyp"), cat("tx"), cat("quality")
    authors, ahash, ohash, day = cat("authors"), cat("ahash"), cat("ohash"), int(chunks[0]["day"])
    N = len(authors)
    print(f"{N} images, designs {designs}, modes {modes}, {len(conds)} conditions")

    # ---------------- decisions
    verified = {}           # (variant, mode, policy) -> bool[N, C]
    wrong = defaultdict(int)
    best_err = {}           # (design, mode) -> int[N, C] errors of best hypothesis on code positions
    for di, d in enumerate(designs):
        for mi, mode in enumerate(modes):
            H = hyp.shape[4]
            valid = np.arange(H)[None, None, :] < nhyp[:, di, mi, :, None]          # [N, C, H]
            bits = np.unpackbits(hyp[:, di, mi], axis=-1)                             # [N, C, H, 128]
            is11 = (grids[:, di, mi, :, :, 0] == 1) & (grids[:, di, mi, :, :, 1] == 1)
            for v in DECODE_VARIANTS[d]:
                c = codecs[v]
                # channel positions that carry code bits (pad positions excluded)
                code_pos = np.zeros(128, bool); code_pos[np.nonzero(c.perm < c.cfg.n_use)[0]] = True
                errs = (bits != tx[:, di, mi][:, None, None, :])[..., code_pos].sum(-1)   # [N, C, H]
                best_err[(d, mode)] = np.where(valid, errs, 10 ** 6).min(-1)
                ok = np.zeros(valid.shape, bool)
                flat = np.nonzero(valid.reshape(-1))[0]
                words = bits.reshape(-1, 128)[flat]
                for s in range(0, len(flat), 20000):
                    msgs, nerr = c.bch_decode_many(c.unwhiten(words[s:s + 20000]))
                    for j in np.nonzero(nerr >= 0)[0]:
                        dd = c.verify_msg(msgs[j])
                        if not dd.ok:
                            continue
                        n_i = flat[s + j] // (len(conds) * H)
                        if dd.fields["author_id"] == int(authors[n_i]) % (1 << c.cfg.id_bits):
                            ok.reshape(-1)[flat[s + j]] = True
                        else:
                            wrong[(v, mode)] += 1
                verified[(v, mode, "search")] = ok.any(-1)
                verified[(v, mode, "nosync")] = ok[..., 0]
                verified[(v, mode, "global_only")] = (ok & is11).any(-1)
                print(f"decided {v}/{mode}: {len(flat)} words", flush=True)

    # ---------------- per-condition table
    rows = []
    for (v, mode, pol), V in verified.items():
        for k, (an, prm) in enumerate(conds):
            kk = int(V[:, k].sum())
            lo, hi = wilson(kk, N)
            rows.append({"variant": v, "mode": mode, "policy": pol, "attack": an, "param": "" if prm is None else prm,
                         "n": N, "verified": kk, "rate": kk / N, "ci_lo": lo, "ci_hi": hi})
    with open(out / "verification.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---------------- mean over conditions with bootstrap over images
    summ = []
    for (v, mode, pol), V in verified.items():
        per_img = V.mean(1)
        lo, hi = bootstrap_mean_ci(per_img)
        summ.append({"variant": v, "mode": mode, "policy": pol, "mean_rate": float(per_img.mean()),
                     "ci_lo": lo, "ci_hi": hi, "wrong_author_accepts": wrong[(v, mode)]})
    with open(out / "verification_mean.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

    # ---------------- McNemar vs A (same images, deployed policy) per condition, Holm-adjusted
    mc = []
    for mode in modes:
        if ("A", mode, "search") not in verified:
            continue
        for v in [x for x in codecs if x != "A"]:
            ps, recs = [], []
            for k, (an, prm) in enumerate(conds):
                A_, B_ = verified[("A", mode, "search")][:, k], verified[(v, mode, "search")][:, k]
                b, c = int((A_ & ~B_).sum()), int((~A_ & B_).sum())
                ps.append(mcnemar_exact(b, c))
                recs.append({"mode": mode, "variant": v, "attack": an, "param": "" if prm is None else prm,
                             "A_rate": A_.mean(), "variant_rate": B_.mean(), "A_only": b, "variant_only": c})
            for r, pv, pa in zip(recs, ps, holm(ps)):
                r["p_mcnemar"], r["p_holm"] = pv, pa
                mc.append(r)
    if mc:
        with open(out / "mcnemar_vs_A.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(mc[0])); w.writeheader(); w.writerows(mc)

    # ---------------- errors of best hypothesis -> TPR(t) curves (E-OP)
    er = []
    for (d, mode), be in best_err.items():
        for k, (an, prm) in enumerate(conds):
            e = be[:, k]
            rec = {"design": d, "mode": mode, "attack": an, "param": "" if prm is None else prm,
                   "median_err": float(np.median(e)), "mean_ber_code": float(e.mean() / codecs[DECODE_VARIANTS[d][0]].cfg.n_use)}
            for t in range(0, 13):
                rec[f"P(e<={t})"] = float((e <= t).mean())
            er.append(rec)
    with open(out / "errors_best_hyp.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(er[0])); w.writeheader(); w.writerows(er)

    # ---------------- quality
    qr = []
    for di, d in enumerate(designs):
        for mi, mode in enumerate(modes):
            for j, name in enumerate(("psnr", "ssim", "lpips")):
                x = q[:, di, mi, j]
                lo, hi = bootstrap_mean_ci(x)
                qr.append({"design": d, "mode": mode, "metric": name, "mean": float(np.nanmean(x)),
                           "std": float(np.nanstd(x)), "min": float(np.nanmin(x)), "ci_lo": lo, "ci_hi": hi})
    with open(out / "quality.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qr[0])); w.writeheader(); w.writerows(qr)

    # ---------------- content-hash drift on genuine attacked images (advisory hash)
    hd = []
    for k, (an, prm) in enumerate(conds):
        dist = np.array([hamming16(ahash[i, 0, 0, k], ohash[i]) for i in range(N)])
        hd.append({"attack": an, "param": "" if prm is None else prm, "share_dist_gt3": float((dist > 3).mean()),
                   "median_dist": float(np.median(dist))})
    with open(out / "hash_drift.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hd[0])); w.writeheader(); w.writerows(hd)
    print("written:", *[f.name for f in out.glob("*.csv")])


if __name__ == "__main__":
    main()
