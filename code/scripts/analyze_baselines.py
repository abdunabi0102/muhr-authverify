"""Offline analysis of E4c (calibration under search), E2 (same-protocol robustness) and U-cross.

E4c decision rule for a k-bit baseline: "marked with message m" iff matches(decoded, m) >= k*(alpha), where
k*(alpha) is the smallest threshold whose binomial tail P[Bin(k, 1/2) >= k*] <= alpha (the rule used in
the HiDDeN / Stable Signature / Video Seal line of work). (i) native single decode, (ii) the verifier
accepts if ANY of our N_geo synchronization hypotheses reaches k*.

Reference messages (the claimed owner's message m):
  random      5 uniform messages per image (independent of the image: the only case the binomial tail models)
  zeros/ones  structured messages (low-entropy IDs)
  majority    per-bit majority of decoded bits, fitted on the first half of the images, tested on the
              second half (an adversary-chosen message exploiting decoder bias; held-out, no test leakage)
Detection channel (Video Seal / Pixel Seal / WAM): fraction of images with detection logit > 0, (i) and (ii).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muhr_exp.stats import bootstrap_mean_ci, clopper_pearson, wilson   # noqa: E402

ALPHAS = (1e-2, 1e-3, 1e-6)
N_RANDOM_REFS = 5


def kstar(k: int, alpha: float) -> int:
    """Smallest t with P[Bin(k,1/2) >= t] <= alpha (k+1 if impossible)."""
    for t in range(k + 1):
        if binom.sf(t - 1, k, 0.5) <= alpha:
            return t
    return k + 1


def load_dir(d: Path):
    zs = [np.load(f, allow_pickle=False) for f in sorted(d.glob("chunk_*.npz"))]
    if not zs:
        return None
    return {key: np.concatenate([z[key] for z in zs]) for key in zs[0].files}


def ci_str(x, n, zero_cp=True):
    lo, hi = clopper_pearson(x, n) if (zero_cp and x == 0) else wilson(x, n)
    return f"[{lo:.2e}, {hi:.2e}]"


# ---------------------------------------------------------------------------------------------- E4c
def analyze_unmarked(root: Path, out: Path, seed: int):
    rows = []
    for md in sorted(p for p in root.iterdir() if p.is_dir()):
        z = load_dir(md)
        if z is None:
            continue
        meta = json.loads((md / "meta.json").read_text())
        k = int(meta["bits"])
        names = z["names"].astype(str)
        natural = ~np.char.startswith(names, "degenerate_")
        nat_idx = np.nonzero(natural)[0]
        native = z["native"].astype(np.float32)                 # (N,T,1+k)
        hyp = z["hyp"].astype(np.float32)                       # (N,T,H,1+k)
        nh = z["nhyp"].astype(int)                              # (N,T)
        N, T, H, _ = hyp.shape
        valid = np.arange(H)[None, None, :] < nh[..., None]     # (N,T,H)
        nb = native[..., 1:] > 0                                # (N,T,k)
        hb = hyp[..., 1:] > 0                                   # (N,T,H,k)
        rng = np.random.default_rng(seed)
        half = nat_idx[: len(nat_idx) // 2]
        refs = {"random": [rng.integers(0, 2, size=(N, k)).astype(bool) for _ in range(N_RANDOM_REFS)],
                "zeros": [np.zeros((N, k), bool)], "ones": [np.ones((N, k), bool)],
                "majority": [np.broadcast_to(nb[half, 0].mean(0) > 0.5, (N, k))]}
        bias = float(np.abs(nb[nat_idx].mean(axis=(0, 1)) - 0.5).mean())
        for ti in range(T):
            tname = meta["transforms"][ti][0]
            for subset, sel in (("natural", natural), ("degenerate", ~natural)):
                for rname, rlist in refs.items():
                    s = sel.copy()
                    if rname == "majority":
                        s[half] = False                         # held-out images only
                    if not s.any():
                        continue
                    m_nat = np.concatenate([(nb[s, ti] == r[s][:, :]).sum(-1) for r in rlist])
                    m_hyp = np.concatenate([np.where(valid[s, ti], (hb[s, ti] == r[s][:, None, :]).sum(-1), -1).max(-1)
                                            for r in rlist])
                    n = len(m_nat)
                    row = {"method": md.name, "bits": k, "transform": tname, "subset": subset, "reference": rname,
                           "trials": n, "mean_n_geo": float(nh[s, ti].mean()), "bit_bias": bias}
                    for a in ALPHAS:
                        t = kstar(k, a)
                        x1, x2 = int((m_nat >= t).sum()), int((m_hyp >= t).sum())
                        row[f"k*_{a:g}"] = t
                        row[f"fpr_native_{a:g}"] = x1 / n
                        row[f"ci_native_{a:g}"] = ci_str(x1, n)
                        row[f"fpr_search_{a:g}"] = x2 / n
                        row[f"ci_search_{a:g}"] = ci_str(x2, n)
                        # threshold the search verifier needs to restore alpha empirically (only meaningful
                        # when alpha*n >= ~10; otherwise reported as "n/a")
                        if a * n >= 10:
                            need = next((u for u in range(k + 2) if (m_hyp >= u).mean() <= a), k + 1)
                            row[f"k_needed_search_{a:g}"] = need
                        else:
                            row[f"k_needed_search_{a:g}"] = "n/a"
                    if rname == "random":                       # detection channel (independent of reference)
                        det_n = native[s, ti, 0] > 0
                        det_h = np.where(valid[s, ti], hyp[s, ti, :, 0] > 0, False).any(-1)
                        has_det = bool(np.any(native[..., 0] != 0))
                        row["det_native"] = float(det_n.mean()) if has_det else ""
                        row["det_search"] = float(det_h.mean()) if has_det else ""
                    rows.append(row)
    _write(rows, out / "e4c_calibration.csv")
    return rows


# ---------------------------------------------------------------------------------------------- E2
def analyze_marked(root: Path, out: Path, ours_keys: int):
    rows, qrows, xrows = [], [], []
    for md in sorted(p for p in root.iterdir() if p.is_dir()):
        z = load_dir(md)
        if z is None:
            continue
        meta = json.loads((md / "meta.json").read_text())
        k = int(meta["bits"])
        conds = meta["conditions"]
        preds = z["preds"].astype(np.float32)                   # (N,C,1+k)
        msgs = z["msgs"].astype(bool)                           # (N,k)
        N = len(msgs)
        q = z["quality"]
        qrow = {"method": md.name, "bits": k, "n": N}
        for j, nm in enumerate(("psnr", "ssim", "lpips")):
            lo, hi = bootstrap_mean_ci(q[:, j])
            qrow[nm] = float(np.nanmean(q[:, j])); qrow[f"{nm}_ci"] = f"[{lo:.4f}, {hi:.4f}]"
        qrows.append(qrow)
        for c, (an, p) in enumerate(conds):
            match = ((preds[:, c, 1:] > 0) == msgs).sum(-1)
            ber = 1 - match / k
            lo, hi = bootstrap_mean_ci(ber)
            row = {"method": md.name, "bits": k, "attack": an, "param": p, "n": N,
                   "ber": float(ber.mean()), "ber_ci": f"[{lo:.4f}, {hi:.4f}]", "exact_msg": float((match == k).mean())}
            for a in ALPHAS:
                t = kstar(k, a)
                x = int((match >= t).sum())
                lo, hi = wilson(x, N)
                row[f"tpr_{a:g}"] = x / N
                row[f"tpr_ci_{a:g}"] = f"[{lo:.4f}, {hi:.4f}]"
            if np.any(preds[:, c, 0] != 0):
                row["det_rate"] = float((preds[:, c, 0] > 0).mean())
            rows.append(row)
        # U-cross: does OUR keyed verifier accept images marked by this baseline?
        xrows += _ucross(md.name, z["ours"], z["ours_n"], ours_keys)
    _write(rows, out / "e2_robustness.csv")
    _write(qrows, out / "e2_quality.csv")
    _write(xrows, out / "ucross.csv")


def _ucross(method, ours, ours_n, n_keys):
    from dataclasses import replace

    from muhr.crypto import derive_keys, generate_master_key
    from muhr_exp.codec2 import DESIGNS, CodecV2
    words, owner = [], []
    for i in range(len(ours)):
        for h in range(int(ours_n[i])):
            words.append(ours[i, h]); owner.append(i)
    words = np.unpackbits(np.stack(words), axis=1)
    owner = np.array(owner)
    rows = []
    for design in ("A", "G8", "G9"):
        for ki in range(n_keys):
            codec = CodecV2(derive_keys(generate_master_key()), replace(DESIGNS[design], whiten=True))
            msgs, nerr = codec.bch_decode_many(codec.unwhiten(words))
            passed = np.nonzero(nerr >= 0)[0]
            acc = [i for i in passed if codec.verify_msg(msgs[i]).ok]
            n_img = len(np.unique(owner[acc])) if acc else 0
            lo, hi = clopper_pearson(n_img, len(ours))
            rows.append({"method": method, "design": design, "key": ki, "images": len(ours), "hyp_words": len(words),
                         "bch_pass": len(passed), "accepted_images": n_img, "cp95_upper": hi})
    return rows


# ---------------------------------------------------------------------------------------------- codec on baselines
def analyze_codec(root: Path, out: Path):
    """Our keyed codec carried in the first 128 bits of another method's message (e2_baselines --codec)."""
    from dataclasses import replace

    from muhr.crypto import derive_keys
    from muhr_exp.codec2 import DESIGNS, CodecV2
    keys = derive_keys((root / "exp_master.key").read_text().strip())
    rows = []
    for md in sorted(p for p in root.iterdir() if p.is_dir()):
        z = load_dir(md)
        if z is None:
            continue
        meta = json.loads((md / "meta.json").read_text())
        emb = meta["codec"]
        preds, msgs = z["preds"].astype(np.float32), z["msgs"].astype(np.uint8)
        N = len(msgs)
        a = meta["args"]
        authors = np.random.default_rng(a["seed"]).integers(0, 2 ** 24, size=meta["n_files"])[z["index"]]
        variants = ["G8", "G9"] if emb in ("G8", "G9") else [emb]
        for v in variants:
            c = CodecV2(keys, replace(DESIGNS[v], whiten=True))
            for ci, (an, p) in enumerate(meta["conditions"]):
                words = (preds[:, ci, 1:129] > 0).astype(np.uint8)
                dm, ne = c.bch_decode_many(c.unwhiten(words))
                ok = wrong = 0
                for i in np.nonzero(ne >= 0)[0]:
                    d = c.verify_msg(dm[i])
                    if d.ok:
                        if d.fields["author_id"] == int(authors[i]) % (1 << c.cfg.id_bits):
                            ok += 1
                        else:
                            wrong += 1
                lo, hi = wilson(ok, N)
                rows.append({"method": md.name, "embedded": emb, "decode": v, "attack": an, "param": p, "n": N,
                             "verified": ok / N, "ci_lo": lo, "ci_hi": hi, "wrong_payload_accepts": wrong})
    _write(rows, out / "codec_on_baselines.csv")


def _write(rows, path):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("written", path, len(rows), "rows")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unmarked", default=None)
    p.add_argument("--marked", default=None)
    p.add_argument("--codec-marked", default=None)
    p.add_argument("--ours-keys", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.unmarked:
        analyze_unmarked(Path(a.unmarked), out, a.seed)
    if a.marked:
        analyze_marked(Path(a.marked), out, a.ours_keys)
    if a.codec_marked:
        analyze_codec(Path(a.codec_marked), out)


if __name__ == "__main__":
    main()
