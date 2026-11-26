"""E3 / E-OP: marked test set under the 27 real attacks; stores hard bits of ALL hypotheses.

Resumable (one .npz per chunk). Decisions are computed offline by scripts/analyze_marked.py.

Example (Kaggle):
  python scripts/e3_marked.py --ckpt /kaggle/input/.../best.pt --data /kaggle/input/coco/val2017 \
      --start 1500 --n 2000 --designs A,E,G --modes global,tiled --out /kaggle/working/e3_coco
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muhr.crypto import derive_keys, generate_master_key, phash16          # noqa: E402
from muhr.image.data import REAL_ATTACKS, apply_real_attack, list_images    # noqa: E402
from muhr.image.watermarker import ImageWatermarker                         # noqa: E402
from muhr_exp.codec2 import DESIGNS, CodecV2                                # noqa: E402
from muhr_exp.runner import (MAX_HYP, LPIPSMeter, all_hypotheses_batch, amp_check, embed_bits,  # noqa: E402
                             env_info, load_rgb, quality, write_json)

EMBED_DESIGNS = {"A": "A", "C": "C", "E": "E", "F": "F", "G": "G9"}   # G8/G9 share the same codeword


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--start", type=int, default=0, help="index into the sorted file list")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--designs", default="A,E,G")
    p.add_argument("--modes", default="global,tiled")
    p.add_argument("--attacks", default=None, help="comma list of attack names (default: all 27 conditions)")
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--max-n", type=int, default=4, help="largest grid side searched (N_geo budget)")
    p.add_argument("--amp", type=int, default=1, help="fp16 decoding if it agrees with fp32 on a sample")
    p.add_argument("--no-stn-infer", action="store_true",
                   help="ablation without retraining: bypass the trained decoder's STN at test time")
    p.add_argument("--regen", action="store_true",
                   help="E9: register regeneration attacks (muhr_exp/regen.py); select them with --attacks")
    return p.parse_args()


def conditions(names=None):
    out = []
    for an, (_, params) in REAL_ATTACKS.items():
        if names is None or an in names or an == "yo'q":
            out += [(an, p) for p in params]
    return out


def main():
    a = parse()
    t_start = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keyfile = out / "exp_master.key"            # fresh key for this experiment (not the dissertation key)
    if not keyfile.exists():
        keyfile.write_text(generate_master_key())
    keys = derive_keys(keyfile.read_text().strip())
    wm = ImageWatermarker(a.ckpt, keys)
    if a.no_stn_infer:
        assert wm.dec.stn is not None, "checkpoint has no STN"
        wm.dec.stn = None
        print("STN bypassed at inference (ablation)", flush=True)
    designs = [d.strip() for d in a.designs.split(",") if d.strip()]
    codecs = {d: CodecV2(keys, DESIGNS[EMBED_DESIGNS[d]]) for d in designs}
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    regen_meta = None
    if a.regen:
        from muhr_exp import regen
        regen_meta = regen.register()
    conds = conditions(a.attacks.split(",") if a.attacks else None)
    files = list_images(a.data)[a.start: a.start + a.n]
    lp = LPIPSMeter(wm.device)
    sample = [load_rgb(f, a.max_side) for f in files[:10]]
    ampinfo = amp_check(wm, sample, a.max_n) if a.amp else {"amp": False, "reason": "disabled"}
    print("AMP check:", ampinfo, flush=True)
    use_amp = bool(ampinfo["amp"])
    write_json(out / "meta.json", {"args": vars(a), "env": env_info(), "ckpt_meta": wm.meta, "amp": ampinfo,
                                   "designs": {d: DESIGNS[EMBED_DESIGNS[d]].__dict__ for d in designs},
                                   "conditions": [[c[0], c[1]] for c in conds], "n_files": len(files),
                                   "max_hyp": MAX_HYP, "modes": modes, "regen": regen_meta})
    rng = np.random.default_rng(a.seed)
    authors = rng.integers(0, 2 ** 24, size=len(files))
    day = 9800                                  # fixed day index (fits 16 and 20 bits)
    n_chunks = (len(files) + a.chunk - 1) // a.chunk
    for ci in range(n_chunks):
        fn = out / f"chunk_{ci:04d}.npz"
        if fn.exists():
            continue
        if (time.time() - t_start) / 3600 > a.max_hours:
            print("time budget reached; rerun to resume", flush=True)
            break
        idx = range(ci * a.chunk, min((ci + 1) * a.chunk, len(files)))
        D, M, C = len(designs), len(modes), len(conds)
        hyp = np.zeros((len(idx), D, M, C, MAX_HYP, 16), np.uint8)       # packed bits
        grids = np.full((len(idx), D, M, C, MAX_HYP, 2), -1, np.int16)
        conf = np.zeros((len(idx), D, M, C, MAX_HYP), np.float16)
        nhyp = np.zeros((len(idx), D, M, C), np.uint8)
        ahash = np.zeros((len(idx), D, M, C), np.uint16)
        tx = np.zeros((len(idx), D, M, 128), np.uint8)
        q = np.zeros((len(idx), D, M, 3), np.float32)
        sizes = np.zeros((len(idx), 2), np.int32)
        ohash = np.zeros(len(idx), np.uint16)
        names = []
        t0 = time.time()
        for j, i in enumerate(idx):
            img = load_rgb(files[i], a.max_side)
            names.append(files[i].name)
            sizes[j] = img.size
            ph = phash16(img); ohash[j] = ph
            for di, d in enumerate(designs):
                bits = codecs[d].encode(int(authors[i]), day, ph, 0)
                for mi, mode in enumerate(modes):
                    stego, _ = embed_bits(wm, img, bits, mode)
                    tx[j, di, mi] = bits
                    qq = quality(img, stego, lp); q[j, di, mi] = (qq["psnr"], qq["ssim"], qq["lpips"])
                    atts = [apply_real_attack(stego, an, p) for an, p in conds]
                    res = all_hypotheses_batch(wm, atts, a.max_n, use_amp)
                    for k, (att, (hb, g, cf, nh)) in enumerate(zip(atts, res)):
                        hyp[j, di, mi, k] = np.packbits(hb, axis=1)
                        grids[j, di, mi, k], conf[j, di, mi, k], nhyp[j, di, mi, k] = g, cf, nh
                        ahash[j, di, mi, k] = phash16(att)
        np.savez_compressed(fn, hyp=hyp, grids=grids, conf=conf, nhyp=nhyp, ahash=ahash, tx=tx, quality=q,
                            sizes=sizes, ohash=ohash, authors=authors[list(idx)], names=np.array(names),
                            day=day, index=np.array(list(idx)))
        print(f"chunk {ci+1}/{n_chunks} saved ({len(idx)} images, {time.time()-t0:.0f}s) | "
              f"PSNR {q[..., 0].mean():.2f}", flush=True)
    print("done" if all((out / f"chunk_{c:04d}.npz").exists() for c in range(n_chunks)) else "partial",
          flush=True)


if __name__ == "__main__":
    main()
