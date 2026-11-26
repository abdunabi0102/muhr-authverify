"""E4a/E4b/E4d: decoder outputs of ALL hypotheses on UNMARKED images (key-independent).

FA for any codec, key, whitening setting and tag length is computed offline (scripts/analyze_fa.py).
Also generates degenerate images (flat colours, black, white, gradients, checkerboards).

Example (Kaggle):
  python scripts/e4_unmarked.py --ckpt .../best.pt --data /kaggle/input/coco/test2017 --n 10000 \
      --degenerate 500 --out /kaggle/working/e4_unmarked
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muhr.crypto import derive_keys, generate_master_key                    # noqa: E402
from muhr.image.data import apply_real_attack, list_images                  # noqa: E402
from muhr.image.watermarker import ImageWatermarker                         # noqa: E402
from muhr_exp.runner import MAX_HYP, all_hypotheses_batch, amp_check, env_info, load_rgb, write_json  # noqa: E402

BENIGN = [("yo'q", None), ("JPEG", 80), ("Masshtablash", 0.75), ("Kesish", 0.8), ("WebP", 80)]


def degenerate_images(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    kinds = ["flat", "black", "white", "hgrad", "vgrad", "checker"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        h, w = int(rng.integers(200, 1025)), int(rng.integers(200, 1025))
        if kind == "flat":
            a = np.ones((h, w, 3)) * rng.integers(0, 256, 3)
        elif kind == "black":
            a = np.zeros((h, w, 3))
        elif kind == "white":
            a = np.full((h, w, 3), 255.0)
        elif kind == "hgrad":
            a = np.repeat(np.linspace(0, 255, w)[None, :, None], h, 0).repeat(3, 2)
        elif kind == "vgrad":
            a = np.repeat(np.linspace(0, 255, h)[:, None, None], w, 1).repeat(3, 2)
        else:
            s = int(rng.integers(4, 64)); yy, xx = np.mgrid[0:h, 0:w]
            a = (((yy // s + xx // s) % 2) * 255.0)[..., None].repeat(3, 2)
        yield f"degenerate_{kind}_{i:04d}", Image.fromarray(a.astype(np.uint8))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--degenerate", type=int, default=500)
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--out", required=True)
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--max-n", type=int, default=4, help="largest grid side searched (N_geo budget)")
    p.add_argument("--amp", type=int, default=1, help="fp16 decoding if it agrees with fp32 on a sample")
    return p.parse_args()


def main():
    a = parse()
    t_start = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # decoder is keyless; a throwaway key is only needed to construct the watermarker object
    wm = ImageWatermarker(a.ckpt, derive_keys(generate_master_key()))
    files = list_images(a.data)[a.start: a.start + a.n]
    items = [(f.name, f) for f in files] + list(degenerate_images(a.degenerate))
    sample = [load_rgb(f, a.max_side) for f in files[:10]]
    ampinfo = amp_check(wm, sample, a.max_n) if a.amp else {"amp": False, "reason": "disabled"}
    print("AMP check:", ampinfo, flush=True)
    use_amp = bool(ampinfo["amp"])
    write_json(out / "meta.json", {"args": vars(a), "env": env_info(), "ckpt_meta": wm.meta, "amp": ampinfo,
                                   "benign": BENIGN, "n_natural": len(files), "n_degenerate": a.degenerate,
                                   "max_hyp": MAX_HYP})
    n_chunks = (len(items) + a.chunk - 1) // a.chunk
    for ci in range(n_chunks):
        fn = out / f"chunk_{ci:04d}.npz"
        if fn.exists():
            continue
        if (time.time() - t_start) / 3600 > a.max_hours:
            print("time budget reached; rerun to resume", flush=True)
            break
        part = items[ci * a.chunk:(ci + 1) * a.chunk]
        T = len(BENIGN)
        hyp = np.zeros((len(part), T, MAX_HYP, 16), np.uint8)
        grids = np.full((len(part), T, MAX_HYP, 2), -1, np.int16)
        conf = np.zeros((len(part), T, MAX_HYP), np.float16)
        nhyp = np.zeros((len(part), T), np.uint8)
        names, t0 = [], time.time()
        for j, (name, src) in enumerate(part):
            img = src if isinstance(src, Image.Image) else load_rgb(src, a.max_side)
            names.append(name)
            res = all_hypotheses_batch(wm, [apply_real_attack(img, an, p) for an, p in BENIGN], a.max_n, use_amp)
            for k, (hb, g, cf, nh) in enumerate(res):
                hyp[j, k], grids[j, k], conf[j, k], nhyp[j, k] = np.packbits(hb, axis=1), g, cf, nh
        np.savez_compressed(fn, hyp=hyp, grids=grids, conf=conf, nhyp=nhyp, names=np.array(names))
        print(f"chunk {ci+1}/{n_chunks} saved ({len(part)} images, {time.time()-t0:.0f}s)", flush=True)
    print("done" if all((out / f"chunk_{c:04d}.npz").exists() for c in range(n_chunks)) else "partial",
          flush=True)


if __name__ == "__main__":
    main()
