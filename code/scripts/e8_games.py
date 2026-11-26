"""E8: measured security games for the short-tag verifier (threat model A3-A6, methodology.md).

For every game we store the hard bits of all synchronization hypotheses of the image the verifier sees;
acceptance, *whose* payload is accepted, and the advisory content-hash check are decided offline
(scripts/analyze_e8.py) with the experiment key saved in --out.

Games (victim images V = T-COCO[start : start+n], hosts H = the next n images):
  copy_oracle    residual (stego - cover) of V added to H                 (upper bound: attacker knows the cover)
  copy_denoise   residual estimated as stego - median3(stego), added to H   (realistic blind copy)
  copy_avg       residual averaged over --avg-k stegos of the SAME author, added to H (estimation/averaging)
  collage        left half of the stego of V pasted onto H
  overwrite      stego of V re-embedded with the SAME model and a different author (A6, public-model setting)
  overwrite_jpg  as overwrite, then JPEG 75
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muhr.crypto import derive_keys, generate_master_key, phash16          # noqa: E402
from muhr.image.data import apply_real_attack, list_images                  # noqa: E402
from muhr.image.watermarker import ImageWatermarker                         # noqa: E402
from muhr_exp.codec2 import DESIGNS, CodecV2                                # noqa: E402
from muhr_exp.runner import MAX_HYP, all_hypotheses_batch, embed_bits, env_info, load_rgb, write_json  # noqa: E402

GAMES = ["copy_oracle", "copy_denoise", "copy_avg", "collage", "overwrite", "overwrite_jpg"]
DAY = 9800


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--design", default="G9")
    p.add_argument("--avg-k", type=int, default=8)
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--max-n", type=int, default=4)
    p.add_argument("--chunk", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--max-hours", type=float, default=11.0)
    return p.parse_args()


def arr(img):
    return np.asarray(img.convert("RGB"), np.float32)


def to_img(a):
    return Image.fromarray(np.clip(np.rint(a), 0, 255).astype(np.uint8))


def resize_like(a, ref):
    """resize an HxWx3 float array to ref's size (bilinear)."""
    h, w = ref.shape[:2]
    if a.shape[:2] == (h, w):
        return a
    chans = [np.asarray(Image.fromarray(a[..., c]).resize((w, h), Image.Resampling.BILINEAR)) for c in range(3)]
    return np.stack(chans, -1)


def main():
    a = parse()
    t_start = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keyfile = out / "exp_master.key"
    if not keyfile.exists():
        keyfile.write_text(generate_master_key())
    keys = derive_keys(keyfile.read_text().strip())
    wm = ImageWatermarker(a.ckpt, keys)
    codec = CodecV2(keys, DESIGNS[a.design])
    files = list_images(a.data)[a.start: a.start + 2 * a.n]
    victims, hosts = files[: a.n], files[a.n: 2 * a.n]
    rng = np.random.default_rng(a.seed)
    authors = rng.integers(0, 2 ** 24, size=a.n)              # victim authors
    attackers = rng.integers(0, 2 ** 24, size=a.n)            # attacker authors (overwrite)
    # copy_avg: groups of avg_k victims share one author (the averaging attacker collects k images of one owner)
    group = np.arange(a.n) // a.avg_k
    group_author = rng.integers(0, 2 ** 24, size=group.max() + 1)
    write_json(out / "meta.json", {"args": vars(a), "env": env_info(), "games": GAMES, "design": a.design,
                                   "day": DAY, "max_hyp": MAX_HYP})
    np.savez(out / "authors.npz", authors=authors, attackers=attackers, group=group, group_author=group_author)

    def stego_of(img, author):
        bits = codec.encode(int(author), DAY, phash16(img), 0)
        s, _ = embed_bits(wm, img, bits, "global")
        return s

    n_chunks = (a.n + a.chunk - 1) // a.chunk
    # residual pool for copy_avg (per group), built lazily per chunk from that chunk's victims
    for ci in range(n_chunks):
        fn = out / f"chunk_{ci:04d}.npz"
        if fn.exists():
            continue
        if (time.time() - t_start) / 3600 > a.max_hours:
            print("time budget reached; rerun to resume", flush=True)
            break
        idx = list(range(ci * a.chunk, min((ci + 1) * a.chunk, a.n)))
        G = len(GAMES)
        hyp = np.zeros((len(idx), G, MAX_HYP, 16), np.uint8)
        nhyp = np.zeros((len(idx), G), np.uint8)
        ahash = np.zeros((len(idx), G), np.uint16)
        vhash = np.zeros(len(idx), np.uint16)
        t0 = time.time()
        avg_cache = {}
        for j, i in enumerate(idx):
            V, H = load_rgb(victims[i], a.max_side), load_rgb(hosts[i], a.max_side)
            vhash[j] = phash16(V)
            SV = stego_of(V, authors[i])
            res_oracle = arr(SV) - arr(V)
            res_den = arr(SV) - arr(SV.filter(ImageFilter.MedianFilter(3)))
            g = int(group[i])
            if g not in avg_cache:                               # average residual of the group's k stegos
                members = [m for m in range(a.n) if group[m] == g]
                acc = None
                for m in members:
                    Vm = load_rgb(victims[m], a.max_side)
                    Sm = stego_of(Vm, group_author[g])
                    r = arr(Sm) - arr(Sm.filter(ImageFilter.MedianFilter(3)))
                    r = resize_like(r, arr(H)) if acc is None else resize_like(r, acc)
                    acc = r if acc is None else acc + r
                avg_cache[g] = acc / len(members)
            Ha = arr(H)
            seen = {
                "copy_oracle": to_img(Ha + resize_like(res_oracle, Ha)),
                "copy_denoise": to_img(Ha + resize_like(res_den, Ha)),
                "copy_avg": to_img(Ha + resize_like(avg_cache[g], Ha)),
            }
            Sa = arr(SV)
            Hc = resize_like(Ha, Sa).copy()
            Hc[:, : Sa.shape[1] // 2] = Sa[:, : Sa.shape[1] // 2]
            seen["collage"] = to_img(Hc)
            ow = stego_of(SV, attackers[i])
            seen["overwrite"] = ow
            seen["overwrite_jpg"] = apply_real_attack(ow, "JPEG", 75)
            imgs = [seen[gname] for gname in GAMES]
            res = all_hypotheses_batch(wm, imgs, a.max_n, False)
            for k, (hb, _, _, nh) in enumerate(res):
                hyp[j, k], nhyp[j, k] = np.packbits(hb, axis=1), nh
                ahash[j, k] = phash16(imgs[k])
        np.savez_compressed(fn, hyp=hyp, nhyp=nhyp, ahash=ahash, vhash=vhash, index=np.array(idx))
        print(f"chunk {ci+1}/{n_chunks} saved ({len(idx)} victims, {time.time()-t0:.0f}s)", flush=True)
    print("done" if all((out / f"chunk_{c:04d}.npz").exists() for c in range(n_chunks)) else "partial", flush=True)


if __name__ == "__main__":
    main()
