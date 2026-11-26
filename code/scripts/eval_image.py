"""Tasvir suv belgisini haqiqiy hujumlar ostida baholash (IV bob, 4.2–4.3-§ jadvallari uchun).

Misol:
    python scripts/eval_image.py --ckpt runs/taklif/best.pt --data data/test --n 200 --out results/image
Ablyatsiya: har bir o'qitilgan variant uchun alohida ishga tushiring va --name bering:
    python scripts/eval_image.py --ckpt runs/no_stn/best.pt --name "STNsiz" ...
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muhr.crypto import derive_keys, load_or_create_master_key       # noqa: E402
from muhr.image.data import REAL_ATTACKS, apply_real_attack, list_images  # noqa: E402
from muhr.image.watermarker import ImageWatermarker                    # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="sinov tasvirlari (o'qitishda ishlatilmagan!)")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--max-side", type=int, default=1024, help="katta tasvirlarni kichraytirish")
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--key-file", default="keys/master.key")
    p.add_argument("--name", default="Taklif")
    p.add_argument("--modes", default="tiled,global",
                   help="muhrlash rejimlari: tiled (bloklar to'ri), global (butun tasvir bitta blok)")
    p.add_argument("--attacks", default=None, help="vergul bilan: JPEG,Kesish,... (default: hammasi)")
    p.add_argument("--out", default="results/image")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse()
    np.random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keys = derive_keys(load_or_create_master_key(a.key_file)[0])
    wm = ImageWatermarker(a.ckpt, keys)
    print(f"Model: {a.ckpt} | blok {wm.block} | {wm.msg_len} bit | {wm.codec.describe()}")
    files = list_images(a.data)[: a.n]
    attacks = {k: v for k, v in REAL_ATTACKS.items()
               if a.attacks is None or k in a.attacks.split(",") or k == "yo'q"}

    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    mode_names = {"tiled": "bloklar", "global": "global"}
    imgs = []
    for f in files:
        img = Image.open(f).convert("RGB")
        if max(img.size) > a.max_side:
            s = a.max_side / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)
        imgs.append((f, img, int(np.random.randint(0, 2 ** 24))))

    for mode in modes:
        name = f"{a.name}_{mode_names.get(mode, mode)}"
        print(f"\n===== Rejim: {mode} ({name}) =====", flush=True)
        rows, quality = [], []
        t0 = time.time()
        for i, (f, img, author) in enumerate(imgs):
            stego, info = wm.embed(img, author, beta=a.beta, mode=mode)
            quality.append((info["psnr"], info["ssim"]))
            tx = info["bits"]
            for an, (_, params) in attacks.items():
                for p in params:
                    att = apply_real_attack(stego, an, p)
                    r = wm.extract(att)
                    probs = r.probs if r.probs is not None else np.full(len(tx), 0.5)
                    bit_acc = float(np.mean((probs > 0.5).astype(np.uint8) == tx))
                    rows.append({"image": f.name, "attack": an, "param": "" if p is None else p,
                                 "bit_acc": round(bit_acc, 4),
                                 "verified": bool(r.ok and r.payload.author_id == author),
                                 "content_match": r.content_match, "corrected": r.corrected_bits,
                                 "grid": "x".join(map(str, r.grid)) if r.grid else "", "reason": r.reason})
            if (i + 1) % 10 == 0 or i + 1 == len(imgs):
                print(f"[{i+1}/{len(imgs)}] {time.time()-t0:.0f}s | PSNR={np.mean([q[0] for q in quality]):.2f}",
                      flush=True)

        with open(out / f"{name}_details.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        agg = defaultdict(list)
        for r in rows:
            agg[(r["attack"], r["param"])].append(r)
        ps, ss = np.array([q[0] for q in quality]), np.array([q[1] for q in quality])
        md = [f"# Tasvir suv belgisi: {name}\n",
              f"Rejim: {mode} | tasvirlar: {len(imgs)} | blok {wm.block} | xabar {wm.msg_len} bit | "
              f"{wm.codec.describe()}\n",
              "## Ko'rinmaslik\n", "| Ko'rsatkich | O'rtacha | Std | Min |", "|---|---|---|---|",
              f"| PSNR, dB | {ps.mean():.2f} | {ps.std():.2f} | {ps.min():.2f} |",
              f"| SSIM | {ss.mean():.4f} | {ss.std():.4f} | {ss.min():.4f} |",
              "\n## Bardoshlilik\n",
              "| Hujum | Parametr | Bit aniqligi (BCH gacha) | BER | Tasdiqlash ulushi |", "|---|---|---|---|---|"]
        summary = []
        for (an, p), rs in agg.items():
            acc = float(np.mean([r["bit_acc"] for r in rs]))
            ver = float(np.mean([r["verified"] for r in rs]))
            summary.append({"name": name, "mode": mode, "attack": an, "param": p, "bit_acc": acc,
                            "ber": 1 - acc, "verified": ver})
            md.append(f"| {an} | {p} | {acc:.4f} | {1-acc:.4f} | {ver:.2%} |")
        with open(out / f"{name}_summary.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
        (out / f"{name}_summary.md").write_text("\n".join(md), encoding="utf-8")
        print("\n".join(md), flush=True)


if __name__ == "__main__":
    main()
