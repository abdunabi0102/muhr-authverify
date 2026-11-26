"""E2 / E4c / U-cross: deep-watermarking baselines under the same images, attacks and search as ours.

Stages (--stage):
  smoke     4 images per method: clean / JPEG50 bit accuracy, PSNR, throughput (sanity check of loading).
  unmarked  E4c. Unmarked COCO test2017 (+ degenerate images): soft outputs of (i) the native single
            decode and (ii) every synchronization hypothesis of OUR verifier (same N_geo, same max_n),
            each baseline's detector used as tile decoder. Decisions/thresholds are computed offline.
  marked    E2 + U-cross. T-COCO images marked by each baseline (random message, native resolution),
            PSNR/SSIM/LPIPS, native decode under the same 27 attack conditions as E3; plus OUR decoder's
            hypotheses on the clean baseline-marked image (does our keyed verifier accept foreign marks?).

All raw outputs are stored (float16 logits); nothing is thresholded on the GPU. Resumable per chunk.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e4_unmarked import BENIGN, degenerate_images                                   # noqa: E402
from muhr.crypto import derive_keys, generate_master_key                            # noqa: E402
from muhr.image.data import REAL_ATTACKS, apply_real_attack, list_images            # noqa: E402
from muhr.image.watermarker import ImageWatermarker, pil_to_tensor, tensor_to_pil   # noqa: E402
from muhr_exp import baselines as B                                                 # noqa: E402
from muhr_exp.runner import (MAX_HYP, LPIPSMeter, all_hypotheses_batch, env_info,  # noqa: E402
                             load_rgb, quality, write_json)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["smoke", "unmarked", "marked"])
    p.add_argument("--methods", default=",".join(B.ALL_METHODS))
    p.add_argument("--vs-root", required=True, help="Video Seal checkout with ckpts/ and configs/")
    p.add_argument("--ckpt", required=True, help="OUR best.pt (grid hypotheses; U-cross decoding)")
    p.add_argument("--data", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--degenerate", type=int, default=0)
    p.add_argument("--transforms", default="yo'q", help="unmarked: comma list of BENIGN names")
    p.add_argument("--attacks", default=None, help="marked: comma list of attack names (default all 27)")
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--max-n", type=int, default=4)
    p.add_argument("--chunk", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--max-hours", type=float, default=11.0)
    p.add_argument("--codec", default=None,
                   help="marked: embed OUR keyed codeword (design name, e.g. G9) in the first 128 bits; "
                        "remaining bits random. Verifies the codec on top of another decoder")
    p.add_argument("--regen", action="store_true",
                   help="E9: register regeneration attacks (muhr_exp/regen.py); select them with --attacks")
    return p.parse_args()


def to_t(img, dev):
    return pil_to_tensor(img.convert("RGB")).to(dev)


def conditions(names=None):
    out = []
    for an, (_, params) in REAL_ATTACKS.items():
        if names is None or an in names or an == "yo'q":
            out += [(an, p) for p in params]
    return out


def msg_for(seed: int, i: int, k: int) -> torch.Tensor:
    g = np.random.default_rng([seed, i])
    return torch.from_numpy(g.integers(0, 2, size=(1, k))).float()


def bitacc(pred_row: torch.Tensor, msg: torch.Tensor) -> float:
    return float(((pred_row[1:] > 0).float().cpu() == msg[0]).float().mean())


# ----------------------------------------------------------------------------------------------
def stage_smoke(a, wm, dev, methods, files):
    imgs = [load_rgb(f, a.max_side) for f in files[:4]]
    lp = LPIPSMeter(dev)
    report = {}
    for m in methods:
        t0 = time.time()
        try:
            model = B.load_method(m, a.vs_root, dev)
        except Exception as e:                                     # report and continue with the others
            report[m] = {"error": repr(e)}
            print(f"[{m}] LOAD FAILED: {e!r}", flush=True)
            continue
        k, S = B.n_bits(model), B.proc_size(model)
        acc_c, acc_j, ps, lps = [], [], [], []
        for i, img in enumerate(imgs):
            msg = msg_for(a.seed, i, k)
            xw = B.embed(model, to_t(img, dev), msg)
            stego = tensor_to_pil(xw)
            q = quality(img, stego, lp); ps.append(q["psnr"]); lps.append(q["lpips"])
            acc_c.append(bitacc(B.detect(model, to_t(stego, dev))[0], msg))
            acc_j.append(bitacc(B.detect(model, to_t(apply_real_attack(stego, "JPEG", 50), dev))[0], msg))
        x = to_t(imgs[0], dev)
        cands = wm._grid_candidates(imgs[0].height, imgs[0].width, a.max_n)[:MAX_HYP]
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(3):
            B.detect_hypotheses(model, x, cands, MAX_HYP)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        hyp_s = (time.time() - t1) / 3
        report[m] = {"bits": k, "proc_size": S, "acc_clean": float(np.mean(acc_c)), "acc_jpeg50": float(np.mean(acc_j)),
                     "psnr": float(np.mean(ps)), "lpips": float(np.mean(lps)), "n_hyp": len(cands),
                     "sec_per_image_search": hyp_s, "load_s": time.time() - t0}
        report[m]["loaded_ok"] = report[m]["acc_clean"] > 0.9    # weights really loaded (cards use strict=False)
        print(f"[{m}] {'OK ' if report[m]['loaded_ok'] else 'CHECK WEIGHTS!'} k={k} S={S} acc_clean={report[m]['acc_clean']:.3f} acc_jpeg50={report[m]['acc_jpeg50']:.3f} "
              f"PSNR={report[m]['psnr']:.2f} LPIPS={report[m]['lpips']:.4f} search={hyp_s:.3f}s/img "
              f"({len(cands)} hyp)", flush=True)
        del model
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    write_json(Path(a.out) / "smoke.json", {"env": env_info(), "args": vars(a), "methods": report})


# ----------------------------------------------------------------------------------------------
def stage_unmarked(a, wm, dev, methods, files, t_start):
    items = [(f.name, f) for f in files] + list(degenerate_images(a.degenerate))
    trans = [t for t in BENIGN if t[0] in a.transforms.split(",")]
    for m in methods:
        out = Path(a.out) / m; out.mkdir(parents=True, exist_ok=True)
        model = B.load_method(m, a.vs_root, dev)
        k = B.n_bits(model)
        write_json(out / "meta.json", {"args": vars(a), "env": env_info(), "method": m, "bits": k,
                                       "proc_size": B.proc_size(model), "transforms": trans,
                                       "n_natural": len(files), "n_degenerate": a.degenerate, "max_hyp": MAX_HYP})
        n_chunks = (len(items) + a.chunk - 1) // a.chunk
        for ci in range(n_chunks):
            fn = out / f"chunk_{ci:04d}.npz"
            if fn.exists():
                continue
            if (time.time() - t_start) / 3600 > a.max_hours:
                print("time budget reached; rerun to resume", flush=True)
                return
            part = items[ci * a.chunk:(ci + 1) * a.chunk]
            T = len(trans)
            native = np.zeros((len(part), T, 1 + k), np.float16)
            hyp = np.zeros((len(part), T, MAX_HYP, 1 + k), np.float16)
            grids = np.full((len(part), T, MAX_HYP, 2), -1, np.int16)
            nhyp = np.zeros((len(part), T), np.uint8)
            names, t0 = [], time.time()
            for j, (name, src) in enumerate(part):
                img = src if isinstance(src, Image.Image) else load_rgb(src, a.max_side)
                names.append(name)
                for ti, (tn, tp) in enumerate(trans):
                    im = apply_real_attack(img, tn, tp).convert("RGB")
                    x = to_t(im, dev)
                    cands = wm._grid_candidates(im.height, im.width, a.max_n)[:MAX_HYP]
                    native[j, ti] = B.detect(model, x)[0].cpu().numpy()
                    hyp[j, ti] = B.detect_hypotheses(model, x, cands, MAX_HYP).cpu().numpy()
                    grids[j, ti, :len(cands)] = np.array(cands, np.int16)
                    nhyp[j, ti] = len(cands)
            np.savez_compressed(fn, native=native, hyp=hyp, grids=grids, nhyp=nhyp, names=np.array(names))
            print(f"[{m}] chunk {ci+1}/{n_chunks} saved ({len(part)} images, {time.time()-t0:.0f}s)", flush=True)
        del model
        torch.cuda.empty_cache() if dev.type == "cuda" else None


# ----------------------------------------------------------------------------------------------
def stage_marked(a, wm, dev, methods, files, t_start):
    conds = conditions(a.attacks.split(",") if a.attacks else None)
    lp = LPIPSMeter(dev)
    codec = None
    if a.codec:
        from muhr.crypto import phash16
        from muhr_exp.codec2 import DESIGNS, CodecV2
        keyfile = Path(a.out) / "exp_master.key"          # fresh experiment key (needed offline to verify)
        if not keyfile.exists():
            keyfile.write_text(generate_master_key())
        codec = CodecV2(derive_keys(keyfile.read_text().strip()), DESIGNS[a.codec])
        authors = np.random.default_rng(a.seed).integers(0, 2 ** 24, size=len(files))
    for m in methods:
        out = Path(a.out) / m; out.mkdir(parents=True, exist_ok=True)
        model = B.load_method(m, a.vs_root, dev)
        k = B.n_bits(model)
        write_json(out / "meta.json", {"args": vars(a), "env": env_info(), "method": m, "bits": k,
                                       "proc_size": B.proc_size(model), "conditions": [[c[0], c[1]] for c in conds],
                                       "n_files": len(files), "max_hyp": MAX_HYP, "codec": a.codec, "day": 9800})
        n_chunks = (len(files) + a.chunk - 1) // a.chunk
        for ci in range(n_chunks):
            fn = out / f"chunk_{ci:04d}.npz"
            if fn.exists():
                continue
            if (time.time() - t_start) / 3600 > a.max_hours:
                print("time budget reached; rerun to resume", flush=True)
                return
            idx = list(range(ci * a.chunk, min((ci + 1) * a.chunk, len(files))))
            C = len(conds)
            preds = np.zeros((len(idx), C, 1 + k), np.float16)
            msgs = np.zeros((len(idx), k), np.uint8)
            q = np.zeros((len(idx), 3), np.float32)
            ours = np.zeros((len(idx), MAX_HYP, 16), np.uint8)      # U-cross: our decoder on foreign marks
            ours_n = np.zeros(len(idx), np.uint8)
            sizes = np.zeros((len(idx), 2), np.int32)
            names, t0 = [], time.time()
            for j, i in enumerate(idx):
                img = load_rgb(files[i], a.max_side)
                names.append(files[i].name); sizes[j] = img.size
                msg = msg_for(a.seed, i, k)
                if codec is not None:
                    assert k >= 128, "codec needs >= 128 bits"
                    cw = codec.encode(int(authors[i]), 9800, phash16(img), 0)
                    msg[0, :128] = torch.from_numpy(cw.astype(np.float32))
                msgs[j] = msg[0].numpy().astype(np.uint8)
                stego = tensor_to_pil(B.embed(model, to_t(img, dev), msg))
                qq = quality(img, stego, lp); q[j] = (qq["psnr"], qq["ssim"], qq["lpips"])
                atts = [to_t(apply_real_attack(stego, an, p).convert("RGB"), dev) for an, p in conds]
                for c, xa in enumerate(atts):
                    preds[j, c] = B.detect(model, xa)[0].cpu().numpy()
                hb, _, _, nh = all_hypotheses_batch(wm, [stego], a.max_n, False)[0]
                ours[j], ours_n[j] = np.packbits(hb, axis=1), nh
            np.savez_compressed(fn, preds=preds, msgs=msgs, quality=q, ours=ours, ours_n=ours_n, sizes=sizes,
                                names=np.array(names), index=np.array(idx))
            print(f"[{m}] chunk {ci+1}/{n_chunks} saved ({len(idx)} images, {time.time()-t0:.0f}s) | "
                  f"PSNR {q[:, 0].mean():.2f} | clean acc "
                  f"{np.mean((preds[:, 0, 1:] > 0) == msgs.astype(bool)):.3f}", flush=True)
        del model
        torch.cuda.empty_cache() if dev.type == "cuda" else None


def main():
    a = parse()
    t_start = time.time()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm = ImageWatermarker(a.ckpt, derive_keys(generate_master_key()))     # keyless decoder; throwaway key
    a.regen_meta = None
    if a.regen:
        from muhr_exp import regen
        a.regen_meta = regen.register()
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    files = list_images(a.data)[a.start: a.start + a.n]
    print("device", dev, "| methods", methods, "| images", len(files), flush=True)
    if a.stage == "smoke":
        stage_smoke(a, wm, dev, methods, files)
    elif a.stage == "unmarked":
        stage_unmarked(a, wm, dev, methods, files, t_start)
    else:
        stage_marked(a, wm, dev, methods, files, t_start)
    print("done", flush=True)


if __name__ == "__main__":
    main()
