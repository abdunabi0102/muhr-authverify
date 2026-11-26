"""E10: example marked images for the paper figure (no statistics).

For a few fixed T-COCO images: cover, our host network + G9 (global), Pixel Seal + G9 (codec in the first
128 of 256 bits, as in bl_codec), and Video Seal + G9. Saves PNGs and per-image PSNR / SSIM / LPIPS, and checks
that the keyed verifier accepts each clean marked image.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from muhr.crypto import derive_keys, generate_master_key, phash16                  # noqa: E402
from muhr.image.data import list_images                                            # noqa: E402
from muhr.image.watermarker import ImageWatermarker, tensor_to_pil                 # noqa: E402
from muhr_exp import baselines as B                                                # noqa: E402
from muhr_exp.codec2 import DESIGNS, CodecV2                                       # noqa: E402
from muhr_exp.runner import LPIPSMeter, all_hypotheses_batch, embed_bits, load_rgb, quality, write_json  # noqa: E402


def to_t(img, dev):
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(dev)


def verify_ours(wm, codec, img, max_n):
    hb, _, _, nh = all_hypotheses_batch(wm, [img], max_n, False)[0]
    msgs, ne = codec.bch_decode_many(codec.unwhiten(hb[:nh]))
    for h in np.nonzero(ne >= 0)[0]:
        r = codec.verify_msg(msgs[h])
        if r.ok:
            return True
    return False


def verify_bits(codec, bits128):
    msgs, ne = codec.bch_decode_many(codec.unwhiten(bits128[None]))
    return bool(ne[0] >= 0 and codec.verify_msg(msgs[0]).ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vs-root", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--indices", default="0,7,23,41,58,96")
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--max-n", type=int, default=4)
    p.add_argument("--hosts", default="pixelseal,videoseal")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keys = derive_keys(generate_master_key())                  # throwaway key, not saved
    wm = ImageWatermarker(a.ckpt, keys)
    codec = CodecV2(keys, DESIGNS["G9"])
    lp = LPIPSMeter(dev)
    files = list_images(a.data)
    idx = [int(x) for x in a.indices.split(",")]
    hosts = {h: B.load_method(h, a.vs_root, dev) for h in a.hosts.split(",") if h}
    rows = []
    rng = np.random.default_rng(0)
    for i in idx:
        img = load_rgb(files[i], a.max_side)
        stem = f"{i:04d}"
        img.save(out / f"{stem}_cover.png")
        bits = codec.encode(int(rng.integers(0, 2 ** 24)), 9800, phash16(img), 0)
        stego, _ = embed_bits(wm, img, bits, "global")
        stego.save(out / f"{stem}_ours.png")
        q = quality(img, stego, lp)
        rows.append({"index": i, "file": files[i].name, "host": "ours", **{k: float(v) for k, v in q.items()}, "verified": verify_ours(wm, codec, stego, a.max_n)})
        for name, model in hosts.items():
            k = B.n_bits(model)
            msg = torch.from_numpy(rng.integers(0, 2, size=(1, k)).astype(np.float32))
            msg[0, :128] = torch.from_numpy(bits.astype(np.float32))
            st = tensor_to_pil(B.embed(model, to_t(img, dev), msg))
            st.save(out / f"{stem}_{name}.png")
            pred = B.detect(model, to_t(st, dev))[0].cpu().numpy()
            dec = (pred[1:129] > 0).astype(np.uint8)
            rows.append({"index": i, "file": files[i].name, "host": name, **{kk: float(v) for kk, v in quality(img, st, lp).items()},
                         "verified": verify_bits(codec, dec)})
        print(rows[-3:], flush=True)
    write_json(out / "examples.json", rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
