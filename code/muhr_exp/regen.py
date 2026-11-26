"""Regeneration (purification) attacks for E9, registered into muhr.image.data.REAL_ATTACKS on demand.

Conditions (name, parameter):
  Regen-ctrl  512        resize to 512x512 and back (the resize used by the diffusion attack, alone)
  Regen-VAE   kl         Stable Diffusion KL-VAE encode -> decode at 512x512 (latent mode, no noise)
  Regen-VAE   bmshj3     learned image codec bmshj2018-factorized, quality 3, native resolution
  Regen-VAE   cheng3     learned image codec cheng2020-anchor, quality 3, native resolution
  Regen-Diff  0.1 / 0.2  SD img2img at 512x512, empty prompt, no CFG, 20 scheduler steps x strength,
                         fixed seed (Zhao et al. 2024 style regeneration), then resized back

Models load lazily on the first call. If an optional model cannot be loaded (no internet, missing wheel),
the condition raises; run with --attacks restricted to what loads (see the notebook's smoke cell).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

SD_IDS = ["stable-diffusion-v1-5/stable-diffusion-v1-5", "CompVis/stable-diffusion-v1-4",
          "stabilityai/stable-diffusion-2-1-base"]
SIDE = 512
STEPS = 20
_cache: dict = {}


def _dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pipe():
    if "pipe" not in _cache:
        from diffusers import StableDiffusionImg2ImgPipeline
        dt = torch.float16 if _dev().type == "cuda" else torch.float32
        err = None
        for mid in SD_IDS:
            try:
                p = StableDiffusionImg2ImgPipeline.from_pretrained(mid, torch_dtype=dt, safety_checker=None,
                                                                   requires_safety_checker=False)
                p = p.to(_dev()); p.set_progress_bar_config(disable=True)
                _cache["pipe"], _cache["sd_id"] = p, mid
                print("regen: loaded", mid, flush=True)
                break
            except Exception as e:          # try the next mirror
                err = e
        if "pipe" not in _cache:
            raise RuntimeError(f"no Stable Diffusion checkpoint could be loaded: {err!r}")
    return _cache["pipe"]


def _codec(name: str):
    if name not in _cache:
        from compressai.zoo import bmshj2018_factorized, cheng2020_anchor
        fn = {"bmshj3": bmshj2018_factorized, "cheng3": cheng2020_anchor}[name]
        _cache[name] = fn(quality=3, pretrained=True).eval().to(_dev())
    return _cache[name]


def _to_t(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(_dev())


def _to_pil(x: torch.Tensor) -> Image.Image:
    a = (x[0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(a)


def _sq(img):
    return img.convert("RGB").resize((SIDE, SIDE), Image.Resampling.BICUBIC)


def _back(out, img):
    return out.resize(img.size, Image.Resampling.BICUBIC)


def regen_ctrl(img, p):
    return _back(_sq(img), img)


@torch.no_grad()
def regen_vae(img, p):
    if p == "kl":
        pipe = _pipe()
        x = _to_t(_sq(img)).to(pipe.vae.dtype) * 2 - 1
        z = pipe.vae.encode(x).latent_dist.mode()
        y = pipe.vae.decode(z).sample
        return _back(_to_pil((y.float() + 1) / 2), img)
    net = _codec(p)
    x = _to_t(img)
    H, W = x.shape[-2:]
    ph, pw = (-H) % 64, (-W) % 64
    xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")
    y = net(xp)["x_hat"][..., :H, :W]
    return _to_pil(y)


@torch.no_grad()
def regen_diff(img, s):
    pipe = _pipe()
    g = torch.Generator(device=_dev()).manual_seed(0)
    out = pipe(prompt="", image=_sq(img), strength=float(s), num_inference_steps=STEPS, guidance_scale=1.0,
               generator=g, output_type="pil").images[0]
    return _back(out, img)


REGEN = {
    "Regen-ctrl": (regen_ctrl, [SIDE]),
    "Regen-VAE": (regen_vae, ["kl", "bmshj3", "cheng3"]),
    "Regen-Diff": (regen_diff, [0.1, 0.2]),
}


def preflight() -> dict:
    """Run every condition once on a small test image; drop parameters whose model cannot be loaded."""
    test = Image.fromarray((np.random.default_rng(0).random((200, 300, 3)) * 255).astype(np.uint8))
    report = {}
    for name, (fn, params) in REGEN.items():
        keep = []
        for p in params:
            try:
                out = fn(test, p)
                assert out.size == test.size
                keep.append(p); report[f"{name}:{p}"] = "ok"
            except Exception as e:
                report[f"{name}:{p}"] = f"DROPPED: {e!r}"[:300]
        REGEN[name] = (fn, keep)
    return report


def register(check: bool = True) -> dict:
    """Add the regeneration conditions to REAL_ATTACKS (in place, so scripts that imported it see them)."""
    from muhr.image.data import REAL_ATTACKS
    rep = preflight() if check else {}
    for k, (fn, params) in REGEN.items():
        if params:
            REAL_ATTACKS[k] = (fn, params)
    print("regen preflight:", rep, flush=True)
    return {"sd_ids": SD_IDS, "sd_loaded": _cache.get("sd_id"), "side": SIDE, "steps": STEPS,
            "guidance": 1.0, "seed": 0, "preflight": rep}


def info() -> dict:
    return {"sd_id": _cache.get("sd_id"), "side": SIDE, "steps": STEPS}
