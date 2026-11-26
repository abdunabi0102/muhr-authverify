"""Tasvir suv belgisi tarmoqlarini o'qitish (2.1-algoritm).

Misol:
    python scripts/train_image.py --data data/train --val-data data/val --out runs/taklif
Ablyatsiya (IV bob uchun):
    ... --no-stn            # fazoviy transformatorsiz
    ... --no-adv            # diskriminatorsiz
    ... --no-curriculum     # kurrikulumsiz (boshidan barcha hujumlar)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from muhr.image.data import ImageFolderDataset                      # noqa: E402
from muhr.image.metrics import bit_accuracy, psnr, ssim             # noqa: E402
from muhr.image.models import Decoder, Discriminator, Encoder, count_params  # noqa: E402
from muhr.image.noise import ATTACKS, VALIDATION_SUITE, NoiseLayer  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description="Muhr: tasvir suv belgisi tarmoqlarini o'qitish")
    p.add_argument("--data", required=True, help="o'quv tasvirlari papkasi")
    p.add_argument("--val-data", help="validatsiya papkasi (bo'lmasa o'quvdan 10%% ajratiladi)")
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--msg-len", type=int, default=128)
    p.add_argument("--base", type=int, default=64, help="enkoder/dekoder kanallar soni")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--lam-img", type=float, default=0.7)
    p.add_argument("--lam-ssim", type=float, default=0.1)
    p.add_argument("--lam-msg", type=float, default=1.5)
    p.add_argument("--lam-adv", type=float, default=1e-3)
    p.add_argument("--adv-start", type=int, default=5, help="lambda_A shu epoxadan oshira boshlanadi")
    p.add_argument("--adv-ramp", type=int, default=10)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--msg-warmup", type=int, default=3000,
                   help="dastlabki shuncha qadamda faqat xabar o'rganiladi (tasvir yo'qotishi vazni 0)")
    p.add_argument("--img-ramp", type=int, default=5000,
                   help="so'ng tasvir yo'qotishi vazni shuncha qadam davomida 0 dan 1 gacha oshiriladi")
    p.add_argument("--no-noise", action="store_true", help="hujumsiz o'qitish (ablyatsiya/tekshiruv)")
    p.add_argument("--no-stn", action="store_true")
    p.add_argument("--no-adv", action="store_true")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="tasvirlar sonini cheklash (sinov uchun)")
    p.add_argument("--val-limit", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--psnr-goal", type=float, default=40.0,
                   help="moslashuvchan lambda_I nazoratchisi ushlab turadigan PSNR maqsadi, dB")
    p.add_argument("--adapt-rate", type=float, default=4e-5,
                   help="nazoratchi qadami: log(mu) har qadamda rate * (maqsad - PSNR) ga o'zgaradi")
    p.add_argument("--acc-floor", type=float, default=0.93,
                   help="o'quv bit aniqligi (hujumlar ostida) shundan pastga tushsa, mu kamaytiriladi")
    p.add_argument("--mu-max", type=float, default=1e4, help="lambda_I ko'paytuvchisining yuqori chegarasi")
    p.add_argument("--no-adapt", action="store_true", help="moslashuvchan nazoratchini o'chirish (ablyatsiya)")
    p.add_argument("--target-psnr", type=float, default=38.0,
                   help="eng yaxshi modelni tanlashda talab qilinadigan minimal PSNR")
    p.add_argument("--resume", help="checkpoint yo'li")
    p.add_argument("--max-hours", type=float, default=None,
                   help="shuncha soatdan keyin keyingi epoxani boshlamay to'xtaydi (Kaggle/Colab sessiyalari uchun)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def validate(enc, dec, loader, args, device):
    enc.eval(); dec.eval()
    stats = {name: [] for name, _ in VALIDATION_SUITE}
    ps, ss = [], []
    g = torch.Generator().manual_seed(0)
    for cover in loader:
        cover = cover.to(device)
        msgs = torch.randint(0, 2, (cover.shape[0], args.msg_len), generator=g).float().to(device)
        stego = (cover + args.beta * enc(cover, msgs)).clamp(0, 1)
        stego = torch.round(stego * 255) / 255            # 8 bitli kvantlash
        ps.append(psnr(stego, cover)); ss.append(ssim(stego, cover).item())
        for name, params in VALIDATION_SUITE:
            att = ATTACKS[name].apply(stego, cover, **params)
            stats[name].append(bit_accuracy(dec(att), msgs))
    enc.train(); dec.train()
    out = {"psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}
    out.update({f"acc_{k}": float(np.mean(v)) for k, v in stats.items()})
    out["acc_mean"] = float(np.mean([out[f"acc_{k}"] for k, _ in VALIDATION_SUITE]))
    return out


def main():
    args = parse()
    try:
        sys.stdout.reconfigure(line_buffering=True)   # loglar darhol ko'rinsin
    except AttributeError:
        pass
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    full = ImageFolderDataset(args.data, args.block, train=True, limit=args.limit)
    if args.val_data:
        train_ds = full
        val_ds = ImageFolderDataset(args.val_data, args.block, train=False, limit=args.val_limit)
    else:
        n_val = max(1, len(full) // 10)
        files = full.files[:]
        random.Random(0).shuffle(files)
        val_ds = ImageFolderDataset.__new__(ImageFolderDataset)
        val_ds.files, val_ds.block, val_ds.train = files[:n_val][: args.val_limit], args.block, False
        full.files = files[n_val:]
        train_ds = full
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                          pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, num_workers=args.workers)

    enc = Encoder(args.msg_len, args.base).to(device)
    dec = Decoder(args.msg_len, args.base, use_stn=not args.no_stn).to(device)
    disc = Discriminator(args.base).to(device)
    print(f"Qurilma: {device} | o'quv: {len(train_ds)} | validatsiya: {len(val_ds)}")
    print(f"Parametrlar: enkoder {count_params(enc)/1e6:.2f}M, dekoder {count_params(dec)/1e6:.2f}M, "
          f"diskriminator {count_params(disc)/1e6:.2f}M")

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=args.lr, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)
    noise = NoiseLayer(args.epochs, curriculum=not args.no_curriculum)
    start, best, gstep = 0, -1.0, 0
    log_mu = 0.0          # moslashuvchan ko'paytuvchi: mu = exp(log_mu), lambda_I_eff = mu * lambda_I
    ema_psnr = ema_acc = None

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        enc.load_state_dict(ck["encoder"]); dec.load_state_dict(ck["decoder"])
        if "disc" in ck:
            disc.load_state_dict(ck["disc"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"]); opt_d.load_state_dict(ck["opt_d"]); sched.load_state_dict(ck["sched"])
        start, best, gstep = ck.get("epoch", 0) + 1, ck.get("best", -1.0), ck.get("gstep", 0)
        log_mu = ck.get("log_mu", 0.0)
        print(f"Davom ettirilmoqda: {start}-epoxadan")

    cfg = {"block": args.block, "msg_len": args.msg_len, "base": args.base, "stn": not args.no_stn,
           "beta": args.beta}
    log_path = out / "log.csv"
    log_new = not log_path.exists() or start == 0
    log_f = open(log_path, "w" if log_new else "a", newline="")
    writer = None

    run_t0, last_epoch_s = time.time(), 0.0
    for epoch in range(start, args.epochs):
        if args.max_hours is not None and epoch > start:
            elapsed = time.time() - run_t0
            if elapsed + 1.2 * last_epoch_s > args.max_hours * 3600:
                print(f"VAQT CHEGARASI: {elapsed/3600:.2f} soat o'tdi, keyingi epoxa sig'maydi. "
                      f"{epoch}-epoxadan --resume bilan davom ettiring.")
                break
        noise.set_epoch(epoch)
        if args.no_adv:
            lam_a = 0.0
        else:
            lam_a = args.lam_adv * min(1.0, max(0.0, (epoch - args.adv_start) / max(1, args.adv_ramp)))
        img_w = min(1.0, max(0.0, (gstep - args.msg_warmup) / max(1, args.img_ramp)))
        t0, agg, n = time.time(), {"loss": 0, "img": 0, "ssim": 0, "msg": 0, "adv": 0, "acc": 0, "d": 0}, 0
        for step, cover in enumerate(train_dl):
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break
            cover = cover.to(device, non_blocking=True)
            img_w = min(1.0, max(0.0, (gstep - args.msg_warmup) / max(1, args.img_ramp)))
            gstep += 1
            msgs = torch.randint(0, 2, (cover.shape[0], args.msg_len), device=device).float()
            # 2-qadam: oldinga hisoblash
            stego = cover + args.beta * enc(cover, msgs)
            stego = stego + (stego.clamp(0, 1) - stego).detach()     # STE: oldinga clamp, orqaga 1
            attacked = stego if args.no_noise else noise(stego, cover)[0]
            logits = dec(attacked)
            # 3-qadam: (2.10) maqsad funksiyasi
            l_img = F.mse_loss(stego, cover)
            l_ssim = 1 - ssim(stego, cover)
            l_msg = F.binary_cross_entropy_with_logits(logits, msgs)
            mu = math.exp(log_mu)
            loss = img_w * mu * (args.lam_img * l_img + args.lam_ssim * l_ssim) + args.lam_msg * l_msg
            l_adv = torch.zeros((), device=device)
            if lam_a > 0:
                l_adv = F.binary_cross_entropy_with_logits(disc(stego), torch.ones(cover.shape[0], 1, device=device))
                loss = loss + lam_a * l_adv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), args.clip)
            opt.step()
            # moslashuvchan nazoratchi (2.10a), ikki tomonlama:
            #  - bit aniqligi chegaradan past bo'lsa — mu kamayadi (xabar ustuvor);
            #  - aks holda PSNR maqsaddan past bo'lsa mu oshadi, yuqori bo'lsa kamayadi.
            psnr_b = -10.0 * math.log10(max(float(l_img.detach()), 1e-10))
            acc_b = bit_accuracy(logits.detach(), msgs)
            ema_psnr = psnr_b if ema_psnr is None else 0.99 * ema_psnr + 0.01 * psnr_b
            ema_acc = acc_b if ema_acc is None else 0.99 * ema_acc + 0.01 * acc_b
            if img_w >= 1.0 and not args.no_adapt:
                if ema_acc < args.acc_floor:
                    d = -args.adapt_rate * 400.0 * (args.acc_floor - ema_acc)
                else:
                    d = args.adapt_rate * (args.psnr_goal - ema_psnr)
                log_mu = min(math.log(args.mu_max), max(0.0, log_mu + max(-0.01, min(0.01, d))))
            # 4-qadam: diskriminator (2.11)
            l_d = torch.zeros(())
            if not args.no_adv and epoch >= args.adv_start:
                real = cover + 0.01 * torch.randn_like(cover)       # kichik instance-shovqin
                fake = stego.detach() + 0.01 * torch.randn_like(cover)
                ones = torch.ones(cover.shape[0], 1, device=device)
                l_d = F.binary_cross_entropy_with_logits(disc(real), ones) + \
                    F.binary_cross_entropy_with_logits(disc(fake), torch.zeros_like(ones))
                opt_d.zero_grad(set_to_none=True); l_d.backward(); opt_d.step()
            n += 1
            for k, v in (("loss", loss.detach()), ("img", l_img), ("ssim", l_ssim), ("msg", l_msg), ("adv", l_adv), ("d", l_d)):
                agg[k] += float(v)
            agg["acc"] += acc_b
        sched.step()
        agg = {k: v / max(1, n) for k, v in agg.items()}
        epoch_t0 = t0
        val = validate(enc, dec, val_dl, args, device)   # 5-qadam
        row = {"epoch": epoch, "lr": opt.param_groups[0]["lr"], "lam_adv": lam_a, "img_w": round(img_w, 3),
               "mu": round(math.exp(log_mu), 3),
               "time_s": round(time.time() - t0, 1), **{f"train_{k}": round(v, 5) for k, v in agg.items()},
               **{k: round(v, 5) for k, v in val.items()}}
        if writer is None:
            fields = list(row)
            if not log_new:
                # eski sessiya logi boshqa ustunlarga ega bo'lsa — yangi sarlavha bilan qayta yozish
                log_f.close()
                with open(log_path, encoding="utf-8") as fh:
                    old = list(csv.DictReader(fh))
                if old and list(old[0]) != fields:
                    fields += [k for k in old[0] if k not in fields]
                    with open(log_path, "w", newline="", encoding="utf-8") as fh:
                        w = csv.DictWriter(fh, fieldnames=fields, restval="")
                        w.writeheader(); w.writerows(old)
                log_f = open(log_path, "a", newline="", encoding="utf-8")
            writer = csv.DictWriter(log_f, fieldnames=fields, restval="")
            if log_new:
                writer.writeheader()
        writer.writerow(row); log_f.flush()
        print(f"[{epoch+1}/{args.epochs}] loss={agg['loss']:.4f} acc={agg['acc']:.3f} | "
              f"val PSNR={val['psnr']:.2f} SSIM={val['ssim']:.4f} acc_mean={val['acc_mean']:.4f} "
              f"jpeg50={val['acc_jpeg']:.3f} crop={val['acc_crop']:.3f} rot={val['acc_rotate']:.3f} "
              f"mu={row['mu']:.1f} ({row['time_s']}s)")

        state = {"encoder": enc.state_dict(), "decoder": dec.state_dict(), "disc": disc.state_dict(),
                 "opt": opt.state_dict(), "opt_d": opt_d.state_dict(), "sched": sched.state_dict(),
                 "config": cfg, "epoch": epoch, "val": val, "best": best, "gstep": gstep, "log_mu": log_mu}
        torch.save(state, out / "last.pt")
        score = val["acc_mean"] if val["psnr"] >= args.target_psnr else val["acc_mean"] - 1.0
        if score > best:
            best = score
            state["best"] = best
            torch.save({k: state[k] for k in ("encoder", "decoder", "config", "epoch", "val")}, out / "best.pt")
            print(f"   -> eng yaxshi model saqlandi (acc_mean={val['acc_mean']:.4f})")
        last_epoch_s = time.time() - epoch_t0
    log_f.close()
    done = torch.load(out / "last.pt", map_location="cpu", weights_only=False)["epoch"] + 1 >= args.epochs
    print(("O'QITISH TO'LIQ TUGADI." if done else "Sessiya yakunlandi, o'qitish hali tugamagan.")
          + f" Eng yaxshi model: {out / 'best.pt'}")


if __name__ == "__main__":
    main()
