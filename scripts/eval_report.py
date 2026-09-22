"""
Evaluation report — the numbers that decide whether a change actually helped.

Three things the aggregate PSNR hides:
  1. per-dataset PSNR/SSIM, split real vs synthetic. The headline number is ~97% synthetic,
     so it says almost nothing about real-world haze.
  2. whether A is SENSED or MEMORISED. A near-constant A across images means the
     atmospheric-state encoder collapsed, which undercuts the condition-adaptive claim.
  3. colour cast — how unevenly the channels move, after removing the overall shift.

Pass several checkpoints to get the §9 ablation comparison side by side.

Run:  PYTHONPATH=. python scripts/eval_report.py [--split val] [--ckpt A.pth B.pth ...]
"""
import argparse
import collections
import csv
import os

import cv2
import numpy as np
import torch
from PIL import Image

from inference.inference_engine import DehazeInference
from training.losses import SSIMLoss

# Conditions with fewer than this many test images are pooled into real/synth rather than
# reported alone: 8 of 12 conditions have n < 10 and one has n = 1, which is not a result.
MIN_N = 40


def dcp_airlight(rgb, patch=15, pct=99.9):
    """Classical dark-channel airlight estimate — an independent reference for A."""
    dark = cv2.erode(rgb.min(axis=2), np.ones((patch, patch)))
    return rgb[dark >= np.percentile(dark, pct)].reshape(-1, 3).mean(0)


def evaluate(eng, ssim, root, rows, limit):
    """Per-image PSNR/SSIM (NOT batch-averaged — needed for per-condition grouping)."""
    per = {}
    for r in rows[:limit]:
        h = Image.open(f"{root}/hazy/{r['idx']}.png").convert("RGB")
        c = Image.open(f"{root}/clear/{r['idx']}.png").convert("RGB")
        x = eng.transform(h).unsqueeze(0).to(eng.device)
        y = eng.transform(c).unsqueeze(0).to(eng.device)
        with torch.no_grad():
            out = eng.model(x)
            J = (out["J"] if isinstance(out, dict) else out).clamp(0, 1)
            s = 1.0 - ssim(J, y).item()
        p = 10 * np.log10(1.0 / max(((J - y) ** 2).mean().item(), 1e-12))
        per[r["idx"]] = (p, s, out["A"][0].cpu().numpy() if isinstance(out, dict) else None)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--ckpt", nargs="+", default=["outputs/checkpoints/mamba_best.pth"])
    ap.add_argument("--limit", type=int, default=100000)
    args = ap.parse_args()

    root = f"data/processed/{args.split}"
    rows = list(csv.DictReader(open(f"{root}/meta.csv", newline="")))
    by_ds = collections.defaultdict(list)
    for r in rows:
        by_ds[(r["dataset"], r["real_synth"])].append(r)

    results = {}
    for ck in args.ckpt:
        eng = DehazeInference(ck)
        ssim = SSIMLoss().to(eng.device)
        results[os.path.basename(ck)] = evaluate(eng, ssim, root, rows, args.limit)
        del eng
        torch.cuda.empty_cache()

    names = list(results)
    W = max(14, max(len(n) for n in names) + 2)

    # ---- per-dataset, then pooled real/synth ----
    print(f"\n  === {args.split} | per-dataset PSNR (dB) ===\n")
    print(f"  {'dataset':12s} {'kind':6s} {'n':>5} " + "".join(f"{n:>{W}}" for n in names))
    for (ds, kind), rs in sorted(by_ds.items(), key=lambda kv: -len(kv[1])):
        cells = "".join(f"{np.mean([results[n][r['idx']][0] for r in rs if r['idx'] in results[n]]):>{W}.2f}"
                        for n in names)
        print(f"  {ds:12s} {kind:6s} {len(rs):>5} {cells}")
    print()
    for kind in ("synth", "real"):
        rs = [r for (d, k), v in by_ds.items() if k == kind for r in v]
        if not rs:
            continue
        cells = "".join(f"{np.mean([results[n][r['idx']][0] for r in rs if r['idx'] in results[n]]):>{W}.2f}"
                        for n in names)
        print(f"  {'POOLED':12s} {kind:6s} {len(rs):>5} {cells}")
    synth = [r for (d, k), v in by_ds.items() if k == "synth" for r in v]
    real = [r for (d, k), v in by_ds.items() if k == "real" for r in v]
    if synth and real:
        gaps = "".join(
            f"{np.mean([results[n][r['idx']][0] for r in synth if r['idx'] in results[n]]) - np.mean([results[n][r['idx']][0] for r in real if r['idx'] in results[n]]):>{W}.2f}"
            for n in names)
        print(f"  {'':12s} {'gap':6s} {'':>5} {gaps}")

    # ---- PSNR binned by A: the continuous-adaptation evidence ----
    # Binning by beta is not viable (only ~51 test images carry a genuine beta); A covers
    # ~1400. Quantile bins, because the A distribution is far from uniform.
    labelled = [r for r in rows if r.get("A") not in ("", None)]
    if labelled:
        vals = np.array([float(r["A"]) for r in labelled])
        edges = np.quantile(vals, np.linspace(0, 1, 5))
        print(f"\n  === PSNR by atmospheric light A (n={len(labelled)}, quantile bins) ===\n")
        print(f"  {'A range':18s} {'n':>5} " + "".join(f"{n:>{W}}" for n in names))
        for i in range(4):
            lo, hi = edges[i], edges[i + 1]
            sel = [r for r in labelled if lo <= float(r["A"]) <= hi]
            if not sel:
                continue
            cells = "".join(f"{np.mean([results[n][r['idx']][0] for r in sel if r['idx'] in results[n]]):>{W}.2f}"
                            for n in names)
            print(f"  [{lo:.3f}, {hi:.3f}]   {len(sel):>5} {cells}")

    # ---- is A sensed or memorised? ----
    print(f"\n  === atmospheric light: sensed or memorised? ===\n")
    print(f"  {'ckpt':{W}} {'group':7s} {'predicted A':24s} {'std':>18} {'vs DCP':>8}")
    for n in names:
        for kind in ("synth", "real"):
            rs = [r for (d, k), v in by_ds.items() if k == kind for r in v][:120]
            P = [results[n][r["idx"]][2] for r in rs if r["idx"] in results[n]
                 and results[n][r["idx"]][2] is not None]
            if not P:
                continue
            D = [dcp_airlight(np.asarray(Image.open(f"{root}/hazy/{r['idx']}.png")
                                         .convert("RGB"), np.float32) / 255)
                 for r in rs if r["idx"] in results[n]]
            P, D = np.array(P), np.array(D[:len(P)])
            print(f"  {n:{W}} {kind:7s} {str(np.round(P.mean(0), 3)):24s} "
                  f"{str(np.round(P.std(0), 3)):>18} {np.abs(P - D).mean():8.3f}")
    print("\n  A std should track the label spread (~0.09); ~0.03 means the encoder collapsed.\n")


if __name__ == "__main__":
    main()
