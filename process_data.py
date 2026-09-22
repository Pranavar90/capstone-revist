"""
Leakage-free, metadata-preserving preprocessor.
================================================
Fixes the old pipeline (merge -> shuffle -> random split), which leaked the same
clear scene across train/val/test and discarded every environmental label.

This version:
  - Parses ALL datasets, keeping each pair's condition metadata.
  - Splits by SCENE (group = dataset:scene_id), stratified per dataset, so no
    scene's variants ever cross splits -> no content leakage.
  - Writes data/processed/{split}/{hazy,clear}/{i}.png AND a meta.csv per split
    carrying: dataset, scene_id, beta, condition, real_synth  (for FiLM + physics).
  - Resizes pairs to 256x256 (model input size).

Transmission-map supervision is intentionally NOT used (archive trans maps
distrusted); t is left to a dark-channel prior / physics-consistency downstream.

Run:  python process_data.py            (full run, overwrites data/processed)
      python process_data.py --dry-run  (parse + split + leakage check, no writes)
"""
import os
import re
import csv
import sys
import random
from collections import Counter

import cv2
from tqdm import tqdm

IMAGE_SIZE = 256
SEED = 42
RAW = "data/raw"
OUT = "data/processed"
SPLIT = (0.8, 0.1, 0.1)   # train, val, test  (by scene)
EXTS = (".png", ".jpg", ".jpeg")


def _rec(hazy, clear, dataset, scene_id, condition, real_synth, beta=None, A=None):
    return dict(hazy=hazy, clear=clear, dataset=dataset, scene_id=str(scene_id),
                condition=condition, real_synth=real_synth, beta=beta, A=A)


def collect():
    """Walk every raw dataset -> list of pair records with metadata."""
    recs = []
    T = os.path.join(RAW, "thesis")

    # archive(1): hazy/{id}_{variant}_{A}.png , clear/{id}.png
    # toks[2] is ATMOSPHERIC LIGHT, not beta: measured within-scene correlation with the
    # airlight estimate is +0.904 while correlation with contrast destruction is -0.087,
    # and it is bounded in [0.70, 1.00]. It was previously written into the `beta` column,
    # which fed airlight into the density head for 93% of all labels.
    a_h, a_c = os.path.join(T, "archive(1)", "hazy"), os.path.join(T, "archive(1)", "clear")
    if os.path.isdir(a_h):
        for f in os.listdir(a_h):
            if not f.lower().endswith(EXTS):
                continue
            toks = os.path.splitext(f)[0].split("_")
            sid = toks[0]
            clear = os.path.join(a_c, sid + ".png")
            if not os.path.exists(clear):
                clear = os.path.join(a_c, sid + ".jpg")
            if not os.path.exists(clear):
                continue
            A = None
            try:
                A = float(toks[2])
            except (ValueError, IndexError):
                pass
            recs.append(_rec(os.path.join(a_h, f), clear, "archive", sid, "synth", "synth",
                             beta=None, A=A))

    # NH-HAZE (real, non-homogeneous): {n}_hazy.png / {n}_GT.png
    nh = os.path.join(T, "NH-HAZE", "NH-HAZE")
    if os.path.isdir(nh):
        for f in os.listdir(nh):
            if f.endswith("_hazy.png"):
                gt = os.path.join(nh, f.replace("_hazy.png", "_GT.png"))
                if os.path.exists(gt):
                    recs.append(_rec(os.path.join(nh, f), gt, "NH-HAZE",
                                     f.split("_")[0], "nonhomogeneous", "real"))

    # I-HAZE / O-HAZE (real): {ds}/{sub}/{hazy,GT}
    for ds, sub, cond in [("I-HAZE", "I-HAZY NTIRE 2018", "indoor"),
                          ("O-HAZE", "O-HAZY NTIRE 2018", "outdoor")]:
        hd, gd = os.path.join(T, ds, sub, "hazy"), os.path.join(T, ds, sub, "GT")
        if not os.path.isdir(hd):
            continue
        gt_files = os.listdir(gd) if os.path.isdir(gd) else []
        for f in os.listdir(hd):
            if not f.lower().endswith(EXTS):
                continue
            sid = re.split(r"[_#]", os.path.splitext(f)[0])[0]
            gt = os.path.join(gd, f)
            if not os.path.exists(gt):
                gt = os.path.join(gd, f.replace("hazy", "GT"))
            if not os.path.exists(gt):
                cands = [g for g in gt_files if g.startswith(sid)]
                gt = os.path.join(gd, cands[0]) if cands else gt
            if os.path.exists(gt):
                recs.append(_rec(os.path.join(hd, f), gt, ds, sid, cond, "real"))

    # Dense_Haze (real, extreme): {hazy,GT}, {n}_hazy.png / {n}_GT.png
    dh = os.path.join(T, "Dense_Haze_NTIRE19")
    hd, gd = os.path.join(dh, "hazy"), os.path.join(dh, "GT")
    if os.path.isdir(hd):
        for f in os.listdir(hd):
            if not f.lower().endswith(EXTS):
                continue
            gt = os.path.join(gd, f.replace("_hazy", "_GT"))
            if not os.path.exists(gt):
                gt = os.path.join(gd, f)
            if os.path.exists(gt):
                recs.append(_rec(os.path.join(hd, f), gt, "Dense_Haze",
                                 f.split("_")[0], "dense", "real"))

    # SOTS (synth): indoor {id}_{v}.png ; outdoor {id}_{A}_{beta}.jpg ; gt {id}.*
    for mode in ["indoor", "outdoor"]:
        hd = os.path.join(T, "SOTS", "SOTS", mode, "hazy")
        gd = os.path.join(T, "SOTS", "SOTS", mode, "gt")
        if not os.path.isdir(hd):
            continue
        for f in os.listdir(hd):
            if not f.lower().endswith(EXTS):
                continue
            toks = os.path.splitext(f)[0].split("_")
            sid = toks[0]
            gt = next((os.path.join(gd, sid + e) for e in EXTS
                       if os.path.exists(os.path.join(gd, sid + e))), None)
            if not gt:
                continue
            # outdoor filenames are {id}_{A}_{beta}.jpg — the ONLY source of genuine beta.
            beta = A = None
            if mode == "outdoor":
                try:
                    A, beta = float(toks[1]), float(toks[-1])
                except (ValueError, IndexError):
                    pass
            recs.append(_rec(os.path.join(hd, f), gt, "SOTS", sid,
                             f"{mode}_synth", "synth", beta=beta, A=A))

    # BeDDE (real fog): {city}/fog/*  ->  {city}/gt/{city}_clear.*   (group by city)
    bd = os.path.join(T, "BeDDE", "BeDDE")
    if os.path.isdir(bd):
        for city in os.listdir(bd):
            cd = os.path.join(bd, city)
            fog = os.path.join(cd, "fog")
            gt = next((os.path.join(cd, "gt", f"{city}_clear{e}") for e in (".png", ".jpg")
                       if os.path.exists(os.path.join(cd, "gt", f"{city}_clear{e}"))), None)
            if os.path.isdir(fog) and gt:
                for f in os.listdir(fog):
                    if f.lower().endswith(EXTS):
                        recs.append(_rec(os.path.join(fog, f), gt, "BeDDE", city, "real_fog", "real"))

    # Haze1k (synth, density-stratified): {sub}/input , {sub}/target
    hz = os.path.join(RAW, "haze1k", "Distributed_haze1k")
    if os.path.isdir(hz):
        for sub in os.listdir(hz):
            ind, tgt = os.path.join(hz, sub, "input"), os.path.join(hz, sub, "target")
            if not (os.path.isdir(ind) and os.path.isdir(tgt)):
                continue
            dens = next((d for d in ("thin", "moderate", "thick") if d in sub), "mixed")
            tgt_files = os.listdir(tgt)
            for f in os.listdir(ind):
                if not f.lower().endswith(EXTS):
                    continue
                clear = os.path.join(tgt, f)
                if not os.path.exists(clear):
                    stem = os.path.splitext(f)[0]
                    cands = [g for g in tgt_files if os.path.splitext(g)[0] == stem]
                    clear = os.path.join(tgt, cands[0]) if cands else clear
                if os.path.exists(clear):
                    recs.append(_rec(os.path.join(ind, f), clear, "Haze1k",
                                     f"{sub}:{os.path.splitext(f)[0]}", f"haze1k_{dens}", "synth"))
    return recs


def split_scene_grouped(recs):
    """Assign each (dataset, scene_id) group to one split, stratified per dataset."""
    random.seed(SEED)
    by_ds = {}
    for r in recs:
        by_ds.setdefault(r["dataset"], set()).add(r["scene_id"])

    assign = {}
    for ds, scene_set in by_ds.items():
        sids = sorted(scene_set)
        random.shuffle(sids)
        n = len(sids)
        tr, va = int(SPLIT[0] * n), int((SPLIT[0] + SPLIT[1]) * n)
        for i, sid in enumerate(sids):
            assign[(ds, sid)] = "train" if i < tr else "val" if i < va else "test"

    splits = {"train": [], "val": [], "test": []}
    for r in recs:
        splits[assign[(r["dataset"], r["scene_id"])]].append(r)
    return splits


def _fit_short_side(img, size=IMAGE_SIZE):
    """Resize so the SHORTER side is `size`, preserving aspect ratio.

    The old unconditional resize(img, (256, 256)) squashed every image to square. Source
    aspect ratios run 0.80-1.53, so geometry was distorted by a different amount per
    dataset. The loader crops to a square afterwards (random for train, centre for eval).
    """
    h, w = img.shape[:2]
    s = size / min(h, w)
    return cv2.resize(img, (max(size, round(w * s)), max(size, round(h * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def write_split(name, recs):
    hazy_out, clear_out = os.path.join(OUT, name, "hazy"), os.path.join(OUT, name, "clear")
    os.makedirs(hazy_out, exist_ok=True)
    os.makedirs(clear_out, exist_ok=True)
    meta = []
    for i, r in enumerate(tqdm(recs, desc=name)):
        h, c = cv2.imread(r["hazy"]), cv2.imread(r["clear"])
        if h is None or c is None:
            continue
        # Pairs must stay pixel-aligned: give the clear image the hazy one's exact size.
        h = _fit_short_side(h)
        c = cv2.resize(c, (h.shape[1], h.shape[0]),
                       interpolation=cv2.INTER_AREA if c.shape[0] > h.shape[0] else cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(hazy_out, f"{i}.png"), h)
        cv2.imwrite(os.path.join(clear_out, f"{i}.png"), c)
        meta.append(dict(idx=i, dataset=r["dataset"], scene_id=r["scene_id"],
                         beta=("" if r["beta"] is None else r["beta"]),
                         A=("" if r["A"] is None else r["A"]),
                         condition=r["condition"], real_synth=r["real_synth"]))
    with open(os.path.join(OUT, name, "meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "dataset", "scene_id", "beta", "A",
                                          "condition", "real_synth"])
        w.writeheader()
        w.writerows(meta)
    return len(meta)


def main(dry_run=False):
    recs = collect()
    print(f"Collected {len(recs)} pairs:")
    print("  by dataset:", dict(Counter(r["dataset"] for r in recs)))
    print("  with beta :", sum(1 for r in recs if r["beta"] is not None), "(SOTS-outdoor only)")
    print("  with A    :", sum(1 for r in recs if r["A"] is not None), "(archive + SOTS-outdoor)")
    if not recs:
        print("ERROR: no pairs found. Is data/raw populated? Run download_datasets.py.")
        return

    splits = split_scene_grouped(recs)

    # LEAKAGE GUARD: a (dataset, scene_id) must live in exactly one split.
    where = {}
    for name, rs in splits.items():
        for r in rs:
            where.setdefault((r["dataset"], r["scene_id"]), set()).add(name)
    leaks = [k for k, v in where.items() if len(v) > 1]
    assert not leaks, f"LEAKAGE: {len(leaks)} scenes span splits, e.g. {leaks[:3]}"
    print("[check] scene-grouped split is leakage-free OK")

    for name in ["train", "val", "test"]:
        by_ds = dict(Counter(r["dataset"] for r in splits[name]))
        print(f"  {name}: {len(splits[name])} pairs | {by_ds}")

    if dry_run:
        print("\n[dry-run] no images written.")
        return

    for name in ["train", "val", "test"]:
        n = write_split(name, splits[name])
        print(f"  wrote {name}: {n} images + meta.csv")


if __name__ == "__main__":
    main(dry_run="--dry-run" in sys.argv)
