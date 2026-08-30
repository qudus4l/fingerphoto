"""Multi-capture hand test: split every hand photo into per-finger template
sets, then for each pair of captures print the full finger-vs-finger score
matrix, a greedy position assignment, and the FUSED (sum-rule) score.

Genuine pairs (two captures of the same hand) should fuse far above
len(assignment) x threshold-ish; different hands should stay flat.
Single-finger photos are matched against every finger of every hand.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fingerphoto import afis, identity
from fingerphoto.hand_split import split_fingers
from fingerphoto.pipeline import photo_to_ridge_image

PHOTO_DIR = ROOT / "data" / "samples" / "photo"
OUT = ROOT / "out"
BAND = (5.0, 22.0)
HANDS = ["hand_a", "hand_b", "hand_c", "sis_hand_a", "sis_hand_b", "sis_hand_c"]
SINGLES = ["sis_finger_a", "sis_finger_b"]


def finger_set(crop, mask) -> list[bytes]:
    ridge, info = photo_to_ridge_image(crop, period_band=BAND, mask=mask)
    base = info["ridge_period_px"]
    templates = []
    for m in identity.SCALE_SWEEP:
        try:
            r = ridge if m == 1.0 else photo_to_ridge_image(crop, mask=mask, period=base * m)[0]
            t = afis.extract_template(r)
        except ValueError:
            continue
        n = afis.minutiae_count(t)
        if n is None or n >= identity.MIN_MINUTIAE:
            templates.append(t)
    return templates


def hand_sets(name: str) -> dict[str, list[bytes]]:
    bgr = cv2.imread(str(PHOTO_DIR / f"{name}.jpg"))
    if bgr is None:
        return {}
    (OUT / "hand").mkdir(parents=True, exist_ok=True)
    try:
        fingers = split_fingers(bgr, debug_path=str(OUT / "hand" / f"{name}_debug.png"))
    except ValueError as e:
        print(f"  {name}: split failed ({e})")
        return {}
    sets = {}
    for label, crop, mask in fingers:
        d = OUT / f"{name}_{label}"
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "0_crop.png"), crop)
        try:
            s = finger_set(crop, mask)
        except ValueError:
            s = []
        if s:
            sets[label] = s
    print(f"  {name}: {len(sets)} usable fingers ({', '.join(sorted(sets))})")
    return sets


def fused(a: dict, b: dict) -> tuple[float, list]:
    """Greedy one-to-one position assignment maximizing summed score."""
    pairs = sorted(
        ((identity.set_score(a[i], b[j]), i, j) for i in a for j in b), reverse=True
    )
    used_a, used_b, picked = set(), set(), []
    for s, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        picked.append((i, j, s))
    return sum(s for _, _, s in picked), picked


def main() -> None:
    print("splitting hand captures:")
    hands = {n: hand_sets(n) for n in HANDS}
    hands = {k: v for k, v in hands.items() if v}

    print("\nsingle-finger photos:")
    singles = {}
    for n in SINGLES:
        bgr = cv2.imread(str(PHOTO_DIR / f"{n}.jpg"))
        if bgr is None:
            continue
        try:
            singles[n] = identity.photo_template_set(bgr, out_dir=OUT / n)
            print(f"  {n}: {len(singles[n])} templates")
        except ValueError as e:
            print(f"  {n}: unusable ({e})")

    thr = afis.THRESHOLD
    print("\ncapture-pair fusion (per-finger matrix -> greedy assignment -> summed score):")
    for na, nb in combinations(hands, 2):
        total, picked = fused(hands[na], hands[nb])
        detail = "  ".join(f"{i}-{j}:{s:.0f}" for i, j, s in picked)
        n_over = sum(1 for _, _, s in picked if s >= thr)
        print(f"  {na} vs {nb}: fused={total:7.1f}  fingers>=thr: {n_over}/{len(picked)}   [{detail}]")

    print("\nsingle fingers vs every hand capture (best finger score):")
    for sn, sset in singles.items():
        for hn, hsets in hands.items():
            best = max((identity.set_score(sset, fs), lbl) for lbl, fs in hsets.items())
            print(f"  {sn} vs {hn}: {best[0]:6.1f} (best: {best[1]})")

    if len(singles) == 2:
        a, b = list(singles)
        s = identity.set_score(singles[a], singles[b])
        print(f"\n{a} vs {b} (same finger, two photos): {s:.1f}  "
              f"{'MATCH' if s >= thr else 'below threshold'}")


if __name__ == "__main__":
    main()
