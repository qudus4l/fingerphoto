"""Validate the SourceAFIS matching core on FVC2000 DB1_B.

10 fingers x 8 impressions of contact-sensor images at 500dpi. Genuine pairs
must score far above impostor pairs; SourceAFIS's nominal threshold is 40.
These are contact images, so no photo preprocessing is applied — this isolates
and proves the template + matching stages (06-07) of the pipeline.
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fingerphoto import afis

DATA = Path(__file__).resolve().parent.parent / "data" / "fvc"


def main() -> None:
    files = sorted(DATA.glob("*.tif"))
    if not files:
        sys.exit(f"no .tif files in {DATA}")
    by_finger: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        by_finger[f.stem.split("_")[0]].append(f)

    # CLAHE rescues faint/dry prints (finger 110: 0 minutiae raw -> ~44 enhanced)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(15, 15))
    print(f"extracting templates for {len(files)} images...")
    t0 = time.time()
    templates: dict[str, bytes] = {}
    for f in files:
        gray = clahe.apply(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))
        templates[f.stem] = afis.extract_template(gray, dpi=500)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s ({dt / len(files) * 1000:.0f} ms/image)")

    genuine: list[float] = []
    for finger, fs in sorted(by_finger.items()):
        for a, b in combinations([f.stem for f in fs], 2):
            genuine.append(afis.match_one(templates[a], templates[b]))

    impostor: list[float] = []
    fingers = sorted(by_finger)
    for fa, fb in combinations(fingers, 2):
        for imp in ("1", "2"):
            a = f"{fa}_{imp}"
            b = f"{fb}_{imp}"
            impostor.append(afis.match_one(templates[a], templates[b]))

    g, i = np.array(genuine), np.array(impostor)
    thr = afis.THRESHOLD
    print(f"\ngenuine  pairs: {len(g):4d}  mean {g.mean():6.1f}  min {g.min():6.1f}  max {g.max():6.1f}")
    print(f"impostor pairs: {len(i):4d}  mean {i.mean():6.1f}  min {i.min():6.1f}  max {i.max():6.1f}")
    print(f"\n@threshold {thr}:")
    print(f"  false rejects  (genuine < {thr}): {(g < thr).sum()} / {len(g)}  ({(g < thr).mean() * 100:.1f}%)")
    print(f"  false accepts (impostor >= {thr}): {(i >= thr).sum()} / {len(i)}  ({(i >= thr).mean() * 100:.1f}%)")

    thresholds = np.linspace(0, max(g.max(), i.max()), 2000)
    frr = np.array([(g < t).mean() for t in thresholds])
    far = np.array([(i >= t).mean() for t in thresholds])
    k = int(np.argmin(np.abs(frr - far)))
    print(f"  EER ~ {(frr[k] + far[k]) / 2 * 100:.2f}% at threshold {thresholds[k]:.1f}")


if __name__ == "__main__":
    main()
