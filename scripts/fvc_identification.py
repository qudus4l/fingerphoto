"""1:N identification on FVC2000 DB1_B with set-based enrollment — the actual
product scenario: an identity = several templates, identify = max score across
all identities' template sets.

Enroll: impressions 1-4 of each of the 10 fingers. Probe: impressions 5-8.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fingerphoto import afis

DATA = Path(__file__).resolve().parent.parent / "data" / "fvc"
ENROLL_IMPRESSIONS = {"1", "2", "3", "4"}


def main() -> None:
    # CLAHE rescues faint/dry prints (finger 110: 0 minutiae raw -> ~44 enhanced)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(15, 15))
    gallery: dict[str, list[bytes]] = defaultdict(list)
    probes: list[tuple[str, bytes]] = []
    for f in sorted(DATA.glob("*.tif")):
        finger, impression = f.stem.split("_")
        gray = clahe.apply(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))
        template = afis.extract_template(gray, dpi=500)
        if impression in ENROLL_IMPRESSIONS:
            gallery[finger].append(template)
        else:
            probes.append((finger, template))

    correct = rejected = wrong = 0
    margins: list[float] = []
    for true_finger, probe in probes:
        best_finger, best_score, second = None, -1.0, -1.0
        for finger, templates in gallery.items():
            score = max(afis.match_many(probe, templates))
            if score > best_score:
                best_finger, second, best_score = finger, best_score, score
            elif score > second:
                second = score
        if best_score < afis.THRESHOLD:
            rejected += 1
            status = "REJECTED (below threshold)"
        elif best_finger == true_finger:
            correct += 1
            margins.append(best_score - second)
            status = "ok"
        else:
            wrong += 1
            status = f"WRONG -> {best_finger}"
        print(f"probe {true_finger}: best={best_finger} score={best_score:6.1f} runner-up={second:6.1f}  {status}")

    n = len(probes)
    print(f"\nidentified correctly: {correct}/{n} ({correct / n * 100:.0f}%)")
    print(f"rejected (no match >= {afis.THRESHOLD}): {rejected}/{n}")
    print(f"misidentified: {wrong}/{n}")
    if margins:
        print(f"mean margin over runner-up when correct: {np.mean(margins):.1f}")


if __name__ == "__main__":
    main()
