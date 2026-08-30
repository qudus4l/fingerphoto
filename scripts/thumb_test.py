"""The honest test: three REAL photos of the same thumb (different shots),
cross-matched against each other (genuine pairs) and against other fingers
(impostors: the Wikimedia macro fingertip photo + an FVC contact print).
Uses the identity layer: multi-scale template sets, max-score set matching.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fingerphoto import afis, identity

PHOTO_DIR = ROOT / "data" / "samples" / "photo"
THUMBS = sorted(PHOTO_DIR.glob("thumb_*.jpg"))
OUT = ROOT / "out"


def main() -> None:
    if len(THUMBS) < 2:
        sys.exit(f"need >=2 thumb_*.jpg in {PHOTO_DIR}")

    print("building template sets (distal crop + scale sweep):")
    sets = {}
    for p in THUMBS:
        sets[p.stem] = identity.photo_template_set(cv2.imread(str(p)), out_dir=OUT / p.stem)
        print(f"  {p.stem}: {len(sets[p.stem])} templates")
    macro = identity.photo_template_set(
        cv2.imread(str(PHOTO_DIR / "fingertip_macro.jpg")), crop=False, out_dir=OUT / "fingertip_macro"
    )
    fvc = [
        afis.extract_template(
            cv2.createCLAHE(4.0, (15, 15)).apply(
                cv2.imread(str(ROOT / "data" / "fvc" / "101_1.tif"), cv2.IMREAD_GRAYSCALE)
            ),
            dpi=500,
        )
    ]

    print(f"\ngenuine pairs (same thumb, different real photos), threshold {afis.THRESHOLD}:")
    for a, b in combinations(sorted(sets), 2):
        s = identity.set_score(sets[a], sets[b])
        print(f"  {a} vs {b}: {s:6.1f}  {'MATCH' if s >= afis.THRESHOLD else 'below threshold'}")

    print("\nimpostor pairs (different fingers):")
    for name in sorted(sets):
        s1 = identity.set_score(sets[name], macro)
        s2 = identity.set_score(sets[name], fvc)
        print(f"  {name} vs macro-photo finger: {s1:6.1f}   vs FVC contact 101: {s2:6.1f}")

    print("\n1:N identification (enroll 6172+6174 as 'Q-thumb', probe with 6173):")
    names = sorted(sets)
    gallery = {
        "Q-thumb": sets[names[0]] + sets[names[2]],
        "macro-finger": macro,
        "fvc-101": fvc,
    }
    who, score = identity.identify(sets[names[1]], gallery)
    print(f"  probe {names[1]} -> {who} (score {score:.1f})")


if __name__ == "__main__":
    main()
