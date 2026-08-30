"""4-finger capture test: split the hand photo into per-finger pads, run each
through the pipeline, then sanity-check matching:
  - different fingers of the same hand are IMPOSTORS to each other
  - every finger is an IMPOSTOR to Q's enrolled thumb
(A genuine per-finger test needs a second 4-finger capture.)
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fingerphoto import afis, identity
from fingerphoto.hand_split import split_fingers
from fingerphoto.pipeline import photo_to_ridge_image

PHOTO_DIR = ROOT / "data" / "samples" / "photo"
OUT = ROOT / "out"
BAND = (5.0, 22.0)  # absolute ridge band; distant capture = finer pixels


def finger_template_set(label: str, crop, mask, out_dir: Path) -> list[bytes]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "0_crop.png"), crop)
    ridge, info = photo_to_ridge_image(crop, out_dir=out_dir, period_band=BAND, mask=mask)
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
    print(f"  {label}: crop {crop.shape[1]}x{crop.shape[0]}, period {base}px, "
          f"{len(templates)} usable templates")
    return templates


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "hand_a"
    bgr = cv2.imread(str(PHOTO_DIR / f"{name}.jpg"))
    if bgr is None:
        sys.exit(f"no {name}.jpg in {PHOTO_DIR}")

    print(f"splitting {name} ({bgr.shape[1]}x{bgr.shape[0]}):")
    (OUT / "hand").mkdir(parents=True, exist_ok=True)
    fingers = split_fingers(bgr, debug_path=str(OUT / "hand" / f"{name}_debug.png"))
    sets = {}
    for label, crop, mask in fingers:
        try:
            s = finger_template_set(label, crop, mask, OUT / f"{name}_{label}")
        except ValueError as e:
            print(f"  {label}: unusable ({e})")
            continue
        if s:
            sets[label] = s

    thr = afis.THRESHOLD
    print(f"\ncross-finger pairs (same hand, DIFFERENT fingers -> must NOT match), threshold {thr}:")
    for a, b in combinations(sorted(sets), 2):
        s = identity.set_score(sets[a], sets[b])
        print(f"  {a} vs {b}: {s:6.1f}  {'ok' if s < thr else 'FALSE ACCEPT'}")

    print("\nvs Q's enrolled thumb (impostor):")
    q_thumb = [t for p in sorted(PHOTO_DIR.glob("thumb_*.jpg"))
               for t in identity.photo_template_set(cv2.imread(str(p)))]
    for label in sorted(sets):
        s = identity.set_score(sets[label], q_thumb)
        print(f"  {label} vs Q-thumb: {s:6.1f}  {'ok' if s < thr else 'FALSE ACCEPT'}")


if __name__ == "__main__":
    main()
