"""Run the photo pipeline end-to-end on a real camera image of a fingertip.

Sanity checks, honestly labeled:
  1. photo -> ridge image -> template (does the pipeline produce minutiae?)
  2. genuine-ish probe: the SAME photo re-run through the pipeline after a
     synthetic perspective warp + blur (stand-in for "another photo of the same
     finger" until we have real multi-shot captures — real testing needs those)
  3. impostor probe: an FVC contact print of a different finger (score must be low)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fingerphoto import afis
from fingerphoto.pipeline import photo_to_ridge_image

PHOTO = ROOT / "data" / "samples" / "photo" / "fingertip_macro.jpg"
IMPOSTOR = ROOT / "data" / "fvc" / "101_1.tif"
OUT = ROOT / "out"


def synthetic_second_shot(bgr: np.ndarray) -> np.ndarray:
    """Mild perspective warp + rotation + blur, simulating a second capture."""
    h, w = bgr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32(
        [[w * 0.03, h * 0.02], [w * 0.985, h * 0.015], [w * 0.97, h * 0.98], [w * 0.02, h * 0.96]]
    )
    warped = cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(src, dst), (w, h))
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), 4.0, 1.0)
    rotated = cv2.warpAffine(warped, rot, (w, h))
    return cv2.GaussianBlur(rotated, (3, 3), 0.8)


def main() -> None:
    bgr = cv2.imread(str(PHOTO))
    if bgr is None:
        sys.exit(f"could not read {PHOTO}")
    print(f"photo: {PHOTO.name}  {bgr.shape[1]}x{bgr.shape[0]}")

    ridge, info = photo_to_ridge_image(bgr, out_dir=OUT / "enroll")
    print(f"pipeline info: {info}")
    enroll = afis.extract_template(ridge)
    n = afis.minutiae_count(enroll)
    print(f"enroll template: {len(enroll)} bytes, minutiae: {n if n is not None else 'n/a'}")

    probe_ridge, probe_info = photo_to_ridge_image(synthetic_second_shot(bgr), out_dir=OUT / "probe")
    probe = afis.extract_template(probe_ridge)
    n2 = afis.minutiae_count(probe)
    print(f"probe  template: {len(probe)} bytes, minutiae: {n2 if n2 is not None else 'n/a'}")

    genuine = afis.match_one(probe, enroll)

    imp_gray = cv2.imread(str(IMPOSTOR), cv2.IMREAD_GRAYSCALE)
    impostor = afis.match_one(probe, afis.extract_template(imp_gray, dpi=500))

    print(f"\nscores (threshold {afis.THRESHOLD}):")
    print(f"  same finger, warped re-capture (synthetic): {genuine:.1f}")
    print(f"  different finger (FVC {IMPOSTOR.stem}):       {impostor:.1f}")
    verdict = "PASS" if genuine >= afis.THRESHOLD > impostor else "NEEDS WORK"
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
