"""Identity layer: photos -> multi-scale template sets -> 1:N identification.

An enrolled identity is a SET of templates: several photos, each extracted at
the estimated ridge period and at +/-13% scale variants. Set-vs-set score is
the max over the cross product — multi-sample enrollment is the single biggest
accuracy lever in the contactless literature, and the scale sweep absorbs the
residual error the per-image period estimator can't remove.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import afis
from .errors import AcquisitionError, ReasonCode
from .pipeline import photo_to_ridge_image
from .scale import ridge_period_band_for_fingertip
from .segment import crop_distal

SCALE_SWEEP = (0.87, 1.0, 1.15)
MIN_MINUTIAE = 15  # quality gate: refuse templates with fewer minutiae


def _templates_for_crop(
    bgr: np.ndarray,
    mask: np.ndarray | None,
    hint: float | None,
    sweep: tuple[float, ...],
    out_dir: Path | None,
    min_minutiae: int = MIN_MINUTIAE,
) -> list[bytes]:
    band = ridge_period_band_for_fingertip(bgr.shape[1])
    ridge, info = photo_to_ridge_image(
        bgr, out_dir=out_dir, period_band=band, mask=mask, initial_period=hint
    )
    base = info["ridge_period_px"]
    templates = []
    for m in sweep:
        try:
            if m == 1.0:
                r = ridge
            else:
                r, _ = photo_to_ridge_image(bgr, mask=mask, period=base * m)
            t = afis.extract_template(r)
        except ValueError:
            continue  # this scale variant degenerated; others may survive
        n = afis.minutiae_count(t)
        if n is None or n >= min_minutiae:
            templates.append(t)
    return templates


def photo_template_set(
    bgr: np.ndarray,
    crop: bool = True,
    sweep: tuple[float, ...] = SCALE_SWEEP,
    out_dir: Path | None = None,
    min_minutiae: int = MIN_MINUTIAE,
    stats: dict | None = None,
) -> list[bytes]:
    """All templates for one photo — the union over crop strategies (upright
    color crop AND texture localization, when both work) and scale variants.
    Set-based max scoring downstream picks whichever crop was right, so a
    photo that suits one strategy never loses to the other's failure.
    Raises AcquisitionError (a ValueError) if nothing passes the quality gate;
    `stats`, when given, is filled with per-stage wall times and counts."""
    if stats is None:
        stats = {}
    if not crop:
        t0 = time.monotonic()
        templates = _templates_for_crop(bgr, None, None, sweep, out_dir, min_minutiae)
        stats["full_frame_s"] = round(time.monotonic() - t0, 3)
        stats["templates"] = len(templates)
        if not templates:
            raise AcquisitionError(
                ReasonCode.QUALITY_TOO_LOW,
                f"no scale variant produced >= {min_minutiae} minutiae",
            )
        return templates

    templates: list[bytes] = []
    crops_found = 0
    t0 = time.monotonic()
    try:  # proven path for protocol (finger-up) captures
        c, m = crop_distal(bgr)
        crops_found += 1
        templates += _templates_for_crop(c, m, None, sweep, out_dir, min_minutiae)
    except (ValueError, cv2.error):
        pass
    stats["crop_distal_s"] = round(time.monotonic() - t0, 3)
    t0 = time.monotonic()
    try:  # orientation-free texture localization (tilted/sideways fingers)
        from .locate import crop_pad

        c, m, hint = crop_pad(bgr)
        crops_found += 1
        templates += _templates_for_crop(
            c, m, hint, sweep, (out_dir / "pad" if out_dir else None), min_minutiae
        )
    except (ValueError, cv2.error):
        pass
    stats["crop_pad_s"] = round(time.monotonic() - t0, 3)
    stats["templates"] = len(templates)
    if not templates:
        if crops_found == 0:
            raise AcquisitionError(
                ReasonCode.NO_FINGER_FOUND, "no crop strategy located a fingertip"
            )
        raise AcquisitionError(
            ReasonCode.QUALITY_TOO_LOW,
            f"no crop strategy produced >= {min_minutiae} minutiae",
        )
    return templates


def set_score(probe_set: list[bytes], enrolled_set: list[bytes]) -> float:
    return max(max(afis.match_many(p, enrolled_set)) for p in probe_set)


def verify(
    probe_set: list[bytes],
    enrolled_set: list[bytes],
    threshold: float = afis.THRESHOLD,
) -> tuple[bool, float]:
    """1:1 verification: does the probe belong to this enrolled identity?"""
    score = set_score(probe_set, enrolled_set)
    return score >= threshold, score


def enroll(photos: list[np.ndarray], crop: bool = True) -> list[bytes]:
    """Identity = union of template sets from several photos of one finger."""
    identity: list[bytes] = []
    for bgr in photos:
        identity.extend(photo_template_set(bgr, crop=crop))
    return identity


def identify(
    probe_set: list[bytes],
    gallery: dict[str, list[bytes]],
    threshold: float = afis.THRESHOLD,
) -> tuple[str | None, float]:
    """Best (identity, score) if the score clears the threshold, else (None, best)."""
    best_name, best = None, -1.0
    for name, enrolled in gallery.items():
        s = set_score(probe_set, enrolled)
        if s > best:
            best_name, best = name, s
    if best < threshold:
        return None, best
    return best_name, best
