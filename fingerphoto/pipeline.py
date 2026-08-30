"""Photo -> matchable ridge image.

Stages (parameters from published contactless pipelines — Kauba/Ruzicka 2023,
Priesnitz 2022):
  1. grayscale
  2. estimate ridge period, rescale to pseudo-500dpi (scale.py)
  3. invert polarity (camera ridges are BRIGHT; contact convention is dark)
  4. bilateral filter (d=2, sigma_color=4, sigma_space=1)
  5. CLAHE (clip 4, 15x15 tiles)
  6. oriented Gabor enhancement (Hong 1998 via fingerprint_enhancer)
  7. polarity fix: dark ridges on white for SourceAFIS

Scale runs closed-loop: the raw-skin period estimate can be pulled off by
creases/specular highlights, so after enhancement we re-measure the period on
the (much cleaner) binary ridge map and correct until it hits TARGET_PERIOD.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .scale import TARGET_PERIOD, estimate_ridge_period, normalize_scale

MAX_SCALE_ITERATIONS = 3
PERIOD_TOLERANCE = 0.6  # px, on the enhanced map


def photo_to_ridge_image(
    bgr: np.ndarray,
    out_dir: Path | None = None,
    period_band: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    period: float | None = None,
    initial_period: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Returns (ridge_image, info). ridge_image is uint8, dark ridges on white,
    pseudo-500dpi. If out_dir is set, intermediate stages are saved there.
    period_band optionally bounds the ridge-period search (px), e.g. from
    scale.ridge_period_band_for_fingertip() when bgr is a fingertip crop.
    mask (same size as bgr, nonzero = finger) suppresses everything outside
    the finger after enhancement. period forces the initial estimate and
    disables the closed-loop correction (diagnostics). initial_period seeds
    the loop (e.g. from locate.find_pad) but still lets it correct."""
    info: dict = {}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()
    _save(out_dir, "1_gray.png", gray)

    forced = period is not None
    if not forced:
        if initial_period is not None:
            period = initial_period
        elif period_band is not None:
            period = estimate_ridge_period(gray, min_period=period_band[0], max_period=period_band[1])
        else:
            period = estimate_ridge_period(gray)
    info["initial_period_px"] = round(period, 2)

    for attempt in range(1, MAX_SCALE_ITERATIONS + 1):
        scaled, factor = normalize_scale(gray, period)

        inv = 255 - scaled
        smooth = cv2.bilateralFilter(inv, d=2, sigmaColor=4, sigmaSpace=1)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(15, 15))
        contrasted = clahe.apply(smooth)

        try:
            enhanced = _gabor_enhance(contrasted)
        except ValueError:
            if attempt > 1:
                break  # keep the previous iteration's result
            raise

        if mask is not None:
            scaled_mask = cv2.resize(
                mask, (scaled.shape[1], scaled.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            # erode ~2 ridge periods so the boundary contour itself is wiped out
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
            scaled_mask = cv2.erode(scaled_mask, k)
            enhanced[scaled_mask == 0] = 255  # white = valley/background

        if forced:
            break
        try:
            achieved = estimate_ridge_period(
                enhanced, min_period=TARGET_PERIOD * 0.6, max_period=TARGET_PERIOD * 1.8
            )
        except ValueError:
            break
        info["achieved_period_px"] = round(achieved, 2)
        if abs(achieved - TARGET_PERIOD) <= PERIOD_TOLERANCE:
            break
        period *= achieved / TARGET_PERIOD

    info["ridge_period_px"] = round(period, 2)
    info["scale_factor"] = round(factor, 4)
    info["scale_iterations"] = attempt
    info["scaled_shape"] = scaled.shape
    _save(out_dir, "2_scaled.png", scaled)
    _save(out_dir, "3_contrast.png", contrasted)
    _save(out_dir, "4_enhanced.png", enhanced)

    return enhanced, info


def _gabor_enhance(gray: np.ndarray) -> np.ndarray:
    """Hong 1998 oriented-Gabor enhancement; output dark ridges on white."""
    import fingerprint_enhancer

    fn = getattr(fingerprint_enhancer, "enhance_fingerprint", None) or getattr(
        fingerprint_enhancer, "enhance_Fingerprint"
    )
    try:
        out = np.asarray(fn(gray))
    except (IndexError, ZeroDivisionError) as e:
        # the enhancer hard-crashes when its internal frequency estimation
        # finds no ridges at all (badly scaled or textureless input)
        raise ValueError(f"gabor enhancement found no ridges: {e}") from e
    if out.dtype == bool:
        out = out.astype(np.uint8) * 255
    elif out.max() <= 1:
        out = (out.astype(np.uint8)) * 255

    # The enhancer marks the ridge class as 255. Ridges are the pixels that were
    # DARK in our input (we inverted upstream) — flip so ridges end up dark.
    ridge_in_mean = gray[out == 255].mean() if (out == 255).any() else 255
    valley_in_mean = gray[out == 0].mean() if (out == 0).any() else 0
    if ridge_in_mean < valley_in_mean:
        out = 255 - out
    return out


def _save(out_dir: Path | None, name: str, img: np.ndarray) -> None:
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / name), img)
