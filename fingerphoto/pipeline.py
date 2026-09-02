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

`info` also carries the ridge-quality measurement of the winning iteration
(quality.ridge_quality on the scale-normalized grayscale, BEFORE enhancement --
the enhancer invents ridges from blur, so a later measurement proves nothing)
plus the period-band diagnostics the caller's sanity rules need:
period_band_px, period_clamped (the seed estimate fell outside the physical
band for this crop width) and period_at_band_edge (the loop was still pushing
against a limit when it stopped).

Scale runs closed-loop: the raw-skin period estimate can be pulled off by
creases/specular highlights, so after enhancement we re-measure the period on
the (much cleaner) binary ridge map and correct until it hits TARGET_PERIOD.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .quality import ridge_quality
from .scale import TARGET_PERIOD, estimate_ridge_period, normalize_scale

MAX_SCALE_ITERATIONS = 3
PERIOD_TOLERANCE = 0.6  # px, on the enhanced map
BAND_EDGE_TOL = 0.02  # fraction of the band limit that still counts as "at the edge"


def photo_to_ridge_image(
    bgr: np.ndarray,
    out_dir: Path | None = None,
    period_band: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
    period: float | None = None,
    initial_period: float | None = None,
    quality: bool = True,
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
    seed_period = period  # kept pre-clamp: the band test below compares against it

    # The closed loop may only refine WITHIN the physical prior. The correction
    # below used to be unbounded, and period_band constrained only the initial
    # estimate — so on a soft crop, where the enhanced map carries no true
    # ridges, `achieved` comes back biased high (few FFT bins survive on an
    # ever-shrinking image, and they quantise upward) and the period inflated
    # ~1.33x per iteration without ever converging: 28.8 -> 38.3 -> 49.5 ->
    # 59.4 px on a 1023 px crop, i.e. 17 ridges across a thumb, rescaling the
    # print down to 165x181 px. The loop then returned that LAST, worst value.
    # Templates extracted from such a run still cleared the minutiae gate, so
    # the garbage entered the gallery silently and matched nothing.
    lo, hi = period_band if period_band is not None else (period / 2.0, period * 2.0)

    # Keep the iteration that landed CLOSEST to target, not the most recent —
    # a diverging loop's last step is its worst.
    best: tuple | None = None
    for attempt in range(1, MAX_SCALE_ITERATIONS + 1):
        period = min(max(period, lo), hi)
        scaled, factor = normalize_scale(gray, period)
        scaled_mask = _scaled_mask(mask, scaled.shape)

        inv = 255 - scaled
        smooth = cv2.bilateralFilter(inv, d=2, sigmaColor=4, sigmaSpace=1)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(15, 15))
        contrasted = clahe.apply(smooth)

        try:
            enhanced = _gabor_enhance(contrasted)
        except ValueError:
            if best is not None:
                break  # keep the best earlier iteration
            raise

        if scaled_mask is not None:
            enhanced[scaled_mask == 0] = 255  # white = valley/background

        if forced:
            best = (0.0, enhanced, period, factor, scaled.shape, scaled, contrasted, scaled_mask)
            break
        try:
            achieved = estimate_ridge_period(
                enhanced, min_period=TARGET_PERIOD * 0.6, max_period=TARGET_PERIOD * 1.8
            )
        except ValueError:
            if best is None:
                best = (float("inf"), enhanced, period, factor, scaled.shape, scaled,
                        contrasted, scaled_mask)
            break
        err = abs(achieved - TARGET_PERIOD)
        if best is None or err < best[0]:
            best = (err, enhanced, period, factor, scaled.shape, scaled, contrasted, scaled_mask)
            info["achieved_period_px"] = round(achieved, 2)
        if err <= PERIOD_TOLERANCE:
            break
        nxt = min(max(period * achieved / TARGET_PERIOD, lo), hi)
        if abs(nxt - period) < 1e-3:
            break  # the correction wants to leave the physical band — stop
        period = nxt

    err, enhanced, period, factor, shape, scaled, contrasted, scaled_mask = best
    info["ridge_period_px"] = round(period, 2)
    info["scale_factor"] = round(factor, 4)
    info["scale_iterations"] = attempt
    info["scaled_shape"] = shape
    # Did the loop actually reach the target? Callers (and the quality gate)
    # need to know that a template came from an unconverged scale estimate.
    info["scale_converged"] = bool(err <= PERIOD_TOLERANCE)
    # Band diagnostics for the caller's crop-sanity rules. A seed outside the
    # physical band means the crop is not a fingertip (a whole hand gives a
    # period the band cannot contain); landing on a limit means the loop was
    # still pushing outwards when it ran out of room.
    info["period_band_px"] = (round(lo, 2), round(hi, 2))
    info["period_clamped"] = bool(seed_period < lo or seed_period > hi)
    info["period_at_band_edge"] = bool(
        period <= lo * (1 + BAND_EDGE_TOL) or period >= hi * (1 - BAND_EDGE_TOL)
    )
    if quality:
        info["quality"] = ridge_quality(scaled, scaled_mask).as_dict()
    _save(out_dir, "2_scaled.png", scaled)
    _save(out_dir, "3_contrast.png", contrasted)
    _save(out_dir, "4_enhanced.png", enhanced)

    return enhanced, info


def _scaled_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Crop mask at the normalized scale, eroded ~2 ridge periods so the mask
    boundary itself cannot spawn ridges or minutiae. One helper, used both to
    blank the enhanced map and to bound the quality measurement, so the two can
    never drift apart."""
    if mask is None:
        return None
    out = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    return cv2.erode(out, k)


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
