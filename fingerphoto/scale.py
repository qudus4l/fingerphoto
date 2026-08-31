"""Scale normalization — the make-or-break step for camera images.

A photo has unknown DPI. We estimate the dominant ridge period (px between
ridge peaks) from the FFT power spectrum of the central patch, then rescale so
ridges sit TARGET_PERIOD px apart, matching what 500-dpi-calibrated tools
(Gabor enhancement, SourceAFIS) expect.
"""
from __future__ import annotations

import cv2
import numpy as np

TARGET_PERIOD = 8.0  # px between ridges at pseudo-500dpi (literature: 7-9)


def _patch_period(
    patch: np.ndarray,
    min_period: float,
    max_period: float,
    harmonic_check: bool = False,
) -> tuple[float, float] | None:
    """(period, confidence) from one patch's radial FFT profile, or None if no
    convincing spectral peak (specular wash, crease, flat skin).
    harmonic_check is for the wide-band pad LOCALIZER only (locate.py), where
    the sub-harmonic trap is real; inside the pipeline's narrow calibrated
    band it mis-halves valid periods — keep it off there."""
    from scipy.signal import find_peaks

    s = patch.shape[0]
    p = patch.astype(np.float64)
    p -= p.mean()
    win = np.hanning(s)
    p *= np.outer(win, win)

    power = np.abs(np.fft.fftshift(np.fft.fft2(p))) ** 2
    yy, xx = np.indices(power.shape)
    r = np.hypot(yy - s // 2, xx - s // 2).astype(np.int64)
    radial = np.bincount(r.ravel(), weights=power.ravel()) / np.maximum(np.bincount(r.ravel()), 1)

    r_lo = max(2, int(np.ceil(s / max_period)))
    r_hi = min(len(radial) - 1, int(np.floor(s / min_period)))
    if r_hi <= r_lo + 2:
        return None
    band = radial[r_lo : r_hi + 1]

    peaks, props = find_peaks(band, prominence=0)
    if not len(peaks):
        return None
    best = int(np.argmax(props["prominences"]))
    r_peak = r_lo + int(peaks[best])

    # Harmonic check: fingerprint spectra often put a strong peak at HALF the
    # ridge frequency (period doubling via every-other-ridge contrast). If
    # there is substantial energy at 2x the chosen frequency, the finer period
    # is the real one.
    if harmonic_check:
        r_double = 2 * r_peak
        if r_double <= r_hi:
            w_half = max(1, r_peak // 8)
            neighborhood = band[max(0, r_double - r_lo - w_half) : r_double - r_lo + w_half + 1]
            if len(neighborhood) and neighborhood.max() >= 0.35 * band[r_peak - r_lo]:
                r_peak = r_double - r_lo - w_half + int(np.argmax(neighborhood)) + r_lo

    # confidence: peak power must stand out from the band's typical power
    ratio = float(band[r_peak - r_lo] / (np.median(band) + 1e-12))
    if ratio < 2.0:
        return None
    return s / r_peak, ratio


def _grid(gray: np.ndarray, grid: int):
    """CLAHE'd copy plus the patch geometry shared by the estimators."""
    g = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    h, w = gray.shape
    side = int(min(h, w) * 0.9)
    y0, x0 = (h - side) // 2, (w - side) // 2
    grid = max(1, min(grid, side // 48))  # small crops get a coarser grid
    s = side // grid
    s -= s % 2
    if s < 32:
        raise ValueError("image too small for grid period estimation")
    return g, y0, x0, grid, s, side


def ridge_evidence(
    gray: np.ndarray,
    min_period: float,
    max_period: float,
    grid: int = 6,
) -> tuple[float, float]:
    """Does this image contain ridge-frequency energy AT ALL?

    Returns (fraction of grid patches with a convincing spectral peak, median
    peak prominence). Measured on the RAW image, deliberately before Gabor
    enhancement: the enhancer is an oriented band-pass filter, so fed a blurred
    finger it still emits ridge-like texture and SourceAFIS finds plenty of
    minutiae in the hallucination. A minutiae count therefore cannot tell a real
    print from an invented one — this can.

    NEGATIVE RESULT — do not gate on this without new evidence. The idea was
    to reject captures with no real ridges before the enhancer invents some.
    On three photos it looked decisive, but across the full fixture set every
    summary statistic overlaps (fraction of patches >= 2.0, median, p75, mean,
    mean of top 3), measured on finger-mask patches only:

        matches      thumb_6172 0.73  thumb_6174 0.72  sister_6214 0.79
                     thumb_6173 0.46  sister_6176 0.05  sister_6175 0.07
        never match  QD 0.53  W 0.50  probes 0.16-0.22

    sister_6176 (0.05) sits below every failed web capture, and QD (0.53) above
    several photos that do match. So low spectral prominence in the raw crop
    does not imply an unusable print, and high prominence does not imply a
    usable one. Kept because the number is worth logging, and because the next
    attempt should start from these measurements rather than repeat them.
    """
    try:
        g, y0, x0, grid, s, _ = _grid(gray, grid)
    except ValueError:
        return 0.0, 0.0
    ratios = []
    for gy in range(grid):
        for gx in range(grid):
            patch = g[y0 + gy * s : y0 + (gy + 1) * s, x0 + gx * s : x0 + (gx + 1) * s]
            res = _patch_period(patch, min_period, max_period)
            ratios.append(res[1] if res is not None else 0.0)
    r = np.array(ratios)
    return float((r >= 2.0).mean()), float(np.median(r))


def estimate_ridge_period(
    gray: np.ndarray,
    min_period: float = 4.0,
    max_period: float = 120.0,
    grid: int = 4,
) -> float:
    """Ridge period in px as the confidence-weighted MEDIAN over a grid of
    patches. A single global FFT gets captured by creases/specular highlights;
    block-wise + median (Hong 1998 practice) lets good blocks outvote bad ones.
    Runs on a CLAHE'd copy so weak-ridge blocks still show a spectral peak."""
    g, y0, x0, grid, s, side = _grid(gray, grid)

    estimates: list[tuple[float, float]] = []
    for gy in range(grid):
        for gx in range(grid):
            patch = g[y0 + gy * s : y0 + (gy + 1) * s, x0 + gx * s : x0 + (gx + 1) * s]
            res = _patch_period(patch, min_period, max_period)
            if res is not None:
                estimates.append(res)
    if not estimates:
        # last resort: one global patch, no confidence gate
        res = _patch_period(g[y0 : y0 + side, x0 : x0 + side], min_period, max_period)
        if res is None:
            raise ValueError("could not find a ridge frequency")
        return res[0]

    periods = np.array([e[0] for e in estimates])
    weights = np.array([e[1] for e in estimates])
    order = np.argsort(periods)
    cum = np.cumsum(weights[order])
    median_idx = order[int(np.searchsorted(cum, cum[-1] / 2))]
    return float(periods[median_idx])


def ridge_period_band_for_fingertip(crop_width_px: int) -> tuple[float, float]:
    """Physical prior: a fingertip pad is ~16-25mm wide with ridge periods of
    0.3-0.6mm, i.e. roughly 25-85 ridges across the crop. Bounds the search so
    it cannot lock onto illumination gradients or background texture."""
    return crop_width_px / 90.0, crop_width_px / 25.0


def normalize_scale(gray: np.ndarray, period: float | None = None) -> tuple[np.ndarray, float]:
    """Rescale so ridge period becomes TARGET_PERIOD. Returns (image, factor)."""
    if period is None:
        period = estimate_ridge_period(gray)
    factor = TARGET_PERIOD / period
    resized = cv2.resize(
        gray,
        None,
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC,
    )
    return resized, factor
