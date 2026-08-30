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
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    g = clahe.apply(gray)

    h, w = gray.shape
    side = int(min(h, w) * 0.9)
    y0, x0 = (h - side) // 2, (w - side) // 2
    grid = max(1, min(grid, side // 48))  # small crops get a coarser grid
    s = side // grid
    s -= s % 2
    if s < 32:
        raise ValueError("image too small for grid period estimation")

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
