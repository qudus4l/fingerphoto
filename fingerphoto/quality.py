"""Ridge-structure quality of a scale-normalized fingertip crop.

The "ridge detector": does this crop actually contain a fingerprint, or is it
blurred skin, a crease pattern, or a whole hand? Measured on the grayscale
AFTER scale normalization (ridge period ~TARGET_PERIOD px) and BEFORE Gabor
enhancement, because the enhancer is an oriented band-pass filter: fed an
out-of-focus finger it emits ridge-like texture and SourceAFIS happily finds
minutiae in the hallucination. Nothing downstream of the enhancer can tell a
real print from an invented one, so the measurement has to happen here.

NEGATIVE RESULT, do not repeat it: naive block orientation coherence (Sobel on
a lightly blurred image, the textbook feature and the one the contactless
quality literature ranks highest) does NOT separate on this data. Fraction of
blocks with coherence >= 0.5, measured on the project's own crops:

    good crops 0.54-0.71     bad crops 0.38-0.66  (a no-ridge web capture: 0.66)

Blurred skin has strongly coherent low-frequency shading. Two changes fix it:
band-pass the image first (difference of Gaussians, passband ~5-20 px at the
8 px normalized scale, which removes shading), and require a minimum gradient
energy per block (which removes the scale-free coherence of near-flat regions).
The same reasoning explains why scale.ridge_evidence failed: it measured raw
crops over a wide period band.

Three features are combined as a product (they are AND conditions, so one
failure sinks the score):

    coherence_frac  band-passed, energy-floored orientation coherence
    edge_density    Canny edge density inside the mask
    spectral_frac   fraction of patches with a ridge-band spectral peak

Spectral fraction is the strongest single feature and the only one that
catches whole-hand crops (they have coherent shading and plenty of edges, but
no energy at ridge frequencies once the crop is normalized against the wrong
period).

Measured on 23 labelled crops from this project's photos (2026-09-01,
scripts/quality_calibration.py):

    feature          AUC     good min    bad max
    score            1.000      0.733      0.071
    spectral_frac    1.000      0.420      0.135
    coherence_frac   0.952      0.467      0.546   <- overlaps; not usable alone
    edge_density     0.746      0.119      0.363   <- overlaps; not usable alone
    score_nospec     0.952      0.733      0.892   <- why spectral stays in the product

score_nospec is the same product without the spectral factor: a whole-hand crop
scores 0.89 on coherence and edges alone, which is why dropping spectral breaks
the separation.

Constants come from scripts/quality_calibration.py; re-run it whenever
TARGET_PERIOD, the scale loop, or the crop strategies change.

Port notes (the capture UI will want this on live frames): everything here is
two Gaussian blurs, a Sobel, box sums, and Canny. The frame must first be
rescaled so ridges sit ~8 px apart; capture.html already estimates that as
outline_px * 0.023.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .scale import _patch_period

BLOCK = 16
DOG_SIGMA_FINE = 1.0
DOG_SIGMA_COARSE = 3.0
COHERENCE_MIN = 0.5
ENERGY_RMS_MIN = 3.0
MASK_BLOCK_FRAC = 0.9      # a block counts only if this much of it is inside the mask
MIN_BLOCKS = 24            # fewer than this and the ROI is too small to judge
CANNY_LO, CANNY_HI = 50, 150
CLAHE_CLIP, CLAHE_TILES = 4.0, (15, 15)
SPECTRAL_PATCH = 48        # px at the normalized scale = 6 ridge periods (32 does not separate)
SPECTRAL_BAND = (5.0, 12.0)  # ridge period band in px, tight because the scale is normalized

# Ramp endpoints: below LO the factor is 0, above HI it is 1. Set by calibration.
CF_LO, CF_HI = 0.10, 0.60
ED_LO, ED_HI = 0.02, 0.08
SF_LO, SF_HI = 0.05, 0.30
USE_SPECTRAL = True
MIN_RIDGE_QUALITY = 0.40


@dataclass
class RidgeQuality:
    coherence_mean: float
    coherence_frac: float
    ocl_mean: float
    edge_density: float
    spectral_frac: float
    energy_median: float
    n_blocks: int
    score: float
    valid: bool

    def as_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = round(v, 4)
        d["n_blocks"] = int(self.n_blocks)
        d["valid"] = bool(self.valid)
        return d


def _empty(n_blocks: int = 0) -> RidgeQuality:
    return RidgeQuality(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, n_blocks, 0.0, False)


def _block_sum(a: np.ndarray, block: int) -> np.ndarray:
    """Sum over non-overlapping block x block tiles; trailing partial tiles are
    dropped so every tile has the same area."""
    by, bx = a.shape[0] // block, a.shape[1] // block
    if by == 0 or bx == 0:
        return np.zeros((0, 0), np.float32)
    trimmed = a[: by * block, : bx * block]
    return trimmed.reshape(by, block, bx, block).sum(axis=(1, 3))


def _ramp(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def combine(
    coherence_frac: float,
    edge_density: float,
    spectral_frac: float,
    use_spectral: bool = USE_SPECTRAL,
) -> float:
    """Product of three clipped ramps: every factor must be satisfied."""
    score = _ramp(coherence_frac, CF_LO, CF_HI) * _ramp(edge_density, ED_LO, ED_HI)
    if use_spectral:
        score *= _ramp(spectral_frac, SF_LO, SF_HI)
    return float(score)


def orientation_tensor(gray: np.ndarray, block: int = BLOCK):
    """Band-passed structure tensor, summed over blocks. Returns (Gxx, Gyy, Gxy)."""
    f = gray.astype(np.float32)
    bp = cv2.GaussianBlur(f, (0, 0), DOG_SIGMA_FINE) - cv2.GaussianBlur(f, (0, 0), DOG_SIGMA_COARSE)
    gx = cv2.Sobel(bp, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(bp, cv2.CV_32F, 0, 1, ksize=3)
    return _block_sum(gx * gx, block), _block_sum(gy * gy, block), _block_sum(gx * gy, block)


def edge_density(eq: np.ndarray, mask: np.ndarray | None) -> float:
    edges = cv2.Canny(eq, CANNY_LO, CANNY_HI)
    if mask is None:
        return float((edges > 0).mean())
    inside = cv2.countNonZero(mask)
    if inside == 0:
        return 0.0
    return float(cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask)) / inside)


def spectral_fraction(
    eq: np.ndarray,
    mask: np.ndarray | None,
    patch: int = SPECTRAL_PATCH,
    band: tuple[float, float] = SPECTRAL_BAND,
) -> float:
    """Fraction of patches showing a convincing spectral peak in the ridge band."""
    h, w = eq.shape
    hits = total = 0
    for y in range(0, h - patch + 1, patch):
        for x in range(0, w - patch + 1, patch):
            if mask is not None:
                m = mask[y : y + patch, x : x + patch]
                if cv2.countNonZero(m) < MASK_BLOCK_FRAC * patch * patch:
                    continue
            total += 1
            if _patch_period(eq[y : y + patch, x : x + patch], band[0], band[1]) is not None:
                hits += 1
    return float(hits / total) if total else 0.0


def ridge_quality(
    gray_scaled: np.ndarray, mask_scaled: np.ndarray | None = None, block: int = BLOCK
) -> RidgeQuality:
    """Score a scale-normalized fingertip ROI. `gray_scaled` is the grayscale at
    ~TARGET_PERIOD px ridge spacing; `mask_scaled` (nonzero = finger) should
    already be eroded away from the crop boundary."""
    if gray_scaled.ndim == 3:
        gray_scaled = cv2.cvtColor(gray_scaled, cv2.COLOR_BGR2GRAY)
    gray = np.ascontiguousarray(gray_scaled, dtype=np.uint8)
    h, w = gray.shape

    if mask_scaled is not None:
        if mask_scaled.shape[:2] != (h, w):
            mask_scaled = cv2.resize(mask_scaled, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = (mask_scaled > 0).astype(np.uint8) * 255
        if cv2.countNonZero(mask) == 0:
            return _empty()
    else:
        mask = None

    gxx, gyy, gxy = orientation_tensor(gray, block)
    if gxx.size == 0:
        return _empty()
    if mask is None:
        inside = np.ones(gxx.shape, bool)
    else:
        inside = _block_sum((mask > 0).astype(np.float32), block) >= MASK_BLOCK_FRAC * block * block
    n_blocks = int(inside.sum())
    if n_blocks < MIN_BLOCKS:
        return _empty(n_blocks)

    eps = 1e-6
    a, b, c = gxx[inside], gyy[inside], gxy[inside]
    tr = a + b
    coherence = np.sqrt((a - b) ** 2 + 4 * c * c) / (tr + eps)
    disc = np.sqrt(np.maximum(tr * tr / 4 - (a * b - c * c), 0.0))
    lmax, lmin = tr / 2 + disc, tr / 2 - disc
    ocl = 1.0 - lmin / (lmax + eps)
    energy_rms = np.sqrt(tr / (block * block))

    eq = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILES).apply(gray)
    ed = edge_density(eq, mask)
    sf = spectral_fraction(eq, mask)
    cf = float(((coherence >= COHERENCE_MIN) & (energy_rms >= ENERGY_RMS_MIN)).mean())

    return RidgeQuality(
        coherence_mean=float(coherence.mean()),
        coherence_frac=cf,
        ocl_mean=float(ocl.mean()),
        edge_density=ed,
        spectral_frac=sf,
        energy_median=float(np.median(energy_rms)),
        n_blocks=n_blocks,
        score=combine(cf, ed, sf),
        valid=True,
    )
