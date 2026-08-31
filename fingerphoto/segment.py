"""Stage 01: fingertip (distal phalange) segmentation and crop.

Skin detection via the Cr channel of YCrCb plus an HSV hue gate (Priesnitz
2022's recipe), largest connected component, then crop from the top of the
finger down by ~1.8x the finger width — the distal pad. Background pixels are
flattened to the median skin gray so the mask edge can't spawn false minutiae.
"""
from __future__ import annotations

import cv2
import numpy as np


def color_gate(bgr: np.ndarray) -> np.ndarray:
    """Raw skin-color mask (all regions, specular holes filled).

    The Cr lower bound adapts via Otsu (Priesnitz 2022): fixed bounds admit
    warm backgrounds (beige tiles, wood). Otsu can only TIGHTEN the gate —
    the fixed 135 floor still applies."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    hue = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 0]
    warm = (cr < 185) & ((hue < 30) | (hue > 150))
    raw = ((cr > 135) & warm).astype(np.uint8) * 255
    # Only when the fixed gate swallows an implausible share of the frame
    # (warm background like beige tiles) tighten adaptively via Otsu — the
    # Otsu bound on clean scenes can cut into shadowed skin.
    if cv2.countNonZero(raw) > 0.45 * raw.size:
        otsu_t, _ = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lo = max(135.0, float(otsu_t))
        raw = ((cr > lo) & warm).astype(np.uint8) * 255
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    return cv2.morphologyEx(raw, cv2.MORPH_CLOSE, small)  # fill specular holes


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    raw = color_gate(bgr)

    # Warm shadows on background can pass the color gate, attached to the
    # finger by thin bridges. Erode hard to cut bridges, keep the largest core,
    # then reconstruct back inside the raw mask so the finger keeps its shape.
    bridge = max(31, int(min(bgr.shape[:2]) * 0.02) | 1)
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge, bridge))
    eroded = cv2.erode(raw, k_bridge)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(eroded)
    if n < 2:
        raise ValueError("no skin region found")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sel = (labels == largest).astype(np.uint8) * 255
    sel = cv2.dilate(sel, k_bridge)
    return cv2.bitwise_and(sel, raw)


def crop_distal(
    bgr: np.ndarray, height_factor: float = 1.8, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Crop the fingertip pad, assuming the finger points roughly upward
    (topmost skin = tip). Returns (BGR crop with background flattened, mask).

    `mask` overrides the color-based segmentation — pass a learned mask (see
    localize_sam.find_finger) when the color gate is unreliable (dim scenes,
    warm backgrounds)."""
    if mask is None:
        mask = skin_mask(bgr)
    rows = np.where(mask.any(axis=1))[0]
    y_top = int(rows.min())

    # finger width: median blob width over the first stretch below the tip.
    # 400 is empirically tuned; the closed-loop scale estimator downstream is
    # sensitive enough that even 400->403 measurably shifts genuine scores,
    # so do not "generalize" this constant without an A/B on real photos.
    probe_rows = mask[y_top : min(y_top + 400, mask.shape[0])]
    widths = probe_rows.sum(axis=1) / 255
    width = float(np.median(widths[widths > 0]))

    y1 = min(mask.shape[0], int(y_top + height_factor * width))
    band = mask[y_top:y1]
    cols = np.where(band.any(axis=0))[0]
    pad = int(0.05 * width)
    x0 = max(0, int(cols.min()) - pad)
    x1 = min(mask.shape[1], int(cols.max()) + pad)

    crop = bgr[y_top:y1, x0:x1].copy()
    crop_mask = mask[y_top:y1, x0:x1]

    gray_skin = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[crop_mask > 0]
    flat = int(np.median(gray_skin))
    crop[crop_mask == 0] = (flat, flat, flat)
    return crop, crop_mask
