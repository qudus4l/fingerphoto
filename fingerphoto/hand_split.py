"""Split a multi-finger hand photo into individual distal-pad crops.

k-curvature fingertip detection: fingertip caps are sustained high-curvature
CONVEX arcs of the hand contour — webbings are concave (chord midpoint falls
outside the mask), inter-finger seams of TOUCHING fingers are straight, so
this finds tips whether fingers are spread or held together, at any
orientation. High-curvature corners along the border where the hand enters
the frame (wrist/knuckles) are excluded.

Each finger's axis and width come from the cap chord; crops are bounded by
finger width and the mask is restricted to the central finger strip so
touching neighbors cannot contribute (shared) minutiae.

Fingers are labeled f1..fN ordered around the entry anchor, so positions stay
consistent between enrollment and probe as long as the same hand side faces
the camera.
"""
from __future__ import annotations

import cv2
import numpy as np

from .locate import crop_guided
from .segment import skin_mask

BORDER_MARGIN = 12
ENTRY_STRIP = 10  # px strip along each border used to find the wrist side
ENTRY_EXCLUSION = 0.18  # fraction of the frame near the entry border: no tips
CURVATURE_PROMINENCE = 0.25  # radians
STRIP_HALF = 0.42  # central mask strip, fraction of finger width


def _entry(mask: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """(anchor point, side) where the hand enters the frame, or (None, None)."""
    h, w = mask.shape
    strips = {
        "top": (mask[:ENTRY_STRIP, :], lambda xs, ys: np.array([xs.mean(), 0.0])),
        "bottom": (mask[h - ENTRY_STRIP :, :], lambda xs, ys: np.array([xs.mean(), float(h - 1)])),
        "left": (mask[:, :ENTRY_STRIP], lambda xs, ys: np.array([0.0, ys.mean()])),
        "right": (mask[:, w - ENTRY_STRIP :], lambda xs, ys: np.array([float(w - 1), ys.mean()])),
    }
    best_side, best_count = None, 0
    for side, (strip, _) in strips.items():
        c = cv2.countNonZero(strip)
        if c > best_count:
            best_side, best_count = side, c
    if best_side is None or best_count == 0:
        return None, None
    strip, to_point = strips[best_side]
    ys, xs = np.nonzero(strip)
    return to_point(xs.astype(float), ys.astype(float)), best_side


def _curvature_tips(mask: np.ndarray, entry_side: str | None, max_tips: int):
    """[(tip, joint, width)] via k-curvature peaks on the hand contour."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    n = len(cnt)
    h, w = mask.shape
    k = max(60, int(0.055 * max(h, w)))
    if n < 3 * k:
        raise ValueError("hand contour too short for curvature analysis")

    a = np.roll(cnt, k, axis=0)
    c = np.roll(cnt, -k, axis=0)
    v1 = a - cnt
    v2 = c - cnt
    cosang = (v1 * v2).sum(axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9
    )
    sharp = np.pi - np.arccos(np.clip(cosang, -1, 1))

    mid = ((a + c) / 2).astype(int)
    mid[:, 0] = np.clip(mid[:, 0], 0, w - 1)
    mid[:, 1] = np.clip(mid[:, 1], 0, h - 1)
    convex = mask[mid[:, 1], mid[:, 0]] > 0

    ok = (
        (cnt[:, 0] > BORDER_MARGIN)
        & (cnt[:, 0] < w - BORDER_MARGIN)
        & (cnt[:, 1] > BORDER_MARGIN)
        & (cnt[:, 1] < h - BORDER_MARGIN)
    )
    if entry_side == "bottom":
        ok &= cnt[:, 1] < (1 - ENTRY_EXCLUSION) * h
    elif entry_side == "top":
        ok &= cnt[:, 1] > ENTRY_EXCLUSION * h
    elif entry_side == "left":
        ok &= cnt[:, 0] > ENTRY_EXCLUSION * w
    elif entry_side == "right":
        ok &= cnt[:, 0] < (1 - ENTRY_EXCLUSION) * w

    score = np.where(convex & ok, sharp, 0.0)
    score = uniform_filter1d(score, size=31, mode="wrap")

    peaks, _ = find_peaks(
        np.concatenate([score, score[:k]]), prominence=CURVATURE_PROMINENCE, distance=2 * k
    )
    peaks = np.unique(peaks % n)
    if len(peaks) == 0:
        raise ValueError("no fingertip curvature peaks found")
    order = np.argsort(-score[peaks])[:max_tips]
    tips_idx = peaks[order]

    out = []
    for i in tips_idx:
        tip = cnt[i]
        chord_mid = (a[i] + c[i]) / 2
        width = float(np.linalg.norm(a[i] - c[i]))
        u = tip - chord_mid
        u = u / (np.linalg.norm(u) + 1e-9)
        out.append((tip, tip - u * 1.2 * width, width))
    return out


def split_fingers(
    bgr: np.ndarray, max_fingers: int = 4, debug_path: str | None = None
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Returns up to max_fingers of (label, crop_bgr, mask), ordered around
    the entry anchor. Raises ValueError if no fingers are found."""
    mask = skin_mask(bgr)
    anchor, side = _entry(mask)
    fingers = _curvature_tips(mask, side, max_tips=max_fingers + 2)

    # drop duplicate peaks that resolved to (nearly) the same fingertip
    fingers.sort(key=lambda f: -f[2])
    deduped: list = []
    for tip, joint, width in fingers:
        if all(np.linalg.norm(tip - t2) > 1.2 * max(width, w2) for t2, _, w2 in deduped):
            deduped.append((tip, joint, width))
    fingers = deduped[:max_fingers]

    if debug_path:
        vis = bgr.copy()
        if anchor is not None:
            cv2.circle(vis, tuple(np.round(anchor).astype(int)), 40, (255, 0, 0), -1)
        for tip, joint, width in fingers:
            cv2.circle(vis, tuple(np.round(tip).astype(int)), 30, (0, 0, 255), -1)
            cv2.circle(vis, tuple(np.round(joint).astype(int)), 30, (0, 255, 0), -1)
            cv2.line(vis, tuple(np.round(tip).astype(int)), tuple(np.round(joint).astype(int)), (0, 255, 255), 8)
        cv2.imwrite(debug_path, cv2.resize(vis, None, fx=0.35, fy=0.35))

    if anchor is None:
        anchor = np.array([bgr.shape[1] / 2, bgr.shape[0] / 2])
    fingers.sort(key=lambda f: float(np.arctan2(*(f[0] - anchor)[::-1])))

    out = []
    for i, (tip, joint, width) in enumerate(fingers, start=1):
        crop, crop_mask = crop_guided(bgr, tuple(tip), tuple(joint), width=width)
        # touching fingers: keep only the central finger strip of the mask so
        # neighbors' slivers can't contribute (shared) minutiae
        cx = crop.shape[1] // 2
        half = int(STRIP_HALF * width)
        strip = np.zeros_like(crop_mask)
        strip[:, max(0, cx - half) : cx + half] = 255
        crop_mask = cv2.bitwise_and(crop_mask, strip)
        out.append((f"f{i}", crop, crop_mask))
    return out
