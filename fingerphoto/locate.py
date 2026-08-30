"""Orientation-free fingertip localization by ridge texture.

Hand detectors (MediaPipe) fail on macro close-ups, and the color crop in
segment.py assumes an upright finger with no other skin in frame. This module
finds the fingertip pad directly: it is the only skin region with strong
periodic energy at plausible ridge frequencies. Steps:

  1. downscale, skin color gate
  2. grid of patches over skin: spectral peak confidence in the ridge band
     (reuses scale._patch_period) -> winning patch = pad center + ridge period
  3. finger axis from minAreaRect of the local skin blob around the center
  4. tip polarity: the axis direction with NO skin beyond the pad is the tip
  5. rotate upright around the center, window out the finger, refine with the
     proven segment.crop_distal color crop
"""
from __future__ import annotations

import cv2
import numpy as np

from .scale import _patch_period
from .segment import color_gate, crop_distal

# True pad ridges on a phone photo at arm's length: both validated captures
# measured ~13-17px. Creases/skin folds read 22-30px — keep them OUT of band.
RIDGE_PERIOD_ORIG_BAND = (8.0, 20.0)
RIDGES_ACROSS_PAD = 60.0
MIN_PATCH_BRIGHTNESS = 80  # flash-lit ridges are bright; hair/shadows are not


def find_pad(bgr: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(center_xy in original px, ridge period in original px, confidence).
    Raises ValueError if no patch shows convincing ridge texture. The period
    is the median over the top-confidence patches — one patch can sit on a
    crease or pick a harmonic; the pad yields many agreeing patches."""
    s = 2400.0 / max(bgr.shape[:2])
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mask = color_gate(small)
    gray = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(
        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    )

    band = (max(3.5, RIDGE_PERIOD_ORIG_BAND[0] * s), RIDGE_PERIOD_ORIG_BAND[1] * s)
    patch = 160
    step = patch // 2
    candidates: list[tuple[float, np.ndarray, float]] = []  # (conf, center, period)
    h, w = gray.shape
    for y in range(0, h - patch, step):
        for x in range(0, w - patch, step):
            if cv2.countNonZero(mask[y : y + patch, x : x + patch]) < 0.7 * patch * patch:
                continue
            if np.median(gray[y : y + patch, x : x + patch]) < MIN_PATCH_BRIGHTNESS:
                continue
            res = _patch_period(
                gray[y : y + patch, x : x + patch], band[0], band[1], harmonic_check=True
            )
            if res is None:
                continue
            period, conf = res
            candidates.append((conf, np.array([x + patch / 2, y + patch / 2]), period))
    if not candidates:
        raise ValueError("no ridge-textured skin region found")

    candidates.sort(key=lambda c: -c[0])
    top = candidates[: min(5, len(candidates))]
    period_small = float(np.median([c[2] for c in top]))
    # center: best-confidence patch whose period agrees with the consensus
    for conf, center_small, p in top:
        if 0.7 <= p / period_small <= 1.4:
            return center_small / s, period_small / s, conf
    conf, center_small, _ = top[0]
    return center_small / s, period_small / s, conf


def crop_guided(
    bgr: np.ndarray,
    tip_xy: tuple[float, float],
    joint_xy: tuple[float, float],
    width: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Distal-pad crop given the finger tip and the distal joint (crease) as
    two points (px). Rotates so the finger points up, then refines with the
    color crop. width (px), when known, bounds the window so neighboring
    fingers stay out of the crop."""
    tip = np.asarray(tip_xy, dtype=float)
    joint = np.asarray(joint_xy, dtype=float)
    v = tip - joint
    length = float(np.linalg.norm(v))
    angle = float(np.degrees(np.arctan2(v[1], v[0]))) + 90.0

    if width is not None:
        out_w, out_h = int(1.8 * width), int(2.8 * width)
        m = cv2.getRotationMatrix2D((float(tip[0]), float(tip[1])), angle, 1.0)
        m[0, 2] += out_w / 2 - tip[0]
        m[1, 2] += out_h * 0.12 - tip[1]
    else:
        center = (tip + joint) / 2
        out_w, out_h = int(2.6 * length), int(3.2 * length)
        m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
        m[0, 2] += out_w / 2 - center[0]
        m[1, 2] += out_h * 0.45 - center[1]
    sub = cv2.warpAffine(bgr, m, (out_w, out_h), flags=cv2.INTER_CUBIC)
    return crop_distal(sub)


def crop_pad(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(distal-pad BGR crop, mask, ridge-period hint in px), any finger
    orientation, cluttered scenes OK. The period hint comes from the winning
    texture patch and is usually a better initial scale estimate than a global
    re-estimate on the crop."""
    center, period, _ = find_pad(bgr)
    w_est = float(np.clip(RIDGES_ACROSS_PAD * period, 300, 2500))

    full_mask = color_gate(bgr)

    # local skin blob around the pad -> finger axis
    window = np.zeros_like(full_mask)
    cv2.circle(window, tuple(np.round(center).astype(int)), int(w_est), 255, -1)
    local = cv2.bitwise_and(full_mask, window)
    n, labels = cv2.connectedComponents(local)
    label = labels[int(round(center[1])), int(round(center[0]))]
    if label == 0:  # center fell on a masked-out pixel; take nearest component
        ys, xs = np.nonzero(local)
        if len(ys) == 0:
            raise ValueError("no skin around ridge-textured region")
        i = int(np.argmin((xs - center[0]) ** 2 + (ys - center[1]) ** 2))
        label = labels[ys[i], xs[i]]
    pts = np.column_stack(np.nonzero(labels == label))[:, ::-1].astype(np.float32)
    (rcx, rcy), (rw, rh), ang = cv2.minAreaRect(pts)
    axis_deg = ang + 90.0 if rw < rh else ang
    d = np.array([np.cos(np.radians(axis_deg)), np.sin(np.radians(axis_deg))])

    # tip = axis direction with no skin beyond the pad
    def skin_beyond(direction: np.ndarray) -> int:
        probe = np.zeros_like(full_mask)
        p = center + direction * 1.3 * w_est
        p = np.clip(p, [0, 0], [full_mask.shape[1] - 1, full_mask.shape[0] - 1])
        cv2.circle(probe, tuple(np.round(p).astype(int)), int(0.4 * w_est), 255, -1)
        return cv2.countNonZero(cv2.bitwise_and(full_mask, probe))

    tip_dir = d if skin_beyond(d) < skin_beyond(-d) else -d

    # rotate so tip_dir points up, windowed directly around the pad center
    angle = float(np.degrees(np.arctan2(tip_dir[1], tip_dir[0]))) + 90.0
    out_w, out_h = int(2.6 * w_est), int(3.0 * w_est)
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
    m[0, 2] += out_w / 2 - center[0]
    m[1, 2] += out_h * 0.4 - center[1]
    sub = cv2.warpAffine(bgr, m, (out_w, out_h), flags=cv2.INTER_CUBIC)

    crop, mask = crop_distal(sub)
    return crop, mask, period
