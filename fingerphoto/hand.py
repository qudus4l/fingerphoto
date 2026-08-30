"""Orientation-aware thumb crop via MediaPipe Hands.

The color-mask crop in segment.py assumes the finger points up and is the only
skin in frame. Real photos have tilted thumbs, faces, arms. MediaPipe gives us
the thumb tip (landmark 4) and IP joint (landmark 3): rotate the image so the
thumb points up, take a generous crop around the tip, then hand off to the
proven skin-mask crop for the precise distal-pad cut.
"""
from __future__ import annotations

import cv2
import numpy as np

from pathlib import Path

from .segment import crop_distal

THUMB_TIP, THUMB_IP = 4, 3
MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "hand_landmarker.task"

_landmarker = None


def _get_landmarker():
    global _landmarker
    if _landmarker is None:
        from mediapipe.tasks.python import vision

        try:
            from mediapipe.tasks.python import BaseOptions
        except ImportError:
            from mediapipe.tasks.python.core.base_options import BaseOptions

        _landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                # CPU delegate: the Metal path aborts on this machine
                base_options=BaseOptions(
                    model_asset_path=str(MODEL_PATH), delegate=BaseOptions.Delegate.CPU
                ),
                num_hands=2,
                min_hand_detection_confidence=0.2,
            )
        )
    return _landmarker


def _thumb_axis(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(tip_xy, ip_xy) in pixels, from the hand whose thumb is most prominent."""
    import mediapipe as mp

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = _get_landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.hand_landmarks:
        return None

    best = None
    best_len = -1.0
    for hand in res.hand_landmarks:
        tip = np.array([hand[THUMB_TIP].x * w, hand[THUMB_TIP].y * h])
        ip = np.array([hand[THUMB_IP].x * w, hand[THUMB_IP].y * h])
        length = float(np.linalg.norm(tip - ip))
        if length > best_len:
            best, best_len = (tip, ip), length
    return best


def crop_thumb(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (distal-pad crop, mask) for the most prominent thumb, any
    orientation. Falls back to the finger-up color crop if no hand is found."""
    axis = _thumb_axis(bgr)
    if axis is None:
        return crop_distal(bgr)
    tip, ip = axis

    # rotate the full image so tip-minus-ip points straight up
    v = tip - ip
    angle_deg = 90.0 + float(np.degrees(np.arctan2(v[1], v[0])))
    h, w = bgr.shape[:2]
    m = cv2.getRotationMatrix2D((float(tip[0]), float(tip[1])), angle_deg, 1.0)
    # canvas large enough that nothing clips
    side = int(np.hypot(h, w))
    m[0, 2] += side / 2 - tip[0]
    m[1, 2] += side / 2 - tip[1]
    upright = cv2.warpAffine(bgr, m, (side, side), flags=cv2.INTER_CUBIC)

    # generous region around the tip (distal length L = |tip-ip|):
    # a bit above the tip, ~3L down, ~1.6L to each side
    length = float(np.linalg.norm(v))
    cx, cy = side / 2, side / 2
    x0 = max(0, int(cx - 1.6 * length))
    x1 = min(side, int(cx + 1.6 * length))
    y0 = max(0, int(cy - 0.5 * length))
    y1 = min(side, int(cy + 3.0 * length))
    region = upright[y0:y1, x0:x1]

    return crop_distal(region)
