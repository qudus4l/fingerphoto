"""Conservative geometric normalization for guided finger photographs.

A frontal camera sees the finger pad as the projection of a curved surface.
Ridges near the sides are therefore compressed relative to ridges in the
centre.  ``cylindrical_unwrap`` reverses that projection under a deliberately
simple cylindrical model.  It only resamples existing pixels; it does not
synthesize ridge detail.

This is an experimental probe-side transform.  The unwarped template is added
to, rather than substituted for, the original probe so an imperfect curvature
assumption cannot lower the best set-match score.
"""
from __future__ import annotations

import cv2
import numpy as np


# A fingertip photographed from the front exposes roughly this much of either
# side of its curved pad.  55 degrees produces a modest correction: about 17%
# centre compression in the output, with the recovered space assigned to the
# foreshortened sides.  Treat this as an A/B parameter, not anatomy ground
# truth; tune it only against genuine and impostor captures.
GUIDED_HALF_ANGLE_DEG = 55.0
# Curvature hypotheses for the slow no-match fallback.  A small corpus A/B
# found that different placements benefited at the mild and strong ends; no
# single angle dominated.  Keep the set deliberately small because each one
# requires an authoritative template extraction.
GUIDED_HALF_ANGLES_DEG = (35.0, 55.0, 65.0)


def cylindrical_unwrap(
    image: np.ndarray,
    half_angle_deg: float = GUIDED_HALF_ANGLE_DEG,
) -> np.ndarray:
    """Unwrap a centred, upright finger crop using an orthographic cylinder.

    The input and output have identical shape and dtype.  For a surface angle
    ``theta``, an orthographic photograph records horizontal position as
    ``sin(theta)``.  The inverse-normalization samples the photograph at
    ``sin(theta)`` while output columns advance linearly in ``theta``.  This
    expands the compressed sides of the pad without changing the API crop or
    its global dimensions.

    ``half_angle_deg=0`` is an exact copy and is useful as the explicit
    no-normalization baseline.  Values must stay below 90 degrees because the
    cylindrical projection becomes singular there.
    """
    if image.ndim not in (2, 3):
        raise ValueError("finger crop must be a 2-D or 3-D image")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("finger crop is too small to normalize")
    if not 0.0 <= half_angle_deg < 90.0:
        raise ValueError("half_angle_deg must be in [0, 90)")
    if half_angle_deg == 0.0:
        return image.copy()

    height, width = image.shape[:2]
    half_angle = np.deg2rad(float(half_angle_deg))
    output_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    source_x = np.sin(half_angle * output_x) / np.sin(half_angle)
    # NumPy promotes the expression to float64 because ``half_angle`` is a
    # Python float; remap/convertMaps require float32 maps exactly.
    source_x = ((source_x + 1.0) * (width - 1) / 2.0).astype(np.float32)

    map_x = np.broadcast_to(source_x, (height, width)).copy()
    map_y = np.broadcast_to(
        np.arange(height, dtype=np.float32)[:, None], (height, width)
    ).copy()
    # OpenCV 4 accepts the two float maps directly.  OpenCV 5's optimized
    # genericWarp path requires the equivalent fixed-point representation.
    # Converting explicitly keeps the code portable across both versions and
    # is cheaper when remap performs cubic interpolation.
    fixed_xy, fractions = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
    return cv2.remap(
        image,
        fixed_xy,
        fractions,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
