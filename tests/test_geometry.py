from __future__ import annotations

import numpy as np
import pytest

from fingerphoto.geometry import cylindrical_unwrap


def test_cylindrical_unwrap_preserves_shape_dtype_and_edges():
    image = np.tile(np.arange(101, dtype=np.uint8), (40, 1))

    out = cylindrical_unwrap(image, half_angle_deg=55)

    assert out.shape == image.shape
    assert out.dtype == image.dtype
    assert out[20, 0] == image[20, 0]
    assert out[20, -1] == image[20, -1]


def test_cylindrical_unwrap_expands_the_sides():
    # Pixel values encode their source x coordinate.  At one-quarter of the
    # output width the cylindrical remap samples closer to the edge than a
    # linear/identity map, which is the local side expansion we need.
    image = np.tile(np.arange(101, dtype=np.float32), (20, 1))

    out = cylindrical_unwrap(image, half_angle_deg=55)

    assert out[10, 25] < image[10, 25]
    assert out[10, 75] > image[10, 75]
    assert out[10, 50] == pytest.approx(image[10, 50], abs=0.6)


def test_zero_angle_is_an_independent_identity_copy():
    image = np.arange(60, dtype=np.uint8).reshape(5, 12)

    out = cylindrical_unwrap(image, half_angle_deg=0)

    assert np.array_equal(out, image)
    assert out is not image


@pytest.mark.parametrize("angle", [-1, 90, 120])
def test_invalid_angle_is_refused(angle):
    with pytest.raises(ValueError, match="half_angle_deg"):
        cylindrical_unwrap(np.zeros((20, 20), np.uint8), angle)
