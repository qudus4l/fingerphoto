"""Decode uploaded image bytes into BGR ndarrays.

PIL is the decoder rather than cv2.imdecode: cv2.imdecode ignores EXIF
orientation (cv2.imread applies it, imdecode does not), and portrait iPhone
captures rely on that rotation for the finger-up assumption in segmentation.
pillow-heif extends PIL to HEIC, the iPhone default format.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import AcquisitionError, ReasonCode

# Decompression-bomb guard; an iPhone 48 MP capture is 48M pixels.
Image.MAX_IMAGE_PIXELS = 80_000_000

_heif_registered = False


def _register_heif() -> None:
    global _heif_registered
    if not _heif_registered:
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except ImportError:
            pass  # HEIC uploads will fail with UNSUPPORTED_MEDIA_TYPE
        _heif_registered = True


def decode_image(data: bytes, filename: str = "") -> np.ndarray:
    """Bytes -> BGR uint8 ndarray with EXIF orientation applied."""
    _register_heif()
    label = filename or "image"
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        rgb = img.convert("RGB")
    except Image.DecompressionBombError:
        raise AcquisitionError(
            ReasonCode.IMAGE_TOO_LARGE,
            f"{label} exceeds the {Image.MAX_IMAGE_PIXELS} pixel limit",
        )
    except UnidentifiedImageError:
        raise AcquisitionError(
            ReasonCode.UNSUPPORTED_MEDIA_TYPE, f"{label} is not a decodable image format"
        )
    except OSError as e:
        raise AcquisitionError(ReasonCode.UNREADABLE_IMAGE, f"{label} could not be decoded: {e}")
    return np.asarray(rgb)[:, :, ::-1].copy()
