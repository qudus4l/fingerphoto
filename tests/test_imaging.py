from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from fingerphoto import imaging
from fingerphoto.errors import AcquisitionError, ReasonCode

from .conftest import ROOT


def _jpeg_bytes(width: int, height: int, orientation: int | None = None) -> bytes:
    img = Image.fromarray(np.full((height, width, 3), 90, dtype=np.uint8))
    buf = io.BytesIO()
    if orientation:
        exif = Image.Exif()
        exif[0x0112] = orientation
        img.save(buf, "JPEG", exif=exif)
    else:
        img.save(buf, "JPEG")
    return buf.getvalue()


def test_decodes_jpeg():
    bgr = imaging.decode_image(_jpeg_bytes(40, 30))
    assert bgr.shape == (30, 40, 3) and bgr.dtype == np.uint8


def test_applies_exif_orientation():
    # orientation 6 = rotate 90° CW on display; decoded dims must swap
    bgr = imaging.decode_image(_jpeg_bytes(40, 30, orientation=6))
    assert bgr.shape == (40, 30, 3)


def test_garbage_rejected_with_code():
    with pytest.raises(AcquisitionError) as e:
        imaging.decode_image(b"definitely not an image")
    assert e.value.code is ReasonCode.UNSUPPORTED_MEDIA_TYPE


def test_decompression_bomb_rejected(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(AcquisitionError) as e:
        imaging.decode_image(_jpeg_bytes(40, 30))
    assert e.value.code is ReasonCode.IMAGE_TOO_LARGE


def test_decodes_heic_if_sample_present():
    heics = sorted(ROOT.glob("data/*.heic")) + sorted(ROOT.glob("data/*.HEIC"))
    if not heics:
        pytest.skip("no HEIC sample on disk")
    bgr = imaging.decode_image(heics[0].read_bytes(), heics[0].name)
    assert bgr.ndim == 3 and bgr.shape[2] == 3
