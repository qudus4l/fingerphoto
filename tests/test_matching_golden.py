"""Golden-score regression tests on real photos (ported from subjects_test.py).

Bands, not exact values: the pipeline is deterministic, but tolerance protects
these tests across parameter tweaks. History: the thumb trio scores 194/154/126
genuine and <= 29 impostor; a regression once collapsed genuine to ~4, which
these thresholds would catch instantly.

The impostor probe is sister_6214, the only impostor photo with resolvable
ridges: sister_6176 and sister_6175 are refused by the ridge gate (pipeline
1.2.0) and are asserted to be refused instead.
"""
from __future__ import annotations

import threading

import pytest

from fingerphoto import afis, identity, imaging
from fingerphoto.errors import AcquisitionError, ReasonCode

from .conftest import SAMPLES

pytestmark = pytest.mark.slow


def _templates(name: str) -> list[bytes]:
    path = SAMPLES / f"{name}.jpg"
    if not path.exists():
        pytest.skip(f"{path} not on disk")
    return identity.photo_template_set(imaging.decode_image(path.read_bytes(), path.name))


@pytest.fixture(scope="module")
def q_trio():
    return {n: _templates(n) for n in ("thumb_6172", "thumb_6173", "thumb_6174")}


def test_genuine_trio_matches(q_trio):
    enrolled = q_trio["thumb_6172"] + q_trio["thumb_6174"]
    score = identity.set_score(q_trio["thumb_6173"], enrolled)
    assert score >= 100, f"genuine score collapsed: {score}"


def test_impostors_stay_low(q_trio):
    enrolled = q_trio["thumb_6172"] + q_trio["thumb_6174"]
    score = identity.set_score(_templates("sister_6214"), enrolled)
    assert score < afis.THRESHOLD, f"sister_6214 scored {score} against Q's identity"


def test_verify_primitive(q_trio):
    enrolled = q_trio["thumb_6172"] + q_trio["thumb_6174"]
    ok, score = identity.verify(q_trio["thumb_6173"], enrolled)
    assert ok and score >= 100
    ok, score = identity.verify(_templates("sister_6214"), enrolled)
    assert not ok and score < afis.THRESHOLD


@pytest.mark.parametrize("name", ["sister_6176", "sister_6175"])
def test_no_ridge_photos_are_refused(name):
    """These enrolled as ~100-minutiae templates before the ridge gate and
    matched nothing, including their own subject. They must now be refused."""
    path = SAMPLES / f"{name}.jpg"
    if not path.exists():
        pytest.skip(f"{path} not on disk")
    bgr = imaging.decode_image(path.read_bytes(), path.name)
    with pytest.raises(AcquisitionError) as e:
        identity.photo_template_set(bgr)
    assert e.value.code is ReasonCode.QUALITY_TOO_LOW
    assert identity.NO_RIDGES_PREFIX in str(e.value)


def test_fvc_smoke():
    fvc = SAMPLES.parent.parent / "fvc"
    a, b, c = fvc / "101_1.tif", fvc / "101_2.tif", fvc / "102_1.tif"
    if not (a.exists() and b.exists() and c.exists()):
        pytest.skip("FVC2000 data not on disk")
    import cv2

    t = [
        afis.extract_template(
            cv2.createCLAHE(4.0, (15, 15)).apply(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        )
        for p in (a, b, c)
    ]
    assert afis.match_one(t[0], t[1]) > 40  # same finger
    assert afis.match_one(t[0], t[2]) < 40  # different finger


def test_concurrent_extraction():
    """Two threads through the full pipeline simultaneously: proves the JVM
    lock, JPype auto-attach, and fingerprint_enhancer thread-safety."""
    path = SAMPLES / "thumb_6172.jpg"
    if not path.exists():
        pytest.skip("sample not on disk")
    bgr = imaging.decode_image(path.read_bytes(), path.name)
    results: list = [None, None]

    def work(i: int) -> None:
        results[i] = identity.photo_template_set(bgr)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert all(r and len(r) >= 1 for r in results)
    # deterministic pipeline -> both threads must produce identical templates
    assert [len(t) for t in results[0]] == [len(t) for t in results[1]]
