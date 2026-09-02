"""The gate wired into identity: which crops are refused, and what the caller
is told. Pipeline and matcher are monkeypatched, so no JVM and no photos."""
from __future__ import annotations

import numpy as np
import pytest

from fingerphoto import afis, identity, locate, segment
from fingerphoto.errors import AcquisitionError, ReasonCode

CROP = np.full((400, 300, 3), 128, np.uint8)
MASK = np.full((400, 300), 255, np.uint8)
RIDGE = np.full((200, 150), 255, np.uint8)


def info(score=0.9, valid=True, clamped=False, at_edge=False, period=15.0):
    return {
        "ridge_period_px": period,
        "initial_period_px": period,
        "period_band_px": (10.0, 35.0),
        "period_clamped": clamped,
        "period_at_band_edge": at_edge,
        "scale_converged": True,
        "quality": {"score": score, "valid": valid, "coherence_frac": 0.5,
                    "edge_density": 0.2, "spectral_frac": 0.5},
    }


@pytest.fixture()
def fake_pipeline_stack(monkeypatch):
    """Both crop strategies succeed; the pipeline returns whatever info the
    test asks for; extraction always yields one usable template."""
    state = {"info": info()}

    monkeypatch.setattr(identity, "photo_to_ridge_image",
                        lambda bgr, **kw: (RIDGE, dict(state["info"])))
    monkeypatch.setattr(identity, "crop_distal", lambda bgr: (CROP, MASK))
    monkeypatch.setattr(locate, "crop_pad", lambda bgr: (CROP, MASK, 15.0))
    monkeypatch.setattr(afis, "extract_template", lambda img: b"t")
    monkeypatch.setattr(afis, "minutiae_count", lambda t: 40)
    return state


def test_gate_rejects_low_quality(fake_pipeline_stack):
    fake_pipeline_stack["info"] = info(score=0.10)
    report: dict = {}
    out = identity._templates_for_crop(CROP, MASK, None, (1.0,), None, report=report)
    assert out == [] and report["rejected"] == "ridge_quality"


def test_gate_passes_good_quality(fake_pipeline_stack):
    report: dict = {}
    out = identity._templates_for_crop(CROP, MASK, None, (1.0,), None, report=report)
    assert out == [b"t"] and "rejected" not in report


def test_zero_min_quality_disables_the_score_gate(fake_pipeline_stack):
    fake_pipeline_stack["info"] = info(score=0.01)
    report: dict = {}
    out = identity._templates_for_crop(CROP, MASK, None, (1.0,), None,
                                       min_quality=0.0, report=report)
    assert out == [b"t"] and "rejected" not in report


def test_invalid_quality_is_rejected(fake_pipeline_stack):
    """Too small a region to judge is a refusal, not a pass."""
    fake_pipeline_stack["info"] = info(score=0.0, valid=False)
    report: dict = {}
    assert identity._templates_for_crop(CROP, MASK, None, (1.0,), None, report=report) == []
    assert report["rejected"] == "ridge_quality"


def test_crop_sanity_reason_rules():
    frame = (4000, 3000)
    assert identity.crop_sanity_reason(info(), "distal", (400, 300), frame) is None
    assert identity.crop_sanity_reason(info(clamped=True), "distal", (400, 300), frame) \
        == "period_outside_band"
    assert identity.crop_sanity_reason(info(at_edge=True), "distal", (400, 300), frame) \
        == "period_at_band_edge"
    # a localizer crop covering half the frame is a scene, not a fingertip
    assert identity.crop_sanity_reason(info(), "pad", (3000, 2000), frame) == "crop_too_large"
    # the colour crop legitimately fills the frame on a close-up
    assert identity.crop_sanity_reason(info(), "distal", (3000, 2000), frame) is None


def test_sanity_beats_quality_in_the_reported_reason(fake_pipeline_stack):
    fake_pipeline_stack["info"] = info(score=0.9, clamped=True)
    report: dict = {}
    assert identity._templates_for_crop(CROP, MASK, None, (1.0,), None, report=report) == []
    assert report["rejected"] == "period_outside_band"


def test_photo_template_set_reports_no_ridges(fake_pipeline_stack):
    fake_pipeline_stack["info"] = info(score=0.10)
    stats: dict = {}
    with pytest.raises(AcquisitionError) as e:
        identity.photo_template_set(np.zeros((4000, 3000, 3), np.uint8), stats=stats)
    assert e.value.code is ReasonCode.QUALITY_TOO_LOW
    assert identity.NO_RIDGES_PREFIX in str(e.value)
    assert "distal" in str(e.value) and "0.10" in str(e.value)
    assert stats["crops"]["distal"]["rejected"] == "ridge_quality"
    assert stats["crops"]["pad"]["rejected"] == "ridge_quality"
    assert stats["rejections"] == {"distal": "ridge_quality", "pad": "ridge_quality"}
    assert stats["quality"] == 0.0


def test_minutiae_failure_keeps_its_own_message(fake_pipeline_stack, monkeypatch):
    """A crop with real ridges that yields too few minutiae is a different
    failure from one with no ridges, and must say so."""
    monkeypatch.setattr(afis, "minutiae_count", lambda t: 5)
    stats: dict = {}
    with pytest.raises(AcquisitionError) as e:
        identity.photo_template_set(np.zeros((4000, 3000, 3), np.uint8), stats=stats)
    assert e.value.code is ReasonCode.QUALITY_TOO_LOW
    assert identity.NO_RIDGES_PREFIX not in str(e.value)
    assert "minutiae" in str(e.value) and "ridge quality" in str(e.value)


def test_no_finger_found_when_no_crop_succeeds(monkeypatch):
    monkeypatch.setattr(identity, "crop_distal",
                        lambda bgr: (_ for _ in ()).throw(ValueError("no skin")))
    monkeypatch.setattr(locate, "crop_pad",
                        lambda bgr: (_ for _ in ()).throw(ValueError("no texture")))
    with pytest.raises(AcquisitionError) as e:
        identity.photo_template_set(np.zeros((400, 300, 3), np.uint8))
    assert e.value.code is ReasonCode.NO_FINGER_FOUND


def test_template_quality_is_aligned_with_templates(fake_pipeline_stack):
    stats: dict = {}
    templates = identity.photo_template_set(
        np.zeros((4000, 3000, 3), np.uint8), sweep=(0.87, 1.0, 1.15), stats=stats
    )
    assert len(stats["template_quality"]) == len(templates)
    assert set(stats["template_quality"]) == {0.9}
    assert stats["quality"] == 0.9


def test_one_good_crop_survives_a_rejected_one(fake_pipeline_stack, monkeypatch):
    """The union over strategies is the whole point: a bad pad crop must not
    take a good distal crop down with it."""
    calls = {"n": 0}

    def alternating(bgr, **kw):
        calls["n"] += 1
        return RIDGE, info(score=0.9 if calls["n"] == 1 else 0.05)

    monkeypatch.setattr(identity, "photo_to_ridge_image", alternating)
    stats: dict = {}
    out = identity.photo_template_set(np.zeros((4000, 3000, 3), np.uint8),
                                      sweep=(1.0,), stats=stats)
    assert out == [b"t"]
    assert stats["rejections"] == {"pad": "ridge_quality"}
    assert stats["quality"] == 0.9
