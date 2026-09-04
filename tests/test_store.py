from __future__ import annotations

import numpy as np
import pytest
from cryptography.fernet import Fernet

from fingerphoto import crypto, store, version
from fingerphoto.errors import AcquisitionError, ReasonCode, StoreError


def test_consent_required(tmp_store, fake_pipeline):
    with pytest.raises(StoreError) as e:
        store.create_identity("NoConsent", consent=False)
    assert e.value.code is ReasonCode.CONSENT_REQUIRED


def test_duplicate_name_rejected(tmp_store, fake_pipeline):
    store.create_identity("Q", consent=True)
    with pytest.raises(StoreError) as e:
        store.create_identity("Q", consent=True)
    assert e.value.code is ReasonCode.IDENTITY_EXISTS


def test_enroll_and_stamps(tmp_store, fake_pipeline, bgr_image, monkeypatch):
    monkeypatch.setenv("FP_STORE_ORIGINALS", "true")
    ident = store.create_identity("Q", consent=True)
    out = store.enroll_photos(
        ident["uuid"], [("photo.jpg", bgr_image, b"original-bytes")], finger="right_thumb"
    )
    (r,) = out["results"]
    assert r["status"] == "enrolled" and r["templates"] == 2 and r["quality"] == 0.9
    con = store.connect()
    rows = con.execute(
        "SELECT finger, pipeline_version, sourceafis_version, enc, quality FROM templates"
    ).fetchall()
    con.close()
    assert all(
        f == "right_thumb" and pv == version.PIPELINE_VERSION and enc is None and q == 0.9
        for f, pv, sv, enc, q in rows
    )
    # stored originals are uniquified with a timestamp so same-named captures
    # cannot overwrite each other; the sanitized stem is still the prefix
    (stored,) = list((store.photo_store() / ident["uuid"]).glob("photo_*.jpg"))
    assert stored.read_bytes() == b"original-bytes"
    photos = store.list_photos(ident["uuid"])
    assert len(photos) == 1 and photos[0]["available"] and photos[0]["filename"] == stored.name
    data, media = store.original_photo_bytes(ident["uuid"], stored.name)
    assert data == b"original-bytes" and media == "image/jpeg"


def test_invalid_finger_position(tmp_store, fake_pipeline, bgr_image):
    ident = store.create_identity("Q", consent=True)
    with pytest.raises(StoreError) as e:
        store.enroll_photos(ident["uuid"], [("p.jpg", bgr_image, b"x")], finger="thumb2")
    assert e.value.code is ReasonCode.INVALID_FINGER_POSITION


def test_encryption_round_trip(tmp_store, fake_pipeline, bgr_image, monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("FP_ENCRYPTION_KEY", key)
    monkeypatch.setenv("FP_STORE_ORIGINALS", "true")
    ident = store.create_identity("Q", consent=True)
    store.enroll_photos(ident["uuid"], [("photo.jpg", bgr_image, b"original-bytes")])
    con = store.connect()
    blob, enc = con.execute("SELECT template, enc FROM templates LIMIT 1").fetchone()
    con.close()
    assert enc == crypto.SCHEME_FERNET and blob not in (b"fake-template-1", b"fake-template-2")
    assert store.gallery()["Q"][0] in (b"fake-template-1", b"fake-template-2")
    (stored,) = list((store.photo_store() / ident["uuid"]).glob("photo_*.jpg.enc"))
    assert crypto.decrypt(stored.read_bytes(), crypto.SCHEME_FERNET, key) == b"original-bytes"


def test_store_originals_opt_out(tmp_store, fake_pipeline, bgr_image, monkeypatch):
    monkeypatch.setenv("FP_STORE_ORIGINALS", "false")
    ident = store.create_identity("Q", consent=True)
    store.enroll_photos(ident["uuid"], [("photo.jpg", bgr_image, b"original-bytes")])
    assert not (store.photo_store() / ident["uuid"]).exists()


def test_guided_enrollment_requires_two_matching_scans(
    tmp_store, fake_pipeline, bgr_image, monkeypatch
):
    from fingerphoto import identity

    extraction_options = []

    def observed_extract(bgr, **kwargs):
        extraction_options.append(kwargs)
        return fake_pipeline(bgr, **kwargs)

    monkeypatch.setattr(identity, "photo_template_set", observed_extract)
    ident = store.create_identity("Guided", consent=True)
    second = bgr_image.copy()
    second[0, 0] = 129
    out = store.enroll_photos(
        ident["uuid"],
        [("scan-1.jpg", bgr_image, b"one"), ("scan-2.jpg", second, b"two")],
        finger="right_thumb",
        guided=True,
    )
    assert out["profile"] == "guided"
    assert out["self_check"]["passed"] is True
    assert out["self_check"]["score"] == 100.0
    assert [r["status"] for r in out["results"]] == ["enrolled", "enrolled"]
    assert len(extraction_options) == 2
    assert all(o["crop"] is False and o["sweep"] == (1.0,) for o in extraction_options)
    assert not (store.photo_store() / ident["uuid"]).exists()


def test_guided_enrollment_rejects_cross_scan_mismatch(
    tmp_store, fake_pipeline, bgr_image, monkeypatch
):
    from fingerphoto import identity

    monkeypatch.setattr(identity, "set_score", lambda probe, enrolled: 12.0)
    ident = store.create_identity("Mismatch", consent=True)
    second = bgr_image.copy()
    second[0, 0] = 129
    out = store.enroll_photos(
        ident["uuid"],
        [("scan-1.jpg", bgr_image, b"one"), ("scan-2.jpg", second, b"two")],
        guided=True,
    )
    assert out["self_check"]["passed"] is False
    assert all(r["reason_code"] == "ENROLLMENT_MISMATCH" for r in out["results"])
    con = store.connect()
    assert con.execute("SELECT count(*) FROM templates").fetchone()[0] == 0
    con.close()


def test_guided_enrollment_rejects_one_scan(tmp_store, fake_pipeline, bgr_image):
    ident = store.create_identity("One", consent=True)
    out = store.enroll_photos(
        ident["uuid"], [("scan-1.jpg", bgr_image, b"one")], guided=True
    )
    assert out["self_check"] == {"passed": False, "reason": "two_scans_required"}
    assert out["results"][0]["reason_code"] == "ENROLLMENT_MISMATCH"


def test_guided_enrollment_rejects_duplicate_scan(tmp_store, fake_pipeline, bgr_image):
    ident = store.create_identity("Duplicate", consent=True)
    out = store.enroll_photos(
        ident["uuid"],
        [("scan-1.jpg", bgr_image, b"one"), ("scan-2.jpg", bgr_image.copy(), b"two")],
        guided=True,
    )
    assert out["self_check"] == {"passed": False, "reason": "duplicate_scan"}
    assert out["results"][0]["reason_code"] == "ENROLLMENT_MISMATCH"


def test_identify_and_verify(tmp_store, fake_pipeline, bgr_image):
    ident = store.create_identity("Q", consent=True)
    store.enroll_photos(ident["uuid"], [("p.jpg", bgr_image, b"x")])
    result = store.identify_bgr(bgr_image)  # fake set_score = 100 >= default threshold
    assert result["match"]["name"] == "Q" and result["match"]["score"] == 100.0
    result = store.identify_bgr(bgr_image, match_threshold=110.0)
    assert result["match"] is None and result["best_candidate"]["name"] == "Q"
    v = store.verify_bgr(ident["uuid"], bgr_image)
    assert v["verified"] is True and v["score"] == 100.0
    v = store.verify_bgr(ident["uuid"], bgr_image, match_threshold=110.0)
    assert v["verified"] is False


def test_guided_identify_fast_path_does_not_extract_fallback_scales(
    tmp_store, fake_pipeline, bgr_image, monkeypatch
):
    from fingerphoto import identity

    calls = []

    def extract(bgr, **kwargs):
        calls.append(kwargs["sweep"])
        fake_pipeline(bgr, **kwargs)
        return [b"base"]

    monkeypatch.setattr(identity, "photo_template_set", extract)
    ident = store.create_identity("Fast", consent=True)
    store.enroll_photos(ident["uuid"], [("enroll.jpg", bgr_image, b"e")])
    calls.clear()
    monkeypatch.setattr(identity, "set_score", lambda probe, enrolled: 70.0)

    out = store.identify_uploads(
        [("probe.jpg", bgr_image, b"p")], guided=True
    )

    assert out["match"]["name"] == "Fast"
    assert calls == [store.GUIDED_INITIAL_SWEEP]
    assert out["scale_fallback"] == {
        "attempted": False,
        "variants": list(store.GUIDED_FALLBACK_SWEEP),
        "added_templates": 0,
        "score_before": 70.0,
        "score_after": 70.0,
        "processing_s": 0.0,
    }
    assert out["geometry_fallback"]["attempted"] is False


def test_guided_identify_retries_two_scales_once_after_no_match(
    tmp_store, fake_pipeline, bgr_image, monkeypatch
):
    from fingerphoto import identity

    calls = []

    def extract(bgr, **kwargs):
        sweep = kwargs["sweep"]
        calls.append(sweep)
        fake_pipeline(bgr, **kwargs)
        return [b"fallback"] if sweep == store.GUIDED_FALLBACK_SWEEP else [b"base"]

    def score(probe, enrolled):
        return 75.0 if b"fallback" in probe else 20.0

    monkeypatch.setattr(identity, "photo_template_set", extract)
    ident = store.create_identity("Recovered", consent=True)
    store.enroll_photos(ident["uuid"], [("enroll.jpg", bgr_image, b"e")])
    calls.clear()
    monkeypatch.setattr(identity, "set_score", score)

    out = store.identify_uploads(
        [("probe.jpg", bgr_image, b"p")], guided=True
    )

    assert out["match"]["name"] == "Recovered"
    assert calls == [store.GUIDED_INITIAL_SWEEP, store.GUIDED_FALLBACK_SWEEP]
    fallback = out["scale_fallback"]
    assert fallback["attempted"] is True
    assert fallback["variants"] == [0.87, 1.15]
    assert fallback["added_templates"] == 1
    assert fallback["score_before"] == 20.0
    assert fallback["score_after"] == 75.0
    assert fallback["processing_s"] >= 0
    assert out["geometry_fallback"]["attempted"] is False


def test_guided_verify_retries_scales_but_standard_no_match_does_not(
    tmp_store, fake_pipeline, bgr_image, monkeypatch
):
    from fingerphoto import identity

    calls = []

    def extract(bgr, **kwargs):
        sweep = kwargs["sweep"]
        calls.append(sweep)
        fake_pipeline(bgr, **kwargs)
        return [b"fallback"] if sweep == store.GUIDED_FALLBACK_SWEEP else [b"base"]

    def verify(probe, enrolled, threshold):
        score = 72.0 if b"fallback" in probe else 18.0
        return score >= threshold, score

    monkeypatch.setattr(identity, "photo_template_set", extract)
    ident = store.create_identity("Verify", consent=True)
    store.enroll_photos(ident["uuid"], [("enroll.jpg", bgr_image, b"e")])
    calls.clear()
    monkeypatch.setattr(identity, "verify", verify)

    guided = store.verify_uploads(
        ident["uuid"], [("probe.jpg", bgr_image, b"p")], guided=True
    )
    assert guided["verified"] is True
    assert guided["scale_fallback"]["score_before"] == 18.0
    assert guided["scale_fallback"]["score_after"] == 72.0
    assert guided["geometry_fallback"]["attempted"] is False
    assert calls == [store.GUIDED_INITIAL_SWEEP, store.GUIDED_FALLBACK_SWEEP]

    calls.clear()
    standard = store.verify_uploads(
        ident["uuid"], [("probe.jpg", bgr_image, b"p")], guided=False
    )
    assert standard["verified"] is False
    assert "scale_fallback" not in standard
    assert "geometry_fallback" not in standard
    assert calls == [identity.SCALE_SWEEP]


def test_guided_identify_tries_geometric_unwrap_once_after_scale_miss(
    tmp_store, fake_pipeline, bgr_image, monkeypatch
):
    from fingerphoto import geometry, identity

    calls = []

    def unwrap(bgr, half_angle_deg):
        calls.append(f"unwrap-{half_angle_deg:g}")
        return np.full_like(bgr, 201)

    def extract(bgr, **kwargs):
        sweep = kwargs["sweep"]
        kind = "geometry" if int(bgr.mean()) == 201 else (
            "scale" if sweep == store.GUIDED_FALLBACK_SWEEP else "base"
        )
        if kind == "geometry":
            assert kwargs["min_quality"] == 0
        calls.append(kind)
        fake_pipeline(bgr, **kwargs)
        return [kind.encode()]

    def score(probe, enrolled):
        return 76.0 if b"geometry" in probe else 20.0

    monkeypatch.setattr(geometry, "cylindrical_unwrap", unwrap)
    monkeypatch.setattr(identity, "photo_template_set", extract)
    ident = store.create_identity("Geometry", consent=True)
    store.enroll_photos(ident["uuid"], [("enroll.jpg", bgr_image, b"e")])
    calls.clear()
    monkeypatch.setattr(identity, "set_score", score)

    out = store.identify_uploads(
        [("probe.jpg", bgr_image, b"p")], guided=True
    )

    assert out["match"]["name"] == "Geometry"
    assert calls == [
        "base", "scale",
        "unwrap-35", "geometry",
        "unwrap-55", "geometry",
        "unwrap-65", "geometry",
    ]
    assert out["scale_fallback"]["score_after"] == 20.0
    fallback = out["geometry_fallback"]
    assert fallback["attempted"] is True
    assert fallback["method"] == "cylindrical_unwrap"
    assert fallback["half_angles_deg"] == list(geometry.GUIDED_HALF_ANGLES_DEG)
    assert fallback["added_templates"] == 3
    assert fallback["score_before"] == 20.0
    assert fallback["score_after"] == 76.0
    assert fallback["processing_s"] >= 0


def test_guided_verify_tries_geometric_unwrap_once_after_scale_miss(
    tmp_store, fake_pipeline, bgr_image, monkeypatch
):
    from fingerphoto import geometry, identity

    calls = []

    def unwrap(bgr, half_angle_deg):
        calls.append(f"unwrap-{half_angle_deg:g}")
        return np.full_like(bgr, 202)

    def extract(bgr, **kwargs):
        kind = "geometry" if int(bgr.mean()) == 202 else "original"
        calls.append(kind)
        fake_pipeline(bgr, **kwargs)
        return [kind.encode()]

    def verify(probe, enrolled, threshold):
        score = 77.0 if b"geometry" in probe else 19.0
        return score >= threshold, score

    monkeypatch.setattr(geometry, "cylindrical_unwrap", unwrap)
    monkeypatch.setattr(identity, "photo_template_set", extract)
    ident = store.create_identity("Geometry Verify", consent=True)
    store.enroll_photos(ident["uuid"], [("enroll.jpg", bgr_image, b"e")])
    calls.clear()
    monkeypatch.setattr(identity, "verify", verify)

    out = store.verify_uploads(
        ident["uuid"], [("probe.jpg", bgr_image, b"p")], guided=True
    )

    assert out["verified"] is True
    assert calls == [
        "original", "original",
        "unwrap-35", "geometry",
        "unwrap-55", "geometry",
        "unwrap-65", "geometry",
    ]
    assert out["geometry_fallback"]["attempted"] is True
    assert out["geometry_fallback"]["score_before"] == 19.0
    assert out["geometry_fallback"]["score_after"] == 77.0


def test_geometric_fallback_failure_preserves_completed_no_match(
    tmp_store, fake_pipeline, bgr_image, monkeypatch
):
    from fingerphoto import geometry, identity

    unwrap_calls = []

    def failed_unwrap(bgr, half_angle_deg):
        unwrap_calls.append(1)
        raise ValueError("bad geometry")

    monkeypatch.setattr(geometry, "cylindrical_unwrap", failed_unwrap)
    ident = store.create_identity("Safe Geometry", consent=True)
    store.enroll_photos(ident["uuid"], [("enroll.jpg", bgr_image, b"e")])
    monkeypatch.setattr(identity, "set_score", lambda probe, enrolled: 20.0)

    out = store.identify_uploads(
        [("probe.jpg", bgr_image, b"p")], guided=True
    )

    assert out["match"] is None
    assert unwrap_calls == [1]
    assert out["geometry_fallback"]["attempted"] is True
    assert out["geometry_fallback"]["added_templates"] == 0
    assert out["geometry_fallback"]["error"] == "NORMALIZATION_FAILED"
    assert out["geometry_fallback"]["score_before"] == 20.0
    assert out["geometry_fallback"]["score_after"] == 20.0


def test_failed_guided_scale_fallback_returns_one_completed_no_match(
    tmp_store, fake_pipeline, bgr_image, monkeypatch
):
    from fingerphoto import identity

    calls = []

    def extract(bgr, **kwargs):
        sweep = kwargs["sweep"]
        calls.append(sweep)
        if sweep == store.GUIDED_FALLBACK_SWEEP:
            raise AcquisitionError(ReasonCode.QUALITY_TOO_LOW, "fallback unavailable")
        fake_pipeline(bgr, **kwargs)
        return [b"base"]

    monkeypatch.setattr(identity, "photo_template_set", extract)
    ident = store.create_identity("NoMatch", consent=True)
    store.enroll_photos(ident["uuid"], [("enroll.jpg", bgr_image, b"e")])
    calls.clear()
    monkeypatch.setattr(identity, "set_score", lambda probe, enrolled: 20.0)

    out = store.identify_uploads(
        [("probe.jpg", bgr_image, b"p")], guided=True
    )

    assert out["match"] is None
    assert out["best_candidate"]["score"] == 20.0
    assert calls == [
        store.GUIDED_INITIAL_SWEEP,
        store.GUIDED_FALLBACK_SWEEP,
        store.GUIDED_INITIAL_SWEEP,
        store.GUIDED_INITIAL_SWEEP,
        store.GUIDED_INITIAL_SWEEP,
    ]
    assert out["scale_fallback"]["attempted"] is True
    assert out["scale_fallback"]["added_templates"] == 0
    assert out["scale_fallback"]["error"] == "QUALITY_TOO_LOW"
    assert out["scale_fallback"]["score_before"] == 20.0
    assert out["scale_fallback"]["score_after"] == 20.0
    assert out["geometry_fallback"]["attempted"] is True
    assert out["geometry_fallback"]["score_before"] == 20.0
    assert out["geometry_fallback"]["score_after"] == 20.0


def test_identify_empty_gallery(tmp_store, fake_pipeline, bgr_image):
    with pytest.raises(StoreError) as e:
        store.identify_bgr(bgr_image)
    assert e.value.code is ReasonCode.GALLERY_EMPTY


def test_hard_delete(tmp_store, fake_pipeline, bgr_image, monkeypatch):
    monkeypatch.setenv("FP_STORE_ORIGINALS", "true")
    ident = store.create_identity("Q", consent=True)
    store.enroll_photos(ident["uuid"], [("p.jpg", bgr_image, b"x")])
    assert (store.photo_store() / ident["uuid"]).exists()
    store.delete_identity(ident["uuid"])
    assert not (store.photo_store() / ident["uuid"]).exists()
    con = store.connect()
    assert con.execute("SELECT COUNT(*) FROM templates").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM identities").fetchone()[0] == 0
    events = [r[0] for r in con.execute("SELECT event FROM audit_log ORDER BY id")]
    con.close()
    assert events == ["identity.create", "photo.enroll", "identity.delete"]
    with pytest.raises(StoreError) as e:
        store.get_identity(ident["uuid"])
    assert e.value.code is ReasonCode.IDENTITY_NOT_FOUND


def test_path_traversal_filename_sanitized(
    tmp_store, fake_pipeline, bgr_image, monkeypatch
):
    monkeypatch.setenv("FP_STORE_ORIGINALS", "true")
    ident = store.create_identity("Q", consent=True)
    out = store.enroll_photos(ident["uuid"], [("../../evil.jpg", bgr_image, b"x")])
    assert out["results"][0]["filename"] == "evil.jpg"
    assert list((store.photo_store() / ident["uuid"]).glob("evil_*.jpg"))


# --- video / multi-frame enrollment ----------------------------------------

def _clip(tmp_path, **kw):
    from .conftest import make_clip

    return make_clip(tmp_path / "clip.mp4", n=90, brightness=lambda i: i * 3, **kw).read_bytes()


def test_enroll_video_selects_frames(
    tmp_store, fake_pipeline, fake_frame_score, monkeypatch
):
    monkeypatch.setenv("FP_VIDEO_FRAMES_ENROLL", "2")
    monkeypatch.setenv("FP_STORE_ORIGINALS", "true")
    ident = store.create_identity("V", consent=True)
    out = store.enroll_photos(
        ident["uuid"], [("clip.mp4", None, _clip(tmp_store))], finger="right_thumb"
    )
    assert [r["status"] for r in out["results"]] == ["enrolled", "enrolled"]
    assert out["selection"]["candidates"] == 9 and out["selection"]["selected"] == 2
    # provenance survives to the result
    assert all(r["frame"] is not None and r["candidate_score"] is not None
               for r in out["results"])
    assert out["results"][0]["source"] == "clip.mp4"
    # the frames themselves are archived, not the clip
    stored = sorted((tmp_store / "data" / "enrolled" / ident["uuid"]).glob("clip_f*.jpg"))
    assert len(stored) == 2
    con = store.connect()
    assert con.execute("SELECT count(*) FROM templates").fetchone()[0] == 4
    con.close()


def test_enroll_best_of_pools_uploads(tmp_store, fake_pipeline, fake_frame_score, bgr_image):
    import numpy as np

    dark = np.full((64, 64, 3), 10, np.uint8)
    bright = np.full((64, 64, 3), 250, np.uint8)
    ident = store.create_identity("B", consent=True)
    out = store.enroll_photos(
        ident["uuid"],
        [("a.jpg", dark, b"a"), ("b.jpg", bright, b"b")],
        best_of=1,
    )
    assert len(out["results"]) == 1 and out["results"][0]["source"] == "b.jpg"

    ident2 = store.create_identity("B2", consent=True)
    out2 = store.enroll_photos(
        ident2["uuid"], [("a.jpg", dark, b"a"), ("b.jpg", bright, b"b")]
    )
    assert len(out2["results"]) == 2
    assert all(r["candidate_score"] is None for r in out2["results"])  # unchanged photo path


def test_unreadable_video_is_a_rejected_result(tmp_store, fake_pipeline, fake_frame_score):
    ident = store.create_identity("U", consent=True)
    out = store.enroll_photos(ident["uuid"], [("x.mp4", None, b"junk bytes")])
    (r,) = out["results"]
    assert r["status"] == "rejected" and r["reason_code"] == "VIDEO_UNREADABLE"
    con = store.connect()
    assert con.execute("SELECT count(*) FROM templates").fetchone()[0] == 0
    con.close()


def test_identify_video_unions_frames(
    tmp_store, fake_pipeline, fake_frame_score, bgr_image, monkeypatch
):
    monkeypatch.setenv("FP_VIDEO_FRAMES_IDENTIFY", "2")
    ident = store.create_identity("Q", consent=True)
    store.enroll_photos(ident["uuid"], [("p.jpg", bgr_image, b"x")])
    out = store.identify_uploads([("clip.mp4", None, _clip(tmp_store))])
    assert out["match"]["name"] == "Q"          # fake set_score = 55 >= 40
    assert out["frames"]["used"] == 2 and out["frames"]["candidates"] == 2
    assert out["selection"]["selected"] == 2
