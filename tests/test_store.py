from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from fingerphoto import crypto, store, version
from fingerphoto.errors import ReasonCode, StoreError


def test_consent_required(tmp_store, fake_pipeline):
    with pytest.raises(StoreError) as e:
        store.create_identity("NoConsent", consent=False)
    assert e.value.code is ReasonCode.CONSENT_REQUIRED


def test_duplicate_name_rejected(tmp_store, fake_pipeline):
    store.create_identity("Q", consent=True)
    with pytest.raises(StoreError) as e:
        store.create_identity("Q", consent=True)
    assert e.value.code is ReasonCode.IDENTITY_EXISTS


def test_enroll_and_stamps(tmp_store, fake_pipeline, bgr_image):
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


def test_invalid_finger_position(tmp_store, fake_pipeline, bgr_image):
    ident = store.create_identity("Q", consent=True)
    with pytest.raises(StoreError) as e:
        store.enroll_photos(ident["uuid"], [("p.jpg", bgr_image, b"x")], finger="thumb2")
    assert e.value.code is ReasonCode.INVALID_FINGER_POSITION


def test_encryption_round_trip(tmp_store, fake_pipeline, bgr_image, monkeypatch):
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("FP_ENCRYPTION_KEY", key)
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


def test_identify_and_verify(tmp_store, fake_pipeline, bgr_image):
    ident = store.create_identity("Q", consent=True)
    store.enroll_photos(ident["uuid"], [("p.jpg", bgr_image, b"x")])
    result = store.identify_bgr(bgr_image)  # fake set_score = 55 >= 40
    assert result["match"]["name"] == "Q" and result["match"]["score"] == 55.0
    result = store.identify_bgr(bgr_image, match_threshold=60.0)
    assert result["match"] is None and result["best_candidate"]["name"] == "Q"
    v = store.verify_bgr(ident["uuid"], bgr_image)
    assert v["verified"] is True and v["score"] == 55.0
    v = store.verify_bgr(ident["uuid"], bgr_image, match_threshold=60.0)
    assert v["verified"] is False


def test_identify_empty_gallery(tmp_store, fake_pipeline, bgr_image):
    with pytest.raises(StoreError) as e:
        store.identify_bgr(bgr_image)
    assert e.value.code is ReasonCode.GALLERY_EMPTY


def test_hard_delete(tmp_store, fake_pipeline, bgr_image):
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


def test_path_traversal_filename_sanitized(tmp_store, fake_pipeline, bgr_image):
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
