"""Video and multi-frame capture through the HTTP layer (no JVM, no real pipeline)."""
from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fingerphoto import frames

from .conftest import make_clip

KEY = {"X-API-Key": "video-test-key"}


def _make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("FP_DB_PATH", str(tmp_path / "v.db"))
    monkeypatch.setenv("FP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FP_API_KEYS", "video-test-key")
    monkeypatch.setenv("FP_VIDEO_FRAMES_ENROLL", "2")
    monkeypatch.setenv("FP_VIDEO_FRAMES_IDENTIFY", "2")
    monkeypatch.delenv("FP_ENCRYPTION_KEY", raising=False)
    from fingerphoto.api.app import create_app
    from fingerphoto.api.settings import get_settings

    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


@pytest.fixture()
def client(tmp_path, monkeypatch, fake_pipeline, fake_frame_score, fake_jvm):
    """Stubbed scorer: frame selection is deterministic (brightness ranks)."""
    yield from _make_client(tmp_path, monkeypatch)


@pytest.fixture()
def real_score_client(tmp_path, monkeypatch, fake_pipeline, fake_jvm):
    """Real frames.frame_score, still no JVM — for the /v1/quality contract."""
    yield from _make_client(tmp_path, monkeypatch)


@pytest.fixture()
def clip_bytes(tmp_path):
    return make_clip(tmp_path / "clip.mp4", n=90, brightness=lambda i: i * 3).read_bytes()


def _jpeg(value=200, size=(64, 64)):
    ok, buf = cv2.imencode(".jpg", np.full((size[1], size[0], 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def _poll(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = client.get(f"/v1/jobs/{job_id}", headers=KEY).json()
        if out["status"] in ("succeeded", "failed"):
            return out
        time.sleep(0.05)
    raise TimeoutError(job_id)


def _identity(client, name="V"):
    r = client.post("/v1/identities", headers=KEY, json={"name": name, "consent": True})
    assert r.status_code == 201, r.text
    return r.json()["uuid"]


def test_enroll_a_video(client, clip_bytes):
    uuid = _identity(client)
    r = client.post(
        f"/v1/identities/{uuid}/photos",
        headers=KEY,
        files=[("files", ("clip.mp4", clip_bytes, "video/mp4"))],
    )
    assert r.status_code == 202, r.text
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "succeeded", job
    res = job["result"]
    assert res["selection"]["candidates"] == 9 and res["selection"]["selected"] == 2
    assert [x["status"] for x in res["results"]] == ["enrolled", "enrolled"]
    assert all(x["frame"] is not None for x in res["results"])


def test_enroll_burst_with_best_of(client):
    uuid = _identity(client, "Burst")
    files = [("files", (f"burst_{i}.jpg", _jpeg(40 * (i + 1)), "image/jpeg"))
             for i in range(5)]
    r = client.post(f"/v1/identities/{uuid}/photos", headers=KEY,
                    files=files, data={"best_of": "1"})
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "succeeded", job
    res = job["result"]
    assert len(res["results"]) == 1
    assert res["results"][0]["source"] == "burst_4.jpg"  # brightest = highest stub score
    assert res["selection"]["candidates"] == 5


def test_guided_enrollment_uses_two_scans_and_self_checks(client):
    uuid = _identity(client, "Guided")
    files = [
        ("files", ("fingerprint-crop-1.jpg", _jpeg(180), "image/jpeg")),
        ("files", ("fingerprint-crop-2.jpg", _jpeg(185), "image/jpeg")),
    ]
    r = client.post(
        f"/v1/identities/{uuid}/photos",
        headers=KEY,
        files=files,
        data={"guided": "true", "finger": "right_thumb"},
    )
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "succeeded", job
    assert job["result"]["profile"] == "guided"
    assert job["result"]["self_check"]["passed"] is True
    assert len(job["result"]["results"]) == 2


def test_browser_session_completes_enroll_and_identify_without_api_key_header(client):
    """Exact browser auth path: page cookie -> create -> enroll -> poll -> identify."""
    page = client.get("/capture")
    assert page.status_code == 200 and client.cookies.get("fp-capture-session")

    created = client.post("/v1/identities", json={"name": "Browser E2E", "consent": True})
    assert created.status_code == 201, created.text
    uuid = created.json()["uuid"]
    enrolled = client.post(
        f"/v1/identities/{uuid}/photos",
        files=[
            ("files", ("enrollment-scan-1.jpg", _jpeg(180), "image/jpeg")),
            ("files", ("enrollment-scan-2.jpg", _jpeg(185), "image/jpeg")),
        ],
        data={"guided": "true", "finger": "right_thumb"},
    )
    assert enrolled.status_code == 202, enrolled.text
    job = _poll_without_header(client, enrolled.json()["job_id"])
    assert job["status"] == "succeeded" and job["result"]["self_check"]["passed"]

    identified = client.post(
        "/v1/identify",
        files=[("files", ("fingerprint-probe.jpg", _jpeg(190), "image/jpeg"))],
        data={"guided": "true", "best_of": "1"},
    )
    assert identified.status_code == 202, identified.text
    result = _poll_without_header(client, identified.json()["job_id"])
    assert result["status"] == "succeeded"
    assert result["result"]["match"]["name"] == "Browser E2E"


def _poll_without_header(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = client.get(f"/v1/jobs/{job_id}").json()
        if out["status"] in ("succeeded", "failed"):
            return out
        time.sleep(0.05)
    raise TimeoutError(job_id)


def test_identify_with_files_and_best_of(client):
    uuid = _identity(client, "Known")
    client_files = [("files", ("p.jpg", _jpeg(200), "image/jpeg"))]
    r = client.post(f"/v1/identities/{uuid}/photos", headers=KEY, files=client_files)
    assert _poll(client, r.json()["job_id"])["status"] == "succeeded"

    r = client.post(
        "/v1/identify", headers=KEY,
        files=[("files", ("a.jpg", _jpeg(30), "image/jpeg")),
               ("files", ("b.jpg", _jpeg(240), "image/jpeg"))],
        data={"best_of": "1"},
    )
    assert r.status_code == 202, r.text
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "succeeded", job
    assert job["result"]["frames"]["used"] == 1
    assert job["result"]["match"]["name"] == "Known"


def test_identify_still_accepts_a_single_file(client):
    uuid = _identity(client, "Solo")
    r = client.post(f"/v1/identities/{uuid}/photos", headers=KEY,
                    files=[("files", ("p.jpg", _jpeg(200), "image/jpeg"))])
    assert _poll(client, r.json()["job_id"])["status"] == "succeeded"
    r = client.post("/v1/identify", headers=KEY,
                    files={"file": ("probe.jpg", _jpeg(200), "image/jpeg")})
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "succeeded" and job["result"]["frames"]["used"] == 1


def test_identify_without_any_file_is_422(client):
    _identity(client, "Anyone")
    r = client.post("/v1/identify", headers=KEY, data={"best_of": "1"})
    assert r.status_code == 422, r.text


def test_unreadable_video_upload_is_415(client):
    uuid = _identity(client, "Bad")
    r = client.post(f"/v1/identities/{uuid}/photos", headers=KEY,
                    files=[("files", ("x.mp4", b"not a container", "video/mp4"))])
    assert r.status_code == 415, r.text
    assert r.json()["reason_code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_quality_endpoint(client):
    r = client.post("/v1/quality", headers=KEY,
                    files={"file": ("f.jpg", _jpeg(200), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) >= {"score", "strategy", "pass", "min_ridge_quality", "detail"}
    assert body["min_ridge_quality"] == pytest.approx(0.40)
    # the stub scores by brightness (200/255), so this also proves the route
    # reaches frames.frame_score through the module, as the fixture patches it
    assert body["score"] == pytest.approx(200 / 255, abs=1e-3)
    assert body["pass"] is True


def test_quality_endpoint_uses_guided_scorer(client, monkeypatch):
    monkeypatch.setattr(
        frames,
        "guided_frame_score",
        lambda bgr: (0.65, "full", {"full": {"quality": {"valid": True}}}),
    )
    r = client.post(
        "/v1/quality",
        headers=KEY,
        data={"guided": "true"},
        files={"file": ("fingerprint-crop.jpg", _jpeg(200), "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["score"] == pytest.approx(0.65)
    assert r.json()["strategy"] == "full"


def test_quality_endpoint_scores_a_real_frame(real_score_client):
    """A flat grey frame has skin-free, ridge-free texture: score 0, no strategy."""
    r = real_score_client.post("/v1/quality", headers=KEY,
                               files={"file": ("f.jpg", _jpeg(200), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["score"] == 0.0 and body["pass"] is False and body["strategy"] is None


def test_quality_endpoint_refuses_video(client, clip_bytes):
    r = client.post("/v1/quality", headers=KEY,
                    files={"file": ("clip.mp4", clip_bytes, "video/mp4")})
    assert r.status_code == 415, r.text


def test_quality_endpoint_needs_a_key(client):
    r = client.post("/v1/quality", files={"file": ("f.jpg", _jpeg(200), "image/jpeg")})
    assert r.status_code == 401


def test_upload_limit_is_64mb(client):
    uuid = _identity(client, "Big")
    r = client.post(f"/v1/identities/{uuid}/photos", headers=KEY,
                    files=[("files", ("big.mp4", b"\x00" * (65 * 1024 * 1024), "video/mp4"))])
    assert r.status_code == 413, r.status_code
