"""Full API lifecycle through TestClient: create -> enroll -> poll ->
identify -> verify -> delete. Runs the real pipeline + JVM in-process."""
from __future__ import annotations

import time

import pytest

from .conftest import SAMPLES

pytestmark = pytest.mark.slow

KEY = {"X-API-Key": "e2e-test-key"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    import os

    env = {
        "FP_DB_PATH": str(tmp / "e2e.db"),
        "FP_DATA_DIR": str(tmp / "data"),
        "FP_API_KEYS": "e2e-test-key",
    }
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    from fastapi.testclient import TestClient

    from fingerphoto.api.app import create_app
    from fingerphoto.api.settings import get_settings

    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()
    for k, v in old.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def _poll(client, job_id: str, timeout=300) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        out = client.get(f"/v1/jobs/{job_id}", headers=KEY).json()
        if out["status"] in ("succeeded", "failed"):
            return out
        time.sleep(2)
    raise TimeoutError(job_id)


def _upload(name: str):
    path = SAMPLES / f"{name}.jpg"
    if not path.exists():
        pytest.skip(f"{path} not on disk")
    return (f"{name}.jpg", path.read_bytes(), "image/jpeg")


def test_full_lifecycle(client):
    assert client.get("/readyz").status_code == 200

    # consent literal true is mandatory
    r = client.post("/v1/identities", json={"name": "X", "consent": False}, headers=KEY)
    assert r.status_code == 422 and r.json()["reason_code"] == "CONSENT_REQUIRED"

    r = client.post("/v1/identities", json={"name": "Q-e2e", "consent": True}, headers=KEY)
    assert r.status_code == 201
    uid = r.json()["uuid"]

    r = client.post(
        f"/v1/identities/{uid}/photos",
        files=[("files", _upload("thumb_6172")), ("files", _upload("thumb_6174"))],
        data={"finger": "right_thumb"},
        headers=KEY,
    )
    assert r.status_code == 202 and r.headers["location"].startswith("/v1/jobs/")
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "succeeded"
    assert all(p["status"] == "enrolled" for p in job["result"]["results"])
    assert all(p["quality"] >= 0.4 for p in job["result"]["results"])

    r = client.post("/v1/identify", files={"file": _upload("thumb_6173")}, headers=KEY)
    job = _poll(client, r.json()["job_id"])
    assert job["result"]["match"]["name"] == "Q-e2e"
    assert job["result"]["match"]["score"] >= 100

    r = client.post(
        f"/v1/identities/{uid}/verify", files={"file": _upload("sister_6214")}, headers=KEY
    )
    job = _poll(client, r.json()["job_id"])
    assert job["result"]["verified"] is False and job["result"]["score"] < 40

    # a photo with no resolvable ridges is refused, not silently scored low
    r = client.post(
        f"/v1/identities/{uid}/verify", files={"file": _upload("sister_6176")}, headers=KEY
    )
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "failed"
    assert job["error"]["reason_code"] == "QUALITY_TOO_LOW"
    assert "no resolvable ridge" in job["error"]["detail"]

    r = client.delete(f"/v1/identities/{uid}", headers=KEY)
    assert r.status_code == 200
    assert client.get(f"/v1/identities/{uid}", headers=KEY).status_code == 404


def _clip_from(paths, tmp, fps=30, per=30, size=(1080, 1920)):
    """A clip whose segments are real photos: the first is unusable, the rest
    carry ridges. Selection must land in the usable segment."""
    import cv2
    import numpy as np

    out = tmp / "clip.mp4"
    writer = None
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            break
        writer.release()
    if writer is None or not writer.isOpened():
        pytest.skip("no VideoWriter codec available")
    for path in paths:
        img = cv2.imread(str(path))
        assert img is not None, path
        frame = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        for _ in range(per):
            writer.write(frame)
    writer.release()
    return out.read_bytes()


def test_video_enroll_and_identify(client, tmp_path_factory):
    """A clip that starts with a no-ridge frame and ends with real prints:
    the selector must skip the first segment and enrol only usable frames."""
    for name in ("sister_6176", "thumb_6172", "thumb_6174"):
        if not (SAMPLES / f"{name}.jpg").exists():
            pytest.skip("corpus photos not on disk")
    tmp = tmp_path_factory.mktemp("clip")
    clip = _clip_from(
        [SAMPLES / "sister_6176.jpg", SAMPLES / "thumb_6172.jpg", SAMPLES / "thumb_6174.jpg"],
        tmp,
    )

    r = client.post("/v1/identities", headers=KEY, json={"name": "VidQ", "consent": True})
    assert r.status_code == 201, r.text
    uuid = r.json()["uuid"]

    r = client.post(f"/v1/identities/{uuid}/photos", headers=KEY,
                    files=[("files", ("clip.mp4", clip, "video/mp4"))],
                    data={"finger": "right_thumb"})
    assert r.status_code == 202, r.text
    job = _poll(client, r.json()["job_id"], timeout=600)
    assert job["status"] == "succeeded", job
    res = job["result"]
    enrolled = [x for x in res["results"] if x["status"] == "enrolled"]
    assert enrolled, res
    # every enrolled frame comes from the ridge-bearing part of the clip and
    # clears the gate comfortably
    assert all(x["frame"] >= 30 for x in enrolled), enrolled
    assert all(x["quality"] >= 0.6 for x in enrolled), enrolled
    assert res["selection"]["candidates"] > len(enrolled)

    r = client.post("/v1/identify", headers=KEY,
                    files=[("files", ("clip.mp4", clip, "video/mp4"))])
    job = _poll(client, r.json()["job_id"], timeout=600)
    assert job["status"] == "succeeded", job
    out = job["result"]
    assert out["match"] is not None and out["match"]["name"] == "VidQ", out
    assert out["frames"]["used"] >= 1


def test_photo_path_reports_no_candidate_score(client):
    """The single-photo route must stay exactly what it was: unscored."""
    r = client.post("/v1/identities", headers=KEY, json={"name": "PlainQ", "consent": True})
    uuid = r.json()["uuid"]
    r = client.post(f"/v1/identities/{uuid}/photos", headers=KEY,
                    files=[("files", _upload("thumb_6172"))])
    job = _poll(client, r.json()["job_id"])
    assert job["status"] == "succeeded", job
    (result,) = job["result"]["results"]
    assert result["candidate_score"] is None and result["frame"] is None
    assert result["source"] == "thumb_6172.jpg"
