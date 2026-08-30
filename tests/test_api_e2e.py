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

    r = client.post("/v1/identify", files={"file": _upload("thumb_6173")}, headers=KEY)
    job = _poll(client, r.json()["job_id"])
    assert job["result"]["match"]["name"] == "Q-e2e"
    assert job["result"]["match"]["score"] >= 100

    r = client.post(
        f"/v1/identities/{uid}/verify", files={"file": _upload("sister_6176")}, headers=KEY
    )
    job = _poll(client, r.json()["job_id"])
    assert job["result"]["verified"] is False and job["result"]["score"] < 40

    r = client.delete(f"/v1/identities/{uid}", headers=KEY)
    assert r.status_code == 200
    assert client.get(f"/v1/identities/{uid}", headers=KEY).status_code == 404
