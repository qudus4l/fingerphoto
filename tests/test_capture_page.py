"""Capture page bootstraps a credential-free phone UI and browser session."""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from fingerphoto.api.app import create_app
from fingerphoto.api.routes import system
from fingerphoto.api.settings import get_settings


@pytest.fixture()
def capture_client(tmp_store, fake_jvm, monkeypatch):
    monkeypatch.setenv("FP_API_KEYS", "phone-boot-key")
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        yield c
    get_settings.cache_clear()


def test_capture_uses_http_only_session_and_generated_name(capture_client):
    r = capture_client.get("/capture")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "phone-boot-key" not in body
    cookie = r.headers["set-cookie"]
    assert "fp-capture-session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert re.search(r'"name"\s*:\s*"(?:Joshua|Kyle|Tom)-[0-9A-F]{4}"', body)
    assert "/*__FP_CONFIG__*/" not in body  # marker replaced
    assert 'data-mode="capture"' in body
    assert 'data-mode="verify"' in body
    assert 'data-mode="prints">Enrolled' in body
    assert ">Enroll<" not in body
    assert ">Identify<" not in body
    assert 'type="password"' not in body
    assert 'id="name"' not in body
    assert 'id="key"' not in body
    assert "X-API-Key" not in body
    assert "Pause auto-capture" in body
    assert 'id="camera-tools"' in body
    assert 'id="torch"' in body
    assert "Flash unavailable" in body
    assert "Flash: ${torchOn ? 'On' : 'Off'}" in body
    assert "$('torch').style.display = 'none'" not in body
    assert "color-scheme:light" in body
    assert "Secure capture" in body
    assert "RIDGE_PASSES_REQUIRED = 1" in body
    assert "RIDGE_WINDOW_MS = 5000" in body
    assert "RIDGE_MAX_WAIT_MS = 6500" in body
    assert "AUTO_HOLD_MS = 0" in body
    assert "snapFrame" in body
    assert "commitCapture" in body
    assert "Frame locked" in body
    assert "fingerprint-crop" in body
    assert "enrollment-scan-1" in body and "enrollment-scan-2" in body
    assert "Lift your finger" in body
    assert "fd.append('guided'" in body
    assert "fd.append('guided', 'true')" in body
    assert "locked all-green frame (no re-capture)" in body
    assert "`${ridgePasses}/${RIDGE_PASSES_REQUIRED}`" not in body
    assert "Fingerprint photos are not retained" in body
    assert "photoUrl(" not in body
    assert "return await autoSequence" not in body
    assert "Fingerprint successfully added" in body
    assert "Fingerprint belongs to" in body
    assert "Fingerprint not found" in body
    assert "renewCaptureSession" in body
    assert "sessionFetch" in body
    assert "uploadOnce" in body
    # TestClient keeps the HttpOnly cookie, so the page can call protected APIs.
    assert capture_client.get("/v1/version").status_code == 200


def test_capture_session_survives_a_new_app_instance(capture_client):
    """The cookie must work after a reload or on another worker process."""
    capture_client.get("/capture")
    token = capture_client.cookies.get("fp-capture-session")
    assert token
    with TestClient(create_app()) as another_worker:
        another_worker.cookies.set("fp-capture-session", token)
        assert another_worker.get("/v1/version").status_code == 200


def test_capture_session_can_be_renewed_without_a_page_reload(capture_client):
    capture_client.cookies.clear()
    assert capture_client.get("/v1/version").status_code == 401
    refreshed = capture_client.post("/capture/session")
    assert refreshed.status_code == 200
    assert capture_client.get("/v1/version").status_code == 200


def test_tampered_capture_session_is_rejected(capture_client):
    capture_client.get("/capture")
    token = capture_client.cookies.get("fp-capture-session")
    capture_client.cookies.set("fp-capture-session", token + "tampered")
    assert capture_client.get("/v1/version").status_code == 401


def test_capture_gets_a_fresh_test_identity_on_reload(capture_client, monkeypatch):
    suffixes = iter(("A1B2", "C3D4"))
    monkeypatch.setattr(system.secrets, "choice", lambda _: "Joshua")
    monkeypatch.setattr(system.secrets, "token_hex", lambda _: next(suffixes))
    first = re.search(r'"name"\s*:\s*"([^"]+)"', capture_client.get("/capture").text)
    second = re.search(r'"name"\s*:\s*"([^"]+)"', capture_client.get("/capture").text)
    assert first and second
    assert first.group(1) == "Joshua-A1B2"
    assert second.group(1) == "Joshua-C3D4"
