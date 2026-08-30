from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from fingerphoto.api.security import require_api_key
from fingerphoto.api.settings import Settings, get_settings


@pytest.fixture()
def client(monkeypatch):
    """Minimal app with the auth dependency — no lifespan, no JVM."""
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_api_key)])
    def protected():
        return {"ok": True}

    monkeypatch.setenv("FP_API_KEYS", "key-one, key-two")
    monkeypatch.delenv("FP_DEV_NO_AUTH", raising=False)
    get_settings.cache_clear()
    yield TestClient(app)
    get_settings.cache_clear()


def test_missing_key_401(client):
    assert client.get("/protected").status_code == 401


def test_wrong_key_401(client):
    assert client.get("/protected", headers={"X-API-Key": "nope"}).status_code == 401


def test_any_configured_key_accepted(client):
    for key in ("key-one", "key-two"):
        assert client.get("/protected", headers={"X-API-Key": key}).status_code == 200


def test_dev_no_auth(client, monkeypatch):
    monkeypatch.setenv("FP_API_KEYS", "")
    monkeypatch.setenv("FP_DEV_NO_AUTH", "1")
    get_settings.cache_clear()
    assert client.get("/protected").status_code == 200


def test_keys_parsing():
    s = Settings(api_keys=" a , b ,, ")
    assert s.keys() == ["a", "b"]
