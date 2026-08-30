from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "data" / "samples" / "photo"


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    """Point the store at a temp DB/data dir; no encryption unless a test sets it."""
    monkeypatch.setenv("FP_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("FP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("FP_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("FP_STORE_ORIGINALS", raising=False)
    return tmp_path


@pytest.fixture()
def fake_pipeline(monkeypatch):
    """Replace template extraction + matching so store logic tests need no JVM."""
    from fingerphoto import afis, identity

    def fake_template_set(bgr, **kwargs):
        return [b"fake-template-1", b"fake-template-2"]

    monkeypatch.setattr(identity, "photo_template_set", fake_template_set)
    monkeypatch.setattr(afis, "minutiae_count", lambda t: 20)
    monkeypatch.setattr(identity, "set_score", lambda probe, enrolled: 55.0)
    monkeypatch.setattr(
        identity, "verify", lambda probe, enrolled, thr: (55.0 >= thr, 55.0)
    )
    return fake_template_set


@pytest.fixture()
def bgr_image():
    return np.full((64, 64, 3), 128, dtype=np.uint8)
