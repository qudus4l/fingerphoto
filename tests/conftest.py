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
        stats = kwargs.get("stats")
        if stats is not None:
            stats["quality"] = 0.9
            stats["template_quality"] = [0.9, 0.9]
            stats["rejections"] = {}
        return [b"fake-template-1", b"fake-template-2"]

    monkeypatch.setattr(identity, "photo_template_set", fake_template_set)
    monkeypatch.setattr(afis, "minutiae_count", lambda t: 20)
    # Stay above the configured operational threshold (currently 85) so these
    # tests exercise store flow rather than threshold calibration.
    fake_score = 100.0
    monkeypatch.setattr(identity, "set_score", lambda probe, enrolled: fake_score)
    monkeypatch.setattr(
        identity, "verify", lambda probe, enrolled, thr: (fake_score >= thr, fake_score)
    )
    return fake_template_set


@pytest.fixture()
def bgr_image():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture()
def fake_jvm(monkeypatch):
    """Let create_app()'s lifespan run without Java. app.py calls both of these
    through the module, so attribute patching reaches it."""
    from fingerphoto import afis

    monkeypatch.setattr(afis, "_ensure_jvm", lambda: None)
    monkeypatch.setattr(afis, "extract_template", lambda *a, **k: b"fake-template")


@pytest.fixture()
def fake_frame_score(monkeypatch):
    """Rank frames by mean brightness, so frame-selection tests need neither
    segmentation nor the real scorer. Patched as a module attribute because
    frames.video_candidates calls it through the module."""
    from fingerphoto import frames

    def score(bgr, frame_shape=None):
        return float(np.asarray(bgr).mean()) / 255.0, "distal", {"stub": True}

    monkeypatch.setattr(frames, "frame_score", score)
    return score


def make_clip(path, n=45, fps=30, size=(320, 240), brightness=lambda i: i * 5):
    """Write a clip whose frame i is a solid grey of the given brightness, so a
    test can assert exactly which frames a selector picked."""
    import cv2

    for fourcc in ("avc1", "mp4v", "MJPG"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size
        )
        if writer.isOpened():
            break
        writer.release()
    else:  # pragma: no cover - no codec at all
        pytest.skip("no VideoWriter codec available")
    for i in range(n):
        v = int(max(0, min(255, brightness(i))))
        writer.write(np.full((size[1], size[0], 3), v, np.uint8))
    writer.release()
    return path
