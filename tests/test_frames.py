"""Candidate extraction and best-frame selection (no JVM, no real scorer)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from fingerphoto import frames
from fingerphoto.errors import AcquisitionError, ReasonCode

from .conftest import SAMPLES, make_clip


@pytest.fixture()
def clip(tmp_path, monkeypatch):
    monkeypatch.setenv("FP_DATA_DIR", str(tmp_path / "data"))
    # 90 frames @ 30 fps = 3 s, brightness rising 0..267 so the stub scorer
    # ranks later frames higher.
    path = make_clip(tmp_path / "clip.mp4", n=90, brightness=lambda i: i * 3)
    return path.read_bytes()


def test_video_candidates_samples_and_names(clip, fake_frame_score):
    cands = frames.video_candidates(clip, "clip.mp4", sample_fps=3.0)
    assert [c.index for c in cands] == [0, 10, 20, 30, 40, 50, 60, 70, 80]
    assert cands[0].name == "clip_f0000.jpg" and cands[1].name == "clip_f0010.jpg"
    assert cands[0].source == "clip.mp4"
    assert cands[1].time_s == pytest.approx(10 / 30, abs=1e-6)
    # brightness rises with the frame index, so the scores must too
    assert [c.score for c in cands] == sorted(c.score for c in cands)
    kept = [c for c in cands if c.bgr is not None]
    assert len(kept) == frames.KEEP_FRAMES  # the rest were released after scoring
    assert kept[0].bgr.shape == (240, 320, 3)


def test_video_candidates_caps_count(clip, fake_frame_score):
    cands = frames.video_candidates(clip, "clip.mp4", sample_fps=30.0, max_candidates=3)
    assert len(cands) <= 3
    assert cands[1].index - cands[0].index >= 30


def test_video_unreadable(tmp_path, monkeypatch, fake_frame_score):
    monkeypatch.setenv("FP_DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(AcquisitionError) as e:
        frames.video_candidates(b"definitely not a video", "junk.mp4")
    assert e.value.code is ReasonCode.VIDEO_UNREADABLE


def test_video_too_long(clip, fake_frame_score):
    with pytest.raises(AcquisitionError) as e:
        frames.video_candidates(clip, "clip.mp4", max_video_s=0.5)
    assert e.value.code is ReasonCode.VIDEO_TOO_LONG


def test_temp_file_removed(clip, tmp_path, fake_frame_score):
    frames.video_candidates(clip, "clip.mp4")
    assert list((tmp_path / "data" / "tmp").iterdir()) == []


def _cand(name, score, time_s=None, index=0):
    return frames.Candidate(
        name=name, bgr=None, original=None, source="c.mp4",
        index=index, time_s=time_s, score=score,
    )


def test_select_takes_the_best_k():
    pool = [_cand("a", 0.9), _cand("b", 0.5), _cand("c", 0.7)]
    assert [c.name for c in frames.select(pool, 2, min_quality=0.4)] == ["a", "c"]


def test_select_refuses_below_the_gate():
    pool = [_cand("a", 0.9), _cand("b", 0.2), _cand("c", 0.1)]
    assert [c.name for c in frames.select(pool, 3, min_quality=0.4)] == ["a"]


def test_select_enforces_a_time_gap():
    pool = [_cand("a", 0.9, time_s=1.0), _cand("b", 0.8, time_s=1.2),
            _cand("c", 0.7, time_s=3.0)]
    got = frames.select(pool, 2, min_quality=0.4, min_gap_s=0.5)
    assert [c.name for c in got] == ["a", "c"]  # b is a near-duplicate of a


def test_select_falls_back_to_the_single_best():
    """Nothing clears the gate: send one frame on so the pipeline reports why."""
    pool = [_cand("a", 0.2), _cand("b", 0.3)]
    got = frames.select(pool, 3, min_quality=0.4)
    assert [c.name for c in got] == ["b"]


def _img(v):
    return np.full((64, 64, 3), v, np.uint8)


def test_expand_default_keeps_stills_unscored(clip, fake_frame_score):
    uploads = [("a.jpg", _img(10), b"a"), ("b.jpg", _img(200), b"b"),
               ("clip.mp4", None, clip)]
    chosen, summary = frames.expand(uploads, None, k_video=2, min_quality=0.4)
    stills = [c for c in chosen if c.time_s is None]
    assert [c.name for c in stills] == ["a.jpg", "b.jpg"]
    assert all(c.score is None for c in stills)  # the photo path is unchanged
    # Frames 80 (2.67 s) and 60 (2.00 s): 70 scores higher than 60 but sits
    # 0.33 s from 80, inside MIN_GAP_S, so it is dropped as a near-duplicate.
    assert [c.index for c in chosen if c.time_s is not None] == [80, 60]
    assert summary["candidates"] == 2 + 9 and summary["selected"] == 4


def test_expand_best_of_pools_everything(clip, fake_frame_score):
    uploads = [("a.jpg", _img(10), b"a"), ("b.jpg", _img(250), b"b"),
               ("clip.mp4", None, clip)]
    chosen, summary = frames.expand(uploads, 1, k_video=3, min_quality=0.4)
    assert len(chosen) == 1 and chosen[0].name == "b.jpg"  # brightest of all
    assert summary["best_of"] == 1 and summary["scores"] == [pytest.approx(250 / 255, abs=1e-3)]


def test_candidate_original_bytes_encodes_a_frame():
    c = frames.Candidate(name="f.jpg", bgr=_img(128), original=None,
                         source="c.mp4", index=3, time_s=0.1, score=0.9)
    data = c.original_bytes()
    assert data[:2] == b"\xff\xd8"  # JPEG SOI
    assert c.meta() == {"source": "c.mp4", "frame": 3, "frame_time_s": 0.1,
                        "candidate_score": 0.9, "strategy": None}


def test_candidate_meta_for_a_still():
    c = frames.Candidate(name="p.jpg", bgr=_img(5), original=b"raw",
                         source="p.jpg", index=0, time_s=None)
    assert c.original_bytes() == b"raw"
    assert c.meta()["frame"] is None and c.meta()["candidate_score"] is None


def test_is_video():
    assert frames.is_video("a.MOV") and frames.is_video("a.bin", "video/mp4")
    assert not frames.is_video("a.jpg") and not frames.is_video("a.heic", "image/heic")


# --- the real scorer, still without the JVM ---------------------------------

def test_frame_score_finds_no_finger_in_a_blank_frame():
    score, strategy, detail = frames.frame_score(np.zeros((480, 640, 3), np.uint8))
    assert score == 0.0 and strategy is None
    assert detail.get("reason") == "no_finger"


def test_guided_frame_score_uses_the_supplied_crop_directly():
    x = np.arange(256, dtype=np.float32)
    ridges = np.uint8(np.clip(128 + 60 * np.sin(2 * np.pi * x / 8), 0, 255))
    gray = np.tile(ridges, (256, 1))
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    score, strategy, detail = frames.guided_frame_score(bgr)
    assert score >= 0.9 and strategy == "full"
    assert set(detail) == {"full"}
    assert detail["full"]["quality"]["valid"] is True


def test_guided_frame_score_rejects_a_flat_crop():
    score, strategy, detail = frames.guided_frame_score(
        np.full((256, 256, 3), 128, np.uint8)
    )
    assert score == 0.0 and strategy is None
    assert detail["full"]["reason"] == "no_ridge_period"


def test_frame_score_on_a_real_photo():
    path = SAMPLES / "thumb_6174.jpg"
    if not path.exists():
        pytest.skip("corpus photo not on disk")
    from fingerphoto import imaging

    score, strategy, detail = frames.frame_score(imaging.decode_image(path.read_bytes()))
    assert score >= 0.6 and strategy in ("distal", "pad")
    assert detail[strategy]["quality"]["valid"] is True


def test_scorer_ranks_good_frames_above_bad_ones():
    """The property selection actually depends on.

    The scorer skips the closed scale loop, so on an ambiguous frame it can
    read higher than the gate (measured: 0.46-0.48 on video frames of a photo
    the gate refuses). That costs a wasted extraction, never a bad enrollment.
    What must hold is the ordering: a usable frame has to outrank an unusable
    one, or selection picks the wrong frames out of a clip."""
    from fingerphoto import imaging

    good, bad = [], []
    for name in ("thumb_6172", "thumb_6174"):
        path = SAMPLES / f"{name}.jpg"
        if path.exists():
            good.append(frames.frame_score(imaging.decode_image(path.read_bytes()))[0])
    for name in ("sister_6176", "web_w_capture"):
        path = SAMPLES / f"{name}.jpg"
        if path.exists():
            bad.append(frames.frame_score(imaging.decode_image(path.read_bytes()))[0])
    if not good or not bad:
        pytest.skip("corpus photos not on disk")
    assert min(good) > max(bad), f"good {good} must outrank bad {bad}"


def test_parked_frames_survive_the_memory_cap(clip, fake_frame_score):
    """A frame outside the top KEEP_FRAMES must still be usable.

    Selection ranks by score but skips near-duplicates, so it routinely
    reaches past the frames whose raw pixels were kept. Those candidates are
    parked as JPEG, not discarded."""
    cands = frames.video_candidates(clip, "clip.mp4", keep_frames=2)
    parked = [c for c in cands if c.bgr is None]
    assert parked, "expected the cap to park some frames"
    assert all(c.original is not None for c in parked)
    img = parked[0].pixels()
    assert img.shape == (240, 320, 3)
    assert parked[0].original_bytes()[:2] == b"\xff\xd8"


def test_selection_can_reach_a_parked_frame(clip, fake_frame_score):
    """End to end: with a tight cap, the gap rule picks a parked frame and it
    still yields pixels."""
    cands = frames.video_candidates(clip, "clip.mp4", keep_frames=1)
    chosen = frames.select(cands, 3, min_quality=0.0, min_gap_s=0.5)
    assert len(chosen) == 3
    for c in chosen:
        assert c.pixels().shape == (240, 320, 3)
