"""Candidate frames: score many, extract few.

A photo carries one chance at a readable print; a few seconds of video carries
dozens. This module turns any upload -- a still, a browser burst, or a clip
recorded in the phone's Camera app -- into a pool of `Candidate` frames, scores
every one of them, and hands the best K to the (expensive) template pipeline.

Why score here rather than extract everything: a full extraction is 10-20 s per
frame (`out/eval/latest.json`), while the score below is 0.3-1.0 s because it
stops before the Gabor stage. Scoring 24 candidates and extracting 3 costs about
a minute; extracting 24 would cost six.

The score is `quality.ridge_quality` on the scale-normalized crop -- the same
measurement `identity._templates_for_crop` gates on -- plus the same geometric
sanity rules via `identity.crop_sanity_reason`. So a frame this module selects
is one the gate will accept, and the selector cannot disagree with the gate.
Measured against the gate's verdict on all 17 corpus photos (2026-09-02): every
photo the gate accepted scores >= 0.75 here, every photo it refused scores
<= 0.333, and the gate threshold 0.40 falls in the gap.

The selector is a PRE-FILTER, never the verdict. It skips the closed scale
loop (that needs the Gabor stage), so on an ambiguous frame it can settle on a
different period than the gate does and read higher. Measured on a 1080x1920
H.264 clip built from corpus photos: on frames the gate accepts the two agree
exactly (0.774 and 1.000 from both), while frames of a no-ridge photo scored
0.46-0.48 here and were still refused by the gate as `ridge_quality`. The cost
of that disagreement is a wasted extraction when a clip contains nothing
better; the gate remains the only thing that decides what is enrolled. What
selection must get right is the RANKING, and it does: good frames outrank bad
ones by a wide margin, which is what picks the right frames out of a clip.

SCORE AT FULL RESOLUTION. Ridges sit 8-20 px apart in a 4K frame, so downscaling
destroys the very signal being measured. Measured on the corpus: at a 2560 px
long side thumb_6217 falls 0.91 -> 0.00 and the failed probe match-64 rises
0.17 -> 0.45 (i.e. it crosses the gate); at 1920 px thumb_6173 falls 0.99 ->
0.41. If frame scoring is ever too slow, sample FEWER frames, never smaller ones.

Video decoding is cv2's bundled FFmpeg (the headless wheel in the Docker image
has it too), so H.264 mp4 and HEVC mov both open with no extra dependency.
CAP_PROP_ORIENTATION_AUTO defaults to on, so an iPhone clip stored landscape
with a 90-degree display matrix is delivered as portrait frames -- the finger-up
assumption in segmentation still holds. cv2.VideoCapture needs a real path, so
the upload is written to a temp file and unlinked in a finally block.

Originals stored for a video enrollment are the SELECTED FRAMES as JPEG, not the
clip: those are the images the templates came from, which is what re-extraction
needs, and a 5 s 4K clip is 15-35 MB against ~300 KB per frame.
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import identity
from .errors import AcquisitionError, ReasonCode
from .locate import crop_pad
from .pipeline import _scaled_mask
from .quality import MIN_RIDGE_QUALITY, ridge_quality
from .scale import estimate_ridge_period, normalize_scale, ridge_period_band_for_fingertip
from .segment import crop_distal

logger = logging.getLogger("fingerphoto.frames")

SAMPLE_FPS = 3.0        # candidate frames per second of video
MAX_CANDIDATES = 24     # hard cap per clip (~24 x 0.5 s of scoring)
MAX_VIDEO_S = 20.0      # longer clips are refused: decode cost is per frame, not per candidate
MIN_GAP_S = 0.5         # selected video frames must be this far apart (adjacent frames are near-duplicates)
VIDEO_FRAMES_ENROLL = 3
VIDEO_FRAMES_IDENTIFY = 2
JPEG_QUALITY = 95
VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".webm", ".mkv", ".3gp"}

# Frames whose pixels are kept while scanning a clip. Everything else is scored
# and dropped: 24 candidates x one 4K BGR frame is ~600 MB resident.
KEEP_FRAMES = 4


@dataclass
class Candidate:
    """One frame that could become a template set."""

    name: str                    # "<stem>_f0123.jpg" for a video frame, else the upload's filename
    bgr: np.ndarray | None       # None once dropped by the memory cap (score is still valid)
    original: bytes | None       # upload bytes for a still; None for a video frame (encoded on demand)
    source: str                  # filename of the upload it came from
    index: int                   # frame index in the clip, or position in the upload list
    time_s: float | None         # frame time in the clip; None for stills
    score: float | None = None   # ridge score; None = never scored (single-photo path)
    strategy: str | None = None  # "distal" | "pad" | None
    detail: dict = field(default_factory=dict)

    def original_bytes(self) -> bytes:
        """Bytes to archive for this candidate: the upload as sent, or the frame
        encoded as JPEG (what the templates were actually extracted from)."""
        if self.original is not None:
            return self.original
        if self.bgr is None:
            raise ValueError(f"candidate {self.name} has neither original bytes nor pixels")
        return _encode(self.bgr, self.name)

    def pixels(self) -> np.ndarray:
        """The frame itself, decoding the parked JPEG if the memory cap took
        the raw pixels. Selection ranks by score but skips near-duplicates, so
        it can reach past the frames whose pixels were kept -- without this the
        gap rule would hand the pipeline a candidate with no image."""
        if self.bgr is not None:
            return self.bgr
        if self.original is None:
            raise ValueError(f"candidate {self.name} has no pixels")
        buf = np.frombuffer(self.original, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"could not decode parked frame {self.name}")
        return img

    def meta(self) -> dict:
        """Provenance for the API result, so a caller can see which frame won."""
        return {
            "source": self.source,
            "frame": self.index if self.time_s is not None else None,
            "frame_time_s": round(self.time_s, 3) if self.time_s is not None else None,
            "candidate_score": round(self.score, 3) if self.score is not None else None,
            "strategy": self.strategy,
        }


def frame_score(bgr: np.ndarray, frame_shape: tuple[int, int] | None = None) -> tuple[float, str | None, dict]:
    """Ridge score of one frame: (score, strategy, detail).

    Runs both crop strategies and keeps the better. Stops before Gabor
    enhancement -- the enhancer invents ridges from blur, so anything measured
    after it proves nothing (see quality.py) -- which is also what makes this
    cheap enough to run on every frame of a clip."""
    if frame_shape is None:
        frame_shape = bgr.shape[:2]
    detail: dict = {}
    best_score, best_strategy = 0.0, None

    for strategy in ("distal", "pad"):
        rep: dict = {}
        detail[strategy] = rep
        try:
            if strategy == "distal":
                crop, mask = crop_distal(bgr)
                hint = None
            else:
                crop, mask, hint = crop_pad(bgr)
        except (ValueError, cv2.error) as e:
            rep["reason"] = "no_crop"
            rep["error"] = e.__class__.__name__
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        lo, hi = ridge_period_band_for_fingertip(crop.shape[1])
        rep["crop_shape"] = list(crop.shape[:2])
        rep["band_px"] = (round(lo, 2), round(hi, 2))
        try:
            period = hint if hint is not None else estimate_ridge_period(
                gray, min_period=lo, max_period=hi
            )
        except (ValueError, cv2.error):
            rep["reason"] = "no_period"
            continue
        rep["initial_period_px"] = round(period, 2)

        # Same geometric rules the real gate applies, so this pre-filter can
        # never pass a crop identity.py would refuse on geometry alone. The
        # band-edge test needs the closed scale loop, which has not run here;
        # the out-of-band test is the one that catches whole-hand crops.
        reason = identity.crop_sanity_reason(
            {"period_clamped": period < lo or period > hi, "period_at_band_edge": False},
            strategy,
            crop.shape[:2],
            frame_shape,
        )
        if reason is not None:
            rep["reason"] = reason
            rep["score"] = 0.0
            continue

        period = min(max(period, lo), hi)
        try:
            scaled, _ = normalize_scale(gray, period)
            q = ridge_quality(scaled, _scaled_mask(mask, scaled.shape))
        except (ValueError, cv2.error) as e:
            rep["reason"] = "scale_failed"
            rep["error"] = e.__class__.__name__
            continue

        rep["period_px"] = round(period, 2)
        rep["quality"] = q.as_dict()
        rep["score"] = round(q.score, 4)
        if q.score > best_score:
            best_score, best_strategy = q.score, strategy

    if best_strategy is None and all("score" not in r for r in detail.values()):
        detail["reason"] = "no_finger"
    return best_score, best_strategy, detail


def image_candidates(
    uploads: list[tuple[str, np.ndarray, bytes | None]], score: bool
) -> list[Candidate]:
    """One candidate per decoded still. Scored only when asked: the classic
    single-photo path must stay byte-for-byte what it was, including its cost."""
    out = []
    for i, (name, bgr, data) in enumerate(uploads):
        c = Candidate(
            name=name, bgr=bgr, original=data, source=name, index=i, time_s=None
        )
        if score:
            c.score, c.strategy, c.detail = frame_score(bgr)
        out.append(c)
    return out


def _temp_dir() -> Path:
    d = Path(os.environ.get("FP_DATA_DIR", Path(__file__).resolve().parent.parent / "data")) / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def video_candidates(
    data: bytes,
    name: str,
    *,
    sample_fps: float = SAMPLE_FPS,
    max_candidates: int = MAX_CANDIDATES,
    max_video_s: float = MAX_VIDEO_S,
    keep_frames: int = KEEP_FRAMES,
) -> list[Candidate]:
    """Decode a clip, score every sampled frame, return the candidates.

    Only the top `keep_frames` scoring frames keep their pixels; the rest are
    scored and released, so a long 4K clip does not hold hundreds of megabytes
    of decoded frames at once."""
    stem = Path(name).stem or "clip"
    suffix = Path(name).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        suffix = ".mp4"  # FFmpeg probes the container anyway; the suffix only helps it
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, dir=_temp_dir(), delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return _scan(tmp.name, stem, name, sample_fps, max_candidates, max_video_s, keep_frames)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _scan(
    path: str,
    stem: str,
    source: str,
    sample_fps: float,
    max_candidates: int,
    max_video_s: float,
    keep_frames: int,
) -> list[Candidate]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise AcquisitionError(
            ReasonCode.VIDEO_UNREADABLE, f"{source} is not a decodable video"
        )
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or math.isnan(fps):
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0 and total / fps > max_video_s:
            raise AcquisitionError(
                ReasonCode.VIDEO_TOO_LONG,
                f"{source} is {total / fps:.1f}s; the limit is {max_video_s:.0f}s",
            )
        stride = max(1, round(fps / sample_fps))
        if total > 0:
            stride = max(stride, math.ceil(total / max_candidates))

        out: list[Candidate] = []
        i = 0
        while True:
            # grab() advances without decoding to BGR; only sampled frames pay
            # the retrieve() cost.
            if not cap.grab():
                break
            if i % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    score, strategy, detail = frame_score(frame)
                    out.append(
                        Candidate(
                            name=f"{stem}_f{i:04d}.jpg",
                            bgr=frame,
                            original=None,
                            source=source,
                            index=i,
                            time_s=i / fps,
                            score=score,
                            strategy=strategy,
                            detail=detail,
                        )
                    )
                    _trim(out, keep_frames)
                    if len(out) >= max_candidates:
                        break
            i += 1
            if total <= 0 and i / fps > max_video_s:
                raise AcquisitionError(
                    ReasonCode.VIDEO_TOO_LONG,
                    f"{source} is longer than the {max_video_s:.0f}s limit",
                )
    finally:
        cap.release()

    if not out:
        raise AcquisitionError(ReasonCode.VIDEO_UNREADABLE, f"{source} decoded no frames")
    logger.info(
        "scored clip",
        extra={
            "source": source, "fps": round(fps, 2), "stride": stride,
            "candidates": len(out),
            "best_score": round(max((c.score or 0.0) for c in out), 3),
        },
    )
    return out


def _encode(bgr: np.ndarray, name: str) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise ValueError(f"could not encode frame {name}")
    return buf.tobytes()


def _trim(cands: list[Candidate], keep: int) -> None:
    """Park every candidate outside the current top `keep` as JPEG bytes and
    release its raw pixels: 24 decoded 4K frames is ~600 MB resident, the same
    frames as JPEG are ~7 MB. Parking rather than discarding matters because
    selection's gap rule can choose a frame outside the top `keep`."""
    ranked = sorted(cands, key=lambda c: -(c.score or 0.0))
    for c in ranked[keep:]:
        if c.bgr is not None:
            if c.original is None:
                c.original = _encode(c.bgr, c.name)
            c.bgr = None


def select(
    cands: list[Candidate],
    k: int,
    *,
    min_quality: float = MIN_RIDGE_QUALITY,
    min_gap_s: float = MIN_GAP_S,
) -> list[Candidate]:
    """The best `k` candidates worth extracting.

    Only frames that clear the gate are selected -- extracting a frame the gate
    will refuse spends 15 s to learn what the 0.5 s score already said. When
    nothing clears it, the single best frame is returned anyway so the full
    pipeline produces the usual refusal, with its measured numbers, instead of
    a bare "no candidates" that tells the user nothing."""
    ranked = sorted(cands, key=lambda c: -(c.score or 0.0))
    chosen: list[Candidate] = []
    for c in ranked:
        if (c.score or 0.0) < min_quality:
            break  # ranked, so nothing further can pass either
        if c.time_s is not None and any(
            o.time_s is not None and abs(c.time_s - o.time_s) < min_gap_s for o in chosen
        ):
            continue  # near-duplicate of a frame already chosen
        chosen.append(c)
        if len(chosen) == k:
            break
    return chosen or ranked[:1]


def is_video(name: str, content_type: str | None = None) -> bool:
    if content_type and content_type.startswith("video/"):
        return True
    return Path(name).suffix.lower() in VIDEO_SUFFIXES


def expand(
    uploads: list[tuple[str, np.ndarray | None, bytes | None]],
    best_of: int | None,
    *,
    k_video: int,
    min_quality: float = MIN_RIDGE_QUALITY,
    on_error=None,
) -> tuple[list[Candidate], dict]:
    """Uploads -> the candidates actually worth extracting, plus a summary.

    `uploads` items are (filename, bgr, data); a `bgr` of None marks a video,
    whose frames are decoded here.

    best_of=None (the default, and what clients that predate this send): stills
    pass through untouched and unscored, videos contribute their best `k_video`
    frames. best_of=k: everything -- stills and video frames alike -- competes
    in one pool and the best k win. That is how the browser burst works: it
    uploads all five frames and lets the server pick on ridge score rather than
    on the client's Laplacian guess.

    on_error(name, exc), when given, is called for an upload that cannot be
    expanded at all (an unreadable or over-long clip) and that upload is
    skipped; without it the error propagates. Either way the remaining uploads
    still compete in one pool, which per-upload expansion would have broken."""
    per_video: list[list[Candidate]] = []
    stills: list[tuple[str, np.ndarray, bytes | None]] = []

    for name, bgr, data in uploads:
        if bgr is None:
            try:
                if data is None:
                    raise AcquisitionError(
                        ReasonCode.VIDEO_UNREADABLE, f"{name} has no content"
                    )
                per_video.append(video_candidates(data, name))
            except AcquisitionError as e:
                if on_error is None:
                    raise
                on_error(name, e)
        else:
            stills.append((name, bgr, data))

    n_considered = len(stills) + sum(len(v) for v in per_video)
    if best_of is None:
        # Each still is its own photo; each clip contributes its own best frames.
        chosen = image_candidates(stills, score=False)
        for cands in per_video:
            chosen += select(cands, k_video, min_quality=min_quality)
    else:
        # One pool, one competition.
        pool = image_candidates(stills, score=True)
        for cands in per_video:
            pool += cands
        chosen = select(pool, best_of, min_quality=min_quality)

    summary = {
        "candidates": n_considered,
        "selected": len(chosen),
        "scores": [round(c.score, 3) for c in chosen if c.score is not None],
        "best_of": best_of,
    }
    return chosen, summary
