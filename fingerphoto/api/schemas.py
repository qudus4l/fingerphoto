"""Request/response models. Field names mirror the store layer's dicts."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

FingerPosition = Literal[
    "unknown",
    "right_thumb", "right_index", "right_middle", "right_ring", "right_little",
    "left_thumb", "left_index", "left_middle", "left_ring", "left_little",
]


class IdentityIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    consent: bool = Field(
        description="Must be true: explicit consent to process biometric data "
        "(GDPR Art. 9 / BIPA written release recorded by the caller)."
    )


class IdentityOut(BaseModel):
    uuid: str
    name: str
    consent: bool
    consent_at: float | None = None
    created_at: float | None = None
    templates: int | None = None
    photos: int | None = None


class ScoreEntry(BaseModel):
    uuid: str
    name: str
    score: float


class PhotoResult(BaseModel):
    filename: str
    status: Literal["enrolled", "rejected"]
    reason_code: str | None = None
    detail: str | None = None
    templates: int | None = None
    minutiae: int | None = None
    quality: float | None = None
    timings: dict | None = None
    # Provenance: which upload and, for a video, which frame this came from.
    source: str | None = None
    frame: int | None = None
    frame_time_s: float | None = None
    candidate_score: float | None = None
    strategy: str | None = None


class QualityOut(BaseModel):
    """Ridge score of one frame (POST /v1/quality)."""

    score: float
    strategy: str | None = None
    pass_: bool = Field(alias="pass")
    min_ridge_quality: float
    detail: dict = {}

    model_config = {"populate_by_name": True}


class JobOut(BaseModel):
    id: str
    kind: str
    status: Literal["pending", "running", "succeeded", "failed"]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: dict | None = None


class JobSubmitted(BaseModel):
    job_id: str
    status: str
    poll: str


class VersionOut(BaseModel):
    app: str
    pipeline: str
    sourceafis: str
    threshold: float
    min_minutiae: int
    min_ridge_quality: float
    video_frames_enroll: int
    video_frames_identify: int
    max_upload_mb: int


class Problem(BaseModel):
    """RFC 7807-style error body (application/problem+json)."""

    title: str
    status: int
    detail: str | None = None
    reason_code: str | None = None
    request_id: str | None = None
