"""Identify (1:N), verify (1:1), frame quality, and job polling."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Header, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from ... import frames, store
from ...errors import AcquisitionError, ReasonCode, StoreError
from ..jobs import Job
from ..logging import request_id_var
from ..schemas import FingerPosition, JobOut, JobSubmitted, QualityOut
from ..settings import get_settings
from ._uploads import accepted, read_uploads

router = APIRouter()

Threshold = Annotated[float | None, Query(ge=0, le=1000)]
IdemKey = Annotated[str | None, Header(alias="Idempotency-Key")]
BestOf = Annotated[int | None, Form(ge=1, le=10)]


def _one_or_many(
    file: UploadFile | None, files: list[UploadFile] | None
) -> list[UploadFile]:
    """Accept either the original single `file` or the newer `files` list, so
    clients that predate multi-frame capture keep working unchanged."""
    picked = [f for f in (files or []) if f is not None] or ([file] if file else [])
    if not picked:
        raise StoreError(
            ReasonCode.UNREADABLE_IMAGE, "send a photo or video as `file` or `files`"
        )
    return picked


@router.post("/v1/identify", status_code=202, response_model=JobSubmitted)
async def identify(
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    threshold: Threshold = None,
    best_of: BestOf = None,
    guided: Annotated[bool, Form()] = False,
    idempotency_key: IdemKey = None,
):
    if not store.list_identities():
        raise StoreError(ReasonCode.GALLERY_EMPTY, "no identities enrolled")
    uploads = await read_uploads(_one_or_many(file, files))
    rid = request_id_var.get()
    job = request.app.state.jobs.submit(
        "identify",
        lambda: store.identify_uploads(
            uploads, threshold, best_of, request_id=rid, guided=guided
        ),
        idempotency_key=idempotency_key,
    )
    return accepted(job)


@router.post(
    "/v1/identities/{identity_uuid}/verify", status_code=202, response_model=JobSubmitted
)
async def verify(
    identity_uuid: str,
    request: Request,
    file: Annotated[UploadFile | None, File()] = None,
    files: Annotated[list[UploadFile] | None, File()] = None,
    threshold: Threshold = None,
    best_of: BestOf = None,
    finger: Annotated[FingerPosition | None, Form()] = None,
    guided: Annotated[bool, Form()] = False,
    idempotency_key: IdemKey = None,
):
    store.get_identity(identity_uuid)  # 404 before decoding anything
    uploads = await read_uploads(_one_or_many(file, files))
    rid = request_id_var.get()
    job = request.app.state.jobs.submit(
        "verify",
        lambda: store.verify_uploads(
            identity_uuid, uploads, threshold, best_of, finger=finger,
            request_id=rid, guided=guided
        ),
        idempotency_key=idempotency_key,
    )
    return accepted(job)


@router.post("/v1/quality", response_model=QualityOut)
async def frame_quality(
    file: Annotated[UploadFile, File()],
    guided: Annotated[bool, Form()] = False,
):
    """Ridge score of a single frame, synchronously.

    This is the capture UI's live indicator: it is the same measurement the
    enrollment gate applies, but it stops before Gabor enhancement, so it costs
    well under a second and needs no job. Videos are refused — scoring a clip
    is what /v1/identify does."""
    name = file.filename or "frame"
    if frames.is_video(name, file.content_type):
        raise AcquisitionError(
            ReasonCode.UNSUPPORTED_MEDIA_TYPE, "send a single image, not a video"
        )
    (_, bgr, _), = await read_uploads([file])
    scorer = frames.guided_frame_score if guided else frames.frame_score
    score, strategy, detail = await run_in_threadpool(scorer, bgr)
    threshold = get_settings().min_ridge_quality
    return QualityOut(
        score=round(score, 4),
        strategy=strategy,
        **{"pass": score >= threshold},
        min_ridge_quality=threshold,
        detail=detail,
    )


@router.get("/v1/jobs/{job_id}", response_model=JobOut)
def job_status(job_id: str, request: Request) -> JobOut:
    job: Job | None = request.app.state.jobs.get(job_id)
    if job is None:
        raise StoreError(
            ReasonCode.JOB_NOT_FOUND, f"no job {job_id} (results expire after the TTL)"
        )
    return JobOut(
        id=job.id,
        kind=job.kind,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        result=job.result,
        error=job.error,
    )
