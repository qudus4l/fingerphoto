"""Identify (1:N), verify (1:1), and job polling."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Header, Query, Request, UploadFile

from ... import store
from ...errors import ReasonCode, StoreError
from ..jobs import Job
from ..logging import request_id_var
from ..schemas import FingerPosition, JobOut, JobSubmitted
from ._uploads import accepted, decoded_uploads

router = APIRouter()

Threshold = Annotated[float | None, Query(ge=0, le=1000)]
IdemKey = Annotated[str | None, Header(alias="Idempotency-Key")]


@router.post("/v1/identify", status_code=202, response_model=JobSubmitted)
async def identify(
    request: Request,
    file: Annotated[UploadFile, File()],
    threshold: Threshold = None,
    idempotency_key: IdemKey = None,
):
    if not store.list_identities():
        raise StoreError(ReasonCode.GALLERY_EMPTY, "no identities enrolled")
    _, bgr, _ = (await decoded_uploads([file]))[0]
    rid = request_id_var.get()
    job = request.app.state.jobs.submit(
        "identify",
        lambda: store.identify_bgr(bgr, threshold, request_id=rid),
        idempotency_key=idempotency_key,
    )
    return accepted(job)


@router.post(
    "/v1/identities/{identity_uuid}/verify", status_code=202, response_model=JobSubmitted
)
async def verify(
    identity_uuid: str,
    request: Request,
    file: Annotated[UploadFile, File()],
    threshold: Threshold = None,
    finger: Annotated[FingerPosition | None, Form()] = None,
    idempotency_key: IdemKey = None,
):
    store.get_identity(identity_uuid)  # 404 before decoding anything
    _, bgr, _ = (await decoded_uploads([file]))[0]
    rid = request_id_var.get()
    job = request.app.state.jobs.submit(
        "verify",
        lambda: store.verify_bgr(identity_uuid, bgr, threshold, finger=finger, request_id=rid),
        idempotency_key=idempotency_key,
    )
    return accepted(job)


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
