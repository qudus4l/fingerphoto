"""Identity CRUD + photo enrollment."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Header, Request, UploadFile

from ... import store
from ...errors import ReasonCode, StoreError
from ..logging import request_id_var
from ..schemas import FingerPosition, IdentityIn, IdentityOut, JobSubmitted
from ._uploads import accepted, decoded_uploads

router = APIRouter()


@router.post("/v1/identities", status_code=201, response_model=IdentityOut)
def create_identity(body: IdentityIn) -> dict:
    return store.create_identity(body.name, body.consent, request_id=request_id_var.get())


@router.get("/v1/identities", response_model=list[IdentityOut])
def list_identities() -> list[dict]:
    return store.list_identities()


@router.get("/v1/identities/{identity_uuid}", response_model=IdentityOut)
def get_identity(identity_uuid: str) -> dict:
    return store.get_identity(identity_uuid)


@router.delete("/v1/identities/{identity_uuid}")
def delete_identity(identity_uuid: str) -> dict:
    """GDPR/BIPA hard delete: templates, identity row, and original photos."""
    return store.delete_identity(identity_uuid, request_id=request_id_var.get())


@router.post(
    "/v1/identities/{identity_uuid}/photos", status_code=202, response_model=JobSubmitted
)
async def enroll_photos(
    identity_uuid: str,
    request: Request,
    files: Annotated[list[UploadFile], File()],
    finger: Annotated[FingerPosition, Form()] = "unknown",
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
):
    info = store.get_identity(identity_uuid)  # 404 before decoding anything
    if not info["consent"]:
        raise StoreError(
            ReasonCode.CONSENT_REQUIRED,
            "identity has no recorded consent; re-create it with consent before enrolling",
        )
    photos = await decoded_uploads(files)  # 415/422 before taking a job slot
    rid = request_id_var.get()
    job = request.app.state.jobs.submit(
        "enroll",
        lambda: store.enroll_photos(identity_uuid, photos, finger=finger, request_id=rid),
        idempotency_key=idempotency_key,
    )
    return accepted(job)
