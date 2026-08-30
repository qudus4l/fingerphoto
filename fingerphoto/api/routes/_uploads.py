"""Shared upload helpers for the recognition/enrollment routes."""
from __future__ import annotations

import numpy as np
from fastapi import UploadFile
from fastapi.responses import JSONResponse

from ... import imaging
from ..jobs import Job


async def decoded_uploads(files: list[UploadFile]) -> list[tuple[str, np.ndarray, bytes]]:
    """Read and decode every upload up front (~100 ms each) so garbage is
    rejected with 415/422 immediately instead of wasting a job slot."""
    out = []
    for f in files:
        name = f.filename or "photo"
        data = await f.read()
        out.append((name, imaging.decode_image(data, name), data))
    return out


def accepted(job: Job) -> JSONResponse:
    """202 + Location + Retry-After (async request-reply convention)."""
    poll = f"/v1/jobs/{job.id}"
    return JSONResponse(
        {"job_id": job.id, "status": job.status, "poll": poll},
        status_code=202,
        headers={"Location": poll, "Retry-After": "5"},
    )
