"""Shared upload helpers for the recognition/enrollment routes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import UploadFile
from fastapi.responses import JSONResponse

from ... import frames, imaging
from ...errors import AcquisitionError, ReasonCode
from ..jobs import Job

# Container magic. MP4/MOV/3GP put "ftyp" at offset 4; Matroska/WebM start with
# the EBML header. This is only a fail-fast check so obvious garbage is refused
# before it takes a job slot -- FFmpeg does the real probing later.
_FTYP = b"ftyp"
_EBML = b"\x1a\x45\xdf\xa3"


async def read_uploads(
    files: list[UploadFile],
) -> list[tuple[str, np.ndarray | None, bytes]]:
    """Read every upload; decode the images, leave the videos encoded.

    Images are decoded here (~100 ms each) so garbage is rejected with 415/422
    immediately instead of wasting a job slot. Videos are NOT decoded here:
    that is seconds of work per clip and belongs in the job, so they get a
    cheap container check instead and travel to the worker as bytes (a bgr of
    None is how the store layer recognises one).

    Note the content-type test comes first: HEIC also carries "ftyp" magic, so
    sniffing bytes alone would misfile every iPhone photo as a video."""
    out: list[tuple[str, np.ndarray | None, bytes]] = []
    for f in files:
        name = f.filename or "photo"
        data = await f.read()
        if frames.is_video(name, f.content_type):
            if not (data[4:8] == _FTYP or data[:4] == _EBML):
                raise AcquisitionError(
                    ReasonCode.UNSUPPORTED_MEDIA_TYPE,
                    f"{name} is not a recognised video container",
                )
            out.append((name, None, data))
        else:
            out.append((name, imaging.decode_image(data, name), data))
    return out


async def decoded_uploads(
    files: list[UploadFile],
) -> list[tuple[str, np.ndarray, bytes]]:
    """Images only — kept for callers that cannot handle a video."""
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
