"""Liveness, readiness, and version endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from ... import store, version
from ..schemas import VersionOut
from ..settings import get_settings

public_router = APIRouter()  # mounted without auth
router = APIRouter()  # mounted behind the API key

STATIC = Path(__file__).resolve().parent.parent / "static"


@public_router.get("/capture", include_in_schema=False)
async def capture_page():
    """Guided capture UI. Unauthenticated because it is just a static page —
    the API key is entered in the form and travels on the XHRs it makes.
    Note: browsers only grant camera access over HTTPS or localhost.

    Sent no-store: mobile Safari otherwise serves a stale copy for a long
    time, which silently hides UI fixes during tuning."""
    return FileResponse(
        STATIC / "capture.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
    )


@public_router.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@public_router.get("/readyz")
async def readyz(request: Request):
    problems = []
    if not getattr(request.app.state, "jvm_ready", False):
        problems.append("jvm not started")
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is not None and jobs.saturated():
        problems.append("job queue saturated")
    try:
        con = store.connect()
        con.execute("BEGIN IMMEDIATE")
        con.execute("ROLLBACK")
        con.close()
    except Exception as e:
        problems.append(f"db: {e.__class__.__name__}")
    if problems:
        return JSONResponse({"status": "unavailable", "problems": problems}, status_code=503)
    return {"status": "ready"}


@router.get("/v1/version", response_model=VersionOut)
def version_info() -> VersionOut:
    settings = get_settings()
    return VersionOut(
        app=version.APP_VERSION,
        pipeline=version.PIPELINE_VERSION,
        sourceafis=version.sourceafis_version(),
        threshold=settings.threshold,
        min_minutiae=settings.min_minutiae,
    )
