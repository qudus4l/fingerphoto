"""Liveness, readiness, and version endpoints."""
from __future__ import annotations

import json
import secrets
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ... import store, version
from ..schemas import VersionOut
from ..security import CAPTURE_SESSION_COOKIE, CAPTURE_SESSION_TTL_S, create_capture_session
from ..settings import get_settings

public_router = APIRouter()  # mounted without auth
router = APIRouter()  # mounted behind the API key

STATIC = Path(__file__).resolve().parent.parent / "static"
_CAPTURE_BOOT = "/*__FP_CONFIG__*/"
_TEST_NAMES = ("Joshua", "Kyle", "Tom")


def _capture_test_name() -> str:
    """Give each page load a readable identity that will not mix test prints."""
    return f"{secrets.choice(_TEST_NAMES)}-{secrets.token_hex(2).upper()}"


def _set_capture_cookie(response: JSONResponse | HTMLResponse, request: Request) -> None:
    response.set_cookie(
        CAPTURE_SESSION_COOKIE,
        create_capture_session(request),
        max_age=CAPTURE_SESSION_TTL_S,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )


@public_router.get("/capture", include_in_schema=False)
async def capture_page(request: Request):
    """Guided capture UI for phone testing.

    Creates a short-lived HttpOnly browser session and injects only a default
    enroll name so the page is one-tap (no credential field or exposed key).
    Browsers only grant camera access over HTTPS or localhost.

    Sent no-store: mobile Safari otherwise serves a stale copy for a long
    time, which silently hides UI fixes during tuning."""
    boot = {"name": _capture_test_name()}
    html = (STATIC / "capture.html").read_text(encoding="utf-8")
    html = html.replace(_CAPTURE_BOOT, f"window.__FP_CONFIG__={json.dumps(boot)};", 1)
    response = HTMLResponse(
        html,
        headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
    )
    _set_capture_cookie(response, request)
    return response


@public_router.post("/capture/session", include_in_schema=False)
async def refresh_capture_session(request: Request):
    """Renew the browser credential without exposing the permanent API key."""
    response = JSONResponse({"status": "ok"})
    _set_capture_cookie(response, request)
    return response


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
        min_ridge_quality=settings.min_ridge_quality,
        video_frames_enroll=settings.video_frames_enroll,
        video_frames_identify=settings.video_frames_identify,
        max_upload_mb=settings.max_upload_mb,
    )
