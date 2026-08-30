"""API-key authentication. /healthz and /readyz are mounted without it."""
from __future__ import annotations

import hmac

from fastapi import HTTPException, Request

from .settings import get_settings


def require_api_key(request: Request) -> None:
    settings = get_settings()
    keys = settings.keys()
    if not keys and settings.dev_no_auth:
        return
    provided = request.headers.get("x-api-key", "")
    for key in keys:
        if hmac.compare_digest(provided.encode(), key.encode()):
            return
    raise HTTPException(
        status_code=401,
        detail="missing or invalid API key (X-API-Key header)",
        headers={"WWW-Authenticate": "ApiKey"},
    )
