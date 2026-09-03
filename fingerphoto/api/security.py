"""API-key authentication. /healthz and /readyz are mounted without it."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time

from fastapi import HTTPException, Request

from .settings import get_settings


CAPTURE_SESSION_COOKIE = "fp-capture-session"
CAPTURE_SESSION_TTL_S = 12 * 3600


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def create_capture_session(request: Request, ttl_s: int = CAPTURE_SESSION_TTL_S) -> str:
    """Create a signed browser session without exposing an API key.

    The signature is stateless: it remains valid when the server reloads and
    when requests are handled by different worker processes. The capture page
    is intentionally a no-login test surface, so this remains a convenience
    credential rather than a replacement for network access controls.
    """
    keys = get_settings().keys()
    if not keys:  # dev_no_auth already bypasses authentication
        return secrets.token_urlsafe(32)
    payload = f"{int(time.time()) + ttl_s}.{secrets.token_urlsafe(18)}".encode("ascii")
    signature = hmac.new(keys[0].encode(), payload, hashlib.sha256).digest()
    return f"{_b64encode(payload)}.{_b64encode(signature)}"


def valid_capture_session(token: str, keys: list[str]) -> bool:
    try:
        encoded_payload, encoded_signature = token.split(".", 1)
        payload = _b64decode(encoded_payload)
        signature = _b64decode(encoded_signature)
        expires_text, _ = payload.split(b".", 1)
        if int(expires_text) <= int(time.time()):
            return False
    except (ValueError, TypeError, UnicodeError):
        return False
    return any(
        hmac.compare_digest(
            signature,
            hmac.new(key.encode(), payload, hashlib.sha256).digest(),
        )
        for key in keys
    )


def require_api_key(request: Request) -> None:
    settings = get_settings()
    keys = settings.keys()
    if not keys and settings.dev_no_auth:
        return
    provided = request.headers.get("x-api-key", "")
    for key in keys:
        if hmac.compare_digest(provided.encode(), key.encode()):
            return
    session = request.cookies.get(CAPTURE_SESSION_COOKIE, "")
    if session and valid_capture_session(session, keys):
        return
    raise HTTPException(
        status_code=401,
        detail="missing or invalid API key (X-API-Key header)",
        headers={"WWW-Authenticate": "ApiKey"},
    )
