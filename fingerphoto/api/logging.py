"""JSON structured logging + request-id propagation.

Structural rule: image bytes and templates never reach loggers. Log calls
only ever pass scalars (timings, counts, reason codes, filenames); there is
deliberately no code path that hands ndarray/bytes payloads to logging.
"""
from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = request_id_var.get()
        if rid:
            entry["request_id"] = rid
        for key, value in record.__dict__.items():  # `extra=` fields
            if key not in _RESERVED and not key.startswith("_"):
                entry[key] = value
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def setup_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


class RequestIdMiddleware:
    """Pure ASGI: honor inbound X-Request-ID or mint one; echo it back."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
        rid = headers.get("x-request-id", "")[:64] or uuid.uuid4().hex
        token = request_id_var.set(rid)

        async def send_with_header(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append((b"x-request-id", rid.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            request_id_var.reset(token)


class BodyTooLarge(Exception):
    """Streamed body exceeded the cap mid-read; mapped to 413 by an app
    exception handler (safe: body reads happen before the response starts)."""


async def _send_problem(send, status: int, detail: str, reason_code: str | None) -> None:
    body = json.dumps(
        {"title": "Length Required" if status == 411 else "Payload Too Large",
         "status": status, "detail": detail, "reason_code": reason_code,
         "request_id": request_id_var.get()}
    ).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/problem+json"),
                            (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class BodyLimitMiddleware:
    """Reject oversized request bodies: Content-Length up front (responded to
    directly — exceptions here would bypass the app's handlers), plus a
    streamed byte count so a forged Content-Length can't get a full read."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        headers = {k.lower(): v for k, v in scope.get("headers", [])}
        length = headers.get(b"content-length")
        # Empty POSTs (e.g. /capture/session renew) often omit Content-Length on
        # some mobile browsers. Treat missing as 0 and still count streamed bytes.
        if length is None:
            length = b"0"
        if int(length) > self.max_bytes:
            return await _send_problem(
                send, 413,
                f"request body exceeds the {self.max_bytes // (1024 * 1024)} MB upload limit",
                "IMAGE_TOO_LARGE",
            )
        seen = 0

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise BodyTooLarge()
            return message

        await self.app(scope, limited_receive, send)
