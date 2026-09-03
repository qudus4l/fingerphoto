"""Exception handlers: everything leaves as RFC 7807-style problem JSON."""
from __future__ import annotations

import logging
from http import HTTPStatus

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..errors import CodedError, ReasonCode
from .logging import BodyTooLarge, request_id_var

logger = logging.getLogger("fingerphoto.api")

CODE_STATUS = {
    ReasonCode.IDENTITY_NOT_FOUND: 404,
    ReasonCode.PHOTO_NOT_FOUND: 404,
    ReasonCode.JOB_NOT_FOUND: 404,
    ReasonCode.IDENTITY_EXISTS: 409,
    ReasonCode.IMAGE_TOO_LARGE: 413,
    ReasonCode.UNSUPPORTED_MEDIA_TYPE: 415,
    ReasonCode.VIDEO_UNREADABLE: 415,
    ReasonCode.QUEUE_FULL: 503,
    ReasonCode.INTERNAL_ERROR: 500,
}  # everything else (quality gates, consent, empty gallery, ...) -> 422


def problem(status: int, detail: str, reason_code: str | None = None) -> JSONResponse:
    body = {
        "title": HTTPStatus(status).phrase,
        "status": status,
        "detail": detail,
        "reason_code": reason_code,
        "request_id": request_id_var.get(),
    }
    headers = {"Retry-After": "10"} if status == 503 else None
    return JSONResponse(
        body, status_code=status, media_type="application/problem+json", headers=headers
    )


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(CodedError)
    async def coded(request: Request, exc: CodedError):
        return problem(CODE_STATUS.get(exc.code, 422), str(exc), exc.code.value)

    @app.exception_handler(BodyTooLarge)
    async def too_large(request: Request, exc: BodyTooLarge):
        return problem(413, "request body exceeds the upload limit",
                       ReasonCode.IMAGE_TOO_LARGE.value)

    @app.exception_handler(HTTPException)
    async def http_exc(request: Request, exc: HTTPException):
        response = problem(exc.status_code, str(exc.detail))
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation(request: Request, exc: RequestValidationError):
        return problem(422, str(exc.errors()[:3]))

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        logger.exception("unhandled error")
        return problem(500, "internal error; see server logs",
                       ReasonCode.INTERNAL_ERROR.value)
