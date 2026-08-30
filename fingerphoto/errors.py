"""Typed error taxonomy shared by the pipeline, store, and API.

Both error classes subclass ValueError so the existing scripts' broad
`except ValueError` handlers keep working unchanged.
"""
from __future__ import annotations

from enum import Enum


class ReasonCode(str, Enum):
    UNREADABLE_IMAGE = "UNREADABLE_IMAGE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    NO_FINGER_FOUND = "NO_FINGER_FOUND"
    QUALITY_TOO_LOW = "QUALITY_TOO_LOW"
    GALLERY_EMPTY = "GALLERY_EMPTY"
    IDENTITY_NOT_FOUND = "IDENTITY_NOT_FOUND"
    IDENTITY_EXISTS = "IDENTITY_EXISTS"
    IDENTITY_HAS_NO_TEMPLATES = "IDENTITY_HAS_NO_TEMPLATES"
    CONSENT_REQUIRED = "CONSENT_REQUIRED"
    INVALID_FINGER_POSITION = "INVALID_FINGER_POSITION"
    QUEUE_FULL = "QUEUE_FULL"
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class CodedError(ValueError):
    """A failure with a machine-readable reason code."""

    def __init__(self, code: ReasonCode, message: str):
        super().__init__(message)
        self.code = code


class AcquisitionError(CodedError):
    """A photo could not be turned into usable templates."""


class StoreError(CodedError):
    """An identity-store operation failed (not found, no consent, ...)."""
