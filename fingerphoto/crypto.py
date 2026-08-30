"""At-rest encryption for template blobs and stored photos (Fernet).

No key configured -> plaintext passthrough with scheme None; the scheme is
recorded per row/file so a store can hold a mix during migration
(scripts/reencrypt.py upgrades plaintext rows offline).

Honest limits: the symmetric key lives in the environment, rotation is an
offline decrypt/re-encrypt, and key loss makes stored originals unrecoverable
— which freezes the gallery at the current SourceAFIS version, since template
re-extraction needs the originals.
"""
from __future__ import annotations

SCHEME_FERNET = "fernet-v1"


def encrypt(data: bytes, key: str | None) -> tuple[bytes, str | None]:
    if not key:
        return data, None
    from cryptography.fernet import Fernet

    return Fernet(key.encode()).encrypt(data), SCHEME_FERNET


def decrypt(data: bytes, scheme: str | None, key: str | None) -> bytes:
    if scheme is None:
        return data
    if scheme != SCHEME_FERNET:
        raise ValueError(f"unknown encryption scheme {scheme!r}")
    if not key:
        raise ValueError("data is encrypted but FP_ENCRYPTION_KEY is not set")
    from cryptography.fernet import Fernet

    return Fernet(key.encode()).decrypt(data)
