from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from fingerphoto import crypto


def test_passthrough_without_key():
    data, scheme = crypto.encrypt(b"hello", None)
    assert (data, scheme) == (b"hello", None)
    assert crypto.decrypt(data, scheme, None) == b"hello"


def test_round_trip_with_key():
    key = Fernet.generate_key().decode()
    data, scheme = crypto.encrypt(b"secret", key)
    assert scheme == crypto.SCHEME_FERNET and data != b"secret"
    assert crypto.decrypt(data, scheme, key) == b"secret"


def test_decrypt_requires_key():
    key = Fernet.generate_key().decode()
    data, scheme = crypto.encrypt(b"secret", key)
    with pytest.raises(ValueError):
        crypto.decrypt(data, scheme, None)


def test_unknown_scheme_rejected():
    with pytest.raises(ValueError):
        crypto.decrypt(b"x", "rot13-v1", "key")
