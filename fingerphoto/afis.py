"""SourceAFIS bridge via JPype.

Templates are opaque CBOR blobs (version-locked: keep original images so they
can be re-extracted after a SourceAFIS upgrade). Score threshold 40 roughly
corresponds to 0.01% false-match rate per SourceAFIS docs; tune on real data.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

JARS_DIR = Path(__file__).resolve().parent.parent / "jars"
THRESHOLD = 40.0

# The JVM starts once per process and can never be restarted (JPype/JNI
# limitation) — so it must start lazily AFTER any fork, and exactly once
# even when the first callers are concurrent pool threads.
_JVM_LOCK = threading.Lock()


def _ensure_jvm() -> None:
    import jpype
    import jpype.imports  # registers the Java import hook (enables `from com...`)

    with _JVM_LOCK:
        if not jpype.isJVMStarted():
            jars = sorted(str(p) for p in JARS_DIR.glob("*.jar"))
            if not jars:
                raise RuntimeError(f"no jars found in {JARS_DIR}; run scripts/fetch_jars.sh")
            heap = os.environ.get("FP_JVM_HEAP", "512m")
            jpype.startJVM(f"-Xmx{heap}", classpath=jars)


def _jbytes(data: bytes):
    import jpype

    return jpype.JArray(jpype.JByte)(np.frombuffer(data, dtype=np.int8))


def extract_template(gray: np.ndarray, dpi: float = 500.0) -> bytes:
    """Extract a template from a grayscale uint8 image (dark ridges, light valleys)."""
    import cv2

    _ensure_jvm()
    from com.machinezoo.sourceafis import (
        FingerprintImage,
        FingerprintImageOptions,
        FingerprintTemplate,
    )

    ok, png = cv2.imencode(".png", gray)
    if not ok:
        raise ValueError("PNG encoding failed")
    image = FingerprintImage(_jbytes(png.tobytes()), FingerprintImageOptions().dpi(float(dpi)))
    return bytes(FingerprintTemplate(image).toByteArray())


def match_one(probe: bytes, candidate: bytes) -> float:
    return match_many(probe, [candidate])[0]


def match_many(probe: bytes, candidates: list[bytes]) -> list[float]:
    """Score probe against candidates. Build the matcher once: extraction is the
    expensive step, matching is thousands/sec."""
    _ensure_jvm()
    from com.machinezoo.sourceafis import FingerprintMatcher, FingerprintTemplate

    matcher = FingerprintMatcher(FingerprintTemplate(_jbytes(probe)))
    return [float(matcher.match(FingerprintTemplate(_jbytes(c)))) for c in candidates]


def minutiae_count(template: bytes) -> int | None:
    """Diagnostic: number of minutiae in a serialized template (CBOR)."""
    try:
        import cbor2
    except ImportError:
        return None
    data = cbor2.loads(template)
    for key in ("positionsX", "x", "minutiaeX"):
        if key in data:
            return len(data[key])
    return None
