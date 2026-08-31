"""Version stamps for the app, the pipeline, and SourceAFIS.

PIPELINE_VERSION must be bumped whenever any parameter that affects template
extraction changes (segmentation, scale loop, Gabor, crop strategies, scale
sweep) — templates are only comparable within a pipeline version, and stored
rows are stamped with it so stale sets can be found and re-extracted.
"""
from __future__ import annotations

import re
from pathlib import Path

APP_VERSION = "1.0.0"
# 1.1.0: the closed-loop scale correction is now clamped to the physical ridge
# period band on every iteration and returns its best iteration rather than its
# last. Templates from a diverging 1.0.0 run were extracted at up to 2x the
# correct scale, so 1.0.0 rows are not comparable with 1.1.0 ones — re-extract
# them from stored originals with scripts/reextract.py.
PIPELINE_VERSION = "1.1.0"

JARS_DIR = Path(__file__).resolve().parent.parent / "jars"


def sourceafis_version() -> str:
    """SourceAFIS version, parsed from the jar actually on the classpath."""
    for jar in sorted(JARS_DIR.glob("sourceafis-*.jar")):
        m = re.match(r"sourceafis-(\d+(?:\.\d+)*)\.jar$", jar.name)
        if m:
            return m.group(1)
    return "unknown"
