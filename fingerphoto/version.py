"""Version stamps for the app, the pipeline, and SourceAFIS.

PIPELINE_VERSION must be bumped whenever any parameter that affects template
extraction changes (segmentation, scale loop, Gabor, crop strategies, scale
sweep) — templates are only comparable within a pipeline version, and stored
rows are stamped with it so stale sets can be found and re-extracted.
"""
from __future__ import annotations

import re
from pathlib import Path

# 1.1.0: video and multi-frame capture. Uploads can be clips or frame sets;
# the server scores every candidate frame with the ridge detector (frames.py)
# and extracts only the best few. Adds best_of, video uploads on the three
# capture endpoints, and POST /v1/quality for the live capture indicator.
APP_VERSION = "1.1.0"
# 1.1.0: the closed-loop scale correction is now clamped to the physical ridge
# period band on every iteration and returns its best iteration rather than its
# last. Templates from a diverging 1.0.0 run were extracted at up to 2x the
# correct scale, so 1.0.0 rows are not comparable with 1.1.0 ones — re-extract
# them from stored originals with scripts/reextract.py.
# 1.2.0: ridge-structure quality gate (quality.py) plus crop sanity rules.
# Crops with no resolvable ridges, or with non-fingertip geometry, no longer
# produce templates at all, so the template SET stored for a photo differs from
# 1.1.0 even though an accepted crop is extracted identically. Re-run
# scripts/quality_calibration.py whenever TARGET_PERIOD, the scale loop or the
# crop strategies change — the thresholds are calibrated against them.
PIPELINE_VERSION = "1.2.0"

JARS_DIR = Path(__file__).resolve().parent.parent / "jars"


def sourceafis_version() -> str:
    """SourceAFIS version, parsed from the jar actually on the classpath."""
    for jar in sorted(JARS_DIR.glob("sourceafis-*.jar")):
        m = re.match(r"sourceafis-(\d+(?:\.\d+)*)\.jar$", jar.name)
        if m:
            return m.group(1)
    return "unknown"
