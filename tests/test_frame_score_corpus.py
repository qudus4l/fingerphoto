"""The cheap frame scorer must agree with the full ridge-quality gate.

Selection happens before extraction, so if the scorer and the gate ever
disagree the selector will either throw away usable frames or spend 15 s
extracting frames the gate then refuses. This pins them together on the real
corpus. No JVM: the scorer stops before template extraction.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from fingerphoto import frames, imaging

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "samples" / "manifest.csv"

# Verdicts measured by scripts/evaluate.py at pipeline 1.2.0 (out/eval/latest.json).
ACCEPTED = {
    "thumb_6172", "thumb_6173", "thumb_6174", "thumb_6217", "sister_6214",
    "fingertip_macro", "20260831-211132_match-194",
}
REFUSED = {
    "sister_6175", "sister_6176", "sis_finger_a", "sis_finger_b",
    "web_w_capture", "web_qd_capture", "20260831-193348_nomatch-6",
    "20260831-195511_nomatch-17", "20260831-202624_nomatch-26",
    "20260831-203000_match-64",
}
ACCEPT_FLOOR = 0.60   # measured minimum over ACCEPTED is 0.750
REFUSE_CEILING = 0.40  # measured maximum over REFUSED is 0.333 (= the gate)

FIXME = "re-run scripts/quality_calibration.py before changing quality.py or frames.py"


def _rows():
    if not MANIFEST.exists():
        pytest.skip(f"{MANIFEST} not on disk (local corpus)", allow_module_level=True)
    out = []
    for r in csv.DictReader(MANIFEST.open()):
        if r["path"].startswith("#"):
            continue
        p = ROOT / r["path"]
        if p.exists():
            out.append((p.stem, p))
    return out


@pytest.mark.parametrize("stem,path", _rows(), ids=lambda v: v if isinstance(v, str) else "")
def test_scorer_matches_the_gate(stem, path):
    if stem not in ACCEPTED and stem not in REFUSED:
        pytest.skip(f"{stem} has no recorded gate verdict")
    score, strategy, _ = frames.frame_score(imaging.decode_image(path.read_bytes()))
    if stem in ACCEPTED:
        assert score >= ACCEPT_FLOOR, (
            f"{stem} is accepted by the gate but the frame scorer gives {score:.3f}"
            f" (< {ACCEPT_FLOOR}); selection would discard usable frames. {FIXME}"
        )
        assert strategy in ("distal", "pad")
    else:
        assert score <= REFUSE_CEILING, (
            f"{stem} is refused by the gate but the frame scorer gives {score:.3f}"
            f" (> {REFUSE_CEILING}); selection would spend a full extraction on it. {FIXME}"
        )
