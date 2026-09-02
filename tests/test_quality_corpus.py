"""The ridge gate on the real labelled crops: every crop known to yield matching
templates must pass, every crop with no resolvable ridges must be refused.

This is the regression that protects the calibration in fingerphoto/quality.py.
No JVM (no templates are extracted), but it runs the Gabor stage on ~23 crops."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from fingerphoto import evaluation

pytestmark = pytest.mark.slow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.fixture(scope="module")
def rows():
    if not evaluation.DEFAULT_MANIFEST.exists():
        pytest.skip("no corpus manifest on disk")
    import quality_calibration as qc

    out = qc.collect()
    good = [r for r in out if r["label"] == qc.GOOD]
    bad = [r for r in out if r["label"] == qc.BAD]
    if len(good) < 4 or len(bad) < 4:
        pytest.skip("too few labelled crops resolved from the corpus")
    return out


def test_every_good_crop_passes(rows):
    import quality_calibration as qc

    failed = [f"{r['name']}/{r['strategy']} score {r['score']:.3f} ({r['sanity']})"
              for r in rows if r["label"] == qc.GOOD and r["gate"] != "pass"]
    assert not failed, "gate refused a usable crop: " + "; ".join(failed)


def test_every_bad_crop_is_refused(rows):
    import quality_calibration as qc

    passed = [f"{r['name']}/{r['strategy']} score {r['score']:.3f}"
              for r in rows if r["label"] == qc.BAD and r["gate"] != "reject"]
    assert not passed, "gate accepted a crop with no ridges: " + "; ".join(passed)


def test_scores_separate_with_margin(rows):
    import quality_calibration as qc

    good = min(r["score"] for r in rows if r["label"] == qc.GOOD)
    bad = max(r["score"] for r in rows if r["label"] == qc.BAD)
    assert good > bad, f"score overlap: good min {good:.3f} <= bad max {bad:.3f}"
    assert good - bad > 0.2, f"separation margin collapsed to {good - bad:.3f}"


def test_spectral_factor_is_load_bearing(rows):
    """Dropping the spectral term lets whole-hand crops through — the reason it
    is in the product at all. If this ever stops failing, re-check the labels."""
    import quality_calibration as qc

    good = min(r["score_nospec"] for r in rows if r["label"] == qc.GOOD)
    bad = max(r["score_nospec"] for r in rows if r["label"] == qc.BAD)
    assert good <= bad, (
        "coherence and edges alone now separate the corpus; USE_SPECTRAL could be "
        "reconsidered (and quality.py's docstring updated)"
    )
