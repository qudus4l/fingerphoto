"""The harness end to end on the real corpus, plus a regression guard against a
checked-in baseline. Needs the JVM and the photos; warm cache makes it quick."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fingerphoto import evaluation

pytestmark = pytest.mark.slow

BASELINE = Path(__file__).parent / "baselines" / "eval_baseline.json"


@pytest.fixture(scope="module")
def report():
    if not evaluation.DEFAULT_MANIFEST.exists():
        pytest.skip("no corpus manifest on disk")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rep, _ = evaluation.run(out_dir=Path(tmp), log=None)
    return rep


def test_harness_runs_on_corpus(report):
    assert report["meta"]["n_photos"] >= 1
    assert report["set_max"]["n_genuine"] > 0
    assert all(p["status"] in ("ok", "fta") for p in report["photos"])
    # every acquired photo carries a quality score from the gate
    assert all(p["quality"] is not None for p in report["photos"] if p["status"] == "ok")


def test_writes_a_complete_report(report, tmp_path):
    d = evaluation.write_report(report, tmp_path)
    for name in ("report.json", "photos.csv", "pairs.csv", "loo.csv"):
        assert (d / name).exists(), name


def test_regression_against_baseline(report):
    """Bands, not exact numbers. A deliberate improvement is accepted by
    re-writing the baseline; an accidental collapse fails here first."""
    if not BASELINE.exists():
        pytest.skip("no baseline yet: scripts/evaluate.py --write-baseline")
    base = json.loads(BASELINE.read_text())
    old = {p["name"]: p for p in base["photos"]}

    regressions = []
    for p in report["photos"]:
        o = old.get(p["name"])
        if o is None or o["best_genuine"] is None or p["best_genuine"] is None:
            continue
        if p["best_genuine"] < o["best_genuine"] - max(15.0, 0.2 * o["best_genuine"]):
            regressions.append(f"{p['name']} genuine {o['best_genuine']} -> {p['best_genuine']}")
    assert not regressions, (
        "genuine scores collapsed: " + "; ".join(regressions)
        + " — accept intentionally with: "
        "python scripts/evaluate.py --write-baseline tests/baselines/eval_baseline.json"
    )

    if base["set_max"]["impostor"]["max"] is not None:
        assert report["set_max"]["impostor"]["max"] <= base["set_max"]["impostor"]["max"] + 5
    assert report["identification"]["correct"] >= base["identification"]["correct"] - 1
    assert report["fta"]["count"] <= base["fta"]["count"]
