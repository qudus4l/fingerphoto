"""Metric-layer tests for the evaluation harness. No JVM: afis.match_many and
identity.photo_template_set are monkeypatched, so these run in milliseconds."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from fingerphoto import afis, evaluation, identity, imaging
from fingerphoto.errors import AcquisitionError, ReasonCode


def row(name, subject="Q", finger="right_thumb", source="camera_app", **kw):
    return evaluation.ManifestRow(
        name=name, path=Path(f"/nonexistent/{name}.jpg"), subject=subject,
        finger=finger, source=source, **kw
    )


def result(name, subject="Q", finger="right_thumb", n=2, ok=True, **kw):
    templates = [f"{name}#{i}".encode() for i in range(n)] if ok else []
    return evaluation.PhotoResult(
        row=row(name, subject, finger, **kw),
        templates=templates,
        minutiae=[40] * len(templates),
        error_code=None if ok else "QUALITY_TOO_LOW",
        error=None if ok else "no resolvable ridge structure",
    )


@pytest.fixture()
def fake_match(monkeypatch):
    """match_many(probe, candidates) -> scores looked up by photo name."""
    table: dict[tuple[str, str], float] = {}

    def match_many(probe: bytes, candidates: list[bytes]) -> list[float]:
        a = probe.decode().split("#")[0]
        return [table.get((a, c.decode().split("#")[0]), 1.0) for c in candidates]

    monkeypatch.setattr(afis, "match_many", match_many)
    return table


# --- metrics ---------------------------------------------------------------


def test_eer_separable():
    e, t = evaluation.eer([80, 90, 100], [5, 10, 15])
    assert e == 0.0 and 15 < t < 80


def test_eer_overlap():
    e, _ = evaluation.eer([30, 50, 70], [10, 40, 60])
    assert abs(e - 1 / 3) < 0.02


def test_eer_empty_is_nan():
    e, t = evaluation.eer([], [1, 2])
    assert math.isnan(e) and math.isnan(t)


def test_fmr_fnmr_at_threshold():
    fmr, fnmr = evaluation.fmr_fnmr([30, 50, 70], [10, 45], 40)
    assert abs(fnmr - 1 / 3) < 1e-9 and fmr == 0.5
    fmr, fnmr = evaluation.fmr_fnmr([], [], 40)
    assert math.isnan(fmr) and math.isnan(fnmr)


def test_distribution_empty():
    assert evaluation.distribution([]) == {
        "n": 0, "min": None, "median": None, "mean": None, "max": None
    }


def test_score_stats_dprime():
    s = evaluation.score_stats([100, 100], [0, 0], 40)
    assert s["n_genuine"] == 2 and s["fnmr"] == 0.0 and s["fmr"] == 0.0
    assert s["dprime"] is None  # zero variance in both -> undefined, not a crash


# --- pairing ---------------------------------------------------------------


def test_score_pairs_keys_and_exclusions(fake_match):
    results = [
        result("q1", "Q", "right_thumb"),
        result("q2", "Q", "right_thumb"),
        result("s1", "Sister", "right_thumb"),
        result("s2", "Sister", "right_index"),
        result("probe", evaluation.UNKNOWN, evaluation.UNKNOWN),
        result("bad", "Q", "right_thumb", ok=False),
    ]
    pairs = evaluation.score_pairs(results)
    names = {(p.probe, p.gallery) for p in pairs}
    assert not any("probe" in n or "bad" in n for n in names)
    # 4 usable photos -> 12 ordered pairs; genuine only q1<->q2 (both directions)
    assert len(pairs) == 12
    assert sum(1 for p in pairs if p.genuine) == 2
    # same person, different finger is an impostor pair
    assert not next(p for p in pairs if p.probe == "s1" and p.gallery == "s2").genuine


def test_score_pairs_are_ordered_and_asymmetric(fake_match):
    fake_match[("q1", "q2")] = 90.0
    fake_match[("q2", "q1")] = 30.0
    pairs = {(p.probe, p.gallery): p.set_max
             for p in evaluation.score_pairs([result("q1"), result("q2")])}
    assert pairs[("q1", "q2")] == 90.0 and pairs[("q2", "q1")] == 30.0


def test_set_max_and_per_template_stats(fake_match):
    fake_match[("q1", "q2")] = fake_match[("q2", "q1")] = 90.0
    pairs = evaluation.score_pairs([result("q1"), result("q2"), result("s1", "Sister")])
    sm = evaluation.set_max_stats(pairs, 40)
    pt = evaluation.per_template_stats(pairs, 40)
    assert sm["genuine"]["max"] == 90.0 and sm["n_genuine"] == 2
    assert pt["n_genuine"] == 8  # 2 pairs x 2 probe templates x 2 gallery templates
    assert pt["impostor"]["max"] == 1.0


# --- identification --------------------------------------------------------


def test_leave_one_out_outcomes(fake_match):
    fake_match[("q1", "q2")] = 120.0          # correct
    fake_match[("s1", "q1")] = fake_match[("s1", "q2")] = 80.0   # wrong (beats its own mate)
    fake_match[("s1", "s2")] = 5.0
    results = [
        result("q1", "Q"), result("q2", "Q"),
        result("s1", "Sister"), result("s2", "Sister"),
        result("lone", "Lone"),
    ]
    loo = {x["name"]: x for x in evaluation.leave_one_out(results, 40, {"fvc-101": [b"fvc#0"]})}
    assert loo["q1"]["outcome"] == "correct" and loo["q1"]["rank"] == 1
    assert loo["s1"]["outcome"] == "wrong" and loo["s1"]["predicted"] == "Q/right_thumb"
    assert loo["q2"]["outcome"] == "rejected"      # only scores 1.0 against everything
    assert loo["lone"]["outcome"] == "no_mate" and loo["lone"]["rank"] is None
    assert "fvc-101" in loo["q1"]["scores"]

    summary = evaluation.identification_summary(list(loo.values()))
    assert summary["correct"] == 1 and summary["wrong"] == 1 and summary["no_mate"] == 1
    assert summary["n_with_mate"] == 4


def test_probe_unknowns_reported_not_rated(fake_match):
    fake_match[("p1", "q1")] = 150.0
    results = [result("q1", "Q"), result("p1", evaluation.UNKNOWN, evaluation.UNKNOWN),
               result("p2", evaluation.UNKNOWN, evaluation.UNKNOWN, ok=False)]
    rows = {u["name"]: u for u in evaluation.probe_unknowns(results, 40)}
    assert rows["p1"]["matched"] is True and rows["p1"]["best"] == "Q/right_thumb"
    assert rows["p2"]["matched"] is None and rows["p2"]["error_code"] == "QUALITY_TOO_LOW"


# --- cache -----------------------------------------------------------------


def test_cache_key_sensitivity():
    base = evaluation.extraction_settings(True, 15, None)
    key = evaluation.cache_key(b"photo", base)
    assert key == evaluation.cache_key(b"photo", dict(base))
    assert key != evaluation.cache_key(b"other photo", base)
    for field, value in (("pipeline_version", "9.9.9"), ("code_hash", "deadbeef"),
                         ("crop", False), ("min_ridge_quality", 0.35), ("min_minutiae", 20)):
        assert key != evaluation.cache_key(b"photo", {**base, field: value}), field


def test_extract_cached_roundtrip(tmp_path, monkeypatch):
    photo = tmp_path / "p.jpg"
    photo.write_bytes(b"not really a jpeg")
    calls = []

    def fake_set(bgr, **kw):
        calls.append(kw)
        kw["stats"]["quality"] = 0.8
        return [b"t0", b"t1"]

    monkeypatch.setattr(imaging, "decode_image", lambda d, n="": np.zeros((8, 8, 3), np.uint8))
    monkeypatch.setattr(identity, "photo_template_set", fake_set)
    monkeypatch.setattr(afis, "minutiae_count", lambda t: 42)

    r = evaluation.ManifestRow("p", photo, "Q", "right_thumb", "camera_app")
    first = evaluation.extract_cached(r, tmp_path / "cache")
    assert first.cached is False and first.templates == [b"t0", b"t1"]
    assert first.minutiae == [42, 42] and first.stats["quality"] == 0.8
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 1

    second = evaluation.extract_cached(r, tmp_path / "cache")
    assert second.cached is True and second.templates == first.templates
    assert len(calls) == 1

    evaluation.extract_cached(r, tmp_path / "cache", use_cache=False)
    assert len(calls) == 2


def test_extract_cached_caches_failures(tmp_path, monkeypatch):
    photo = tmp_path / "bad.jpg"
    photo.write_bytes(b"x")
    calls = []

    def fake_set(bgr, **kw):
        calls.append(1)
        raise AcquisitionError(ReasonCode.QUALITY_TOO_LOW, "no resolvable ridge structure")

    monkeypatch.setattr(imaging, "decode_image", lambda d, n="": np.zeros((8, 8, 3), np.uint8))
    monkeypatch.setattr(identity, "photo_template_set", fake_set)

    r = evaluation.ManifestRow("bad", photo, "Q", "right_thumb", "web_burst")
    first = evaluation.extract_cached(r, tmp_path / "cache")
    assert first.ok is False and first.error_code == "QUALITY_TOO_LOW"
    second = evaluation.extract_cached(r, tmp_path / "cache")
    assert second.cached is True and second.error_code == "QUALITY_TOO_LOW"
    assert len(calls) == 1  # a rejected photo is not re-extracted on every run


# --- manifest --------------------------------------------------------------


def _manifest(tmp_path, body: str) -> Path:
    (tmp_path / "a.jpg").write_bytes(b"a")
    (tmp_path / "b.jpg").write_bytes(b"b")
    p = tmp_path / "manifest.csv"
    p.write_text("path,subject,finger,source,gallery,crop,notes\n" + body)
    return p


def test_load_manifest_defaults(tmp_path):
    p = _manifest(tmp_path, "a.jpg,Q,right_thumb,camera_app,1,,first\n"
                            "b.jpg,unknown,unknown,web_burst,0,0,probe\n"
                            "#skip.jpg,X,y,z,1,1,comment\n")
    rows = load = evaluation.load_manifest(p, root=tmp_path)
    assert [r.name for r in rows] == ["a", "b"]
    assert rows[0].gallery is True and rows[0].crop is True
    assert rows[0].key == "Q/right_thumb"
    assert rows[1].gallery is False and rows[1].crop is False and rows[1].key is None
    assert load[0].path == (tmp_path / "a.jpg").resolve()


def test_load_manifest_rejects_duplicates_and_missing(tmp_path):
    p = _manifest(tmp_path, "a.jpg,Q,right_thumb,camera_app,1,1,\n"
                            "a.jpg,Q,right_thumb,camera_app,1,1,dupe\n")
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.load_manifest(p, root=tmp_path)

    p = _manifest(tmp_path, "nope.jpg,Q,right_thumb,camera_app,1,1,\n")
    with pytest.raises(ValueError, match="missing"):
        evaluation.load_manifest(p, root=tmp_path)


# --- report ----------------------------------------------------------------


def _small_report(fake_match) -> dict:
    fake_match[("q1", "q2")] = fake_match[("q2", "q1")] = 120.0
    fake_match[("s1", "q1")] = 55.0     # false match: Sister scores over threshold vs Q
    fake_match[("s1", "s2")] = 90.0     # ...while still matching its own mate
    results = [result("q1", "Q"), result("q2", "Q"), result("s1", "Sister"),
               result("s2", "Sister"), result("probe", evaluation.UNKNOWN, evaluation.UNKNOWN),
               result("bad", "Q", ok=False)]
    pairs = evaluation.score_pairs(results)
    loo = evaluation.leave_one_out(results, 40, {})
    unknowns = evaluation.probe_unknowns(results, 40)
    return evaluation.build_report(results, pairs, loo, unknowns, 40, Path("m.csv"), [],
                                   evaluation.extraction_settings(True, 15, None))


def test_build_and_write_report(tmp_path, fake_match):
    report = _small_report(fake_match)
    verdicts = {p["name"]: p["verdict"] for p in report["photos"]}
    assert verdicts["q1"] == "pass"
    assert verdicts["bad"] == "fta"
    assert verdicts["probe"] == "probe"
    assert verdicts["s1"] == "false_match"
    assert verdicts["s2"] == "false_nonmatch"
    assert report["fta"]["count"] == 1
    assert report["meta"]["n_unknown"] == 1 and report["meta"]["n_labelled"] == 5

    d = evaluation.write_report(report, tmp_path)
    for f in ("report.json", "photos.csv", "pairs.csv", "loo.csv"):
        assert (d / f).exists(), f
    assert (tmp_path / "latest.json").exists()
    assert json.loads((d / "report.json").read_text())["meta"]["threshold"] == 50
    assert evaluation.format_report(report, per_template=True)


def test_strip_for_baseline(fake_match):
    slim = evaluation.strip_for_baseline(_small_report(fake_match))
    assert all("single" not in p for p in slim["pairs"])
    assert all("scores" not in x for x in slim["loo"])
    assert slim["set_max"]["genuine"]["max"] == 120.0


def test_compare_deltas(fake_match):
    prev = _small_report(fake_match)
    curr = json.loads(json.dumps(prev))
    curr["set_max"]["genuine"]["min"] = 20.0
    curr["fta"]["count"] = 3
    deltas = {d["metric"]: d for d in evaluation.compare(prev, curr)}
    assert deltas["set_max.genuine.min"]["flag"] == "!"
    assert deltas["fta.count"]["delta"] == 2 and deltas["fta.count"]["flag"] == "!"
    assert deltas["set_max.genuine.max"]["delta"] == 0 and deltas["set_max.genuine.max"]["flag"] == ""
    assert "photo:q1.best_genuine" in deltas
