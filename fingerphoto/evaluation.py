"""Measurement harness: labelled photo corpus -> genuine/impostor statistics.

Every pipeline change needs a number, not an anecdote. This module turns a CSV
manifest of labelled photos into template sets (cached, because extraction costs
12-24 s per photo while matching costs milliseconds), then reports:

  * all-pairs genuine and impostor score distributions, FMR/FNMR at a threshold,
    and the EER with the threshold that achieves it
  * leave-one-out 1:N identification against the other photos of every subject
    plus optional FVC contact-print distractors
  * failure-to-acquire (photos the pipeline refuses) reported SEPARATELY from
    false non-match - a rejected photo is not a matching failure
  * set-max versus per-template scores, so the inflation the set-max rule adds
    to impostor scores is visible (measured 31.4 vs 16.6 at pipeline 1.1.0)

Identity key is "<subject>/<finger>": two photos are a genuine pair only when
both agree, so one person's index and thumb are impostors of each other.
Photos labelled subject=unknown are scored against the gallery and reported,
but never counted in any rate.

Scores are asymmetric in SourceAFIS, so pairs are ORDERED (probe -> gallery).
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import afis, identity, imaging, version
from .errors import AcquisitionError

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "data" / "samples" / "manifest.csv"
DEFAULT_OUT = ROOT / "out" / "eval"
DEFAULT_CACHE = ROOT / "out" / "eval" / "cache"
FVC_DIR = ROOT / "data" / "fvc"

CACHE_FORMAT = 1
UNKNOWN = "unknown"
# Any edit to these invalidates cached templates even without a version bump.
PIPELINE_SOURCES = (
    "afis.py", "identity.py", "locate.py", "pipeline.py",
    "quality.py", "scale.py", "segment.py",
)


# --- manifest ---------------------------------------------------------------


@dataclass
class ManifestRow:
    name: str
    path: Path
    subject: str
    finger: str
    source: str
    gallery: bool = False
    crop: bool = True
    notes: str = ""

    @property
    def key(self) -> str | None:
        """Identity key, or None for an unlabelled probe."""
        if self.subject == UNKNOWN or self.finger == UNKNOWN:
            return None
        return f"{self.subject}/{self.finger}"


def _flag(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "y")


def load_manifest(path: Path, root: Path = ROOT) -> list[ManifestRow]:
    """Parse the corpus manifest. Raises ValueError on duplicate names or
    missing files, so a typo fails loudly instead of silently shrinking the
    evaluation."""
    rows: list[ManifestRow] = []
    seen: dict[str, Path] = {}
    missing: list[str] = []
    duplicate: list[str] = []
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            rel = (raw.get("path") or "").strip()
            if not rel or rel.startswith("#"):
                continue
            p = (root / rel).resolve()
            name = Path(rel).stem
            if name in seen:
                duplicate.append(name)
                continue
            seen[name] = p
            if not p.exists():
                missing.append(rel)
                continue
            rows.append(
                ManifestRow(
                    name=name,
                    path=p,
                    subject=(raw.get("subject") or UNKNOWN).strip() or UNKNOWN,
                    finger=(raw.get("finger") or UNKNOWN).strip() or UNKNOWN,
                    source=(raw.get("source") or "").strip(),
                    gallery=_flag(raw.get("gallery"), False),
                    crop=_flag(raw.get("crop"), True),
                    notes=(raw.get("notes") or "").strip(),
                )
            )
    problems = []
    if duplicate:
        problems.append(f"duplicate photo names: {sorted(set(duplicate))}")
    if missing:
        problems.append(f"missing files: {missing}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return rows


# --- extraction cache -------------------------------------------------------


@dataclass
class PhotoResult:
    row: ManifestRow
    templates: list[bytes] = field(default_factory=list)
    minutiae: list[int | None] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    error_code: str | None = None
    error: str | None = None
    extract_s: float = 0.0
    cached: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.templates)


def code_hash() -> str:
    """Fingerprint of the pipeline source, so a code edit invalidates the cache."""
    h = hashlib.sha256()
    for name in PIPELINE_SOURCES:
        p = Path(__file__).resolve().parent / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def extraction_settings(
    crop: bool,
    min_minutiae: int = identity.MIN_MINUTIAE,
    min_ridge_quality: float | None = None,
) -> dict:
    return {
        "format": CACHE_FORMAT,
        "pipeline_version": version.PIPELINE_VERSION,
        "sourceafis_version": version.sourceafis_version(),
        "code_hash": code_hash(),
        "crop": bool(crop),
        "sweep": list(identity.SCALE_SWEEP),
        "min_minutiae": min_minutiae,
        "min_ridge_quality": min_ridge_quality,
    }


def cache_key(photo_bytes: bytes, settings: dict) -> str:
    a = hashlib.sha256(photo_bytes).hexdigest()[:16]
    b = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    return f"{a}-{b}"


def extract_cached(
    row: ManifestRow,
    cache_dir: Path = DEFAULT_CACHE,
    use_cache: bool = True,
    min_minutiae: int = identity.MIN_MINUTIAE,
    min_quality: float | None = None,
    log=None,
) -> PhotoResult:
    """Template set for one photo, memoized on disk.

    Failures are cached too: a photo the pipeline refuses must not cost 20 s on
    every run just to be refused again."""
    data = row.path.read_bytes()
    settings = extraction_settings(row.crop, min_minutiae, min_quality)
    path = Path(cache_dir) / f"{cache_key(data, settings)}.pkl"

    if use_cache and path.exists():
        try:
            blob = pickle.loads(path.read_bytes())
        except Exception:  # corrupt cache entry: re-extract rather than crash
            blob = None
        if blob is not None:
            result = PhotoResult(
                row=row,
                templates=blob["templates"],
                minutiae=blob["minutiae"],
                stats=blob.get("stats", {}),
                error_code=blob.get("error_code"),
                error=blob.get("error"),
                extract_s=blob.get("extract_s", 0.0),
                cached=True,
            )
            _log_result(result, log)
            return result

    stats: dict = {}
    kwargs = {"crop": row.crop, "min_minutiae": min_minutiae, "stats": stats}
    if min_quality is not None:
        kwargs["min_quality"] = min_quality
    t0 = time.monotonic()
    templates: list[bytes] = []
    error_code = error = None
    try:
        bgr = imaging.decode_image(data, row.path.name)
        templates = identity.photo_template_set(bgr, **kwargs)
    except AcquisitionError as e:
        error_code, error = e.code.value, str(e)
    extract_s = round(time.monotonic() - t0, 2)
    minutiae = [afis.minutiae_count(t) for t in templates]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        pickle.dumps(
            {
                "format": CACHE_FORMAT,
                "photo": str(row.path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "settings": settings,
                "templates": templates,
                "minutiae": minutiae,
                "stats": stats,
                "error_code": error_code,
                "error": error,
                "extract_s": extract_s,
                "created_at": time.time(),
            }
        )
    )
    result = PhotoResult(
        row=row, templates=templates, minutiae=minutiae, stats=stats,
        error_code=error_code, error=error, extract_s=extract_s, cached=False,
    )
    _log_result(result, log)
    return result


def _log_result(result: PhotoResult, log) -> None:
    if log is None:
        return
    when = "cached" if result.cached else f"{result.extract_s:5.1f}s"
    quality = result.stats.get("quality")
    q = f" q={quality:.2f}" if isinstance(quality, (int, float)) else ""
    if result.error_code:
        log(f"  {result.row.name:28s} {when}  REJECTED {result.error_code}{q}")
    else:
        best = max([m for m in result.minutiae if m is not None], default=0)
        log(f"  {result.row.name:28s} {when}  {len(result.templates)} tmpl, "
            f"{best} minutiae{q}")


def fvc_distractors(fvc_dir: Path = FVC_DIR, fingers=range(101, 111)) -> dict[str, list[bytes]]:
    """FVC2000 contact prints as 1:N gallery distractors (same recipe as
    scripts/fvc_benchmark.py: CLAHE then extract at a known 500 dpi)."""
    import cv2

    out: dict[str, list[bytes]] = {}
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(15, 15))
    for n in fingers:
        p = Path(fvc_dir) / f"{n}_1.tif"
        if not p.exists():
            continue
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        out[f"fvc-{n}"] = [afis.extract_template(clahe.apply(gray), dpi=500)]
    return out


# --- scoring ----------------------------------------------------------------


@dataclass
class PairScore:
    probe: str
    gallery: str
    genuine: bool
    set_max: float
    single: list[float]
    n_probe: int
    n_gallery: int


def score_pairs(results: list[PhotoResult]) -> list[PairScore]:
    """Ordered pairs over labelled, acquired photos. SourceAFIS scores are not
    symmetric, so (a -> b) and (b -> a) are both evaluated."""
    usable = [r for r in results if r.ok and r.row.key is not None]
    pairs: list[PairScore] = []
    for a in usable:
        for b in usable:
            if a is b:
                continue
            single = [s for t in a.templates for s in afis.match_many(t, b.templates)]
            pairs.append(
                PairScore(
                    probe=a.row.name,
                    gallery=b.row.name,
                    genuine=a.row.key == b.row.key,
                    set_max=float(max(single)),
                    single=[float(s) for s in single],
                    n_probe=len(a.templates),
                    n_gallery=len(b.templates),
                )
            )
    return pairs


def fmr_fnmr(genuine, impostor, threshold: float) -> tuple[float, float]:
    g, i = np.asarray(genuine, dtype=float), np.asarray(impostor, dtype=float)
    fmr = float((i >= threshold).mean()) if i.size else float("nan")
    fnmr = float((g < threshold).mean()) if g.size else float("nan")
    return fmr, fnmr


def eer(genuine, impostor, steps: int = 2000) -> tuple[float, float]:
    """Equal error rate and the threshold that achieves it."""
    g, i = np.asarray(genuine, dtype=float), np.asarray(impostor, dtype=float)
    if not g.size or not i.size:
        return float("nan"), float("nan")
    thresholds = np.linspace(0.0, float(max(g.max(), i.max())) + 1.0, steps)
    frr = np.array([(g < t).mean() for t in thresholds])
    far = np.array([(i >= t).mean() for t in thresholds])
    k = int(np.argmin(np.abs(frr - far)))
    return float((frr[k] + far[k]) / 2), float(thresholds[k])


def distribution(x) -> dict:
    a = np.asarray(x, dtype=float)
    if not a.size:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": int(a.size),
        "min": round(float(a.min()), 2),
        "median": round(float(np.median(a)), 2),
        "mean": round(float(a.mean()), 2),
        "max": round(float(a.max()), 2),
    }


def score_stats(genuine: list[float], impostor: list[float], threshold: float) -> dict:
    g, i = np.asarray(genuine, dtype=float), np.asarray(impostor, dtype=float)
    fmr, fnmr = fmr_fnmr(g, i, threshold)
    e, e_t = eer(g, i)
    dprime = None
    if g.size and i.size:
        denom = float(np.sqrt((g.var() + i.var()) / 2))
        if denom > 0:
            dprime = round(float(abs(g.mean() - i.mean()) / denom), 2)
    return {
        "n_genuine": int(g.size),
        "n_impostor": int(i.size),
        "genuine": distribution(g),
        "impostor": distribution(i),
        "fmr": round(fmr, 4) if fmr == fmr else None,
        "fnmr": round(fnmr, 4) if fnmr == fnmr else None,
        "eer": round(e, 4) if e == e else None,
        "eer_threshold": round(e_t, 2) if e_t == e_t else None,
        "dprime": dprime,
    }


def set_max_stats(pairs: list[PairScore], threshold: float) -> dict:
    return score_stats(
        [p.set_max for p in pairs if p.genuine],
        [p.set_max for p in pairs if not p.genuine],
        threshold,
    )


def per_template_stats(pairs: list[PairScore], threshold: float) -> dict:
    return score_stats(
        [s for p in pairs if p.genuine for s in p.single],
        [s for p in pairs if not p.genuine for s in p.single],
        threshold,
    )


def _gallery_score(probe_templates: list[bytes], gallery_templates: list[bytes]) -> float:
    return float(max(max(afis.match_many(t, gallery_templates)) for t in probe_templates))


def leave_one_out(
    results: list[PhotoResult], threshold: float, distractors: dict[str, list[bytes]] | None = None
) -> list[dict]:
    """1:N identification with the probe's own photo held out of the gallery."""
    usable = [r for r in results if r.ok and r.row.key is not None]
    rows: list[dict] = []
    for probe in usable:
        gallery: dict[str, list[bytes]] = {k: list(v) for k, v in (distractors or {}).items()}
        for other in usable:
            if other is probe:
                continue
            gallery.setdefault(other.row.key, []).extend(other.templates)
        if not gallery:
            continue
        scores = {k: _gallery_score(probe.templates, ts) for k, ts in gallery.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        predicted, top = ranked[0]
        key = probe.row.key
        if key not in gallery:
            outcome, rank = "no_mate", None
        else:
            rank = 1 + [k for k, _ in ranked].index(key)
            if top < threshold:
                outcome = "rejected"
            elif predicted == key:
                outcome = "correct"
            else:
                outcome = "wrong"
        rows.append(
            {
                "name": probe.row.name,
                "key": key,
                "predicted": predicted,
                "score": round(top, 1),
                "genuine_score": round(scores[key], 1) if key in scores else None,
                "rank": rank,
                "outcome": outcome,
                "scores": {k: round(v, 1) for k, v in ranked},
            }
        )
    return rows


def probe_unknowns(results: list[PhotoResult], threshold: float) -> list[dict]:
    """Unlabelled probes against the full labelled gallery: reported, never
    counted in a rate (their true identity is unknown)."""
    gallery: dict[str, list[bytes]] = {}
    for r in results:
        if r.ok and r.row.key is not None:
            gallery.setdefault(r.row.key, []).extend(r.templates)
    rows: list[dict] = []
    for r in results:
        if r.row.key is not None:
            continue
        if not r.ok:
            rows.append({"name": r.row.name, "source": r.row.source, "best": None,
                         "score": None, "matched": None, "error_code": r.error_code,
                         "scores": {}})
            continue
        if not gallery:
            continue
        scores = {k: _gallery_score(r.templates, ts) for k, ts in gallery.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best, top = ranked[0]
        rows.append(
            {
                "name": r.row.name,
                "source": r.row.source,
                "best": best,
                "score": round(top, 1),
                "matched": bool(top >= threshold),
                "error_code": None,
                "scores": {k: round(v, 1) for k, v in ranked},
            }
        )
    return rows


def identification_summary(loo: list[dict]) -> dict:
    with_mate = [x for x in loo if x["outcome"] != "no_mate"]
    counts = {k: sum(1 for x in loo if x["outcome"] == k)
              for k in ("correct", "rejected", "wrong", "no_mate")}
    rank1 = sum(1 for x in with_mate if x["rank"] == 1)
    n = len(with_mate)
    return {
        "n_probes": len(loo),
        "n_with_mate": n,
        **counts,
        "rank1": rank1,
        "rank1_rate": round(counts["correct"] / n, 4) if n else None,
        "rank1_rate_any_threshold": round(rank1 / n, 4) if n else None,
    }


def per_photo_rows(
    results: list[PhotoResult], pairs: list[PairScore], loo: list[dict], threshold: float
) -> list[dict]:
    by_loo = {x["name"]: x for x in loo}
    rows: list[dict] = []
    for r in results:
        name = r.row.name
        gen = [(p.set_max, p.gallery) for p in pairs if p.probe == name and p.genuine]
        imp = [(p.set_max, p.gallery) for p in pairs if p.probe == name and not p.genuine]
        best_gen = max(gen, default=(None, None))
        best_imp = max(imp, default=(None, None))
        l = by_loo.get(name, {})
        if not r.ok:
            verdict = "fta"
        elif r.row.key is None:
            verdict = "probe"
        elif best_gen[0] is None:
            verdict = "no_mate"
        else:
            bad = []
            if best_gen[0] < threshold:
                bad.append("false_nonmatch")
            if best_imp[0] is not None and best_imp[0] >= threshold:
                bad.append("false_match")
            verdict = "+".join(bad) if bad else "pass"
        rows.append(
            {
                "name": name,
                "subject": r.row.subject,
                "finger": r.row.finger,
                "key": r.row.key,
                "source": r.row.source,
                "gallery": r.row.gallery,
                "status": "ok" if r.ok else "fta",
                "error_code": r.error_code,
                "templates": len(r.templates),
                "minutiae_max": max([m for m in r.minutiae if m is not None], default=None),
                "minutiae": r.minutiae,
                "quality": r.stats.get("quality"),
                "rejections": r.stats.get("rejections"),
                "extract_s": r.extract_s,
                "cached": r.cached,
                "best_genuine": round(best_gen[0], 1) if best_gen[0] is not None else None,
                "best_genuine_vs": best_gen[1],
                "best_impostor": round(best_imp[0], 1) if best_imp[0] is not None else None,
                "best_impostor_vs": best_imp[1],
                "loo_predicted": l.get("predicted"),
                "loo_score": l.get("score"),
                "loo_rank": l.get("rank"),
                "loo_outcome": l.get("outcome"),
                "verdict": verdict,
            }
        )
    return rows


def build_report(
    results: list[PhotoResult],
    pairs: list[PairScore],
    loo: list[dict],
    unknowns: list[dict],
    threshold: float,
    manifest: Path,
    distractor_names: list[str],
    settings: dict,
) -> dict:
    labelled = [r for r in results if r.row.key is not None]
    fta = [r for r in results if not r.ok]
    sm = set_max_stats(pairs, threshold)
    pt = per_template_stats(pairs, threshold)
    if sm["impostor"]["max"] is not None and pt["impostor"]["max"] is not None:
        pt["inflation_max_impostor"] = round(sm["impostor"]["max"] - pt["impostor"]["max"], 2)
    else:
        pt["inflation_max_impostor"] = None
    return {
        "meta": {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pipeline_version": version.PIPELINE_VERSION,
            "sourceafis_version": version.sourceafis_version(),
            "code_hash": settings.get("code_hash"),
            "threshold": threshold,
            "min_ridge_quality": settings.get("min_ridge_quality"),
            "min_minutiae": settings.get("min_minutiae"),
            "manifest": str(manifest),
            "n_photos": len(results),
            "n_labelled": len(labelled),
            "n_unknown": len(results) - len(labelled),
            "cache_hits": sum(1 for r in results if r.cached),
            "extract_total_s": round(sum(r.extract_s for r in results), 1),
            "distractors": distractor_names,
        },
        "photos": per_photo_rows(results, pairs, loo, threshold),
        "pairs": [asdict(p) for p in pairs],
        "set_max": sm,
        "per_template": pt,
        "identification": identification_summary(loo),
        "loo": loo,
        "unknown_probes": unknowns,
        "fta": {
            "count": len(fta),
            "rate": round(len(fta) / len(labelled), 4) if labelled else None,
            "photos": [{"name": r.row.name, "error_code": r.error_code, "error": r.error}
                       for r in fta],
        },
    }


# --- output -----------------------------------------------------------------


def _csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_report(report: dict, out_root: Path = DEFAULT_OUT) -> Path:
    out_root = Path(out_root)
    d = out_root / f"{time.strftime('%Y%m%d-%H%M%S')}_{report['meta']['pipeline_version']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(report, indent=1, default=str))
    _csv(d / "photos.csv", report["photos"],
         [c for c in report["photos"][0] if c != "minutiae"] if report["photos"] else ["name"])
    _csv(
        d / "pairs.csv",
        [
            {
                "probe": p["probe"], "gallery": p["gallery"], "genuine": p["genuine"],
                "set_max": round(p["set_max"], 1),
                "single_max": round(max(p["single"]), 1) if p["single"] else None,
                "single_median": round(float(np.median(p["single"])), 1) if p["single"] else None,
                "n_probe": p["n_probe"], "n_gallery": p["n_gallery"],
            }
            for p in report["pairs"]
        ],
        ["probe", "gallery", "genuine", "set_max", "single_max", "single_median",
         "n_probe", "n_gallery"],
    )
    _csv(d / "loo.csv", report["loo"],
         ["name", "key", "predicted", "score", "genuine_score", "rank", "outcome"])
    shutil.copyfile(d / "report.json", out_root / "latest.json")
    return d


def strip_for_baseline(report: dict) -> dict:
    """Report without the bulky per-template score lists, for checking in."""
    out = json.loads(json.dumps(report, default=str))
    for p in out.get("pairs", []):
        p.pop("single", None)
    for x in out.get("loo", []):
        x.pop("scores", None)
    for x in out.get("unknown_probes", []):
        x.pop("scores", None)
    return out


def _dig(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


COMPARE_SCORE_METRICS = (
    "set_max.genuine.min", "set_max.genuine.median", "set_max.genuine.max",
    "set_max.impostor.median", "set_max.impostor.max",
    "set_max.eer_threshold", "per_template.impostor.max",
)
COMPARE_RATE_METRICS = ("set_max.fmr", "set_max.fnmr", "set_max.eer",
                        "identification.rank1_rate")
COMPARE_COUNT_METRICS = ("identification.correct", "identification.rejected",
                         "identification.wrong", "fta.count")


def compare(prev: dict, curr: dict) -> list[dict]:
    rows: list[dict] = []

    def add(metric: str, a, b, tol: float) -> None:
        delta = None
        flag = ""
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = round(b - a, 4)
            if abs(delta) > tol:
                flag = "!"
        elif a != b:
            flag = "!"
        rows.append({"metric": metric, "prev": a, "curr": b, "delta": delta, "flag": flag})

    for m in COMPARE_SCORE_METRICS:
        add(m, _dig(prev, m), _dig(curr, m), 10.0)
    for m in COMPARE_RATE_METRICS:
        add(m, _dig(prev, m), _dig(curr, m), 0.05)
    for m in COMPARE_COUNT_METRICS:
        add(m, _dig(prev, m), _dig(curr, m), 0.5)
    prev_photos = {p["name"]: p for p in prev.get("photos", [])}
    for p in curr.get("photos", []):
        old = prev_photos.get(p["name"])
        if old is None:
            continue
        add(f"photo:{p['name']}.best_genuine", old.get("best_genuine"), p.get("best_genuine"), 10.0)
        add(f"photo:{p['name']}.best_impostor", old.get("best_impostor"), p.get("best_impostor"), 10.0)
    return rows


def _fmt(value, width: int = 7, decimals: int = 1) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, float):
        return f"{value:{width}.{decimals}f}"
    return str(value).rjust(width)


def format_report(report: dict, per_template: bool = False, deltas: list[dict] | None = None) -> str:
    m = report["meta"]
    out: list[str] = []
    out.append(
        f"pipeline {m['pipeline_version']}  sourceafis {m['sourceafis_version']}  "
        f"code {m['code_hash']}  threshold {m['threshold']}  "
        f"min_ridge_quality {m['min_ridge_quality']}"
    )
    out.append(
        f"{m['n_photos']} photos ({m['n_labelled']} labelled, {m['n_unknown']} unknown), "
        f"{m['cache_hits']} cached, {m['extract_total_s']}s extracting, "
        f"{len(m['distractors'])} distractors"
    )

    out.append("")
    out.append(f"{'photo':28s} {'subject':10s} {'finger':12s} {'source':13s} "
               f"{'tmpl':>4s} {'minu':>4s} {'qual':>5s} {'genuine':>8s} {'impostor':>8s} "
               f"{'loo':>12s} verdict")
    for p in sorted(report["photos"], key=lambda p: (p["subject"], p["name"])):
        loo = f"{p['loo_outcome'] or '-'}"
        out.append(
            f"{p['name'][:28]:28s} {p['subject'][:10]:10s} {p['finger'][:12]:12s} "
            f"{(p['source'] or '')[:13]:13s} {p['templates']:4d} "
            f"{_fmt(p['minutiae_max'], 4, 0)} {_fmt(p['quality'], 5, 2)} "
            f"{_fmt(p['best_genuine'], 8)} {_fmt(p['best_impostor'], 8)} {loo:>12s} "
            f"{p['verdict']}{'  ' + (p['error_code'] or '') if p['status'] == 'fta' else ''}"
        )

    def block(title: str, s: dict) -> None:
        out.append("")
        out.append(f"{title}:")
        g, i = s["genuine"], s["impostor"]
        out.append(f"  genuine  n={g['n']:5d}  min {_fmt(g['min'])}  median {_fmt(g['median'])}"
                   f"  max {_fmt(g['max'])}")
        out.append(f"  impostor n={i['n']:5d}  max {_fmt(i['max'])}  median {_fmt(i['median'])}")
        out.append(f"  FNMR {s['fnmr']}   FMR {s['fmr']}   EER {s['eer']} @ {s['eer_threshold']}"
                   f"   d' {s['dprime']}")

    block("set-max scores (the operational rule)", report["set_max"])
    if per_template:
        block("per-template scores", report["per_template"])
        out.append(f"  set-max inflation of the impostor maximum: "
                   f"{report['per_template'].get('inflation_max_impostor')}")

    ident = report["identification"]
    out.append("")
    out.append(f"leave-one-out 1:N: {ident['correct']} correct, {ident['rejected']} rejected, "
               f"{ident['wrong']} wrong, {ident['no_mate']} without a mate "
               f"(rank-1 {ident['rank1_rate']})")

    fta = report["fta"]
    out.append(f"failure to acquire: {fta['count']} of {m['n_labelled']} labelled "
               f"(rate {fta['rate']})")
    for f in fta["photos"]:
        out.append(f"  {f['name']:28s} {f['error_code']}: {(f['error'] or '')[:80]}")

    if report["unknown_probes"]:
        out.append("")
        out.append("unlabelled probes (excluded from every rate):")
        for u in report["unknown_probes"]:
            if u.get("error_code"):
                out.append(f"  {u['name']:36s} REJECTED {u['error_code']}")
            else:
                out.append(f"  {u['name']:36s} best {u['best']} at {_fmt(u['score'])}"
                           f"  {'MATCH' if u['matched'] else 'no match'}")

    if deltas:
        out.append("")
        out.append(f"{'metric':44s} {'prev':>9s} {'curr':>9s} {'delta':>9s}")
        for d in deltas:
            if d["prev"] is None and d["curr"] is None:
                continue
            out.append(f"{d['metric'][:44]:44s} {_fmt(d['prev'], 9, 2)} {_fmt(d['curr'], 9, 2)} "
                       f"{_fmt(d['delta'], 9, 2)} {d['flag']}")
    return "\n".join(out)


def run(
    manifest: Path = DEFAULT_MANIFEST,
    out_dir: Path = DEFAULT_OUT,
    cache_dir: Path = DEFAULT_CACHE,
    threshold: float = afis.THRESHOLD,
    use_cache: bool = True,
    distractors: bool = True,
    only: list[str] | None = None,
    min_minutiae: int = identity.MIN_MINUTIAE,
    min_quality: float | None = None,
    log=print,
) -> tuple[dict, Path]:
    """Extract (cached), score, and write a report. Extraction is sequential:
    one JVM per process, and determinism beats speed here."""
    rows = load_manifest(Path(manifest))
    if only:
        wanted = set(only)
        rows = [r for r in rows if r.name in wanted]
    if log:
        log(f"extracting {len(rows)} photos ({manifest}):")
    results = [
        extract_cached(r, cache_dir, use_cache, min_minutiae, min_quality, log)
        for r in rows
    ]
    pairs = score_pairs(results)
    dist = fvc_distractors() if distractors else {}
    loo = leave_one_out(results, threshold, dist)
    unknowns = probe_unknowns(results, threshold)
    settings = extraction_settings(True, min_minutiae, min_quality)
    report = build_report(results, pairs, loo, unknowns, threshold, Path(manifest),
                          sorted(dist), settings)
    return report, write_report(report, out_dir)
