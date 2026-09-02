"""Calibrate and verify the ridge-quality gate against labelled crops.

    ./.venv/bin/python scripts/quality_calibration.py
    ./.venv/bin/python scripts/quality_calibration.py --no-spectral   # expected to FAIL

Runs both crop strategies over the corpus, measures every quality feature on
each crop, and checks that the score separates crops known to yield matching
templates (GOOD) from crops with no resolvable ridges or non-fingertip geometry
(BAD). Prints per-feature AUC and the threshold at the midpoint of the gap.

The labels below are not guesses: the ambiguous ones were settled by running
the real matcher (see the table in the module docstring of fingerphoto/quality.py).
Exit status is 1 when the separation breaks, so this doubles as a regression
check whenever the scale loop, TARGET_PERIOD or the crop strategies change.

No JVM: templates are never extracted here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fingerphoto import evaluation, identity, imaging, quality  # noqa: E402
from fingerphoto.locate import crop_pad  # noqa: E402
from fingerphoto.pipeline import photo_to_ridge_image  # noqa: E402
from fingerphoto.scale import ridge_period_band_for_fingertip  # noqa: E402
from fingerphoto.segment import crop_distal  # noqa: E402

GOOD, BAD, UNCERTAIN = "GOOD", "BAD", "?"

# (photo stem, crop strategy) -> label.
LABELS: dict[tuple[str, str], str] = {
    # Crops whose templates match genuinely.
    ("thumb_6172", "distal"): GOOD, ("thumb_6172", "pad"): GOOD,
    ("thumb_6173", "distal"): GOOD,
    ("thumb_6174", "distal"): GOOD, ("thumb_6174", "pad"): GOOD,
    ("sister_6214", "distal"): GOOD, ("sister_6214", "pad"): GOOD,
    ("20260831-211132_match-194", "distal"): GOOD,
    ("thumb_6217", "pad"): GOOD,          # weak genuine, but real ridges
    # No resolvable ridges, or a whole hand / whole frame masquerading as a pad.
    ("20260831-193348_nomatch-6", "distal"): BAD, ("20260831-193348_nomatch-6", "pad"): BAD,
    ("20260831-195511_nomatch-17", "distal"): BAD, ("20260831-195511_nomatch-17", "pad"): BAD,
    ("20260831-202624_nomatch-26", "distal"): BAD, ("20260831-202624_nomatch-26", "pad"): BAD,
    ("20260831-203000_match-64", "distal"): BAD,
    ("thumb_6217", "distal"): BAD,        # the colour crop took the whole hand
    ("sister_6175", "distal"): BAD, ("sister_6175", "pad"): BAD,
    ("sister_6176", "distal"): BAD,
    ("sister_6176", "pad"): BAD,          # 100 minutiae, but 11.1 against its own subject
    ("web_w_capture", "pad"): BAD, ("web_qd_capture", "pad"): BAD,
    # Reported, not scored: over-wide crops that duplicate a good one, or web
    # captures whose true usability is unknown.
    ("web_w_capture", "distal"): UNCERTAIN, ("web_qd_capture", "distal"): UNCERTAIN,
    ("20260831-203000_match-64", "pad"): UNCERTAIN,
    ("thumb_6173", "pad"): UNCERTAIN,     # 90 ridges wide, period clamped; distal scores 193
    ("20260831-211132_match-194", "pad"): UNCERTAIN,
}

FEATURES = ("coherence_frac", "coherence_mean", "ocl_mean", "edge_density",
            "spectral_frac", "energy_median", "score", "score_nospec")


def crops_of(bgr: np.ndarray):
    """[(strategy, crop, mask, period hint)] for both strategies, failures skipped."""
    out = []
    try:
        c, m = crop_distal(bgr)
        out.append(("distal", c, m, None))
    except (ValueError, cv2.error) as e:
        print(f"    distal crop failed: {type(e).__name__}: {e}")
    try:
        c, m, hint = crop_pad(bgr)
        out.append(("pad", c, m, hint))
    except (ValueError, cv2.error) as e:
        print(f"    pad crop failed: {type(e).__name__}: {e}")
    return out


def crop_row(name, strategy, crop, mask, hint, frame_shape, min_quality) -> dict | None:
    band = ridge_period_band_for_fingertip(crop.shape[1])
    try:
        _, info = photo_to_ridge_image(crop, period_band=band, mask=mask, initial_period=hint)
    except (ValueError, cv2.error) as e:
        print(f"    {name}/{strategy}: enhancement failed ({type(e).__name__})")
        return None
    q = info.get("quality") or {}
    sanity = identity.crop_sanity_reason(info, strategy, crop.shape[:2], frame_shape)
    score = q.get("score", 0.0)
    nospec = quality.combine(q.get("coherence_frac", 0.0), q.get("edge_density", 0.0),
                             q.get("spectral_frac", 0.0), use_spectral=False)
    gated = sanity is not None or not q.get("valid") or score < min_quality
    return {
        "name": name, "strategy": strategy,
        "label": LABELS.get((name, strategy), UNCERTAIN),
        "shape": list(crop.shape[:2]),
        "frame_frac": round(crop.shape[0] * crop.shape[1] / float(frame_shape[0] * frame_shape[1]), 3),
        "period": info.get("ridge_period_px"), "band": info.get("period_band_px"),
        "ridges_across": round(crop.shape[1] / max(info.get("ridge_period_px") or 1, 1e-6)),
        "sanity": sanity, "score_nospec": round(nospec, 4),
        "gate": "reject" if gated else "pass",
        **{k: q.get(k) for k in ("coherence_frac", "coherence_mean", "ocl_mean",
                                 "edge_density", "spectral_frac", "energy_median",
                                 "n_blocks", "score", "valid")},
    }


def collect(manifest: Path = evaluation.DEFAULT_MANIFEST, only=None,
            min_quality: float = quality.MIN_RIDGE_QUALITY) -> list[dict]:
    rows: list[dict] = []
    for r in evaluation.load_manifest(manifest):
        if only and r.name not in only:
            continue
        if not any((r.name, s) in LABELS for s in ("distal", "pad")):
            continue
        bgr = imaging.decode_image(r.path.read_bytes(), r.path.name)
        print(f"  {r.name}")
        for strategy, crop, mask, hint in crops_of(bgr):
            row = crop_row(r.name, strategy, crop, mask, hint, bgr.shape[:2], min_quality)
            if row is not None:
                rows.append(row)
    return rows


def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney: probability a GOOD outranks a BAD, ties counting half."""
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def best_threshold(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """(threshold, balanced accuracy). Sits at the midpoint of a clean gap."""
    if not pos or not neg:
        return float("nan"), float("nan")
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    best = (float("nan"), -1.0)
    for t in sorted(set(p.tolist() + n.tolist())):
        ba = 0.5 * float((p >= t).mean() + (n < t).mean())
        if ba > best[1]:
            best = (t, ba)
    if p.min() > n.max():
        return float((p.min() + n.max()) / 2), 1.0
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, default=evaluation.DEFAULT_MANIFEST)
    ap.add_argument("--only", nargs="+", metavar="NAME")
    ap.add_argument("--min-quality", type=float, default=quality.MIN_RIDGE_QUALITY)
    ap.add_argument("--no-spectral", action="store_true",
                    help="score without the spectral factor (expected to fail; documents "
                         "why the factor is there)")
    ap.add_argument("--json", type=Path, help="write the rows for later inspection")
    args = ap.parse_args()

    t0 = time.monotonic()
    print(f"measuring crops (min quality {args.min_quality}):")
    rows = collect(args.manifest, set(args.only) if args.only else None, args.min_quality)
    key = "score_nospec" if args.no_spectral else "score"
    if args.no_spectral:
        for r in rows:
            r["gate"] = "reject" if (r["sanity"] or r[key] < args.min_quality) else "pass"

    print(f"\n{'lbl':4s} {'crop':38s} {'WxH':>11s} {'frm%':>5s} {'per':>6s} {'ridg':>4s} "
          f"{'blk':>5s} {'cohF':>5s} {'edge':>6s} {'spec':>5s} {'score':>6s} {'nospec':>6s} "
          f"{'sanity':>18s} gate")
    for r in sorted(rows, key=lambda r: (r["label"], -r[key])):
        print(f"{r['label']:4s} {r['name'] + '/' + r['strategy']:38s} "
              f"{str(r['shape'][1]) + 'x' + str(r['shape'][0]):>11s} {r['frame_frac'] * 100:5.0f} "
              f"{r['period']:6.1f} {r['ridges_across']:4d} {r['n_blocks']:5d} "
              f"{r['coherence_frac']:5.2f} {r['edge_density']:6.3f} {r['spectral_frac']:5.2f} "
              f"{r['score']:6.3f} {r['score_nospec']:6.3f} {str(r['sanity']):>18s} {r['gate']}")

    good = [r for r in rows if r["label"] == GOOD]
    bad = [r for r in rows if r["label"] == BAD]
    print(f"\n{len(good)} GOOD, {len(bad)} BAD, "
          f"{sum(1 for r in rows if r['label'] == UNCERTAIN)} uncertain "
          f"({time.monotonic() - t0:.0f}s)")

    print(f"\n{'feature':16s} {'AUC':>6s} {'GOOD min':>9s} {'BAD max':>9s} {'gap':>7s} "
          f"{'threshold':>10s}")
    for f in FEATURES:
        pos, neg = [r[f] for r in good], [r[f] for r in bad]
        if not pos or not neg:
            continue
        t, _ = best_threshold(pos, neg)
        print(f"{f:16s} {auc(pos, neg):6.3f} {min(pos):9.3f} {max(neg):9.3f} "
              f"{min(pos) - max(neg):+7.3f} {t:10.3f}")

    uncertain = [r for r in rows if r["label"] == UNCERTAIN]
    if uncertain:
        print("\nuncertain crops (not scored, shown so a wrong label is visible):")
        for r in sorted(uncertain, key=lambda r: -r[key]):
            print(f"  {r['name'] + '/' + r['strategy']:38s} {r[key]:6.3f}  {r['gate']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=1, default=str))
        print(f"\nrows: {args.json}")

    misgated = [r for r in good if r["gate"] != "pass"] + [r for r in bad if r["gate"] != "reject"]
    if misgated or not good or not bad:
        print("\nSEPARATION FAILED: " + ", ".join(
            f"{r['name']}/{r['strategy']} ({r['label']} -> {r['gate']}, {key} {r[key]:.3f})"
            for r in misgated) or "\nSEPARATION FAILED: no labelled crops resolved")
        return 1
    t, _ = best_threshold([r["score"] for r in good], [r["score"] for r in bad])
    print(f"\nSEPARATION OK — every GOOD passes, every BAD is rejected.")
    print(f"recommended FP_MIN_RIDGE_QUALITY = {t:.2f} (currently {args.min_quality})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
