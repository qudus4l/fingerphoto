"""Measure the pipeline on the labelled corpus.

    ./.venv/bin/python scripts/evaluate.py                    # cold ~6 min, warm < 30 s
    ./.venv/bin/python scripts/evaluate.py --compare out/eval/latest.json
    ./.venv/bin/python scripts/evaluate.py --write-baseline tests/baselines/eval_baseline.json

Reports land in out/eval/<timestamp>_<pipeline version>/ as report.json plus
photos.csv, pairs.csv and loo.csv, and the newest report is copied to
out/eval/latest.json so --compare has an obvious argument.

Exit status is always 0: this is a measurement, not a gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fingerphoto import afis, evaluation, identity  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, default=evaluation.DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=evaluation.DEFAULT_OUT)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(os.environ.get("FP_EVAL_CACHE", evaluation.DEFAULT_CACHE)))
    ap.add_argument("--no-cache", action="store_true", help="re-extract and overwrite the cache")
    ap.add_argument("--threshold", type=float, default=afis.THRESHOLD)
    ap.add_argument("--min-minutiae", type=int, default=identity.MIN_MINUTIAE)
    ap.add_argument("--min-quality", type=float, default=None,
                    help="ridge-quality gate (default: FP_MIN_RIDGE_QUALITY, else the "
                         "pipeline default)")
    ap.add_argument("--no-distractors", action="store_true",
                    help="omit the FVC contact prints from the 1:N gallery")
    ap.add_argument("--per-template", action="store_true",
                    help="also print single-template scores (shows set-max inflation)")
    ap.add_argument("--compare", type=Path, help="previous report.json to diff against")
    ap.add_argument("--write-baseline", type=Path,
                    help="write a slimmed report for the regression test to check against")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="restrict to these photo names")
    args = ap.parse_args()

    min_quality = args.min_quality
    if min_quality is None:
        env = os.environ.get("FP_MIN_RIDGE_QUALITY")
        if env is not None:
            min_quality = float(env)
        elif hasattr(identity, "MIN_RIDGE_QUALITY"):
            min_quality = identity.MIN_RIDGE_QUALITY

    report, out_dir = evaluation.run(
        manifest=args.manifest,
        out_dir=args.out,
        cache_dir=args.cache_dir,
        threshold=args.threshold,
        use_cache=not args.no_cache,
        distractors=not args.no_distractors,
        only=args.only,
        min_minutiae=args.min_minutiae,
        min_quality=min_quality,
    )

    deltas = None
    if args.compare:
        deltas = evaluation.compare(json.loads(Path(args.compare).read_text()), report)

    print()
    print(evaluation.format_report(report, per_template=args.per_template, deltas=deltas))
    print(f"\nreport: {out_dir}")

    if args.write_baseline:
        args.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.write_baseline.write_text(
            json.dumps(evaluation.strip_for_baseline(report), indent=1, default=str)
        )
        print(f"baseline: {args.write_baseline}")


if __name__ == "__main__":
    main()
