"""Re-extract templates whose pipeline version is stale.

SourceAFIS templates are only comparable within one pipeline version, and
PIPELINE_VERSION 1.0.0 shipped a diverging scale loop: on soft photos the ridge
period inflated ~1.33x per iteration without converging, and the loop returned
its last (worst) estimate — templates extracted at up to 2x the correct scale.
Those rows will not match anything a 1.1.0 run produces.

    ./.venv/bin/python scripts/reextract.py            # report what is stale
    ./.venv/bin/python scripts/reextract.py --apply    # re-extract and replace

Needs the stored originals (FP_STORE_ORIGINALS): SourceAFIS templates cannot be
reversed, so a row whose source photo is missing can only be deleted and
recaptured. Stop the API server first — this rewrites rows in place.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fingerphoto import afis, crypto, identity, imaging, store, version  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite rows (default: report only)")
    ap.add_argument("--version", default=version.PIPELINE_VERSION,
                    help=f"target pipeline version (default {version.PIPELINE_VERSION})")
    args = ap.parse_args()

    con = store.connect()
    key = store._encryption_key()
    rows = con.execute(
        "SELECT t.id, t.identity_id, i.uuid, i.name, t.finger, t.source_photo,"
        "       t.pipeline_version"
        "  FROM templates t JOIN identities i ON i.id = t.identity_id"
        " WHERE t.pipeline_version != ?", (args.version,)
    ).fetchall()
    if not rows:
        print(f"nothing stale — every template is at pipeline {args.version}")
        con.close()
        return

    # group by the photo they came from: one re-extraction per source image
    groups: dict[tuple, list[int]] = defaultdict(list)
    for rid, iid, uuid, name, finger, src, pv in rows:
        groups[(iid, uuid, name, finger, src, pv)].append(rid)

    print(f"{len(rows)} stale templates from {len(groups)} photos "
          f"(target pipeline {args.version})\n")
    replaced = dropped = 0
    for (iid, uuid, name, finger, src, pv), ids in sorted(groups.items(), key=lambda g: g[0][2]):
        path = store.photo_store() / uuid / src
        if not path.exists():  # pre-uuid layout stored photos under the name
            path = store.photo_store() / store._safe_filename(name) / src
        if not path.exists():
            print(f"  {name:10s} {src:34s} {len(ids):2d} tmpl  "
                  f"pipeline {pv}  ORIGINAL MISSING — must be recaptured")
            dropped += len(ids)
            continue
        raw = crypto.decrypt(path.read_bytes(), "fernet-v1" if src.endswith(".enc") else None, key)
        try:
            bgr = imaging.decode_image(raw, src)
            templates = identity.photo_template_set(bgr, min_minutiae=store.min_minutiae())
        except Exception as e:
            print(f"  {name:10s} {src:34s} {len(ids):2d} tmpl  "
                  f"pipeline {pv}  RE-EXTRACT FAILED: {type(e).__name__}: {str(e)[:36]}")
            dropped += len(ids)
            continue
        print(f"  {name:10s} {src:34s} {len(ids):2d} -> {len(templates):2d} tmpl  "
              f"pipeline {pv} -> {args.version}")
        if not args.apply:
            continue
        now = time.time()
        with con:
            con.execute(
                f"DELETE FROM templates WHERE id IN ({','.join('?' * len(ids))})", ids
            )
            for t in templates:
                blob, scheme = crypto.encrypt(t, key)
                con.execute(
                    "INSERT INTO templates(identity_id, finger, template, minutiae,"
                    " source_photo, created_at, pipeline_version, sourceafis_version,"
                    " enc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (iid, finger, blob, afis.minutiae_count(t), src, now,
                     args.version, version.sourceafis_version(), scheme),
                )
        replaced += len(templates)

    if args.apply:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        print(f"\nwrote {replaced} templates; {dropped} could not be re-extracted "
              f"and were left in place")
    else:
        print(f"\ndry run — nothing written. {dropped} of {len(rows)} cannot be "
              f"re-extracted. Re-run with --apply.")
    con.close()


if __name__ == "__main__":
    main()
