"""Rebuild the enrolled gallery from the corpus manifest.

    ./.venv/bin/python scripts/rebuild_gallery.py            # dry run, changes nothing
    ./.venv/bin/python scripts/rebuild_gallery.py --apply

SourceAFIS templates are only comparable within one pipeline version, and the
stored gallery predates several. Rather than re-extract in place, this wipes the
identities and re-enrols them from data/samples/manifest.csv, which also records
consent properly at creation (the legacy CLI identities have consent = 0).

Two safety rules, because store.delete_identity removes the stored originals for
both the uuid AND the legacy name directory, and some of those originals exist
nowhere else:

  1. every original is copied into data/samples/photo/ first, and the wipe is
     refused unless each one is verifiably present there (by content hash)
  2. the database and data/enrolled/ are backed up before anything is deleted

Stop the API server first: this rewrites the database.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fingerphoto import crypto, evaluation, store  # noqa: E402

CORPUS = ROOT / "data" / "samples" / "photo"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def corpus_hashes() -> dict[str, Path]:
    return {_sha(p.read_bytes()): p for p in CORPUS.glob("*.jpg")}


def stored_originals() -> list[tuple[str, Path]]:
    """[(identity name, file)] for every stored original, both directory layouts."""
    out = []
    for row in store.list_identities():
        for d in (store.photo_store() / row["uuid"],
                  store.photo_store() / store._safe_filename(row["name"])):
            if d.is_dir():
                out.extend((row["name"], p) for p in sorted(d.iterdir()) if p.is_file())
    return out


def preserve(apply: bool) -> list[str]:
    """Copy any original that exists nowhere else into the corpus. Returns the
    list of files still unaccounted for (which must be empty before a wipe)."""
    known = corpus_hashes()
    key = store._encryption_key()
    unsafe: list[str] = []
    for name, path in stored_originals():
        raw = path.read_bytes()
        if path.suffix == ".enc":
            raw = crypto.decrypt(raw, crypto.SCHEME_FERNET, key)
        digest = _sha(raw)
        if digest in known:
            print(f"  already in the corpus: {path.name} ({name}) -> {known[digest].name}")
            continue
        base = path.name[:-4] if path.suffix == ".enc" else path.name
        dest = CORPUS / f"web_{store._safe_filename(name).lower()}_{base}"
        print(f"  {'copying' if apply else 'would copy'}: {path} -> {dest}")
        if apply:
            dest.write_bytes(raw)
            known[_sha(raw)] = dest
        else:
            unsafe.append(str(path))
    return unsafe


def backup() -> Path:
    dest = ROOT / "out" / "backup" / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    db = store.db_path()
    for p in (db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")):
        if p.exists():
            shutil.copy2(p, dest / p.name)
    if store.photo_store().is_dir():
        shutil.copytree(store.photo_store(), dest / "enrolled", dirs_exist_ok=True)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true", help="actually rebuild (default: dry run)")
    ap.add_argument("--manifest", type=Path, default=evaluation.DEFAULT_MANIFEST)
    args = ap.parse_args()

    con = store.connect()
    try:  # refuse to touch a database another process is writing
        con.execute("BEGIN IMMEDIATE")
        con.execute("ROLLBACK")
    except Exception as e:
        print(f"database is busy ({e}); stop the API server first")
        return 2
    finally:
        con.close()

    rows = evaluation.load_manifest(args.manifest)
    wanted: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for r in rows:
        if r.gallery:
            wanted[(r.subject, r.finger)].append(r.path)
    if not wanted:
        print(f"no rows marked gallery=1 in {args.manifest}")
        return 2

    print("existing identities:")
    for row in store.list_identities():
        print(f"  {row['name']:10s} {row['templates']:3d} templates, {row['photos']} photos,"
              f" consent {row['consent']}")

    print("\npreserving stored originals:")
    unsafe = preserve(args.apply)

    print("\nwill enrol:")
    for (subject, finger), paths in sorted(wanted.items()):
        print(f"  {subject:10s} {finger:12s} {len(paths)} photo(s): "
              f"{', '.join(p.name for p in paths)}")

    if not args.apply:
        print(f"\ndry run — nothing changed. {len(unsafe)} original(s) would be copied to the "
              f"corpus first. Re-run with --apply.")
        return 0

    if unsafe:
        print(f"\nABORT: {len(unsafe)} original(s) are not in the corpus: {unsafe}")
        return 1

    dest = backup()
    print(f"\nbacked up to {dest}")

    for row in store.list_identities():
        store.delete_identity(row["uuid"], request_id="rebuild-cli")
        print(f"  deleted {row['name']}")

    print()
    empty: list[str] = []
    for (subject, finger), paths in sorted(wanted.items()):
        result = store.enroll(subject, paths, finger=finger)
        for photo, n in result["enrolled"]:
            print(f"  {subject}/{finger}: {photo} -> {n} templates")
        for photo, why in result["rejected"]:
            print(f"  {subject}/{finger}: {photo} REJECTED — {str(why)[:110]}")
        if not result["enrolled"]:
            empty.append(f"{subject}/{finger}")

    print("\ngallery now:")
    for row in store.list_identities():
        print(f"  {row['name']:10s} {row['templates']:3d} templates, {row['photos']} photos,"
              f" consent {row['consent']}")

    if empty:
        print(f"\nWARNING: no templates for {', '.join(empty)} — those photos were refused by "
              f"the quality gate and must be recaptured.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
