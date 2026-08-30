"""Offline re-encryption: encrypt plaintext rows/photos with the current key,
or rotate from an old key to the current one.

    FP_ENCRYPTION_KEY=<new> ./.venv/bin/python scripts/reencrypt.py             # encrypt plaintext
    FP_ENCRYPTION_KEY=<new> ./.venv/bin/python scripts/reencrypt.py --old <old> # rotate keys

Stop the API server first — this rewrites rows in place.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fingerphoto import crypto, store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", help="previous key when rotating (default: rows are plaintext)")
    args = ap.parse_args()
    new_key = store._encryption_key()
    if not new_key:
        sys.exit("set FP_ENCRYPTION_KEY to the key to encrypt with")

    con = store.connect()
    rows = con.execute("SELECT id, template, enc FROM templates").fetchall()
    changed = 0
    with con:
        for rowid, blob, enc in rows:
            plain = crypto.decrypt(blob, enc, args.old)
            data, scheme = crypto.encrypt(plain, new_key)
            con.execute("UPDATE templates SET template = ?, enc = ? WHERE id = ?",
                        (data, scheme, rowid))
            changed += 1
    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    print(f"re-encrypted {changed} template rows")

    photos = 0
    for f in store.photo_store().glob("*/*"):
        if not f.is_file():
            continue
        if f.suffix == ".enc":
            if not args.old:
                continue  # already encrypted with the current scheme lineage
            plain = crypto.decrypt(f.read_bytes(), crypto.SCHEME_FERNET, args.old)
            f.write_bytes(crypto.encrypt(plain, new_key)[0])
        else:
            data, scheme = crypto.encrypt(f.read_bytes(), new_key)
            f.with_name(f.name + ".enc").write_bytes(data)
            f.unlink()
        photos += 1
    print(f"re-encrypted {photos} stored photos")


if __name__ == "__main__":
    main()
