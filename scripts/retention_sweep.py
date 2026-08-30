"""BIPA/GDPR retention sweep: find (and optionally delete) identities whose
last interaction is older than the retention window.

    ./.venv/bin/python scripts/retention_sweep.py            # dry run
    ./.venv/bin/python scripts/retention_sweep.py --delete   # actually delete

"Last interaction" = the newest of: identity creation, template enrollment,
or any audit event naming the identity. BIPA requires destruction when the
purpose is satisfied or within 3 years of the last interaction, whichever
comes first (default window here: 3 years).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fingerphoto import store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=3 * 365, help="retention window (days)")
    ap.add_argument("--delete", action="store_true", help="delete instead of just listing")
    args = ap.parse_args()
    cutoff = time.time() - args.days * 86400

    con = store.connect()
    rows = con.execute(
        """SELECT i.uuid, i.name, MAX(x.last) FROM identities i JOIN (
             SELECT id AS iid, created_at AS last FROM identities
             UNION ALL SELECT identity_id, created_at FROM templates
             UNION ALL SELECT i2.id, a.created_at FROM audit_log a
               JOIN identities i2 ON i2.uuid = a.identity_uuid
           ) x ON x.iid = i.id GROUP BY i.id"""
    ).fetchall()
    con.close()

    stale = [(u, n, last) for u, n, last in rows if last < cutoff]
    if not stale:
        print(f"nothing past the {args.days:.0f}-day window ({len(rows)} identities checked)")
        return
    for u, name, last in stale:
        age_days = (time.time() - last) / 86400
        if args.delete:
            store.delete_identity(u, request_id="retention-sweep")
            print(f"DELETED {name} ({u}) — last interaction {age_days:.0f} days ago")
        else:
            print(f"would delete {name} ({u}) — last interaction {age_days:.0f} days ago"
                  " (rerun with --delete)")


if __name__ == "__main__":
    main()
