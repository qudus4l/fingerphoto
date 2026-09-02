"""SQLite connection + migrations-lite.

Migrations are append-only steps gated on PRAGMA user_version; version 0 is
either an empty database or the legacy pre-API schema, so the v1 step is
written to handle both (CREATE IF NOT EXISTS + column-presence checks).
Behavioral pragmas (WAL, busy_timeout, secure_delete, foreign_keys) are set
on every connection, not in migrations. WAL requires a local filesystem —
the Docker data volume must not be NFS.
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

# The legacy (pre-API) schema, also the base the v1 migration builds on.
BASE_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS identities (
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE NOT NULL,
        created_at REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS templates (
        id INTEGER PRIMARY KEY,
        identity_id INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
        finger TEXT NOT NULL DEFAULT 'thumb',
        template BLOB NOT NULL,
        minutiae INTEGER,
        source_photo TEXT NOT NULL,
        created_at REAL NOT NULL
    )""",
]


def connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA journal_mode = WAL")
    con.execute("PRAGMA busy_timeout = 5000")
    con.execute("PRAGMA secure_delete = ON")
    return con


def _columns(con: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in con.execute(f"PRAGMA table_info({table})")}


def _migrate_v1(con: sqlite3.Connection) -> None:
    for stmt in BASE_STATEMENTS:
        con.execute(stmt)

    icols = _columns(con, "identities")
    if "uuid" not in icols:
        con.execute("ALTER TABLE identities ADD COLUMN uuid TEXT")
    if "consent" not in icols:
        con.execute("ALTER TABLE identities ADD COLUMN consent INTEGER NOT NULL DEFAULT 0")
    if "consent_at" not in icols:
        con.execute("ALTER TABLE identities ADD COLUMN consent_at REAL")
    for (rowid,) in con.execute("SELECT id FROM identities WHERE uuid IS NULL").fetchall():
        con.execute("UPDATE identities SET uuid = ? WHERE id = ?", (str(uuid.uuid4()), rowid))
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_identities_uuid ON identities(uuid)")

    tcols = _columns(con, "templates")
    if "pipeline_version" not in tcols:
        con.execute(
            "ALTER TABLE templates ADD COLUMN pipeline_version TEXT NOT NULL DEFAULT 'pre-1.0'"
        )
    if "sourceafis_version" not in tcols:
        # every pre-migration template was extracted with 3.18.1
        con.execute(
            "ALTER TABLE templates ADD COLUMN sourceafis_version TEXT NOT NULL DEFAULT '3.18.1'"
        )
    if "enc" not in tcols:
        con.execute("ALTER TABLE templates ADD COLUMN enc TEXT")

    con.execute(
        """CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY,
            event TEXT NOT NULL,
            identity_uuid TEXT,
            request_id TEXT,
            outcome TEXT NOT NULL,
            detail TEXT,
            created_at REAL NOT NULL
        )"""
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_templates_identity ON templates(identity_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_templates_finger ON templates(finger)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at)")


def _migrate_v2(con: sqlite3.Connection) -> None:
    """Ridge-quality score of the crop each template came from (pipeline 1.2.0)."""
    if "quality" not in _columns(con, "templates"):
        con.execute("ALTER TABLE templates ADD COLUMN quality REAL")


MIGRATIONS = [_migrate_v1, _migrate_v2]


def migrate(con: sqlite3.Connection) -> None:
    current = con.execute("PRAGMA user_version").fetchone()[0]
    for target, step in enumerate(MIGRATIONS[current:], start=current + 1):
        with con:
            step(con)
            con.execute(f"PRAGMA user_version = {target}")
