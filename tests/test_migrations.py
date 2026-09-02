from __future__ import annotations

import sqlite3
import time

from fingerphoto import db


def _make_v0(path) -> None:
    """Synthesize the legacy pre-API schema with live-looking rows."""
    con = sqlite3.connect(path)
    for stmt in db.BASE_STATEMENTS:
        con.execute(stmt)
    now = time.time()
    con.execute("INSERT INTO identities(name, created_at) VALUES ('Q', ?)", (now,))
    con.execute(
        "INSERT INTO templates(identity_id, finger, template, minutiae, source_photo,"
        " created_at) VALUES (1, 'thumb', X'DEADBEEF', 30, 'thumb.jpg', ?)",
        (now,),
    )
    con.commit()
    con.close()


def test_migrates_legacy_v0(tmp_path):
    path = tmp_path / "v0.db"
    _make_v0(path)
    con = db.connect(path)
    db.migrate(con)
    assert con.execute("PRAGMA user_version").fetchone()[0] == len(db.MIGRATIONS)
    u, consent = con.execute("SELECT uuid, consent FROM identities").fetchone()
    assert u and len(u) == 36  # backfilled uuid4
    assert consent == 0  # legacy rows need explicit re-consent
    pv, sv, enc = con.execute(
        "SELECT pipeline_version, sourceafis_version, enc FROM templates"
    ).fetchone()
    assert (pv, sv, enc) == ("pre-1.0", "3.18.1", None)
    assert "quality" in db._columns(con, "templates")  # v2: null on legacy rows
    assert con.execute("SELECT quality FROM templates").fetchone()[0] is None
    assert con.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0
    con.close()


def test_migrate_is_idempotent(tmp_path):
    path = tmp_path / "v0.db"
    _make_v0(path)
    for _ in range(2):
        con = db.connect(path)
        db.migrate(con)
        con.close()
    con = db.connect(path)
    assert con.execute("SELECT COUNT(*) FROM identities").fetchone()[0] == 1
    con.close()


def test_fresh_database(tmp_path):
    con = db.connect(tmp_path / "fresh.db")
    db.migrate(con)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"identities", "templates", "audit_log"} <= tables
    assert "quality" in db._columns(con, "templates")
    con.close()
