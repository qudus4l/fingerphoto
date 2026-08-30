"""Persistent identity store: SQLite for templates, disk for originals.

Design notes:
- An identity holds template SETS from several photos (multi-photo enrollment
  is the biggest accuracy lever; set-max scoring at identification).
- Original photos are kept (encrypted when FP_ENCRYPTION_KEY is set) because
  SourceAFIS templates are version-locked — upgrading the matcher means
  re-extracting from the originals. Storage is opt-out via FP_STORE_ORIGINALS.
- Finger position is stored per template (ANSI/NIST-ITL symbolic names) so
  4+1 fusion can query per position once multi-finger enrollment lands.
- Every mutating or matching operation writes an audit_log row — metadata
  only, never image bytes or templates.
- Configuration is environment-driven (FP_*) so the CLI, the API, and tests
  all resolve the same paths/limits without importing the API layer.
"""
from __future__ import annotations

import os
import re
import shutil
import sqlite3
import time
import uuid as uuidlib
from pathlib import Path

import numpy as np

from . import afis, crypto, db, identity, imaging, version
from .errors import AcquisitionError, ReasonCode, StoreError

ROOT = Path(__file__).resolve().parent.parent

# ANSI/NIST-ITL 1-2011 finger position codes (0 = unknown)
FINGER_POSITIONS = {
    "unknown": 0,
    "right_thumb": 1,
    "right_index": 2,
    "right_middle": 3,
    "right_ring": 4,
    "right_little": 5,
    "left_thumb": 6,
    "left_index": 7,
    "left_middle": 8,
    "left_ring": 9,
    "left_little": 10,
    "thumb": 0,  # legacy value from the pre-API store; treated as unknown side
}


def db_path() -> Path:
    return Path(os.environ.get("FP_DB_PATH", ROOT / "data" / "identities.db"))


def photo_store() -> Path:
    return Path(os.environ.get("FP_DATA_DIR", ROOT / "data")) / "enrolled"


def _encryption_key() -> str | None:
    return os.environ.get("FP_ENCRYPTION_KEY") or None


def _store_originals() -> bool:
    return os.environ.get("FP_STORE_ORIGINALS", "true").lower() not in ("0", "false", "no")


def threshold() -> float:
    return float(os.environ.get("FP_THRESHOLD", afis.THRESHOLD))


def min_minutiae() -> int:
    return int(os.environ.get("FP_MIN_MINUTIAE", identity.MIN_MINUTIAE))


def connect() -> sqlite3.Connection:
    con = db.connect(db_path())
    db.migrate(con)  # no-op after the first call (single PRAGMA read)
    return con


def audit(
    con: sqlite3.Connection,
    event: str,
    identity_uuid: str | None = None,
    request_id: str | None = None,
    outcome: str = "ok",
    detail: str | None = None,
) -> None:
    with con:
        con.execute(
            "INSERT INTO audit_log(event, identity_uuid, request_id, outcome, detail,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (event, identity_uuid, request_id, outcome, detail, time.time()),
        )


def _safe_filename(filename: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)[:120]
    return name.lstrip(".") or "photo"


def _resolve(con: sqlite3.Connection, identity_uuid: str) -> tuple[int, str, int]:
    row = con.execute(
        "SELECT id, name, consent FROM identities WHERE uuid = ?", (identity_uuid,)
    ).fetchone()
    if row is None:
        raise StoreError(ReasonCode.IDENTITY_NOT_FOUND, f"no identity {identity_uuid}")
    return row


def create_identity(name: str, consent: bool, request_id: str | None = None) -> dict:
    if not consent:
        raise StoreError(
            ReasonCode.CONSENT_REQUIRED,
            "explicit consent is required to enroll biometric data",
        )
    con = connect()
    try:
        new_uuid, now = str(uuidlib.uuid4()), time.time()
        try:
            with con:
                con.execute(
                    "INSERT INTO identities(name, uuid, consent, consent_at, created_at)"
                    " VALUES (?, ?, 1, ?, ?)",
                    (name, new_uuid, now, now),
                )
        except sqlite3.IntegrityError:
            raise StoreError(ReasonCode.IDENTITY_EXISTS, f"identity named {name!r} exists")
        audit(con, "identity.create", new_uuid, request_id)
        return {"uuid": new_uuid, "name": name, "consent": True, "created_at": now}
    finally:
        con.close()


def enroll_photos(
    identity_uuid: str,
    photos: list[tuple[str, np.ndarray, bytes | None]],
    finger: str = "unknown",
    request_id: str | None = None,
) -> dict:
    """Enroll decoded photos: list of (filename, bgr, original_bytes).
    Per-photo failures are reported with reason codes, not raised."""
    if finger not in FINGER_POSITIONS:
        raise StoreError(
            ReasonCode.INVALID_FINGER_POSITION,
            f"finger must be one of {sorted(set(FINGER_POSITIONS) - {'thumb'})}",
        )
    con = connect()
    try:
        iid, name, consent = _resolve(con, identity_uuid)
        if not consent:
            raise StoreError(
                ReasonCode.CONSENT_REQUIRED,
                "identity has no recorded consent; record consent before enrolling",
            )
        key = _encryption_key()
        dest = photo_store() / identity_uuid
        results = []
        for filename, bgr, original in photos:
            safe = _safe_filename(filename)
            stats: dict = {}
            try:
                templates = identity.photo_template_set(
                    bgr, min_minutiae=min_minutiae(), stats=stats
                )
            except AcquisitionError as e:
                results.append(
                    {"filename": safe, "status": "rejected",
                     "reason_code": e.code.value, "detail": str(e)}
                )
                audit(con, "photo.enroll", identity_uuid, request_id, "rejected", e.code.value)
                continue
            stored_name = safe
            if _store_originals() and original is not None:
                dest.mkdir(parents=True, exist_ok=True)
                data, scheme = crypto.encrypt(original, key)
                stored_name = safe + (".enc" if scheme else "")
                (dest / stored_name).write_bytes(data)
            now = time.time()
            with con:
                for t in templates:
                    blob, scheme = crypto.encrypt(t, key)
                    con.execute(
                        "INSERT INTO templates(identity_id, finger, template, minutiae,"
                        " source_photo, created_at, pipeline_version, sourceafis_version,"
                        " enc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (iid, finger, blob, afis.minutiae_count(t), stored_name, now,
                         version.PIPELINE_VERSION, version.sourceafis_version(), scheme),
                    )
            best = max((afis.minutiae_count(t) or 0) for t in templates)
            results.append(
                {"filename": safe, "status": "enrolled",
                 "templates": len(templates), "minutiae": best, "timings": stats}
            )
            audit(con, "photo.enroll", identity_uuid, request_id, "enrolled",
                  f"{len(templates)} templates")
        return {"identity": {"uuid": identity_uuid, "name": name}, "results": results}
    finally:
        con.close()


def _decrypted_gallery(
    con: sqlite3.Connection, finger: str | None = None
) -> dict[str, tuple[str, list[bytes]]]:
    q = (
        "SELECT i.uuid, i.name, t.template, t.enc FROM templates t"
        " JOIN identities i ON i.id = t.identity_id"
    )
    args: tuple = ()
    if finger:
        q += " WHERE t.finger = ?"
        args = (finger,)
    key = _encryption_key()
    out: dict[str, tuple[str, list[bytes]]] = {}
    for u, name, blob, enc in con.execute(q, args):
        out.setdefault(u, (name, []))[1].append(crypto.decrypt(blob, enc, key))
    return out


def templates_for_identity(identity_uuid: str, finger: str | None = None) -> list[bytes]:
    con = connect()
    try:
        iid, _, _ = _resolve(con, identity_uuid)
        q = "SELECT template, enc FROM templates WHERE identity_id = ?"
        args: list = [iid]
        if finger:
            q += " AND finger = ?"
            args.append(finger)
        key = _encryption_key()
        return [crypto.decrypt(b, e, key) for b, e in con.execute(q, args)]
    finally:
        con.close()


def identify_bgr(
    bgr: np.ndarray,
    match_threshold: float | None = None,
    request_id: str | None = None,
) -> dict:
    """1:N identification of one decoded photo against every enrolled identity."""
    thr = threshold() if match_threshold is None else match_threshold
    con = connect()
    try:
        gal = _decrypted_gallery(con)
        if not gal:
            raise StoreError(ReasonCode.GALLERY_EMPTY, "no identities enrolled")
        probe = identity.photo_template_set(bgr, min_minutiae=min_minutiae())
        scores = {u: identity.set_score(probe, ts) for u, (_, ts) in gal.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_uuid, best_score = ranked[0]
        matched = best_score >= thr
        audit(con, "identify", best_uuid if matched else None, request_id,
              "match" if matched else "no_match")
        entry = lambda u, s: {"uuid": u, "name": gal[u][0], "score": round(s, 1)}
        return {
            "match": entry(best_uuid, best_score) if matched else None,
            "best_candidate": entry(best_uuid, best_score),
            "scores": [entry(u, s) for u, s in ranked],
            "threshold": thr,
        }
    finally:
        con.close()


def verify_bgr(
    identity_uuid: str,
    bgr: np.ndarray,
    match_threshold: float | None = None,
    finger: str | None = None,
    request_id: str | None = None,
) -> dict:
    """1:1 verification of one decoded photo against a claimed identity."""
    thr = threshold() if match_threshold is None else match_threshold
    con = connect()
    try:
        _, name, _ = _resolve(con, identity_uuid)
    finally:
        con.close()
    enrolled = templates_for_identity(identity_uuid, finger)
    if not enrolled:
        raise StoreError(
            ReasonCode.IDENTITY_HAS_NO_TEMPLATES,
            f"identity {identity_uuid} has no enrolled templates"
            + (f" for finger {finger}" if finger else ""),
        )
    probe = identity.photo_template_set(bgr, min_minutiae=min_minutiae())
    verified, score = identity.verify(probe, enrolled, thr)
    con = connect()
    try:
        audit(con, "verify", identity_uuid, request_id,
              "verified" if verified else "rejected")
    finally:
        con.close()
    return {
        "identity": {"uuid": identity_uuid, "name": name},
        "verified": verified,
        "score": round(score, 1),
        "threshold": thr,
    }


def delete_identity(identity_uuid: str, request_id: str | None = None) -> dict:
    """GDPR/BIPA hard delete: rows (CASCADE), original photos, WAL remnants."""
    con = connect()
    try:
        iid, name, _ = _resolve(con, identity_uuid)
        with con:
            con.execute("DELETE FROM identities WHERE id = ?", (iid,))
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # don't leave blobs in the WAL
        for d in (photo_store() / identity_uuid, photo_store() / _safe_filename(name)):
            if d.is_dir():
                shutil.rmtree(d)
        audit(con, "identity.delete", identity_uuid, request_id)
        return {"deleted": {"uuid": identity_uuid, "name": name}}
    finally:
        con.close()


def get_identity(identity_uuid: str) -> dict:
    con = connect()
    try:
        row = con.execute(
            "SELECT i.uuid, i.name, i.consent, i.consent_at, i.created_at,"
            " COUNT(t.id), COUNT(DISTINCT t.source_photo)"
            " FROM identities i LEFT JOIN templates t ON t.identity_id = i.id"
            " WHERE i.uuid = ? GROUP BY i.id",
            (identity_uuid,),
        ).fetchone()
        if row is None:
            raise StoreError(ReasonCode.IDENTITY_NOT_FOUND, f"no identity {identity_uuid}")
        u, name, consent, consent_at, created_at, templates, photos = row
        return {"uuid": u, "name": name, "consent": bool(consent), "consent_at": consent_at,
                "created_at": created_at, "templates": templates, "photos": photos}
    finally:
        con.close()


def list_identities() -> list[dict]:
    con = connect()
    try:
        rows = con.execute(
            "SELECT i.uuid, i.name, i.consent, COUNT(t.id), COUNT(DISTINCT t.source_photo)"
            " FROM identities i LEFT JOIN templates t ON t.identity_id = i.id"
            " GROUP BY i.id ORDER BY i.name"
        ).fetchall()
        return [
            {"uuid": u, "name": n, "consent": bool(c), "templates": t, "photos": p}
            for u, n, c, t, p in rows
        ]
    finally:
        con.close()


# --- legacy path-based wrappers (keep scripts/enroll.py + identify.py working) ---


def enroll(name: str, photo_paths: list[Path], finger: str = "thumb") -> dict:
    """CLI wrapper: find-or-create identity by name (consent is granted
    interactively by running the command — printed as a notice), read photos
    from disk. Returns the legacy {identity, enrolled, rejected} shape."""
    con = connect()
    try:
        row = con.execute("SELECT uuid, consent FROM identities WHERE name = ?", (name,)).fetchone()
    finally:
        con.close()
    if row is None:
        print(f"notice: creating identity {name!r} with consent recorded (CLI enrollment)")
        identity_uuid = create_identity(name, consent=True)["uuid"]
    else:
        identity_uuid, consent = row
        if not consent:
            print(f"notice: recording consent for existing identity {name!r} (CLI enrollment)")
            con = connect()
            try:
                with con:
                    con.execute(
                        "UPDATE identities SET consent = 1, consent_at = ? WHERE uuid = ?",
                        (time.time(), identity_uuid),
                    )
            finally:
                con.close()

    photos, unreadable = [], []
    for path in map(Path, photo_paths):
        try:
            data = path.read_bytes()
            photos.append((path.name, imaging.decode_image(data, path.name), data))
        except (OSError, AcquisitionError) as e:
            unreadable.append((path.name, str(e)))
    out = enroll_photos(identity_uuid, photos, finger=finger)
    enrolled = [(r["filename"], r["templates"]) for r in out["results"] if r["status"] == "enrolled"]
    rejected = unreadable + [
        (r["filename"], r["detail"]) for r in out["results"] if r["status"] == "rejected"
    ]
    return {"identity": name, "enrolled": enrolled, "rejected": rejected}


def gallery(finger: str | None = None) -> dict[str, list[bytes]]:
    """Legacy shape: name -> decrypted templates."""
    con = connect()
    try:
        return {name: ts for name, ts in _decrypted_gallery(con, finger).values()}
    finally:
        con.close()


def identify_photo(photo_path: Path, match_threshold: float | None = None) -> dict:
    """CLI wrapper around identify_bgr; returns the legacy shape."""
    data = Path(photo_path).read_bytes()
    bgr = imaging.decode_image(data, Path(photo_path).name)
    try:
        result = identify_bgr(bgr, match_threshold)
    except StoreError as e:
        if e.code is ReasonCode.GALLERY_EMPTY:
            return {"match": None, "scores": {}, "note": "gallery is empty"}
        raise
    return {
        "match": result["match"]["name"] if result["match"] else None,
        "best_candidate": result["best_candidate"]["name"],
        "scores": {e["name"]: e["score"] for e in result["scores"]},
        "threshold": result["threshold"],
    }
