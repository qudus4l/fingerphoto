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

from . import afis, crypto, db, frames, identity, imaging, version
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


def min_ridge_quality() -> float:
    """Ridge-structure score a crop must reach before a template is extracted.
    0 disables the score gate (the geometric sanity rules always apply)."""
    return float(os.environ.get("FP_MIN_RIDGE_QUALITY", identity.MIN_RIDGE_QUALITY))


def video_frames_enroll() -> int:
    """How many frames a video enrollment keeps. Each costs a full extraction
    (10-20 s), so this is a latency budget as much as an accuracy knob."""
    return int(os.environ.get("FP_VIDEO_FRAMES_ENROLL", frames.VIDEO_FRAMES_ENROLL))


def video_frames_identify() -> int:
    return int(os.environ.get("FP_VIDEO_FRAMES_IDENTIFY", frames.VIDEO_FRAMES_IDENTIFY))


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


def _expand_uploads(
    uploads: list[tuple[str, np.ndarray | None, bytes | None]],
    best_of: int | None,
    k_video: int,
) -> tuple[list[frames.Candidate], dict, list[dict]]:
    """Uploads -> selected candidates, a selection summary, and one rejected
    result per upload that could not be expanded at all (an unreadable or
    over-long clip). Everything that did expand still competes in one pool."""
    failures: list[dict] = []

    def failed(name: str, exc: AcquisitionError) -> None:
        failures.append(
            {"filename": _safe_filename(name), "status": "rejected",
             "reason_code": exc.code.value, "detail": str(exc)}
        )

    selected, summary = frames.expand(
        uploads, best_of, k_video=k_video,
        min_quality=min_ridge_quality(), on_error=failed,
    )
    return selected, summary, failures


def enroll_photos(
    identity_uuid: str,
    photos: list[tuple[str, np.ndarray | None, bytes | None]],
    finger: str = "unknown",
    request_id: str | None = None,
    best_of: int | None = None,
) -> dict:
    """Enroll uploads: list of (filename, bgr, original_bytes). A bgr of None
    marks a video, whose frames are scored here and the best few enrolled.
    best_of pools every upload and keeps that many overall (the browser burst).
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
        candidates, selection, results = _expand_uploads(
            photos, best_of, video_frames_enroll()
        )
        for r in results:
            audit(con, "photo.enroll", identity_uuid, request_id, "rejected",
                  r["reason_code"])
        for cand in candidates:
            safe = _safe_filename(cand.name)
            stats: dict = {}
            try:
                templates = identity.photo_template_set(
                    cand.pixels(), min_minutiae=min_minutiae(),
                    min_quality=min_ridge_quality(), stats=stats
                )
            except AcquisitionError as e:
                results.append(
                    {"filename": safe, "status": "rejected",
                     "reason_code": e.code.value, "detail": str(e),
                     "quality": stats.get("quality"), "timings": stats, **cand.meta()}
                )
                audit(con, "photo.enroll", identity_uuid, request_id, "rejected", e.code.value)
                continue
            now = time.time()
            stored_name = safe
            if _store_originals():
                # Only encode the frame if it is actually going to be kept.
                dest.mkdir(parents=True, exist_ok=True)
                data, scheme = crypto.encrypt(cand.original_bytes(), key)
                # Uniquify: every capture from the web UI arrives as
                # "capture.jpg", so a bare filename silently OVERWRITES the
                # previous enrollment's original. SourceAFIS templates have no
                # backward compatibility and cannot be re-extracted without the
                # source image, so that overwrite is unrecoverable data loss.
                stem, _, ext = safe.rpartition(".")
                stored_name = (f"{stem or safe}_{int(now)}"
                               + (f".{ext}" if stem else "")
                               + (".enc" if scheme else ""))
                (dest / stored_name).write_bytes(data)
            per_template_quality = stats.get("template_quality") or [None] * len(templates)
            with con:
                for t, tq in zip(templates, per_template_quality):
                    blob, scheme = crypto.encrypt(t, key)
                    con.execute(
                        "INSERT INTO templates(identity_id, finger, template, minutiae,"
                        " source_photo, created_at, pipeline_version, sourceafis_version,"
                        " enc, quality) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (iid, finger, blob, afis.minutiae_count(t), stored_name, now,
                         version.PIPELINE_VERSION, version.sourceafis_version(), scheme, tq),
                    )
            best = max((afis.minutiae_count(t) or 0) for t in templates)
            results.append(
                {"filename": safe, "status": "enrolled",
                 "templates": len(templates), "minutiae": best,
                 "quality": stats.get("quality"), "timings": stats, **cand.meta()}
            )
            audit(con, "photo.enroll", identity_uuid, request_id, "enrolled",
                  f"{len(templates)} templates")
        return {"identity": {"uuid": identity_uuid, "name": name},
                "results": results, "selection": selection}
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


def _save_probe(bgr: np.ndarray, outcome: str) -> None:
    """Diagnostic: keep identify probes when FP_SAVE_PROBES is set. Off by
    default — these are biometric images and identification is normally a
    read-only operation. Invaluable while tuning capture, since a failed
    probe is otherwise invisible."""
    if os.environ.get("FP_SAVE_PROBES", "").lower() not in ("1", "true", "yes"):
        return
    import cv2

    d = Path(os.environ.get("FP_DATA_DIR", ROOT / "data")) / "probes"
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(str(d / f"{stamp}_{outcome}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _probe_set(cands: list[frames.Candidate]) -> tuple[list[bytes], dict]:
    """Templates from every usable candidate, unioned into one probe set.

    Set-max scoring downstream means extra frames can only help: a frame that
    matches nothing simply never wins the max. A candidate that fails to
    acquire is skipped; if none survive, the last failure is re-raised so the
    caller still gets the measured reason."""
    probe: list[bytes] = []
    used: list[frames.Candidate] = []
    last: AcquisitionError | None = None
    for c in cands:
        try:
            probe += identity.photo_template_set(
                c.pixels(), min_minutiae=min_minutiae(), min_quality=min_ridge_quality()
            )
            used.append(c)
        except AcquisitionError as e:
            last = e
            _save_probe(c.pixels(), "acquire-failed")
    if not probe:
        raise last if last is not None else AcquisitionError(
            ReasonCode.NO_FINGER_FOUND, "no candidate frame produced a template"
        )
    return probe, {
        "candidates": len(cands),
        "used": len(used),
        "scores": [round(c.score, 3) for c in used if c.score is not None],
    }


def identify_frames(
    cands: list[frames.Candidate],
    match_threshold: float | None = None,
    request_id: str | None = None,
    selection: dict | None = None,
) -> dict:
    """1:N identification from one or more candidate frames."""
    thr = threshold() if match_threshold is None else match_threshold
    con = connect()
    try:
        gal = _decrypted_gallery(con)
        if not gal:
            raise StoreError(ReasonCode.GALLERY_EMPTY, "no identities enrolled")
        probe, used = _probe_set(cands)
        scores = {u: identity.set_score(probe, ts) for u, (_, ts) in gal.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_uuid, best_score = ranked[0]
        matched = best_score >= thr
        for c in cands:
            _save_probe(c.pixels(), f"{'match' if matched else 'nomatch'}-{best_score:.0f}")
        audit(con, "identify", best_uuid if matched else None, request_id,
              "match" if matched else "no_match")
        entry = lambda u, s: {"uuid": u, "name": gal[u][0], "score": round(s, 1)}
        out = {
            "match": entry(best_uuid, best_score) if matched else None,
            "best_candidate": entry(best_uuid, best_score),
            "scores": [entry(u, s) for u, s in ranked],
            "threshold": thr,
            "frames": used,
        }
        if selection is not None:
            out["selection"] = selection
        return out
    finally:
        con.close()


def _one(bgr: np.ndarray) -> list[frames.Candidate]:
    """A single decoded photo as a one-element candidate list."""
    return [frames.Candidate(name="probe", bgr=bgr, original=None,
                             source="probe", index=0, time_s=None)]


def identify_bgr(
    bgr: np.ndarray,
    match_threshold: float | None = None,
    request_id: str | None = None,
) -> dict:
    """1:N identification of one decoded photo against every enrolled identity."""
    return identify_frames(_one(bgr), match_threshold, request_id)


def identify_uploads(
    uploads: list[tuple[str, np.ndarray | None, bytes | None]],
    match_threshold: float | None = None,
    best_of: int | None = None,
    request_id: str | None = None,
) -> dict:
    """1:N identification from photos and/or videos, best frames first."""
    cands, selection, failures = _expand_uploads(
        uploads, best_of, video_frames_identify()
    )
    if not cands:
        raise AcquisitionError(
            ReasonCode.VIDEO_UNREADABLE,
            "; ".join(f["detail"] for f in failures) or "no usable frames in the upload",
        )
    selection["rejected"] = failures
    return identify_frames(cands, match_threshold, request_id, selection)


def verify_frames(
    identity_uuid: str,
    cands: list[frames.Candidate],
    match_threshold: float | None = None,
    finger: str | None = None,
    request_id: str | None = None,
    selection: dict | None = None,
) -> dict:
    """1:1 verification from one or more candidate frames."""
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
    probe, used = _probe_set(cands)
    verified, score = identity.verify(probe, enrolled, thr)
    con = connect()
    try:
        audit(con, "verify", identity_uuid, request_id,
              "verified" if verified else "rejected")
    finally:
        con.close()
    out = {
        "identity": {"uuid": identity_uuid, "name": name},
        "verified": verified,
        "score": round(score, 1),
        "threshold": thr,
        "frames": used,
    }
    if selection is not None:
        out["selection"] = selection
    return out


def verify_bgr(
    identity_uuid: str,
    bgr: np.ndarray,
    match_threshold: float | None = None,
    finger: str | None = None,
    request_id: str | None = None,
) -> dict:
    """1:1 verification of one decoded photo against a claimed identity."""
    return verify_frames(identity_uuid, _one(bgr), match_threshold, finger, request_id)


def verify_uploads(
    identity_uuid: str,
    uploads: list[tuple[str, np.ndarray | None, bytes | None]],
    match_threshold: float | None = None,
    best_of: int | None = None,
    finger: str | None = None,
    request_id: str | None = None,
) -> dict:
    """1:1 verification from photos and/or videos, best frames first."""
    cands, selection, failures = _expand_uploads(
        uploads, best_of, video_frames_identify()
    )
    if not cands:
        raise AcquisitionError(
            ReasonCode.VIDEO_UNREADABLE,
            "; ".join(f["detail"] for f in failures) or "no usable frames in the upload",
        )
    selection["rejected"] = failures
    return verify_frames(identity_uuid, cands, match_threshold, finger,
                         request_id, selection)


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
    """CLI wrapper around identify_uploads; returns the legacy shape.
    Accepts a video as readily as a photo."""
    path = Path(photo_path)
    data = path.read_bytes()
    if frames.is_video(path.name):
        upload = (path.name, None, data)
    else:
        upload = (path.name, imaging.decode_image(data, path.name), data)
    try:
        result = identify_uploads([upload], match_threshold)
    except StoreError as e:
        if e.code is ReasonCode.GALLERY_EMPTY:
            return {"match": None, "scores": {}, "note": "gallery is empty"}
        raise
    return {
        "match": result["match"]["name"] if result["match"] else None,
        "best_candidate": result["best_candidate"]["name"],
        "scores": {e["name"]: e["score"] for e in result["scores"]},
        "threshold": result["threshold"],
        "frames": result.get("frames"),
    }
