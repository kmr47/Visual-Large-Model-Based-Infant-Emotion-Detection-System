"""Privacy-first SQLite persistence layer for the emotion detection system.

Design principles
-----------------
1. Data minimisation: we persist *emotion events* only. No face images, no video
   bytes, no file names and no raw bounding boxes ever touch the database.
2. Pseudonymisation: children are referenced by ``child_anon_id``, an
   HMAC-SHA256 digest of ``camera_id | track_id`` keyed with a server-side
   secret (``EMOTION_PRIVACY_SECRET``). The mapping is one-way: without the
   secret, a pseudonym cannot be linked back to a raw track identifier, let
   alone to a real child identity.
3. Retention & automatic cleanup: rows carry an ISO-8601 ``created_at``;
   the app lifespan task calls :func:`purge_expired` periodically so events
   older than ``EMOTION_RETENTION_DAYS`` are deleted.
4. Access control lives at the API layer (see ``app.security``), not here.

Threading note: every call opens a short-lived connection guarded by a
module-level lock, which is safe under FastAPI's thread pool and async
event loop while staying dependency-free.
"""

from __future__ import annotations

import hashlib
import hmac
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS children (
    id          TEXT PRIMARY KEY,           -- child_id (e.g. 'child-001')
    nickname    TEXT NOT NULL,
    class_name  TEXT NOT NULL DEFAULT '',
    note        TEXT NOT NULL DEFAULT '',
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS emotion_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_uuid      TEXT NOT NULL,
    child_id        TEXT,                    -- FK-ish -> children.id, NULL = anonymous
    child_anon_id   TEXT NOT NULL,
    camera_id       TEXT NOT NULL DEFAULT 'camera-1',
    source          TEXT NOT NULL DEFAULT 'image',   -- image | video | stream | demo
    emotion         TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.0,
    negative_score  REAL NOT NULL DEFAULT 0.0,
    alert_triggered INTEGER NOT NULL DEFAULT 0,
    event_request_id TEXT,                   -- inference request id (claim support)
    track_label     TEXT,                    -- track label inside that request
    created_at      TEXT NOT NULL                  -- ISO-8601 UTC
);

CREATE INDEX IF NOT EXISTS idx_events_created ON emotion_events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_anon   ON emotion_events(child_anon_id);
CREATE INDEX IF NOT EXISTS idx_events_emotion ON emotion_events(emotion);

CREATE TABLE IF NOT EXISTS alert_events (
    id              TEXT PRIMARY KEY,             -- same uuid as AlertRecord.id
    child_anon_id   TEXT NOT NULL,
    camera_id       TEXT NOT NULL DEFAULT 'camera-1',
    emotion         TEXT,
    negative_score  REAL NOT NULL DEFAULT 0.0,
    severity        TEXT NOT NULL DEFAULT 'warning',
    status          TEXT NOT NULL DEFAULT 'active',
    message         TEXT NOT NULL DEFAULT '',
    duration_seconds REAL NOT NULL DEFAULT 0.0,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_created ON alert_events(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_status  ON alert_events(status);

CREATE TABLE IF NOT EXISTS system_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS identities (
    id            TEXT PRIMARY KEY,             -- identity uuid
    anon_code     TEXT NOT NULL UNIQUE,         -- operator-facing pseudonym (c-...)
    camera_id     TEXT NOT NULL DEFAULT 'camera-1',
    child_id      TEXT,                         -- roster link (NULL = anonymous)
    embedding     BLOB NOT NULL,                -- packed float32 prototype vector (768 dims)
    sample_count  INTEGER NOT NULL DEFAULT 1,   -- number of embeddings folded in
    created_at    TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_identities_child ON identities(child_id);

CREATE TABLE IF NOT EXISTS pending_confirmations (
    id                   TEXT PRIMARY KEY,
    anon_code            TEXT NOT NULL,          -- temporary identity the events carry now
    candidate_anon_code  TEXT,                   -- best automatic candidate (may be NULL)
    candidate_nickname   TEXT,
    similarity           REAL,
    margin               REAL,
    sample_count         INTEGER NOT NULL DEFAULT 1,
    event_request_id     TEXT,
    track_label          TEXT,
    camera_id            TEXT NOT NULL DEFAULT 'camera-1',
    status               TEXT NOT NULL DEFAULT 'pending',  -- pending|resolved|dismissed|expired
    created_at           TEXT NOT NULL,
    resolved_at          TEXT
);

CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_confirmations(status);
"""

_lock = threading.Lock()
_db_path: Path | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def configure(db_path: Path) -> None:
    """Point the layer at a database file (must be called once at startup)."""
    global _db_path
    _db_path = db_path
    _db_path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    if _db_path is None:
        raise RuntimeError("db.configure() was not called before use")
    with _lock:
        conn = sqlite3.connect(str(_db_path), timeout=10)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Lightweight additive migrations for databases created by earlier versions."""
    # Children table may not exist on older DB files -> create if missing.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS children ("
        " id TEXT PRIMARY KEY, nickname TEXT NOT NULL, class_name TEXT NOT NULL DEFAULT '',"
        " note TEXT NOT NULL DEFAULT '', enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL)"
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(emotion_events)").fetchall()}
    if "child_id" not in cols:
        conn.execute("ALTER TABLE emotion_events ADD COLUMN child_id TEXT")
    if "event_request_id" not in cols:
        conn.execute("ALTER TABLE emotion_events ADD COLUMN event_request_id TEXT")
    if "track_label" not in cols:
        conn.execute("ALTER TABLE emotion_events ADD COLUMN track_label TEXT")
    if "child_id" in cols or "child_id" in {r[1] for r in conn.execute("PRAGMA table_info(emotion_events)").fetchall()}:
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_child ON emotion_events(child_id)"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_alerts_created ON alert_events(created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_alerts_status ON alert_events(status)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS identities ("
        " id TEXT PRIMARY KEY, anon_code TEXT NOT NULL UNIQUE, camera_id TEXT NOT NULL DEFAULT 'camera-1',"
        " child_id TEXT, embedding BLOB NOT NULL, sample_count INTEGER NOT NULL DEFAULT 1,"
        " created_at TEXT NOT NULL, last_seen_at TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_identities_child ON identities(child_id)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pending_confirmations ("
        " id TEXT PRIMARY KEY, anon_code TEXT NOT NULL, candidate_anon_code TEXT, candidate_nickname TEXT,"
        " similarity REAL, margin REAL, sample_count INTEGER NOT NULL DEFAULT 1,"
        " event_request_id TEXT, track_label TEXT, camera_id TEXT NOT NULL DEFAULT 'camera-1',"
        " status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL, resolved_at TEXT)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_confirmations(status)"
    )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pseudonymise(camera_id: str, track_id: str, secret: str) -> str:
    """One-way child pseudonym: HMAC-SHA256(camera|track, secret), hex-truncated."""
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{camera_id}|{track_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"c-{digest[:14]}"


def parse_created_at(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value) if value else datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def insert_emotion_event(
    event_uuid: str,
    child_anon_id: str,
    camera_id: str,
    source: str,
    emotion: str,
    confidence: float,
    negative_score: float,
    alert_triggered: bool,
    created_at: str | datetime | None = None,
    child_id: str | None = None,
    event_request_id: str | None = None,
    track_label: str | None = None,
) -> None:
    created = parse_created_at(created_at).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO emotion_events "
            "(event_uuid, child_id, child_anon_id, camera_id, source, emotion, confidence,"
            " negative_score, alert_triggered, event_request_id, track_label, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event_uuid,
                child_id,
                child_anon_id,
                camera_id,
                source,
                emotion,
                float(confidence),
                float(negative_score),
                1 if alert_triggered else 0,
                event_request_id,
                track_label,
                created,
            ),
        )


def insert_alert_event(
    alert_id: str,
    child_anon_id: str,
    camera_id: str,
    emotion: str | None,
    negative_score: float,
    severity: str,
    status: str,
    message: str,
    duration_seconds: float,
    created_at: str | datetime,
    updated_at: str | datetime | None = None,
) -> None:
    created = parse_created_at(created_at).isoformat()
    updated = parse_created_at(updated_at or created_at).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO alert_events "
            "(id, child_anon_id, camera_id, emotion, negative_score, severity, status,"
            " message, duration_seconds, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at,"
            " child_anon_id=excluded.child_anon_id",
            (
                alert_id,
                child_anon_id,
                camera_id,
                emotion,
                float(negative_score),
                severity,
                status,
                message,
                float(duration_seconds),
                created,
                updated,
            ),
        )


def update_alert_status(alert_id: str, status: str, updated_at: str | datetime | None = None) -> None:
    updated = parse_created_at(updated_at).isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute(
            "UPDATE alert_events SET status=?, updated_at=? WHERE id=?",
            (status, updated, alert_id),
        )


def load_recent_alerts(limit: int = 200) -> list[dict[str, Any]]:
    """Used at startup so persisted alerts survive a backend restart."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM alert_events ORDER BY created_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Retention / cleanup
# ---------------------------------------------------------------------------


def purge_expired(retention_days: int) -> int:
    """Delete events and alerts older than ``retention_days``. Returns rows removed.

    Stale anonymous identities (no recent sighting) are deleted too so the
    biometric-template store obeys the same retention window as event data.
    Pending confirmations past retention are closed as ``expired``.
    """
    if retention_days <= 0:
        return 0
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    with _connect() as conn:
        cur1 = conn.execute("DELETE FROM emotion_events WHERE created_at < ?", (cutoff,))
        cur2 = conn.execute("DELETE FROM alert_events WHERE created_at < ?", (cutoff,))
        cur3 = conn.execute("DELETE FROM identities WHERE last_seen_at < ?", (cutoff,))
        cur4 = conn.execute(
            "UPDATE pending_confirmations SET status='expired' "
            "WHERE status='pending' AND created_at < ?",
            (cutoff,),
        )
        return int(cur1.rowcount) + int(cur2.rowcount) + int(cur3.rowcount) + int(cur4.rowcount)


# ---------------------------------------------------------------------------
# Reads / aggregation for the daily & weekly reports
# ---------------------------------------------------------------------------


def _window_bounds(window: str, anchor: str | None) -> tuple[datetime, datetime, str]:
    """Resolve a report window into [start, end) UTC plus a display label.

    ``window``: ``today`` | ``yesterday`` | ``week`` | ``last_week``
    Anchor time comes from the server clock (UTC). Display label is the
    calendar date the operator cares about (local day boundaries would need a
    tz parameter; we keep UTC for simplicity and document it in responses).
    """
    if anchor:
        base = datetime.fromisoformat(anchor).replace(tzinfo=timezone.utc)
    else:
        base = datetime.now(timezone.utc)
    base = base.replace(hour=0, minute=0, second=0, microsecond=0)
    day = timedelta(days=1)
    if window == "today":
        return base, base + day, base.date().isoformat()
    if window == "yesterday":
        return base - day, base, (base - day).date().isoformat()
    # Monday-start week
    monday = base - timedelta(days=base.weekday())
    if window == "week":
        return monday, monday + 7 * day, monday.date().isoformat()
    if window == "last_week":
        return monday - 7 * day, monday, (monday - 7 * day).date().isoformat()
    raise ValueError(f"unknown window: {window}")


def query_emotion_events(
    window: str = "today", anchor: str | None = None
) -> list[dict[str, Any]]:
    start, end, _label = _window_bounds(window, anchor)
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM emotion_events WHERE created_at >= ? AND created_at < ? "
            "ORDER BY created_at ASC",
            (start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
        ).fetchall()
    return [dict(r) for r in rows]


def query_alert_events(
    window: str = "today", anchor: str | None = None
) -> list[dict[str, Any]]:
    start, end, _label = _window_bounds(window, anchor)
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM alert_events WHERE created_at >= ? AND created_at < ? "
            "ORDER BY created_at ASC",
            (start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
        ).fetchall()
    return [dict(r) for r in rows]


def query_emotion_events_raw(start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Query emotion events inside an explicit [start, end) UTC window."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM emotion_events WHERE created_at >= ? AND created_at < ? "
            "ORDER BY created_at ASC",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    return [dict(r) for r in rows]


def query_alert_events_raw(start: datetime, end: datetime) -> list[dict[str, Any]]:
    """Query alert events inside an explicit [start, end) UTC window."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM alert_events WHERE created_at >= ? AND created_at < ? "
            "ORDER BY created_at ASC",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    return [dict(r) for r in rows]


def distinct_child_count(window: str = "today", anchor: str | None = None) -> int:
    start, end, _label = _window_bounds(window, anchor)
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT child_anon_id) AS n FROM emotion_events "
            "WHERE created_at >= ? AND created_at < ?",
            (start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
        ).fetchone()
    return int(row["n"]) if row else 0


def count_rows(window: str = "today", anchor: str | None = None) -> int:
    """Count emotion events in a window; used for change vs previous period."""
    start, end, _label = _window_bounds(window, anchor)
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM emotion_events "
            "WHERE created_at >= ? AND created_at < ?",
            (start.isoformat(timespec="seconds"), end.isoformat(timespec="seconds")),
        ).fetchone()
    return int(row["n"]) if row else 0


def db_stats() -> dict[str, Any]:
    """Small introspection endpoint used by /health/privacy to prove the DB works."""
    with _connect() as conn:
        events = conn.execute("SELECT COUNT(*) FROM emotion_events").fetchone()[0]
        alerts = conn.execute("SELECT COUNT(*) FROM alert_events").fetchone()[0]
        identities = conn.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM pending_confirmations WHERE status='pending'"
        ).fetchone()[0]
        schema = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
    return {
        "events": int(events),
        "alerts": int(alerts),
        "identities": int(identities),
        "pending_confirmations": int(pending),
        "tables": int(schema),
    }


def purge_all() -> dict[str, int]:
    """Manual wipe (only exposed behind the admin token)."""
    with _connect() as conn:
        e = conn.execute("DELETE FROM emotion_events").rowcount
        a = conn.execute("DELETE FROM alert_events").rowcount
        i = conn.execute("DELETE FROM identities").rowcount
        p = conn.execute("DELETE FROM pending_confirmations").rowcount
    return {"emotion_events": int(e), "alert_events": int(a), "identities": int(i), "pending_confirmations": int(p)}


def touch_meta(key: str, value: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO system_meta (key, value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


# ---------------------------------------------------------------------------
# Children roster (pseudonym <-> display-name mapping controlled by teachers)
# ---------------------------------------------------------------------------


def _child_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["enabled"] = bool(d.get("enabled"))
    return d


def list_children(include_disabled: bool = True) -> list[dict[str, Any]]:
    with _connect() as conn:
        if include_disabled:
            rows = conn.execute("SELECT * FROM children ORDER BY class_name, nickname").fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM children WHERE enabled=1 ORDER BY class_name, nickname"
            ).fetchall()
    return [_child_row_to_dict(r) for r in rows]


def get_child(child_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM children WHERE id=?", (child_id,)).fetchone()
    return _child_row_to_dict(row) if row else None


def upsert_child(
    child_id: str,
    nickname: str,
    class_name: str = "",
    note: str = "",
    enabled: bool = True,
    created_at: str | datetime | None = None,
) -> dict[str, Any]:
    created = parse_created_at(created_at).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO children (id, nickname, class_name, note, enabled, created_at) "
            "VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET nickname=excluded.nickname, "
            "class_name=excluded.class_name, note=excluded.note, enabled=excluded.enabled",
            (child_id, nickname, class_name, note, 1 if enabled else 0, created),
        )
    row = get_child(child_id)
    assert row is not None
    return row


def delete_child(child_id: str) -> int:
    """Delete a child and de-link (anonymise) all their events."""
    with _connect() as conn:
        conn.execute("UPDATE emotion_events SET child_id=NULL WHERE child_id=?", (child_id,))
        cur = conn.execute("DELETE FROM children WHERE id=?", (child_id,))
    return int(cur.rowcount)


def claim_events(child_id: str, event_request_id: str, track_label: str) -> int:
    """Link the events of one request+track to a roster child (behavioural claim)."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE emotion_events SET child_id=? "
            "WHERE event_request_id=? AND track_label=? AND child_id IS NULL",
            (child_id, event_request_id, track_label),
        )
    return int(cur.rowcount)


def unclaim_events(child_id: str, event_request_id: str | None = None, track_label: str | None = None) -> int:
    """Remove the link (events return to anonymous)."""
    sql = "UPDATE emotion_events SET child_id=NULL WHERE child_id=?"
    params: list[Any] = [child_id]
    if event_request_id:
        sql += " AND event_request_id=?"
        params.append(event_request_id)
        if track_label:
            sql += " AND track_label=?"
            params.append(track_label)
    with _connect() as conn:
        cur = conn.execute(sql, params)
    return int(cur.rowcount)


def count_events_for_child(child_id: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM emotion_events WHERE child_id=?", (child_id,)
        ).fetchone()
    return int(row["n"]) if row else 0


def query_events_for_child_raw(
    child_id: str, start: datetime, end: datetime
) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM emotion_events WHERE child_id=? AND created_at >= ? AND created_at < ? "
            "ORDER BY created_at ASC",
            (child_id, start.isoformat(), end.isoformat()),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Anonymous identity store (re-ID via face embedding prototypes)
# ---------------------------------------------------------------------------
# Identities hold ONE L2-normalised prototype vector per anonymous child.
# No images, video bytes, file names or bounding boxes are stored. The
# prototype is an averaged face descriptor used only to recognise "the same
# anonymous child as before" across videos.


def insert_identity(
    identity_id: str,
    anon_code: str,
    camera_id: str,
    embedding: bytes,
    child_id: str | None = None,
    sample_count: int = 1,
    created_at: str | datetime | None = None,
) -> None:
    created = parse_created_at(created_at).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO identities (id, anon_code, camera_id, child_id, embedding,"
            " sample_count, created_at, last_seen_at) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(anon_code) DO UPDATE SET last_seen_at=excluded.last_seen_at",
            (identity_id, anon_code, camera_id, child_id, embedding, int(sample_count), created, created),
        )


def update_identity_last_seen(
    anon_code: str,
    embedding: bytes,
    sample_count: int | None = None,
    last_seen_at: str | datetime | None = None,
) -> None:
    seen = parse_created_at(last_seen_at).isoformat()
    with _connect() as conn:
        if sample_count is None:
            conn.execute(
                "UPDATE identities SET embedding=?, last_seen_at=? WHERE anon_code=?",
                (embedding, seen, anon_code),
            )
        else:
            conn.execute(
                "UPDATE identities SET embedding=?, sample_count=?, last_seen_at=? WHERE anon_code=?",
                (embedding, int(sample_count), seen, anon_code),
            )


def get_identity(anon_code: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, anon_code, camera_id, child_id, embedding, sample_count, created_at, last_seen_at "
            "FROM identities WHERE anon_code=?",
            (anon_code,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["embedding"] = bytes(d["embedding"])
    return d


def get_identity_by_child(child_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, anon_code, camera_id, child_id, embedding, sample_count, created_at, last_seen_at "
            "FROM identities WHERE child_id=? LIMIT 1",
            (child_id,),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["embedding"] = bytes(d["embedding"])
    return d


def list_identities(limit: int = 200, include_embedding: bool = False) -> list[dict[str, Any]]:
    cols = "id, anon_code, camera_id, child_id, sample_count, created_at, last_seen_at"
    if include_embedding:
        cols = cols.replace("sample_count", "embedding, sample_count")
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT {cols} FROM identities ORDER BY last_seen_at DESC LIMIT ?", (int(limit),)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if include_embedding:
            d["embedding"] = bytes(d["embedding"])
        out.append(d)
    return out


def link_identity_to_child(anon_code: str, child_id: str) -> int:
    """Link an anonymous identity (and all its events) to a roster child."""
    with _connect() as conn:
        conn.execute("UPDATE identities SET child_id=? WHERE anon_code=?", (child_id, anon_code))
        cur = conn.execute(
            "UPDATE emotion_events SET child_id=? WHERE child_anon_id=? AND child_id IS NULL",
            (child_id, anon_code),
        )
    return int(cur.rowcount)


def unlink_identity(anon_code: str) -> int:
    """Remove the child link; events return to anonymous."""
    with _connect() as conn:
        conn.execute("UPDATE identities SET child_id=NULL WHERE anon_code=?", (anon_code,))
        cur = conn.execute(
            "UPDATE emotion_events SET child_id=NULL WHERE child_anon_id=?", (anon_code,)
        )
    return int(cur.rowcount)


def delete_identity(anon_code: str) -> int:
    """Delete an identity prototype; its events remain as anonymous rows."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM identities WHERE anon_code=?", (anon_code,))
    return int(cur.rowcount)


def count_identity_events(anon_code: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM emotion_events WHERE child_anon_id=?", (anon_code,)
        ).fetchone()
    return int(row["n"]) if row else 0


def merge_events_anon(
    old_anon_code: str,
    new_anon_code: str,
    event_request_id: str | None = None,
    track_label: str | None = None,
) -> int:
    """Re-point events from one pseudonym to another (pending-confirmation link).

    When scoped to request+track, only that upload's rows move; otherwise all
    rows of the temporary pseudonym are merged.
    """
    sql = "UPDATE emotion_events SET child_anon_id=? WHERE child_anon_id=?"
    params: list[Any] = [new_anon_code, old_anon_code]
    if event_request_id:
        sql += " AND event_request_id=?"
        params.append(event_request_id)
        if track_label:
            sql += " AND track_label=?"
            params.append(track_label)
    with _connect() as conn:
        cur = conn.execute(sql, params)
    return int(cur.rowcount)


def update_alert_anon(old_anon_code: str, new_anon_code: str, since_iso: str) -> int:
    """Re-point alert rows written under a temporary pseudonym to the final one."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE alert_events SET child_anon_id=? WHERE child_anon_id=? AND created_at >= ?",
            (new_anon_code, old_anon_code, since_iso),
        )
    return int(cur.rowcount)


def insert_pending_confirmation(
    pending_id: str,
    anon_code: str,
    candidate_anon_code: str | None,
    candidate_nickname: str | None,
    similarity: float,
    margin: float,
    sample_count: int,
    event_request_id: str | None,
    track_label: str | None,
    camera_id: str = "camera-1",
    created_at: str | datetime | None = None,
) -> None:
    created = parse_created_at(created_at).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO pending_confirmations "
            "(id, anon_code, candidate_anon_code, candidate_nickname, similarity, margin,"
            " sample_count, event_request_id, track_label, camera_id, status, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                pending_id,
                anon_code,
                candidate_anon_code,
                candidate_nickname,
                float(similarity),
                float(margin),
                int(sample_count),
                event_request_id,
                track_label,
                camera_id,
                "pending",
                created,
            ),
        )


def list_pending_confirmations(status: str = "pending", limit: int = 100) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT p.*, i.child_id AS candidate_child_id, c.nickname AS candidate_roster_nickname "
            "FROM pending_confirmations p "
            "LEFT JOIN identities i ON i.anon_code = p.candidate_anon_code "
            "LEFT JOIN children c ON c.id = i.child_id "
            "WHERE p.status=? ORDER BY p.created_at DESC LIMIT ?",
            (status, int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]


def get_pending_confirmation(pending_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM pending_confirmations WHERE id=?", (pending_id,)
        ).fetchone()
    return dict(row) if row else None


def resolve_pending_confirmation(pending_id: str, status: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE pending_confirmations SET status=?, resolved_at=? WHERE id=?",
            (status, now_iso(), pending_id),
        )
