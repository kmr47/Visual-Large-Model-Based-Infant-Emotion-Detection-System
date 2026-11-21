"""Event log endpoints: browse inference batches and claim tracks.

Why this exists
---------------
Daily/weekly reports aggregate by calendar day, which hides *which upload*
produced a row. Teachers need to see recent inference batches (image/video
uploads) with their pseudonyms so they can claim a track for a roster child:

    upload video -> see batch rows (time, anon id, emotion)
                  -> claim track of that batch as "豆豆"
                  -> 豆豆's daily report now includes those rows

Display rule: the UI shows the *anonymous pseudonym* (e.g. c-638018c4f7f0da),
never the internal short-lived track label (person-1/2/...). The track label
is only kept server-side so a claim can address one physical face track.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import APIRouter, Query

from app import db

router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("")
def list_event_batches(
    limit: int = Query(default=15, ge=1, le=50),
    request_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return recent events grouped by inference request, newest first.

    Rows are merged per (request, track): a video produces many sampled rows
    for the same face; we collapse them into one card with a count and a
    primary emotion histogram. The visible identity is ``child_anon_id``.
    """
    batches: dict[str, dict[str, Any]] = {}
    track_index: dict[tuple[str, str], int] = {}  # (request, track_label) -> track slot

    with db._connect() as conn:
        sql = (
            "SELECT e.event_request_id AS rid, e.track_label, e.child_anon_id,"
            " e.child_id, e.source, e.emotion, e.confidence, e.negative_score,"
            " e.alert_triggered, e.created_at,"
            " c.nickname AS child_nickname, c.class_name AS child_class,"
            " p.id AS pending_id, p.candidate_anon_code AS pending_candidate,"
            " p.candidate_nickname AS pending_candidate_nickname,"
            " p.similarity AS pending_similarity"
            " FROM emotion_events e"
            " LEFT JOIN children c ON c.id = e.child_id"
            " LEFT JOIN pending_confirmations p ON p.event_request_id = e.event_request_id"
            "   AND p.track_label = e.track_label AND p.status = 'pending'"
        )
        params: list[Any] = []
        if request_id:
            sql += " WHERE e.event_request_id = ?"
            params.append(request_id)
        sql += " ORDER BY e.created_at DESC LIMIT ?"
        params.append(int(limit) * 200)
        rows = conn.execute(sql, params).fetchall()

    for row in rows:
        rid = row["rid"] or "unknown"
        batch = batches.get(rid)
        if batch is None:
            batch = batches[rid] = {
                "request_id": rid,
                "source": row["source"],
                "first_seen": row["created_at"],
                "last_seen": row["created_at"],
                "track_count": 0,
                "tracks": [],
            }
        track_label = row["track_label"] or "?"
        key = (rid, track_label)
        slot = track_index.get(key)
        if slot is None:
            slot = len(batch["tracks"])
            track_index[key] = slot
            batch["tracks"].append(
                {
                    "track_label": track_label,  # internal only; UI shows the pseudonym
                    "child_anon_id": row["child_anon_id"],
                    "child_id": row["child_id"],
                    "child_nickname": row["child_nickname"],
                    "child_class": row["child_class"],
                    "sample_count": 0,
                    "emotion_counts": Counter(),
                    "emotion": row["emotion"],
                    "confidence": round(float(row["confidence"] or 0.0), 3),
                    "negative_score": round(float(row["negative_score"] or 0.0), 3),
                    "alert_triggered": False,
                    "first_seen": row["created_at"],
                    "last_seen": row["created_at"],
                    "pending": None,
                }
            )
        track = batch["tracks"][slot]
        if row["pending_id"] and track["pending"] is None:
            track["pending"] = {
                "pending_id": row["pending_id"],
                "candidate_anon_code": row["pending_candidate"],
                "candidate_nickname": row["pending_candidate_nickname"],
                "similarity": round(float(row["pending_similarity"] or 0.0), 3),
            }
        track["sample_count"] += 1
        track["emotion_counts"][row["emotion"]] += 1
        # Latest sample drives the displayed emotion; max negative score is kept.
        track["emotion"] = row["emotion"]
        track["confidence"] = round(float(row["confidence"] or 0.0), 3)
        track["negative_score"] = max(track["negative_score"], round(float(row["negative_score"] or 0.0), 3))
        track["alert_triggered"] = track["alert_triggered"] or bool(row["alert_triggered"])
        track["last_seen"] = row["created_at"]
        batch["last_seen"] = row["created_at"]
        batch["first_seen"] = min(batch["first_seen"], row["created_at"])

    out = []
    try:
        from app.thumb_buffer import thumb_buffer
        thumb_ok = True
    except Exception:
        thumb_buffer = None
        thumb_ok = False
    for b in batches.values():
        for t in b["tracks"]:
            t["track_count_total"] = sum(t["emotion_counts"].values())
            t["emotion_histogram"] = {k: v for k, v in t["emotion_counts"].items()}
            # Session-only claim thumbnail availability (RAM buffer; never in DB).
            t["has_thumb"] = bool(thumb_ok and thumb_buffer.has(b["request_id"], t["track_label"])) if thumb_ok else False
        b["track_count"] = len(b["tracks"])
        out.append(b)
    out.sort(key=lambda b: b.get("last_seen") or "", reverse=True)
    return {"total_batches": len(out), "batches": out[: int(limit)]}
