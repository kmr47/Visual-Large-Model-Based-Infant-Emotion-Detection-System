"""Children roster: teacher-managed pseudonym <-> display-name mapping.

Privacy story (defence-friendly)
--------------------------------
- The detection pipeline stores *anonymous* events only.
- A teacher may create a roster entry (nickname, class) and then *claim* the
  events of one inference request+track for that child (a behavioural link,
  NOT face recognition — no biometric data is involved).
- Deleting the roster entry anonymises all linked events again
  (GDPR-style right to erasure), returning rows to pseudonym-only state.

Write endpoints require the API token when ``EMOTION_API_TOKEN`` is set.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query

from app import db
from app.security import require_api_token

router = APIRouter(prefix="/api/children", tags=["children"])

EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NEGATIVE = {"anger", "disgust", "fear", "sad"}
LABEL_CN = {
    "anger": "生气", "contempt": "轻蔑", "disgust": "厌恶", "fear": "害怕",
    "happy": "开心", "neutral": "平静", "sad": "伤心", "surprise": "惊讶",
}


# ---------------------------------------------------------------------------
# Roster CRUD
# ---------------------------------------------------------------------------


@router.get("")
def list_children(include_disabled: bool = True) -> list[dict[str, Any]]:
    children = db.list_children(include_disabled=include_disabled)
    out = []
    for c in children:
        out.append({**c, "event_count": db.count_events_for_child(c["id"])})
    return out


@router.post("", dependencies=[Depends(require_api_token)])
def create_child(payload: dict[str, Any]) -> dict[str, Any]:
    nickname = (payload.get("nickname") or "").strip()
    if not nickname:
        raise HTTPException(status_code=400, detail="nickname is required")
    child_id = f"child-{uuid4().hex[:8]}"
    return db.upsert_child(
        child_id=child_id,
        nickname=nickname,
        class_name=(payload.get("class_name") or "").strip(),
        note=(payload.get("note") or "").strip(),
        enabled=bool(payload.get("enabled", True)),
    )


@router.patch("/{child_id}", dependencies=[Depends(require_api_token)])
def update_child(child_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    existing = db.get_child(child_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Child not found")
    return db.upsert_child(
        child_id=child_id,
        nickname=(payload.get("nickname") or existing["nickname"]).strip(),
        class_name=(payload.get("class_name") if "class_name" in payload else existing["class_name"]).strip(),
        note=(payload.get("note") if "note" in payload else existing["note"]).strip(),
        enabled=bool(payload.get("enabled", existing["enabled"])),
    )


@router.delete("/{child_id}", dependencies=[Depends(require_api_token)])
def remove_child(child_id: str) -> dict[str, Any]:
    removed = db.delete_child(child_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Child not found")
    return {"status": "deleted", "child_id": child_id, "anonymised_events": True}


# ---------------------------------------------------------------------------
# Claim / unclaim (behavioural link, no biometrics)
# ---------------------------------------------------------------------------


@router.post("/claim", dependencies=[Depends(require_api_token)])
def claim_child(payload: dict[str, Any]) -> dict[str, Any]:
    child_id = (payload.get("child_id") or "").strip()
    request_id = (payload.get("request_id") or "").strip()
    track_label = (payload.get("track_label") or "").strip()
    anon_code = (payload.get("anon_code") or "").strip()
    if not child_id:
        raise HTTPException(status_code=400, detail="child_id is required")
    if db.get_child(child_id) is None:
        raise HTTPException(status_code=404, detail="Child not found")

    # Preferred: claim the cross-video anonymous identity itself. Future
    # sightings of that identity then link to the child automatically.
    if anon_code and not request_id:
        if db.get_identity(anon_code) is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        linked = db.link_identity_to_child(anon_code, child_id)
        return {
            "status": "claimed",
            "child_id": child_id,
            "anon_code": anon_code,
            "linked_events": linked,
            "note": "behavioural link to an anonymous identity; no biometric data stored",
        }

    # Back-compatible: claim one request+track (single upload batch).
    if not request_id or not track_label:
        raise HTTPException(status_code=400, detail="request_id and track_label (or anon_code) are required")
    linked = db.claim_events(child_id, request_id, track_label)
    return {
        "status": "claimed",
        "child_id": child_id,
        "request_id": request_id,
        "track_label": track_label,
        "linked_events": linked,
        "note": "behavioural link only; no biometric data stored",
    }


@router.post("/{child_id}/unclaim", dependencies=[Depends(require_api_token)])
def unclaim_child(
    child_id: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if db.get_child(child_id) is None:
        raise HTTPException(status_code=404, detail="Child not found")
    payload = payload or {}
    request_id = (payload.get("request_id") or "").strip() or None
    track_label = (payload.get("track_label") or "").strip() or None
    unlinked = db.unclaim_events(child_id, request_id, track_label)
    return {"status": "unclaimed", "child_id": child_id, "unlinked_events": unlinked}


# ---------------------------------------------------------------------------
# Per-child report
# ---------------------------------------------------------------------------


def _day_range(day) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


@router.get("/{child_id}/daily")
def child_daily_report(
    child_id: str,
    date: str | None = Query(default=None, description="YYYY-MM-DD (UTC), default today"),
) -> dict[str, Any]:
    """Per-child daily emotion report (aggregates over linked events only)."""
    child = db.get_child(child_id)
    if child is None:
        raise HTTPException(status_code=404, detail="Child not found")
    if not date:
        day = datetime.now(timezone.utc).date()
    else:
        try:
            day = datetime.fromisoformat(date).date()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid date: {date!r}") from exc

    start, end = _day_range(day)
    rows = db.query_events_for_child_raw(child_id, start, end)

    counts = {e: 0 for e in EMOTIONS}
    hourly = [0] * 24
    negative_counts = {e: 0 for e in NEGATIVE}
    alerts = 0
    for row in rows:
        emo = row["emotion"]
        if emo in counts:
            counts[emo] += 1
        if row.get("alert_triggered"):
            alerts += 1
            if emo in negative_counts:
                negative_counts[emo] += 1
        try:
            hour = datetime.fromisoformat(row["created_at"]).astimezone(timezone.utc).hour
            hourly[hour] += 1
        except Exception:
            pass
    total = sum(counts.values())
    negative_total = sum(negative_counts.values())

    return {
        "type": "child_daily",
        "child_id": child_id,
        "nickname": child["nickname"],
        "class_name": child["class_name"],
        "date": day.isoformat(),
        "label_cn": f"{child['nickname']} {day.month}月{day.day}日 情绪日报",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_events": total,
        "emotion_counts": counts,
        "emotion_ratios": {e: round(counts[e] / total, 4) if total else 0.0 for e in EMOTIONS},
        "negative_events": negative_total,
        "negative_ratio": round(negative_total / total, 4) if total else 0.0,
        "negative_by_emotion": negative_counts,
        "hourly_distribution": hourly,
        "alert_count": alerts,
        "main_emotion": max(counts, key=counts.get) if total else None,
        "privacy_note": "aggregates only teacher-linked events; unlinked events stay anonymous",
    }
