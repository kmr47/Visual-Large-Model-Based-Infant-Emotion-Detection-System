"""Daily / weekly emotion reports + privacy management endpoints.

Aggregation is computed from the anonymous ``emotion_events`` table only —
no images, no names, no raw tracks ever leave the persistence layer.

Endpoints
---------
- GET  /api/reports/daily?date=YYYY-MM-DD   (default: today)
- GET  /api/reports/weekly?start=YYYY-MM-DD (default: current Monday-start week)
- GET  /api/reports/summary                 meta info: retention, row counts
- POST /api/reports/purge                   admin: wipe all stored events
- POST /api/reports/seed                    admin: generate demo events for the
                                            current day (clearly labelled demo)

Access control: when ``EMOTION_API_TOKEN`` is set, the write endpoints
(purge/seed) require the bearer token; reads stay open for the demo UI.
"""

from __future__ import annotations

from collections import Counter
from datetime import date as date_cls
from datetime import datetime, time, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app import db
from app.config import Settings, get_settings
from app.security import require_api_token

router = APIRouter(prefix="/api/reports", tags=["reports"])

EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NEGATIVE = {"anger", "disgust", "fear", "sad"}

LABEL_CN = {
    "anger": "生气",
    "contempt": "轻蔑",
    "disgust": "厌恶",
    "fear": "害怕",
    "happy": "开心",
    "neutral": "平静",
    "sad": "伤心",
    "surprise": "惊讶",
}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _day_range(day: date_cls) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _week_range(anchor: date_cls) -> tuple[date_cls, date_cls]:
    monday = anchor - timedelta(days=anchor.weekday())
    return monday, monday + timedelta(days=6)


def _parse_day(value: str | None) -> date_cls:
    if not value:
        return datetime.now(timezone.utc).date()
    try:
        return date_cls.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date: {value!r}") from exc


def _parse_week_start(value: str | None) -> date_cls:
    day = _parse_day(value)
    return day - timedelta(days=day.weekday())


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _emotion_rows_to_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn raw emotion-event rows into aggregated statistics."""
    counts: Counter[str] = Counter()
    negative_counts: Counter[str] = Counter()
    hourly = [0] * 24
    alerts = 0
    children: set[str] = set()
    by_source: Counter[str] = Counter()
    for row in rows:
        emo = row["emotion"]
        counts[emo] += 1
        by_source[row.get("source", "image")] += 1
        children.add(row["child_anon_id"])
        if row.get("alert_triggered"):
            alerts += 1
        if emo in NEGATIVE:
            negative_counts[emo] += 1
        created = row.get("created_at", "")
        try:
            hour = datetime.fromisoformat(created).astimezone(timezone.utc).hour
            hourly[hour] += 1
        except Exception:
            pass

    total = int(sum(counts.values())) or 1
    return {
        "total_events": int(sum(counts.values())),
        "emotion_counts": {e: counts.get(e, 0) for e in EMOTIONS},
        "emotion_ratios": {e: round(counts.get(e, 0) / total, 4) for e in EMOTIONS},
        "negative_events": int(sum(negative_counts.values())),
        "negative_by_emotion": {e: negative_counts.get(e, 0) for e in NEGATIVE},
        "hourly_distribution": hourly,
        "alert_count": alerts,
        "distinct_children": len(children),
        "by_source": dict(by_source),
    }


def _previous_window_stats(window_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _emotion_rows_to_stats(window_rows)


# ---------------------------------------------------------------------------
# Report endpoints
# ---------------------------------------------------------------------------


@router.get("/daily")
def daily_report(
    date: str | None = Query(default=None, description="YYYY-MM-DD (UTC), default today"),
    compare: bool = Query(default=True, description="include previous-day comparison"),
) -> dict[str, Any]:
    """Aggregate one calendar day. All values are anonymous aggregates."""
    day = _parse_day(date)
    start, end = _day_range(day)
    rows = db.query_emotion_events_raw(start, end)

    stats = _emotion_rows_to_stats(rows)
    label = day.isoformat()

    payload: dict[str, Any] = {
        "type": "daily",
        "date": label,
        "label_cn": f"{day.month}月{day.day}日 情绪日报",
        "generated_at": _iso(datetime.now(timezone.utc)),
        "tz_note": "aggregation uses UTC day boundaries",
        "summary": {
            "main_emotion": max(
                ((e, stats["emotion_counts"][e]) for e in EMOTIONS),
                key=lambda kv: kv[1],
            )[0],
            "negative_ratio": round(
                stats["negative_events"] / max(1, stats["total_events"]), 4
            ),
            "peak_hour": stats["hourly_distribution"].index(
                max(stats["hourly_distribution"])
            ),
        },
        **stats,
    }

    if compare:
        prev_rows = db.query_emotion_events_raw(start - timedelta(days=1), start)
        prev = _previous_window_stats(prev_rows)
        payload["vs_previous_day"] = {
            "total_events_delta": stats["total_events"] - prev["total_events"],
            "negative_events_delta": stats["negative_events"] - prev["negative_events"],
            "previous_total_events": prev["total_events"],
            "previous_negative_events": prev["negative_events"],
        }
    return payload


@router.get("/weekly")
def weekly_report(
    start: str | None = Query(default=None, description="any date in the week (YYYY-MM-DD); week starts Monday"),
    compare: bool = Query(default=True, description="include previous-week comparison"),
) -> dict[str, Any]:
    """Aggregate a Monday-start week, with per-day breakdown."""
    monday = _parse_week_start(start)
    sunday = monday + timedelta(days=6)
    range_start, range_end = _day_range(monday), _day_range(sunday)
    rows = db.query_emotion_events_raw(range_start[0], range_end[1])

    days = []
    for offset in range(7):
        day = monday + timedelta(days=offset)
        ds, de = _day_range(day)
        day_rows = [r for r in rows if ds.isoformat(timespec="seconds") <= r["created_at"] < de.isoformat(timespec="seconds")]
        day_stats = _emotion_rows_to_stats(day_rows)
        days.append(
            {
                "date": day.isoformat(),
                "weekday": "一二三四五六日"[day.weekday()],
                "total_events": day_stats["total_events"],
                "negative_events": day_stats["negative_events"],
                "emotion_counts": day_stats["emotion_counts"],
            }
        )

    week_stats = _emotion_rows_to_stats(rows)
    payload: dict[str, Any] = {
        "type": "weekly",
        "week_start": monday.isoformat(),
        "week_end": sunday.isoformat(),
        "label_cn": f"{monday.month}月{monday.day}日-{sunday.month}月{sunday.day}日 情绪周报",
        "generated_at": _iso(datetime.now(timezone.utc)),
        "daily_series": days,
        **week_stats,
    }

    if compare:
        prev_start, prev_end = _day_range(monday - timedelta(days=7)), _day_range(sunday - timedelta(days=7))
        prev_rows = db.query_emotion_events_raw(prev_start[0], prev_end[1])
        prev = _previous_window_stats(prev_rows)
        payload["vs_previous_week"] = {
            "total_events_delta": week_stats["total_events"] - prev["total_events"],
            "negative_events_delta": week_stats["negative_events"] - prev["negative_events"],
            "previous_total_events": prev["total_events"],
            "previous_negative_events": prev["negative_events"],
        }
    return payload


@router.get("/privacy")
def privacy_status(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Expose the privacy posture + storage stats (for the thesis demo)."""
    from app.thumb_buffer import thumb_buffer
    return {
        "storage": db.db_stats(),
        "retention_days": settings.retention_days,
        "stores_images": False,
        "stores_bboxes": False,
        "pseudonymised": True,
        "pseudonym_algorithm": "HMAC-SHA256(camera_id|track_id, server_secret)",
        "access_control_enabled": bool(settings.api_token),
        # Session-only claim thumbnails: RAM-only, burn-on-claim, never persisted.
        "session_thumbnails": thumb_buffer.stats(),
    }


# ---------------------------------------------------------------------------
# Admin endpoints (token-protected when EMOTION_API_TOKEN is set)
# ---------------------------------------------------------------------------


@router.post("/purge", dependencies=[Depends(require_api_token)])
def purge_all_events() -> dict[str, Any]:
    """Delete every stored emotion/alert event (GDPR-style erasure demo)."""
    removed = db.purge_all()
    return {"status": "purged", **removed}


@router.post("/seed", dependencies=[Depends(require_api_token)])
def seed_demo_events(
    days: int = Query(default=1, ge=1, le=14),
    per_day: int = Query(default=40, ge=1, le=500),
) -> dict[str, Any]:
    """Generate clearly-labelled demo events across the last N days.

    Useful to fill the report pages before real usage. Rows carry
    source='demo' so reports can tell them apart from real detections.
    When roster children exist, each child gets a personalised daily stream
    (child_id set) so the per-child views have data to show; surplus rows
    remain anonymous.
    """
    import random

    secret = get_settings().privacy_secret
    seeded = 0
    now = datetime.now(timezone.utc)
    emotions = EMOTIONS
    tracks = ["person-1", "person-2", "person-3", "person-4"]

    roster = db.list_children(include_disabled=True)
    if not roster:
        # First-time demo: create sample roster entries so per-child views work.
        now0 = datetime.now(timezone.utc)
        for idx, (nick, cls) in enumerate(
            [("豆豆", "小一班"), ("乐乐", "小一班"), ("糖糖", "小二班")], start=1
        ):
            db.upsert_child(
                child_id=f"child-demo-{idx}",
                nickname=nick,
                class_name=cls,
                note="演示档案（可删除，删除后相关事件自动匿名化）",
                created_at=now0 - timedelta(days=7),
            )
        roster = db.list_children(include_disabled=True)
    # Build per-child weight profiles: every child gets one dominant emotion,
    # plus a personal negative tendency so reports differ visibly.
    profiles = []
    for c in roster:
        dominant = random.choices(
            emotions, weights=[4, 1, 2, 3, 26, 40, 7, 5], k=1
        )[0]
        neg_bias = random.uniform(0.1, 0.45)
        profiles.append((c, dominant, neg_bias))

    for offset in range(days):
        day = (now - timedelta(days=offset)).replace(hour=8, minute=0, second=0, microsecond=0)
        n = per_day
        # Per-child share if roster exists.
        if profiles:
            share = max(1, n // (len(profiles) + 1))
            for c, dominant, neg_bias in profiles:
                for i in range(share):
                    ts = day + timedelta(minutes=int((10 * 60) * i / max(1, share)))
                    r = random.random()
                    if r < neg_bias:
                        emo = random.choices(
                            ["sad", "fear", "anger", "neutral", "happy"],
                            weights=[5, 4, 3, 2, 1],
                            k=1,
                        )[0]
                    elif r < neg_bias + 0.55:
                        emo = dominant
                    else:
                        emo = random.choices(
                            emotions, weights=[2, 1, 1, 2, 24, 40, 4, 4], k=1
                        )[0]
                    conf = round(random.uniform(0.55, 0.97), 3)
                    neg = 1.0 if emo in NEGATIVE else round(random.uniform(0.02, 0.45), 3)
                    alert = 1 if (emo in NEGATIVE and neg >= 0.6 and random.random() < 0.4) else 0
                    db.insert_emotion_event(
                        event_uuid=f"demo-{offset}-{c['id']}-{i}",
                        child_id=c["id"],
                        child_anon_id=db.pseudonymise("camera-1", c["id"], secret),
                        camera_id="camera-1",
                        source="demo",
                        emotion=emo,
                        confidence=conf,
                        negative_score=neg,
                        alert_triggered=bool(alert),
                        created_at=ts,
                        event_request_id=f"demo-request-{offset}",
                        track_label="person-1",
                    )
                    seeded += 1
            n -= share * len(profiles)
        for i in range(n):
            minute_step = int((10 * 60) * i / max(1, n))  # spread over a "school day"
            ts = day + timedelta(minutes=minute_step)
            track = random.choice(tracks)
            anon = db.pseudonymise("camera-1", track, secret)
            emo = random.choices(
                emotions,
                weights=[5, 1, 2, 4, 30, 42, 8, 6],
                k=1,
            )[0]
            conf = round(random.uniform(0.55, 0.97), 3)
            neg = 1.0 if emo in NEGATIVE else round(random.uniform(0.02, 0.45), 3)
            alert = 1 if (emo in NEGATIVE and neg >= 0.6 and random.random() < 0.4) else 0
            db.insert_emotion_event(
                event_uuid=f"demo-{offset}-anon-{i}",
                child_anon_id=anon,
                camera_id="camera-1",
                source="demo",
                emotion=emo,
                confidence=conf,
                negative_score=neg,
                alert_triggered=bool(alert),
                created_at=ts,
                event_request_id=f"demo-request-{offset}",
                track_label=track,
            )
            seeded += 1
    return {"status": "seeded", "events": seeded, "roster_children": len(roster), "note": "source=demo rows only"}
