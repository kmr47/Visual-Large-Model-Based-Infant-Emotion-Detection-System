from __future__ import annotations

import asyncio
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings
from app import db
from app.schemas import AlertRecord, CameraConfig


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[Any] = set()

    async def connect(self, websocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket) -> None:
        self._connections.discard(websocket)

    async def broadcast(self, payload: dict) -> None:
        dead = []
        for ws in list(self._connections):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


class AppState:
    """In-memory runtime state, mirrored into the privacy-first SQLite store.

    Memory keeps the real-time dashboard/WebSocket behaviour unchanged;
    ``emotion_events`` / ``alert_events`` tables provide persistence for the
    daily/weekly reports. Only anonymous emotion events are persisted — never
    images, file names or bounding boxes.
    """

    def __init__(self) -> None:
        self.alerts: dict[str, AlertRecord] = {}
        self.cameras: dict[str, CameraConfig] = {
            "camera-1": CameraConfig(id="camera-1", name="Demo Camera", source="demo", location="classroom")
        }
        self.emotion_counts: Counter[str] = Counter()
        self.recent_events: deque[dict] = deque(maxlen=200)
        self.ws = ConnectionManager()
        self.lock = asyncio.Lock()
        self._settings = get_settings()

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #

    def _pseudonym(self, camera_id: str, track_id: str) -> str:
        return db.pseudonymise(camera_id, track_id, self._settings.privacy_secret)

    async def restore_alerts_from_db(self, limit: int = 200) -> int:
        """Load persisted alerts into memory (survives backend restarts)."""
        rows = db.load_recent_alerts(limit=limit)
        restored = 0
        for row in rows:
            try:
                record = AlertRecord(
                    id=row["id"],
                    camera_id=row["camera_id"],
                    track_id="",  # deliberately not persisted; not needed for display
                    emotion=row["emotion"],
                    negative_score=row["negative_score"],
                    severity=row["severity"],
                    status=row["status"],
                    message=row["message"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    snapshot={"duration_seconds": row["duration_seconds"]},
                )
            except Exception:
                continue
            self.alerts[record.id] = record
            restored += 1
        return restored

    # ------------------------------------------------------------------ #
    # Alert lifecycle (memory + db)
    # ------------------------------------------------------------------ #

    async def add_alert(self, alert: AlertRecord) -> AlertRecord:
        async with self.lock:
            self.alerts[alert.id] = alert
            self.recent_events.append({
                "type": "alert",
                "alert_id": alert.id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            # Persist alert metadata only (no image, no bbox).
            db.insert_alert_event(
                alert_id=alert.id,
                child_anon_id=alert.anonymous_id or self._pseudonym(alert.camera_id, alert.track_id),
                camera_id=alert.camera_id,
                emotion=alert.emotion.value if alert.emotion else None,
                negative_score=alert.negative_score,
                severity=alert.severity,
                status=alert.status.value,
                message=alert.message,
                duration_seconds=float(alert.snapshot.get("duration_seconds", 0.0) or 0.0),
                created_at=alert.created_at,
                updated_at=alert.updated_at,
            )
        await self.ws.broadcast({"type": "alert_created", "alert": alert.model_dump(mode="json")})
        return alert

    async def update_alert(self, alert_id: str, status) -> AlertRecord | None:
        async with self.lock:
            alert = self.alerts.get(alert_id)
            if alert is None:
                return None
            alert.status = status
            alert.updated_at = datetime.now(timezone.utc)
            self.alerts[alert_id] = alert
            db.update_alert_status(alert_id, status.value, alert.updated_at)
        await self.ws.broadcast({"type": "alert_updated", "alert": alert.model_dump(mode="json")})
        return alert

    # ------------------------------------------------------------------ #
    # Emotion detections (memory counter + anonymous event row)
    # ------------------------------------------------------------------ #

    def persist_detection(
        self,
        *,
        camera_id: str,
        track_id: str,
        emotion: str,
        confidence: float,
        negative_score: float,
        source: str = "image",
        alert_triggered: bool = False,
        created_at: str | datetime | None = None,
        event_request_id: str | None = None,
        track_label: str | None = None,
        anon_code: str | None = None,
        child_id: str | None = None,
    ) -> str:
        """Write one anonymous emotion event row.

        ``anon_code`` is the re-ID pseudonym when available; otherwise the
        request-scoped HMAC pseudonym is used. ``child_id`` links the row to a
        roster child when the identity was already claimed.
        """
        event_uuid = str(uuid.uuid4())
        db.insert_emotion_event(
            event_uuid=event_uuid,
            child_anon_id=anon_code or self._pseudonym(camera_id, track_id),
            camera_id=camera_id,
            source=source,
            emotion=emotion,
            confidence=confidence,
            negative_score=negative_score,
            alert_triggered=alert_triggered,
            created_at=created_at,
            child_id=child_id,
            event_request_id=event_request_id,
            track_label=track_label,
        )
        return event_uuid

    async def record_detection(self, emotion: str) -> None:
        async with self.lock:
            self.emotion_counts[emotion] += 1
            self.recent_events.append({
                "type": "detection",
                "emotion": emotion,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })


app_state = AppState()
