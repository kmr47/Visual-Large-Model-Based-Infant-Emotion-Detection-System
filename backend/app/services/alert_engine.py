from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from app.config import Settings
from app.schemas import AlertRecord, FaceDetection, NEGATIVE_EMOTIONS


@dataclass
class TrackState:
    first_negative_at: datetime | None = None
    last_negative_at: datetime | None = None
    last_alert_at: datetime | None = None
    recent_scores: list[float] = field(default_factory=list)


class AlertEngine:
    """Rule-based negative emotion alert engine.

    The engine keeps short-lived in-memory state per camera/track. For a real
    deployment this can later move to Redis or SQLite-backed state.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.tracks: dict[tuple[str, str], TrackState] = {}

    def evaluate(self, camera_id: str, detections: list[FaceDetection]) -> list[AlertRecord]:
        now = datetime.now(timezone.utc)
        alerts: list[AlertRecord] = []

        for det in detections:
            key = (camera_id, det.track_id)
            state = self.tracks.setdefault(key, TrackState())
            is_negative = (
                det.primary_emotion.value in NEGATIVE_EMOTIONS
                or det.negative_score >= self.settings.negative_threshold
            )

            state.recent_scores.append(det.negative_score)
            state.recent_scores = state.recent_scores[-30:]

            if is_negative:
                if state.first_negative_at is None:
                    state.first_negative_at = now
                state.last_negative_at = now
            else:
                state.first_negative_at = None
                state.last_negative_at = None
                continue

            duration = (now - state.first_negative_at).total_seconds() if state.first_negative_at else 0.0
            cooldown_ok = (
                state.last_alert_at is None
                or (now - state.last_alert_at).total_seconds() >= self.settings.alert_cooldown_seconds
            )
            duration_ok = duration >= self.settings.alert_duration_seconds

            # For single-image demos, duration may never accumulate. We still expose
            # a low-severity alert if the score is very high so the UI can demonstrate
            # the alert panel without running a long video stream.
            demo_instant = det.negative_score >= max(0.85, self.settings.negative_threshold + 0.2)

            if cooldown_ok and (duration_ok or demo_instant):
                severity = "critical" if det.negative_score >= 0.85 else "warning"
                display_name = det.anonymous_id or det.track_id
                alert = AlertRecord(
                    camera_id=camera_id,
                    track_id=det.track_id,
                    anonymous_id=det.anonymous_id,
                    emotion=det.primary_emotion,
                    negative_score=det.negative_score,
                    severity=severity,
                    message=(
                        f"检测到 {display_name} 持续/高置信负面情绪: "
                        f"{det.primary_emotion.value}, score={det.negative_score:.2f}"
                    ),
                    snapshot={
                        "duration_seconds": duration,
                        "bbox": det.bbox,
                        "primary_confidence": det.primary_confidence,
                        "recent_negative_score_avg": sum(state.recent_scores) / len(state.recent_scores),
                    },
                )
                state.last_alert_at = now
                alerts.append(alert)

        return alerts
