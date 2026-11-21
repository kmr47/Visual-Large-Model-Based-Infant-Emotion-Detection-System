from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field


EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NEGATIVE_EMOTIONS = {"anger", "disgust", "fear", "sad"}


class EmotionName(str, Enum):
    anger = "anger"
    contempt = "contempt"
    disgust = "disgust"
    fear = "fear"
    happy = "happy"
    neutral = "neutral"
    sad = "sad"
    surprise = "surprise"


class AlertStatus(str, Enum):
    active = "active"
    acknowledged = "acknowledged"
    resolved = "resolved"


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str
    inference_mode: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EmotionScore(BaseModel):
    emotion: EmotionName
    confidence: float = Field(ge=0, le=1)
    label_cn: str | None = None


class FaceDetection(BaseModel):
    track_id: str = "person-1"  # internal short-lived tracker key; not shown as identity
    anonymous_id: str | None = None  # privacy pseudonym shown to operators
    bbox: list[int] = Field(default_factory=lambda: [0, 0, 0, 0])
    face_confidence: float = Field(default=1.0, ge=0, le=1)
    emotions: list[EmotionScore]
    primary_emotion: EmotionName
    primary_confidence: float = Field(ge=0, le=1)
    negative_score: float = Field(ge=0, le=1)


class TrackIdentityDecision(BaseModel):
    """One per detected track: the anonymous identity assigned by re-ID.

    ``decision`` is ``match`` (reused an existing pseudonym), ``new`` (created
    a fresh anonymous identity) or ``pending`` (ambiguous; teacher may link it
    to a candidate later). No embedding data is included.
    """
    track_id: str
    anon_code: str
    child_id: str | None = None
    child_nickname: str | None = None
    decision: str = "new"  # match | new | pending
    similarity: float | None = None
    candidate_anon_code: str | None = None
    candidate_nickname: str | None = None
    margin: float | None = None
    sample_count: int = 1
    pending_id: str | None = None


class InferenceResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    source: str = "upload"
    mode: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    num_faces: int = 0
    detections: list[FaceDetection] = Field(default_factory=list)
    alert_triggered: bool = False
    alerts: list["AlertRecord"] = Field(default_factory=list)
    track_identity_decisions: list[TrackIdentityDecision] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class FramePayload(BaseModel):
    camera_id: str = "camera-1"
    track_id: str = "person-1"
    timestamp: datetime | None = None
    detections: list[FaceDetection] = Field(default_factory=list)


class AlertRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    camera_id: str = "camera-1"
    track_id: str = "person-1"
    anonymous_id: str | None = None  # privacy pseudonym for display
    emotion: EmotionName | None = None
    negative_score: float = Field(default=0.0, ge=0, le=1)
    severity: str = "warning"
    status: AlertStatus = AlertStatus.active
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    snapshot: dict[str, Any] = Field(default_factory=dict)


class AlertUpdate(BaseModel):
    status: AlertStatus


class CameraConfig(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    source: str = "demo"
    enabled: bool = True
    location: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CameraCreate(BaseModel):
    name: str
    source: str = "demo"
    location: str | None = None


class StatsResponse(BaseModel):
    total_alerts: int
    active_alerts: int
    emotion_counts: dict[str, int]
    recent_alerts: list[AlertRecord]


class FrameResult(BaseModel):
    frame_index: int
    timestamp_sec: float
    detections: list[FaceDetection] = Field(default_factory=list)
    alert_triggered: bool = False


class VideoInferenceResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    filename: str
    mode: str
    fps: float = 0
    total_frames: int = 0
    processed_frames: int = 0
    duration_sec: float = 0
    frames: list[FrameResult] = Field(default_factory=list)
    aggregate: dict[str, Any] = Field(default_factory=dict)
    track_identity_decisions: list[TrackIdentityDecision] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


InferenceResponse.model_rebuild()
