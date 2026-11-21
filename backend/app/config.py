from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Backend runtime configuration.

    Inference modes:
    - mock: deterministic local demo results, no model required
    - local: real on-device inference (MediaPipe face detection + local ViT
      emotion model; no cloud dependency)
    - replay: replay rows from exported prediction/error CSV files
    - cloud: call a remote AutoDL inference API
    """

    app_name: str = "Child Emotion Detection Backend"
    app_version: str = "0.2.0"  # v0.2: privacy-first SQLite persistence + daily/weekly reports
    env: str = "dev"

    inference_mode: str = Field(default="mock", validation_alias="EMOTION_INFERENCE_MODE")
    cloud_inference_url: str | None = Field(default=None, validation_alias="EMOTION_CLOUD_INFERENCE_URL")
    cloud_timeout_seconds: float = Field(default=20.0, validation_alias="EMOTION_CLOUD_TIMEOUT_SECONDS")

    replay_csv_path: Path = Field(
        default=Path("data/error_analysis_age5_plus_age10/predictions.csv"),
        validation_alias="EMOTION_REPLAY_CSV_PATH",
    )

    negative_threshold: float = Field(default=0.60, validation_alias="EMOTION_NEGATIVE_THRESHOLD")
    alert_duration_seconds: float = Field(default=10.0, validation_alias="EMOTION_ALERT_DURATION_SECONDS")
    alert_cooldown_seconds: float = Field(default=30.0, validation_alias="EMOTION_ALERT_COOLDOWN_SECONDS")

    # ---- Privacy-first persistence -------------------------------------
    # SQLite file holding emotion/alert events only (no images, no bboxes).
    db_path: Path = Field(
        default=Path("data/emotion_events.db"),
        validation_alias="EMOTION_DB_PATH",
    )
    # Events older than this many days are purged by the background task.
    retention_days: int = Field(default=30, validation_alias="EMOTION_RETENTION_DAYS")
    # How often the purge task runs (seconds).
    purge_interval_seconds: int = Field(
        default=3600, validation_alias="EMOTION_PURGE_INTERVAL_SECONDS"
    )
    # Server-side key for one-way child pseudonymisation. MUST be kept secret.
    # In production, change this value; children's pseudonyms are derived from it.
    privacy_secret: str = Field(
        default="dev-only-change-me",
        validation_alias="EMOTION_PRIVACY_SECRET",
    )

    # ---- Optional access control ----------------------------------------
    # When set to a non-empty value, mutating/report/admin endpoints require
    # "Authorization: Bearer <EMOTION_API_TOKEN>". Leave empty to keep the
    # demo open (default). See app.security.require_api_token.
    api_token: str | None = Field(default=None, validation_alias="EMOTION_API_TOKEN")

    # ---- Anonymous re-identification (cross-video identity) -------------
    # Enables matching video tracks against stored face-feature prototypes so
    # the same anonymous child keeps one pseudonym across uploads. The
    # prototype is a normalised 768-dim descriptor; no images are stored.
    reid_enabled: bool = Field(default=True, validation_alias="EMOTION_REID_ENABLED")
    # Cosine similarity at/above which a track is automatically reused.
    reid_auto_match_threshold: float = Field(default=0.80, validation_alias="EMOTION_REID_AUTO_MATCH_THRESHOLD")
    # Below this similarity a track is treated as a new anonymous identity.
    # Calibrated so different faces (~0.2-0.5 in the offline experiments)
    # never fall into the ambiguous grey band.
    reid_new_identity_threshold: float = Field(default=0.62, validation_alias="EMOTION_REID_NEW_IDENTITY_THRESHOLD")
    # Even above auto-match threshold, if top2 is this close we refuse to
    # guess and create one pending confirmation instead.
    reid_ambiguous_margin: float = Field(default=0.04, validation_alias="EMOTION_REID_AMBIGUOUS_MARGIN")
    # How many quality embeddings are averaged into one track descriptor.
    reid_track_sample_limit: int = Field(default=8, validation_alias="EMOTION_REID_TRACK_SAMPLE_LIMIT")
    # Which backbone provides the identity descriptor:
    # - imagenet: a separate ImageNet-pretrained ViT-B/16 (emotion-agnostic,
    #   more stable across expressions)
    # - emotion:  the CLS feature of the emotion ViT itself (one forward pass,
    #   but sensitive to expression changes)
    reid_backbone: str = Field(default="imagenet", validation_alias="EMOTION_REID_BACKBONE")
    # Gallery cap: oldest identities beyond this are not matched (kept, pruned
    # by retention later).
    reid_gallery_limit: int = Field(default=500, validation_alias="EMOTION_REID_GALLERY_LIMIT")
    # Pending confirmations older than this many hours auto-expire to
    # 'keep separate' (events stay under their temporary anonymous code).
    reid_pending_expiry_hours: float = Field(default=48.0, validation_alias="EMOTION_REID_PENDING_EXPIRY_HOURS")

    # ---- Session-only thumbnail buffer (claim support) -------------------
    # One small face thumbnail per track, kept ONLY in memory so teachers can
    # see who an anonymous code refers to when claiming. Never written to the
    # database or disk; destroyed on claim, on TTL expiry, on purge and on
    # process restart.
    thumb_ttl_seconds: int = Field(default=7200, validation_alias="EMOTION_THUMB_TTL_SECONDS")
    thumb_max_entries: int = Field(default=300, validation_alias="EMOTION_THUMB_MAX_ENTRIES")
    thumb_size: int = Field(default=96, validation_alias="EMOTION_THUMB_SIZE")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
