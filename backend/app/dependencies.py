from functools import lru_cache
from app.config import get_settings
from app.services.alert_engine import AlertEngine
from app.services.emotion_service import EmotionService


@lru_cache
def get_emotion_service() -> EmotionService:
    return EmotionService(get_settings())


@lru_cache
def get_alert_engine() -> AlertEngine:
    return AlertEngine(get_settings())
