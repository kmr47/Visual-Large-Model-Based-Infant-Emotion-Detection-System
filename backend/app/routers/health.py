from fastapi import APIRouter, Depends
from app.config import get_settings
from app.schemas import HealthResponse
from app.security import require_api_token

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        app=settings.app_name,
        version=settings.app_version,
        inference_mode=settings.inference_mode,
    )


@router.get("/health/admin", dependencies=[Depends(require_api_token)])
def admin_token_check() -> dict:
    """Lightweight admin-token probe used by the UI to show password validity.

    Reaching this endpoint means ``Authorization: Bearer <EMOTION_API_TOKEN>``
    was accepted (or access control is disabled). Returns nothing sensitive.
    """
    settings = get_settings()
    return {
        "authorized": True,
        "access_control_enabled": bool(settings.api_token),
    }
