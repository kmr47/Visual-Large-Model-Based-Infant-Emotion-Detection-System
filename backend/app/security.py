"""Lightweight API access control.

Design: optional bearer-token gate.
- If ``EMOTION_API_TOKEN`` is empty (default), the demo stays fully open and
  every endpoint works without credentials (matches the original behaviour).
- If a token is configured, write/admin/report endpoints enforce
  ``Authorization: Bearer <token>``; read-only monitoring endpoints
  (health, inference upload for the demo UI) stay open so the live dashboard
  keeps working while sensitive data operations are protected.

This is intentionally simple (constant-time compare, no user store) so the
thesis can present it as a *configurable, deploy-time* access-control layer
without pretending to be a full IAM system.
"""

from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


def require_api_token(
    authorization: Optional[str] = Header(default=None),
    x_api_token: Optional[str] = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """FastAPI dependency: reject requests when a token is configured but absent/wrong."""
    expected = settings.api_token
    if not expected:
        return  # access control disabled -> open demo

    provided = None
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    if provided is None and x_api_token:
        provided = x_api_token.strip()

    if not provided or not hmac.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
