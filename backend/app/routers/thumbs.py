"""Session-only claim thumbnails: serve, burn, purge.

Thumbnails live only in RAM (see app.thumb_buffer); these endpoints expose
them for the claim UI and let teachers incinerate them:

- GET    /api/thumbs/{request_id}/{track_label}  -> image/jpeg (or 404)
- DELETE /api/thumbs/{request_id}                -> burn all thumbs of a batch
- POST   /api/thumbs/purge                       -> burn everything
- GET    /api/thumbs/stats                       -> buffer status (no images)

Reads are intentionally unauthenticated (same level as report reads, and the
buffer is ephemeral); destructive endpoints require the admin token when
EMOTION_API_TOKEN is configured.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response

from app.security import require_api_token
from app.thumb_buffer import thumb_buffer

router = APIRouter(prefix="/api/thumbs", tags=["thumbs"])


@router.get("/{request_id}/{track_label}")
def get_thumb(request_id: str, track_label: str) -> Response:
    jpeg = thumb_buffer.get(request_id, track_label)
    if jpeg is None:
        raise HTTPException(status_code=404, detail="Thumbnail not found (expired, burned or never stored)")
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "X-Session-Only": "true",
        },
    )


@router.delete("/{request_id}", dependencies=[Depends(require_api_token)])
def burn_request_thumbs(request_id: str) -> dict:
    removed = thumb_buffer.delete_request(request_id)
    return {"status": "burned", "request_id": request_id, "removed": removed}


@router.delete("/{request_id}/{track_label}", dependencies=[Depends(require_api_token)])
def burn_track_thumb(request_id: str, track_label: str) -> dict:
    removed = thumb_buffer.delete_track(request_id, track_label)
    return {"status": "burned", "request_id": request_id, "track_label": track_label, "removed": removed}


@router.post("/purge", dependencies=[Depends(require_api_token)])
def purge_thumbs() -> dict:
    removed = thumb_buffer.purge_all()
    return {"status": "burned", "removed": removed}


@router.get("/stats")
def thumb_stats() -> dict:
    return thumb_buffer.stats()
