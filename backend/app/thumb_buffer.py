"""Session-only face-thumbnail buffer for the claim workflow.

Why this exists
---------------
Anonymous re-ID gives teachers a stable pseudonym (c-...), but a pseudonym
alone tells them nothing about *which child* it refers to. To claim a track
the teacher must be able to look at the face. This buffer keeps ONE small
thumbnail per track **in process memory only**:

- never written to SQLite or to disk
- destroyed when the batch is claimed / manually cleared
- auto-expired after ``thumb_ttl_seconds``
- capped at ``thumb_max_entries`` (LRU eviction)
- gone on process restart

Thumbnails are downscaled (default 96px) before being stored. The database
therefore still contains zero images, which keeps the privacy story intact:
images exist only transiently, in RAM, to enable a human decision, and are
incinerated as soon as that decision is made ("claim-to-burn").
"""

from __future__ import annotations

import threading
import time
from io import BytesIO
from typing import Any

from PIL import Image

from app.config import get_settings

_settings = get_settings()


def _downscale(jpeg_bytes: bytes, size: int = 96) -> bytes:
    try:
        img = Image.open(BytesIO(jpeg_bytes)).convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return buf.getvalue()
    except Exception:
        return jpeg_bytes[:64 * 1024]


class ThumbBuffer:
    """In-memory, TTL-limited, LRU-capped thumbnail store."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.max_entries = int(_settings.thumb_max_entries)
        self.ttl_seconds = int(_settings.thumb_ttl_seconds)
        self.thumb_size = int(_settings.thumb_size)

    @staticmethod
    def _key(request_id: str, track_label: str) -> str:
        return f"{request_id}|{track_label}"

    def _expire_locked(self) -> None:
        now = time.time()
        stale = [k for k, v in self._items.items() if now - v["ts"] > self.ttl_seconds]
        for k in stale:
            self._items.pop(k, None)

    def _evict_locked(self) -> None:
        while len(self._items) > self.max_entries:
            oldest = min(self._items, key=lambda k: self._items[k]["ts"])
            self._items.pop(oldest, None)

    def put(
        self,
        request_id: str,
        track_label: str,
        anon_code: str,
        jpeg_bytes: bytes,
    ) -> None:
        if not jpeg_bytes:
            return
        key = self._key(request_id, track_label)
        with self._lock:
            self._expire_locked()
            self._items[key] = {
                "request_id": request_id,
                "track_label": track_label,
                "anon_code": anon_code,
                "jpeg": _downscale(jpeg_bytes, self.thumb_size),
                "ts": time.time(),
            }
            self._evict_locked()

    def get(self, request_id: str, track_label: str) -> bytes | None:
        key = self._key(request_id, track_label)
        with self._lock:
            self._expire_locked()
            item = self._items.get(key)
            if item is None:
                return None
            item["ts"] = time.time()  # refresh on read
            return bytes(item["jpeg"])

    def has(self, request_id: str, track_label: str) -> bool:
        return self.get(request_id, track_label) is not None

    def list_request(self, request_id: str) -> list[dict[str, Any]]:
        with self._lock:
            self._expire_locked()
            out = []
            for item in self._items.values():
                if item["request_id"] == request_id:
                    out.append({
                        "track_label": item["track_label"],
                        "anon_code": item["anon_code"],
                        "created_at": None,
                        "expires_in_seconds": max(0, int(self.ttl_seconds - (time.time() - item["ts"]))),
                    })
            return out

    def delete_request(self, request_id: str) -> int:
        with self._lock:
            keys = [k for k, v in self._items.items() if v["request_id"] == request_id]
            for k in keys:
                self._items.pop(k, None)
            return len(keys)

    def delete_track(self, request_id: str, track_label: str) -> int:
        with self._lock:
            key = self._key(request_id, track_label)
            if key in self._items:
                self._items.pop(key, None)
                return 1
            return 0

    def purge_all(self) -> int:
        with self._lock:
            n = len(self._items)
            self._items.clear()
            return n

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._expire_locked()
            return {
                "entries": len(self._items),
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "thumb_size": self.thumb_size,
                "persisted": False,
                "note": "session-only RAM buffer; never written to database or disk",
            }


thumb_buffer = ThumbBuffer()
