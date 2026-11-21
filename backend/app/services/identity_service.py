"""Anonymous re-identification service (cross-video identity).

How it works
------------
Every confirmed video track contributes several face descriptors (768-dim
normalised CLS embeddings from the local ViT). They are averaged into one
track descriptor, which is compared against stored *identity prototypes*:

- ``sim >= auto_match_threshold``      -> reuse that anonymous code
- ``sim < new_identity_threshold``     -> new anonymous identity
- grey band in between                 -> reuse when the top-2 gap is clear,
                                          otherwise one *pending* record

Pending records are rare, track-level, and auto-expire (teacher never has to
confirm frame-by-frame). If re-ID is disabled or the model is unavailable the
caller falls back to the request-scoped HMAC pseudonym.

Privacy: prototypes are irreversible normalised descriptors. No images, no
video bytes, no file names and no bounding boxes are stored.
"""

from __future__ import annotations

import secrets
import uuid
from typing import Any

import numpy as np

from app import db
from app.config import Settings

_EMB_DIM = 768
_F32 = np.float32


def _pack(emb: np.ndarray) -> bytes:
    emb = np.asarray(emb, dtype=_F32).reshape(-1)
    if emb.shape[0] != _EMB_DIM:
        raise ValueError(f"embedding must be {_EMB_DIM}-dim, got {emb.shape[0]}")
    norm = float(np.linalg.norm(emb))
    if norm > 0:
        emb = emb / norm
    return emb.tobytes()


def _unpack(blob: bytes) -> np.ndarray | None:
    try:
        arr = np.frombuffer(blob, dtype=_F32).reshape(-1)
    except Exception:
        return None
    if arr.shape[0] != _EMB_DIM:
        return None
    norm = float(np.linalg.norm(arr))
    return (arr / norm) if norm > 0 else None


def aggregate_embeddings(embeddings: list[np.ndarray]) -> np.ndarray | None:
    """Mean-pool a track's embeddings into one normalised descriptor."""
    if not embeddings:
        return None
    valid = []
    for e in embeddings:
        e = np.asarray(e, dtype=_F32).reshape(-1)
        if e.shape[0] != _EMB_DIM:
            continue
        n = float(np.linalg.norm(e))
        if n <= 0 or not np.isfinite(e).all():
            continue
        valid.append(e / n)
    if not valid:
        return None
    mean = np.mean(np.stack(valid), axis=0)
    n = float(np.linalg.norm(mean))
    if n <= 0:
        return None
    return (mean / n).astype(_F32)


class IdentityService:
    """Matching + registration of anonymous identities, backed by SQLite."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # -- helpers --------------------------------------------------------- #

    def enabled(self) -> bool:
        return bool(self.settings.reid_enabled)

    def new_anon_code(self) -> str:
        """Fresh random pseudonym. Not derived from camera/track -> cannot be
        linked back to a short-lived track label even with the server secret."""
        return f"c-{secrets.token_hex(7)}"

    def _gallery(self) -> tuple[list[dict[str, Any]], np.ndarray, list[np.ndarray]]:
        rows = db.list_identities(limit=int(self.settings.reid_gallery_limit), include_embedding=True)
        metas: list[dict[str, Any]] = []
        vectors: list[np.ndarray] = []
        for r in rows:
            emb = _unpack(bytes(r.get("embedding") or b""))
            if emb is None:
                continue
            metas.append(r)
            vectors.append(emb)
        if not vectors:
            return metas, np.zeros((0, _EMB_DIM), dtype=_F32), []
        return metas, np.stack(vectors).astype(_F32), vectors

    def match(self, emb: np.ndarray) -> dict[str, Any]:
        """Return top-2 gallery matches for a normalised descriptor."""
        metas, matrix, _ = self._gallery()
        if matrix.shape[0] == 0:
            return {"top1_anon": None, "top1_sim": -1.0, "top2_sim": -1.0, "margin": 1.0, "gallery_size": 0}
        scores = matrix @ emb
        order = np.argsort(-scores)
        top1_idx = int(order[0])
        top2_sim = float(scores[order[1]]) if matrix.shape[0] > 1 else -1.0
        top1 = metas[top1_idx]
        top1_sim = float(scores[top1_idx])
        return {
            "top1_anon": top1["anon_code"],
            "top1_child_id": top1.get("child_id"),
            "top1_child_nickname": None,  # joined later by callers if needed
            "top1_sim": round(top1_sim, 6),
            "top2_sim": round(top2_sim, 6),
            "margin": round(top1_sim - top2_sim, 6),
            "gallery_size": matrix.shape[0],
        }

    def decide(self, emb: np.ndarray) -> dict[str, Any]:
        """Decide match / new / pending for one track descriptor.

        Calibrated against the offline experiments (docs/vit_embedding_spike_*,
        scripts/reid_*_compare.py):
        - within-track descriptors sit at ~0.99
        - different faces sit at ~0.2-0.5 for both backbones
        The grey band [new, auto) always becomes *pending*: we prefer
        fragmenting an identity over mis-merging two children.
        """
        m = self.match(emb)
        auto = float(self.settings.reid_auto_match_threshold)
        low = float(self.settings.reid_new_identity_threshold)
        ambiguous_margin = float(self.settings.reid_ambiguous_margin)
        sim = m["top1_sim"]
        margin = m["margin"]
        decision: dict[str, Any] = {**m, "decision": "new"}
        if m["top1_anon"] is None or sim < low:
            decision["decision"] = "new"
        elif sim >= auto:
            if margin < ambiguous_margin and m["top2_sim"] >= low:
                decision["decision"] = "pending"
            else:
                decision["decision"] = "match"
        else:  # grey band: ambiguous by construction -> teacher sees one row
            decision["decision"] = "pending"
        return decision

    def register_new(self, emb: np.ndarray, camera_id: str = "camera-1") -> dict[str, Any]:
        anon_code = self.new_anon_code()
        db.insert_identity(
            identity_id=str(uuid.uuid4()),
            anon_code=anon_code,
            camera_id=camera_id,
            embedding=_pack(emb),
        )
        return {"anon_code": anon_code, "sample_count": 1}

    def touch(self, anon_code: str, emb: np.ndarray) -> None:
        """Fold a new descriptor into the prototype (90/10 running mean)."""
        current = db.get_identity(anon_code)
        if current is None:
            return
        old = _unpack(bytes(current.get("embedding") or b""))
        new = emb / (float(np.linalg.norm(emb)) or 1.0)
        if old is None:
            merged = new
        else:
            merged = 0.9 * old + 0.1 * new
            n = float(np.linalg.norm(merged))
            merged = merged / n if n > 0 else new
        count = int(current.get("sample_count") or 1) + 1
        db.update_identity_last_seen(
            anon_code, _pack(merged), sample_count=count
        )

    def child_display(self, anon_code: str) -> tuple[str | None, str | None]:
        """(child_id, nickname) for an identity, if linked to a roster child."""
        identity = db.get_identity(anon_code)
        if not identity or not identity.get("child_id"):
            return None, None
        child = db.get_child(identity["child_id"])
        return identity["child_id"], (child or {}).get("nickname")
