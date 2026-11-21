"""Anonymous identity roster: cross-video pseudonyms, pending confirmations.

These endpoints expose the re-ID gallery to the report UI:

- ``GET  /api/identities``             list anonymous identities (no vectors)
- ``GET  /api/identities/pending``     track-level pending confirmations
- ``POST /api/identities/pending/{id}/resolve``  link | keep | dismiss
- ``POST /api/identities/{anon}/claim``  link an identity to a roster child
- ``POST /api/identities/{anon}/unclaim``  remove the roster link
- ``DELETE /api/identities/{anon}``     erase the prototype (events stay anonymous)

Embedding vectors are never returned by any endpoint.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app import db
from app.security import require_api_token

router = APIRouter(prefix="/api/identities", tags=["identities"])


@router.get("")
def list_identities(limit: int = 100) -> dict[str, Any]:
    rows = db.list_identities(limit=int(limit), include_embedding=False)
    out = []
    for r in rows:
        child_nickname = None
        child_class = None
        if r.get("child_id"):
            child = db.get_child(r["child_id"])
            if child:
                child_nickname = child.get("nickname")
                child_class = child.get("class_name")
        out.append({
            "anon_code": r["anon_code"],
            "child_id": r.get("child_id"),
            "child_nickname": child_nickname,
            "child_class": child_class,
            "sample_count": int(r.get("sample_count") or 1),
            "event_count": db.count_identity_events(r["anon_code"]),
            "created_at": r.get("created_at"),
            "last_seen_at": r.get("last_seen_at"),
        })
    return {"total": len(out), "identities": out}


@router.get("/pending")
def list_pending(limit: int = 100) -> dict[str, Any]:
    rows = db.list_pending_confirmations(status="pending", limit=int(limit))
    return {"total": len(rows), "pending": rows}


@router.post("/pending/{pending_id}/resolve", dependencies=[Depends(require_api_token)])
def resolve_pending(pending_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve one track-level confirmation.

    ``action``: ``link`` (merge into the candidate identity), ``keep`` (keep
    the temporary identity as its own anonymous child), ``dismiss`` (keep the
    data anonymous, close the question).
    """
    pending = db.get_pending_confirmation(pending_id)
    if pending is None:
        raise HTTPException(status_code=404, detail="Pending confirmation not found")
    if pending.get("status") != "pending":
        raise HTTPException(status_code=409, detail="Already resolved")

    action = (payload.get("action") or "").strip()
    if action not in {"link", "keep", "dismiss"}:
        raise HTTPException(status_code=400, detail="action must be link | keep | dismiss")

    linked_events = 0
    if action == "link":
        candidate = pending.get("candidate_anon_code")
        if not candidate:
            raise HTTPException(status_code=400, detail="No candidate to link to")
        # Move this upload's rows under the candidate pseudonym.
        linked_events = db.merge_events_anon(
            pending["anon_code"],
            candidate,
            event_request_id=pending.get("event_request_id"),
            track_label=pending.get("track_label"),
        )
        # Drop the temporary identity if it has no remaining events.
        if db.count_identity_events(pending["anon_code"]) == 0:
            db.delete_identity(pending["anon_code"])
        db.resolve_pending_confirmation(pending_id, "resolved")
    elif action == "keep":
        # Temporary identity becomes a permanent anonymous identity.
        db.resolve_pending_confirmation(pending_id, "resolved")
    else:
        db.resolve_pending_confirmation(pending_id, "dismissed")

    return {
        "status": "ok",
        "pending_id": pending_id,
        "action": action,
        "linked_events": linked_events,
        "note": "no biometric data involved; events stay anonymous unless claimed",
    }


@router.post("/{anon_code}/claim", dependencies=[Depends(require_api_token)])
def claim_identity(anon_code: str, payload: dict[str, Any]) -> dict[str, Any]:
    child_id = (payload.get("child_id") or "").strip()
    if not child_id:
        raise HTTPException(status_code=400, detail="child_id is required")
    if db.get_child(child_id) is None:
        raise HTTPException(status_code=404, detail="Child not found")
    if db.get_identity(anon_code) is None:
        raise HTTPException(status_code=404, detail="Identity not found")
    linked = db.link_identity_to_child(anon_code, child_id)
    return {
        "status": "claimed",
        "anon_code": anon_code,
        "child_id": child_id,
        "linked_events": linked,
        "note": "future sightings of this anonymous identity link to the child automatically",
    }


@router.post("/{anon_code}/unclaim", dependencies=[Depends(require_api_token)])
def unclaim_identity(anon_code: str) -> dict[str, Any]:
    if db.get_identity(anon_code) is None:
        raise HTTPException(status_code=404, detail="Identity not found")
    unlinked = db.unlink_identity(anon_code)
    return {"status": "unclaimed", "anon_code": anon_code, "unlinked_events": unlinked}


@router.delete("/{anon_code}", dependencies=[Depends(require_api_token)])
def delete_identity(anon_code: str) -> dict[str, Any]:
    """Erase the identity prototype. Its emotion events remain, fully anonymous."""
    removed = db.delete_identity(anon_code)
    if not removed:
        raise HTTPException(status_code=404, detail="Identity not found")
    return {
        "status": "deleted",
        "anon_code": anon_code,
        "note": "prototype erased; events remain anonymous and are still subject to retention purge",
    }
