from fastapi import APIRouter, HTTPException
from app.schemas import AlertRecord, AlertUpdate, StatsResponse
from app.state import app_state

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertRecord])
def list_alerts(status: str | None = None) -> list[AlertRecord]:
    alerts = list(app_state.alerts.values())
    if status:
        alerts = [a for a in alerts if a.status.value == status]
    return sorted(alerts, key=lambda a: a.created_at, reverse=True)


@router.get("/stats/summary", response_model=StatsResponse)
def stats_summary() -> StatsResponse:
    alerts = list(app_state.alerts.values())
    active = [a for a in alerts if a.status.value == "active"]
    recent = sorted(alerts, key=lambda a: a.created_at, reverse=True)[:10]
    return StatsResponse(
        total_alerts=len(alerts),
        active_alerts=len(active),
        emotion_counts=dict(app_state.emotion_counts),
        recent_alerts=recent,
    )


@router.get("/{alert_id}", response_model=AlertRecord)
def get_alert(alert_id: str) -> AlertRecord:
    alert = app_state.alerts.get(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert


@router.patch("/{alert_id}", response_model=AlertRecord)
async def update_alert(alert_id: str, payload: AlertUpdate) -> AlertRecord:
    alert = await app_state.update_alert(alert_id, payload.status)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert
