from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from app.dependencies import get_alert_engine, get_emotion_service
from app.schemas import InferenceResponse, VideoInferenceResponse
from app.services.alert_engine import AlertEngine
from app.services.emotion_service import EmotionService
from app.state import app_state

router = APIRouter(prefix="/api/infer", tags=["inference"])


@router.post("/image", response_model=InferenceResponse)
async def infer_image(
    file: UploadFile = File(...),
    camera_id: str = "camera-1",
    emotion_service: EmotionService = Depends(get_emotion_service),
    alert_engine: AlertEngine = Depends(get_alert_engine),
) -> InferenceResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        result = await emotion_service.predict_image(image_bytes, filename=file.filename)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Inference failed: {exc}") from exc

    # re-ID decisions: track_id -> final pseudonym + roster link
    decision_by_track = {d.track_id: d for d in result.track_identity_decisions}

    alerts = alert_engine.evaluate(camera_id=camera_id, detections=result.detections)
    alert_tracks = {a.track_id for a in alerts}
    for alert in alerts:
        dec = decision_by_track.get(alert.track_id)
        if dec:
            alert.anonymous_id = dec.anon_code

    for det in result.detections:
        dec = decision_by_track.get(det.track_id)
        await app_state.record_detection(det.primary_emotion.value)
        # Persist one anonymous emotion event per detected face (no image, no bbox).
        app_state.persist_detection(
            camera_id=camera_id,
            track_id=det.track_id,
            emotion=det.primary_emotion.value,
            confidence=det.primary_confidence,
            negative_score=det.negative_score,
            source="image",
            alert_triggered=det.track_id in alert_tracks,
            event_request_id=result.request_id,
            track_label=det.track_id,
            anon_code=dec.anon_code if dec else det.anonymous_id,
            child_id=dec.child_id if dec else None,
        )
    for alert in alerts:
        await app_state.add_alert(alert)

    result.alerts = alerts
    result.alert_triggered = bool(alerts)
    await app_state.ws.broadcast({"type": "inference", "result": result.model_dump(mode="json")})
    return result


@router.post("/video", response_model=VideoInferenceResponse)
async def infer_video(
    file: UploadFile = File(...),
    emotion_service: EmotionService = Depends(get_emotion_service),
    alert_engine: AlertEngine = Depends(get_alert_engine),
) -> VideoInferenceResponse:
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video uploads are supported")

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty video file")

    try:
        result = await emotion_service.predict_video(video_bytes, filename=file.filename)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Video inference failed: {exc}") from exc

    # One re-ID decision per physical face track (decided during the run).
    decision_by_track = {d.track_id: d for d in result.track_identity_decisions}

    # Sampling cadence for anonymous persistence: one row per person per ~2 s.
    # Video frames are dense (typically 5 fps) and writing every frame would flood
    # the daily report with near-duplicate rows from a single clip.
    SAMPLE_INTERVAL_SEC = 2.0
    last_persist: dict[tuple[str, str], float] = {}

    # Fire alerts per frame
    for fr in result.frames:
        alerts = alert_engine.evaluate(camera_id="camera-1", detections=fr.detections)
        alert_tracks = {a.track_id for a in alerts}
        for alert in alerts:
            dec = decision_by_track.get(alert.track_id)
            if dec:
                alert.anonymous_id = dec.anon_code
        ts = float(fr.timestamp_sec or 0.0)
        for det in fr.detections:
            dec = decision_by_track.get(det.track_id)
            anon_code = dec.anon_code if dec else det.anonymous_id
            child_id = dec.child_id if dec else None
            await app_state.record_detection(det.primary_emotion.value)
            key = ("camera-1", det.track_id)
            if key not in last_persist or ts - last_persist[key] >= SAMPLE_INTERVAL_SEC:
                last_persist[key] = ts
                app_state.persist_detection(
                    camera_id="camera-1",
                    track_id=det.track_id,
                    emotion=det.primary_emotion.value,
                    confidence=det.primary_confidence,
                    negative_score=det.negative_score,
                    source="video",
                    alert_triggered=det.track_id in alert_tracks,
                    event_request_id=result.request_id,
                    track_label=det.track_id,
                    anon_code=anon_code,
                    child_id=child_id,
                )
        for alert in alerts:
            await app_state.add_alert(alert)

    await app_state.ws.broadcast({"type": "video_inference", "result": result.model_dump(mode="json")})
    return result
