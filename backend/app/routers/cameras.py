from fastapi import APIRouter, HTTPException
from app.schemas import CameraConfig, CameraCreate
from app.state import app_state

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


@router.get("", response_model=list[CameraConfig])
def list_cameras() -> list[CameraConfig]:
    return list(app_state.cameras.values())


@router.post("", response_model=CameraConfig)
def create_camera(payload: CameraCreate) -> CameraConfig:
    camera = CameraConfig(name=payload.name, source=payload.source, location=payload.location)
    app_state.cameras[camera.id] = camera
    return camera


@router.get("/{camera_id}", response_model=CameraConfig)
def get_camera(camera_id: str) -> CameraConfig:
    camera = app_state.cameras.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.delete("/{camera_id}")
def delete_camera(camera_id: str) -> dict[str, str]:
    if camera_id not in app_state.cameras:
        raise HTTPException(status_code=404, detail="Camera not found")
    del app_state.cameras[camera_id]
    return {"status": "deleted", "camera_id": camera_id}
