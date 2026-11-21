from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.state import app_state

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await app_state.ws.connect(websocket)
    try:
        await websocket.send_json({"type": "connected", "message": "emotion stream ready"})
        while True:
            # Keep the socket open and allow clients to send ping/config messages.
            message = await websocket.receive_json()
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        app_state.ws.disconnect(websocket)
    except Exception:
        app_state.ws.disconnect(websocket)
