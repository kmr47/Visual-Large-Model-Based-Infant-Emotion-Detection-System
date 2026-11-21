# Phase 3 Backend

FastAPI backend for the child emotion detection system.

## Run

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open:

- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- WebSocket: ws://127.0.0.1:8000/ws/stream

## Inference Modes

### mock

Default. No model required. Useful for local UI/backend demos.

```powershell
$env:EMOTION_INFERENCE_MODE="mock"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### replay

Replays predictions from exported CSV files, useful for recorded demo flows.

```powershell
$env:EMOTION_INFERENCE_MODE="replay"
$env:EMOTION_REPLAY_CSV_PATH="C:\path\to\your\authorized\predictions.csv"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### cloud

Calls a remote AutoDL inference API if one is later exposed.

```powershell
$env:EMOTION_INFERENCE_MODE="cloud"
$env:EMOTION_CLOUD_INFERENCE_URL="http://host:port/api/infer/image"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Main Endpoints

- `GET /health`
- `POST /api/infer/image`
- `GET /api/alerts`
- `PATCH /api/alerts/{alert_id}`
- `GET /api/alerts/stats/summary`
- `GET /api/cameras`
- `POST /api/cameras`
- `WS /ws/stream`
