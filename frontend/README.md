# Emotion Detection Frontend

Vue 3 + Vite dashboard for the child emotion detection system.

## Run

Start the backend first from the repository root:

```powershell
Copy-Item backend/.env.example backend/.env
# Set different random values for EMOTION_PRIVACY_SECRET and EMOTION_API_TOKEN in backend/.env.
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8002
```

Start frontend:

```powershell
Copy-Item frontend/.env.development.example frontend/.env.development
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5174
```

Open:

- Dashboard: http://127.0.0.1:5174
- Backend API: http://127.0.0.1:8002/docs

## Modes

The frontend reads backend data from `VITE_API_BASE`, set in `frontend/.env.development` to `http://127.0.0.1:8002`.

```powershell
$env:VITE_API_BASE="http://127.0.0.1:8002"
npm run dev
```

If the backend is unavailable, the UI falls back to demo data. Mock mode does not require model weights. Real local inference requires separately obtained authorized weights; datasets, videos, databases and weights are not included in this repository.

## Recording Demo Flow

1. Start backend in `mock` mode for the built-in demonstration, or configure a local model/replay data separately.
2. Start frontend and open `http://127.0.0.1:5173`.
3. Click `演示播放` in the monitor panel.
4. The dashboard will generate demo frames, call `/api/infer/image`, receive replayed emotion results, update the warning queue, and refresh statistics.
5. Use `暂停演示` to pause or the step button to advance one frame manually.
