$ErrorActionPreference = "Stop"
$backendRoot = $PSScriptRoot
$projectRoot = Split-Path $backendRoot -Parent
Set-Location $projectRoot
if (-not (Test-Path (Join-Path $backendRoot ".env"))) {
  throw "Missing backend/.env. Copy backend/.env.example to backend/.env and configure strong secrets first."
}
if (-not $env:EMOTION_INFERENCE_MODE) {
  $env:EMOTION_INFERENCE_MODE = "mock"
}
python -m uvicorn app.main:app --app-dir $backendRoot --host 127.0.0.1 --port 8002 --reload
