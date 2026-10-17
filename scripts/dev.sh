#!/usr/bin/env bash
# Start the demo locally: API (real-time quotes) on :8000 and the web app on :5173.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
.venv/bin/uvicorn app.main:app --app-dir services/api --host 127.0.0.1 --port 8000 > logs/api.log 2>&1 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT
echo "API  -> http://127.0.0.1:8000/docs   (logs/api.log)"
echo "Web  -> http://127.0.0.1:5173"
cd web && npx vite --port 5173 --host 127.0.0.1
