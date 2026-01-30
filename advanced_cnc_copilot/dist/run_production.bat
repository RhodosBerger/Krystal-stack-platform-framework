@echo off
TITLE FANUC RISE // PRODUCTION
echo STARTING PRODUCTION SERVER...
start "RISE_CORE" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo SERVING FRONTEND...
cd frontend_build
python -m http.server 3000
