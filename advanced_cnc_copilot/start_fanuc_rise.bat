@echo off
TITLE FANUC RISE // SYSTEM LAUNCHER
COLOR 0A
CLS

ECHO ========================================================
ECHO    FANUC RISE // NEURAL MANUFACTURING CO-PILOT
ECHO    INITIATING UPLINK PROTOCOL...
ECHO ========================================================
ECHO.

:: 1. START BACKEND
ECHO [1/3] ACTIVATING NEURAL CORE (BACKEND)...
start "RISE_BACKEND" cmd /k "venv\Scripts\activate && uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

:: 2. START FRONTEND
ECHO [2/3] LOADING VISUAL INTERFACE (FRONTEND)...
cd frontend-react
start "RISE_FRONTEND" cmd /k "npm run dev"
cd ..

:: 3. LAUNCH BROWSER
ECHO [3/3] ESTABLISHING SECURE LINK...
timeout /t 5 >nul
start http://localhost:3000/?role=ADMIN

ECHO.
ECHO ========================================================
ECHO    SYSTEM ONLINE.
ECHO    ACCESSING: http://localhost:3000
ECHO    ROLE:      ADMIN (Root Access)
ECHO ========================================================
ECHO.
PAUSE
