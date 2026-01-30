@echo off
TITLE FANUC RISE // SYSTEM REPAIR & LAUNCH
COLOR 0E
CLS

ECHO ========================================================
ECHO    FANUC RISE // SYSTEM REPAIR PROTOCOL
ECHO    DETECTING CORRUPTED DEPENDENCIES...
ECHO ========================================================
ECHO.

cd frontend-react

IF EXIST "node_modules" (
    ECHO [1/4] REMOVING CORRUPTED NODE_MODULES...
    rmdir /s /q node_modules
)

ECHO [2/4] INSTALLING FRESH DEPENDENCIES (This may take a minute)...
call npm install
IF %ERRORLEVEL% NEQ 0 (
    COLOR 0C
    ECHO [ERROR] NPM INSTALL FAILED. PLEASE INSTALL NODE.JS.
    PAUSE
    EXIT /B
)

cd ..

ECHO.
ECHO [3/4] ACTIVATING NEURAL CORE (BACKEND)...
start "RISE_BACKEND" cmd /k "venv\Scripts\activate && uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

ECHO [4/4] STARTING VISUAL INTERFACE (FRONTEND)...
cd frontend-react
start "RISE_FRONTEND" cmd /k "npm run dev"
cd ..

ECHO.
ECHO ========================================================
ECHO    REPAIR COMPLETE.
ECHO    LAUNCHING BROWSER IN 5 SECONDS...
ECHO ========================================================
timeout /t 5 >nul
start http://localhost:3000/?role=ADMIN

PAUSE
