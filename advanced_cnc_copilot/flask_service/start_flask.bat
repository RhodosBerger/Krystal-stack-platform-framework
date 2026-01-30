@echo off
REM Quick Start Script for Flask Service

echo ============================================
echo FANUC RISE - FLASK MICROSERVICE STARTUP
echo ============================================
echo.

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call ..\venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)

REM Set environment variable
echo [2/3] Setting PYTHONPATH...
set PYTHONPATH=%CD%\..

REM Start Flask server
echo [3/3] Starting Flask microservice...
echo.
echo ============================================
echo Server will start on: http://localhost:5000
echo WebSocket: ws://localhost:5000
echo Health Check: http://localhost:5000/health
echo ============================================
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py
