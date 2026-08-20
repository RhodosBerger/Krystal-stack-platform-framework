@echo off
REM Quick Start Script for Fanuc Rise

echo ============================================
echo FANUC RISE - DJANGO SERVER STARTUP
echo ============================================
echo.

REM Activate virtual environment
echo [1/4] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)

REM Set environment variable
echo [2/4] Setting PYTHONPATH...
set PYTHONPATH=%CD%

REM Check if migrations exist
echo [3/4] Checking database...
python manage.py showmigrations > nul 2>&1
if errorlevel 1 (
    echo Creating database migrations...
    python manage.py makemigrations
    python manage.py migrate
)

REM Start Django server
echo [4/4] Starting Django server...
echo.
echo ============================================
echo Server will start on: http://localhost:8000
echo Django Admin: http://localhost:8000/admin
echo REST API: http://localhost:8000/api/
echo ============================================
echo.
echo Press Ctrl+C to stop the server
echo.

python manage.py runserver 0.0.0.0:8000
