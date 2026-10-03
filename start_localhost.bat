@echo off
title Krystal-Stack Localhost Mission Control
cls
echo =======================================================================
echo          KRYSTAL-STACK HETEROGENEOUS NEURAL ASCII PLATFORM
echo =======================================================================
echo.
echo [1/2] Checking Python environment...
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python was not found in PATH. Please install Python 3.10+
    pause
    exit /b 1
)

echo [2/2] Starting Localhost Mission Control on http://localhost:8080 ...
echo.
python start_localhost.py
pause
