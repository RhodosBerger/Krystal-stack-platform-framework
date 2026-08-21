@echo off
title KrystalVino Architecture Setup
color 0A

echo ========================================================
echo    KRYSTALVINO / GAMESA ADVANCED SETUP UTILITY
echo ========================================================
echo.
echo [1/3] Checking environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)
echo [OK] Python detected.

echo.
echo [2/3] Installing dependencies...
echo (In a real scenario, this would install numpy, openvino, etc.)
echo Skipping pip install for this demonstration environment.

echo.
echo [3/3] Launching Setup Wizard & Benchmarks...
echo.
python krystalvino_setup_wizard.py

echo.
echo ========================================================
echo    SETUP COMPLETE. SYSTEM READY.
echo ========================================================
pause
