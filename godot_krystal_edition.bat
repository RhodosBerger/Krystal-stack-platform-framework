@echo off
REM Godot Krystal Edition Launcher (Windows)
REM Invariant: VITAL_MAX_HP == 6
echo ========================================================
echo   LAUNCHING GODOT KRYSTAL EDITION (v1.0.0-SOVEREIGN)
echo ========================================================
set GODOT_EXE=%~dp0godot_project\godot.exe
if not exist "%GODOT_EXE%" (
    echo [INFO] godot.exe not in local directory, checking PATH...
    where godot >nul 2>&1
    if %ERRORLEVEL% EQU 0 ( set GODOT_EXE=godot ) else (
        echo [ERROR] Godot 4.x executable not found. Please install Godot 4.2+ Forward+.
        exit /b 1
    )
)
"%GODOT_EXE%" --path "%~dp0godot_project" res://scenes/GodotKrystalEditionStage.tscn %*
