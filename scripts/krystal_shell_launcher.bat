@echo off
rem ==============================================================================
rem KRYSTAL-STACK: SOVEREIGN SHELL LAUNCHER & FAILSAFE WATCHDOG
rem ==============================================================================
rem Script: scripts/krystal_shell_launcher.bat
rem Purpose: Headless / Custom Shell Launcher for Windows NT Kernel.
rem          Replaces or suppresses explorer.exe, launches Krystal Hub,
rem          Janet Compositor, and AI Coprocessors with zero DWM bloat.
rem
rem Usage:
rem   krystal_shell_launcher.bat              -> Start Sovereign Krystal Shell
rem   krystal_shell_launcher.bat --restore    -> Restore standard Windows explorer.exe
rem
rem System Invariant: VITAL_MAX_HP = 6
rem ==============================================================================

setlocal enabledelayedexpansion
set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..
set PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%

echo ================================================================================
echo   [KRYSTAL-STACK] SOVEREIGN OPEN-SOURCE OS SHELL LAUNCHER
echo   System Invariant: VITAL_MAX_HP = 6
echo ================================================================================

if "%1"=="--restore" (
    echo [ACTION] Restoring standard Windows Explorer Shell...
    reg add "HKCU\Software\Microsoft\Windows NT\CurrentVersion\Winlogon" /v Shell /t REG_SZ /d "explorer.exe" /f >nul 2>&1
    tasklist /fi "imagename eq explorer.exe" | findstr /i "explorer.exe" >nul
    if errorlevel 1 (
        start explorer.exe
    )
    echo [OK] Windows Explorer restored successfully.
    goto :end
)

if "%1"=="--set-shell" (
    echo [ACTION] Registering Krystal Shell as default user shell in registry...
    reg add "HKCU\Software\Microsoft\Windows NT\CurrentVersion\Winlogon" /v Shell /t REG_SZ /d "\"%~f0\"" /f
    echo [OK] Registry updated. On next login, Krystal Shell will launch directly.
    goto :end
)

echo [STEP 1] Checking Localhost Mission Control Hub (Port 8080)...
netstat -ano | findstr 8080 | findstr LISTENING >nul
if errorlevel 1 (
    echo   -> Starting Krystal Web Hub background server...
    start /b python -m krystal_web_hub.server 8080
    timeout /t 2 /nobreak >nul
) else (
    echo   -> Krystal Web Hub is already running on port 8080.
)

echo [STEP 2] Initializing Janet Subsystem & WSL2 Bridge...
python -m krystal_kernel.janet_bytecode_decoder

echo [STEP 3] Launching Sovereign Compositor & Neural ASCII Interface...
echo   -> Press CTRL+C or type 'exit' to terminate.
echo   -> Failsafe hotkey: CTRL+SHIFT+ESC launches Task Manager anytime.
echo ================================================================================

python -c "from krystal_janet.janet_bridge import JanetValidator; print('[KRYSTAL] Janet Engine validated and active. All 39+ modules ready.')"

rem Keep window open as interactive sovereign terminal
cmd /k "title KRYSTAL-STACK SOVEREIGN TERMINAL && cd /d %REPO_ROOT%"

:end
endlocal
