@echo off
rem ============================================================================
rem Krystal-Stack Janet Subproject Launcher (Batch)
rem ============================================================================

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..
set PYTHONPATH=%REPO_ROOT%;%PYTHONPATH%
cd /d "%SCRIPT_DIR%"

where janet >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [KRYSTAL-JANET] Native Janet runtime detected in PATH.
    janet main.janet %*
) else (
    echo [KRYSTAL-JANET] Native Janet binary not found in PATH.
    echo [KRYSTAL-JANET] Launching via Krystal Janet Bridge Engine...
    python -m krystal_janet.janet_bridge %*
)
