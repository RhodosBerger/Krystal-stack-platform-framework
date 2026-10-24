@echo off
REM sovereign_run_tester.bat
REM One-Click Automated Run-Testing for Sovereign Layers (Windows)
REM Invariant: VITAL_MAX_HP == 6
echo ========================================================
echo   KRYSTAL STACK // SOVEREIGN RUN-TESTER (WINDOWS)
echo ========================================================
python "%~dp0run_sovereign_tests.py" %*
exit /b %ERRORLEVEL%
