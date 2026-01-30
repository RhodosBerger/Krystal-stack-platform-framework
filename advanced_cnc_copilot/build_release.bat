@echo off
TITLE FANUC RISE // BUILD & RELEASE
COLOR 0B
CLS

ECHO ========================================================
ECHO    FANUC RISE // BUILD PROTOCOL
ECHO    PREPARING RELEASE ARTIFACTS...
ECHO ========================================================
ECHO.

:: 1. Clean Dist
IF EXIST "dist" (
    ECHO [1/5] CLEANING PREVIOUS BUILD...
    rmdir /s /q dist
)
mkdir dist
mkdir dist\backend
mkdir dist\extensions
mkdir dist\docs

:: 2. Build Frontend
ECHO [2/5] BUILDING FRONTEND (REACT)...
cd frontend-react
call npm run build
IF %ERRORLEVEL% NEQ 0 (
    ECHO [ERROR] FRONTEND BUILD FAILED.
    PAUSE
    EXIT /B
)
xcopy /E /I dist ..\dist\frontend_build
cd ..

:: 3. Package Backend
ECHO [3/5] PACKAGING BACKEND CORE...
xcopy /E /I backend dist\backend
xcopy /E /I cms dist\cms
copy config.py dist\
copy requirements.txt dist\
copy .env.example dist\.env

:: 4. Extensions & Docs
ECHO [4/5] COLLECTING EXTENSIONS & DOCS...
xcopy /E /I extensions dist\extensions
copy *.md dist\docs\ 

:: 5. Create Launcher
ECHO [5/5] FINALIZING LAUNCHER...
(
echo @echo off
echo TITLE FANUC RISE // PRODUCTION
echo echo STARTING PRODUCTION SERVER...
echo start "RISE_CORE" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
echo echo SERVING FRONTEND...
echo cd frontend_build
echo python -m http.server 3000
) > dist\run_production.bat

ECHO.
ECHO ========================================================
ECHO    BUILD COMPLETE.
ECHO    ARTIFACTS LOCATED IN: /dist
ECHO ========================================================
PAUSE
