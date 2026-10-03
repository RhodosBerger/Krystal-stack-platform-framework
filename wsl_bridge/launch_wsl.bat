@echo off
title Krystal-Stack WSL2 Bridge
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\wsl_launcher.ps1"
pause
