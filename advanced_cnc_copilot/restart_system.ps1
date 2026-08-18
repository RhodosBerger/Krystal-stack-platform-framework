Write-Host "=== FANUC RISE: BACKEND RESTART ===" -ForegroundColor Cyan
Write-Host "1. Stopping existing services..."

Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "uvicorn" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "node" -Force -ErrorAction SilentlyContinue

Write-Host "2. Relaunching System..."
.\rescue_launch.ps1
