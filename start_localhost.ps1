# Krystal-Stack Localhost Mission Control Launcher (PowerShell)
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "         KRYSTAL-STACK HETEROGENEOUS NEURAL ASCII PLATFORM" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host ""

$pythonCheck = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCheck) {
    Write-Host "[ERROR] Python was not found in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Python detected at: $($pythonCheck.Source)" -ForegroundColor Green
Write-Host "[OK] Launching Localhost Hub on http://localhost:8080 ..." -ForegroundColor Yellow
Write-Host ""

python start_localhost.py
