<#
.SYNOPSIS
    Optimized Launcher for FANUC RISE (FastAPI + React)
    Checks dependencies, verifies Python version, and launches services.
#>

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "   FANUC RISE // OPTIMIZED LAUNCHER" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python Version
Write-Host "[1/4] Checking Python Environment..." -ForegroundColor Yellow
try {
    $pyVersion = python --version 2>&1
    Write-Host "      Detected: $pyVersion" -ForegroundColor Gray
    
    if ($pyVersion -match "3\.14") {
        Write-Host "      [CRITICAL WARNING] Python 3.14 detected!" -ForegroundColor Red
        Write-Host "      This version is currently INCOMPATIBLE with core libraries." -ForegroundColor Red
        Write-Host "      Please install Python 3.11 or 3.12 for stability." -ForegroundColor Red
        Write-Host "      Continuing at your own risk..." -ForegroundColor DarkGray
        Start-Sleep -Seconds 3
    }
}
catch {
    Write-Error "Python not found! Please install Python 3.11+."
    exit 1
}

# 2. Check Virtual Environment
if (-not (Test-Path "venv")) {
    Write-Host "[2/4] Creating Virtual Environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "      Installing Dependencies..."
    .\venv\Scripts\pip install -r requirements.txt
}
else {
    Write-Host "      Virtual Environment found." -ForegroundColor Gray
}

# 3. Check Docker Status (Container Deployment)
Write-Host "[3/5] Checking Container Services..." -ForegroundColor Yellow
$dockerStatus = docker info 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "      Docker Daemon is RUNNING." -ForegroundColor Green
}
else {
    Write-Host "      [WARNING] Docker Desktop is NOT running or requires Admin privileges." -ForegroundColor Red
    Write-Host "      The 'Deployment' stack will fail, but you can run locally." -ForegroundColor Gray
}

# 4. Launch Backend (FastAPI)
Write-Host "[4/5] Launching Backend (FastAPI :8000)..." -ForegroundColor Yellow
$backendProcess = Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", ".\venv\Scripts\python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000" -PassThru

# 4. Launch Frontend (Vite)
Write-Host "[4/4] Launching Frontend (React :3000)..." -ForegroundColor Yellow
if (Test-Path "frontend-react") {
    Push-Location "frontend-react"
    if (-not (Test-Path "node_modules")) {
        Write-Host "      Installing Node Modules..." -ForegroundColor Gray
        npm install
    }
    $frontendProcess = Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "npm run dev" -PassThru
    Pop-Location
}
else {
    Write-Error "frontend-react directory not found!"
}

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "   SYSTEM STARTING..." -ForegroundColor Green
Write-Host "   Backend:  http://localhost:8000/docs" -ForegroundColor Green
Write-Host "   Frontend: http://localhost:3000" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
