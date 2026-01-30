Write-Host "=== FANUC RISE RESCUE LAUNCHER ===" -ForegroundColor Cyan
Write-Host "Diagnosing startup issues..."

# 1. Check Python Environment
if (-not (Test-Path "venv")) {
    Write-Host "[!] Virtual Environment not found. Creating..." -ForegroundColor Yellow
    python -m venv venv
}
Write-Host "[*] Activating Python Virtual Environment..."
.\[venv](venv)\Scripts\Activate.ps1

# 2. Check Backend Dependencies
Write-Host "[*] Checking Backend Dependencies..."
pip install -r requirements.txt | Out-Null

# 3. Check Frontend Dependencies
$frontendPath = "frontend-react"
if (-not (Test-Path "$frontendPath\node_modules")) {
    Write-Host "[!] Frontend dependencies missing. Installing..." -ForegroundColor Yellow
    Push-Location $frontendPath
    npm install
    Pop-Location
}

# 4. Launch Services
Write-Host "[*] Starting Backend Service (Port 8000)..." -ForegroundColor Green
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", ".\venv\Scripts\Activate.ps1; uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"

Write-Host "[*] Starting Frontend Service (Port 3000+)..." -ForegroundColor Green
Push-Location $frontendPath
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "npm run dev"
Pop-Location

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "System Launching..."
Write-Host "1. Look for two new PowerShell windows."
Write-Host "2. Wait ~10 seconds for servers to boot."
Write-Host "3. Browser will open automatically."
Write-Host "==============================================" -ForegroundColor Cyan

Start-Sleep -Seconds 5
Start-Process "http://localhost:3000/?role=ADMIN"
