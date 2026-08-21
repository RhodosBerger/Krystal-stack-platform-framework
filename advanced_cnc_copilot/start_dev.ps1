
Write-Host "=== FANUC RISE SYSTEM LAUNCHER ==="
Write-Host "1. Installing Frontend Dependencies (if missing)..."

$frontendPath = ".\frontend-vue"
if (Test-Path $frontendPath) {
    Push-Location $frontendPath
    
    Write-Host "1a. Checking Node.js..."
    if (-not (Get-Command "npm" -ErrorAction SilentlyContinue)) {
        Write-Error "CRITICAL: NPM not found. Install Node.js!"
        exit
    }

    Write-Host "1b. Installing/Updating Frontend Dependencies..."
    # Fix for Rollup/Win32 error: Clean install is often required
    if (Test-Path "node_modules") { Remove-Item -Recurse -Force "node_modules" }
    if (Test-Path "package-lock.json") { Remove-Item -Force "package-lock.json" }
    
    npm install
    
    Pop-Location
}
else {
    Write-Error "Frontend directory not found!"
    exit
}

Write-Host "1b. Installing Python Dependencies (if missing)..."
if (Test-Path "requirements.txt") {
    try {
        Write-Host "Installing/Updating Python requirements..."
        python -m pip install -r requirements.txt
    }
    catch {
        Write-Error "Failed to install Python dependencies. Please check your Python installation."
        Write-Host "Press any key to continue..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    }
}

Write-Host "2. Launching Backend (FastAPI)..."
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "python -m uvicorn backend.main:app --reload --port 8000"

Write-Host "3. Launching Frontend (Vite)..."
Push-Location $frontendPath
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "npm run dev"
Pop-Location

Write-Host "4. Opening Browser..."
Start-Sleep -Seconds 3
Start-Process "http://localhost:5174"

Write-Host "System Running. Check the new PowerShell windows for logs."
