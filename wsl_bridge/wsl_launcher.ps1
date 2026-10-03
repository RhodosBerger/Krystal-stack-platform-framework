# ==============================================================================
# KRYSTAL-STACK // MICROSOFT WSL2 LAUNCHER & ORCHESTRATOR
# ==============================================================================
Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "         KRYSTAL-STACK WSL2 SUBSYSTEM ORCHESTRATOR" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan

# 1. Verify WSL status
Write-Host "[1/3] Checking Microsoft WSL2 Ubuntu Subsystem..." -ForegroundColor Yellow
$wslCheck = wsl --list --running 2>$null
Write-Host "Active WSL Distributions:" -ForegroundColor DarkGray
wsl -l -v

# 2. Path mapping
$currentWinPath = (Get-Location).Path
$wslPath = (wsl wslpath -u "'$currentWinPath'").Trim()
Write-Host "[2/3] Mapped Workspace in WSL: $wslPath" -ForegroundColor Green

# 3. Launch Bridge Daemon in WSL
Write-Host "[3/3] Starting WSL Bridge Daemon inside Linux subsystem..." -ForegroundColor Cyan
Write-Host "Command: wsl bash -c 'cd $wslPath/wsl_bridge && python3 bridge_daemon.py'" -ForegroundColor DarkGray

wsl bash -c "cd '$wslPath/wsl_bridge' && python3 bridge_daemon.py"
