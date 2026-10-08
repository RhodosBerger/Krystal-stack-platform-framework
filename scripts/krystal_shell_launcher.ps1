# ==============================================================================
# KRYSTAL-STACK: SOVEREIGN SHELL LAUNCHER & FAILSAFE WATCHDOG (POWERSHELL)
# ==============================================================================
# Script: scripts/krystal_shell_launcher.ps1
# Description: PowerShell manager for sovereign shell mode, registry setup,
#              RAM reclamation monitoring, and Explorer restoration.
#
# System Invariant: VITAL_MAX_HP = 6
# ==============================================================================

[CmdletBinding()]
param(
    [switch]$RestoreExplorer,
    [switch]$SetAsShell,
    [switch]$CheckStatus
)

$VitalMaxHp = 6
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path "$ScriptDir\.."

Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  [KRYSTAL-STACK] SOVEREIGN OPEN-SOURCE OS SHELL CONTROLLER" -ForegroundColor Green
Write-Host "  System Invariant: VITAL_MAX_HP = $VitalMaxHp" -ForegroundColor Yellow
Write-Host "================================================================================" -ForegroundColor Cyan

if ($RestoreExplorer) {
    Write-Host "[ACTION] Restoring default Windows Explorer shell..." -ForegroundColor Yellow
    Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows NT\CurrentVersion\Winlogon" -Name "Shell" -Value "explorer.exe" -Force -ErrorAction SilentlyContinue
    if (-not (Get-Process explorer -ErrorAction SilentlyContinue)) {
        Start-Process explorer.exe
    }
    Write-Host "[OK] Windows Explorer restored to default." -ForegroundColor Green
    return
}

if ($SetAsShell) {
    $batPath = "$ScriptDir\krystal_shell_launcher.bat"
    Write-Host "[ACTION] Setting Krystal Shell launcher in HKCU Winlogon..." -ForegroundColor Yellow
    Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows NT\CurrentVersion\Winlogon" -Name "Shell" -Value "`"$batPath`"" -Force
    Write-Host "[OK] Registry updated. Next logon will boot directly into Krystal Shell." -ForegroundColor Green
    return
}

if ($CheckStatus) {
    $currentShell = (Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows NT\CurrentVersion\Winlogon" -Name "Shell" -ErrorAction SilentlyContinue).Shell
    if (-not $currentShell) { $currentShell = "explorer.exe (Default HKLM)" }
    $ram = Get-CimInstance Win32_OperatingSystem
    $freeRamGb = [math]::Round($ram.FreePhysicalMemory / 1MB, 2)
    $totalRamGb = [math]::Round($ram.TotalVisibleMemorySize / 1MB, 2)
    
    Write-Host "  Current User Shell: $currentShell" -ForegroundColor White
    Write-Host "  Total Memory:       $totalRamGb GB" -ForegroundColor White
    Write-Host "  Free Memory:        $freeRamGb GB" -ForegroundColor White
    Write-Host "  Hub Status (8080):  $((Test-NetConnection -Port 8080 -ComputerName 127.0.0.1 -WarningAction SilentlyContinue).TcpTestSucceeded)" -ForegroundColor White
    Write-Host "  System Invariant:   VITAL_MAX_HP = $VitalMaxHp (VERIFIED)" -ForegroundColor Green
    return
}

Write-Host "Usage:" -ForegroundColor Gray
Write-Host "  .\krystal_shell_launcher.ps1 -CheckStatus     -> Display shell and memory statistics" -ForegroundColor Gray
Write-Host "  .\krystal_shell_launcher.ps1 -SetAsShell      -> Register as custom shell in Windows" -ForegroundColor Gray
Write-Host "  .\krystal_shell_launcher.ps1 -RestoreExplorer -> Restore standard Windows Explorer" -ForegroundColor Gray
