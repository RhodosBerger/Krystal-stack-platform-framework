# ============================================================================
# Krystal-Stack Janet Subproject Launcher (PowerShell)
# ============================================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path "$ScriptDir\.."
$env:PYTHONPATH = "$RepoRoot;$env:PYTHONPATH"
Set-Location $ScriptDir

$JanetCmd = Get-Command janet -ErrorAction SilentlyContinue

if ($JanetCmd) {
    Write-Host "[KRYSTAL-JANET] Native Janet runtime detected in PATH: $($JanetCmd.Source)" -ForegroundColor Green
    & janet main.janet $args
} else {
    Write-Host "[KRYSTAL-JANET] Native Janet binary not found in PATH." -ForegroundColor Yellow
    Write-Host "[KRYSTAL-JANET] Launching via Krystal Janet Bridge Engine..." -ForegroundColor Cyan
    python -m krystal_janet.janet_bridge $args
}
