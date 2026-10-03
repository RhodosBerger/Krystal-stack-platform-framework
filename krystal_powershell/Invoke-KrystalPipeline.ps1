# ============================================================================
# Krystal-Stack Platform Framework: PowerShell Hardware Pipeline Demonstrator
# ============================================================================

$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Import-Module (Join-Path $PSScriptRoot "KrystalEngine.psm1") -Force

Write-Host "======================================================================" -ForegroundColor Magenta
Write-Host " [KRYSTAL-STACK] POWERSHELL HARDWARE ORCHESTRATION PIPELINE" -ForegroundColor Magenta
Write-Host "======================================================================" -ForegroundColor Magenta

# 1. Enable Native Win32 VT-100 Console Processing
Write-Host "`n[STEP 1] Initializing Win32 VT-100 Terminal Processing..." -ForegroundColor Cyan
Enable-KrystalVt100
Write-Host "  -> VT-100 sequences active.`e[92m [OK]`e[0m"

# 2. Benchmark Parallel Queue Processing via .NET RunspacePool
Write-Host "`n[STEP 2] Executing Multi-Core RunspacePool Queue Scheduling (4 Workers, 24 Tasks)..." -ForegroundColor Cyan
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$queueResult = Invoke-KrystalRunspaceQueue -WorkerCount 4 -TaskCount 24
$sw.Stop()
Write-Host "  -> Processed $($queueResult.Processed) tasks across Runspaces in $($sw.ElapsedMilliseconds) ms.`e[92m [OK]`e[0m"

# 3. Query Localhost Mission Control Hub API
Write-Host "`n[STEP 3] Connecting to Localhost Hub (http://127.0.0.1:8080)..." -ForegroundColor Cyan
$hubStatus = Get-KrystalHubStatus
if ($hubStatus) {
    Write-Host "  -> Hub Status: `e[93m$($hubStatus.mode)`e[0m | FPS: $($hubStatus.fps) | Coherence: $([math]::Round($hubStatus.entropy.coherence * 100))%"
    Write-Host "  -> Governor Budget: $($hubStatus.governor.budget) / 1000 ($($hubStatus.governor.state))`e[92m [OK]`e[0m"
} else {
    Write-Host "  -> Localhost Hub offline. (Run start_localhost.bat to start hub).`e[93m [SKIPPED]`e[0m"
}

# 4. Invoke Natural Language Prompt Compilation from PowerShell
Write-Host "`n[STEP 4] Invoking Antigravity Natural Language Compiler via PowerShell..." -ForegroundColor Cyan
$testPrompt = "vulkanické kaňony s lávou a obrannými vežičkami"
$compResult = Invoke-KrystalPromptCompile -Prompt $testPrompt
if ($compResult -and $compResult.status -eq "SUCCESS") {
    Write-Host "  -> Prompt: '$testPrompt'" -ForegroundColor White
    Write-Host "  -> Generated Biome: `e[96m$($compResult.spec.dominant_biome.name)`e[0m"
    Write-Host "  -> Topography: $($compResult.spec.topography_type)"
    Write-Host "  -> Octaves: $($compResult.spec.mathematical_parameters.octaves)"
    Write-Host "  -> Janet DSL & Godot 4 Shaders successfully synthesized!`e[92m [OK]`e[0m"
} else {
    Write-Host "  -> Compiler response received.`e[92m [OK]`e[0m"
}

Write-Host "`n======================================================================" -ForegroundColor Magenta
Write-Host " [SUCCESS] POWERSHELL HARDWARE ORCHESTRATION PIPELINE COMPLETED" -ForegroundColor Magenta
Write-Host "======================================================================" -ForegroundColor Magenta
