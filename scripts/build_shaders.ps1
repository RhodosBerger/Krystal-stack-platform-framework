# Compiles CrystalSec taxonomy compute shaders to SPIR-V and validates them.
# Requires the Vulkan SDK (glslc, spirv-val). The resulting .spv files are committed so that
# runtimes without the SDK can still dispatch the kernels.
param(
    [string]$ShaderDir = (Join-Path $PSScriptRoot "..\src\shaders\taxonomy")
)
$ErrorActionPreference = "Stop"

$glslc = (Get-Command glslc -ErrorAction SilentlyContinue).Source
if (-not $glslc -and $env:VULKAN_SDK) { $glslc = Join-Path $env:VULKAN_SDK "Bin\glslc.exe" }
$spirvVal = (Get-Command spirv-val -ErrorAction SilentlyContinue).Source
if (-not $spirvVal -and $env:VULKAN_SDK) { $spirvVal = Join-Path $env:VULKAN_SDK "Bin\spirv-val.exe" }
if (-not $glslc -or -not (Test-Path $glslc)) { throw "glslc not found (install the Vulkan SDK)." }

$failed = 0
Get-ChildItem -Path $ShaderDir -Filter *.comp | ForEach-Object {
    $spv = [System.IO.Path]::ChangeExtension($_.FullName, ".spv")
    & $glslc -O --target-env=vulkan1.1 -o $spv $_.FullName
    if ($LASTEXITCODE -ne 0) { Write-Host "FAIL compile $($_.Name)"; $failed++; return }
    if ($spirvVal) {
        & $spirvVal --target-env vulkan1.1 $spv
        if ($LASTEXITCODE -ne 0) { Write-Host "FAIL validate $($_.Name)"; $failed++; return }
    }
    Write-Host ("OK   {0} -> {1} ({2} bytes)" -f $_.Name, (Split-Path $spv -Leaf), (Get-Item $spv).Length)
}
if ($failed -gt 0) { exit 1 }
