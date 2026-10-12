# CrystalSec 3.0 Phase 5.3C - Native CPU SIMD Reproducible Build Script
# Compiles src/c/taxonomy_simd.c into taxonomy_simd.dll and generates SHA-256 manifest.
param(
    [string]$SourceDir = (Join-Path $PSScriptRoot "..\src\c"),
    [string]$OutputDir = (Join-Path $PSScriptRoot "..\src\c"),
    [string]$ManifestDir = (Join-Path $PSScriptRoot "..\artifacts"),
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$cSource = Join-Path $SourceDir "taxonomy_simd.c"
$cHeader = Join-Path $SourceDir "taxonomy_simd.h"
$dllPath = Join-Path $OutputDir "taxonomy_simd.dll"
$manifestPath = Join-Path $ManifestDir "taxonomy_simd_manifest.json"

if (-not (Test-Path $cSource)) {
    throw "Source file not found: $cSource"
}

Write-Host "=================================================="
Write-Host "CrystalSec 3.0 - Building Native CPU SIMD Kernels"
Write-Host "=================================================="

$compiler = $null
$compilerType = $null
$compilerVersion = $null
$compilerFlags = @()

# Detect toolchains
$clCmd = Get-Command cl -ErrorAction SilentlyContinue
$clangCmd = Get-Command clang -ErrorAction SilentlyContinue
$gccCmd = Get-Command gcc -ErrorAction SilentlyContinue

if ($clCmd) {
    $compiler = $clCmd.Source
    $compilerType = "MSVC"
    $compilerFlags = @("/O2", "/W4", "/LD", "/arch:AVX2", "/fp:precise", "/Fe:`"$dllPath`"", "`"$cSource`"")
} elseif ($clangCmd) {
    $compiler = $clangCmd.Source
    $compilerType = "Clang"
    $compilerFlags = @("-O3", "-Wall", "-shared", "-mavx2", "-mfma", "-fPIC", "-o", "`"$dllPath`"", "`"$cSource`"")
} elseif ($gccCmd) {
    $compiler = $gccCmd.Source
    $compilerType = "GCC"
    $compilerFlags = @("-O3", "-Wall", "-shared", "-mavx2", "-mfma", "-fPIC", "-o", "`"$dllPath`"", "`"$cSource`"")
}

if (-not $compiler) {
    Write-Host "INFO: No native C compiler (cl.exe, clang, gcc) found in PATH."
    Write-Host "Skipping DLL compilation. Source files and verification scripts remain available."
    exit 0
}

Write-Host "Detected Compiler: $compilerType ($compiler)"
& $compiler $compilerFlags
if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE"
}

if (Test-Path $dllPath) {
    $dllHash = (Get-FileHash -Path $dllPath -Algorithm SHA256).Hash.ToLower()
    $srcHash = (Get-FileHash -Path $cSource -Algorithm SHA256).Hash.ToLower()

    $manifest = @{
        "schema_version" = "1.0.0"
        "target_binary" = "taxonomy_simd.dll"
        "target_architecture" = "x86_64"
        "compiler" = $compilerType
        "compiler_path" = $compiler
        "compiler_flags" = $compilerFlags
        "build_timestamp" = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
        "dll_sha256" = $dllHash
        "source_sha256" = $srcHash
        "supported_kernels" = @(
            "taxonomy_int8_dot",
            "taxonomy_fp32_poly",
            "taxonomy_memory_stream",
            "taxonomy_branch_control",
            "taxonomy_mixed_compute_memory"
        )
        "evidence_tier" = "T1_NATIVE_CPU"
    }

    if (-not (Test-Path $ManifestDir)) {
        New-Item -ItemType Directory -Path $ManifestDir -Force | Out-Null
    }
    $manifestJson = $manifest | ConvertTo-Json -Depth 4
    Set-Content -Path $manifestPath -Value $manifestJson
    Write-Host "SUCCESS: Built $dllPath"
    Write-Host "Manifest written to: $manifestPath (SHA-256: $dllHash)"
}
