<#
.SYNOPSIS
    Single documented entry point for running the Krystal-Stack test suites on Windows.

.DESCRIPTION
    Uses the standard-library unittest runner, so no extra packages (pytest) are required.
    The repository's `venv/` folder has a Linux layout (bin/, lib64) and cannot be used on
    Windows; `.venv-keras` does not contain pytest. This script therefore uses `python` from PATH
    unless -Python is given.

.EXAMPLE
    ./scripts/run_tests.ps1                     # CrystalSec suites only (fast)
    ./scripts/run_tests.ps1 -Pattern "test_*.py" # whole tests/ directory
    ./scripts/run_tests.ps1 -Python C:\path\to\python.exe
#>
param(
    [string]$Pattern = "test_crystalsec*.py",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    & $Python --version
    & $Python -m unittest discover -s tests -p $Pattern
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $code
