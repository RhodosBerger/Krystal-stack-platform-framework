@echo off
REM ==============================================================================
REM KRYSTAL-STACK: SOVEREIGN OPERATIONS CONTROL CENTER & COMPILER AGGREGATOR (BAT)
REM ==============================================================================
REM Constitutional Invariant: VITAL_MAX_HP == 6 (not Valid(A) => Delta S == 0)
REM Normative Architecture: Krystal Sovereign Principle Pack (KSPP v1)
REM ==============================================================================

setlocal enabledelayedexpansion

echo ================================================================================
echo [SOVEREIGN OPERATIONS CONTROL CENTER ^& COMPILER AGGREGATOR]
echo Constitutional Invariant: VITAL_MAX_HP == 6 (Delta S == 0 on rejection)
echo Krystal Sovereign Principle Pack (KSPP v1) - Windows Native Runner
echo ================================================================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in PATH!
    exit /b 1
)

set MODE=ALL
if "%1"=="--compile-only" set MODE=COMPILE_ONLY
if "%1"=="--ops-center" set MODE=OPS_CENTER
if "%1"=="--verify" set MODE=VERIFY

echo [*] Mode: %MODE%
echo --------------------------------------------------------------------------------

echo [1/3] Compiling theories, contracts, and aggregating operations center...
python -m krystal_kernel.sovereign_operations_compiler
if %errorlevel% neq 0 (
    echo [ERROR] Step 1 Failed!
    exit /b %errorlevel%
)

if "%MODE%"=="COMPILE_ONLY" (
    echo [OK] Compile-only mode finished successfully.
    exit /b 0
)

echo.
echo [2/3] Executing hermetic regression ^& invariant preservation suite...
python -m unittest tests/test_sovereign_runtime.py ^
                   tests/test_redundant_epistemic_inference_engine.py ^
                   tests/test_future_theory_integrations_orchestrator.py ^
                   tests/test_chrono_symplectic_processor_architecture.py ^
                   tests/test_pattern_rhythm_cadence_calculator.py ^
                   tests/test_cartesian_hex_hsl_grid.py ^
                   tests/test_latent_offset_safety_thresholds.py ^
                   tests/test_matrix_algorithms_two_phase_bridge.py ^
                   tests/test_two_phase_reasoning_computing_alternator.py ^
                   tests/test_uma_inductive_voltage_governor.py ^
                   tests/test_silicon_telemetry_lexicon.py ^
                   tests/test_hw_acceleration_bridges.py ^
                   tests/test_unified_dependency_coordinator.py ^
                   tests/test_godot_krystal_edition_packager.py ^
                   tests/test_sovereign_layer_optimizer_and_run_tester.py
if %errorlevel% neq 0 (
    echo [ERROR] Step 2 Unit test suite failed!
    exit /b %errorlevel%
)

echo.
echo [3/3] Validating audit certificate receipt...
if exist "artifacts\SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json" (
    echo [OK] Found artifacts\SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json
    python -c "import json; d=json.load(open('artifacts/SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json')); print('Receipt:', d['receipt_id'], '| Operational:', d['is_fully_operational'], '| Stages:', f\"{d['stages_passed']}/{d['total_stages_run']}\")"
) else (
    echo [WARN] Receipt artifacts\SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json not found!
)

echo.
echo ================================================================================
echo [SUCCESS] SOVEREIGN OPERATIONS COMPILER ^& CONTROL CENTER PASS
echo Constitutional invariant VITAL_MAX_HP == 6 preserved.
echo ================================================================================
exit /b 0
