@echo off
setlocal

echo ================================================================================
echo  KRYSTAL SYSTEMS LANGUAGE [KSL] - SOVEREIGN COMPILER DEMONSTRATION
echo  Constitutional Invariant: VITAL_MAX_HP == 6 [Delta S == 0 on breach]
echo ================================================================================

python demo_ksl_pipeline.py
if errorlevel 1 (
    echo [ERROR] KSL demonstration encountered an error.
    exit /b 1
)

echo [OK] KSL demonstration executed successfully.
exit /b 0
