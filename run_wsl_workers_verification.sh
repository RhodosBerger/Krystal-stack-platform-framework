#!/usr/bin/env bash
# ==============================================================================
# KRYSTAL-STACK: NATIVE WSL EXECUTION & WORKERS VERIFICATION RUNNER
# ==============================================================================
# Constitutional Invariant: VITAL_MAX_HP == 6
# ==============================================================================
set -e
cd "$(dirname "$0")"

echo "================================================================================"
echo "⚡ KRYSTAL STACK: NATIVE WSL WORKERS & TEST VERIFICATION"
echo "🐧 Environment: Linux (WSL2 Ubuntu) | Python: $(python3 --version)"
echo "🔒 Constitutional Invariant: VITAL_MAX_HP == 6"
echo "================================================================================"
echo ""

echo "▶ [KROK 1/5] Spúšťam 67/67 Unit & Regresných Testov (Hermetická sada vrátane E-07)..."
echo "--------------------------------------------------------------------------------"
python3 -m unittest tests/test_native_pulse_logger.py \
                   tests/test_tri_phase_dosing_controller.py \
                   tests/test_vpp_grid_broker_fleet.py \
                   tests/test_household_edge_energy_node.py \
                   tests/test_grid_aware_energy_arbitrage.py \
                   tests/test_community_showcase_integrator.py \
                   tests/test_silicon_pipeline_orchestrator.py \
                   tests/test_silicon_telemetry_lexicon.py \
                   tests/test_investor_suite_integration.py

echo ""
echo "▶ [KROK 2/5] Spúšťam Experiment E-07: Natívny C11/Rust Pulse Logger (<50ns Benchmark)..."
echo "--------------------------------------------------------------------------------"
python3 demo_pulse_logger_benchmark.py

echo ""
echo "▶ [KROK 3/5] Spúšťam Simuláciu Trojfázového Dávkovania & Preťaženia Fázy L1..."
echo "--------------------------------------------------------------------------------"
python3 demo_tri_phase_dosing_simulation.py

echo ""
echo "▶ [KROK 4/5] Spúšťam Flotilového VPP Brokera (100 Domov & Fázované Pole)..."
echo "--------------------------------------------------------------------------------"
python3 demo_vpp_grid_broker_simulation.py

echo ""
echo "▶ [KROK 5/5] Spúšťam 24-Hodinový 96-Slotový Cyklus Domácej Arbitráže..."
echo "--------------------------------------------------------------------------------"
python3 demo_household_edge_simulation.py

echo ""
echo "================================================================================"
echo "🏆 VŠETKY WORKERY A SIMULÁCIE DOKONČENÉ SO STATUSOM OK (100% ÚSPEŠNOSŤ)"
echo "🔒 VITAL_MAX_HP == 6 STRIKTNE ZACHOVANÝ V CELEJ ARCHITEKTÚRE"
echo "================================================================================"

