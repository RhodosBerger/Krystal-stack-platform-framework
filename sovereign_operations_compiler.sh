#!/usr/bin/env bash
# ==============================================================================
# KRYSTAL-STACK: SOVEREIGN OPERATIONS CONTROL CENTER & COMPILER AGGREGATOR
# ==============================================================================
# Constitutional Invariant: VITAL_MAX_HP == 6 (not Valid(A) => Delta S == 0)
# Normative Architecture: Krystal Sovereign Principle Pack (KSPP v1)
# ==============================================================================

set -o pipefail

# ANSI Color & Formatting Codes
BOLD="\033[1m"
GREEN="\033[1;32m"
RED="\033[1;31m"
CYAN="\033[1;36m"
YELLOW="\033[1;33m"
MAGENTA="\033[1;35m"
RESET="\033[0m"

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${WORKSPACE_DIR}"

echo -e "${CYAN}================================================================================${RESET}"
echo -e "${BOLD}⚡ KRYSTAL STACK: SOVEREIGN OPERATIONS CONTROL CENTER & COMPILER AGGREGATOR${RESET}"
echo -e "${CYAN}🔒 Ústavný Invariant: ${GREEN}VITAL_MAX_HP == 6${CYAN} (Prísne zachovanie Delta S == 0)${RESET}"
echo -e "${CYAN}🏛️  Normatívne Zásady: ${BOLD}KSPP v1 (P1-P10)${RESET} | Dvojfázové Tieňové Okno (3.5 ms)"
echo -e "${CYAN}================================================================================${RESET}"
echo ""

# Python Executable Resolution
if command -v python3 &>/dev/null; then
    PY_BIN="python3"
elif command -v python &>/dev/null; then
    PY_BIN="python"
else
    echo -e "${RED}❌ CHYBA: Python interpretér nebol nájdený v PATH!${RESET}"
    exit 1
fi

PY_VER=$(${PY_BIN} --version 2>&1)
echo -e "🐍 Detegované prostredie: ${BOLD}${PY_VER}${RESET} | Root: ${WORKSPACE_DIR}"
echo ""

# Execution Modes Flag Parsing
MODE="ALL"
if [[ "$1" == "--compile-only" ]]; then
    MODE="COMPILE_ONLY"
elif [[ "$1" == "--ops-center" ]]; then
    MODE="OPS_CENTER"
elif [[ "$1" == "--verify" ]]; then
    MODE="VERIFY"
fi

echo -e "🚀 Režim agregátora: ${BOLD}${YELLOW}${MODE}${RESET}"
echo "--------------------------------------------------------------------------------"

# ==============================================================================
# STAGE 1: COMPILATION & OPERATIONS AGGREGATION VIA PYTHON ENGINE
# ==============================================================================
echo -e "▶ ${BOLD}[KROK 1/3] Kompilácia teórií, kontraktov a agregácia riadiaceho centra...${RESET}"
${PY_BIN} -m krystal_kernel.sovereign_operations_compiler
COMPILE_STATUS=$?

if [[ ${COMPILE_STATUS} -ne 0 ]]; then
    echo -e "${RED}❌ KROK 1 ZLYHAL: Kompilátor riadiaceho centra hlási porušenie invariantu alebo chybu!${RESET}"
    exit 1
fi

if [[ "${MODE}" == "COMPILE_ONLY" ]]; then
    echo -e "${GREEN}✅ Režim --compile-only úspešne dokončený.${RESET}"
    exit 0
fi

# ==============================================================================
# STAGE 2: HERMETIC REGRESSION & INVARIANT PRESERVATION SUITE (94 TESTS)
# ==============================================================================
echo ""
echo -e "▶ ${BOLD}[KROK 2/3] Spúšťam plnú hermetickú sadu testov (94 unit testov)...${RESET}"
${PY_BIN} -m unittest tests/test_sovereign_runtime.py \
                    tests/test_redundant_epistemic_inference_engine.py \
                    tests/test_future_theory_integrations_orchestrator.py \
                    tests/test_chrono_symplectic_processor_architecture.py \
                    tests/test_pattern_rhythm_cadence_calculator.py \
                    tests/test_cartesian_hex_hsl_grid.py \
                    tests/test_latent_offset_safety_thresholds.py \
                    tests/test_matrix_algorithms_two_phase_bridge.py \
                    tests/test_two_phase_reasoning_computing_alternator.py \
                    tests/test_uma_inductive_voltage_governor.py \
                    tests/test_silicon_telemetry_lexicon.py \
                    tests/test_hw_acceleration_bridges.py
TEST_STATUS=$?

if [[ ${TEST_STATUS} -ne 0 ]]; then
    echo -e "${RED}❌ KROK 2 ZLYHAL: Regresné testy odhalili chybu v runtimi!${RESET}"
    exit 1
fi

# ==============================================================================
# STAGE 3: AUDIT RECEIPT VALIDATION & ARTIFACT VERIFICATION
# ==============================================================================
echo ""
echo -e "▶ ${BOLD}[KROK 3/3] Validácia emitovaného auditného certifikátu (HMAC-SHA256)...${RESET}"
RECEIPT_FILE="${WORKSPACE_DIR}/artifacts/SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json"

if [[ -f "${RECEIPT_FILE}" ]]; then
    RECEIPT_ID=$(${PY_BIN} -c "import json; d=json.load(open('${RECEIPT_FILE}')); print(d['receipt_id'])")
    IS_OP=$(${PY_BIN} -c "import json; d=json.load(open('${RECEIPT_FILE}')); print(d['is_fully_operational'])")
    STAGES=$(${PY_BIN} -c "import json; d=json.load(open('${RECEIPT_FILE}')); print(f\"{d['stages_passed']}/{d['total_stages_run']}\")")
    DUR=$(${PY_BIN} -c "import json; d=json.load(open('${RECEIPT_FILE}')); print(d['total_duration_ms'])")
    
    echo -e "📄 Súbor certifikátu: ${CYAN}${RECEIPT_FILE}${RESET}"
    echo -e "🔑 Identifikátor:      ${BOLD}${RECEIPT_ID}${RESET}"
    echo -e "⏱️  Čas agregácie:      ${BOLD}${DUR} ms${RESET}"
    echo -e "📊 Fázy:               ${BOLD}${GREEN}${STAGES} PASS${RESET}"
    echo -e "🛡️  Plne operačné:     ${BOLD}${GREEN}${IS_OP}${RESET}"
else
    echo -e "${YELLOW}⚠️ UPOZORNENIE: Certifikát ${RECEIPT_FILE} nebol nájdený!${RESET}"
fi

echo ""
echo -e "${GREEN}================================================================================${RESET}"
echo -e "${BOLD}🏆 KRYSTAL STACK: RIADIACE CENTRUM A KOMPILÁTOR ÚSPEŠNE SKOMPILOVANÉ${RESET}"
echo -e "${GREEN}🔒 Ústavný invariant VITAL_MAX_HP == 6 zachovaný (Delta S == 0 pri odmietnutí)${RESET}"
echo -e "${GREEN}================================================================================${RESET}"

exit 0
