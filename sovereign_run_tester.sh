#!/usr/bin/env bash
# sovereign_run_tester.sh
# One-Click Automated Run-Testing for Sovereign Layers (POSIX / WSL)
# Invariant: VITAL_MAX_HP == 6
set -euo pipefail
echo "========================================================"
echo "  KRYSTAL STACK // SOVEREIGN RUN-TESTER (POSIX / WSL)"
echo "========================================================"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${REPO_DIR}/run_sovereign_tests.py" "$@"
