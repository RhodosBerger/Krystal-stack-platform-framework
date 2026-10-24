#!/usr/bin/env bash
# Godot Krystal Edition Launcher (POSIX / WSL)
# Invariant: VITAL_MAX_HP == 6
set -euo pipefail
echo "========================================================"
echo "  LAUNCHING GODOT KRYSTAL EDITION (v1.0.0-SOVEREIGN)"
echo "========================================================"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GODOT_BIN="${REPO_DIR}/godot_project/godot.x86_64"
if [ ! -f "$GODOT_BIN" ]; then
    if command -v godot >/dev/null 2>&1; then
        GODOT_BIN="godot"
    elif command -v godot4 >/dev/null 2>&1; then
        GODOT_BIN="godot4"
    else
        echo "[ERROR] Godot 4.x executable not found in PATH or godot_project/."
        exit 1
    fi
fi
"$GODOT_BIN" --path "${REPO_DIR}/godot_project" res://scenes/GodotKrystalEditionStage.tscn "$@"
