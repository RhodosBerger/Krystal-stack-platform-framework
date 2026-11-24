#!/usr/bin/env bash
# Spúšťač modulov GNOME + GDM session + Godot 4 WebGL2/GLES3 priamo cez WSLg v termináli
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GODOT_PROJ="${SCRIPT_DIR}/godot_project"

export KRYSTAL_VITAL_MAX_HP=6
export KRYSTAL_GDM_SESSION_MODE="${KRYSTAL_GDM_SESSION_MODE:-MODE_02_NIRI_SCROLLABLE_STRIP}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-zink}"

echo "[KRYSTAL-WSL-GNOME-GDM] Session=${KRYSTAL_GDM_SESSION_MODE} | VITAL_MAX_HP=${KRYSTAL_VITAL_MAX_HP}"

if command -v godot4 >/dev/null 2>&1; then
  exec godot4 --rendering-driver opengl3 --path "${GODOT_PROJ}" "$@"
elif command -v godot >/dev/null 2>&1; then
  exec godot --rendering-driver opengl3 --path "${GODOT_PROJ}" "$@"
else
  echo "[KRYSTAL-WSL-GNOME-GDM] Binárka godot4 nie je nainštalovaná vo WSL; spúšťam terminálový prehliadač Krystal TUI:"
  exec python3 "${REPO_ROOT}/scripts/krystal_terminal_browser_and_projections.py" --ansi
fi
