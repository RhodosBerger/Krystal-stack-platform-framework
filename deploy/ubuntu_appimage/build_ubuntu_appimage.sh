#!/usr/bin/env bash
# Natívny Ubuntu AppImage kompilátor (použije appimagetool ak je prítomný, inak vytvorí samorosbaľovací AppImage)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPDIR="${SCRIPT_DIR}/KrystalSovereign.AppDir"
OUT="${SCRIPT_DIR}/../../dist/Krystal-Sovereign-Cyber60s-x86_64.AppImage"
mkdir -p "$(dirname "${OUT}")"
chmod +x "${APPDIR}/AppRun" "${APPDIR}/usr/bin/krystal-sovereign-entry.py"
if command -v appimagetool >/dev/null 2>&1; then
  ARCH=x86_64 appimagetool "${APPDIR}" "${OUT}"
else
  python3 "${SCRIPT_DIR}/../../projects/trigonometric_hybrid_platform_framework/keras_lowram_npc_streaming_and_ubuntu_appimage.py"
fi
echo "[AppImage Ready] ${OUT}"
