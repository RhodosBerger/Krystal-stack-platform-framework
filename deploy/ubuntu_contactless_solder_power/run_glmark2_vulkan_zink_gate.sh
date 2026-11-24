#!/usr/bin/env bash
set -euo pipefail

# Krystal Stack v2.0 — Dual Verification Gate:
#   1) OpenVINO <-> Kisak-Gallium Zero-Copy Tensor & Neural Frame Synthesis Parity Gate
#   2) Ubuntu glmark2 Vulkan/Zink/ANV Non-Regression Gate
# Ak klesne výkon glmark2 alebo zlyhá parita OpenVINO-Gallium, zmena sa okamžite zamietne (Delta_S = 0).

BASELINE_MIN_SCORE="${1:-2500}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_BIN="${PYTHON_BIN:-python3}"

echo "[Krystal Dual Gate 1/2] Overujem bezkópiové prepojenie OpenVINO <-> Kisak-Gallium (VITAL_MAX_HP == 6)..."
PYTHONPATH="${REPO_ROOT}" "${PY_BIN}" -O -c "
from krystal_kernel.openvino_gallium_driver_bridge import get_global_openvino_gallium_bridge
rep = get_global_openvino_gallium_bridge().execute_full_openvino_gallium_suite()
if not rep['neural_gallium_frame_synthesizer']['parity_verified']:
    raise SystemExit('[ROLLBACK] Porušená parita CPU vs GPU v OpenVINO-Gallium -> Delta_S = 0!')
print('[OK] OpenVINO-Gallium Zero-Copy & NIR prediktor overený:', rep['active_gallium_openvino_channel']['gallium_driver_plane'])
"

echo "[Krystal Dual Gate 2/2] Overujem prítomnosť glmark2 a Mesa Zink/Gallium v Ubuntu..."
if ! command -v glmark2 >/dev/null 2>&1; then
  echo "[INFO] glmark2 nie je nainštalovaný na tomto uzle; OpenVINO-Gallium brána prešla, Vulkan vrstva pripravená."
  exit 0
fi

export VK_ADD_LAYER_PATH="${REPO_ROOT}/deploy/kisak_mesa_gallium_port:${SCRIPT_DIR}"
export VK_INSTANCE_LAYERS="VK_LAYER_KRYSTAL_SOVEREIGN_GALLIUM:VK_LAYER_KRYSTAL_CONTACTLESS_SOLDER"
export MESA_LOADER_DRIVER_OVERRIDE="zink"
export MESA_GLTHREAD="true"
export RUSTICL_ENABLE="iris,llvmpipe"

OUT_LOG="$(mktemp)"
glmark2 --off-screen -b build:use-vbo=true -b shading:shading=phong | tee "${OUT_LOG}"
SCORE="$(grep -Eo 'glmark2 Score: [0-9]+' "${OUT_LOG}" | awk '{print $3}' || echo 0)"
rm -f "${OUT_LOG}"

echo "[Krystal Dual Gate] Namerané skóre cez Kisak-Gallium/Zink + OpenVINO vrstvu: ${SCORE} (minimum: ${BASELINE_MIN_SCORE})"
if [[ "${SCORE}" -lt "${BASELINE_MIN_SCORE}" ]]; then
  echo "[ROLLBACK] Detegovaná regresia výkonu (${SCORE} < ${BASELINE_MIN_SCORE}) -> Delta_S = 0!"
  exit 1
fi
echo "[APPROVED] Obidve brány (OpenVINO-Gallium aj glmark2 Vulkan/Zink) potvrdené!"
