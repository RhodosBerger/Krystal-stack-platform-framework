#!/usr/bin/env bash
# Krystal Stack v2.0 — Unified Installer: Kisak-Mesa Gallium3D + OpenVINO GPU/Rusticl Bridge + #![no_std] Rust Layer
# Respects KSPP P1 (Zero-Touch Interposition: upstream Kisak & OpenVINO binaries remain unmodified).
set -euo pipefail

echo "[KRYSTAL-OV-GALLIUM] Overujem ústavný invariant VITAL_MAX_HP == 6..."
export KRYSTAL_VITAL_MAX_HP=6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if command -v add-apt-repository >/dev/null 2>&1; then
  echo "[KRYSTAL-OV-GALLIUM] Konfigurujem ppa:kisak/kisak-mesa (Iris, ANV, Zink, Rusticl OpenCL 3.0, DMA-BUF)..."
  sudo add-apt-repository -y ppa:kisak/kisak-mesa
  sudo apt-get update
  sudo apt-get install -y \
    mesa-vulkan-drivers \
    mesa-opencl-icd \
    ocl-icd-libopencl1 \
    libgl1-mesa-dri \
    intel-opencl-icd \
    vulkan-tools \
    glmark2 \
    clinfo
fi

echo "[KRYSTAL-OV-GALLIUM] Kompilujem natívne #![no_std] Rust jadrá (libkrystal_sovereign_firmware_rs.so)..."
python3 "${REPO_ROOT}/scripts/build_concurrent_rust_modules.py"

sudo mkdir -p /etc/vulkan/implicit_layer.d /etc/krystal-gallium /etc/OpenCL/vendors /var/cache/krystal-openvino-gallium
sudo chmod 1777 /var/cache/krystal-openvino-gallium
sudo cp "${SCRIPT_DIR}/VK_LAYER_KRYSTAL_SOVEREIGN_GALLIUM.json" /etc/vulkan/implicit_layer.d/
sudo cp "${SCRIPT_DIR}/krystal_kisak_gallium_env.conf" /etc/krystal-gallium/

if [[ -f /usr/lib/x86_64-linux-gnu/libRusticlOpenCL.so.1 ]]; then
  echo "/usr/lib/x86_64-linux-gnu/libRusticlOpenCL.so.1" | sudo tee /etc/OpenCL/vendors/krystal-rusticl.icd >/dev/null
  echo "[KRYSTAL-OV-GALLIUM] Gallium Rusticl OpenCL 3.0 ICD prepojené pre OpenVINO GPU runtime."
fi

echo "[KRYSTAL-OV-GALLIUM] Spúšťam verifikáciu OpenVINO <-> Gallium bezkópiového mosta..."
PYTHONPATH="${REPO_ROOT}" python3 -O -c "from krystal_kernel.openvino_gallium_driver_bridge import get_global_openvino_gallium_bridge; r = get_global_openvino_gallium_bridge().execute_full_openvino_gallium_suite(); print('[OK] OpenVINO-Gallium parity:', r['neural_gallium_frame_synthesizer']['parity_verified'])"
