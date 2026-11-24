#!/usr/bin/env bash
# Krystal Stack — Autonómny Zero-Touch VPS Inštalačný a Adaptačný Skript
# Stačí naklonovať repozitár na čisté Ubuntu/Debian VPS a spustiť tento skript:
#   git clone https://github.com/RhodosBerger/Krystal-stack-platform-framework.git && cd Krystal-stack-platform-framework && bash deploy/vps_cloud_bootstrap/bootstrap_krystal_vps.sh
set -euo pipefail

export KRYSTAL_VITAL_MAX_HP=6
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[KRYSTAL-VPS 1/5] Inštalujem systémové balíky pre Python, C11, Rust a Vulkan/Mesa..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y python3 python3-venv python3-pip build-essential gcc clang curl git
fi

echo "[KRYSTAL-VPS 2/5] Vytváram izolované virtuálne prostredie a sťahujem knižnice frameworku..."
cd "${REPO_ROOT}"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy psutil pyyaml openvino pytest

echo "[KRYSTAL-VPS 3/5] Kompilujem C11 ISA diagramy a natívne #![no_std] Rust moduly (ak je prítomný cargo)..."
mkdir -p artifacts/bin
gcc -std=c11 -O3 -Wall -Wextra -Werror -Iinclude src/c/krystal_isa_vulkan_webgl_diagrams.c -o artifacts/bin/krystal_isa_diagrams
./artifacts/bin/krystal_isa_diagrams
if command -v cargo >/dev/null 2>&1; then
  python3 scripts/build_concurrent_rust_modules.py
fi

echo "[KRYSTAL-VPS 4/5] Spúšťam OpenVINO skenovanie hardvéru VPS, priraďujem adaptívny preset a vytváram počiatočný SHA-256 backup..."
PYTHONPATH="${REPO_ROOT}" python3 -O -c "
from projects.trigonometric_hybrid_platform_framework.last_tribe_cyber60s_vps_cloud_and_blender_bench import OpenVinoVpsHardwareScannerAndBackup
mgr = OpenVinoVpsHardwareScannerAndBackup()
scan = mgr.scan_hardware_and_route_preset()
bkp = mgr.create_signed_backup(scan)
restored = mgr.restore_from_backup(bkp['backup_path'])
print('[OK] Detegovaný profil VPS:', scan['selected_hardware_preset'], '| Obnova zo zálohy overená:', restored['restored_ok'])
"

echo "[KRYSTAL-VPS 5/5] VPS uzol je pripravený. Spustenie cloudového servera: KRYSTAL_HOST=0.0.0.0 KRYSTAL_PORT=8090 python3 -O -m krystal_kernel.sovereign_openvino_oneapi_app"
