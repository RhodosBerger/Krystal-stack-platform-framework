#!/usr/bin/env bash
set -euo pipefail

# Krystal Stack — Inštalačný a verifikačný skript pre Ubuntu 22.04 / 24.04 LTS
# Aktualizuje oficiálne balíky firmvéru/mikrokódu (fwupd, intel-microcode) a nasadzuje adaptívny plán napájania.

MODE="${1:---check}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[Krystal Ubuntu] Kontrola prostredia a senzorov (VITAL_MAX_HP == 6)..."
python3 "${SCRIPT_DIR}/krystal_ubuntu_power_daemon.py"

if [[ "${MODE}" == "--install" ]]; then
  sudo mkdir -p /opt/krystal-stack
  sudo cp "${SCRIPT_DIR}/krystal_ubuntu_power_daemon.py" /opt/krystal-stack/krystal_ubuntu_power_daemon.py
  sudo chmod 0755 /opt/krystal-stack/krystal_ubuntu_power_daemon.py
  sudo cp "${SCRIPT_DIR}/krystal-contactless-solder.service" /etc/systemd/system/krystal-contactless-solder.service
  sudo systemctl daemon-reload
  sudo systemctl enable --now krystal-contactless-solder.service
  echo "[Krystal Ubuntu] Služba krystal-contactless-solder.service bola úspešne aktivovaná."
fi
