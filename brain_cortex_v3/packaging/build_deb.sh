#!/usr/bin/env bash
# =============================================================================
#  Brain Cortex V3 — Debian Package Builder
#  build_deb.sh
#
#  Balík: brain-cortex-v3
#  Obsah:
#    • Python runtime (advice_grid, instruction_bus, power_monitor,
#                      arenas, cortex_game, cortex_css)
#    • C++ CortexBot binárka (skompilovaná -march=tigerlake)
#    • systemd service:  brain-cortex-v3.service
#    • Manpage / help:   /usr/share/doc/brain-cortex-v3/
#    • Env konfig:       /etc/default/brain-cortex-v3
#    • Wrapper binárky:  cortex-game, cortex-bot, cortex-css, cortex-demo
#
#  Použitie:
#    bash packaging/build_deb.sh [VERSION]
#    bash packaging/build_deb.sh 3.1.0
#
#  Výstup: dist/brain-cortex-v3_3.1.0_amd64.deb
# =============================================================================

set -euo pipefail

# ── Metadáta ─────────────────────────────────────────────────────────────────
PKG_NAME="brain-cortex-v3"
VERSION="${1:-3.0.0}"
ARCH="$(dpkg --print-architecture 2>/dev/null || echo "amd64")"
MAINTAINER="Dušan Kopecký <dusan.kopecky0101@gmail.com>"

# ── Cesty ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/packaging/.build/${PKG_NAME}_${VERSION}_${ARCH}"
DIST_DIR="${ROOT_DIR}/dist"
INSTALL_PREFIX="/usr/lib/${PKG_NAME}"
BOT_BUILD_DIR="${ROOT_DIR}/cortex_bot/build"

# ── Farby ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()   { echo -e "${CYAN}[build]${NC} $*"; }
ok()    { echo -e "${GREEN}[ok]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${NC}  $*"; }
err()   { echo -e "${RED}[err]${NC}   $*" >&2; exit 1; }

# =============================================================================
echo -e "\n${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║   Brain Cortex V3 — .deb Package Builder             ║${NC}"
echo -e "${BOLD}║   Intel i5-1135G7 / Iris Xe / Ubuntu                 ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}\n"
log "Package: ${PKG_NAME}_${VERSION}_${ARCH}"
log "Source:  ${ROOT_DIR}"
log "Output:  ${DIST_DIR}/${PKG_NAME}_${VERSION}_${ARCH}.deb"
echo

# =============================================================================
# KROK 1: Skontroluj závislosti
# =============================================================================
log "Krok 1/6: Kontrola závislostí..."
for cmd in dpkg-deb python3 cmake g++; do
    if command -v "$cmd" >/dev/null 2>&1; then
        ok "$cmd found"
    else
        warn "$cmd not found — niektoré časti môžu byť vynechané"
    fi
done

# =============================================================================
# KROK 2: Kompiluj C++ CortexBot
# =============================================================================
log "Krok 2/6: Kompilacia C++ CortexBot (-march=tigerlake)..."
CORTEX_BOT_BIN=""

if command -v cmake >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then
    mkdir -p "${BOT_BUILD_DIR}"
    cmake \
        -B "${BOT_BUILD_DIR}" \
        -S "${ROOT_DIR}/cortex_bot" \
        -DWITH_OPENVINO=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -Wno-dev \
        --log-level=WARNING \
        2>&1 | grep -v "^--" || true

    cmake --build "${BOT_BUILD_DIR}" --parallel "$(nproc)" 2>&1

    if [ -f "${BOT_BUILD_DIR}/cortex_bot" ]; then
        CORTEX_BOT_BIN="${BOT_BUILD_DIR}/cortex_bot"
        ok "cortex_bot binary: $(du -sh "${CORTEX_BOT_BIN}" | cut -f1)"
    else
        warn "cortex_bot binary not found — package will be Python-only"
    fi
else
    warn "cmake/g++ not available — preskakujem C++ kompiliu"
fi

# =============================================================================
# KROK 3: Vytvor adresárovú štruktúru balíka
# =============================================================================
log "Krok 3/6: Tvorba adresárovej štruktúry .deb..."

rm -rf "${BUILD_DIR}"
mkdir -p \
    "${BUILD_DIR}/DEBIAN" \
    "${BUILD_DIR}${INSTALL_PREFIX}" \
    "${BUILD_DIR}${INSTALL_PREFIX}/cortex_css/styles" \
    "${BUILD_DIR}${INSTALL_PREFIX}/cortex_css/evaluator" \
    "${BUILD_DIR}${INSTALL_PREFIX}/cortex_css/dashboard" \
    "${BUILD_DIR}${INSTALL_PREFIX}/cortex_bot" \
    "${BUILD_DIR}/usr/bin" \
    "${BUILD_DIR}/usr/share/doc/${PKG_NAME}" \
    "${BUILD_DIR}/usr/share/man/man1" \
    "${BUILD_DIR}/etc/default" \
    "${BUILD_DIR}/lib/systemd/system" \
    "${BUILD_DIR}/var/log/${PKG_NAME}" \
    "${BUILD_DIR}/var/lib/${PKG_NAME}" \
    "${DIST_DIR}"

# =============================================================================
# KROK 4: Skopíruj Python runtime
# =============================================================================
log "Krok 4/6: Kopírujem Python runtime..."

# Jadro projektu
for f in advice_grid.py instruction_bus.py power_monitor.py \
          arenas.py cortex_game.py requirements.txt README.md; do
    if [ -f "${ROOT_DIR}/${f}" ]; then
        cp "${ROOT_DIR}/${f}" "${BUILD_DIR}${INSTALL_PREFIX}/"
        ok "  ✓ ${f}"
    fi
done

# LICENSE
[ -f "${ROOT_DIR}/../LICENSE" ] && \
    cp "${ROOT_DIR}/../LICENSE" "${BUILD_DIR}${INSTALL_PREFIX}/LICENSE" || true

# CortexCSS — Computation Style System
for f in __init__.py demo.py; do
    [ -f "${ROOT_DIR}/cortex_css/${f}" ] && \
        cp "${ROOT_DIR}/cortex_css/${f}" \
           "${BUILD_DIR}${INSTALL_PREFIX}/cortex_css/" || true
done
for f in __init__.py compute_styles.py; do
    [ -f "${ROOT_DIR}/cortex_css/styles/${f}" ] && \
        cp "${ROOT_DIR}/cortex_css/styles/${f}" \
           "${BUILD_DIR}${INSTALL_PREFIX}/cortex_css/styles/" || true
done
for f in __init__.py cascade_engine.py; do
    [ -f "${ROOT_DIR}/cortex_css/evaluator/${f}" ] && \
        cp "${ROOT_DIR}/cortex_css/evaluator/${f}" \
           "${BUILD_DIR}${INSTALL_PREFIX}/cortex_css/evaluator/" || true
done
for f in server.py index.html; do
    [ -f "${ROOT_DIR}/cortex_css/dashboard/${f}" ] && \
        cp "${ROOT_DIR}/cortex_css/dashboard/${f}" \
           "${BUILD_DIR}${INSTALL_PREFIX}/cortex_css/dashboard/" || true
done
ok "CortexCSS skopírovaný"

# C++ CortexBot binárka
if [ -n "${CORTEX_BOT_BIN}" ]; then
    cp "${CORTEX_BOT_BIN}" "${BUILD_DIR}${INSTALL_PREFIX}/cortex_bot/cortex_bot"
    chmod 0755 "${BUILD_DIR}${INSTALL_PREFIX}/cortex_bot/cortex_bot"
    ok "cortex_bot binárka skopírovaná"
fi

# Vyčisti __pycache__ a .pyc
find "${BUILD_DIR}${INSTALL_PREFIX}" \
    \( -name "__pycache__" -type d -o -name "*.pyc" \) \
    -prune -exec rm -rf {} + 2>/dev/null || true
ok "Cache súbory vyčistené"

# =============================================================================
# KROK 5: Vytvor DEBIAN control súbory + wrapper binárky + systemd
# =============================================================================
log "Krok 5/6: Generujem DEBIAN metadata, wrappery, systemd..."

# ── control ──────────────────────────────────────────────────────────────────
cat > "${BUILD_DIR}/DEBIAN/control" <<EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: science
Priority: optional
Architecture: ${ARCH}
Maintainer: ${MAINTAINER}
Depends: python3 (>= 3.10), python3-psutil, libstdc++6
Recommends: python3-pip, cmake, build-essential, libdrm-dev, vulkan-tools
Suggests: openvino-runtime, intel-gpu-tools, i7z, powertop
Description: Brain Cortex V3 — HW Inference Orchestration Platform
 Inteligentny plánovač inferencie pre Intel Tiger Lake (i5-1135G7).
 .
 Features:
  - Advice Grid: Vulkan log -> OpenVINO inštrukcia prekladac
  - Instruction Bus: OpenVINO + OpenAPI dual-path executor
  - Power Monitor: sysfs RAPL + thermal reader
  - Four Inference Arenas: ALPHA/BETA/GAMMA/DELTA game modes
  - CortexCSS: CSS-analogny computation style system
  - CortexBot (C++): OBSERVE->EVALUATE->ACT real-time control loop
  - Web Dashboard: FastAPI + WebSocket live visualization
  - Tiger Lake specific: Iris Xe 80EU, HETERO:GPU,CPU routing
 .
 Compiled with -march=tigerlake for maximum performance.
 Licensed under Apache 2.0.
EOF

# ── conffiles ─────────────────────────────────────────────────────────────────
cat > "${BUILD_DIR}/DEBIAN/conffiles" <<EOF
/etc/default/${PKG_NAME}
EOF

# ── /etc/default/brain-cortex-v3 ──────────────────────────────────────────────
cat > "${BUILD_DIR}/etc/default/${PKG_NAME}" <<'EOF'
# Brain Cortex V3 — Runtime Environment
# Upravte podľa vašej konfigurácie

# Python interpreter
BCX_PYTHON=/usr/bin/python3

# Default power mode: eco | balanced | overdrive
BCX_POWER_MODE=balanced

# Target inference latency (ms)
BCX_TARGET_LATENCY_MS=20

# CortexBot tick period (ms) — 8ms = 125Hz
BCX_TICK_MS=8

# OpenAPI endpoint (Ollama local)
BCX_OPENAPI_URL=http://localhost:11434/v1

# Web dashboard port
BCX_DASHBOARD_PORT=7860

# Log directory
BCX_LOG_DIR=/var/log/brain-cortex-v3
EOF

# ── Wrapper: cortex-game ───────────────────────────────────────────────────────
cat > "${BUILD_DIR}/usr/bin/cortex-game" <<EOF
#!/usr/bin/env sh
# Brain Cortex V3 — Game Interface
set -eu
[ -f /etc/default/${PKG_NAME} ] && . /etc/default/${PKG_NAME}
export BCX_LOG_DIR="\${BCX_LOG_DIR:-/var/log/${PKG_NAME}}"
cd /usr/lib/${PKG_NAME}
exec "\${BCX_PYTHON:-python3}" cortex_game.py "\$@"
EOF

# ── Wrapper: cortex-bot ────────────────────────────────────────────────────────
cat > "${BUILD_DIR}/usr/bin/cortex-bot" <<'EOF'
#!/usr/bin/env sh
# Brain Cortex V3 — C++ CortexBot
set -eu
[ -f /etc/default/brain-cortex-v3 ] && . /etc/default/brain-cortex-v3
BOT_BIN=/usr/lib/brain-cortex-v3/cortex_bot/cortex_bot
if [ ! -x "$BOT_BIN" ]; then
    echo "[cortex-bot] Binary not found. Run: cortex-game --arena ALL" >&2
    exit 1
fi
exec "$BOT_BIN" --simulate "${@}"
EOF

# ── Wrapper: cortex-css ────────────────────────────────────────────────────────
cat > "${BUILD_DIR}/usr/bin/cortex-css" <<EOF
#!/usr/bin/env sh
# Brain Cortex V3 — CortexCSS Demo
set -eu
[ -f /etc/default/${PKG_NAME} ] && . /etc/default/${PKG_NAME}
cd /usr/lib/${PKG_NAME}/cortex_css
exec "\${BCX_PYTHON:-python3}" demo.py "\$@"
EOF

# ── Wrapper: cortex-dashboard ──────────────────────────────────────────────────
cat > "${BUILD_DIR}/usr/bin/cortex-dashboard" <<EOF
#!/usr/bin/env sh
# Brain Cortex V3 — Web Dashboard
set -eu
[ -f /etc/default/${PKG_NAME} ] && . /etc/default/${PKG_NAME}
PORT="\${BCX_DASHBOARD_PORT:-7860}"
echo "CortexCSS Dashboard starting on http://localhost:\${PORT}"
cd /usr/lib/${PKG_NAME}/cortex_css
exec "\${BCX_PYTHON:-python3}" dashboard/server.py "\$@"
EOF

# ── Wrapper: cortex-demo ───────────────────────────────────────────────────────
cat > "${BUILD_DIR}/usr/bin/cortex-demo" <<EOF
#!/usr/bin/env sh
# Brain Cortex V3 — Live Advice Grid Demo
set -eu
[ -f /etc/default/${PKG_NAME} ] && . /etc/default/${PKG_NAME}
cd /usr/lib/${PKG_NAME}
exec "\${BCX_PYTHON:-python3}" cortex_game.py --demo --frames "\${1:-50}"
EOF

chmod 0755 \
    "${BUILD_DIR}/usr/bin/cortex-game" \
    "${BUILD_DIR}/usr/bin/cortex-bot" \
    "${BUILD_DIR}/usr/bin/cortex-css" \
    "${BUILD_DIR}/usr/bin/cortex-dashboard" \
    "${BUILD_DIR}/usr/bin/cortex-demo"

ok "Wrapper binárky vytvorené"

# ── systemd service ───────────────────────────────────────────────────────────
cat > "${BUILD_DIR}/lib/systemd/system/${PKG_NAME}.service" <<EOF
[Unit]
Description=Brain Cortex V3 — HW Inference Orchestration Bot
Documentation=https://github.com/Dev-contitional/brain_cortex_v3
After=network.target
Wants=network.target

[Service]
Type=simple
User=cortex-bot
Group=cortex-bot
EnvironmentFile=-/etc/default/${PKG_NAME}
WorkingDirectory=/var/lib/${PKG_NAME}
ExecStart=/usr/bin/cortex-dashboard
ExecReload=/bin/kill -HUP \$MAINPID
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${PKG_NAME}
Restart=on-failure
RestartSec=5
TimeoutStopSec=10

# Bezpečnostné parametre
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/var/log/${PKG_NAME} /var/lib/${PKG_NAME}

[Install]
WantedBy=multi-user.target
EOF
ok "systemd service vytvorený"

# ── man page ──────────────────────────────────────────────────────────────────
cat > "${BUILD_DIR}/usr/share/man/man1/cortex-game.1" <<'EOF'
.TH CORTEX-GAME 1 "2026-03-15" "Brain Cortex V3" "User Commands"
.SH NAME
cortex-game \- Brain Cortex V3 Hardware Inference Game
.SH SYNOPSIS
.B cortex-game
[\fB--arena\fR \fIARENAME\fR] [\fB--power\fR \fIMODE\fR] [\fB--demo\fR]
.SH DESCRIPTION
Interaktivna hra pre Intel i5-1135G7 (Tiger Lake). Simuluje a riadi
inferenciu cez OpenVINO na CPU/GPU/HETERO zariadeniach.
.SH ARENAS
.TP
.B ALPHA
CPU vs NPU klasifikacny zavod (Intel GNA 3720 fallback na HETERO).
.TP
.B BETA
Battery endurance: udrz >=30fps na minimalnom watt odbere.
.TP
.B GAMMA
Advice Grid rychlost: kolko Vulkan snimok/sekundu vie prepisat.
.TP
.B DELTA
Tepelne prezitie: neudrz CPU pod 85 degC pocas rastucich scenárov.
.TP
.B ALL
Spusti vsetky areny v poradí.
.SH OPTIONS
.TP
.B --power eco|balanced|overdrive
Rezim napajacej sprarvy pre BETA arenu.
.TP
.B --demo [--frames N]
Zivi stream Vulkan -> Advice Grid -> Instruction.
.SH COMMANDS
.TP
.B cortex-bot
Real-time C++ OBSERVE->EVALUATE->ACT smycka.
.TP
.B cortex-css
CortexCSS Computation Style System demo.
.TP
.B cortex-dashboard
Web dasboard na http://localhost:7860.
.SH FILES
.I /etc/default/brain-cortex-v3
Konfiguracny subor s parametrami.
.I /var/log/brain-cortex-v3/
Logy.
.SH SEE ALSO
openvino-runtime(1), i7z(1)
.SH AUTHOR
Dusan Kopecky <dusan.kopecky0101@gmail.com>
EOF

gzip -9 "${BUILD_DIR}/usr/share/man/man1/cortex-game.1" 2>/dev/null || true

# ── doc ───────────────────────────────────────────────────────────────────────
cp "${ROOT_DIR}/README.md" "${BUILD_DIR}/usr/share/doc/${PKG_NAME}/"
cat > "${BUILD_DIR}/usr/share/doc/${PKG_NAME}/changelog.gz.stub" <<'EOF'
brain-cortex-v3 (3.0.0) unstable; urgency=low
  * Initial release of Brain Cortex V3
  * Advice Grid: Vulkan log -> OpenVINO instruction translation
  * CortexBot C++: sysfs real-time HW control loop
  * CortexCSS: Computation Style System with parallel evaluator
  * Four game arenas: ALPHA/BETA/GAMMA/DELTA
  * Web dashboard: FastAPI + WebSocket
 -- Dusan Kopecky <dusan.kopecky0101@gmail.com>  Sun, 15 Mar 2026 07:50:00 +0100
EOF

# ── DEBIAN/postinst ───────────────────────────────────────────────────────────
cat > "${BUILD_DIR}/DEBIAN/postinst" <<POSTINST
#!/usr/bin/env sh
set -eu
PKG="brain-cortex-v3"
BOT_USER="cortex-bot"
LOG_DIR="/var/log/\${PKG}"
LIB_DIR="/var/lib/\${PKG}"
INSTALL_DIR="/usr/lib/\${PKG}"

echo "[postinst] Konfigurujem \${PKG}..."

# Vytvor system usera pre bezpecnye spustanie
if ! getent group "\${BOT_USER}" >/dev/null 2>&1; then
    addgroup --system "\${BOT_USER}" >/dev/null 2>&1 || true
fi
if ! getent passwd "\${BOT_USER}" >/dev/null 2>&1; then
    adduser --system --ingroup "\${BOT_USER}" \
        --home "\${LIB_DIR}" --no-create-home \
        --shell /usr/sbin/nologin \
        --gecos "Brain Cortex V3 Service" \
        "\${BOT_USER}" >/dev/null 2>&1 || true
fi

# Adresare pre logy a runtime state
mkdir -p "\${LOG_DIR}" "\${LIB_DIR}"
chown -R "\${BOT_USER}:\${BOT_USER}" "\${LOG_DIR}" "\${LIB_DIR}" 2>/dev/null || true
chmod 0775 "\${LOG_DIR}" "\${LIB_DIR}" 2>/dev/null || true

# Nastav capabilities pre cortex-bot (sysfs pristup bez root)
BOT_BIN="\${INSTALL_DIR}/cortex_bot/cortex_bot"
if [ -x "\${BOT_BIN}" ] && command -v setcap >/dev/null 2>&1; then
    setcap 'cap_sys_admin+ep' "\${BOT_BIN}" 2>/dev/null || \
        echo "[postinst] Tip: Spusti 'sudo setcap cap_sys_admin+ep \${BOT_BIN}'"
fi

# Nainštaluj Python závislosti do systémovej virty / globálne
INSTALL_DIR_PY="/usr/lib/${PKG_NAME}"
if command -v pip3 >/dev/null 2>&1; then
    pip3 install --quiet --no-warn-script-location \
        rich psutil requests fastapi uvicorn 2>/dev/null || \
        echo "[postinst] Tip: pip3 install rich psutil requests fastapi uvicorn"
fi

# Reload systemd a enable service
if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload >/dev/null 2>&1 || true
    echo ""
    echo "==================================================================="
    echo "  Brain Cortex V3 nainštalovaný!"
    echo ""
    echo "  Rýchly štart:"
    echo "    cortex-game              # interaktívne menu"
    echo "    cortex-game --arena ALL  # plný gauntlet"
    echo "    cortex-demo              # živý stream"
    echo "    cortex-css               # computation style race"
    echo "    cortex-dashboard         # web UI na :7860"
    echo "    cortex-bot --simulate    # C++ control loop"
    echo ""
    echo "  Ako service:"
    echo "    sudo systemctl enable --now brain-cortex-v3"
    echo "==================================================================="
fi
exit 0
POSTINST
chmod 0755 "${BUILD_DIR}/DEBIAN/postinst"

# ── DEBIAN/prerm ──────────────────────────────────────────────────────────────
cat > "${BUILD_DIR}/DEBIAN/prerm" <<'EOF'
#!/usr/bin/env sh
set -eu
if command -v systemctl >/dev/null 2>&1; then
    systemctl stop brain-cortex-v3.service >/dev/null 2>&1 || true
    systemctl disable brain-cortex-v3.service >/dev/null 2>&1 || true
fi
exit 0
EOF
chmod 0755 "${BUILD_DIR}/DEBIAN/prerm"

# ── DEBIAN/postrm ─────────────────────────────────────────────────────────────
cat > "${BUILD_DIR}/DEBIAN/postrm" <<'EOF'
#!/usr/bin/env sh
set -eu
if command -v systemctl >/dev/null 2>&1; then
    systemctl daemon-reload >/dev/null 2>&1 || true
fi
if [ "$1" = "purge" ]; then
    rm -rf /var/log/brain-cortex-v3 /var/lib/brain-cortex-v3 2>/dev/null || true
fi
exit 0
EOF
chmod 0755 "${BUILD_DIR}/DEBIAN/postrm"

ok "DEBIAN skripty vygenerované"

# =============================================================================
# KROK 6: Zostav .deb
# =============================================================================
log "Krok 6/6: Zostavujem .deb balík..."

OUT_DEB="${DIST_DIR}/${PKG_NAME}_${VERSION}_${ARCH}.deb"
dpkg-deb --root-owner-group --build "${BUILD_DIR}" "${OUT_DEB}"

DEB_SIZE="$(du -sh "${OUT_DEB}" | cut -f1)"
ok "Hotovo!"

echo
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║  .deb úspešne zostrojený!                                ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo -e "  Súbor:   ${BOLD}${OUT_DEB}${NC}"
echo -e "  Veľkosť: ${BOLD}${DEB_SIZE}${NC}"
echo
echo -e "  ${BOLD}Inštalácia:${NC}"
echo -e "    sudo dpkg -i ${OUT_DEB}"
echo -e "    # alebo"
echo -e "    sudo apt install --fix-broken ${OUT_DEB}"
echo
echo -e "  ${BOLD}Overenie:${NC}"
echo -e "    dpkg -I ${OUT_DEB}            # info"
echo -e "    dpkg -c ${OUT_DEB}            # obsah"
echo -e "    dpkg --list | grep cortex     # po inštalácii"
echo
echo -e "  ${BOLD}Prvé spustenie:${NC}"
echo -e "    cortex-game                   # interaktívne menu"
echo -e "    cortex-game --arena ALL       # plný gauntlet"
echo -e "    cortex-demo                   # živý Vulkan stream"
echo -e "    cortex-dashboard              # web UI :7860"
echo -e "    cortex-bot --simulate         # C++ real-time loop"
echo
