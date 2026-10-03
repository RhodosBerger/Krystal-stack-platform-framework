#!/bin/bash
# ==============================================================================
# KRYSTAL-STACK WSL2 LINUX SUBSYSTEM SETUP & OPTIMIZATION ENVIRONMENT
# ==============================================================================
# Framework: Gamesa Cortex V2 / Krystal-Stack Platform
# Target: Microsoft Windows Subsystem for Linux (WSL2 - Ubuntu 22.04 / 24.04)
# ==============================================================================

set -e

echo "======================================================================"
echo " 🌌 KRYSTAL-STACK WSL2 LINUX ENVIRONMENT INITIALIZATION"
echo "======================================================================"

# 1. Detect WSL2 Environment
if grep -qi microsoft /proc/version; then
    echo "[OK] Detected Microsoft WSL2 Kernel: $(uname -r)"
else
    echo "[WARNING] Not running inside WSL2. Proceeding in standard Linux mode."
fi

# 2. Check WSL Direct3D 12 GPU Passthrough
echo "[CHECK] Checking WSL Direct3D 12 /dev/dxg GPU passthrough..."
if [ -e /dev/dxg ]; then
    echo "[OK] /dev/dxg device node present! GPU acceleration available."
    ls -l /dev/dxg
else
    echo "[INFO] /dev/dxg not found. Running in CPU emulation mode."
fi

# 3. Check WSL Graphics Libraries
echo "[CHECK] Checking WSL graphics libraries (/usr/lib/wsl/lib)..."
if [ -d /usr/lib/wsl/lib ]; then
    echo "[OK] Found WSL driver libraries in /usr/lib/wsl/lib:"
    ls -la /usr/lib/wsl/lib/libd3d12* 2>/dev/null || true
    # Export driver path
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
fi

# 4. Install Core System Prerequisites
echo "[SETUP] Updating package indices and installing build tools..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-venv \
    libvulkan-dev \
    vulkan-tools \
    socat \
    curl \
    git

# 5. Create WSL Virtual Environment if needed
WSL_VENV="/opt/krystal_wsl_venv"
if [ ! -d "$WSL_VENV" ]; then
    echo "[SETUP] Creating dedicated WSL Python virtual environment at $WSL_VENV..."
    sudo python3 -m venv "$WSL_VENV"
    sudo "$WSL_VENV/bin/pip" install --upgrade pip
    sudo "$WSL_VENV/bin/pip" install numpy scipy
fi

# 6. Setup Cross-Boundary IPC Tunnel
# Bridges /tmp/krystal_ob.sock in WSL to Windows localhost:8080 or Windows Named Pipe
echo "[SETUP] Configuring IPC socket tunnel..."
WSL_SOCK="/tmp/krystal_wsl.sock"
rm -f "$WSL_SOCK"

echo "======================================================================"
echo " [OK] WSL2 Linux Environment Configured Successfully!"
echo " Launch the bridge with: python3 wsl_bridge_daemon.py"
echo "======================================================================"
