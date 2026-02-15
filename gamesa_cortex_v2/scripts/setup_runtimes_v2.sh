#!/bin/bash
set -e

# Gamesa Cortex V2 - Runtime Setup Script
# Installs: System Deps (Apt), Rust (Cargo), Python (Venv + Pip)

echo "============================================================"
echo "    Gamesa Cortex V2: Runtime Installer"
echo "============================================================"

# 1. System Dependencies (Requires Sudo)
echo "[1/4] Installing System Dependencies (Apt)..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    python3-dev \
    python3-venv \
    vulkan-tools \
    libvulkan-dev \
    ocl-icd-opencl-dev \
    opencl-headers

echo "System dependencies installed."

# 2. Rust Toolchain
echo "[2/4] Installing Rust Toolchain..."
if ! command -v cargo &> /dev/null; then
    echo "Cargo not found. Installing Rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "Rust is already installed."
fi

# Ensure cargo is in path for this session
export PATH="$HOME/.cargo/bin:$PATH"
rustc --version
cargo --version

# 3. Python Virtual Environment
echo "[3/4] Setting up Python Environment..."
VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Activate Venv
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel

# 4. Install Python Libraries
echo "[4/4] Installing Python Libraries..."

# Install core requirements if file exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found in current directory."
fi

# Install AI & Graphics specific libs
echo "Installing compute libraries..."
pip install numpy scipy wgpu-py pyopencl

# Optional: PyTorch/OpenVINO (Checking if user wants them heavy libs)
# For now, we install basic CPU versions to keep it light, user can upgrade later.
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "============================================================"
echo "    Setup Complete!"
echo "============================================================"
echo "To activate the environment, run: source venv/bin/activate"
