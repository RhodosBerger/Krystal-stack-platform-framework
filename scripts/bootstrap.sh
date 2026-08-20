#!/bin/bash
# GAMESA/KrystalStack Bootstrap Script
# One-click setup for development environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[BOOTSTRAP]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
    else
        error "Unsupported OS"
    fi
    log "Detected: $OS $VERSION"
}

# Install system dependencies
install_system_deps() {
    log "Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y \
        build-essential cmake ninja-build pkg-config \
        git curl wget \
        python3 python3-pip python3-venv \
        libssl-dev libffi-dev \
        clang llvm \
        libudev-dev libdrm-dev \
        mesa-common-dev libgl1-mesa-dev \
        vainfo intel-gpu-tools
}

# Install Intel GPU stack
install_intel_stack() {
    log "Installing Intel GPU/OpenVINO stack..."

    # Intel compute runtime
    sudo apt-get install -y \
        intel-opencl-icd intel-level-zero-gpu level-zero \
        intel-media-va-driver-non-free libmfx1 libmfxgen1 || true

    # OpenVINO
    if ! python3 -c "import openvino" 2>/dev/null; then
        pip3 install openvino openvino-dev
    fi
}

# Install NVIDIA stack (if detected)
install_nvidia_stack() {
    if command -v nvidia-smi &>/dev/null; then
        log "NVIDIA GPU detected, installing stack..."
        pip3 install pynvml
        # TensorRT requires manual install from NVIDIA
        warn "TensorRT requires manual installation from NVIDIA Developer"
    fi
}

# Install AMD stack (if detected)
install_amd_stack() {
    if [ -d "/sys/class/drm/card0/device" ]; then
        vendor=$(cat /sys/class/drm/card0/device/vendor 2>/dev/null || echo "")
        if [ "$vendor" = "0x1002" ]; then
            log "AMD GPU detected, installing ROCm tools..."
            sudo apt-get install -y rocm-smi-lib || warn "ROCm not available in repos"
        fi
    fi
}

# Install Rust toolchain
install_rust() {
    if ! command -v rustup &>/dev/null; then
        log "Installing Rust toolchain..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
    rustup default stable
    rustup component add clippy rustfmt
}

# Setup Python virtual environment
setup_python_venv() {
    log "Setting up Python virtual environment..."
    cd "$ROOT_DIR"

    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi

    source venv/bin/activate
    pip install --upgrade pip wheel
    pip install -r requirements.txt
}

# Build C core runtime
build_c_core() {
    log "Building C core runtime..."
    cd "$ROOT_DIR/src/c"

    if [ ! -d "build" ]; then
        mkdir build
    fi

    cd build
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release
    ninja

    log "C core built successfully"
}

# Build Rust bot
build_rust_bot() {
    log "Building Rust bot..."
    cd "$ROOT_DIR/src/rust-bot"

    cargo build --release
    cargo clippy -- -D warnings || warn "Clippy warnings present"

    log "Rust bot built successfully"
}

# Build Mesa-GAMESA driver (optional)
build_mesa_driver() {
    if [ "$1" = "--with-mesa" ]; then
        log "Building Mesa-GAMESA driver..."
        cd "$ROOT_DIR/mesa-gamesa-drivers"

        if [ -f "Makefile" ]; then
            make
            sudo make install
            sudo make kernel-install || warn "Kernel module install requires reboot"
        else
            warn "Mesa driver Makefile not found, skipping"
        fi
    fi
}

# Verify installation
verify_install() {
    log "Verifying installation..."

    cd "$ROOT_DIR"
    source venv/bin/activate

    python3 scripts/verify_stack.py || warn "Some checks failed"

    # Test imports
    python3 -c "from src.python import create_gpu_optimizer; print('Python imports OK')"

    # Test Rust
    cd src/rust-bot && cargo test --release || warn "Rust tests failed"

    log "Verification complete"
}

# Main
main() {
    log "GAMESA/KrystalStack Bootstrap"
    log "=============================="

    detect_os

    case "$1" in
        --deps-only)
            install_system_deps
            ;;
        --python-only)
            setup_python_venv
            ;;
        --rust-only)
            install_rust
            build_rust_bot
            ;;
        --verify)
            verify_install
            ;;
        *)
            install_system_deps
            install_intel_stack
            install_nvidia_stack
            install_amd_stack
            install_rust
            setup_python_venv
            build_c_core
            build_rust_bot
            build_mesa_driver "$1"
            verify_install
            ;;
    esac

    log "Bootstrap complete!"
}

main "$@"
