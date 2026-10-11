# GAMESA/KrystalStack Base Image
# Ubuntu 22.04 with Intel/OpenVINO stack

FROM ubuntu:22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build pkg-config \
    git curl wget \
    python3 python3-pip python3-venv \
    libssl-dev libffi-dev \
    clang llvm \
    libudev-dev libdrm-dev \
    mesa-common-dev libgl1-mesa-dev \
    vainfo intel-gpu-tools \
    && rm -rf /var/lib/apt/lists/*

# Intel GPU stack
RUN apt-get update && apt-get install -y \
    intel-opencl-icd intel-level-zero-gpu level-zero \
    intel-media-va-driver-non-free libmfx1 libmfxgen1 \
    || true && rm -rf /var/lib/apt/lists/*

# Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default stable && rustup component add clippy rustfmt

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY scripts/ scripts/
COPY Makefile .

# Build Rust bot
RUN cd src/rust-bot && cargo build --release

# Verify imports
RUN python3 -c "from src.python import create_gpu_optimizer"

ENTRYPOINT ["/bin/bash"]

# ============================================================
# GPU-enabled variant
# ============================================================
FROM base AS gpu

# Additional GPU tools
RUN apt-get update && apt-get install -y \
    vulkan-tools mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# OpenVINO with GPU support
RUN pip3 install --no-cache-dir openvino openvino-dev

# Entry for GPU workloads
CMD ["python3", "scripts/verify_stack.py"]
