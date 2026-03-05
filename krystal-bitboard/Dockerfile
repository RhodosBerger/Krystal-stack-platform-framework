FROM rust:1.75 AS builder

WORKDIR /app
COPY . .

# Vulkan build deps if shaders are to be compiled, plus dependencies
RUN apt-get update && apt-get install -y \
    libvulkan-dev glslang-tools \
    && rm -rf /var/lib/apt/lists/*

# V reálnej farme by tu bol --release flag, takto sa len overí stav
RUN cargo build --release

FROM ubuntu:24.04
WORKDIR /app

# Inštalácia driverov dôležitých na edge host s GPU, nvml, a hwmon (z kernelu)
RUN apt-get update && apt-get install -y \
    vulkan-tools \
    libvulkan1 \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/krystal-bitboard /app/krystal-bitboard

# vGPU Slicing flag na Nvidia MPS (Hybrid Mining a AI), limit 40% pre mining
ENV CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=40
ENV RUST_LOG=info

# ENTRYPOINT pre orchestrátor v 10-100 GPU farme
ENTRYPOINT ["/app/krystal-bitboard"]
