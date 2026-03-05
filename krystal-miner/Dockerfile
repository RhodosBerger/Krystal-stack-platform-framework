FROM rust:1.75 AS builder

WORKDIR /app
COPY . .

# Inštalácia Vulkan SDK závislostí pre vulkano-shaders kompiláciu SPIR-V runtime
RUN apt-get update && apt-get install -y \
    libvulkan-dev \
    glslang-tools \
    && rm -rf /var/lib/apt/lists/*

RUN cargo build --release

FROM ubuntu:24.04
WORKDIR /app

# Inštalácia driverov dôležitých na edge host s GPU
RUN apt-get update && apt-get install -y \
    vulkan-tools \
    libvulkan1 \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/krystal-miner /app/krystal-miner

ENTRYPOINT ["/app/krystal-miner"]
