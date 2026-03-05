# Krystal Bitboard: The Dual-Earn Engine (Mining + Edge Render) 🌌⛏️

**Attention GPU Miners! Stop letting market winter drain your electricity budget.**  
What if your rig could mine Bitcoin *AND* earn premium rendering payouts simultaneously, adapting to whatever currently pays more? 

**Krystal Bitboard** is an experimental module within the Krystal-stack-platform-framework that introduces **Hybrid GPU Dynamic Slicing**. By adaptively partitioning your VRAM and utilizing Vulkan Compute, we combine **Bitcoin Mining (SHA-256)** and **3D Node Rendering (Blender Cycles/Octane)** into one synchronized operation.

### 💰 Why Choose One When You Can Dual-Earn?
If you mine Bitcoin but suddenly receive a high-paying rendering job, the standard approach would interrupt your mining software, incur downtime latency, wait for rendering to finish, and try to restart mining. 
With **Krystal Bitboard**:
1. **Zero-Latency Network Bidding**: We reduce render job execution latencies because VRAM zones dynamically expand and retract instantly via the **Economic Governor**.
2. **Adaptive Resource Allocation**: When rendering commands premium rates (>1.50€/h), the grid allocates 85% of your GPU to rendering, leaving 10% running the Stratum protocol so you stay connected to your pool with zero reconnection penalties.
3. **Double Earnings**: During quiet periods (nighttime), 80-100% of GPU shifts back to mining when electricity is cheap, creating a relentless dual-income strategy.

## 🚀 Key Features for Miners

*   **Dynamic VRAM Memory Grid**: Your GPU memory (e.g., 8GB on an RTX 3060) is partitioned into specific, hard-locked zones (Mining: 30%, Render: 50%, AI: 15%, Buffer: 5%). Say goodbye to Out-Of-Memory rendering crashes that kill your background miners.
*   **Vulkan Native Mining Pipeline**: Completely bypassed CUDA limitations; running highly optimized SHA-256 midstate compute shaders over the Vulkan API for cross-compatibility (NVIDIA and AMD). Zero-copy mapping!
*   **Shadow Council Overclocking**: Autonomous overclock agents that apply variable clocks. 
    *   *Creator Agent* suggests aggressive clocks (+150 Mem, +50 Core).
    *   *Auditor Agent* approves based on a 5-minute linear trajectory temperature prediction (Rolling Buffer).
    *   *Accountant Agent* shuts down mining entirely if €/kWh costs vs BTC price isn't profitable.
*   **Day/Night Cycle Mining**: The Economic Governor shifts 100% of the grid to mining at night (cheaper electricity tariffs) and returns 80% to AI/Render tasks during the day.
*   **Docker Ready Farm Nodes**: Deploys via `k3s-deploy.yaml` utilizing `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` for sub-millisecond hardware orchestration limits.

---

## 🧪 Community Test Program: What We Need You To Test

To bring this hybrid framework to mainstream stability, we need the GPU mining community to rigorously test specific hardware and telemetry components. We rely on your rigs!

### 1. Hardware Telemetry & NVML/ROCm Bindings
*   **Component**: `krystal-ob/src/telemetry.rs`
*   **Goal**: Ensure precise capturing of metrics across different architectures.
*   **Test**: Run the telemetry daemon on your mining rigs (mix of AMD RX 6000/7000 and NVIDIA RTX 30/40 series). Verify the exported `gpu_temp`, `vram_temp`, `fan_rpm`, and `power_percent`. Report any kernel panics or missing sensor bindings.

### 2. Vulkan Memory Grid Isolation (VRAM Slicing)
*   **Component**: `krystal-bitboard/src/grid.rs`
*   **Goal**: Ensure zero-stutter inference when mining is running simultaneously.
*   **Test**: We need community members to load the default 8GB/12GB/24GB memory limit variants. Monitor if the Vulkan `lock_memory()` strict zones appropriately throttle the mining loop when large chunk AI/Render jobs are scheduled. Verify via `get_grid_json()` Glass Brain API.

### 3. Economic Governor Volatility (Dual-Earning Tests)
*   **Component**: `krystal-bitboard/src/governor.rs`
*   **Goal**: Test rapid profitability context switching between Mining and Rendering.
*   **Test**: Manually tweak the `mining_profit_eur_per_hr` vs `render_profit_eur_per_hr` local variables. Observe the latency it takes for the Memory Grid to dynamically shift its limits. If rendering spikes to > 2.00 €/hr, the system must drop the Mining VRAM zone (resizing to near 0MB) and allocate it to the Render zone (85%) seamlessly without dropping Stratum validation packets.

### 4. Shadow Council Linear Regression (Overclock Safeties)
*   **Component**: `krystal-bitboard/src/ob.rs` / `TemperaturePredictor`
*   **Goal**: Verify cooling predictions don't fry GPU silicon.
*   **Test**: Manually stop GPU fan curves or run stress tests to force a thermal ramp. Confirm that the buffer notices a 3°C jump in 10 seconds and instantly executes `reset_overclock()`. 

---

## 🛠️ Requirements & Setup

You will need a Linux environment (Ubuntu 22.04 / 24.04 recommended) to compile and run the crates.

1.  **Vulkan SDK & Dependencies**:
    ```bash
    sudo apt-get update
    sudo apt-get install -y libvulkan-dev glslang-tools vulkan-tools libvulkan1 mesa-vulkan-drivers
    ```
2.  **Clone & Build**:
    The system is part of a Rust Workspace.
    ```bash
    git clone https://github.com/RhodosBerger/Krystal-stack-platform-framework
    cd Krystal-stack-platform-framework
    cargo build -p krystal-bitboard --release
    ```
3.  **Docker Farm Deployment**:
    For large farms, deploy scaling via our supplied configs utilizing Multi-Process Service (MPS).
    ```bash
    docker build -t krystal-bitboard ./krystal-bitboard
    # Refer to krystal-miner/k3s-deploy.yaml
    ```

## 🤝 Contributing

This is a bleeding-edge prototype. Your feedback, stack traces, and pull requests regarding AMD ROCm driver compatibility and Vulkano API implementations are critical. Let's redefine GPU farm economics together!
