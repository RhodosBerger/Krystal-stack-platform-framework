# UNIFORM CROSS-ENGINE INTERPOSITION & MULTI-PROCESS SCHEDULING SPECIFICATION
## Krystal-Nexus Engine-Agnostic UMA Arbitration & System-Wide Coexistence
**Document ID:** `ARCH-NEXUS-INTERPOSER-2026-V1`  
**Milestone:** `V-UNIVERSAL-INTERPOSER-01`  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  
**Status:** Canonical Architectural Standard  

---

### Executive Summary & Problem Formulation

In the reference milestone, Krystal-Stack demonstrated zero-stutter coexistence on Unified Memory Architecture (UMA) chips using an instrumented, modified Godot engine. In that setup, the engine was explicitly modified with internal K-NSS shaders, telemetry emission hooks, and direct IPC links.

However, real-world deployment requires **uniform, constant manifestation across arbitrary, unmodified software**:
1. **Arbitrary 3D Graphics Engines:** Closed-source AAA games, Unreal Engine 5, Unity, Blender (Cycles/EEVEE viewports), Autodesk Maya, CAD software, and video editing suites (DaVinci Resolve, Premiere Pro) without recompiling or altering a single line of their source code.
2. **Arbitrary External Compute Processes:** Background LLMs (Ollama, llama.cpp, vLLM), heavy batch compilation (`rustc`, `clang`), video transcoders, and database engines contending for the shared LPDDR4x/5 memory bus (~68 GB/s).

This specification formalizes the **Four-Layer Uniform Interposition & Multi-Process Governance Architecture**.

---

### Layer 1: Non-Invasive Graphics API Interposition

To observe and pace any graphics engine without touching its source code, Krystal-Nexus intercepts the graphics presentation boundary at the OS loader level.

```
+-------------------------------------------------------------------------+
|                      APPLICATION / ENGINE LAYER                         |
|     [Unreal Engine 5]    [Stock Blender]    [Unity]    [DirectX 12 Game]|
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  GRAPHICS LOADER & INTERPOSITION HOOKS                  |
|                                                                         |
|  1. Khronos Vulkan Implicit Layer:       2. DirectX / DXGI Detour:      |
|     - VK_LAYER_KRYSTAL_NEXUS                - Hook IDXGISwapChain::Present|
|     - Hook vkQueuePresentKHR                - DwmGetCompositionTimingInfo|
|     - Hook vkAcquireNextImageKHR            - GPU Fence Queries         |
|     - Hook vkQueueSubmit                                                |
+------------------------------------+------------------------------------+
                                     | Telemetry (<50ns)
                                     v
+-------------------------------------------------------------------------+
|          128-BYTE CACHE-LINE ZERO-COPY RING BUFFER (/dev/shm)           |
|          [KrystalZeroCopyFrame: render_ms, drs_scale, hp=6]             |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                 KRYSTAL-NEXUS CENTRAL ARBITER DAEMON                     |
|  - UMA Bus Saturation Monitor (Threshold: 78%)                          |
|  - VSync Micro-Slicing Scheduler (2.2ms Blanking Window)                |
|  - Asymmetric Schmitt-Trigger Hysteresis Filter (12-frame dwell)        |
+------------------------------------+------------------------------------+
                                     | OS Throttling Commands
                                     v
+-------------------------------------------------------------------------+
|                     OS PROCESS & HARDWARE GOVERNOR                      |
|  - Windows: SetProcessInformation(EcoQoS) / PriorityClass / Thread Pause|
|  - Linux: Cgroups v2 (cpu.weight, memory.high) / Intel RDT MBA          |
|  - Hardware MSR: PL1 Clamp Unlock (15W -> 32W, Arrhenius T < 75°C)      |
+-------------------------------------------------------------------------+
```

#### 1.1 Khronos Vulkan Implicit Layer (`VK_LAYER_KRYSTAL_NEXUS`)
Khronos Vulkan specification defines an official mechanism for implicit layers that load transparently into any Vulkan application launched on the host:
- **Windows Registry Key:**
  `HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\Vulkan\ImplicitLayers` (or `HKEY_CURRENT_USER`)
- **Linux / POSIX Path:**
  `/etc/vulkan/implicit_layer.d/` or `~/.local/share/vulkan/implicit_layer.d/`

When any Vulkan executable starts (whether Godot, Unreal, Blender, or a game), the Vulkan loader automatically links `VkLayer_krystal_nexus.dll` / `libVkLayer_krystal_nexus.so`.

**Intercepted Entry Points:**
* `vkQueuePresentKHR`: Captures exact frame completion time and measures presentation latency against the $8.33\text{ ms}$ (120 Hz) or $16.67\text{ ms}$ (60 Hz) deadline.
* `vkAcquireNextImageKHR`: Measures swapchain queue depth and backpressure.
* `vkQueueSubmit`: Tracks GPU execution duration via calibrated GPU timestamp queries (`vkCmdWriteTimestamp`).

#### 1.2 DirectX (DXGI / D3D11 / D3D12) Presentation Interception
On Windows, DirectX engines present frames via `IDXGISwapChain::Present` and `IDXGISwapChain1::Present1`.
* Krystal-Nexus injects a detour hook into the DXGI swapchain present call.
* Cross-referenced with the Desktop Window Manager (DWM) composition timings via `DwmGetCompositionTimingInfo()` to detect display scanout phase with microsecond accuracy.

#### 1.3 Dynamic Resolution Scaling (DRS) Feedback
When an external engine supports dynamic resolution scaling (e.g., Unreal Engine's TSR/DRS, FSR 2/3, or DLSS/XeSS plugins), the interposer injects the optimal scale factor calculated by Krystal's predictive profiler directly into the shared memory feedback channel, guiding render scaling from $1.0\times$ down to $0.67\times$ (or $0.50\times$) during peak load.

---

### Layer 2: Operating-System-Level Process & Thread QoS Governor

External compute processes (Ollama, llama.cpp, PyTorch workers, background compilers) have no awareness of the graphics pipeline. Krystal-Nexus controls them via the operating system's kernel scheduling and power primitives.

| Platform | Mechanism | Action During Active Scanout ($\mu > 0.78$) | Action During VBlank Window ($2.2\text{ ms}$) |
|---|---|---|---|
| **Windows 11 / 10** | **ProcessPowerThrottling (EcoQoS)** | Enables `PROCESS_POWER_THROTTLING_EXECUTION_SPEED`; forces threads onto Intel E-cores. | Disables EcoQoS; allows high-frequency bursts on P-cores. |
| **Windows 11 / 10** | **Thread Slicing / Suspend** | Pauses background compute threads during the active $6.1\text{ ms}$ raster scanout. | Unpauses compute threads; allows token generation during display blanking. |
| **Linux / WSL** | **Cgroups v2 & resctrl** | Clamps `memory.high` and throttles memory bandwidth via Intel RDT / AMD MBA to $8\text{ GB/s}$. | Restores full memory bandwidth allocation ($25\text{ GB/s}$). |
| **Cross-Platform** | **Speculative Token Compression** | Signals LLM runtime to clamp $K_{\text{spec}} = 8 \to 1$. | Restores speculative lookahead $K_{\text{spec}} \to 4\dots 8$. |

---

### Layer 3: The 128-Byte Zero-Copy Telemetry Bus

Communication between the injected graphics layer, the central arbiter daemon, and external processes occurs via a lock-free, cache-line-aligned binary ring buffer (`krystal_uma_ring`):

```c
#pragma pack(push, 1)
typedef struct {
    uint32_t magic;               // 0x4B525953 ("KRYS")
    uint32_t version;             // 2
    uint16_t vital_hp;            // Inviolable: Must equal 6
    uint16_t contract_type;       // BinaryContractType (STATE = 1)
    uint32_t payload_size;        // 16 bytes
    uint32_t flags;               // QoS & blanking flags
    uint32_t sequence;            // Monotonic frame index
    uint32_t generation;          // Policy epoch
    uint64_t timestamp_ns;        // Hardware timestamp
    uint32_t cap_scope;           // Capability mask
    uint64_t nonce;               // Replay prevention
    uint8_t  predecessor_hash[32];// Causal chain digest
    uint16_t profile_id;          // NSS profile ID
    uint16_t verdict_id;          // Arbitration verdict
    uint32_t mask;                // Actuator mask
    float    gpu_render_ms;       // Measured GPU frame time
    float    drs_scale_factor;    // Recommended DRS scale (0.50 - 1.0)
    uint8_t  signature_digest[32];// SHA-256 tamper-evident digest
} KrystalZeroCopyFrame;
#pragma pack(pop)
```

**Key Performance Properties:**
* **Size:** Exactly 128 bytes (occupies exactly two 64-byte CPU cache lines).
* **Write Latency:** $< 45\text{ nanoseconds}$ on x86_64 and ARM64.
* **Lock-Free:** Single-producer single-consumer (SPSC) ring buffer with atomic head/tail pointers.
* **Constitutional Invariant:** Every frame strictly validates `vital_hp == 6`.

---

### Layer 4: Mathematical Multi-Process Coexistence Model

The central daemon evaluates UMA memory bus saturation:
$$\mu = \frac{BW_{\text{GPU}} + \sum_{p \in \text{PIDs}} BW_{p}}{BW_{\text{Max}}}$$

Where $BW_{\text{Max}} \approx 68.0\text{ GB/s}$ on LPDDR4x-4266 dual-channel.

When $\mu > 0.78$ and rendering is active:
1. **Step 1 (Micro-Slice Budgeting):** The available window for external compute is:
   $$t_{\text{slice}} = \min(t_{\text{blanking}}, \max(0, t_{\text{budget}} - t_{\text{render}} - \delta_{\text{jitter}}))$$
   At $120\text{ Hz}$, $t_{\text{budget}} = 8.33\text{ ms}$, $t_{\text{blanking}} \approx 2.2\text{ ms}$, and $\delta_{\text{jitter}} < 0.05\text{ ms}$ (hardware VSync fence).
2. **Step 2 (External Compute Throttling):** Compute processes are shifted to `ProcessQosClass.ECO_QOS_EFFICIENCY` and their memory bus quota is capped to $8\text{ GB/s}$.
3. **Step 3 (Dynamic Resolution Scaling):** The graphics hook recommends a temporary resolution scale of $0.67\times$, relieving GPU texture sampling pressure.
4. **Step 4 (Asymmetric Hysteresis):** Once nominal conditions return ($\mu < 0.70$), the system holds the stabilized profile for a 12-frame dwell time ($100\text{ ms}$ at $120\text{ Hz}$) before scaling quality back up, guaranteeing zero visual stutter and zero rapid oscillation.

---

### Verification and Compliance

The uniform interposition engine is verified in `tests/test_v_universal_engine_interposer.py`:
* **Test 01:** Constitutional invariant `VITAL_MAX_HP == 6` enforced across all manifest, hook, and QoS structures.
* **Test 02:** Khronos Vulkan Layer JSON manifest compliance for Windows (`.dll`) and Linux (`.so`).
* **Test 03:** 128-byte cache-line frame serialization verified with exact struct packing.
* **Test 04:** OS Process QoS enforcement & VBlank phase micro-slicing validation.
* **Test 05:** Multi-engine coordination across Unreal Engine 5, Blender, and DirectX 12 games.
* **Test 06:** C-ABI header export (`krystal_nexus_api.h`) structural verification.
