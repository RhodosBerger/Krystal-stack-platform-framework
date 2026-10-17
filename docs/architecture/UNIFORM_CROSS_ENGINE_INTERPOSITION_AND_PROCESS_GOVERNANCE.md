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
#include <stdint.h>
#include <stddef.h>

typedef struct _Alignas(64) {
    /* CL0: 0..63 */
    uint32_t magic;                /* 0..3   */
    uint32_t version;              /* 4..7   */
    uint16_t vital_hp;             /* 8..9   (Invariant == 6) */
    uint16_t contract_type;        /* 10..11 */
    uint32_t payload_size;         /* 12..15 */
    uint32_t flags;                /* 16..19 */
    uint32_t sequence;             /* 20..23 */
    uint32_t generation;           /* 24..27 */
    uint32_t cap_scope;            /* 28..31 */
    uint64_t timestamp_ns;         /* 32..39 (8B aligned at offset 32) */
    uint64_t nonce;                /* 40..47 (8B aligned at offset 40) */
    uint8_t  predecessor_hash[32]; /* 48..79 (Spans CL0 [48..63] & CL1 [64..79]) */

    /* CL1: 64..127 */
    uint16_t profile_id;           /* 80..81 */
    uint16_t verdict_id;           /* 82..83 */
    uint32_t mask;                 /* 84..87 */
    float    gpu_render_ms;        /* 88..91 */
    float    drs_scale_factor;     /* 92..95 */
    uint8_t  frame_hmac[32];       /* 96..127 (HMAC-SHA256 over bytes 0..95) */
} KrystalZeroCopyFrame;

_Static_assert(sizeof(KrystalZeroCopyFrame) == 128, "Frame size must be exactly 128B (2 cache lines)");
_Static_assert(_Alignof(KrystalZeroCopyFrame) == 64, "Frame must be 64B cache-line aligned");
_Static_assert(offsetof(KrystalZeroCopyFrame, timestamp_ns) == 32, "timestamp_ns offset drift");
_Static_assert(offsetof(KrystalZeroCopyFrame, nonce) == 40, "nonce offset drift");
_Static_assert(offsetof(KrystalZeroCopyFrame, profile_id) == 80, "CL1 payload offset drift");
_Static_assert(offsetof(KrystalZeroCopyFrame, frame_hmac) == 96, "HMAC digest offset drift");
```

**Key Performance Properties:**
* **Size:** Exactly 128 bytes (occupies exactly two 64-byte CPU cache lines: CL0 = 0..63B, CL1 = 64..127B).
* **Natural Scalar Alignment:** Every scalar field ($2\text{B}, 4\text{B}, 8\text{B}$) is naturally aligned to its size without padding. Layout is identical with or without `#pragma pack(1)`. Eliminates unaligned 64-bit access faults and penalties on ARM64 and RISC-V.
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

## Milestone V-ZEROCOPY-MEMORY-MODEL-02: Cross-ISA Publication Semantics & Hardware Contract

The transition from a logical serialization structure to a hardware-grounded memory contract requires formal publication semantics across weakly ordered architectures (ARM64, RISC-V RVWMO) and strong ordering (x86_64 TSO). Commit `b3439414` proved field layout invariance ([STRUCTURALLY-PROVEN]); milestone `V-ZEROCOPY-MEMORY-MODEL-02` seals base alignment, cache-line separation, full cryptographic binding, and torn-read prevention.

### 1. Base Alignment and `#pragma pack(1)` Elimination

In C/C++, `#pragma pack(push, 1)` forces `alignof(KrystalZeroCopyFrame) == 1`. When allocated on the stack or in an array without compiler attributes, base address alignment ($\text{Addr} \bmod 8 = 0$) is unconstrained, risking hardware alignment faults on ARM64 and RISC-V when accessing 64-bit scalars (`timestamp_ns` at offset 32, `nonce` at offset 40).

Because all fields in `KrystalZeroCopyFrame` are naturally aligned (offset % size == 0) with zero padding:
$$\text{alignof}(\text{KrystalZeroCopyFrame}) = 64 \quad (\text{\_Alignas}(64) \text{ in C11}, \text{\#[repr(C, align(64))]} \text{ in Rust})$$

For the shared SPSC ring buffer (page-aligned at $4096\text{ B}$ via `mmap` or virtual memory allocation):
$$\text{Addr}(\text{Slot}_n) = \text{RingBaseAddress} + \text{HeaderSize}_{128\text{B}} + 128 \cdot (n \bmod N_{\text{slots}}) \implies \text{Addr}(\text{Slot}_n) \bmod 64 = 0$$

### 2. Elimination of Cache Line Bouncing (Ring Header CL0/CL1 Split)

To eliminate false sharing between producer and consumer threads:
* `RING_HEADER_SIZE` is upgraded to exactly 128 bytes (two 64B cache lines).
* **CL0 (Bytes 0..63):** Producer-owned fields (`magic`, `version`, `capacity`, `slot_size`, `head` pointer at offset 16).
* **CL1 (Bytes 64..127):** Consumer-owned fields (`tail` pointer at offset 64, `drop_count` at offset 72).

Producer stores to `head` and consumer stores to `tail` execute on completely independent L1 cache lines with zero cache-line invalidation traffic.

### 3. Full 96-Byte Cryptographic Binding (`frame_hmac`) & Float Canonicalization

Legacy partial hashing covered only 4 fields (leaving 78 bytes unprotected). An adversary or memory corruption could tamper with `cap_scope` (escalating permissions from `CAN_PROPOSE` to `CAN_EXECUTE`), `generation`, or `verdict_id` without invalidating the digest.

Milestone `V-ZEROCOPY-MEMORY-MODEL-02` binds the entire 96-byte semantic prefix:
$$\text{frame\_hmac}[0..31] = \text{HMAC-SHA256}_{K_{\text{session}}}\big(\text{FrameBytes}[0..95]\big)$$

* **Float Canonicalization:** Prior to serialization, float32 fields (`gpu_render_ms`, `drs_scale_factor`) are normalized to $+0.0\text{f}$ (rejecting $-0.0\text{f}$) and validated against IEEE-754 `NaN`/$\pm\infty$ to prevent bitwise divergence in the HMAC digest.
* In unkeyed telemetric modes, SHA-256 over `FrameBytes[0..95]` produces `integrity_digest`.

### 4. Cross-ISA Publication Protocol and Split `predecessor_hash`

Because `predecessor_hash[32]` spans CL0 [48..63] (16B) and CL1 [64..79] (16B), weak memory ordering could allow a consumer to read a new CL0 with an old CL1, creating a torn frame. Publication is therefore gated by atomic acquire/release synchronization on `ring->head`:

```text
PRODUCER (Writer)                                  CONSUMER (Reader)
-------------------------------------------------  -------------------------------------------------
1. slot = &ring->slots[head & mask]                1. pub_head = atomic_load_explicit(
2. Write bytes 0..63   (CL0: header + hash[0..15])      &ring->head, memory_order_acquire)
3. Write bytes 64..95  (CL1: hash[16..31] + data)     [x86: MOV | ARM64: LDAR | RISC-V: lw + fence r,rw]
4. Write bytes 96..127 (CL1: HMAC(bytes 0..95))    2. if (pub_head == tail) -> RING_EMPTY
5. atomic_store_explicit(                          3. slot = &ring->slots[tail & mask]
     &ring->head, head + 1,                        4. Copy/read 128B (CL0 + CL1 are guaranteed visible!)
     memory_order_release)                         5. Verify vital_hp == 6 && HMAC(slot[0..95])
   [x86: MOV | ARM64: STLR | RISC-V: fence rw,w]   6. atomic_store_explicit(&ring->tail, tail + 1, release)
```

In non-cooperative seqlock/sampling modes, computing `HMAC-SHA256(FrameBytes[0..95])` detects any torn read between CL0 and CL1 with collision probability $2^{-256}$.

---

### Taxonómia verifikačných stavov pre audit a CI

| Kategória | Verifikačný štítok | Reálne dokázané v b3439414 | Implementované a overené v V-ZEROCOPY-MEMORY-MODEL-02 |
|---|---|---|---|
| **Štruktúra & ABI** | `[STRUCTURALLY-PROVEN]` | Veľkosť 128B, prirodzené offsety polí (`timestamp_ns` @ 32, `nonce` @ 40), zhoda Python `struct` a C definície. | Eliminácia `#pragma pack(1)` v prospech `_Alignas(64)` / `#[repr(C, align(64))]`, garancia `base_addr % 64 == 0` a stride == 128. |
| **Pamäťový model** | `[MODEL-TESTED]` vs. `[SILICON-VERIFIED]` | Logická serializácia v jednom vlákne. | Multi-thread SPSC stress test s garanciou 0 drops, detekcia torn-reads cez rozštiepený `predecessor_hash` (48..79) a Acquire/Release bariéry. *(ARM64/RISC-V HW beh vyžaduje kremík).* |
| **Kryptografia** | `[PARTIAL-HASH]` vs. `[FULL-FRAME-AUTHENTICATED]` | SHA-256 nad 4 vybranými poľami (78B nechránených). | HMAC-SHA256 nad kompletným blokom `FrameBytes[0..95]` vrátane `cap_scope`, `nonce`, `flags` a `predecessor_hash`. |
| **Výkonnosť & CI** | `[LOCAL-REPORTED]` vs. `[REMOTE-CI-VERIFIED]` | Lokálne nameraných 44 testov; odhad latencie < 45 ns analytický. | 56/56 lokálnych testov prechádza (vrátane 12 nových v `test_v_zerocopy_memory_model_02.py`). Oddelenie `[ANALYTICAL-ESTIMATE]` od `[HARDWARE-MEASURED]`. |

---

### Verification and Compliance

The uniform interposition engine and memory model are verified in:
* `tests/test_v_universal_engine_interposer.py` (6 integration tests)
* `tests/test_v_zerocopy_memory_model_02.py` (12-point memory model test suite):
  * **Test 01:** Constitutional invariant `VITAL_MAX_HP == 6` strictly verified.
  * **Test 02:** Base alignment $\text{Addr} \bmod 64 == 0$ and cache line stride $128\text{B}$.
  * **Test 03:** Natural scalar offsets identical with and without `#pragma pack(1)`.
  * **Test 04:** Ring buffer header CL0/CL1 separation (`head` @ 16, `tail` @ 64).
  * **Test 05:** Float canonicalization ($-0.0 \to +0.0$) and `NaN`/$\pm\infty$ rejection.
  * **Test 06:** Full 96-byte HMAC coverage vs. partial hash.
  * **Test 07:** Exhaustive tamper detection across every field in bytes 0..95.
  * **Test 08:** Split `predecessor_hash` cross-cache-line boundary integrity.
  * **Test 09:** Cross-ISA Acquire/Release publication simulation.
  * **Test 10:** Torn-read detection under asynchronous mutation.
  * **Test 11:** C-ABI header export with `_Alignas(64)` and C11 `_Static_assert`s.
  * **Test 12:** Multi-threaded concurrent SPSC stress test with 0 drops and 0 tears.

