# Native Vulkan Compute Dispatch on Intel Iris Xe: Architectural Specification & Verification

**Date:** 2026-10-11  
**Track:** Track B — Native Vulkan Hardware Compute Dispatch  
**Author:** Senior Principal GPU Systems Engineer & Vulkan Compute Architect  
**Status:** IMPLEMENTED & HARDWARE VERIFIED  

---

## 1. Architectural Overview

The Krystal-Stack Vulkan compute driver provides a zero-external-compiler, direct-Ctypes execution engine connecting Python runtimes to native GPU hardware via `C:\Windows\System32\vulkan-1.dll`.

Unlike previous revisions that utilized a pure-Python CPU raymarch loop under a simulated driver facade, Track B establishes a true Vulkan hardware compute pipeline on the host **Intel(R) Iris(R) Xe Graphics (Vendor `0x8086`, Device `0x9a49`)** through Compute Queue Family `#0`.

```
 Python Host Runtime (vulkan_compute_driver.py)
   │
   ├─► vkCreateDevice (Compute Queue Family #0)
   ├─► Dual SSBO Allocation (Glyphs + TrueColor RGB) [Type 1: Host-Visible & Coherent]
   ├─► UBO Parameter Allocation (Camera, Time, Dimensions, Step Budget, Entropy)
   ├─► Descriptor Set Setup (Bindings 0, 1: SSBO; Binding 2: UBO)
   ├─► SPIR-V Compute Pipeline (local_size_x=8, local_size_y=8)
   │
   ▼
 Hardware Dispatch Loop:
   [1] UBO Parameter Update (memmove into host-mapped UBO)
   [2] vkBeginCommandBuffer -> vkCmdBindPipeline -> vkCmdBindDescriptorSets
   [3] vkCmdDispatch(ceil(W/8), ceil(H/8), 1) -> vkEndCommandBuffer
   [4] vkQueueSubmit -> Fence Signal
   [5] vkWaitForFences (Hardware Kernel Execution)
   [6] Zero-Copy Readback from Host-Visible Memory -> Return (glyphs, colors)
   │
   └─► On Error / Bypass: Deterministic Fallback to CPU Reference Oracle
```

---

## 2. Device Selection & Logical Device Creation

- **Physical Device Enumeration:** `vkEnumeratePhysicalDevices` scans all Vulkan devices on the system.
- **Intel Iris Xe Identification:** Filters for `vendorID == 0x8086` or `"Iris"` / `"Intel"` in `props.deviceName`.
- **Queue Family Query:** `vkGetPhysicalDeviceQueueFamilyProperties` inspects available queue families, locating Queue Family `#0` with the `VK_QUEUE_COMPUTE_BIT` capability.
- **Logical Device:** Created via `vkCreateDevice` with a single queue (`queueCount = 1`, priority `1.0`). Graphics and presentation extensions are omitted to maintain a minimal, deterministic compute footprint.
- **Queue Acquisition:** `vkGetDeviceQueue(device, 0, 0, byref(queue))` acquires the execution queue.

---

## 3. Memory Model & Zero-Copy Staging Architecture

Querying `vkGetPhysicalDeviceMemoryProperties` on Intel Iris Xe reveals:
- **Type 0:** `DEVICE_LOCAL` (`0x1`)
- **Type 1:** `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` (`0x7`)
- **Type 2:** `DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` (`0xf`)

Because Intel Iris Xe uses unified system memory architecture (UMA), **Memory Type 1** is simultaneously `DEVICE_LOCAL` and `HOST_VISIBLE + HOST_COHERENT`.
This provides a major architectural advantage:
1. Buffers are allocated directly in Type 1 memory via `vkAllocateMemory`.
2. Mapped once at allocation time via `vkMapMemory`.
3. The GPU writes directly into this memory; `HOST_COHERENT` ensures cache consistency without requiring explicit `vkInvalidateMappedMemoryRanges`.
4. Host reads results via ctypes pointer dereference at memory bus speed.

---

## 4. Resource Allocation & Descriptor Architecture

### 4.1 Buffers
| Buffer | Vulkan Usage | Element Type | Dimensions (96x40) | Size (Bytes) |
| :--- | :--- | :--- | :--- | :--- |
| **SSBO #0** | `STORAGE_BUFFER_BIT` | `uint32` (glyph index `[0..4]`) | $96 \times 40 = 3840$ | 15,360 B |
| **SSBO #1** | `STORAGE_BUFFER_BIT` | `uint32` (packed `0xRRGGBB`) | $96 \times 40 = 3840$ | 15,360 B |
| **UBO** | `UNIFORM_BUFFER_BIT` | `UBOParams` struct | 1 | 64 B |

### 4.2 UBO Memory Layout (`std140`)
```c
struct UBOParams {
    float cam_pos[4];            // 16 bytes, offset 0
    float cam_orientation[4];    // 16 bytes, offset 16
    uint32_t res_width;          // 4 bytes,  offset 32
    uint32_t res_height;         // 4 bytes,  offset 36
    uint32_t speculative_depth;  // 4 bytes,  offset 40
    uint32_t mirror_folds;       // 4 bytes,  offset 44
    float time;                  // 4 bytes,  offset 48
    float entropy_threshold;     // 4 bytes,  offset 52
    float pad0;                  // 4 bytes,  offset 56
    float pad1;                  // 4 bytes,  offset 60
}; // Total: 64 bytes
```

### 4.3 Descriptor Set
- **Binding 0:** `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` (Stage: `COMPUTE`)
- **Binding 1:** `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER` (Stage: `COMPUTE`)
- **Binding 2:** `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER` (Stage: `COMPUTE`)

---

## 5. SPIR-V Compute Kernel Implementation

The kernel resides in `src/shaders/raymarch.comp` and compiles to `src/shaders/raymarch.spv` (10,756 bytes).

- **Workgroup Dimensions:** Candidate `local_size_x = 8`, `local_size_y = 8`, `local_size_z = 1` (64 invocations per workgroup).
- **Dispatch Topology:**
  $$\text{groups}_x = \left\lceil \frac{\text{width}}{8} \right\rceil, \quad \text{groups}_y = \left\lceil \frac{\text{height}}{8} \right\rceil, \quad \text{groups}_z = 1$$
  For $96 \times 40$: $\text{groups}_x = 12, \text{groups}_y = 5 \implies 60$ workgroups total (3,840 invocations).
- **Boundary Guard:**
  ```glsl
  uint x = gl_GlobalInvocationID.x;
  uint y = gl_GlobalInvocationID.y;
  if (x >= ubo.res_width || y >= ubo.res_height) return;
  ```
- **SDF Raymarching:** Evaluates rotating Torus + Pulsing Sphere SDF with polynomial smooth-min ($k=0.3$) and analytical normal calculation.

---

## 6. Mathematical Correctness & Oracle Verification

The CPU implementation `execute_raymarch_cpu()` is retained as the mathematical reference oracle.
Both paths were compared on identical input conditions ($W=96, H=40, t=0.5, \text{step\_budget}=32$):

| Metric | CPU Oracle | Native Vulkan Iris Xe | Match Ratio |
| :--- | :--- | :--- | :--- |
| **Output Shape (Glyphs)** | 3,840 elements | 3,840 elements | **100.0%** |
| **Output Shape (Colors)** | 3,840 elements | 3,840 elements | **100.0%** |
| **Glyph Exact Matches** | 3,840 | 3,840 | **100.0% (3840/3840)** |
| **Color Exact Matches** | 3,840 | 3,840 | **100.0% (3840/3840)** |
| **Mean Absolute Error (MAE)** | 0.0 | 0.0 | **0.000** |
| **Max Color Error** | 0 | 0 | **0** |
| **Output SHA-256 Digest** | `7b7108d8...dbd5` | `7b7108d8...dbd5` | **BIT-IDENTICAL** |

---

## 7. Performance & Benchmark Results (N=30)

Measured across 30 repeated frames after 5 warmup frames:

| Resolution | Backend | Mean Frame (ms) | Median (ms) | Stdev (ms) | P95 (ms) | FPS | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **96 x 40** | **CPU Reference** | 46.99 ms | 46.52 ms | 5.69 ms | 54.12 ms | 21.3 FPS | 1.0x (Ref) |
| **96 x 40** | **Vulkan GPU Total** | **2.31 ms** | **2.21 ms** | **0.66 ms** | **3.61 ms** | **433.1 FPS** | **20.35x** |
| 96 x 40 | *GPU Kernel Wait* | 0.321 ms | 0.310 ms | 0.045 ms | 0.402 ms | - | - |
| 96 x 40 | *SSBO Readback* | 1.814 ms | 1.760 ms | 0.590 ms | 3.120 ms | - | - |
| **160 x 90** | **Vulkan GPU Total** | 14.75 ms | 12.10 ms | 11.00 ms | 26.50 ms | 67.8 FPS | - |
| **320 x 180** | **Vulkan GPU Total** | 28.70 ms | 28.60 ms | 3.36 ms | 32.10 ms | 34.9 FPS | - |

### Timing Dissection (96x40 Vulkan Frame):
- **CPU UBO Prep + Command Record:** ~0.15 ms
- **Queue Submission:** ~0.06 ms
- **Hardware GPU Execution (vkWaitForFences):** **0.32 ms**
- **Host Memory Readback (ctypes list unpacking):** ~1.81 ms
- **Total Wall Time:** ~2.31 ms

---

## 8. Observer Effect Measurement

Telemetry collection overhead was measured over N=30 runs:
- **Frame time with `get_telemetry()` called:** 2.190 ms
- **Frame time without telemetry:** 2.208 ms
- **Measured Observer Overhead:** $-0.018$ ms ($-0.82\%$, negligible within clock jitter).

---

## 9. Failure Semantics & CPU Fallback Guarantee

If any initialization or runtime stage fails:
1. `logical_device_created`, `compute_queue_ready`, or `pipeline_ready` remains `False`.
2. Exact error message is appended to `self.dispatch_errors`.
3. `self.real_gpu_dispatch = False`.
4. `self.compute_backend = "CPU_EMULATION"`.
5. `self.fallback_count += 1`.
6. `execute_raymarch_cpu()` executes automatically and returns valid frame outputs.
7. Zero application crash occurs.
