# CrystalSec 3.0 Phase 5.3C: Native CPU SIMD Evidence Path & Fair Cross-Tier Calibration

**Document Status:** Empirical Verification & Architectural Specification (Local Phase 5.3C Implementation)  
**Version:** 3.0.53C  
**Target Hardware:** Intel Core i5-1135G7 (Tiger Lake Family 6 Model 140, Willow Cove Core) + Intel Iris Xe Graphics G7 (80 EUs, 640 ALU threads)  
**Host OS:** Windows 11 AMD64 (with DeviceGuard / UMCI Enforced)

---

## 1. Executive Summary & Epistemic Boundaries

In Phase 5.0–5.2, CrystalSec established a heterogeneous execution model across Tier 0 (Scalar Reference), Tier 1 (Intel Willow Cove CPU), and Tier 2 (Intel Iris Xe GPU). However, an epistemic asymmetry remained:
- **Tier 0 (DirectSim):** Pure Python scalar loop (correctness oracle).
- **Tier 2 (Iris Xe):** Real compiled SPIR-V compute kernels dispatched via Vulkan driver (`vulkan_kernel_runner.py`), measured with wall-clock time on physical GPU hardware.
- **Tier 1 (CPU VNNI):** Still running interpreted Python loops in `cpu_vnni_backend.py` (labeled `HOST_CPU_EMULATED`).

### Crucial Epistemic Boundary (Phase 5.3C Status)
To avoid any recurrence of Phase 5.1-style evidence inflation, we strictly separate three distinct states:
1. **`CAPABILITY_DETECTED`:** CPUID and XGETBV verify live that AVX2, AVX-512F/BW/VL/VNNI and OS XSTATE are physically present and enabled by the Windows kernel.
2. **`SOURCE_IMPLEMENTED` / `BINARY_PROVENANCE_VERIFIED`:** Native C implementation ([`src/c/taxonomy_simd.c`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/c/taxonomy_simd.c)), header ([`src/c/taxonomy_simd.h`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/c/taxonomy_simd.h)), build script ([`scripts/build_taxonomy_simd.ps1`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/scripts/build_taxonomy_simd.ps1)), cryptographic manifest ([`artifacts/taxonomy_simd_manifest.json`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/artifacts/taxonomy_simd_manifest.json)), and verification script ([`scripts/verify_taxonomy_simd_binary.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/scripts/verify_taxonomy_simd_binary.py)) exist and match cryptographic hashes.
3. **`NATIVE_SIMD_EXECUTION_MEASURED`:** Occurs ONLY when the compiled DLL is loaded by the runtime process and the native symbols physically execute on CPU silicon. On this specific host, active Windows User Mode Code Integrity (UMCI / DeviceGuard Enforced) blocks loading unsigned DLLs with `WinError 4551`. Therefore, Tier 1 native execution is honestly labeled `UNAVAILABLE` in the runner, and hardware tests are skipped rather than faked.

> [!IMPORTANT]
> The empirical break-even measured in Phase 5.3C is **T0 Python Reference vs T2 Real Vulkan GPU**, NOT yet the fully fair **T1 Native SIMD vs T2 Real Vulkan GPU**. True native T1 vs T2 comparison is the explicit objective of Phase 5.3D.

---

## 2. Canonical Numerical Contract: INT8 Semantics

A critical microarchitectural distinction on Tiger Lake (and x86 SIMD in general) is that:
- `VPDPBUSD` (AVX-512 VNNI / AVX-VNNI), and
- `_mm256_maddubs_epi16` (AVX2)

operate with **unsigned byte × signed byte** semantics, accumulating into signed 32-bit integers (`uint8 activations × int8 weights → int32 accumulator`).

### Clarified Canonical Contract:
- **Input A (Activations):** `uint8_t` (range $[0, 255]$).
- **Input B (Weights):** `int8_t` (range $[-128, 127]$).
- **Accumulator:** Signed 32-bit integer (`int32_t`).
- **Canonical Operation:** Pure dot-product accumulation:
  $$\text{acc} = \sum_{i=0}^{n-1} a_i \cdot w_i$$
- **Post-Op Activation:** Explicit and separate (e.g. ReLU $\max(0, \text{acc})$ applied to the final accumulated sum, NOT per individual product).

*(Note: The legacy Phase 5.3 SPIR-V shader `int8_dot.spv` applied `max(0, a * b)` per product; Phase 5.3D will align the SPIR-V shader to this canonical reduction contract).*

---

## 3. Physical Hardware Capability Discovery (Live Host Probe)

Live discovery on the host physical processor using direct x86_64 CPUID (`leaf 1`, `leaf 7 sub-leaf 0`) and `XGETBV` confirms that the Willow Cove core physically supports all modern vector extensions, and the Windows OS has fully enabled extended context switching:

```json
{
  "cpu_model": "Intel64 Family 6 Model 140 Stepping 1, GenuineIntel",
  "architecture": "AMD64",
  "capabilities": {
    "sse42": true,
    "avx": true,
    "avx2": true,
    "fma": true,
    "avx512f": true,
    "avx512bw": true,
    "avx512vl": true,
    "avx512_vnni": true,
    "os_avx_state": true,
    "os_avx512_state": true
  }
}
```

### XSTATE Verification
- CPUID indicates hardware capability, but AVX/AVX-512 instructions will trigger `#UD` (Illegal Instruction) unless the operating system enables context saving in XCR0.
- Live `XGETBV(0)` read on this host returns `XCR0 = 0x00000000000000E7`, verifying:
  - Bit 0: x87 FPU state (`1`)
  - Bit 1: SSE state (`1`)
  - Bit 2: AVX state (`1`)
  - Bit 5: AVX-512 opmask state (`1`)
  - Bit 6: AVX-512 ZMM_Hi256 state (`1`)
  - Bit 7: AVX-512 Hi16_ZMM state (`1`)
- Both `os_avx_state` and `os_avx512_state` are verified `True`.

---

## 4. Native Binary Strategy & Windows UMCI Hardening

### Why RWX JIT was Rejected as Default
Creating `PAGE_EXECUTE_READWRITE` memory allocations at runtime introduces:
1. **Security & DEP conflicts:** Violates $W \oplus X$ (Write XOR Execute) memory protection.
2. **Antivirus / EDR intervention:** Triggers heuristic malware blocking.
3. **Auditability failure:** Lacks a static binary fingerprint (SHA-256) traceable to source git commits.

### The Phase 5.3C Pipeline:
1. **Canonical C Source & Header:** [`src/c/taxonomy_simd.h`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/c/taxonomy_simd.h) and [`src/c/taxonomy_simd.c`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/c/taxonomy_simd.c).
2. **Reproducible Build Script:** [`scripts/build_taxonomy_simd.ps1`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/scripts/build_taxonomy_simd.ps1) supporting MSVC (`cl.exe`), Clang, and GCC.
3. **Cryptographic Manifest:** [`artifacts/taxonomy_simd_manifest.json`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/artifacts/taxonomy_simd_manifest.json) recording compiler flags, target architecture, and SHA-256 hashes of sources and binary.
4. **Binary Verifier:** [`scripts/verify_taxonomy_simd_binary.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/scripts/verify_taxonomy_simd_binary.py).
5. **Runtime Loader:** [`src/python/native_cpu_simd_runner.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/src/python/native_cpu_simd_runner.py) checks the manifest and attempts `ctypes.CDLL`.
6. **Graceful Handling of Windows UMCI Policy:** On Windows systems with DeviceGuard / Smart App Control enforced (such as this host where `UsermodeCodeIntegrityPolicyEnforcementStatus == 2`), unsigned DLLs loaded from user directories are blocked by OS policy (`WinError 4551`). The runner explicitly reports `available() == False` with the exact OS reason, causing native hardware verification tests to **skip rather than faking execution with Python**.

---

## 5. Workload Taxonomy Implementations

All 5 taxonomy workloads are implemented natively in C with strict algorithmic parity:

| Workload Class | Native C Algorithm | SIMD Implementation | Numerical Contract |
| :--- | :--- | :--- | :--- |
| `INT8_DOT_PRODUCT` | Pure reduction $\sum a_i w_i$ & product $a_i w_i$ | AVX2 maddubs/madd & VNNI `_mm512_dpbusd_epi32` | $u8 \times s8 \to s32$, post-op separate |
| `FP32_VECTOR_MATH` | Horner form cubic polynomial $((c_3 x + c_2)x + c_1)x + c_0$ | AVX2 + FMA (`_mm256_fmadd_ps`) | IEEE-754 FP32 ($10^{-5}$ tol) |
| `MEMORY_BOUND_STREAM` | Strided buffer load: $out[i] = buf[i \cdot s] \times 1.001f$ | AVX2 streaming loads | Exact float representation |
| `BRANCH_HEAVY_CONTROL` | Genuine 5-branch decision ladder (modulo 11, 7, 3, 2) | Native scalar branch code (tests CPU branch predictor) | Exact integer transitions |
| `MIXED_COMPUTE_MEMORY` | Paired memory traversal and dot-product compute | AVX2 vectorized compute + stream | Matches constituent contracts |

---

## 6. Empirical Cross-Tier Calibration (T0 Reference vs T2 Vulkan GPU)

Running the benchmark sweep ([`scripts/benchmark_native_cpu_vs_vulkan.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/scripts/benchmark_native_cpu_vs_vulkan.py)) across problem sizes $N \in [64, 16384]$ on physical hardware produces genuine break-even distributions:

### Measured Results: `INT8_DOT_PRODUCT` (T0 Reference vs T2 Real GPU)

| $N$ (Size) | T0 Python (Reference) | T2 Vulkan (Kernel) | T2 Vulkan (End-to-End) | Winner (Kernel) | Winner (E2E) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **64** | 0.0322 ms | 0.2591 ms | 0.2849 ms | **CPU** | **CPU** |
| **256** | 0.0543 ms | 0.2652 ms | 0.3316 ms | **CPU** | **CPU** |
| **1,024** | 0.4215 ms | 0.2804 ms | 0.4717 ms | **GPU** ($N^*_k = 1024$) | **CPU** (Marshal delays GPU) |
| **4,096** | 1.3127 ms | 0.3224 ms | 0.9556 ms | **GPU** | **GPU** ($N^*_{\text{e2e}} = 4096$) |
| **16,384** | 3.8388 ms | 0.3473 ms | 2.0419 ms | **GPU** | **GPU** |

### Crucial Empirical Confirmation: $N^*_{\text{kernel}} < N^*_{\text{e2e}}$
As predicted in the heterogeneous architecture model, **the GPU wins kernel execution long before it wins end-to-end latency**:
- At $N = 1024$, GPU kernel execution is **33% faster** than the CPU reference ($0.2804\text{ ms}$ vs $0.4215\text{ ms}$).
- However, when taking into account data marshaling, host-visible UMA upload, and readback, GPU end-to-end latency is **$0.4717\text{ ms}$**, meaning the **CPU reference still wins on end-to-end latency**.
- The persistent end-to-end crossover point occurs at $N \ge 4096$.

---

## 7. Next Target: Phase 5.3D — Trusted Native SIMD Execution

To achieve true $T_1\text{ Native SIMD vs } T_2\text{ Vulkan}$ empirical measurement, Phase 5.3D will establish native execution trust:

```text
Phase 5.3D Definition of Done:
1. Native DLL loads under active Windows policy (trusted/signed binary).
2. Executed symbol is recorded in telemetry.
3. AVX2 path is observed live on CPU silicon.
4. VNNI path is observed live where selected.
5. No Python code participates in cpu_native_ms.
6. Mutation delay changes measured cpu_native_ms.
7. Disassembly confirms expected SIMD ISA instructions.
8. T1 output matches T0 reference.
9. T1 vs T2 size sweep executes in the exact same benchmark session.
10. Break-even crossover is derived from native CPU, not Python reference.
```
