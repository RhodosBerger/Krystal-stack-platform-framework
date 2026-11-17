# Stock Baseline Solutions vs. Krystal Stack Platform — Measured Benchmark Differences

> **Constitutional Invariant:** `VITAL_MAX_HP == 6` | **Hardware Scope:** `4C/8T`, OpenVINO `['CPU', 'GPU']` | **Evidence Level:** `LOCAL_MEASURED`

## 1. Head-to-Head Measured Difference Table (`BM_01..BM_08`)

| ID | Subsystem Domain | Stock Median (`P95`) | Krystal Median (`P95`) | Exact Difference (`Krystal - Stock`) | Gain / Ratio | Winner |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `BM_01_OPENVINO_ZERO_COPY_4T_INFER` | OpenVINO Inference & Memory Handoff (`[64x512] -> [64x256]`) | `0.2618 ms` (`1.36`) | **`0.1511 ms`** (`0.5377`) | `-0.1107 ms` (`+42.28%`) | **`1.73x`** | `KRYSTAL_PLATFORM` |
| `BM_02_IEEE754_SUBNORMAL_BOUNDARY_GUARD` | x86-64 FPU / SIMD Underflow & Decay Resilience (`N = 16,384`) | `114.3 us` (`136.9`) | **`6.75 us`** (`12.9`) | `-107.55 us` (`+94.09%`) | **`16.93x`** | `KRYSTAL_PLATFORM` |
| `BM_03_FREIVALDS_AUDIT_VS_FULL_RECOMPUTE` | Epistemic Tensor Result Verification (`256 x 256` GEMM Audit) | `660.05 us` (`1826.8`) | **`150.8 us`** (`449.9`) | `-509.25 us` (`+77.15%`) | **`4.38x`** | `KRYSTAL_PLATFORM` |
| `BM_04_IPC_JSON_VS_192B_HYPERTRANSPORT` | Cross-Boundary IPC Serialization & Cryptographic Attestation | `23350.9766 ns/frame` (`29714.4531`) | **`2681.6406 ns/frame`** (`3978.9062`) | `-20669.336 ns/frame` (`+88.52%`) | **`8.71x`** | `KRYSTAL_PLATFORM` |
| `BM_05_KERNEL_ALLOCATOR_HEAP_VS_BITMAP64_SLAB` | Kernel Resource Slot Allocation & Bitwise Authority Gate (`64` Slots) | `1796.9727 ns/op` (`2646.9727`) | **`763.6719 ns/op`** (`1494.7266`) | `-1033.3008 ns/op` (`+57.5%`) | **`2.35x`** | `KRYSTAL_PLATFORM` |
| `BM_06_DYNAMIC_4T_LOGGING_MUTEX_VS_SHARDED_SPSC` | 4-Thread Concurrent Telemetry & AI-Chip Connection Logging (`2,048` Events) | `16.1976 ms` (`51.1301`) | **`3.3446 ms`** (`6.99`) | `-12.853 ms` (`+79.35%`) | **`4.84x`** | `KRYSTAL_PLATFORM` |
| `BM_07_XERJ_AST_FASTPATH_VS_STOCK_RAG_ENVELOPE` | Agent Command Compilation & Token Footprint (Local Envelope + Parse) | `48.45 us` (`116.1`) | **`134.9 us`** (`827.4`) | `+86.45 us` (`-178.43%`) | **`0.36x`** | `STOCK_SOLUTION` |
| `BM_08_HYPERVISOR_JITTER_THRASH_TRANSITIONS` | 64-Frame Noisy OS Telemetry Stability (8 Isolated 1-Tick `0.82` Jitter Spikes) | `16.0 mode_flips` (`16.0`) | **`2.0 mode_flips`** (`2.0`) | `-14.0 mode_flips` (`+87.5%`) | **`8.0x`** | `KRYSTAL_PLATFORM` |

## 2. Summary & Epistemic Interpretation
- **Benchmarks Won by Krystal Platform:** **`7 / 8`** (Stock wins on pure unvalidated string dump if excluding LLM call wait: `1`).
- **Geometric Mean Efficiency Ratio Across All 8 Domains:** **`3.72x`**.
- **Machine-Readable Artifacts Written:**
  - `reports/stock_vs_krystal_benchmark_diff.json`
  - `reports/stock_vs_krystal_benchmark_diff.csv`
