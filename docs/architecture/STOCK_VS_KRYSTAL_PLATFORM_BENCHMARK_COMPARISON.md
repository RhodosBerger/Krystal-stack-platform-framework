# Stock Baseline Solutions vs. Krystal Stack Platform — Measured Benchmark Differences

> **Constitutional Invariant:** `VITAL_MAX_HP == 6` | **Hardware Scope:** `4C/8T`, OpenVINO `['CPU', 'GPU']` | **Evidence Level:** `LOCAL_MEASURED`

## 1. Head-to-Head Measured Difference Table (`BM_01..BM_08`)

| ID | Subsystem Domain | Stock Median (`P95`) | Krystal Median (`P95`) | Exact Difference (`Krystal - Stock`) | Gain / Ratio | Winner |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `BM_01_OPENVINO_ZERO_COPY_4T_INFER` | OpenVINO Inference & Memory Handoff (`[64x512] -> [64x256]`) | `0.378 ms` (`0.9084`) | **`0.111 ms`** (`0.2463`) | `-0.267 ms` (`+70.63%`) | **`3.41x`** | `KRYSTAL_PLATFORM` |
| `BM_02_IEEE754_SUBNORMAL_BOUNDARY_GUARD` | x86-64 FPU / SIMD Underflow & Decay Resilience (`N = 16,384`) | `127.3 us` (`4961.0`) | **`7.5 us`** (`45.6`) | `-119.8 us` (`+94.11%`) | **`16.97x`** | `KRYSTAL_PLATFORM` |
| `BM_03_FREIVALDS_AUDIT_VS_FULL_RECOMPUTE` | Epistemic Tensor Result Verification (`256 x 256` GEMM Audit) | `797.85 us` (`7799.7`) | **`153.7 us`** (`1974.2`) | `-644.15 us` (`+80.74%`) | **`5.19x`** | `KRYSTAL_PLATFORM` |
| `BM_04_IPC_JSON_VS_192B_HYPERTRANSPORT` | Cross-Boundary IPC Serialization & Cryptographic Attestation | `48936.3281 ns/frame` (`66844.9219`) | **`4450.9766 ns/frame`** (`9234.7656`) | `-44485.3515 ns/frame` (`+90.9%`) | **`10.99x`** | `KRYSTAL_PLATFORM` |
| `BM_05_KERNEL_ALLOCATOR_HEAP_VS_BITMAP64_SLAB` | Kernel Resource Slot Allocation & Bitwise Authority Gate (`64` Slots) | `2928.418 ns/op` (`4599.8047`) | **`1043.7012 ns/op`** (`6559.082`) | `-1884.7168 ns/op` (`+64.36%`) | **`2.81x`** | `KRYSTAL_PLATFORM` |
| `BM_06_DYNAMIC_4T_LOGGING_MUTEX_VS_SHARDED_SPSC` | 4-Thread Concurrent Telemetry & AI-Chip Connection Logging (`2,048` Events) | `44.9878 ms` (`151.0505`) | **`10.4205 ms`** (`62.949`) | `-34.5673 ms` (`+76.84%`) | **`4.32x`** | `KRYSTAL_PLATFORM` |
| `BM_07_XERJ_AST_FASTPATH_VS_STOCK_RAG_ENVELOPE` | Agent Command Compilation & Token Footprint (Local Envelope + Parse) | `51.25 us` (`142.6`) | **`134.55 us`** (`7389.8`) | `+83.3 us` (`-162.54%`) | **`0.38x`** | `STOCK_SOLUTION` |
| `BM_08_HYPERVISOR_JITTER_THRASH_TRANSITIONS` | 64-Frame Noisy OS Telemetry Stability (8 Isolated 1-Tick `0.82` Jitter Spikes) | `16.0 mode_flips` (`16.0`) | **`2.0 mode_flips`** (`2.0`) | `-14.0 mode_flips` (`+87.5%`) | **`8.0x`** | `KRYSTAL_PLATFORM` |

## 2. Summary & Epistemic Interpretation
- **Benchmarks Won by Krystal Platform:** **`7 / 8`** (Stock wins on pure unvalidated string dump if excluding LLM call wait: `1`).
- **Geometric Mean Efficiency Ratio Across All 8 Domains:** **`4.32x`**.
- **Machine-Readable Artifacts Written:**
  - `reports/stock_vs_krystal_benchmark_diff.json`
  - `reports/stock_vs_krystal_benchmark_diff.csv`
