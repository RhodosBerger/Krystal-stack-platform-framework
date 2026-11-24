# Hybridná Integrácia Rustu, Linuxové Ovládače a CUDA-Ekvivalentná Vrstva pre Stream Procesory, AI Jadrá a NPU

> **Ústavný invariant:** `VITAL_MAX_HP == 6` (`\neg Valid \implies \Delta S = 0`) | **Schéma:** `schemas/hybrid_rust_intel_xe_stream_integration.schema.json`

## 1. Päť rovín hybridnej integrácie Rustu (`HRI_P1..P5`)
| Rovina | Jazyk & ABI | Kľúčové štruktúry | Kontrakt pamäte |
| :--- | :--- | :--- | :--- |
| `HRI_P1_RUST_MEMORY_SAFE_CORE` | Rust (`#[repr(C, align(128))]` + typová mreža v `src/rust-bot/src/`) | `HybridRustXeLaunchDescriptor128B, UniversalMmioBar0Aperture` | Vlastní stavové prechody, O(1) TZCNT bitmapovú alokáciu a neporušiteľnosť Read-Only registrov (Delta_S == 0). |
| `HRI_P2_C11_ZERO_MANGLE_MMIO_ABI` | ANSI C11 (`extern "C"` v `include/krystal_universal_mmio_aperture.h` & `kr_authority_bits.h`) | `KrystalMmioCommand, kr_authority_input` | Stabilné 16 KB `mmap` okno (`0x0000` BAR0, `0x1000` 64B hierarchia, `0x2000` 128B SPSC zbernica) bez C++ name-manglingu. |
| `HRI_P3_UNIFIED_STREAM_AI_NPU_SILICON` | SPIR-V / Level Zero / SYCL Sub-Groups / OpenVINO Hetero Runtime | `XeDim3 (gridDim, blockDim), XeStream (asynchrónny prúd)` | 64B-zarovnaná USM nulová kópia, `SLM <= 64 KB/Subslice`, `K_tile < 449` proti rozliatiu 128 GRF registrov. |
| `HRI_P4_OPENVINO_ONEAPI_EXECUTION` | OpenVINO C++/Python `ov.Tensor(shared_memory=True)` + `oneDNN` + `oneTBB` | `UMADeviceAssetBuffer, InferRequest Async Ping-Pong Queue` | Sanitizácia IEEE-754 denormálov na hranici (18.05x-37.37x ochrana) a pripnutie na 4 fyzické jadrá. |
| `HRI_P5_PYTHON_XERJ_SCHEMA_ORCHESTRATOR` | Python 3 + JSON Schema Draft 2020-12 + XERJ deterministický AST prekladač | `UniversalKnowledgeHierarchyKernel, UnifiedBackendUiAccessGateway (UAP_01 read-only)` | Nulová mutácia surových registrov z UI; validácia schém pred každým spustením úlohy. |

## 2. Ekvivalencia NVIDIA CUDA <-> Intel Xe Stream Procesory, AI Jadrá & NPU
| Pojem v CUDA | Ekvivalent v Krystal Stacku (GPU Stream / AI Matrix / NPU) | Primitívum API |
| :--- | :--- | :--- |
| Streaming Multiprocessor (`SM`) | Intel `Xe Subslice / Xe-Core` (`6 Subslices` na Iris Xe, po `16 EUs`) | NPU `Neural Compute Tile` | `xeGetDeviceProperties()['subslices_xe_cores']` |
| CUDA Cores (`FP32 / INT32 ALUs`) | Intel `Stream Processors` (`96 EUs x 8 FP32 = 768 FP32` prúdových jednotiek, `1536 FP16`, `3072 INT8`) | `TARGET_GPU_STREAM_PROCESSORS` (`SIMD8 / SIMD16 / SIMD32`) |
| Tensor Cores (`WMMA / MMA`) | AI maticové jadrá (`Intel XMX` / `DP4A` / `AVX-512 VNNI` INT8/BF16 dlaždice) | `TARGET_AI_MATRIX_CORES` (automatické dlaždicovanie `K_tile = 256 < 449`) |
| DLA / Low-Power Inference Engine | Dedikované `NPU` (`Intel AI Boost` / `OpenVINO NPU` / `AXI4 NPU` s 1-6 W odberom) | `TARGET_NPU_NEURAL_TILES` (`KR_PROFILE_BALANCED_NPU`, automatický GPU/VNNI fallback) |
| Warp (`32` pevne zviazaných vlákien) | Dynamická `EU Subgroup` s voliteľnou šírkou `SIMD8`, `SIMD16` alebo `SIMD32` podľa tlaku na `128 GRF` registrov | `xeSelectOptimalSubgroupWidth(grf_per_thread, reduction_k)` |
| `__shared__` pamäť SM (`48-100 KB`) | `SLM (Shared Local Memory)` — `64 KB` na každý Subslice (`384 KB` spolu na čipe) | `slm_shared_bytes <= 65536` v `HybridRustXeLaunchDescriptor128B` |
| `cudaStream_t` + `cudaMemcpyAsync` (Pinned) | Asynchrónny `XeStream` (`InferRequest.start_async`) nad `64B` zero-copy `kuMallocUMA` + `BAR0` Doorbell | `UnifiedStreamAiNpuAccelerator.launch_async_dual_stream(...)` |

## 3. Nameraný prínos 2-prúdového asynchrónneho `XeStream` prekrývania (`LOCAL_MEASURED`)
- **Zariadenie:** `GPU` (`8` dávok `[128x256]` dvojvrstvového GEMM)
- **Sekvenčný 1-stream beh:** `61.66 ms`
- **Asynchrónny 2-stream `XeStream` ping-pong:** **`36.719 ms`** (**`1.68x` rýchlejšie**).
