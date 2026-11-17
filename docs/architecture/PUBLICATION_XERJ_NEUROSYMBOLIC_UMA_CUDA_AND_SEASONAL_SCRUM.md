# Sovereign Neurosymbolic Execution, Krystal-UMA-CUDA (`ku*`) & Seasonal Scrum Prompting
### Architectural Publication & Reference Contribution Aligned with XERJ (`xerj-org/xerj`)

> **Abstract:** We present a hardware-grounded neurosymbolic execution architecture uniting deterministic AST/schema query compilation (aligned with **Ivan Novikov's XERJ** design philosophy of eliminating raw-token RAG bloat and unverifiable agent outputs) with **`Krystal-UMA-CUDA` (`ku*`)** — a CUDA-inspired zero-copy memory, defensive Vulkan endpoint advisor, zero-touch kernel operation injector, and `192 B` (`64 B WNTC + 128 B KPKT`) Hypervisor HyperTransport stack on Intel Willow Cove AVX-512 + Iris Xe UMA (`VITAL_MAX_HP == 6`, `UMA < 0.78`). Finally, we formalize a full-season Scrum Retrospective-to-Prompt Knowledge Base (`20` SHA-256-attested records) governing continuous engineering sprints.

---

## 1. Solving the XERJ Neurosymbolic Grounding Challenge
- **Organization & Founder Context:** `XERJ (github.com/xerj-org/xerj · xerj.org)` (founded by **Ivan Novikov**).
- **Core Problem Addressed:** Classical LLM/RAG agent pipelines waste hundreds to thousands of tokens on raw unstructured text, suffer non-deterministic hallucinations, and lack hardware-verified provenance when actuating tools or code.
- **Unique Krystal + XERJ Solution:** Couple XERJ-style deterministic AST/Query-1 symbolic compilation (< 2 ms, 0 LLM tokens on fast-path, 999 -> 39 token prefill compression) directly to the Krystal-UMA-CUDA (`ku*`) hardware runtime, ConvolutionalHypervisorStepBridge, and 192B (64B WNTC + 128B KPKT) HyperTransport packets.
- **Repository Contribution Map:**
  - `README.md (XERJ Corpus Hub Selected Apache-2.0 Reference Implementation)`
  - `krystal_kernel/slovak_llm_adapter.py (SHA-256 weight fingerprinting & Win32 Memory Headroom Guard)`
  - `krystal_kernel/double_time_coefficient_emulator.py (ARCH_EL_01_XERJ_QUERY1_AST 999->39 tok compiler)`
  - `projects/trigonometric_hybrid_platform_framework/krystal_uma_cuda_vulkan_library.py (ku* UMA-CUDA & Vulkan defense)`
  - `projects/trigonometric_hybrid_platform_framework/engine_point_and_function_parser.py (AST catalog over all modules)`

---

## 2. `Krystal-UMA-CUDA` (`ku*`), Defensive Vulkan Endpoints & HyperTransport Methodics
- **Bound Repository Schemas (`schemas/`):** `10` JSON schemas coordinated.
- **Bound Benchmark Scripts (`scripts/`):** `16` benchmark scripts coordinated.
- **Live Defended `K=512` MatMul Proof (`kuLaunchMatMulDefended`):** Executed `2` tiles (`K < 449` law), `64 B` cache-line aligned = `True`, `192 B` HyperTransport verified = `True`.
- **Zero-Touch Kernel Injection (`@kuInjectSimplifiedKernelOp`):** Wrapped existing function with `0` source delta (`result = 14.0`, `192 B` receipt = `True`).

| Methodic ID | Title | Constitutional Rule |
| :--- | :--- | :--- |
| `HHT_M1_CACHELINE_PACKET_QUANTA` | **Exact 64B Cache-Line Multiple Framing (64B WNTC + 128B KPKT = 192B)** | Never split a hypervisor control/telemetry descriptor across unaligned page boundaries; pack into 3 x 64B cache lines. |
| `HHT_M2_VBLANK_CREDIT_FLOW_CONTROL` | **Chronometric VBlank Phase-Shadow Credit Window** | Grant bulk DMA/HyperTransport credits strictly inside the VBlank shadow (tau_spec) so active 60/120Hz scanout sees zero bus contention. |
| `HHT_M3_HOST_COHERENT_ZERO_COPY_SLAB` | **Pinned 64B-Aligned UMA Views Over ov.Tensor & Vulkan Host-Coherent Memory** | Bind one 64B-aligned host allocation across NumPy, OpenVINO CPU/GPU, and Vulkan endpoints with 0 memcpy bytes. |
| `HHT_M4_INSTANT_CREDIT_REVOCATION` | **0-Frame Credit Revocation & WSL2 Quota Trim on UMA >= 0.78** | Revoke GPU/DMA HyperTransport credits in 0 frames when UMA >= 0.78, trim WSL2 RAM 10->8 GB, and require 12 clean samples to restore. |

---

## 3. Seasonal Scrum Management: Problem -> Rule -> Next-Sprint Prompt Compiler

### 3.1 Permanent Error-Prevention Rules Extracted from Past Sprints (`W1..W6`)
| Problem ID | Observed Failure / Bottleneck | Permanent Prevention Rule |
| :--- | :--- | :--- |
| `PROB_01_AST_CUBIC_STALL` | 23m48s (1428.94 s) pytest stall from redundant O(M³) AST parsing & repeated OpenCL init. | EPR_01: Parse module ASTs once per pass and memoize shared OpenVINO Core (`126.4x` speedup -> 11.30 s). |
| `PROB_02_OPENVINO_SDPA_2D_CRASH` | Windows C++ access violation in OpenVINO 2026.4.1 SDPA matcher on unbatched 2D QK^T + CPU CACHE_DIR. | EPR_02: Scope CACHE_DIR strictly to static GPU blobs and compile contiguous FFN+Head opset13 graphs. |
| `PROB_03_X86_SUBNORMAL_MICROCODE_ASSIST` | 14.00x (NumPy) and 37.37x (OpenVINO CPU) slowdown on IEEE-754 subnormal float32 values. | EPR_03: Enforce `canonicalize_float32` / `kuMallocUMA` subnormal flush at every ingress boundary. |
| `PROB_04_DOUBLE_TIME_128B_OVERLAP_AND_120HZ_OVERRUN` | 64B BUF_00/01 truncated 128B frames and k=4 (9.80 ms) overran 120 Hz (6.713 ms) shadow window. | EPR_04: Align BUF_00 (0x000..0x07F) & BUF_01 (0x080..0x0FF) to 128B and clamp k <= floor(shadow/2.45). |
| `PROB_05_PRODUCTION_BARE_ASSERTS` | 6 bare `assert` statements in production modules could be stripped under `python -O`. | EPR_05: Ban `ast.Assert` in production packages via CI test; raise typed `ContractViolation`. |

### 3.2 Next-Season Sprint Goals & Auto-Compiled Prompts (`SPRINT_S7..SPRINT_S10`)
| Sprint | Goal | Definition of Done (`DoD`) | Compiled Guiding Prompt |
| :--- | :--- | :--- | :--- |
| **`SPRINT_S7_RUST_FFI_AND_WSL2_192B`** | Compile Rust `cdylib` for 37-bit `u64` hex bitboard (`DF_05`) and wire `192B WNTC+KPKT` (`DF_01`) into `wsl_bridge/bridge_daemon.py`. | >= 10x move-check speedup over Python & verified 1,000-packet 192B CRC32+SHA256 loopback. | `SPRINT S7 PROMPT: Lock VITAL_MAX_HP==6. Implement C-ABI 128B/192B ring exchange between combinatorial_derivative_functions_engine.py, wsl_bridge/bridge_daemon.py, and krystal-bitboard/src/edge.rs. Run CDT benchmark (n>=30, median/p95/stdev) and reject if FFI overhead > 0.05 ms.` |
| **`SPRINT_S8_VULKAN_HOST_COHERENT_AND_GODOT_ANIM`** | Bind `kuMallocUMA` 64B slabs + `.tres` animation tracks (`DF_03`) into `godot_project/scenes/HoloStage.tscn` and Vulkan endpoints. | 10/20 active Godot layers at UMA < 0.78, automatic 0x1f clamp at UMA >= 0.78, 0.00 ms VSync overrun. | `SPRINT S8 PROMPT: Connect krystal_uma_cuda_vulkan_library.py (`kuAdviseAndDefend`) and `krystal_siege_anim_track.tres` into godot_project/scenes/*.tscn. Verify 0-frame GPU veto when UMA >= 0.78.` |
| **`SPRINT_S9_MONOLITH_DECOMPOSITION_AND_TYPED_IO`** | Decompose the 9 legacy files > 500 LOC through `dep_bridge_*` / `kuInjectSimplifiedKernelOp` and replace 10 broad `except Exception` blocks. | All new/refactored modules <= 450 LOC, 0 bare asserts, 100% typed I/O & OpenCL exception handlers. | `SPRINT S9 PROMPT: Refactor modules > 500 LOC using `@kuInjectSimplifiedKernelOp` without changing public signatures. Keep full pytest regression green and verify under `python -O`.` |
| **`SPRINT_S10_XERJ_CORPUS_BENCHMARK_AND_GEMMA_ADAPTERS`** | Run full local Gemma 2B/12B adapter evaluation on `gemma_neurosymbolic_finetune_sets.jsonl` + online calibration over `bot_priority_calibration_trace.jsonl`. | Zero train/eval leakage, ECE <= 0.05 on held-out test split, 100% `[F_...]` citation grounding, zero OpenAI calls. | `SPRINT S10 PROMPT: Train and temperature-calibrate local OpenVINO/Gemma adapters on disjoint sets SET_01..04. Report Accuracy, Macro-F1, Brier, and ECE with 95% bootstrap CIs for XERJ Corpus Hub publication.` |

---

## 4. Epistemic Scope & Limitations
- **Measured Locally (`LOCAL_MEASURED`):** x86 cache/DRAM hierarchy (`X01`), AVX-512 physical vs SMT jitter (`X02`), IEEE-754 subnormal penalties (`X03`), `64 B` alignment & zero-copy `ov.Tensor` (`X04`), and `128 B / 192 B` binary speedups (`X05`).
- **Verified Architectural Contracts (`VERIFIED_WITH_SCOPE`):** `StepExecutionGateProxy`, `kuAdviseAndDefend` Vulkan GPU veto at `UMA >= 0.78`, `Double-Time` 120 Hz clamping, and `0` bare `assert`s under `python -O`.
- **External Gates (`UNKNOWN` until target hardware/OS run):** Ring-0 MSR/RAPL hardware Joule counters on Windows and live `AF_VSOCK` inside a booted Linux guest.
