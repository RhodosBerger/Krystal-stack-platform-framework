# Project Audit — Heavy Progress & Compute Patterns

> **Provenance.** Every number in the tables below is produced by [`src/python/project_intelligence.py`](../src/python/project_intelligence.py)
> (stdlib-only static analysis of the working tree + a few earlier benchmarks that are named explicitly).
> Re-generate live with `python src/python/project_intelligence.py overview` or `GET /api/project/overview`
> (see [PROJECT_INTELLIGENCE_API.md](PROJECT_INTELLIGENCE_API.md)). Snapshot date: **2026-10-03**.
> Nothing here is a projection unless it is labelled *target*.

---

## 1. Where the heavy progress actually is

Progress is concentrated in **CPU-side, pure-Python, dependency-free engines that have real tests**:

| Subsystem | Code LOC | Discoverable tests | Runtime-verified here | Readiness* |
|---|---:|---:|:---:|---:|
| Symplectic Hamiltonian organism | 297 | 6 | ✅ | 1.00 |
| Krystal-Lang compiler & topological VM | 869 | 6 | ✅ | 0.88 |
| Antigravity prompt engine & AR mirror | 804 | 5 | ✅ | 0.85 |
| OpenWorld semantic compiler & renderer | 1 154 | 5 | ✅ | 0.77 |
| Vulkan compute driver (ctypes) | 409 | 4 | ❌ no dispatch | 0.69 |
| Janet sub-project | 1 307 | 9 | ❌ Janet not installed | 0.58 |
| Neural ASCII engine & compositor | 1 271 | 0 | ❌ | 0.30 |
| Localhost Mission Control (HTTP+SSE) | 3 993 | 0 | ❌ | 0.30 |
| Mimicry compositor | 1 307 | 0 | ❌ | 0.30 |
| NPU / OpenVINO / ONNX | 4 958 | 0 | ❌ runtime absent | 0.30 |
| Godot project & shader | 488 | 0 | ❌ Godot absent | 0.00 |
| WSL2 bridge | 270 | 0 | ❌ | 0.00 |
| Rust crates | 5 131 | 0 | ❌ cargo absent | 0.00 |
| Industrial CNC (legacy) | 77 033 | 0 | ❌ | 0.30 |

\* `readiness = 0.4·min(1, tests per 100 LOC) + 0.3·docs present + 0.3·runtime-verified`. It is a triage heuristic, not a quality score.
"Discoverable tests" counts `def test_` methods that `unittest discover` can find — script-style checks are deliberately not counted.

**Reading the table.** The recent heavy progress (cyclic organism → Krystal-Lang → AR/OpenWorld) is well covered.
The *largest* code volumes (CNC legacy, Rust, NPU) are the least verified in this environment: the toolchains are absent
(`cargo`, `rustc`, `janet`, `godot`, `numpy`, `onnxruntime`, `openvino` all missing on the host; `git` and `wsl` present).

Test suite at snapshot: **47 passing** (`python -m unittest discover tests`).

---

## 2. Compute-pattern catalogue

Detected from source signatures (the analyzer's own file is excluded from evidence). `status` is demoted when the
pattern cannot be proven on this machine.

| Pattern | Complexity | Heat | Status | Target tier | Honest note |
|---|---|---|---|---|---|
| SDF sphere tracing | O(W·H·K·C_sdf) | very high | **implemented, CPU only** | GPU / vectorized | Dominant cost. Measured **~37–46 ms/frame at 96×40** in pure Python (≈22 FPS ceiling). |
| Symplectic velocity-Verlet | O(N), N=4 | low | implemented | CPU | Measured max relative energy drift **4.34·10⁻⁴ over 10 000 steps** (γ=0, β=0, no economic forcing, dt=0.033, H₀=40.79). |
| Fixed-capacity ring-buffer channels | O(1) | medium | implemented | CPU / Rust | **4.39×** faster than `queue.Queue` in the earlier micro-benchmark; GIL means this is *not* lock-free. |
| Queue-partitioned bytecode VM | O(pipelines·batch) | medium | implemented | CPU | Krystal-Lang VM. |
| Visual-entropy back-pressure governor | O(W·H) | medium | implemented | CPU | Closed-loop FPS/entropy governor in the hub. |
| Multi-octave fBm terrain | O(W·H·octaves) | high | implemented | vectorized / GPU | Pure `math`; no SIMD. |
| Cooperative chunk streaming (fibers) | O(chunks) | low | **unverified** | CPU | Janet code never executed on a real Janet. |
| Dual-SSBO glyph+RGB readback | O(W·H) bytes (5 B/px) | medium | **emulated** | GPU | A buffer *contract*; no GPU buffers exist yet. |
| SIMD / vector math | O(N/lanes) | high | implemented (plain `math`) | Rust / numpy | `fast_math.py` uses `math`+`array`. Docs naming an AVX2 / "krystal-math-simd" crate describe a target; that crate does not exist. |
| INT8 NPU inference offload | O(L·M²) | high | **unverified** | NPU | `phantom_net.onnx` is never loaded; no runtime installed. |

### 2.1 What the hot path really is

```
SSE frame (30 Hz target)
 └─ engine_worker_loop (pure-Python thread)
     └─ VulkanComputeDriver.execute_raymarch   ← CPU loop; ~40 ms ⇒ ~22 FPS ceiling
         ├─ real:   vkCreateInstance, device enumeration (Intel Iris Xe 0x8086/0x9a49), compute queue family
         └─ absent: vkCreateDevice, SPIR-V, pipeline, SSBO, vkCmdDispatch
```

The driver is a **device probe plus a CPU reference kernel**. `/api/status` therefore reports
`vulkan.accelerated=false`, `compute_backend="CPU_EMULATION"`, `device_detected=true`.
`readback_mb_s` is host-list throughput, not PCIe bandwidth.

---

## 3. Corrections made to earlier claims

Earlier iterations of this project claimed GPU acceleration, "<1.5 µs readback", "60–120+ FPS" and "PCIe DMA".
None were measured and they were false. The following were corrected in code/UI; the rest is tracked by the
`claims-vs-measurements-audit` priority:

| Item | Was | Now |
|---|---|---|
| `VulkanComputeDriver` docstring/telemetry | "GPU accelerated" | `device_detected`, `real_gpu_dispatch=False`, `compute_backend=CPU_EMULATION` |
| Mission Control Vulkan card | "GPU ACCELERATED" | "DETECTED · CPU KERNEL", "HOST BUFFER RATE (SIMULATED)" |
| Janet sources | `math/max`, `math/min`, `float` | replaced (these symbols do not exist in the Janet Core API) |
| Analyzer itself | docstring mention of `vkCmdDispatch` made Vulkan look "verified" | detectors now read AST identifiers/strings, never docstrings or comments (regression-tested) |

Still open: [`DEEP_PERFORMANCE_OPTIMIZATION_AND_CORE_ARCHITECTURE.md`](research/DEEP_PERFORMANCE_OPTIMIZATION_AND_CORE_ARCHITECTURE.md) §4.5
and [`CYCLIC_THEORY_AND_ALTERNATIVE_ARCHITECTURAL_SCHEMAS.md`](research/CYCLIC_THEORY_AND_ALTERNATIVE_ARCHITECTURAL_SCHEMAS.md) §5 still contain unmeasured figures.

---

## 4. Prioritisation model

```
gap    = min(1, Σ severity of failing detectors)         # measured from the repo
score  = 100 · (0.35·impact + 0.35·gap + 0.15·compute_heat + 0.15·leverage)
value  = score / effort_points                            # S=1, M=3, L=8
```

`gap` is **measured** (each detector inspects files/toolchain). `impact`, `compute_heat`, `leverage`, effort size are
**editorial weights** declared next to each candidate in the source — reviewable, not hidden. `sort=score` answers
"what matters most"; `sort=value` answers "what is the best return per unit of effort".

### 4.1 Current ranking (open items)

| # | Priority | Score | Value | Effort | Category | Expertise tracks |
|--:|---|---:|---:|:--:|---|---|
| 1 | `vulkan-real-dispatch` — logical device → SPIR-V → pipeline → SSBO → `vkCmdDispatch` | 96.0 | 12.0 | L | compute | gpu-compute, systems-ffi, perf-engineering |
| 2 | `python-hotpath-vectorize` — cut the pure-Python raymarch hot path | 87.2 | **87.2** | S | compute | numerical-python, perf-engineering |
| 3 | `rust-native-kernels` — native tier the docs depend on | 74.2 | 9.3 | L | compute | rust-native, perf-engineering |
| 4 | `npu-path-verification` — prove NPU/ONNX end-to-end | 74.0 | 24.7 | M | compute | npu-inference |
| 5 | `mimicry-ar-test-coverage` — bring mimicry & AR into unittest | 68.0 | 68.0 | S | quality | qa-automation, procedural-geometry |
| 6 | `ci-reality-check` — CI targets paths that don't exist | 68.0 | 68.0 | S | devops | devops-release, qa-automation |
| 7 | `roadmap-phase1-5` — Windows installer (roadmap F1) | 67.0 | 8.4 | L | roadmap | devops-release |
| 8 | `repo-hygiene-untrack-artifacts` — stop tracking `venv/`, `target/`, logs | 63.0 | 63.0 | S | devops | devops-release |
| 9 | `godot-verification` — headless Godot check | 61.5 | 20.5 | M | integration | game-engines |
| 10 | `harden-localhost-api` — wildcard CORS + file-writing POSTs | 60.5 | 60.5 | S | security | web-security |
| 11 | `claims-vs-measurements-audit` — reconcile docs with measurements | 60.0 | 60.0 | S | quality | perf-engineering, qa-automation |
| 12 | `janet-runtime-verify` — run Janet on a real interpreter | 51.2 | 51.2 | S | language | lisp-functional, language-runtime |

**Quick wins (best value):** #2 (87.2), #5/#6 (68.0), #8 (63.0), #10 (60.5).
**Big bets (highest score, L effort):** #1 Vulkan dispatch, #3 Rust tier.

Items flagged `requires_user_approval` (currently `repo-hygiene-untrack-artifacts`, which implies `git rm --cached`) are
**never executed automatically**; the API only reports them.

### 4.2 Repo-hygiene finding (needs a decision)

`git ls-files` lists **25 522** tracked files, including a Linux `venv/` (CPython 3.12 `.so` files),
`target/debug/*.rlib` (up to 46 MB) and a 25.8 MB `backend_run_log_stable.txt`; `.gitignore` has no venv/target/dist/log entries.

---

## 5. Expertise map

`staffing_signal` is derived from the readiness of the subsystems a track covers:
`in_house_strong` (≥ 0.7) · `in_house_partial` (0.4–0.7) · `gap` (< 0.4).
`demand_score` is the summed score of open priorities that need the track.

| Track | Signal | Demand | Readiness | Suggested engagement |
|---|---|---:|---:|---|
| Performance engineering & benchmarking | in_house_strong | 317.4 | 0.78 | extend in-house; design review before large changes |
| DevOps, CI & release engineering | **gap** | 198.0 | 0.00 | dedicated spike or external specialist; self-study first |
| Test automation & verification | in_house_strong | 196.0 | 0.88 | extend in-house |
| GPU compute & Vulkan | in_house_partial | 96.0 | 0.49 | pair-build session; spike with measurable exit criteria |
| Native FFI & systems (ctypes, ABI) | in_house_strong | 96.0 | 0.69 | extend in-house |
| Numerical Python & vectorization | in_house_partial | 87.2 | 0.54 | pair-build session |
| Rust native kernels & PyO3 | **gap** | 74.2 | 0.00 | dedicated spike or specialist |
| NPU / OpenVINO / ONNX | **gap** | 74.0 | 0.30 | dedicated spike or specialist |
| Procedural generation, SDF & geometry | in_house_partial | 68.0 | 0.64 | pair-build session |
| Godot & game-engine integration | **gap** | 61.5 | 0.00 | dedicated spike or specialist |
| Web/API security for local daemons | **gap** | 60.5 | 0.30 | dedicated spike or specialist |
| Compilers, bytecode & VMs | in_house_strong | 51.2 | 0.73 | extend in-house |
| Janet / Lisp functional engines | in_house_partial | 51.2 | 0.58 | pair-build session |
| Hamiltonian / cyclic control theory | in_house_strong | 0 | 1.00 | extend in-house |
| Industrial cognitive manufacturing (CNC) | gap | 0 | 0.30 | spike / specialist (no open demand) |
| Product & platform architecture | gap | 0 | 0.30 | spike / specialist (no open demand) |

**Takeaway.** Demand concentrates on three *gap* tracks — **DevOps/CI**, **Rust**, **NPU** — plus **web security**
and **Godot**. The in-house strengths (performance, QA, FFI, language runtime, dynamical systems) are exactly what the
two top compute priorities need, so those two can start without outside help.

---

## 6. Recommended sequence

1. **Quick, measurable, no new tooling:** `claims-vs-measurements-audit` → `ci-reality-check` → `mimicry-ar-test-coverage` → `harden-localhost-api`.
2. **Decision gate A (numpy):** allow an *optional* numpy kernel in a Windows venv to attack `python-hotpath-vectorize`. Keep stdlib as the default so "zero-dependency" stays true. Record ms/frame in a JSON artifact **before** quoting any speedup.
3. **Decision gate B (SPIR-V source):** hand-assembled module (zero toolchain) vs. installing the Vulkan SDK (`glslc`) for `vulkan-real-dispatch`. Keep `execute_raymarch()` as the correctness oracle.
4. **Decision gate C (repo hygiene):** approve or reject the `git rm --cached` plan for `venv/`, `target/`, logs.
5. **Toolchain installs** (Janet, Rust, Godot) only with explicit approval; each unlocks a currently "unverified" subsystem.
