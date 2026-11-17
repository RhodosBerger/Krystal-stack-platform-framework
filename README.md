# KRYSTAL STACK: Heterogeneous AI Runtime, OpenVINO & Vulkan Compute Orchestrator 🌌
### Heterogeneous Compute (CPU · GPU · NPU) · Intel OpenVINO & Vulkan Compute · 128B Zero-Copy UMA IPC · Deterministic AI Safety Gate · Digital Twin Suite

<div align="center">

[![Tests](https://img.shields.io/badge/Unit%20Tests-58%2F58%20Passing%20(0.026s)-22c55e?style=for-the-badge&logo=pytest)](tests/)
[![Zero-Copy Protocol](https://img.shields.io/badge/Protocol-128B%20Zero--Copy%20Aligned-06b6d4?style=for-the-badge&logo=rust)](krystal_kernel/tri_phase_dosing_controller.py)
[![Grid Standard](https://img.shields.io/badge/Grid%20Standard-EN%2050160%20Compliant-f59e0b?style=for-the-badge)](artifacts/benchmarks/VPP_GRID_BROKER_AND_SWITCHED_MATRIX_SPEC.md)
[![Inference](https://img.shields.io/badge/Inference-OpenVINO%20%7C%20DirectML%20%7C%20Vulkan-ea580c?style=for-the-badge&logo=intel)](krystal_kernel/)
[![State Invariant](https://img.shields.io/badge/Constitutional%20Rule-VITAL__MAX__HP%20%3D%3D%206-ef4444?style=for-the-badge)](artifacts/benchmarks/)
[![XERJ Hub](https://img.shields.io/badge/XERJ%20Corpus%20Hub-Selected%20Reference-8b5cf6?style=for-the-badge)](krystal_kernel/slovak_llm_adapter.py)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Du%C5%A1an%20Kopeck%C3%BD-0a66c2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/du%C5%A1an-kopeck%C3%BD-bbba33301/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge)](LICENSE)

---

### 🌐 Multilingual Documentation Hub / Jazykové Mutácie
[ **🇬🇧 English (Primary)** ](README.md) &nbsp;|&nbsp; [ **🇸🇰 Slovenčina** ](docs/i18n/sk/README.md) &nbsp;|&nbsp; [ **🇩🇪 Deutsch** ](docs/i18n/de/README.md) &nbsp;|&nbsp; [ **🇨🇿 Čeština** ](docs/i18n/cs/README.md)

---

</div>

> **Project Description & Search Summary:**  
> **Krystal Stack** is an open-source **Heterogeneous AI Execution & Hardware Acceleration Framework** for **x86_64** and **ARM64** systems. Sitting above **Intel OpenVINO**, **Vulkan Compute**, **DirectML**, and **Keras 3**, it eliminates the **Unified Memory (UMA) Bandwidth Wall** using **128-byte lock-free Zero-Copy SPSC ring buffers**, **Chronometric Shadow Multiplexing**, and a formal **Epistemic Invariant Gate** (`VITAL_MAX_HP == 6`, guaranteeing $\Delta S = 0$ on invalid transitions). Proven with a **20.35x Vulkan GPU speedup**, **497.66 MB/s saved memory bandwidth**, and selected for the **XERJ Corpus Hub** as a production reference for AI coding agents.

---

## 🏆 External Validation & OpenVINO / Vulkan Reference Highlights

* **Selected for XERJ Corpus Hub (Apache-2.0 Reference Implementation):**
  Our OpenVINO tensor substrate with cryptographic SHA-256 weight fingerprinting and Win32 OS Memory Headroom Guard ([`krystal_kernel/slovak_llm_adapter.py`](krystal_kernel/slovak_llm_adapter.py)) was independently selected by **XERJ** as a worked reference architecture for coding agents.
* **Intel OpenVINO Innovator Validation & Positioning Dossier:**
  Full architectural breakdown of the Adaptive Execution Processor (AEP) closed-loop orchestration above OpenVINO Runtime: [`docs/openvino/OPENVINO_INNOVATOR_VALIDATION_POSITIONING.md`](docs/openvino/OPENVINO_INNOVATOR_VALIDATION_POSITIONING.md).
* **Verified 20.35x Vulkan Compute UMA Speedup & Zero-Copy Telemetry:**
  Hardware benchmark on Intel Iris Xe UMA (`0x8086:0x9a49`) reducing frame latency from `46.99 ms` (CPU baseline) to **`2.31 ms`** (`433.13 FPS`, `0.321 ms` kernel wait) and saving **`497.66 MB/s`** of memory bandwidth at 60 FPS: [`artifacts/VULKAN_IRIS_XE_BENCHMARK_RESULTS.json`](artifacts/VULKAN_IRIS_XE_BENCHMARK_RESULTS.json) & [`artifacts/GPU_ZERO_COPY_READBACK_BENCHMARK.json`](artifacts/GPU_ZERO_COPY_READBACK_BENCHMARK.json).
* **Sovereign OpenVINO Companion v2, Manifest Knowledge Layer, Kernel E-Shop & Blender/V-Ray UI Pipeline:**
  24+ deterministic runtime, Vulkan/USM 3D grid, binary matrix, C++20 Zero-GC, Morse hardware parser, contractive oscillation stabilization, ISA cache-tier portability, Kernel E-Shop (`/kernel_eshop`, `/api/v1/runtime/kernel_eshop`), USB `swap.img` / `.kryf` compiler & RT-AX58U V2 profile (`/api/v1/runtime/adaptive_vision_usb_swap`), and Blender + V-Ray / Airport B2B endpoints served via `/blender_vray_ui` and `/api/v1/runtime/blender_vray_sdk_pipeline`:
  - [`projects/sovereign_openvino_companion_v2/README.md`](projects/sovereign_openvino_companion_v2/README.md)
  - [`docs/architecture/KRYSTAL_UNIVERSAL_AGENT_KERNEL_MANIFEST.md`](docs/architecture/KRYSTAL_UNIVERSAL_AGENT_KERNEL_MANIFEST.md)
  - [`docs/architecture/KRYSTAL_MANIFEST_PROCESS_SCHEMAS_AND_CHIPSET_KNOWLEDGE_LAYER.md`](docs/architecture/KRYSTAL_MANIFEST_PROCESS_SCHEMAS_AND_CHIPSET_KNOWLEDGE_LAYER.md)
  - [`docs/architecture/VULKAN_ZERO_COPY_SOCKET_GEARBOX_AND_FOUR_SCALE_COMPONENT_MANUAL.md`](docs/architecture/VULKAN_ZERO_COPY_SOCKET_GEARBOX_AND_FOUR_SCALE_COMPONENT_MANUAL.md)
* **Trigonometric Hybrid Platform, `Krystal-UMA-CUDA` (`ku*`), XERJ Publication, OpenVINO Add-On Kernel & Lightweight Rust+Vulkan Drivers:**
  Unified XERJ zero-token AST synthesis, `kuMallocUMA` / `kuAdviseAndDefend` (`UMA < 0.78`, `K < 449`), 5-phase OpenVINO telemetry add-on kernel (`PH_D1..D5`), `O(1)` 64-bit bitmap slab & 4T sharded `128 B` SPSC connection logger, symmetric oneAPI + Conda LLM presets (`configs/conda_krystal_oneapi_llm_presets.yaml`), and Windows/Unix lightweight Rust+Vulkan drivers:
  - [`projects/trigonometric_hybrid_platform_framework/README.md`](projects/trigonometric_hybrid_platform_framework/README.md)
  - [`docs/architecture/PUBLICATION_XERJ_NEUROSYMBOLIC_UMA_CUDA_AND_SEASONAL_SCRUM.md`](docs/architecture/PUBLICATION_XERJ_NEUROSYMBOLIC_UMA_CUDA_AND_SEASONAL_SCRUM.md)
  - [`docs/architecture/UNIX_RUST_VULKAN_LIGHTWEIGHT_DRIVERS_AND_KERNEL_DATA_STRUCTURES.md`](docs/architecture/UNIX_RUST_VULKAN_LIGHTWEIGHT_DRIVERS_AND_KERNEL_DATA_STRUCTURES.md)
  - [`docs/architecture/STOCK_VS_KRYSTAL_PLATFORM_BENCHMARK_COMPARISON.md`](docs/architecture/STOCK_VS_KRYSTAL_PLATFORM_BENCHMARK_COMPARISON.md)
  - [`reports/README_ALL_INVENTIONS_AND_MONETIZATION_REPORT.md`](reports/README_ALL_INVENTIONS_AND_MONETIZATION_REPORT.md)
* **Community Offering — Universal `16 KB mmap` MMIO Kernel, Shadow-Price Arbitration Engine, Core-Wake Strategies (`CORE_WAKE_01..06`), OpenVINO MT Presets & Autonomous OS/GPU Opportunity Bot:**
  Delivers a **`245.97x` lower-overhead** `BAR0` telemetry hot-path (`133.01 ns` vs `32,715.62 ns` driver query), **`59.18x` faster** in-memory hardware hierarchy lookup (`9` nodes / `375` repo artifacts absorbed), **`2.90x` CPU Lookahead Whisperer** speedup, **`+11.17 dB` PSNR** visual quality gain within the `120 Hz` (`8.333 ms`) frame budget (`2.098 ms`), CUDA-style Intel Stream/AI/NPU layer (`768` FP32 lanes, `64 KB SLM` Flash-Attention), Windows 11 DirectX 12 (`64 KB` shared heap) & Vulkan backward-compatibility layers, and `krystal_bot` `/overseer`, `/arbitrate`, and `/opportunities` autonomous governance:
  - [`docs/architecture/COMMUNITY_OFFERING_ARBITRATION_AND_DRIVER_OPPORTUNITIES.md`](docs/architecture/COMMUNITY_OFFERING_ARBITRATION_AND_DRIVER_OPPORTUNITIES.md)
  - [`docs/architecture/UNIVERSAL_MMIO_DEVICE_ADDRESS_KERNEL_SPEC.md`](docs/architecture/UNIVERSAL_MMIO_DEVICE_ADDRESS_KERNEL_SPEC.md)
  - [`docs/architecture/HYBRID_RUST_LINUX_DRIVERS_AND_INTEL_XE_CUDA_STREAMS.md`](docs/architecture/HYBRID_RUST_LINUX_DRIVERS_AND_INTEL_XE_CUDA_STREAMS.md)
  - [`configs/openvino_multithreading_presets.yaml`](configs/openvino_multithreading_presets.yaml) & [`krystal_bot/integrated_efficiency_overseer.py`](krystal_bot/integrated_efficiency_overseer.py)
  - [`docs/research/OBSERVABLE_STUDY_SYNTHETIC_ML_THEORY_ENGINEERING_REPORT.md`](docs/research/OBSERVABLE_STUDY_SYNTHETIC_ML_THEORY_ENGINEERING_REPORT.md)
* **⚖️ Proprietary Extensions & Fair Commercial / Personal Licensing Notice (All Rights Reserved — Dušan Kopecký):**
  All proprietary Krystal Stack extensions (`Intel Adrenalin Panel + Krystal CMS`, `16 KB mmap` MMIO Kernel, Shadow-Price Resource Arbitration, and Driver-Assist Layers) have **All Rights Reserved by codebase owner Dušan Kopecký (`dusan.kopecky0101@gmail.com` · `@RhodosBerger`)**. Personal use requires prior owner consent; commercial deployment requires contacting the owner and submitting a commercial licensing proposal (with fair high-ROI tiers starting from Indie `€19/mo` up to value-shared Enterprise agreements).
  * **Authorized Partner Addendum (`XERJ` / `xerj.org`):** **Ivan Novikov** is expressly granted permission by codebase owner **Dušan Kopecký** to use the newly delivered Krystal Stack features—specifically highlighting the **zero-copy shared-memory tensor integrations (`kuMallocUMA`, `ov.Tensor` `GPU_USM_MEMORY`, `64 B / 64 KB` boundary alignment, IEEE-754 subnormal `FTZ/DAZ` guards) and data-driven OpenVINO configurations (`OV_MT_01..05` multithreading presets, `HETERO` / `AUTO` accelerator routing, and warm blob caching)**—on his project (**XERJ**, `xerj.org` / `github.com/xerj-org/xerj`) in accordance with their mutual agreement, with appreciation for the collaboration with `xerj.org`.

---

## 🧭 Architecture Highlights by Domain (Role-Based Showcase)

This repository is structured as a modular reference architecture. Engineering leaders and recruiters can directly evaluate the specific domain relevant to their organization:

| Engineering Domain / Role | Core Competencies Demonstrated | Primary Modules & Artifacts | Target Technical Keywords |
| :--- | :--- | :--- | :--- |
| **🦀 Systems / Embedded / Low-Level Core**<br>*(Networking, Automotive, Telco, IoT)* | • 128-byte naturally aligned Zero-Copy protocol (`#[repr(C, align(128))]`)<br>• Lock-free SPSC ring buffers and memory barrier synchronization<br>• Kernel-bypass telemetry via eBPF/XDP descriptors<br>• Fixed-point Q32.32 arithmetic with bit-exact reproducibility | [`krystal_kernel/tri_phase_dosing_controller.py`](krystal_kernel/tri_phase_dosing_controller.py)<br>[`krystal_kernel/household_edge_energy_node.py`](krystal_kernel/household_edge_energy_node.py)<br>[`artifacts/benchmarks/TRI_PHASE_POWER_DOSING_SPEC.md`](artifacts/benchmarks/TRI_PHASE_POWER_DOSING_SPEC.md) | *Zero-Copy IPC, Cache Line Alignment (128B), eBPF/XDP, Lock-Free Rings, FFI, Deterministic State Machine, Microsecond Latency* |
| **🧠 AI / ML Infrastructure & MLOps**<br>*(AI Startups, Hyperscalers, Inference Labs)* | • Heterogeneous multi-model cascade dispatch (Intel OpenVINO, DirectML, Vulkan)<br>• Dynamic task sharding between AI inference and compute workloads<br>• UMA shared-memory zero-copy buffer passing<br>• Real-time Token-per-Watt and thermal telemetry budgeting | [`krystal_kernel/krystal_vino_runtime.py`](krystal_kernel/krystal_vino_runtime.py)<br>[`krystal_kernel/grid_aware_energy_arbitrage.py`](krystal_kernel/grid_aware_energy_arbitrage.py)<br>[`artifacts/benchmarks/H07_SILICON_TELEMETRY_AND_OPENVINO_LEXICON_SPEC.md`](artifacts/benchmarks/H07_SILICON_TELEMETRY_AND_OPENVINO_LEXICON_SPEC.md) | *Heterogeneous Computing, OpenVINO, Multi-Model Dispatch, UMA Architecture, Token/Watt Optimization, Thermal Derating* |
| **⚡ EnergyTech / Smart Grid / FinTech**<br>*(VPP Aggregators, Storage OEMs, Algo-Trading)* | • 24h 96-slot spot electricity market arbitrage (OKTE Day-Ahead / Intraday)<br>• Sub-10ms Peak Shaving protecting 3x25A main service breakers<br>• 100-home Virtual Power Plant (VPP) coordination with Phased-Array voltage control<br>• SiC MOSFET switched-matrix 800V DC bus reducing cable $I^2 R$ losses by 277x | [`krystal_kernel/vpp_grid_broker_fleet.py`](krystal_kernel/vpp_grid_broker_fleet.py)<br>[`krystal_kernel/household_edge_energy_node.py`](krystal_kernel/household_edge_energy_node.py)<br>[`demo_tri_phase_dosing_simulation.py`](demo_tri_phase_dosing_simulation.py) | *Spot Market Arbitrage, Dynamic Peak Shaving, VPP (Virtual Power Plant), EN 50160 Grid Code, Phased Array Inverter Control* |
| **🛡️ DevSecOps / Enterprise Architecture**<br>*(Fintech, Defense, Enterprise Platforms)* | • Automated open-source supply chain audit and license firewall<br>• Runtime isolation and IPC quarantine of viral GPL/copyleft dependencies<br>• Cryptographic state provenance (SHA-256 / Ed25519 HMAC telemetry frames)<br>• 100% test pass rate with zero flaky external dependencies | [`krystal_kernel/community_showcase_integrator.py`](krystal_kernel/community_showcase_integrator.py)<br>[`artifacts/benchmarks/COMMUNITY_FOLLOWER_ECOSYSTEM_AUDIT.md`](artifacts/benchmarks/COMMUNITY_FOLLOWER_ECOSYSTEM_AUDIT.md)<br>[`tests/test_tri_phase_dosing_controller.py`](tests/test_tri_phase_dosing_controller.py) | *Supply Chain Security, Automated License Compliance, Copyleft Quarantine, Cryptographic Provenance, Hermetic Unit Testing* |

---

## 🔬 Industry Engineering Translation

Krystal Stack uses rigorous mathematical and physical invariants. Here is how core project paradigms translate into industry-standard engineering terms:

* **Constitutional Invariant (`assert VITAL_MAX_HP == 6`) $\implies$ Formal State Invariant / Safety Gate:**  
  Guarantees that no state mutation occurs unless an action is strictly authorized and geometrically bounded. If an action is invalid, $\Delta S = 0$ holds across all layers, preventing state corruption, memory overflows, or electrical breaker overload.
* **Time-Sliced Power Dosing $\implies$ Phase-Shifted PWM Duty-Cycle Load Balancing:**  
  Alternates peak power windows across phases ($L_1, L_2, L_3$) to prevent thermal accumulation in domestic terminal blocks while active $120^\circ$ inverter PWM offsets eliminate 3rd harmonic (150 Hz) accumulation on Neutral ($N$).
* **Digital Twin & Simulation-First Engineering $\implies$ Hardware-in-the-Loop (HIL) Pre-Validation:**  
  Validates complex thermal, electrical, and market interactions n-dimensionally in software before committing physical actuation, protecting expensive multi-kW hardware and electrical wiring.

---

## 🔌 Hardware-in-the-Loop (HIL) Architecture & Electrical Blueprint

The framework models and controls an asymmetric 3-phase hybrid microgrid tied to an AI compute cluster:

```text
[ Utility Grid 3x 230V/400V - 25A Main Breaker ]
      │ (CT Current Transducers + 1-Wire Terminal Thermal Probes @ 10 ms)
      ▼
[ KRYSTAL POWER SPINE (Low-Loss Bus CYKY-J 5x10 mm²) ]
 ├──► Phase L1 (max 23.9 A) ──► [ Inverter #1 (5 kW) ] ──► PSU #1 (1 800 W) ──► GPUs 0, 1, 2 + EPYC #1
 ├──► Phase L2 (max 23.9 A) ──► [ Inverter #2 (5 kW) ] ──► PSU #2 (1 800 W) ──► GPUs 3, 4, 5 + EPYC #2
 ├──► Phase L3 (max 23.9 A) ──► [ Inverter #3 (5 kW) ] ──► PSU #3 (1 800 W) ──► GPUs 6, 7, 8, 9 + Cooling Pumps
 └──► Neutral Conductor (N) ──► Doubled 10 mm² with Active 120° Harmonic Cancellation
```

### Protocol Layout: 128-Byte Naturally Aligned Contract Frame
Every telemetry frame across IPC and shared memory adheres to an exact 128-byte boundary:
- **Bytes 0..57 (58 B):** Telemetry header, 3-phase CT current & voltage readings, battery state-of-charge, grid frequency.
- **Bytes 58..73 (16 B):** Cryptographic SHA-256 state hash for tamper-evident data integrity.
- **Bytes 74..95 (22 B):** L3 Grid Broker fields: transformer ID, reactive command ($Q$), VPP dispatch target.
- **Bytes 96..117 (22 B):** Tri-Phase Dosing extension: per-phase cable temperatures ($0.1^\circ\text{C}$), GPU sharded power, 3-way cooling valve state.
- **Bytes 118..127 (10 B):** Tail zero-padding for 128-byte cache-line alignment.

$$\mathbf{58 + 16 + 22 + 22 + 10 = 128\ \text{Bytes}}$$

---

## 🚀 Runnable Demos & Simulation Scenarios

Execute any of the built-in, zero-dependency simulation suites directly from the command line:

```bash
# 1. Tri-Phase Power Dosing & Single-Phase Overload Mitigation (Sub-10ms response)
python demo_tri_phase_dosing_simulation.py

# 2. Virtual Power Plant (VPP) 100-Home Grid Broker & 800V SiC Switched Matrix
python demo_vpp_grid_broker_simulation.py

# 3. 24-Hour 96-Slot Household Edge Simulation (+2,714 €/yr Arbitrage & Heat Recovery)
python demo_household_edge_simulation.py

# 4. Grid Frequency FCR Stabilization (50 Hz primary reserve response)
python demo_grid_aware_arbitrage.py
```

### Snapshot of Live Overload Simulation Output (`demo_tri_phase_dosing_simulation.py`)
```text
▶ CRITICAL EVENT: STOVE OVERLOAD (3.0 kW) ON L1 + SPOT PRICE SPIKE (350 €/MWh) + T_L1 = 46.2 °C!
  ⚡ Reaction Latency:        0.027 ms (sub-10ms control cycle guaranteed)
  🚨 Terminal Thermal Alarm:  True (T_L1 = 46.2 °C >= 45.0 °C safe threshold)
  🛡️  Phase Relief Applied:   True
  🔄 Phase-Aware Task Shift:
     ├── L1 (GPUs 0, 1, 2):   Throttled to AI_INFERENCE (1240 W -> 295 W: 945 W freed instantly)
     ├── L2 (GPUs 3, 4, 5):   1240 W [CRYPTO_MINING] (Full throughput maintained)
     └── L3 (GPUs 6..9):      1660 W [CRYPTO_MINING] (Full throughput maintained)
  🔋 Time-Sliced Power Dosing:
     ├── Inverter #1 (L1):    40% Duty (export throttled, cable L1 safely cools down)
     ├── Inverter #2 (L2):    95% Duty (full export compensation)
     └── Inverter #3 (L3):    95% Duty (full export compensation)
  📉 L1 Service Breaker:      Held at ~16.2 A (Safely below 23.9 A ceiling of 25 A breaker)
  ✨ Outcome: Breaker never tripped, cables cooled down, AI inference and arbitrage revenue preserved!
```

---

## 💼 Ready-to-Use CV & Portfolio Highlights (Copy-Paste Snippets)

When highlighting Krystal Stack in your CV, portfolio, or technical interview discussions, use these tailored bullet points:

> **Krystal Stack – High-Performance Compute & Grid-Edge Optimization Framework** *(Architect & Lead Engineer)*
> * **Low-Level Architecture & Protocol Design:** Engineered a 128-byte naturally aligned Zero-Copy binary protocol (`#[repr(C, align(128))]`) for microsecond IPC and eBPF/XDP network telemetry passing, eliminating Garbage Collection overhead on high-frequency control loops.
> * **Digital Twin & Smart Grid Simulation:** Implemented a full 24-hour, 96-slot spot market arbitrage engine (OKTE Day-Ahead / Intraday) with asymmetric 3-phase balancing ($L_1, L_2, L_3$), sub-10ms dynamic peak shaving for 3x25A main breakers, and thermodynamic server heat recovery.
> * **Heterogeneous AI & Workload Scheduling:** Developed an adaptive hardware dispatcher dynamically arbitrating 10 AMD GPUs and dual EPYC nodes between low-power, high-margin AI inference (Intel OpenVINO / DirectML) and high-throughput compute based on instantaneous phase thermal telemetry.
> * **Automated Governance & Supply Chain Security:** Architected an automated license compliance firewall (`CommunityAuditGuard`) providing IPC process isolation and boundary quarantine for third-party open-source dependencies, backed by a 100% passing test suite.

---

## 🧪 Verification & Test Suite

The test suite runs with zero external network or database dependencies:

```bash
# Run the complete energy, grid, and infrastructure test suite:
python -m unittest tests/test_tri_phase_dosing_controller.py \
                   tests/test_vpp_grid_broker_fleet.py \
                   tests/test_household_edge_energy_node.py \
                   tests/test_grid_aware_energy_arbitrage.py \
                   tests/test_community_showcase_integrator.py \
                   tests/test_silicon_pipeline_orchestrator.py \
                   tests/test_silicon_telemetry_lexicon.py \
                   tests/test_investor_suite_integration.py
```
**Result:** 58/58 tests passing in **0.026s** with zero regressions.

---

## 👥 Authors & Licensing

* **System Architect & Lead Engineer**: Dušan Kopecký ([@RhodosBerger](https://github.com/RhodosBerger))
* **Contact**: `dusan.kopecky0101@gmail.com`
* **LinkedIn**: [Dušan Kopecký](https://www.linkedin.com/in/du%C5%A1an-kopeck%C3%BD-bbba33301/)
* **License**: Apache License 2.0 (See [`LICENSE`](LICENSE))
