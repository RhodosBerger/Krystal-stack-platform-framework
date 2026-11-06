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
