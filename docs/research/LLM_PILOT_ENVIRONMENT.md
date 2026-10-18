# LLM PILOT ENVIRONMENT
**Krystal Stack // Subagent 2: Hardware & Runtime Discovery**  
**Host Platform:** Intel Tiger Lake UP3 / Windows 11  
**Measurement Class:** `[PHYSICAL-LOCAL-MEASURED]` // `[SYSTEM-DISCOVERED]`

---

## 1. Physical Hardware Telemetry

| Parameter | Measured Specification | Source / Tool |
|---|---|---|
| **Processor** | 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz | WMI / `platform.processor()` |
| **Microarchitecture** | Tiger Lake (Willow Cove Core, 10nm SuperFin) | Intel Architecture Spec |
| **Topology** | 4 Physical Cores / 8 Logical Threads | Windows Kernel PSAPI |
| **Vector Instruction Sets** | AVX2, AVX-512, Intel Deep Learning Boost (VNNI) | CPUID Flags |
| **Integrated GPU** | Intel(R) Iris(R) Xe Graphics (iGPU, 80/96 EUs) | OpenVINO `get_property` |
| **Total Physical RAM** | 24,852,824 KB (23.70 GiB / 25.45 GB) | `Win32_OperatingSystem` |
| **Free Physical RAM** | 8,377,216 KB (7.99 GiB / 8.58 GB) | `Win32_OperatingSystem` (Measured) |
| **Memory Architecture** | Unified Memory Architecture (UMA, shared DDR4/LPDDR4x) | Intel Tiger Lake SoC Bus |
| **Thermal Ceiling (Tj_max)** | 100.0 °C (Throttle Trip), 85.0 °C (Governor Guardrail) | Hardware Spec / `ActionGate` |
| **Power Profile** | PL1 = 28 W (Sustained), PL2 = 64 W (Short Burst) | Intel DPTF / MSR Telemetry |

---

## 2. Software Runtime & Library Inventory

### 2.1 Active Python Interpreter
- **Version:** `Python 3.14.7 (tags/v3.14.7:823f032)` 64-bit.
- **Key Installed Packages:**
  - `numpy 2.5.4`
  - `openvino 2026.4.1` (Build 22982-e213a147257)
  - `z3-solver 5.3.0.0`
  - `pytest 9.1.1`
  - `ruff 0.17.0`
- **Isolated Secondary Environment (`.venv-keras`):**
  - Contains `torch 2.14.1+cpu` and `keras 3.15.1`.

### 2.2 OpenVINO Hardware Discovery
Executing `openvino.Core().available_devices` returns:
```json
["CPU", "GPU"]
```
- `CPU`: `11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz`
  - Capabilities: FP32, FP16, INT8, VNNI int8 dot-products (`dp4a`).
- `GPU`: `Intel(R) Iris(R) Xe Graphics (iGPU)`
  - Capabilities: FP16, INT8 execution units, Gen12 display-scanout sharing.

### 2.3 Environmental Constraints & Blockers
1. **Windows Application Control:** JAX / jaxlib execution is strictly blocked by local Windows policy.
2. **Missing CLI Binaries:** Neither `ollama.exe` nor `llama-cli.exe` are present on the system `PATH`.
3. **OpenVINO GenAI Package:** `openvino_genai` Python bindings are not installed; however, core OpenVINO graph execution (`openvino.Model`, `CompiledModel`, `InferRequest`) is fully operational.

---

## 3. Workload Memory Sizing for LLM Inference

Given total free physical RAM of **$7.99\text{ GiB}$** on a UMA system, the maximum feasible parameter footprints are:

```
[ TOTAL SYSTEM RAM: 23.7 GiB ]
├── Operating System & Applications In Use: ~15.7 GiB
└── Free Available UMA Memory: 8.0 GiB
     ├── [14B FP16]  --> 28.0 GiB  [IMPOSSIBLE - OOM PÁD]
     ├── [14B Q6_K]  --> 11.8 GiB  [CRITICAL STALL - HEAVY SWAP]
     ├── [14B Q4_K]  -->  8.9 GiB  [UNSAFE - PUSHES MEMORY TO 99%]
     ├── [ 7B Q4_K]  -->  4.8 GiB  [STABLE - FITS IN FREE RAM]
     └── [ 3B Q4_K]  -->  2.2 GiB  [OPTIMAL - LOW LATENCY & SAFE]
```

### Strategic Conclusion
- **14B Models:** Running full 14B models on this host without a dedicated discrete GPU or external offload server will exhaust physical RAM and trigger severe disk swapping, violating the real-time deadline invariants of Krystal Stack.
- **Governor Role:** The `AepInferenceGovernor` must actively inspect model weight footprint against `FreePhysicalMemory`. If weight size $> \text{FreePhysicalMemory} \times 0.75$, the governor must assert `DISCARDED_BACKPRESSURE_REJECTED` with reason `VRAM_BUDGET_EXCEEDED` ($\Delta S = 0$).
