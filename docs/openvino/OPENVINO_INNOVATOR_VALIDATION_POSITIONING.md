# OpenVINO Innovator Validation Positioning

**Status:** Engineering evidence / corporate communication profile  
**Scope:** Krystal Stack + Adaptive Execution Processor (AEP) + OpenVINO Runtime  
**Audience:** Intel / OpenVINO engineers, technical reviewers, Innovator Program discovery call

---

## 1. Core Positioning

Krystal Stack should be presented as an **adaptive orchestration and telemetry layer around heterogeneous AI execution**, not as a replacement for the operating-system scheduler or OpenVINO Runtime.

> **Krystal Stack is a constraint-governed orchestration and telemetry layer for heterogeneous AI workloads. It observes runtime pressure, selects an execution policy, constrains available resources, dispatches work through OpenVINO or other backends, and records the resulting latency, throughput, thermal and resource-utilization behaviour.**

OpenVINO remains the inference runtime. Krystal Stack adds a closed-loop policy layer above the runtime and below the application-level workload logic.

```text
Application / Workload
        |
        v
Krystal AEP
Observe -> Estimate -> Constrain -> Select Policy -> Schedule
        |
        v
OpenVINO Runtime
        |
        v
CPU / GPU / NPU
        |
        v
Telemetry -> pressure / latency / utilization / thermals
        +-------------------------------------------> AEP feedback
```

---

## 2. What Is Different From the Conventional Stack

### Conventional execution path

```text
Application
   -> OS scheduler / power policy
   -> OpenVINO device/runtime policy
   -> CPU/GPU/NPU execution
```

The operating system and OpenVINO already perform sophisticated scheduling and resource selection. OpenVINO exposes portable performance hints and lower-level properties such as inference thread limits, stream counts, CPU pinning, hyper-threading use and core-type selection.

### Krystal execution path

```text
Application
   -> AEP Observation Engine
   -> Policy Processor
   -> Workload Scheduler
   -> proof-carrying ExecutionEnvelope
   -> OpenVINO runtime/device configuration
   -> OS scheduler / hardware
   -> measurement + feedback
```

Krystal's differentiator is not that the underlying CPU becomes physically faster. The differentiator under investigation is that the software stack may spend **less wall-clock time in non-productive states** and may use the available hardware execution budget more effectively.

Potential mechanisms include:

- lower control-path overhead,
- reduced scheduling and policy recomputation,
- better worker-count selection,
- improved core/backend selection,
- reduced queue pressure,
- fewer unnecessary context transitions,
- more stable thermal/power envelopes,
- better alignment of concurrency with the active workload,
- better use of OpenVINO threads, streams and device policy.

---

## 3. Important Terminology: Governor Quota vs. Useful Work

Do **not** state that Krystal has proven that it "increases CPU performance at the same clock" unless fixed-frequency, cycle-level evidence exists.

The current Krystal `CustomHardwareGovernor` emits an immutable `ExecutionEnvelope` containing, among other fields:

- `allowed_backends`,
- `max_workers`,
- `max_power_w`,
- `max_tick_hz`,
- `thermal_budget_c`,
- `memory_pressure_limit`.

This means the implemented mechanism governs the **resource envelope available to the workload**. It does not by itself prove higher instructions-per-cycle (IPC), microarchitectural efficiency, or silicon-level performance.

### Safe engineering claim

> **At comparable hardware conditions, Krystal's adaptive planning can increase useful completed work per unit of wall-clock time by reducing orchestration overhead and selecting a more appropriate execution envelope.**

### Stronger claim that requires additional proof

> **At a fixed effective CPU frequency, Krystal increases useful work per cycle / instructions per cycle.**

This stronger statement must remain a **hypothesis** until verified with hardware counters and a controlled fixed-frequency experiment.

---

## 4. Existing Local Evidence

The current benchmark evidence registry contains a controlled Phase-Q comparison on an Intel Tiger Lake-class 4C/8T host.

### Governor stage

| Metric | Q0 baseline | Q-AB adaptive | Change |
|---|---:|---:|---:|
| Governor p50 | 86.0 us | 74.0 us | -13.95% |
| Governor p99 | 246.2 us | 217.8 us | -11.53% |

### Total software path

| Metric | Q0 baseline | Q-AB adaptive | Change |
|---|---:|---:|---:|
| Total p50 | 120.4 us | 108.8 us | -9.63% |
| Total p99 | 347.9 us | 317.4 us | -8.77% |

The benchmark also reports zero semantic mismatches in paired comparisons and 30,000/30,000 invariant checks passing.

**Interpretation:** this supports a measured reduction in software/governor-path latency under the captured local benchmark. It does **not yet prove** increased CPU IPC, equal-frequency execution efficiency, or a universal OpenVINO inference speedup.

---

## 5. Proposed "Same-Clock Useful Work" Validation

The next benchmark should explicitly test the user's observation that the processor appears to complete more useful work under Krystal planning while operating at a comparable clock.

### Hypothesis H-SCW-01

> Under a controlled workload and comparable effective CPU frequency, the Krystal adaptive execution policy increases completed application work per second by reducing scheduling, queueing and orchestration overhead relative to the baseline stack.

### Controlled variants

**B0 — Baseline**

```text
Application -> OpenVINO default/performance-hint configuration -> OS -> CPU
```

**K1 — Krystal adaptive**

```text
Application -> Krystal AEP -> selected OpenVINO configuration -> OS -> CPU
```

Keep identical:

- model,
- model precision,
- input dataset,
- batch size unless batch is the variable under study,
- hardware,
- OS and kernel,
- OpenVINO version,
- thermal start window,
- power source,
- benchmark duration,
- background services.

### Required measurements

Record for every run:

1. wall-clock throughput / completed tasks,
2. latency p50 / p95 / p99,
3. average and distribution of effective CPU frequency,
4. CPU cycles,
5. retired instructions,
6. IPC = retired instructions / cycles,
7. context switches,
8. CPU migrations,
9. cache misses where available,
10. task-clock / CPU utilization,
11. package power / energy where available,
12. thermal state and throttling flags,
13. OpenVINO compiled-model properties,
14. OpenVINO stream/thread/pinning/core-selection configuration.

On Linux, suitable evidence may be collected with `perf stat`, RAPL/powercap where available, CPU frequency telemetry, and OpenVINO `benchmark_app` / compiled-model property output.

### Primary derived metrics

```text
UsefulWorkRate = completed_work_units / wall_time
CyclesPerWork  = total_cpu_cycles / completed_work_units
EnergyPerWork  = joules / completed_work_units
IPC            = retired_instructions / cpu_cycles
```

The claim "same clock, more useful work" is supported only if the frequency distributions are comparable and `UsefulWorkRate` improves without an uncontrolled increase in frequency/power envelope.

---

## 6. Relationship to OpenVINO Runtime

OpenVINO already supports high-level `LATENCY` and `THROUGHPUT` performance hints and exposes lower-level CPU execution controls, including:

- inference thread count,
- stream count,
- core-type scheduling,
- hyper-threading use,
- CPU pinning,
- CPU reservation.

Therefore, the clean research question is not:

> "Can Krystal out-schedule OpenVINO?"

It is:

> **"Can an application-level closed-loop controller choose between OpenVINO execution policies and system execution envelopes more effectively when workload pressure, thermal state, queue depth and heterogeneous device availability change over time?"**

This formulation makes the project complementary to OpenVINO rather than adversarial to it.

Official OpenVINO references:

- Performance Hints and Thread Scheduling: https://docs.openvino.ai/nightly/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/performance-hint-and-thread-scheduling.html
- Optimize Inference: https://docs.openvino.ai/2026/openvino-workflow/running-inference/optimize-inference.html
- Benchmark Tool: https://docs.openvino.ai/2026/get-started/learn-openvino/openvino-samples/benchmark-tool.html

---

## 7. Recommended Intel / OpenVINO Introductory Language

> **Krystal Stack adds a closed-loop execution-policy layer around OpenVINO rather than replacing the runtime or operating-system scheduler. In our local controlled benchmarks, the adaptive governor path reduced p50 and p99 software-path latency while preserving semantic correctness. We have also observed workloads completing more useful work under adaptive planning at apparently comparable clock conditions. We currently treat that second observation as a hypothesis rather than a silicon-level performance claim, and our next validation step is to measure effective frequency, CPU cycles, retired instructions, IPC, context switches, OpenVINO execution properties, power and thermals under baseline-versus-Krystal runs. Our goal is to make the claim reproducible and falsifiable across Intel CPU/GPU/NPU platforms.**

This wording is intentionally conservative: it highlights the interesting result while making a clear boundary between what is already measured and what still requires hardware-counter validation.

---

## 8. Claim Matrix

| Claim | Status | Evidence needed / available |
|---|---|---|
| AEP is a closed-loop execution architecture | **ARCHITECTURAL** | `docs/architecture/AEP_CANONICAL_SPEC.md` |
| Governor emits constrained execution envelopes | **IMPLEMENTED** | `krystal_kernel/custom_governor.py` |
| Phase-Q governor p50/p99 improved in current local benchmark | **MEASURED (LOCAL)** | `artifacts/BENCHMARK_EVIDENCE_REGISTRY_V1.json` |
| Total software-path p50/p99 improved in current local benchmark | **MEASURED (LOCAL)** | `artifacts/BENCHMARK_EVIDENCE_REGISTRY_V1.json` |
| Krystal causes a CPU to execute more useful work at the same effective frequency | **HYPOTHESIS** | Requires H-SCW-01 controlled fixed/comparable-frequency test |
| Krystal increases CPU IPC | **UNPROVEN HYPOTHESIS** | Requires retired-instruction + cycle counters |
| Krystal universally outperforms OpenVINO defaults | **NOT CLAIMED** | Requires multi-model / multi-device benchmark campaign |

---

## 9. Corporate Communication Principle

The strongest position is not:

> "We made Intel processors faster."

The strongest technically defensible position is:

> **"We built an adaptive control layer that appears to increase the amount of useful workload completed inside the same practical execution window. We already have local latency evidence, and we are designing the next benchmark so Intel engineers can independently reproduce or falsify the same-clock efficiency hypothesis."**

That is the claim to carry into the OpenVINO Innovator introduction and discovery call.
