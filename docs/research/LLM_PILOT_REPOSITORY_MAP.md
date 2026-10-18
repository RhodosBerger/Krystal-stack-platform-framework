# LLM PILOT REPOSITORY MAP
**Krystal Stack // Subagent 1: Architectural Mapping of AEP, Governors, and Telemetry for Inference Workload**  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  
**Epistemic Rule:** *The existing codebase is authoritative. Models are mapped without renaming or generic restructuring.*

---

## 1. Executive Summary

This mapping audits the existing Krystal Stack runtime architecture to ground the new `INFERENCE WORKLOAD` within validated patterns. Krystal Stack possesses mature state machine machinery, deterministic action gating, and telemetry feedback, but previously lacked a formal `INFERENCE` workload classification within the Adaptive Execution Processor (AEP).

```
[ Telemetry Layer: Hardware, VRAM, RAPL, UMA Bus ]
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Governor Layer (Heuristic / Predictive)                │
│  - krystal_kernel/cortex_openvino_engine.py            │
│  - src/rust-bot/src/unified_core.rs                    │
│  - krystal_bot/quota.py & llm_gateway.py               │
└────────────────────────────────────────────────────────┘
                         │
                         ▼ Proposed ActionPayload
┌────────────────────────────────────────────────────────┐
│ AEP Transactional Core (Authoritative State Gate)      │
│  - krystal_kernel/aep_engine.py (L0 - L8)              │
│  - WorkloadType: RENDER, ECONOMY, RESEARCH (+INFERENCE)│
│  - Invariants: VITAL_MAX_HP == 6, Delta S == 0         │
└────────────────────────────────────────────────────────┘
                         │
                         ▼ Execution & Receipt
[ Physical Execution & AepReceipt Audit Logging ]
```

---

## 2. Inventory of Existing Core Components

### 2.1 AEP Transactional Engine (`krystal_kernel/aep_engine.py`)
- **Authority:** Central arbiter of all state transitions.
- **Contract:** Two-phase execution with automatic rollback on violation:
  $$\neg\text{Valid}(S_t, a) \implies S_{t+1} \equiv S_t \land \Delta S = 0$$
- **Invariant Rules:** Formally enforced via `InvariantRule` subclasses (`DoubleEntryConservationRule`, `NonNegativeBalanceRule`).
- **Existing Workload Types:**
  - `WorkloadType.RENDER`
  - `WorkloadType.ECONOMY`
  - `WorkloadType.RESEARCH`
  - *Extension:* `WorkloadType.INFERENCE` (to be introduced).
- **AEP Receipt (`AepReceipt`):** Emits cryptographic HMAC-SHA256 authenticated execution logs containing `operation_id`, `runtime_id`, `state_before`, `state_after`, `execution_us`, `bus_pressure`, `routing`, and `verdict`.

### 2.2 Process Governor & Hardware Routing (`krystal_kernel/cortex_openvino_engine.py` & `openvino_aep_runtime.py`)
- **Governor:** `OpenVINOProcessGovernor` monitors system load, process memory, and context-switch rates to assign execution priorities (`REALTIME`, `BATCH`, `IDLE`).
- **Hardware Crossover (`HardwareCrossoverPolicy`):**
  - Payload $< 16\text{ KiB} \implies \text{CPU (Willow Cove)}$ (Latency $\sim 30\text{ µs}$).
  - Payload $\ge 16\text{ KiB} \land \mu_{\text{UMA}} < 0.70 \implies \text{GPU (Intel Iris Xe)}$ (Latency $\sim 1.66\text{ ms}$).
  - High Bus Pressure ($\mu_{\text{UMA}} \ge 0.70$) $\implies$ Fallback to CPU to protect display scanout.
- **Anti-Self-Certification:** Enforces strict provenance contract:
  `requested_device` vs `actual_device` vs `fallback_reason`.

### 2.3 Bot LLM Gateway & Quota Manager (`krystal_bot/llm_gateway.py` & `quota.py`)
- **Gateway:** `krystal_bot.llm_gateway.LLMConfig` supports OpenAI-compatible endpoints (`/v1/chat/completions`) with deny-by-default security policies.
- **Quota Enforcer:** `QuotaManager` tracks sliding-window request-per-minute and token-per-day budgets.
- **Breaker Protection:** Automatically trips circuit breaker after $N=3$ consecutive network failures with 60s cooldown.
- **Redaction Filter:** Sanitizes API keys, authorization tokens, and personal file paths.

### 2.4 Epistemic Runtime & ABI (`src/rust-bot/src/epistemic_abi.rs` & `rust/epistemic_runtime`)
- **16-bit Status ABI:** Encodes state into 4-bit nibble families:
  - `0x1000`: Epistemic state (`EpiObserved`, `EpiPredicted`, `EpiFalsified`).
  - `0x3000`: Authority state (`AuthCommitApproved`, `AuthVetoed`, `AuthQuarantined`).
  - `0x4000`: Invariant health (`InvNominal6hp = 0x4006`).
  - `0x6000`: Transaction lifecycle (`TxnPrepared`, `TxnCommitted`, `TxnRolledBack`).
  - `0xE000`: Policy state (`PolActive`, `PolExperimental`, `PolRejected`).

---

## 3. Gap Analysis for Slovak LLM Pilot

| Architectural Requirement | Existing Implementation | Required Extension |
|---|---|---|
| Workload Classification | `RENDER`, `ECONOMY`, `RESEARCH` | Add `WorkloadType.INFERENCE` to `krystal_kernel/aep_engine.py` |
| Token & Context Accounting | Token counters in `krystal_bot/quota.py` | Embed `prompt_tokens`, `completion_tokens`, `ttft_us` in `AepReceipt` |
| Slovak Model Adapter | Mock OpenVINO neural projection | Introduce `SlovakLLMAdapter` supporting GGUF, OpenVINO IR, and Mock |
| Bus-Aware Backpressure | Crossover threshold based on raw payload bytes | Dynamic token pacing & pre-emption based on UMA bus pressure $> 0.70$ |
| Conversation State Integrity | Monolithic JSON conversation logs | Transactional state rollback ($\Delta S = 0$) on inference abort/rejection |

---

## 4. Integration Blueprint

The new inference subsystem will dock directly into the existing `AepTransactionalEngine` without modifying existing render or economy code paths:

```python
# Canonical Transaction Invocation for LLM Inference
result = aep_engine.execute_transition(
    runtime_id="slovak-llm-pilot",
    workload=WorkloadType.INFERENCE,
    action="GENERATE_TOKENS",
    current_state=conversation_state,
    transition_fn=inference_transition_fn,
    action_payload={"prompt": prompt, "max_tokens": 128},
    policy_id="slovak-llm-adaptive-governor",
    execution_lane=ExecutionLane.COMPUTE,
    current_bus_pressure=telemetry.current_bus_pressure,
)
```
