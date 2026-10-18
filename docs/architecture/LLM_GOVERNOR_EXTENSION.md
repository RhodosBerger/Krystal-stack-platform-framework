# KRYSTAL STACK // LLM GOVERNOR EXTENSION
**Chronometric Shadow Multiplexing & Adaptive Pacing Engine**  
*Classification:* `[IMPLEMENTED]` // `[ARCHITECTURE-SPEC]`  
*Author:* Dušan Kopecký & Krystal-Stack Architecture Council (2026)  
*Constitutional Invariant:* `VITAL_MAX_HP == 6`

---

## 1. Architectural Motivation

In integrated GPU/CPU systems such as Intel Tiger Lake (Core i5-1135G7 with Iris Xe Graphics), both computing cores and the display controller share a common unified memory bus (UMA).

When a large language model inference workload saturates the memory bus ($\mu_{\text{UMA}} > 0.70$), it starves the display compositor. This leads to dropped frames, display stutter, and VSync deadline overruns.

To solve this, Krystal Stack implements **Chronometric Shadow Multiplexing (KSPP Principle 3)** through the `AepInferenceGovernor`:
$$\Phi_{\text{Scanout}}(t) \cdot \Phi_{\text{Compute}}(t) = 0 \land \mu_{\text{UMA}} < 0.78$$

Inference workloads execute in the phase shadows of display scanout and are actively paced or rerouted when bus contention threatens system responsiveness.

---

## 2. Dual Governor Architecture

The governor extension introduces two selectable operating modes:

```
                            +-----------------------------------+
                            |      Inference Action Request     |
                            +-----------------+-----------------+
                                              |
                                              v
                              [ AepInferenceGovernor Mode? ]
                                      /               \
                                     /                 \
                  +-----------------+                   +------------------+
                  | STATIC_BASELINE |                   | ADAPTIVE_GOVERNOR|
                  +--------+--------+                   +--------+---------+
                           |                                     |
                           v                                     v
                  [ Greedy Dispatch ]                 [ Sample Bus Pressure ]
                           |                                     |
                           |                        +------------+------------+
                           |                        |  Green: < 0.40  -> GPU  |
                           |                        |  Yellow: 0.40-0.70->Pace|
                           |                        |  Orange: 0.70-0.85->CPU |
                           |                        |  Red:   >= 0.85 -> DROP |
                           |                        +------------+------------+
                           \                                     /
                            \                                   /
                             v                                 v
                          +---------------------------------------+
                          |      AepTransactionalEngine (L8)      |
                          |     Invariant Check: VITAL_MAX_HP == 6|
                          |      Backpressure Gate: mu <= 0.85    |
                          +-------------------+-------------------+
                                              |
                                     +--------+--------+
                                     |                 |
                                     v                 v
                               [ COMMITTED ]     [ DISCARDED ]
                                (St+1 = S')      (Delta S == 0)
```

### 2.1 Mode 1: `STATIC_BASELINE`
- **Behavior:** Greedy, non-adaptive dispatch.
- **Token Budget:** Always dispatches 100% of requested tokens.
- **Device Target:** Always attempts execution on requested accelerator (default `GPU`).
- **Failure Mode:** When $\mu_{\text{UMA}} \ge 0.85$, the AEP Transactional Gate rejects the transaction immediately, resulting in 100% aborted operations with zero graceful degradation.

### 2.2 Mode 2: `ADAPTIVE_GOVERNOR`
- **Behavior:** Dynamic bus-aware pacing and transparent fallback.
- **Control Rules:**
  1. **Green Zone ($\mu < 0.40$):** Full throughput, unconstrained token generation, assigned to `ExecutionLane.COMPUTE` on GPU.
  2. **Yellow Zone ($0.40 \le \mu < 0.70$):** Bus throttling active. Token generation clamped by 50% (minimum 16 tokens) to reduce burst memory bandwidth.
  3. **Orange Zone ($0.70 \le \mu < 0.85$):** Severe contention. Token generation compressed to emergency brevity ($\le 24$ tokens), routed strictly to CPU on `ExecutionLane.BACKGROUND`. This liberates 100% of GPU UMA bandwidth for the 60 FPS display compositor.
  4. **Red Zone ($\mu \ge 0.85$):** Critical bus saturation. The transaction is discarded cleanly prior to heavy compute, returning `DISCARDED_BACKPRESSURE_REJECTED` and ensuring zero-dirty state mutation ($\Delta S \equiv 0$).

---

## 3. Transactional State Representation

The governor tracks authoritative progress in `LLMInferenceState`:

```python
@dataclass
class LLMInferenceState:
    session_id: str
    active_model: str
    completed_requests: int = 0
    total_tokens_generated: int = 0
    total_prompt_tokens: int = 0
    accumulated_ttft_us: int = 0
    accumulated_execution_us: int = 0
    backpressure_drops: int = 0
    last_action: str = "INIT"
    vital_max_hp: int = 6  # Constitutional Invariant
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Pure Transition Function ($S' = T(S_t, A)$)
```python
def transition_llm_inference(state: LLMInferenceState, action_payload: Dict[str, Any]) -> Dict[str, Any]:
    ...
```
- If the action is discarded, `backpressure_drops` increments while all authoritative state values are frozen.
- If committed, tokens and microsecond execution latencies are atomically integrated into the state.
- Invariant `vital_max_hp == 6` is re-asserted on every transition.

---

## 4. Hardware Safety & License Guardrails

The governor extension integrates directly with `SlovakLLMAdapter` to enforce two constitutional stop conditions:

1. **Stop on License Non-Compliance:**
   ```python
   APPROVED_LICENSES = {"Apache-2.0", "MIT", "BSD-3-Clause", "CC-BY-4.0"}
   ```
   If a model specification contains a closed, proprietary, or unverified license, `validate_license()` fails, causing immediate termination with `PermissionError` and zero state mutation.

2. **Stop on Memory Overrun:**
   ```python
   fits, avail_mb, req_mb = adapter.check_memory_headroom()
   if not fits:
       raise MemoryError(...)
   ```
   If the estimated working set memory footprint exceeds 85% of available physical host RAM, the governor refuses to load or execute the unquantized model, preventing OS swapping and kernel lockups.

---

## 5. Verification & Audit Trail

The governor extension is verified by:
- 11 unit tests in `tests/test_v_aep_inference_workload.py`
- Empirical multi-pressure benchmark in `scripts/benchmark_llm_governor_efficiency.py`
- Immutable JSON telemetry in `artifacts/benchmarks/slovak_llm_governor_benchmark.json`
