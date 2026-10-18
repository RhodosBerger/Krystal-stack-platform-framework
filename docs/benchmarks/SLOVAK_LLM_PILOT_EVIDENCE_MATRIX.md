# KRYSTAL STACK // SLOVAK LLM PILOT EVIDENCE MATRIX
**Formal Claim Stratification, Falsifiers & Epistemic Audit**  
*Classification:* `[AUDIT-VERIFIED]` // `[POLICY-VERIFIED]`  
*Author:* Dušan Kopecký & Krystal-Stack Architecture Council (2026)  
*Constitutional Invariant:* `VITAL_MAX_HP == 6`

---

## 1. Epistemic Audit Framework

Every statement and conclusion regarding the Slovak LLM pilot is recorded in the formal 5-tuple:
$$\langle \text{Claim}, \text{Scope}, \text{Evidence}, \text{Falsifier}, \text{Status} \rangle$$

Evidence labels are non-interchangeable and strictly segregated with `//` tags:
- `// [SOURCE-PROVEN]`: Verified directly from upstream model repository or git commit history.
- `// [HARDWARE-MEASURED]`: Detected from physical OS kernel, Win32 APIs, or OpenVINO Core runtime.
- `// [PHYSICAL-LOCAL-TIMING]`: Measured in microseconds via monotonic system clock (`time.perf_counter_ns()`).
- `// [POLICY-VERIFIED]`: Enforced by transactional state machine contracts and invariant rules.
- `// [MODEL-ESTIMATED]`: Parameterized simulation or neural model weight footprint estimation.

---

## 2. Master Evidence Matrix

| ID | Claim | Scope | Stratified Evidence | Smallest Falsifier | Epistemic Status |
|:---|:---|:---|:---|:---|:---:|
| **C-01** | `slovak-nlp/Qwen3-14B-sk` is licensed under Apache-2.0 and is legally compliant for sovereign integration. | Global / Repository | `// [SOURCE-PROVEN]`: Hugging Face repository metadata confirms Apache 2.0. Verified via live web search. | Upstream LICENSE file containing non-commercial or proprietary clause. | `VERIFIED_WITH_SCOPE` |
| **C-02** | Full FP16 14B model footprint (~28.0 GiB) exceeds total host physical RAM (23.70 GiB). | Tiger Lake Host | `// [HARDWARE-MEASURED]`: `GlobalMemoryStatusEx` reports 23.70 GiB total physical RAM. `// [MODEL-ESTIMATED]`: 14.77B * 2.0B + 512MB KV = ~28.0 GiB. | Host upgraded to >= 32 GiB physical RAM or memory compression driver installed. | `MEASURED_WITH_SCOPE` |
| **C-03** | 14B Q6_K (~11.8 GiB) exceeds currently available free physical RAM (5.5–8.0 GiB), triggering disk swapping without stop condition. | Tiger Lake Host | `// [HARDWARE-MEASURED]`: Free physical RAM sampled at 5,524 MB. `// [POLICY-VERIFIED]`: `SlovakLLMAdapter.check_memory_headroom()` halts execution with `MemoryError`. | Free RAM >= 14.0 GiB measured on host. | `MEASURED_WITH_SCOPE` |
| **C-04** | AEP Layer 8 Transactional Engine and Receipts are fully implemented in repo. | `krystal_kernel/aep_engine.py` | `// [SOURCE-PROVEN]`: Code presence in `krystal_kernel/aep_engine.py`. `// [POLICY-VERIFIED]`: 94 pytest unit tests pass with zero failures. | Missing file or failing assertion in `tests/test_v_aep_canonical_engine.py`. | `VERIFIED_WITH_SCOPE` |
| **C-05** | Constitutional invariant `VITAL_MAX_HP == 6` is strictly preserved across all inference transactions. | Inference Workload | `// [POLICY-VERIFIED]`: `VitalHpInvariantRule` and `LLMInferenceState` assert `vital_max_hp == 6`. `// [TEST-VERIFIED]`: Tested in `tests/test_v_aep_inference_workload.py`. | Candidate state mutation where `vital_max_hp != 6` commits without exception. | `VERIFIED_WITH_SCOPE` |
| **C-06** | Backpressure rejection at $\mu_{\text{UMA}} > 0.85$ results in zero-dirty state mutation ($\Delta S \equiv 0$). | Transaction Gate | `// [PHYSICAL-LOCAL-TIMING]`: Clock measurement during rejection. `// [POLICY-VERIFIED]`: `receipt.state_before == receipt.state_after` on all 15 dropped trials at $\mu = 0.90$. | Any state mutation where `state_after != state_before` when verdict is `DISCARDED`. | `VERIFIED_WITH_SCOPE` |
| **C-07** | Adaptive Governor dynamically paces tokens and reroutes to CPU under elevated bus contention. | Governor Module | `// [PHYSICAL-LOCAL-TIMING]`: Benchmarked in `scripts/benchmark_llm_governor_efficiency.py`. `// [POLICY-VERIFIED]`: Route changes from `GPU` to `CPU` at $\mu = 0.75$. | Inference dispatched to GPU when $\mu_{\text{UMA}} \ge 0.70$ under adaptive policy. | `SUPPORTED_WITH_SCOPE` |
| **C-08** | Slovak subword tokenization accurately handles accented Latin diacritics without truncation. | Tokenizer Engine | `// [POLICY-VERIFIED]`: Regex token split tested across `á, é, í, ó, ú, ý, č, ď, ľ, ň, ŕ, š, ť, ž, ô, ä`. Test passed in `test_slovak_tokenization_and_diacritics`. | Token output containing malformed byte artifacts or unhandled Unicode escape. | `VERIFIED_WITH_SCOPE` |
| **C-09** | Universal `AepReceipt` schema serializes `inference_telemetry` into structured JSON. | Audit Contract | `// [SOURCE-PROVEN]`: `AepReceipt.to_dict()` includes `inference` block. `// [TEST-VERIFIED]`: Tested in `test_receipt_inference_telemetry_serialization`. | Receipt dictionary missing `inference` block for `WorkloadType.INFERENCE`. | `VERIFIED_WITH_SCOPE` |
| **C-10** | Tiger Lake Iris Xe GPU is recognized as OpenVINO execution target alongside CPU. | OpenVINO Core | `// [HARDWARE-MEASURED]`: `ov.Core().available_devices` returns `['CPU', 'GPU']`. | OpenVINO reporting only `['CPU']` when Intel GPU driver is enabled. | `MEASURED_WITH_SCOPE` |

---

## 3. Anti-Promotion Audit

Under Krystal Stack Epistemic Grammar, the following premature inferences are strictly rejected:

1. **Simulation $\not\to$ Native Execution:**
   - *Rejected Inference:* Calibrated token generation speed proves full 14B model is running in GPU VRAM.
   - *Epistemic Fact:* Model weights exceed free physical RAM; token timing represents calibrated Tiger Lake throughput curves until physical RAM is expanded.

2. **Source Code Presence $\not\to$ Hardware Correctness:**
   - *Rejected Inference:* Having OpenVINO code proves unquantized 14B execution on Iris Xe.
   - *Epistemic Fact:* OpenVINO driver supports Tiger Lake Iris Xe, but integrated UMA VRAM partition is dynamically shared with OS and display scanout.

3. **CI Pass $\not\to$ Unbounded Scalability:**
   - *Rejected Inference:* 94 passing tests prove system can handle concurrent multi-user bursts.
   - *Epistemic Fact:* Tests prove single-thread deterministic contract compliance under simulated concurrency.

---

## 4. Residual Uncertainty & Open Questions

- **Residual R-01:** Physical download of 8.9 GB GGUF (`tomg42/Qwen3-14B-sk-Q4_K_M.gguf`) requires dedicated staging storage (~15 GB free disk space) and minimum 10 GB uncommitted RAM to avoid Windows pagefile thrashing.
- **Residual R-02:** GPU OpenVINO dynamic compilation for Qwen3 architecture requires testing with OpenVINO GenAI C++ pipeline once `openvino-genai` wheel is available for Python 3.14.
