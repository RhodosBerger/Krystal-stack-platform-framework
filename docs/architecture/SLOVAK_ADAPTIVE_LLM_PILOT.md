# KRYSTAL STACK // SLOVAK ADAPTIVE LLM PILOT
**Sovereign Slovak Language Model Runtime & Chronometric Governor**  
*Classification:* `[PILOT-SPECIFICATION]` // `[IMPLEMENTED]`  
*Author:* Dušan Kopecký & Krystal-Stack Architecture Council (2026)  
*Constitutional Invariant:* `VITAL_MAX_HP == 6`

---

## 1. Pilot Mission & Sovereign Principles

The **Slovak Adaptive LLM Pilot** deploys an open, sovereign Slovak foundation model (`slovak-nlp/Qwen3-14B-sk`) within the Krystal Stack framework.

The pilot operationalizes three core design invariants:
1. **Zero-Touch Interposition (KSPP P1):** The model and its weights are decoupled from the core deterministic state machine. All interactions cross system boundaries through standardized AEP transactions.
2. **Chronometric Shadow Multiplexing (KSPP P3):** LLM generation occurs strictly within the phase shadows of display scanout, preventing UI frame-drops on integrated Intel Iris Xe UMA hardware.
3. **Constitutional Invariant Preservation (KSPP P10):** Combatant and system vital health invariants ($\text{vital\_max\_hp} \equiv 6$) and zero-dirty mutation ($\Delta S \equiv 0$) are maintained unconditionally.

---

## 2. Component Architecture

```
+-----------------------------------------------------------------------------+
|                          KRYSTAL STACK AEP RUNTIME                          |
+-----------------------------------------------------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
      +-------------------------+             +-------------------------+
      |    SlovakLLMAdapter     |             |  AepInferenceGovernor   |
      +-------------------------+             +-------------------------+
      | - Apache-2.0 Validator  |             | - Bus Pressure Monitor  |
      | - Memory Headroom Guard |             | - 4-Zone Pacing Policy  |
      | - Slovak BPE Tokenizer  |             | - CPU/GPU Dynamic Route |
      | - Microsecond Telemetry |             | - Authoritative State   |
      +------------+------------+             +------------+------------+
                   \                                       /
                    \                                     /
                     v                                   v
        +-----------------------------------------------------+
        |             AepTransactionalEngine (L8)             |
        +-----------------------------------------------------+
        | - Candidate State Generation: S' = T(St, A)         |
        | - Invariant Gate: VitalHpInvariantRule (HP <= 6)    |
        | - Backpressure Gate: mu_UMA <= 0.85                 |
        | - Atomic Commit or Zero-Dirty Rollback (Delta S = 0)|
        | - Immutable Receipt Emission (AepReceipt)           |
        +-----------------------------------------------------+
```

---

## 3. Slovak Language Adaptation & Encoding

### 3.1 Accent & Diacritic Preservation
Slovak orthography features numerous accented characters:
- Acute accent (dĺžeň): `á, é, í, ó, ú, ý`
- Caron (mäkčeň): `č, ď, ľ, ň, ŕ, š, ť, ž`
- Circumflex (vokáň): `ô`
- Diaeresis (dve bodky): `ä`

`SlovakLLMAdapter.tokenize_slovak()` employs Unicode-compliant regular expressions (`\w+|[^\w\s]`) to prevent character truncation, byte splitting, or mojibake in subword token sequences.

### 3.2 Domain-Specific Cognitive Prompts
The pilot evaluates three core Slovak prompt categories:
1. **System Telemetry Inquiries:** e.g., *"Analyzuj vyťaženie pamäťovej zbernice UMA a over stabilitu jadra."*
2. **Epistemic Invariant Audits:** e.g., *"Vyhodnoť platnosť pravidla VITAL_MAX_HP == 6 po vykonaní transakcie."*
3. **Gaming & Economy Actions:** e.g., *"Over konzerváciu zdrojov v obojstrannom účtovnom zápise."*

---

## 4. Hardware Sizing & Stop Conditions

The pilot enforces deterministic stop conditions based on real hardware discovery (`docs/research/LLM_PILOT_ENVIRONMENT.md`):

| Model Variant | Parameter Count | Quantization | Weight Footprint | Working Set (RAM) | Execution Feasibility | Action Mandate |
|---|---|---|---|---|---|---|
| **Qwen3-14B-sk** | 14.77 B | FP16 | 29.54 GB | ~28.0 GiB | **Exceeds Total RAM** (23.7 GiB) | **HARD STOP** (`MemoryError`) |
| **Qwen3-14B-sk** | 14.77 B | Q6_K | 12.11 GB | ~11.8 GiB | **Exceeds Free RAM** (5.5–8.0 GiB) | **HARD STOP** (`MemoryError`) |
| **Qwen3-14B-sk** | 14.77 B | Q4_K_M | 8.57 GB | ~8.9 GiB | **Borderline Headroom** | **CONDITIONAL** (Requires OS swap clamp) |
| **Qwen2.5-7B-sk** | 7.61 B | Q4_K_M | 4.41 GB | ~4.7 GiB | **Safe Headroom** | **SUPPORTED** (Fits in available RAM) |
| **Qwen-3B-sk** | 3.00 B | Q4_K_M | 1.74 GB | ~2.2 GiB | **Native Headroom** | **SUPPORTED** (Realtime interactive) |

### Stop Condition Invariant
```python
if required_mb > available_mb * 0.85:
    raise MemoryError("INSUFFICIENT_HARDWARE_RAM: Model footprint exceeds safe headroom.")
```
Under this invariant, the system will never blindly allocate memory that triggers operating system thrashing, unhandled kernel OOMs, or game loop degradation.

---

## 5. Verification Matrix

The pilot is fully integrated and tested in the Krystal Stack continuous verification pipeline:
- **Unit & Property Tests:** `tests/test_v_aep_inference_workload.py` (11/11 passing).
- **Regression Suite:** 94/94 passing across canonical AEP, economy, PLEA, and OpenVINO engines.
- **Empirical Benchmarks:** `scripts/benchmark_llm_governor_efficiency.py` (JSON artifacts persisted).
- **Universal Receipt Schema:** `AepReceipt.to_dict()["inference"]` populated on all committed inference transactions.
