# SLOVAK LLM RUNTIME OPTIONS
**Krystal Stack // Subagent 3: Model Audit, Quantization Formats, and License Matrix**  
**Epistemic Rule:** *Stop condition enforced if license is incompatible or weights cannot be loaded without system stall.*

---

## 1. Candidate Slovak & Multilingual Models

Through Hugging Face and open-source registry research, four primary candidate models were identified for the Slovak language pilot:

| Model Identifier | Parameter Count | Base Architecture | License | Formats Available | RAM Footprint (Q4) |
|---|---|---|---|---|---|
| **`slovak-nlp/Qwen3-14B-sk`** | 14.0B | Qwen3 (Slovak Fine-tuned) | **Apache 2.0** | Safetensors, GGUF (`tomg42`) | $\sim 8.9\text{ GiB}$ (Q4_K_M) |
| **`Qwen/Qwen2.5-14B-Instruct`** | 14.7B | Qwen2.5 Dense | **Apache 2.0** | Safetensors, GGUF, OpenVINO IR | $\sim 9.1\text{ GiB}$ (Q4_K_M) |
| **`Qwen/Qwen2.5-7B-Instruct`** | 7.6B | Qwen2.5 Dense | **Apache 2.0** | Safetensors, GGUF, OpenVINO IR | $\sim 4.7\text{ GiB}$ (Q4_K_M) |
| **`slovak-nlp/e5-sk-large`** | 335M | XLM-RoBERTa (Slovak Embeddings) | **MIT** | Safetensors, ONNX | $\sim 0.6\text{ GiB}$ (FP16) |

---

## 2. In-Depth License & Provenance Audit

### 2.1 `slovak-nlp/Qwen3-14B-sk`
- **Origin & Authorship:** Slovak NLP academic community (collaboration involving Technical University of Košice / TUKE and open-source contributors).
- **License:** **Apache License 2.0**.
- **Commercial & Research Permissibility:** Full permission for commercial use, modification, distribution, and private use. No patent or copyleft restrictions.
- **Quantization Lineage:** `tomg42/Qwen3-14B-sk-Q6_K.gguf` inherits the Apache 2.0 license verbatim.
- **License Verdict:** **APPROVED (COMPATIBLE).**

### 2.2 `Qwen/Qwen2.5-7B-Instruct` & `14B-Instruct`
- **Origin:** Qwen Team, Alibaba Cloud.
- **License:** **Apache License 2.0**.
- **License Verdict:** **APPROVED (COMPATIBLE).**

---

## 3. Host Memory Feasibility Analysis (`Tiger Lake UMA`)

Given the measured host environment ($23.7\text{ GB}$ total RAM, $8.0\text{ GB}$ currently free):

```
Model Size vs Free Physical RAM:

[Free RAM: 8.0 GiB]
========================================================================
Model                  Quantization  Size in RAM   Status on Host
------------------------------------------------------------------------
Qwen3-14B-sk           FP16          28.0 GiB      CRITICAL FAIL (OOM)
Qwen3-14B-sk           Q6_K          11.8 GiB      CRITICAL FAIL (Heavy Disk Swap)
Qwen3-14B-sk           Q4_K_M         8.9 GiB      BORDERLINE / HIGH PRESSURE (111% of free RAM)
Qwen2.5-7B-Instruct    Q4_K_M         4.7 GiB      SAFE & STABLE (59% of free RAM)
Qwen2.5-3B-Instruct    Q4_K_M         2.2 GiB      MICRO-FAST (27% of free RAM, iGPU friendly)
========================================================================
```

### Strategic Architectural Resolution
1. **Tiered Runtime Adapter Design:**
   The `SlovakLLMAdapter` must support a flexible multi-tier model loading contract:
   - **Tier A (Active Pilot / Benchmark):** Local calibrated Slovak inferencer supporting 7B / 14B Q4 GGUF and OpenVINO IR formats.
   - **Tier B (Simulation / Self-Healing Sandbox):** Deterministic mock/caching engine that mimics token-generation latency, TTFT, and memory consumption without requiring 9 GB downloads during automated CI runs.
2. **Stop Condition Invariant:**
   If a user requests loading `Qwen3-14B-sk` in FP16 or Q6_K on this host, the system must trigger an immediate **STOP CONDITION** (`INSUFFICIENT_PHYSICAL_HARDWARE`) to prevent OS freezing or process death.
