# Krystal RAG-Conditioned LLM Training & Adaptive Architecture Fabric

> **Účel Systému**: Trénovanie, jemné doladenie (fine-tuning) a priebežná adaptácia jazykových modelov (trieda Edge SLM ~1.5B) prispôsobených špecifickým účelom Krystal Stacku (procedurálna generácia, raymarching, shaderové uniformy, NPU akčné puzzle a epistemické invarianty) s dynamickou podmienenosťou na aktuálne dostupnej konfigurácii RAG modelu.

---

## 1. Základný Princíp: Adaptácia na Konfiguráciu RAG

V tradičnom ML prístupe je LLM trénovaný staticky bez ohľadu na to, či RAG vracia 4 kvalitné hybridné citácie alebo či je RAG subsystém preťažený, obmedzený pamäťovým tlakom UMA alebo úplne offline.

**Krystal RAG-Conditioned LLM Training Fabric** mení paradigmu:
$$P(y \mid x, \mathbf{c}_{\text{rag}}) = \text{Softmax}\left( \mathbf{W}_{\text{lm}} \cdot \text{LayerNorm}\left( \mathbf{h}_{\text{film}}(\mathbf{c}_{\text{rag}}) + \frac{\alpha(\mathbf{c}_{\text{rag}})}{r} \mathbf{x} \mathbf{A} \mathbf{B} \right) \right)$$

Kde $\mathbf{c}_{\text{rag}} \in \mathbb{R}^8$ je normalizovaný vektor reprezentujúci aktuálny stav RAG modelu:
- `retrieval_mode`: One-hot kódovanie (`BM25_LEXICAL`, `DENSE_COSINE`, `HYBRID_FUSION`, `DEGRADED_EMPTY`),
- `top_k_documents`: Počet vyhľadaných úsekov ($k \in [0, 8]$),
- `embedding_dimension`: Dimenzia vektorov ($128$ pre natívny micro-vektor, $768$ pre štandard),
- `relevance_threshold`: Prah kosínusovej podobnosti ($\theta \in [0.4, 0.9]$),
- `avg_search_latency_ms`: Nameraná latencia vyhľadávania,
- `cache_hit_rate`: Pomer zásahov v pamäťovej vyrovnávacej pamäti ($0.0 \dots 1.0$),
- `authority_boundary_enforced`: Striktné dodržanie invariantu `VITAL_MAX_HP == 6`.

---

## 2. Architektonické Komponenty

```mermaid
graph TD
    RAG_State[RAG Model Configuration c_rag in R^8] --> HyperNet[Hyper-Network 8 -> 32 -> 2*d]
    HyperNet --> FiLM[FiLM Modulation gamma, beta]
    HyperNet --> ElasticLoRA[Elastic LoRA Controller rank r in {4, 8, 16}]
    
    Tokens[Input Sequence w_1 ... w_t] --> Embedding[Embedding Table 128 x 64]
    Embedding --> Trunk[Causal Transformer Trunk]
    
    Trunk --> AdaptBlock[RAG-Conditioned Adaptive Block]
    FiLM --> AdaptBlock
    ElasticLoRA --> AdaptBlock
    
    AdaptBlock --> LayerNorm[Stabilized LayerNorm]
    
    LayerNorm --> Head1[Head 1: Causal LM Logits]
    LayerNorm --> Head2[Head 2: Citation Fidelity Head]
    LayerNorm --> Head3[Head 3: Invariant Hazard Head]
    LayerNorm --> Head4[Head 4: Epistemic Abstain Head]
```

### A. Hyper-Network a FiLM Modulácia ([`RagConditionedAdapterNetwork`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/krystal_rag_llm_trainer.py#L125-L200))
Hyper-sieť mapuje 8-rozmerný stav RAG na parametre Feature-wise Linear Modulation (FiLM):
$$\mathbf{h}' = \gamma(\mathbf{c}_{\text{rag}}) \odot \mathbf{h} + \beta(\mathbf{c}_{\text{rag}})$$
- Ak je RAG hybridný a presný, $\gamma$ a $\beta$ zosilňujú kontextové prepájanie a citation heads.
- Ak je RAG offline, FiLM vrstvy prepínajú model na parametrickú bázu vedomostí alebo iniciujú abstinenciu.

### B. Elastické LoRA (Elastic Rank Adaptation)
Kapacita nízkorangovej adaptácie sa prispôsobuje podľa dostupnosti RAG faktov:
- **High-Fidelity Hybrid ($k \ge 4$)**: $r = 4$, $\alpha = 1.0$ (model nepotrebuje ukladať fakty do váh, fakty dodáva RAG).
- **Dense/BM25 ($k = 2$)**: $r = 8$, $\alpha = 1.5$ (stredná miera parametrickej kompenzácie).
- **Degraded Offline Empty ($k = 0$)**: $r = 16$, $\alpha = 2.5$ (model aktivuje plnú parametrickú kapacitu).

### C. Multi-Task Stratová Funkcia
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + 1.2 \cdot \mathcal{L}_{\text{citation}} + 2.0 \cdot \mathcal{L}_{\text{hazard}} + 1.0 \cdot \mathcal{L}_{\text{abstain}}$$
- $\mathcal{L}_{\text{LM}}$: Krížová entropia na predikciu ďalšieho tokenu.
- $\mathcal{L}_{\text{citation}}$: Binárna krížová entropia nad maskou relevantných úsekov (bráni halucinovaným citáciám).
- $\mathcal{L}_{\text{hazard}}$: Trestá akýkoľvek pokus o generovanie neplatných systémových stavov mimo `VITAL_MAX_HP == 6`.
- $\mathcal{L}_{\text{abstain}}$: Kalibrovaná neistota – núti model generovať `EPI_UNVERIFIED` / `AUTH_DEFERRED`, ak RAG dôkazy chýbajú.

---

## 3. Výsledky Benchmarku naprieč Konfiguráciami RAG

Výsledky evaluácie na testovacej množine s **95% intervalmi spoľahlivosti (bootstrapping 500 opakovaní)** podľa smerníc `ml-best-practices`:

| Konfigurácia RAG | Režim | Top-$k$ | LoRA Rank | Perplexita (PPL) | 95% CI Perplexity | Citation F1 | Abstain Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **High-Fidelity Hybrid** | `HYBRID_FUSION` | 4 | **4** | **221.78** | **[127.36, 392.12]** | **0.4291** | 100% |
| **Low-Latency Dense** | `DENSE_COSINE` | 2 | **8** | 249.82 | [162.60, 372.04] | 0.2167 | 100% |
| **Hermetic BM25** | `BM25_LEXICAL` | 2 | **8** | 201.58 | [121.21, 326.63] | 0.2778 | 100% |
| **Degraded Offline** | `DEGRADED_EMPTY` | 0 | **16** | 284.71 | [176.23, 466.90] | 0.4111 | **100%** |

---

## 4. MCP a REST API Rozhranie

### A. MCP Nástroje v Antigravity IDE:
1. `krystal-llm-train-epoch`:
   Spúšťa trénovaciu epochu s AdamW/SGD gradientným krokom, vyhodnocuje trénovaciu a validačnú stratu, perplexitu a citačný F1-skóre.
2. `krystal-llm-rag-benchmark`:
   Spúšťa porovnávací benchmark natrénovaného modelu naprieč 4 kanonickými RAG presetmi s 95% intervalmi spoľahlivosti.

### B. REST Endpoints v `server.py`:
- `GET /api/llm/rag/benchmark`: Vracia kompletnú porovnávaciu správu s bootstrapovanými metrikami.
- `POST /api/llm/train/epoch`: Prijíma parametre `{ samples_count: 60, epoch: 1, learning_rate: 0.005 }` a vracia metriky epochy.

---

## 5. Invariantná Integrita

Všetky štruktúry, dátové vzorky, váhy adaptérov a výstupy striktne vynucujú platformový invariant:
$$\text{assert } VITAL\_MAX\_HP == 6$$
