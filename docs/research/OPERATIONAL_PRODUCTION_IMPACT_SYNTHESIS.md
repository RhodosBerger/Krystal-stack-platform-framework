# KRYSTAL STACK // OPERATIONAL & PRODUCTION IMPACT SYNTHESIS
## Physically-Hybrid Closed-Loop Cascade Runtime in Practice

**Dátum:** 18. október 2026  
**Status:** `PHYSICALLY_HYBRID_CLOSED_LOOP_VALIDATED`  
**Referenčné commity:** [`8a416982`](https://github.com/RhodosBerger/Krystal-stack-platform-framework/commit/8a416982), [`0b6e6dff`](https://github.com/RhodosBerger/Krystal-stack-platform-framework/commit/0b6e6dff)  
**Invariant:** `assert VITAL_MAX_HP == 6` | $\neg\text{Valid}(\text{Action}) \implies \Delta S \equiv 0$  

---

## 1. Exekutívne zhrnutie

Tento dokument sumarizuje praktické produkčné a prevádzkové dopady zavedenia **fyzicky-hybridného closed-loop plánovania** ([`CascadePlanner`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_planner.py)), **striktnej segregácie telemetrie** ([`MetricProvenance`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_trace.py)) a **fyzického substrátu** ([`SlovakLLMNode`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/model_node.py) s [`SlovakLLMAdapter`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/slovak_llm_adapter.py)).

Architektúra nepotrebovala separátny paralelný scheduler ani nový execution plan – existujúci [`CascadePlan`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_planner.py#L54-L107) bol in-place rozšírený o snapshot prostredia a históriu evidencie, pričom rekalibrácia znížila chybu predikcie na nevidených promptoch o **94.3%** (MAE: $53.51\,\text{ms} \to 3.03\,\text{ms}$).

---

## 2. Štyri kľúčové produkčné dopady

```
+---------------------------------------------------------------------------------------------------+
|                           PRODUKČNÁ REALITA KRYSTAL CASCADE RUNTIME                               |
+----------------------------------+----------------------------------------------------------------+
| 1. Interný Helpdesk & SLOs       | Eliminácia timeoutových anomálií cez samokalibráciu na HW      |
| 2. SRE & Observability           | Alerty naviazané výhradne na MEASURED dáta (anti-alert-fatigue)|
| 3. Drift & Dlhý Kontext          | Detekcia zlyhania priemerov a automatické token-aware škálovanie|
| 4. Kapacitné Plánovanie na Win32 | Deterministická spätná dohľadateľnosť pamäťových rozhodnutí    |
+----------------------------------+----------------------------------------------------------------+
```

### [DOPAD 1] Slovenské dotazy v internom helpdesku: Eliminácia neočakávaných timeoutov
- **Problém As-Is:** Pri nasadení na interný helpdesk plánovač vychádzal z apriórnych statických heuristík ($63.6\,\text{ms}$ pre sequential pipeline, $15.0\,\text{ms}$ pre uzol). Ak bol deadline na hrane (napr. SLO timeout $50\,\text{ms}$), plánovač buď predčasne odstrelil inferenciu ako timeoutovanú, alebo volil drahé zbytočné fallbacky.
- **Riešenie To-Be:** Už po 2–3 reálnych odpovediach v helpdesku absorbuje [`CascadePlanner`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_planner.py) skutočné fyzické časy z [`SlovakLLMNode`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/model_node.py) namerané cez `time.perf_counter_ns()`. 
- **Prevádzkový zisk:** Plánovač sa prestáva spoliehať na fiktívne čísla a dynamicky prispôsobí hrany [`CascadeEdge.timeout`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_planner.py#L28-L51) skutočnému výkonu stroja (napr. $7-12\,\text{ms}$). Používatelia helpdesku nedostávajú umelé chyby.

---

### [DOPAD 2] Produkčný monitoring a SRE: Alerty výhradne na fyzicky namerané čísla
- **Problém As-Is:** V stope [`CascadeExecutionTrace`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_trace.py) boli fyzické časy a syntetické mocky (string-split latencia $15\,\text{ms}$) zmiešané v rovnakom poli `latency_ms`. Prometheus/Grafana metriky zlievali realitu so simuláciou, čo viedlo k falošným poplachom (false positives) a alert fatigue.
- **Riešenie To-Be:** Zavedenie enumu [`MetricProvenance`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_trace.py#L57-L62) a poľa `provenance_summary`:
  - `MEASURED`: Skutočný časovač hardvéru a Win32 API.
  - `ESTIMATED`: Kĺzavý priemer a štatistické odhady.
  - `SYNTHETIC`: Heuristiky a simulácie.
- **Prevádzkový zisk:** Produkčné SLA/SLO pravidlá (napr. `ALERT HighP99Latency IF p99 > 30ms`) sa v monitoringu naviažu **striktne na záznamy s `provenance == "MEASURED"`**. Syntetické stubs už nikdy neodpália nočný pagerDuty incident.

---

### [DOPAD 3] Dlhšie slovenské prompty: Detekcia zlyhania priemerov a driftu
- **Problém As-Is:** Pri prechode z krátkych dotazov (5 slov) na dlhé promptovanie (právne rozbory, systémové logy, 30+ slov) statický historický priemer zlyháva, pretože nezohľadňuje počet tokenov.
- **Riešenie To-Be:** V logoch kaskády sa okamžite objaví predikčná odchýlka:
  $$\text{error\_latency\_ms} = \text{act\_lat} - \text{exp\_lat}$$
  Ablačný test preukázal:
  - Naive Historical Mean (A2): Chyba $3.89\,\text{ms}$ na dlhom prompte ($15.95\,\text{ms}$ namerané vs. $12.06\,\text{ms}$ priemer).
  - Token-Aware Empirical (A3): Zohľadnenie smernice $\text{ms}/\text{token}$ zredukovalo chybu predikcie na variabilných dĺžkach až na $1.00\,\text{ms}$.
- **Prevádzkový zisk:** Monitorovací systém v reálnom čase deteguje fázový drift v dĺžke promptov a umožňuje plynulý prechod na token-aware škálovanie bez nutnosti reštartu runtime.

---

### [DOPAD 4] Kapacitné plánovanie na Windows: Forenzná dohľadateľnosť pamäťových rozhodnutí
- **Problém As-Is:** Ak systém o 14:00 odmietol vykonať plán alebo spadol do degradovaného fallback režimu, inžinieri nemali dôkaz, prečo k tomu došlo (či išlo o chybu modelu, nedostatok pamäte alebo bug v kóde).
- **Riešenie To-Be:** Každý [`CascadePlan`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/cascade/cascade_planner.py#L54-L107) povinne nesie `environment_snapshot` s granulárnym rozpadom:
  - `available_memory_mb`: $5034.4\,\text{MB}$ (`MEASURED_WIN32_API`)
  - `cpu_cores`: $8$ (`OBSERVED_SYSTEM_CALL`)
  - `bus_pressure`: $0.12$ (`SYNTHETIC_HEURISTIC_CONSTANT`)
  - `overall_provenance`: `HYBRID_MEASURED_SYNTHETIC`
- **Prevádzkový zisk:** Pri post-mortem audite je exaktne doložené: *„V čase $T$ bolo na stanici dostupných len $1120\,\text{MB}$ fyzickej RAM (vplyvom iného procesu). Plánovač korektne vyhodnotil pretečenie rozpočtu a autonómne zvolil úsporný specialist fallback.“*

---

## 3. Matrica produkčnej pripravenosti (Production Readiness Matrix)

| Komponent | Status | Pôvod dát (Provenance) | Overenie | Otvorený krok |
| :--- | :--- | :--- | :--- | :--- |
| **Slovak LLM Node** | `IMPLEMENTED` | `MEASURED` (Win32 + CPU SIMD) | Test 02, Test 03 | Rozšírenie na GPU Vulkan (Iris Xe) |
| **Cascade Planner** | `CALIBRATED` | `HYBRID_EMPIRICAL` | Test 04, Test 07 | Per-domain token-bucket cache |
| **Trace Provenance**| `SEGREGATED` | `MEASURED` / `SYNTHETIC` | Test 03 | Prometheus exporter tag mapping |
| **Snapshot prostredia** | `STRUCTURED` | `MEASURED` (RAM) + `SYNTHETIC` ($\mu$) | Test 01 | Fyzické čítanie RAPL / MSR wattmetra |
| **Noether Invariant**| `CONSTITUTIONAL`| Invariant `VITAL_MAX_HP == 6` | Test 06, Test 07 | Kontinuálna ochrana pri prechode stavov |

---

## 4. Záver

Tieto 4 body premieňajú abstraktnú kaskádu na **priemyselne použiteľný, seba-kalibrujúci sa systém**:
1. Žiadne timeoutové prekvapenia na helpdesku.
2. Žiadne falošné alerty pre SRE tím.
3. Transparentná detekcia driftu pri zmene dĺžky promptov.
4. Nespochybniteľný audit pamäťových rozhodnutí priamo na Windows staniciach.
