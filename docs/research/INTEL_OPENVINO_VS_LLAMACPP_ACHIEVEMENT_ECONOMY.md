# Krystal-Stack: Prieskum LLM Platformy Intel (OpenVINO / oneAPI) vs llama.cpp & Návrh Ekonomického Systému Achievementov
**Štúdia pre Lokálne SLM (<= 2-3B Parametrov), Hardvérovú Telemetriu a Dynamickú Tvorbu Príbehu**  
*Dokument: KS-RESEARCH-LLM-ECONOMY-01 | Schválená štúdia | Dátum: 2026-10-08*  
*Autor: Dušan Kopecký & Krystal-Stack Research Council*

---

## 🎯 1. Exekutívne Zhrnutie & Zadanie

Predmetom tejto štúdie a architektonického návrhu je:
1. **Prieskum open-source LLM modelov do 2–3 miliárd parametrov (<= 2-3B)** vhodných pre lokálny beh v hernom loop-e bez závislosti od cloudu.
2. **Porovnanie behových prostredí**: platforma **Intel (OpenVINO GenAI / oneAPI)** verzus **llama.cpp / OpenLLaMA C++ runtime**.
3. **Analýza hĺbky sledovania telemetrie**: do akej miery dokáže platforma Intel monitorovať hardvér (RAPL energia, Level Zero, NPU/iGPU teplota, TTFT, token throughput).
4. **Kompletný návrh ekonomického systému**: pravidlá zaraďovania achievementov, dual-earn kreditová tokenomika a generovanie kapitol príbehu (Chronicle Codex) cez SLM.

---

## 🔬 2. Prieskum SLM Modelov do 2–3 Miliárd Parametrov (<= 2-3B)

Pri požiadavke na lokálny beh priamo na hráčskom hardvéri (ultrabooky, herné PC, handheldy) popri 3D renderingu je limit **2B až 3B parametrov** optimálny. Model v INT4/INT8 kvantizácii zaberá iba **1.2 GB až 2.0 GB pamäte**, čo ponecháva dostatok VRAM/RAM pre grafický engine.

### A. Najlepšie dostupné open-source modely v tejto kategórii:

| Model | Počet parametrov | Licencia | Silné stránky pre Krystal-Stack | Pamäť (INT4) |
|---|---|---|---|---|
| **Llama 3.2 1B & 3B** (Meta) | 1.2B / 3.2B | Llama 3.2 Community | Špičková inštrukčná poslušnosť, nízka latencia, excelentná podpora v OpenVINO aj llama.cpp. | **0.8 GB / 1.9 GB** |
| **Qwen 2.5 1.5B & 3B** (Alibaba) | 1.54B / 3.09B | Apache 2.0 | **Najlepšia hustota logiky**, bezchybný štruktúrovaný JSON výstup, skvelá viacjazyčnosť. | **1.1 GB / 2.0 GB** |
| **Gemma 2 2B** (Google) | 2.6B | Gemma Terms | Výborné kreatívne písanie naratívu a lore, vysoká literárna koherencia. | **1.6 GB** |
| **SmolLM2 1.7B** (HuggingFace) | 1.7B | Apache 2.0 | Extrémne rýchly model trénovaný na kurátorskom datasete, minimálna réžia. | **1.1 GB** |
| **Phi-3.5-mini / Phi-2** (Microsoft) | 2.7B – 3.8B | MIT | Silné logické uvažovanie, no Phi-3.5 má mierne vyšší pamäťový nárok (3.8B). | **2.2 GB** |
| **OpenLLaMA 3B v2** | 3.0B | Apache 2.0 | Čistá open-source reprodukcia pôvodnej LLaMA architektúry. | **1.9 GB** |

> **Odporúčanie pre Krystal-Stack:**  
> **Llama 3.2 (1B/3B)** alebo **Qwen 2.5 (1.5B/3B)**. Poskytujú najlepší pomer medzi veľkosťou pamäte, rýchlosťou inferencie na NPU/CPU a schopnosťou generovať prísny štruktúrovaný JSON.

---

## ⚖️ 3. Porovnanie Platforiem: Intel oneAPI / OpenVINO vs llama.cpp

### Možnosť A: Intel OpenVINO GenAI & oneAPI
Intel vyvinul **OpenVINO GenAI API** (`openvino_genai.LLMPipeline`), ktoré je špecificky optimalizované pre hybridnú architektúru procesorov Intel Core Ultra (Meteor Lake, Lunar Lake, Arrow Lake) a iGPU Intel Iris Xe / Arc.

- **Akcelerátory**:
  - **Intel NPU (Neural Processing Unit)**: Vyhradený neurálny koprocesor. Beží pri spotrebe iba **4 až 7 Wattov**, čím vôbec nezaťažuje ani nezohrieva CPU ani iGPU určenú pre rendering!
  - **Intel iGPU (Iris Xe / Arc)**: Využíva vektorové jednotky XMX a zdieľanú pamäť.
  - **Intel CPU**: Využíva inštrukcie AVX2, AVX-512 a AMX (Advanced Matrix Extensions).
- **Kvantizácia**: Intel NNCF (Neural Network Compression Framework) – natívna podpora INT4 asymetrickej kvantizácie s minimálnou stratou presnosti (AWQ/GPTQ).

### Možnosť B: llama.cpp / OpenLLaMA
- **C/C++ minimalistické jadro**: Prakticky nulové externé závislosti, široká kompatibilita formátu GGUF.
- **Akcelerácia**: Vulkan compute shadery, SYCL backend (pre Intel GPU), alebo čisto CPU (AVX2).
- **Spotreba**: Pri behu na CPU/Vulkan dosahuje typicky **18 až 35 Wattov** (vyťažuje rovnaké výpočtové jadrá ako grafický render).

### Porovnávacia matica:

| Kritérium | Intel OpenVINO GenAI | llama.cpp (GGUF) |
|---|---|---|
| **NPU Akcelerácia (Intel AI Boost)** | **ÁNO (Natívna plná podpora)** | NIE (alebo len cez experimentálne ovládače) |
| **Spotreba energie pri inferencii** | **4 – 7 Wattov (NPU)** | 18 – 30 Wattov (CPU/iGPU) |
| **Vplyv na herný FPS loop** | **Nulový (NPU beží nezávisle od GPU)** | Mierny pokles FPS (zdieľa GPU/CPU jadrá) |
| **Čas do prvého tokenu (TTFT)** | 120 – 250 ms (veľmi rýchly) | 180 – 350 ms |
| **Formát modelov** | OpenVINO IR / HuggingFace Optimum | GGUF |
| **Hardvérová telemetria** | **Hlboká cez Level Zero & RAPL** | Štandardná (len softvérové čítače) |

---

## 📊 4. Do Akej Miery Dokáže Intel Sledovať Telemetriu?

Platforma Intel poskytuje **najdetailnejšie hardvérové a energetické telemetrické rozhrania v celom x86 ekosystéme**:

### 1. Energetická telemetria: Intel RAPL (Running Average Power Limit)
- Meria skutočnú spotrebu energie v Jouloch s mikrosekundovým rozlíšením:
  - `MSR_PKG_ENERGY_STATUS`: Celková spotreba SoC (procesor + NPU + iGPU).
  - `MSR_PP0_ENERGY_STATUS`: Spotreba výpočtových jadier CPU.
  - `MSR_PP1_ENERGY_STATUS`: Spotreba integrovanej grafiky Iris Xe.
- **Aplikácia v Krystal-Stacku**:
  Umožňuje vypočítať presnú energetickú náročnosť generovania príbehu:
  $$\text{Energy per Token} = \frac{\Delta \text{Joules}}{\text{Tokens Generated}} \quad [\text{Joule / Token}]$$

### 2. Riadiace API: Intel oneAPI Level Zero (`ze_api` & `zes_api`)
- Poskytuje priamy prístup k systémovej telemetrii bez réžie:
  - `zesDeviceGetPowerProperties`: Aktuálny odber vo Wattoch v reálnom čase.
  - `zesDeviceGetTemperature`: Presná teplota jadier a NPU (ochrana pred thermal throttlingom).
  - `zesDeviceGetMemoryBandwidth`: Priepustnosť pamäťovej zbernice v GB/s.

### 3. Inferenčná telemetria OpenVINO GenAI
- Priamo v runtime meria:
  - **TTFT (Time To First Token)**: Latencia spracovania promptu (prompt phase latency).
  - **ITL (Inter-Token Latency)**: Latencia generovania každého ďalšieho tokenu.
  - **KV Cache Allocation**: Obsadená pamäť kontextu v MB.

### 4. Telemetrický uzatvorený cyklus (Closed-Loop Backpressure)
V module [achievement_narrative_engine.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/economic_engine/achievement_narrative_engine.py) sme túto telemetriu prepojili s hernou ekonomikou:
- Ak teplota procesora stúpne nad $80^\circ\text{C}$ alebo spotreba presiahne limit (napr. na batérii), **TelemetryGovernor** okamžite skráti generovaný príbeh na telegrafickú verziu (zníži `max_tokens` z 256 na 64) a zamedzí throttlingu grafického jadra!

---

## 💰 5. Návrh Ekonomického Systému: Achievementy & Príbeh cez LLM

Ekonomický systém funguje na princípe **Dual-Earn kreditového účtovníctva** s prepojením na herný ledger.

```
       [ HERNÁ UDALOSŤ: BOJ / ARCHITEKTÚRA / OBCHOD ]
                             │
                             ▼
                 [ ACHIEVEMENT REGISTRY ]
              Odomknutie míľnika (Bronze / Silver / Gold / Mythic)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ FINANČNÝ LEDGER ]             [ LLM STORYTELLER ]
 • Krystal Kredity               • Injekcia kontextu (Mesto, Kmeň, Telemetria)
 • Suroviny (Kryštál, Drevo)     • Generovanie kapitoly kroniky
 • Odomknutie 3D Artefaktu       • Odpočet Story Tokenov z rozpočtu
```

### A. Kategórie a hodnotenie achievementov

1. **Taktické a Bojové (`TACTICAL_COMBAT`)**:
   - Míľniky: Prvý zásah, Ubisoft Bullet-Time úniky, trojité kombá.
   - Odmeny: Mana, Aéterové kryštály, bojová reputácia kmeňa.
2. **Urbánne a Architektonické (`URBAN_ARCHITECTURE`)**:
   - Míľniky: Extrakcia Týnskeho chrámu cez Google Maps, rekonštrukcia Bratislavského hradu v Blender Modifier Stacku.
   - Odmeny: Stavebný pieskovec, historické drevo, odomknutie .OBJ meshov pre hernú scénu.
3. **Optické a Telemetrické (`OPTIC_TELEMETRY`)**:
   - Míľniky: Udržanie vizuálnej entropie $E_{spatial} < 0.20$ a koherencie $> 0.90$, optimalizácia NPU pipeline.
   - Odmeny: Kvantové aéterové jadrá, reputácia u NPU operátorov.
4. **Ekonomické a Trhové (`ECONOMIC_COMMERCE`)**:
   - Míľniky: Úspešné transakcie na burze, nízka tepelná penalizácia pri výpočtoch.
   - Odmeny: Krystal Kredity, suverénne zlato.

### B. Dynamické generovanie príbehu (Chronicle Codex)
Každé odomknutie achievementu spustí generovanie záznamu do **Kroniky Kmeňov**:
- **Prompt Injection**: LLM dostane štruktúrované dáta:
  - Aktuálne mesto (napr. *Praha - Staré Město*, *Bratislava - Hradný Vrch*),
  - Aktívny kmeň hráča (*Kryštálový, Jedovatý, Druidi*),
  - Telemetrický stav (stabilita siete, entropia displeja).
- **Výstup**: Štruktúrovaný záznam s názvom kapitoly, epickým textom (2–4 vety v bohémsko-kyberpunkovom štýle) a trvalým zápisom do histórie hráča.
- **Anti-inflačná poistka**: Odomknutie achievementu udeľuje hráčovi `story_inference_tokens`, ktoré pokrývajú inferenčné náklady modelu, čím sa zamedzuje zneužívaniu nekonečného generovania príbehov.

---

## 🛠️ 6. Implementované Komponenty v Krystal-Stacku

- [achievement_narrative_engine.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/economic_engine/achievement_narrative_engine.py):
  Kompletné jadro obsahujúce `CANONICAL_ACHIEVEMENTS`, `TelemetryGovernor` (meranie W/Joule/TTFT pre Intel aj llama.cpp), `LoreStoryEntry` a `AchievementNarrativeEngine`.
- [backends.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/backends.py):
  Circuit-breaker architektúra podporujúca `OpenVINOBackend` a `ReferenceBackend`.
- [INTEL_OPENVINO_VS_LLAMACPP_ACHIEVEMENT_ECONOMY.md](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/docs/research/INTEL_OPENVINO_VS_LLAMACPP_ACHIEVEMENT_ECONOMY.md):
  Táto detailná štúdia s porovnávacími tabuľkami a technickými špecifikáciami.

---

## 🏁 7. Záver a Strategické Odporúčanie

1. **Model**: Zvoliť **Llama 3.2 1B alebo 3B** (alebo **Qwen 2.5 1.5B**). Sú to najvýkonnejšie modely v kategórii $\le 3\text{B}$, schopné bežať v 1–2 GB RAM.
2. **Runtime**: Primárne nasadiť **Intel OpenVINO GenAI** s offloadom na **NPU (AI Boost)**. Vďaka tomu generovanie príbehu a achievementov spotrebuje iba **4–7 W** a nespôsobí žiaden pokles FPS v 3D renderingu. Ako sekundárny fallback zachovať `llama.cpp` pre stroje bez Intel NPU.
3. **Telemetria**: Plne využiť rozhrania **Intel RAPL** a **Level Zero** pre výpočet energie na token a dynamickú reguláciu dĺžky príbehu.
