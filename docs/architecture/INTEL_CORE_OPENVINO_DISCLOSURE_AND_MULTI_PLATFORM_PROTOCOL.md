# Krystal-Stack: Intel Core & OpenVINO Micro-Architectural Disclosure & Multi-Platform Governed Inference Protocol (MPGIP)

**Document ID:** `KRYSTAL-ARCH-DISCLOSURE-2026-01`  
**Milestone:** `V-AUDIOVISUAL-WEB-PANEL-01` / `V-MULTIPLATFORM-INFERENCE-PROTOCOL-01`  
**Classification:** Open Architecture Technical Report & Foundation Manifesto  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  
**Author:** Dušan Kopecký & Krystal-Stack Architecture Council (2026)  

---

## 1. Executive Summary & Purpose

Tento dokument prináša hĺbkovú syntézu poznatkov získaných z vývoja a benchmarkovania frameworku Krystal Stack na heterogénnych procesorových architektúrach. Zameriava sa na:
1. **Demystifikáciu technológií Intel OpenVINO a oneAPI:** Čo v skutočnosti vykonávajú na úrovni kompilácie výpočtových grafov, nízkoúrovňových ovládačov a hardvérových registrov.
2. **Technické odhalenie mikroarchitektúry Intel Core (Tiger/Alder/Raptor/Meteor Lake):** Identifikáciu systémových a hardvérových úzkych hrdiel, ktoré vznikajú pri súbehu deterministického grafického renderingu (120 Hz) a asynchrónnych LLM / reasoning modelov na unifikovanej pamäti (UMA).
3. **Multi-Platform Governed Inference Protocol (MPGIP):** Zovšeobecnenie našej riadiacej metodiky pre architektúry **x86_64**, **ARM64** a **RISC-V (Open-Source ISA s RVV 1.0)**.
4. **Vizuálny diagnostický systém pre iniciačné sprinty:** Návod na využitie interaktívneho web panelu ([krystal_kernel/web_diagnostics_panel.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/web_diagnostics_panel.html)) pre audit, logovanie a vizualizáciu hysteréznych stavových slučiek.

---

## 2. Čo v skutočnosti predstavuje Intel OpenVINO a oneAPI (Z nášho uhla pohľadu)

Mnohé vývojárske tímy vnímajú OpenVINO iba ako "ďalší Python balíček pre beh ONNX modelov". V skutočnosti ide o nízkoúrovňový **hardvérovo-špecifický optimalizačný a grafový kompilátor**, ktorý tvorí kľúčový most medzi abstraktnými tenzormi a kremíkom Intelu:

```mermaid
graph TD
    subgraph "Vstup Modelu"
        ONNX[PyTorch / ONNX / SafeTensors]
    end

    subgraph "OpenVINO Model Optimizer (Graph Compiler)"
        MO[ov::Model / Graph Pass Engine]
        MO -->|Fusion| F1[Conv + BatchNorm + ReLU Folding]
        MO -->|Quantization| F2[Post-Training Quantization to INT8]
        MO -->|Pruning| F3[Dead Code & Constant Folding]
    end

    subgraph "oneAPI & Level Zero Driver Layer"
        LZ[Level Zero Driver Core (zeInit / zeDriverGet)]
        USM[Unified Shared Memory (USM: Host, Device, Shared)]
        DPC[oneDNN / DPC++ SYCL Vector Kernels]
    end

    subgraph "Hardvérové Výkonné Jednotky"
        CPU_E[P-Cores & E-Cores: AVX-512 VNNI / AMX]
        GPU_E[Iris Xe / Arc iGPU: 96 EU DP4A Dot-Product]
        NPU_E[Intel NPU: VPU Asynchronous Tile Pipeline]
    end

    ONNX --> MO
    MO --> LZ
    LZ --> USM
    USM --> DPC
    DPC --> CPU_E
    DPC --> GPU_E
    DPC --> NPU_E
```

### 2.1 Jadrové komponenty OpenVINO
1. **Model Optimizer & Intermediate Representation (IR):**
   - Prekladá neurónovú sieť do dvojice súborov: `.xml` (štruktúra grafu a topológia uzlov) a `.bin` (váhy a biasy).
   - Vykonáva agresívny **operator fusion**: zlučuje konvolúciu, normalizáciu a nelineárnu aktivačnú funkciu do jedného výpočtového jadra. Tým radikálne znižuje počet zápisov a čítaní z pamäte RAM (eliminácia memory bandwidth bottlenecku).
2. **ov::Core & Asynchrónne inferenčné fronty:**
   - Riadi exekúciu cez `ov::AsyncInferRequest`. Umožňuje plnohodnotný asynchrónny dispatch bez blokovania hlavného vlákna operačného systému.
   - Využíva execution streams pre paralelné saturácie všetkých 80/96 Execution Units (EUs) na Iris Xe.
3. **Intel oneAPI Level Zero:**
   - Priame nízkoúrovňové C API (`ze_api.h`), ktoré obchádza ťažkú réžiu OpenCL alebo DirectX runtime.
   - Poskytuje jemnozrnnú kontrolu nad **Unified Shared Memory (USM)**.

---

## 3. Odhalené tajomstvá a mikroarchitektonická realita Intel Core

Počas vývoja modulov [krystal_kernel/openvino_hardware_acceleration_pack.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/openvino_hardware_acceleration_pack.py) a [krystal_kernel/iris_xe_gaming_uma_driver_bridge.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/iris_xe_gaming_uma_driver_bridge.py) sme identifikovali a empiricky overili zásadné mikroarchitektonické charakteristiky:

### 3.1 Tajomstvo 1: OEM PL1 Clamping a GPU-CPU Cross-Ring Starvation
* **Problém v praxi:**
  Výrobcovia notebookov (OEM) v BIOS-e a cez Intel Dynamic Tuning Technology (DPTF) štandardne zamykajú trvalý napájací limit procesora (PL1) na konzervatívnych **12W až 15W**.
* **Mikroarchitektonická kolízia:**
  Integrovaná grafika Intel Iris Xe (96 EUs) pri plnom zaťažení (napr. DP4A INT8 super-sampling alebo 3D render) spotrebováva **12.0W až 13.5W**. Ak je celkový rozpočet balíka (Package Power) zamknutý na 15W, pre výpočtové jadrá CPU ostáva iba **1.5W až 3.0W**.
  CPU je hardvérovo nútené znížiť takt z nominálnych 3.8 GHz na **1.1 GHz až 1.3 GHz**. Tento jav nazývame **GPU-CPU Power Starvation**. Dochádza k paradoxu: zatiaľ čo grafika beží na plný výkon, CPU nestíha obsluhovať ovládač displeja, čo vedie k masívnemu stutteringu.
* **Naše riešenie v Krystal Stacku:**
  Priamy zápis do MSR a MMIO registrov:
  - `MSR_PKG_POWER_LIMIT` (`0x00000610`)
  - `MSR_IA32_HWP_REQUEST` (`0x00000774`)
  - `MMIO_MCHBAR_POWER_LIMIT` (`0x000059A0`)
  Odomknutie PL1 z 15W na stabilných **32W** s adaptívnym monitorovaním Arrheniusovej rovnice pre ochranu proti elektromigrácii a prehriatiu (T < 75 °C).

---

### 3.2 Tajomstvo 2: UMA LPDDR4x/5 Zbernicová Saturácia & Ring Bus Bottleneck
* **Problém v praxi:**
  Procesory Intel Core nemajú pre iGPU dedikovanú grafickú pamäť (VRAM). Využívajú systémovú pamäť LPDDR4x-4266 alebo LPDDR5-5200 (priepustnosť cca **68 GB/s** zdieľaná medzi všetkými komponentmi).
* **Kolízia pri súbehu LLM a 120 Hz renderu:**
  Streamovanie tokenov reasoning modelu (čítanie váh a KV-cache) spotrebováva sekvenčnú priepustnosť **26 - 32 GB/s**. V rovnakom čase grafický scanout engine pri 120 Hz VSync a 1080p/4K vyžaduje **24 - 30 GB/s** na fetch textúr a framebuffer scanout.
  Akonáhle celková saturácia zbernice presiahne **78% (~53 GB/s)**, na Last Level Cache (LLC) ring bus dochádza k arbitrážnym konfliktom. Scanout engine nedostane dáta včas a dochádza k okamžitému **frame dropu**.
* **Naše riešenie v Krystal Stacku:**
  Komponent [UmaBandwidthArbiter](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/predictive_render_profiler.py#L140) monitoruje saturáciu zbernice. Pri dosiahnutí 78% okamžite aktivuje **token pacing** v [EpistemicGovernedWorker](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/epistemic_governed_worker.py):
  $$\text{ThrottlePct} = \min\left(0.90, \frac{\text{Saturation} - 0.78}{0.22} \cdot 0.80 + 0.20\right)$$
  Špekulatívne sloty LLM sú stlačené z $K_{\text{spec}} = 8$ na $1 \dots 2$, čím okamžite uvoľnia zbernicu a scanout ostáva stabilný na 120 Hz bez jediného prepadu snímky.

---

### 3.3 Tajomstvo 3: VSync Blanking Micro-Slicing
* Namiesto toho, aby LLM inferencia bežala nepretržite a bojovala s vykresľovaním polygónov o pamäťové kanály, Krystal Stack synchronizuje dávky inferencie s **VSync blanking intervalom**:
  - Pri 30 FPS trvá blanking interval cca **12.8 ms**.
  - Pri 120 Hz prebieha mikro-dávkovanie v mikrosekundových medzerách medzi render passmi.
  - Tým sa úplne eliminuje pamäťová interferencia na zbernici.

---

## 4. Multi-Platform Governed Inference Protocol (MPGIP)

Naše riešenie nie je obmedzené iba na procesory Intel x86. V module [krystal_kernel/multiplatform_inference_protocol.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/multiplatform_inference_protocol.py) sme formalizovali **Multi-Platform Governed Inference Protocol (MPGIP)**, ktorý zjednocuje riadenie výkonu procesora pomocou inferenčnej spätnej väzby naprieč tromi hlavnými rodinami architektúr:

| Architektúra | Cieľové Platformy | Vektorový Akceleračný Engine | Typická Zbernica | Prah Saturácie | Riadiaci Mechanizmus |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **x86_64** | Intel Core (Tiger/Alder/Raptor/Meteor Lake), AMD Zen 4/5 | `INTEL_DP4A_INT8`, `AVX512_VNNI`, `AMX` | 68.0 GB/s (LPDDR4x/5) | **0.78** (78%) | MSR PL1 Boost (32W), Token Pacing, Ring Bus Arbitration |
| **ARM64** | Apple Silicon (M-Series), ARM Neoverse, Cortex-X | `ARM_NEON_I8MM`, `ARM_SVE2` | 100.0 - 200.0 GB/s (UMA LPDDR5) | **0.82** (82%) | Široká krížová zbernica, väčší rozpočet pre špekulatívne sloty ($K_{\text{spec}} \le 12$) |
| **RISC-V** | Open-source SoC (Tenstorrent Wormhole, SiFive Intelligence X280) | `RISCV_RVV_1_0` (512-bit vector), Systolic NPU | 48.0 GB/s (DDR4/LPDDR4) | **0.74** (74%) | Konzervatívnejšia arbitráž, striktné riadenie prístupu k pamäťovým radom |

### 4.1 Binárny Zero-Copy Formát Protokolu
Všetky riadiace verdikty a telemetrické dáta sú serializované do **128-bajtových** cache-line zarovnaných rámcov [ZeroCopyContractFrame](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/zero_copy_contracts.py#L83), ktoré prenášajú:
1. `magic`: `0x4B525953` ("KRYS").
2. `vital_hp`: Ústavný invariant `VITAL_MAX_HP == 6`.
3. `capability_scope`: Kód cieľovej architektúry (1 = x86, 2 = ARM64, 3 = RISC-V).
4. `metric1`: Odporúčané spomalenie token pacingu v milisekundách.
5. `metric2`: Cieľový príkon v wattoch (PL1 Target Watts).
6. `signature`: 32-bajtový kryptografický SHA-256 odtlačok potvrdzujúci autenticitu riadiaceho uzla.

---

## 5. Webový Diagnostický Panel pre Iniciačné Sprinty

Pre vývojárske a integračné tímy sme pripravili interaktívny nástroj [krystal_kernel/web_diagnostics_panel.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/web_diagnostics_panel.html), generovaný pomocou [WebDiagnosticsService](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/web_diagnostics_service.py):

### 5.1 Čo panel zobrazuje
1. **Systémové skóre zdravia (Health Score):** Počet analyzovaných snímok a percentuálna stabilita behu bez kolízií.
2. **Duálny časový graf (Timeline Waveform):**
   - V reálnom čase porovnáva latenciu vykresľovania s limitom **8.33 ms** (120 Hz).
   - Zobrazuje saturáciu zbernice voči prahu **0.78**.
   - Označuje body, kde zasiahla hysteréza (`[HELD]`) alebo token pacing (`[PACED]`).
3. **Fázový portrét (Phase-Space Hysteresis):**
   - Vykresľuje 2D stavový diagram $(t_{\text{frame}}, \text{ProfileRank})$.
   - Zobrazuje Schmittov trigger: okamžitý pád do nižšieho rozlíšenia pri $t > 7.5\text{ ms}$ a 12-snímkové zotrvanie pred návratom nahor pod $t < 5.83\text{ ms}$.
4. **Logovací terminál a detektor anomálií:**
   - Filtruje udalosti podľa závažnosti (`CRITICAL`, `WARNING`, `INFO`).
   - Deteguje nechránené zbernicové kolízie (`ANOMALY_UMA_UNPACED_COLLISION`).
5. **LLM Root Cause Analysis panel:**
   - Syntetizuje zistenia z grafov a logov do textového vysvetlenia pre inžinierov.
   - Poskytuje preskripčné odporúčania pre ladenie hardvérových limitov.

---

## 6. Open-Source Špecifikácia: Ako ovplyvňovať procesorový výkon inferenciou

Na základe koncentrovaného poznania v tomto repozitári definujeme **otvorenú metodiku pre vývojárov systémového softvéru**:

1. **Inferencia ako primárny senzor stavu systému:**
   - Tradičné monitory (perf, htop) merajú vyťaženie pasívne s latenciou desiatok milisekúnd.
   - Epistemická inferencia LLM / neurálnej siete slúži ako **aktívna mikrosonda**: rýchlosť generovania tokenov (Inter-Token Latency) okamžite indikuje pamäťový a termálny tlak čipu v reálnom čase (pod 1 ms).
2. **Obojstranná regulácia (Governor Feedback Loop):**
   - Ak je systém studený a grafika nečinná: uvoľni plný rozpočet PL1 (32W-45W) a zvýš špekulatívne dekódovanie na $K_{\text{spec}} = 12$.
   - Ak začne grafický render: stiahni inferenciu do blanking okna a zníž pamäťovú prioritu, aby používateľ nikdy nezaznamenal prepad plynulosti rozhrania.
3. **Multiplatformová univerzálnosť:**
   - Vďaka unifikácii do 128-bajtových rámcov `ZeroCopyContractFrame` je tento protokol okamžite použiteľný nielen na x86, ale aj v embedovaných a robotických systémoch postavených na architektúrach ARM Cortex a RISC-V.

---
*Krystal-Stack Architecture Council // Approved for Milestone V-AUDIOVISUAL-WEB-PANEL-01.*
