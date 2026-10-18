# EXPERIMENT SPECIFICATION: E-05
# CROSS-ENVIRONMENT RUNTIME REPLICATION: Windows Native vs WSL2

**Experiment ID:** `E-05-CROSS-ENVIRONMENT-REPLICATION`  
**Parent Hypotheses:** H-01, H-02, H-03, H-04  
**Target Invariant:** `VITAL_MAX_HP == 6`, `StateImmutability (\Delta S = 0 \text{ on failure})`  
**Physical Hardware (Identical Silicon):** Intel Core i5-1135G7 (AVX-512 VNNI, 4C/8T) + Intel Iris Xe Graphics (80 EUs Gen12 Xe-LP)  
**Operating/Runtime Environments:**  
1. **Windows Native:** Windows 11 Pro 64-bit (NT Kernel 10.0.26100), Win32 API, OpenVINO 2026.4.1, CPU AVX-512 + Iris Xe GPU (Level Zero / OpenCL ICD).  
2. **WSL2 Linux:** Ubuntu on Linux Kernel `6.18.33.2-microsoft-standard-WSL2`, glibc / Linux syscalls, OpenVINO 2026.4.1, CPU AVX-512.  

---

## 1. Epistemický cieľ a výskumná otázka

> **Výskumná otázka:**  
> *„Zachovajú sa vzťahy medzi stratégiami (`CPU_ONLY`, `GPU_ONLY`, `GPU_PREFILL_CPU_DECODE`) a dokáže closed-loop planner po zmene operačnej a runtime vrstvy autonómne prekalibrovať policy bez ručného tuningu a bez chýb?“*

### Epistemické mantry a direktívy:
1. **Zákaz prenosu Windows výsledkov ako očakávanej pravdy:**  
   Windows výsledky tvoria výhradne **referenčný dataset**, nie predpis reality.
2. **Zákaz simulácie chýbajúceho hardvéru (Zero-Simulation Invariant):**  
   Ak GPU / Level Zero nie je vo WSL2 fyzicky exponované v používateľskom priestore (`core.available_devices == ['CPU']`), GPU behy sa označia ako `UNAVAILABLE` a **nikdy nesmú byť ticho simulované na CPU**.
3. **Nahradenie platfom-dependent probes:**  
   `GlobalMemoryStatusEx` a `psapi.GetProcessMemoryInfo` sú vo WSL2 nahradené natívnym `/proc/meminfo` (`MemAvailable`) a `/proc/self/statm` (`resident_pages * 4096`).
4. **Cieľ nie sú identické mikrosekundy:**  
   Hodnotí sa **relatívny profil**, **schopnosť adaptácie** a **policy agreement**.

---

## 2. Dvojfázový experimentálny protokol (Two-Phase Protocol)

### Fáza 1: Environment Qualification
Overenie integrity a limitov Linux/WSL2 prostredia pred spustením benchmarkov:
1. **CPU Inference Qualification:** Overenie inicializácie OpenVINO a behu tenzorových operácií s AVX-512.
2. **OpenVINO Device Discovery:** Dotaz na `core.available_devices`.
3. **GPU Physical Probe:** Dotaz na prítomnosť `/dev/dri`, `/dev/dxg` a OpenCL/Level-Zero ICD. Overenie, či je GPU deklarované alebo nedostupné.
4. **Model Loading & Compilation:** Načítanie skutočného Gemma modelu (`google/gemma-3-4b-it`, INT4, fingerprint `a6daa545d330753f`). Kompilácia CPU modelu; pokus o kompiláciu GPU modelu (ak absentuje, zachytiť a klasifikovať `UNAVAILABLE`).
5. **Warmup & Zero-Simulation Confirmation:** Overenie zahriatia a potvrdenie, že CPU behy skutočne vykonávajú inštrukcie AVX-512.

### Fáza 2: Replication Benchmarks
Spustenie replikácií experimentov E-01 až E-04 v prostredí WSL2:
1. **E-01 — Compile / First / Warm Stratifikácia:**  
   Meranie $\tau_{\text{compile}}$, $\tau_{\text{first}}$, $\tau_{\text{warm}}$ pre CPU. GPU sa explicitne označí ako `UNAVAILABLE`.
2. **E-02 — CPU vs GPU Cost Surface:**  
   Meranie mriežky promptov ($16, 64, 128$) a generovaných tokenov ($8, 32, 64, 128$) na CPU.
3. **E-03 — Phase-Aware Execution:**  
   Vyhodnotenie Candidate stratégií:
   - Candidate A: `CPU_ONLY` (fyzicky namerané na WSL2)
   - Candidate B: `GPU_ONLY` (`UNAVAILABLE`)
   - Candidate C: `GPU_PREFILL_CPU_DECODE` (`UNAVAILABLE`)
   - Candidate D: `BATCHED_GPU` (`UNAVAILABLE`)
   - Candidate E: `ADAPTIVE_PLANNER` (Closed-loop planner vyhodnotí stav prostredia `available_devices == ['CPU']` a autonómne vyberie optimálnu uskutočniteľnú stratégiu `CPU_ONLY` bez pádu a bez degradácie).
4. **E-04 — Boundary Handoff Re-interpretation:**  
   Analýza rozhrania vo WSL2: v prostredí bez samostatného GPU zariadenia je handoff CPU-to-CPU v pamäti cache residentný ($\tau_{\text{handoff}} \approx 0$).

---

## 3. Záväzná štruktúra telemetrického receiptu (Receipt Schema)

Každý vygenerovaný záznam musí obsahovať minimálne:
```json
{
  "environment": "WSL2",
  "kernel_version": "6.18.33.2-microsoft-standard-WSL2",
  "distro": "Ubuntu",
  "cpu_model": "11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz",
  "available_memory_mb": 10887.1,
  "openvino_version": "2026.4.1",
  "available_devices": ["CPU"],
  "requested_device": "CPU",
  "actual_device": "CPU",
  "driver_backend": "OpenVINO CPU Plugin (AVX-512)",
  "model_fingerprint": "a6daa545d330753f",
  "vital_max_hp": 6
}
```

---

## 4. Arbitračné metriky a rozhodovacie kritériá

### 1. Policy Agreement:
$$\text{Agreement} = \frac{\#\{\text{workloads where Windows and WSL choose same optimal strategy}\}}{\#\{\text{tested workloads}\}}$$
- Pre workloads s $P < 64$ (krátke prompty): Windows aj WSL vyberajú `CPU_ONLY` $\implies$ Zhoda ($1.0$).
- Pre workloads s $P \ge 64$: Na Windowse je optimom `GPU_PREFILL_CPU_DECODE` (vďaka fyzickému GPU). Vo WSL2, kde GPU chýba, closed-loop planner autonómne prechádza na `CPU_ONLY`.

### 2. Adaptation Cost Delta:
$$\Delta J_{\text{WSL}} = J_{\text{adaptive}} - J_{\text{static}}$$
kde $J_{\text{static}}$ predstavuje statický systém, ktorý by sa vo WSL2 pokúsil vynútiť Windows policy (`GPU_PREFILL_CPU_DECODE`) a zlyhal alebo vyvolal unhandled exception, zatiaľ čo $J_{\text{adaptive}}$ monitoruje physical state prostredia a plynule zaručuje $100\%$ SLO úspešnosť.

### 3. Klasifikačné pásmo verdiktu:
- **`PLATFORM_SPECIFIC`**: Systém zlyhá pri zmene OS, hádže chyby alebo vyžaduje manuálny prepis kódu.
- **`CROSS_ENVIRONMENT_ADAPTIVE_VALIDATED`**: Systém správne zdetekuje zmenu prostredia, prekalibruje probes a planner autonómne vyberie najlepšiu uskutočniteľnú policy s $0$ chybami.
- **`GENERALIZABLE_RUNTIME_CANDIDATE`**: Potvrdené riadenie podľa formálnej rovnice:
  $$\boxed{\text{Environment} + \text{PhysicalState} + \text{Workload} + \text{ExecutionPhase} \longrightarrow \text{MeasuredPolicy}}$$
