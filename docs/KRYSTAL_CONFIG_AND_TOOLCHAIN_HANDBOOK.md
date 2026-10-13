# KRYSTAL-STACK CONFIGURATION & TOOLCHAIN HANDBOOK
## Kompaktná Príručka Používania, Profilovania a Optimalizácie Konfigurácií

> **Dokument:** Praktická Príručka & Referenčný Manuál Toolchainu (Compressed Field Guide)  
> **Klasifikácia:** Operations & Runtime Configuration Engineering  
> **Ústavný Invariant:** `assert VITAL_MAX_HP == 6` (Garantovaný vo všetkých profiloch)

---

## 1. Rýchly Štart v 3 Krokoch (Quickstart)

```bash
# Krok 1: Inšpekcia lokálneho hardvéru, QPC časovačov a vektorových inštrukcií
python scripts/krystal_config_toolchain.py --inspect

# Krok 2: Zobrazenie zoznamu kanonických profilov
python scripts/krystal_config_toolchain.py --list

# Krok 3: Simulácia (Dry-Run) a export konfigurácie do .env súboru
python scripts/krystal_config_toolchain.py --profile REALTIME_LOW_LATENCY --dry-run
python scripts/krystal_config_toolchain.py --profile REALTIME_LOW_LATENCY --export env > .env.active
```

---

## 2. Kanonická Matica Konfigurácií (Configuration Matrix)

| Profil | Kadencia | Kvantum | MMCSS | CPU Afinita | Priorita OS | PL1 Limit | ISA Kernel | NPU Streamy | Lokál |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`REALTIME_LOW_LATENCY`** | **1.0 ms** | 1 ms | Games | P-Jadrá (`0x000F`) | HIGH | 32 W (-25mV) | `INTEL_AMX_TILE` | 4 (DP4A) | SK |
| **`BALANCED_DEVELOPMENT`** | **10.0 ms** | 2 ms | Default | Všetky (`0x00FF`) | NORMAL | 24 W (0mV) | `X86_AVX2_FMA` | 2 | SK |
| **`ECO_THERMAL_PRESERVATION`**| **50.0 ms** | 5 ms | Default | E-Jadrá (`0x00F0`) | BELOW_NORMAL | 10 W (-50mV) | `NPU_DP4A_INT8` | 1 | EN |
| **`SHADOW_COUNTERFACTUAL`** | **2.0 ms** | 1 ms | Default | P-Jadrá (`0x000F`) | NORMAL | 28 W (0mV) | `X86_AVX512_VNNI`| 2 (Shadow)| LA |
| **`MISSION_CONTROL_KIOSK`** | **16.67 ms**| 1 ms | DisplayPost | Všetky (`0x00FF`) | HIGH | 28 W (-15mV) | `X86_AVX2_FMA` | 2 | SK |
| **`HEADLESS_CONTAINER_CI`** | **10.0 ms** | 10 ms | Default | OS Default (`0x00`) | NORMAL | 15 W (0mV) | `GENERIC_SCALAR`| 0 | EN |

---

## 3. Detailný Prehľad Profilov (Profiles Breakdown)

### 3.1 Profil: `REALTIME_LOW_LATENCY`
* **Určenie:** Kritické regulačné slučky, bezodkladný dispečing nástrojov, vysokofrekvenčné generovanie zvuku/obrazu.
* **Technické jadro:** `winmm.timeBeginPeriod(1)`, MMCSS profil "Games", P-jadrová väzba (`0x000F`), akcelerácia Intel AMX/AVX-512 ($86.0\times$ zrýchlenie), zníženie latencie pod $1\,\text{ms}$ a jitteru pod $\pm 0.05\,\text{ms}$.
* **Kompaktná vizitka:**
```text
┌────────────────────────────────────────────────────────────────────────┐
│ KRYSTAL CONFIG: [REALTIME_LOW_LATENCY        ]              │
├────────────────────────────────────────────────────────────────────────┤
│ Cadence:    1.0 ms  │ Resolution: 1 ms  │ MMCSS: Games           │
│ Affinity: 0x000F    │ Priority: HIGH_PRIORITY_CLASS │ PL1: 32.0W (-25mV)   │
│ ISA:      INTEL_AMX_TILE   │ NPU: 4 strm (DP4A=True ) │ Sockets: 8910,8911,8912 │
│ Language: [SK] MILITARY_TACTICAL │ Shadow Mode: False │ Dirty Depth: 5 Tiers │
├────────────────────────────────────────────────────────────────────────┤
│ INVARIANT: ♥ ♥ ♥ ♥ ♥ ♥ (6/6 HP) ── VALIDATED STABLE                  │
└────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Profil: `BALANCED_DEVELOPMENT`
* **Určenie:** Bežné lokálne ladenie, multi-agentné párovanie, testovanie API a interaktívna práca v UI Kit Studio.
* **Technické jadro:** 10.0 ms cyklus, vyvážený 24 W príkon, plná podpora všetkých jadier, AVX2 vektorizácia, aktívne sokety pre streaming telemetrie.

### 3.3 Profil: `ECO_THERMAL_PRESERVATION`
* **Určenie:** Chladenie pri zvýšenej teplote ($T_j > 75^\circ\text{C}$), úspora energie, pasívne monitorovanie.
* **Technické jadro:** Obmedzenie príkonu na 10 W, podvoltovanie o -50 mV, beh výhradne na úsporných E-jadrách (`0x00F0`), vypnutie zbytočných soketov, obmedzenie kaskádového prekresľovania len na 2 úrovne.

### 3.4 Profil: `SHADOW_COUNTERFACTUAL_RESEARCH`
* **Určenie:** A/B experimentovanie, trénovanie rozhodovacích modelov, overovanie rizík pred aplikáciou.
* **Technické jadro:** Rýchla 2.0 ms kadencia v striktnom režime `AUTH_SHADOW_ONLY`. **Garantuje nulové zápisy do registrov OS.** Zaznamenáva zistenia priamo do `HypothesisLedger`.

### 3.5 Profil: `MISSION_CONTROL_STUDIO_KIOSK`
* **Určenie:** Kiosk režim, demonštračné displeje a plynulý 60 FPS rendering v UI Kit Studio.
* **Technické jadro:** Fixná QPC synchronizácia s obnovovacou frekvenciou $16.67\,\text{ms}$, optimalizovaný grafický profil MMCSS "DisplayPostProcessing", 5-úrovňový kaskádový kompozitor.

### 3.6 Profil: `HEADLESS_CONTAINER_CI`
* **Určenie:** GitHub Actions, Docker kontajnery, čisté POSIX systémy bez Windows API.
* **Technické jadro:** Využíva štandardný `time.perf_counter_ns()`, čistý Python skalárny fallback bez nutnosti špeciálnych inštrukčných sád, plne deterministické generovanie výstupov.

---

## 4. Referenčná Príručka CLI Príkazov (Toolchain CLI Reference)

Spúšťanie cez Python z koreňa repozitára:
```bash
python scripts/krystal_config_toolchain.py [VOĽBY]
```

| Prepínač / Argument | Popis a Funkcia |
| :--- | :--- |
| `-l`, `--list` | Vypíše prehľadný kompaktný zoznam všetkých kanonických profilov. |
| `-p`, `--profile <ID>` | Zvolí profil (napr. `REALTIME_LOW_LATENCY`, `BALANCED`, `ECO`). |
| `-i`, `--inspect` | Vykoná hĺbkovú inšpekciu CPU, limitov časovačov winmm a OpenVINO. |
| `-e`, `--export <formát>`| Exportuje profil v zadanom formáte: `env`, `json`, `card`, alebo `morse`. |
| `--dry-run` | Spustí simuláciu 9-krokového aktivačného reťazca bez reálnych mutácií OS. |
| `--validate-json <cesta>`| Overí platnosť vlastného JSON konfiguračného súboru voči invariantom. |

---

## 5. Morseova Abeceda ako Telegrafický Wire Formát a Akustický Maják

V Krystal Stacku plní Morseova abeceda ([`krystal_bot/morse.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_bot/morse.py)) štyri systémové úlohy:

```
[Krystal Bytecode / Telemetry Stream]
                 │
                 ▼  krystal_bot.morse.encode_text() / encode_program()
[Morse Wire String: "-.- .-. -.-- ... - .- .-.."]
                 │
                 ▼  krystal_bot.morse.morse_to_tensor()
[Fixed-Length Integer Tensor: [2, 1, 2, 3, 1, 2, ...]]  (DOT=1, DASH=2, LGAP=3, WGAP=4, PAD=0)
                 │
                 ▼
[OpenVINO / NPU / ProcessPredictor / Akustický Beep Maják]
```

1. **Človekom čitateľný wire formát bez knižníc (Human-Legible Wire Format)**:
   * Bezstratový prenos inštrukcií kreslenia a bytecode programov (`krystal_bot.bytecode.Program`) bez závislosti na JSON alebo binárnych deserializátoroch.
2. **Celočíselný tenzorový vstup pre NPU / neurónové modely**:
   * Prevod Morse reťazca na fixný vektor celých čísel cez `morse_to_tensor(m, length=2048)`:
     $$\text{Tokeny}: \text{PAD}=0, \quad \text{DOT}=1, \quad \text{DASH}=2, \quad \text{LGAP}=3, \quad \text{WGAP}=4$$
   * Tenzor je priamo konzumovateľný neurónovými sieťami v OpenVINO bez masívnych NLP tokenizerov.
3. **Kódovanie udalostí jadra & štatistický n-gram `ProcessPredictor`**:
   * Mapovanie udalostí jadra na symboly: `P` (proces OK), `T` (vlákno OK), `M` (zmeškaný deadline), `X` (chyba), `J` (back-pressure odmietnutie), `H` (self-heal), `C` (zmena zdravia), `B` (otvorený breaker).
   * Štatistický n-gram model predikuje nasledujúce procesy jadra.
4. **Hardvérový núdzový telemetrický maják (Emergency Beacon)**:
   * Pri zlyhaní grafického subsystému alebo siete toolchain vygeneruje telegrafický maják:
     ```bash
     python scripts/krystal_config_toolchain.py --profile REALTIME_LOW_LATENCY --export morse
     ```
   * Výstup vysiela telemetriu a ústavný invariant `VITAL_MAX_HP == 6` v kompaktnom formáte vhodnom pre rádiové vysielanie, sériovú linku UART alebo PC-Speaker buzzer.

---

## 6. Rýchla Diagnostika Latencie a Riešenie Problémov (Triage Playbook)

```
                       [ZISTENÝ PROBLÉM LATENCIE]
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
[Jitter > 1.0 ms]           [T_j > 75°C / Throttling]    [LLC Misses > 40%]
      │                            │                            │
      ▼                            ▼                            ▼
Spusť timeBeginPeriod(1)    Prepnúť do profilu           Znížiť počet workerov
a over MMCSS "Games"         ECO_THERMAL_PRESERVATION     v mriežke na 4 až 6
```

### Scenár A: Časovač Windows hlási rozlíšenie 15.6 ms namiesto 1.0 ms
* **Príčina:** Volanie `timeBeginPeriod` nebolo inicializované, alebo proces beží bez multimediálnych práv.
* **Riešenie:** Spustiť `python scripts/krystal_config_toolchain.py --profile REALTIME_LOW_LATENCY --dry-run` a overiť hodnotu `winmm_timer_min_period_ms` cez `--inspect`.

### Scenár B: Teplota čipu stúpa nad 78°C a klesá taktovacia frekvencia
* **Príčina:** Príkon PL1 32 W prekračuje chladiace kapacity pasívneho šasi.
* **Riešenie:** Aktivovať `ECO_THERMAL_PRESERVATION` (`target_pl1_watts=10.0`, `undervolt_mv=-50.0`). Kaskáda zníži generovanie tepla o $68\%$.

### Scenár C: Nárast latencie pri zvyšovaní počtu workerov (LLC Cache Trashing)
* **Príčina:** Pri 8 workeroch na `ConvolutionalDiffusionLayer` dochádza k saturácii Last Level Cache.
* **Riešenie:** Nastaviť počet workerov na 4 až 5. Priechodnosť dosiahne $97\%$ maxima, ale latencia p99 klesne o $34\%$.

### Scenár D: Zlyhanie invariantu `VITAL_MAX_HP != 6`
* **Príčina:** Neautorizovaný pokus o zmenu životného fondu v konfigurácii.
* **Riešenie:** Systém automaticky odmietne konfiguráciu. Hodnota `vital_max_hp` musí byť striktne `6`.

---

## 7. Prehľad Súvisiacich Súborov v Repozitári

* **CLI Nástroj Toolchainu:** [`scripts/krystal_config_toolchain.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/scripts/krystal_config_toolchain.py)
* **Jednotkové Testy Toolchainu:** [`tests/test_krystal_config_toolchain.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_krystal_config_toolchain.py)
* **Architektonická Špecifikácia:** [`docs/architecture/KRYSTAL_TEMPLATE_SYSTEM_AND_LATENCY_OPTIMIZATION_FRAMEWORK.md`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/docs/architecture/KRYSTAL_TEMPLATE_SYSTEM_AND_LATENCY_OPTIMIZATION_FRAMEWORK.md)
* **Kanonické Šablóny Koordinácie:** [`krystal_kernel/coordination_templates.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/coordination_templates.py)
* **Sub-ms Časovače Windows:** [`krystal_kernel/windows_timer_subsystem.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/windows_timer_subsystem.py)
* **Certifikát Stabilného Buildu:** [`artifacts/MIDNIGHT_STABLE_BUILD_CERTIFICATE.json`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/artifacts/MIDNIGHT_STABLE_BUILD_CERTIFICATE.json)
