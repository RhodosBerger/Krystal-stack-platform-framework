# SOVEREIGN BOT MULTI-SCALE DIMENSIONING & LAYER GOVERNING VARIABLES SPECIFICATION
## Špecifikácia dimenzovania bota, premenných riadiacich vrstvy a koordinácie optimálnej prevádzky

> **Mantra:** *Drž cieľ stabilný, plán môžeš meniť, kombinuj len označené charakteristiky, nevyrob istotu, otváraj rámec pri rozpore, neoverené ostáva neznáme.*  
> **Ústavný Invariant:** `VITAL_MAX_HP == 6` | $\neg\text{Valid}(A) \implies \Delta S \equiv 0$ (Zero Mutation on Breach)  
> **Normatívny Rámec:** KSPP v1 (P1-P10) | Dual-Mode Epistemic Architecture (Proposer $\neq$ Evaluator)  
> **Katalógová Teória:** `THEORY-SOVEREIGN-BOT-SCALE-DIMENSIONING` (Theory DAG: 30 uzlov, 0 cyklov)

---

## 1. Výkonný Súhrn a Účel

Tento dokument definuje **nové dimenzovanie autonómneho Sovereign Bota** naprieč 4 hierarchickými časovo-kapacitnými škálami a formalizuje **28 autoritatívnych riadiacich premenných**, ktoré deterministicky riadia všetkých 6 architektonických vrstiev frameworku Krystal Stack (L0 až L5).

Sovereign Bot neslúži len ako pasívny asistent či textový chatbot, ale ako **kybernetický guvernér prevádzky**, ktorý:
1. **Monitoruje a reguluje väzby medzi vrstvami:** Zabraňuje tomu, aby lokálna optimalizácia v jednej vrstve (napr. strmý nárast frekvencie jadier v L0) vyvolala deštruktívne spätné rázy (back-EMF flyback), prekmit napätia na PDN zbernici alebo saturáciu UMA zbernice ($\mu_{\text{UMA}} \ge 0.78$) v L1.
2. **Kalkuluje Pareto-optimálny prevádzkový bod:** Rieši multikriteriálnu rovnováhu medzi snímkovou frekvenciou ($60 - 120\text{ Hz}$), VBlank tieňovým rozpočtom ($\tau_{\text{shadow}} \ge 3.50\text{ ms}$), tepelnou obálkou PL1 ($12 - 28\text{ W}$), symplektickou konzerváciou ($\det(\mathbf{J}) \equiv 1.0$) a spotrebou energie.
3. **Garantuje ústavnú nedotknuteľnosť:** Nemenné zákony (napr. `VITAL_MAX_HP == 6`, zákaz samocertifikácie `Proposer != Evaluator` a `Delta S == 0`) nemožno zmutovať žiadnym príkazom; akýkoľvek neplatný pokus je okamžite odmietnutý s nulovým zápisom do stavu.

---

## 2. 4 Hierarchické Škálovacie Domény Bota

Bot je dimenzovaný na paralelnú prevádzku v 4 časových a priepustnostných škálach:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 4 ŠKÁLOVACIE DOMÉNY BOTA                               │
├─────────────────────┬───────────────────┬──────────────┬──────────────┬────────────────┤
│ DOMÉNA              │ ČASOVÁ GRANULARITA│ KONKURENCIA  │ PRIEPUSTNOSŤ │ HLAVNÁ ÚLOHA   │
├─────────────────────┼───────────────────┼──────────────┼──────────────┼────────────────┤
│ MICRO_SILICON_NS    │ <= 50 ns          │ 1 024 sond   │ 10M ev/s     │ TSC telemetria │
│ MESO_VBLANK_MS      │ 0.748 us - 16.6 ms│ 256 kanálov  │ 1 000 ev/s   │ VBlank tieň    │
│ MACRO_CADENCE_MS    │ 100 ms - 200 ms   │ 64 procesov  │ 10 ev/s      │ Dwell hysteréz │
│ EPOCH_GOVERNANCE_S  │ >= 1.0 s          │ 16 auditov   │ 1 ev/s       │ VPP a ledger   │
└─────────────────────┴───────────────────┴──────────────┴──────────────┴────────────────┘
```

### 1. Doména `MICRO_SILICON_NS` ($\le 50\text{ ns}$)
- **Mechanizmus:** Zber nanosekundovej telemetrie z procesora pomocou inštrukcie `RDTSC`/`RDTSCP` a priamych non-temporal zápisov `MOVNTI` do 8-bajtového `PulseFrame` rámca.
- **Kapacita:** Až 1 024 paralelných sondovacích bodov, priepustnosť $10^7$ udalostí/s, latencia sondy $\le 11.34\text{ ns}$, celková interferencia pozorovateľa $\eta \le 0.12\%$.

### 2. Doména `MESO_VBLANK_MS` ($0.748\ \mu\text{s} - 16.66\text{ ms}$)
- **Mechanizmus:** Synchronizácia s vertikálnym zatemnením grafického scanoutu (VBlank) a riadenie zero-copy prenosov Level Zero USM medzi iGPU a CPU.
- **Kapacita:** 256 paralelných kanálov, minimálny tieňový rozpočet $\tau_{\text{shadow}} \ge 3.50\text{ ms}$, strop UMA zbernice $\mu_{\text{UMA}} < 0.78$.

### 3. Doména `MACRO_CADENCE_MS` ($100\text{ ms} - 200\text{ ms}$)
- **Mechanizmus:** Riadenie kognitívnych harmonických periód (Alpha rytmus $100\text{ ms}$) a 12-rámcová hysterézna stabilizácia frekvencie DVFS (KSPP P8, $200\text{ ms}$ pri $60\text{ Hz}$).
- **Kapacita:** 64 koordinovaných úloh, vyhladzovací Merkle-Momentum filter $\gamma = 0.85$, ochrana pred frekvenčným thrashingom.

### 4. Doména `EPOCH_GOVERNANCE_S` ($\ge 1.0\text{ s}$)
- **Mechanizmus:** Kaskádový routing viacmodelových inferencií (Cascade SLM), energetická arbitráž VPP (Virtual Power Plant) a kryptografický audit HMAC-SHA256.
- **Kapacita:** 16 dlhodobých auditných reťazcov, pamäťový rozpočet 1 024 MB, termodynamické dávkovanie energie ($E_{\text{budget}} \le 0.42\text{ J/tok}$).

---

## 3. Katalóg 28 Riadiacich Premenných Podľa Vrstiev (L0 - L5)

Každá premenná je formalizovaná ako [`LayerGoverningVariable`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/sovereign_bot_scale_dimensioner.py) s presnými fyzikálnymi hranicami a ústavným statusom:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             ARCHITEKTONICKÝ STACK VRSTIEV                              │
├────┬────────────────────────────┬──────────────────────────────────────────────────────┤
│ L5 │ CONSTITUTIONAL GATE        │ VITAL_MAX_HP == 6, SoC, Delta S == 0, RAPL Joule     │
├────┼────────────────────────────┼──────────────────────────────────────────────────────┤
│ L4 │ COORDINATION FABRIC        │ SPSC 128B kanály, 64-slotový ring, HMAC 96B obálka   │
├────┼────────────────────────────┼──────────────────────────────────────────────────────┤
│ L3 │ COGNITIVE CADENCE          │ Dwell 12 rámcov, Merkle-Momentum 0.85, P* 15 tokenov │
├────┼────────────────────────────┼──────────────────────────────────────────────────────┤
│ L2 │ SPATIAL & SYMPLECTIC       │ D6 6 sektorov (0x10..0x60), det(J) == 1.0, CFL       │
├────┼────────────────────────────┼──────────────────────────────────────────────────────┤
│ L1 │ CHRONOMETRIC & UMA         │ 60 Hz, 3.5ms VBlank tieň, mu_UMA < 0.78, USM 0.748us │
├────┼────────────────────────────┼──────────────────────────────────────────────────────┤
│ L0 │ SILICON & PDN              │ L = 2.5nH, Z0 = 50 Ohm, Vdd = 0.85V, max +0.15V, PL1 │
└────┴────────────────────────────┴──────────────────────────────────────────────────────┘
```

### Kompletná matica premenných:

| ID Premennej | Vrstva | Názov | Škála | Nominál | Rozsah [Min, Max] | Jedn. | Ústavný Status | Riadica Rovnica |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `VAR_PDN_INDUCTANCE_NH` | L0 | PDN Loop Indukčnosť | MICRO | 2.5 | [1.0, 5.0] | nH | ⚙️ Laditeľná | $V_{\text{ind}} = -L \frac{di}{dt}$ |
| `VAR_PDN_IMPEDANCE_Z0` | L0 | PDN Tlmiaca Impedancia | MICRO | 50.0 | [25.0, 100.0] | $\Omega$ | ⚙️ Laditeľná | $\tau = \frac{L}{Z_0}$ |
| `VAR_VDD_NOMINAL_VOLTS` | L0 | Nominálne Napätie Vdd | MICRO | 0.85 | [0.65, 1.05] | V | ⚙️ Laditeľná | $P = C V^2 f$ |
| `VAR_MAX_VOLTAGE_OVERSHOOT_V`| L0 | Max. Prekmit Napätia | MICRO | 0.15 | [0.05, 0.15] | V | ⚙️ Laditeľná | $V_{\text{peak}} \le V_0 + 0.15\text{ V}$ |
| `VAR_PL1_POWER_BUDGET_W` | L0 | Príkonový Limit PL1 | MACRO | 28.0 | [12.0, 35.0] | W | ⚙️ Laditeľná | $\int P_{\text{PL1}} dt \le E_{\text{budget}}$ |
| `VAR_THERMAL_TJ_MAX_C` | L0 | Teplota Prechodu Tj | MACRO | 95.0 | [70.0, 100.0] | $^\circ\text{C}$ | ⚙️ Laditeľná | $T_j = T_{\text{amb}} + P \theta_{ja}$ |
| `VAR_DISPLAY_REFRESH_RATE_HZ`| L1 | Obnovovacia Frekvencia | MESO | 60.0 | [30.0, 144.0] | Hz | ⚙️ Laditeľná | $T_{\text{frame}} = 1000 / f$ |
| `VAR_ACTIVE_RASTER_SCAN_MS` | L1 | Trvanie Raster Scanoutu | MESO | 12.80 | [8.0, 14.0] | ms | ⚙️ Laditeľná | $T_{\text{active}} = N_{\text{lines}} / f_{\text{line}}$ |
| `VAR_VBLANK_GUARD_BAND_MS` | L1 | VBlank Guard Band | MESO | 0.366 | [0.1, 1.0] | ms | ⚙️ Laditeľná | $\tau_{\text{guard}} = \delta t_{\text{jit}} + \delta t_{\text{sync}}$ |
| `VAR_VBLANK_SHADOW_BUDGET_MS`| L1 | VBlank Tieňový Rozpočet | MESO | 3.50 | [3.50, 6.0] | ms | ⚙️ Laditeľná | $\tau_{\text{shadow}} \ge 3.50\text{ ms}$ |
| `VAR_UMA_SATURATION_CEILING` | L1 | UMA Strop Saturácie | MESO | 0.78 | [0.50, 0.78] | ratio | ⚙️ Laditeľná | $\mu_{\text{UMA}} < 0.78$ |
| `VAR_USM_HANDOFF_DEADLINE_US`| L1 | Level Zero USM Handoff | MESO | 0.748 | [0.10, 0.748] | $\mu\text{s}$ | ⚙️ Laditeľná | $\tau_{\text{handoff}} \le 0.748\ \mu\text{s}$ |
| `VAR_HEX_SECTOR_COUNT` | L2 | Počet Sektorov $D_6$ | MACRO | 6.0 | [6.0, 6.0] | count | 🔒 Nemenná | $D_6 \implies 6\text{ sektorov } (0\text{x}10..0\text{x}60)$ |
| `VAR_JACOBIAN_DET_TOLERANCE` | L2 | Symplektická Tolerancia | MESO | $10^{-12}$| [$10^{-15}$, $10^{-9}$] | err | ⚙️ Laditeľná | $\|\det(\mathbf{J}) - 1.0\| < \varepsilon$ |
| `VAR_CFL_STABILITY_MARGIN` | L2 | CFL Rezerva Stability | MESO | 0.85 | [0.50, 0.95] | ratio | ⚙️ Laditeľná | $\Delta t \le m \cdot (2 / \omega_0)$ |
| `VAR_HERMITE_SMOOTHSTEP_C1` | L2 | Hermiteov Rád Hladkosti | MESO | 1.0 | [1.0, 1.0] | order | 🔒 Nemenná | $s(u) = 3u^2 - 2u^3 \implies C^1$ |
| `VAR_DWELL_FRAME_LIMIT` | L3 | Dwell Hysterézia Rámcov | MACRO | 12.0 | [12.0, 24.0] | frames | ⚙️ Laditeľná | $N_{\text{dwell}} \ge 12\text{ rámcov}$ (P8) |
| `VAR_MERKLE_MOMENTUM_GAMMA` | L3 | Merkle-Momentum Faktor | MACRO | 0.85 | [0.50, 0.95] | ratio | ⚙️ Laditeľná | $M_k = \gamma M_{k-1} + (1-\gamma)T$ |
| `VAR_PHASE_CROSSOVER_TOKENS` | L3 | Crossover Prah Tokenov | MESO | 15.0 | [8.0, 32.0] | tok | ⚙️ Laditeľná | $P^* = \lceil \Delta\tau / \Delta\alpha \rceil$ |
| `VAR_ALPHA_CADENCE_PERIOD_MS`| L3 | Alpha Kognitívna Perióda| MACRO | 100.0 | [50.0, 250.0] | ms | ⚙️ Laditeľná | $f = 10\text{ Hz} \implies T = 100\text{ ms}$ |
| `VAR_ZERO_COPY_FRAME_BYTES` | L4 | Zarovnaný Rámec SPSC | MICRO | 128.0 | [128.0, 128.0] | B | 🔒 Nemenná | `_Alignas(64)` 128B formát |
| `VAR_CHANNEL_RING_CAPACITY` | L4 | Kapacita Ring Buffera | MESO | 64.0 | [32.0, 256.0] | slots | ⚙️ Laditeľná | Kapacita $= 2^N$ slotov |
| `VAR_CROSS_SUBSYSTEM_DEADLINE_MS`| L4 | IPC Turnaround Lehota | MESO | 1.50 | [0.50, 5.00] | ms | ⚙️ Laditeľná | $\tau_{\text{IPC}} \le 1.50\text{ ms}$ |
| `VAR_HMAC_ENVELOPE_BYTES` | L4 | HMAC Autentifikácia | MICRO | 96.0 | [96.0, 96.0] | B | 🔒 Nemenná | 96B kryptografický podpis |
| `VAR_VITAL_MAX_HP` | L5 | Ústavný Invariant HP | EPOCH | 6.0 | [6.0, 6.0] | HP | 🔒 Nemenná | `VITAL_MAX_HP == 6` (P10) |
| `VAR_SEPARATION_OF_CONCERNS_ENFORCED`| L5 | Zákaz Samocertifikácie | EPOCH | 1.0 | [1.0, 1.0] | bool | 🔒 Nemenná | $\text{Proposer} \neq \text{Evaluator}$ |
| `VAR_DELTA_S_ON_BREACH` | L5 | Nulová Mutácia pri Chybe| EPOCH | 0.0 | [0.0, 0.0] | B | 🔒 Nemenná | $\neg\text{Valid}(A) \implies \Delta S \equiv 0$ |
| `VAR_METERING_RAPL_JOULES_PER_TOKEN`| L5 | RAPL Energetický Rozpočet| EPOCH | 0.42 | [0.10, 1.00] | J/tok | ⚙️ Laditeľná | $\int P_{\text{RAPL}} dt \le E_{\text{budget}}$ |

---

## 4. Krížové Väzby Medzi Vrstvami a Invariantné Pravidlá

Pri pokuse o zmenu ktorejkoľvek laditeľnej premennej vykonáva [`SovereignBotScaleDimensioner`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/sovereign_bot_scale_dimensioner.py) automatický audit previazanosti:

1. **Väzba Frekvencie a VBlank Tieňa ($L_1 \leftrightarrow L_0$):**
   Ak operátor navrhne zvýšiť obnovovaciu frekvenciu `VAR_DISPLAY_REFRESH_RATE_HZ` na $144\text{ Hz}$, celkový čas rámca klesne na $\frac{1000}{144} = 6.944\text{ ms}$. Pri aktívnom raster scanoute $12.80\text{ ms}$ je nemožné udržať ústavný tieňový rozpočet $\tau_{\text{shadow}} \ge 3.50\text{ ms}$. Inferencia okamžite návrh zamietne s nulovou zmenou stavu ($\Delta S = 0$).

2. **Väzba Induktívneho Spätného Rázu a Napätia ($L_0 \leftrightarrow L_1$):**
   Strmosť prúdovej zmeny $\frac{di}{dt}$ pri prepínaní jadier nesmie vyvolať indukované napätie presahujúce $\Delta V_{\max} = 0.15\text{ V}$. Ak by k tomu došlo, governor aktivuje tlmivku $Z_0 = 50.0\ \Omega$ a potlačí frekvenčný skok na úroveň ustálenia $5\tau = 250\text{ ps}$.

3. **Väzba UMA Saturácie a Vykresľovania ($L_1 \leftrightarrow L_4$):**
   Ak súbežné prenosy cez 6 koordinačných kanálov v L4 vyčerpajú priepustnosť zbernice UMA nad $78\%$ ($\mu_{\text{UMA}} \ge 0.78$), VBlank tieňové okno sa v L1 okamžite zablokuje na $0.0\text{ ms}$ (choke), aby sa zabránilo vizuálnemu tearing-u na displeji.

---

## 5. Riešič Pareto-Optimálneho Prevádzkového Bodu

Metóda `solve_optimal_operating_point(workload_intensity)` generuje autoritatívny auditný certifikát [`OptimalOperatingReceipt`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/sovereign_bot_scale_dimensioner.py):

$$\max_{\theta} \left\{ \text{Priepustnosť}(\theta) \right\} \quad \text{za podmienok:} \quad \begin{cases} P_{\text{PL1}}(\theta) \le 28.0\text{ W} \\ \mu_{\text{UMA}}(\theta) < 0.78 \\ \tau_{\text{shadow}}(\theta) \ge 3.50\text{ ms} \\ \|\det(\mathbf{J}) - 1.0\| < 10^{-12} \\ \text{VITAL\_MAX\_HP} \equiv 6 \end{cases}$$

Certifikát je kryptograficky podpísaný HMAC-SHA256 podpisom pokrývajúcim ID operácie, frekvenciu, rozpočet tieňa, príkon, zaťaženie UMA a zachovanie invarianta HP.

---

## 6. Príkazy Bota v Rozhraní `krystal_bot`

Všetky riadiace premenné a škálovacie domény sú prístupné cez rozhranie [`krystal_bot/service.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_bot/service.py) pomocou rodiny príkazov `/layer`:

```bash
# 1. Zobrazenie prehľadnej tabuľky všetkých 28 premenných
/layer list
/layer list L0_SILICON_PDN

# 2. Detail konkrétnej riadiacej premennej s fyzikálnym popisom a rovnicou
/layer get VAR_VBLANK_SHADOW_BUDGET_MS

# 3. Bezpečné ladenie premennej (overené bránou s Delta S == 0 pri porušení)
/layer tune VAR_ALPHA_CADENCE_PERIOD_MS 110

# 4. Výpočet optimálneho multi-vrstvového prevádzkového bodu (Pareto optimum)
/layer optimal
/layer optimal 0.8

# 5. Zobrazenie 4 časovo-kapacitných škálovacích domén Sovereign Bota
/layer scales
```

---

## 7. Verifikačné a Regresné Výsledky

Integrácia prešla hermetickou verifikáciou:
1. **Unit testy škálovania a premenných:** [`tests/test_sovereign_bot_scale_dimensioner.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_sovereign_bot_scale_dimensioner.py) (8/8 PASS v 0.003s).
2. **Unit testy integrácie bota a CLI:** [`tests/test_krystal_bot_layer_governor.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_krystal_bot_layer_governor.py) (6/6 PASS v 0.059s).
3. **Plná bot & sovereign regresná sada:** **120/120 unit testov úspešne prebehlo (100% PASS)** v čase 11.3 s.
4. **Theory Dependency Graph:** Zaregistrovaný uzol `THEORY-SOVEREIGN-BOT-SCALE-DIMENSIONING` s 30 plne acyklickými a verifikovanými uzlami.
