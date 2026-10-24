# KRYSTAL STACK: UČEBNICOVÉ UKÁŽKY INFEROVATEĽNÝCH PATTERNOV
## Formálna špecifikácia, fyzikálne odvodenia, 6-zložková evidencia a referenčná implementácia

> **Mantra:** *Drž cieľ stabilný, plán môžeš meniť, kombinuj len označené charakteristiky, nevyrob istotu, otváraj rámec pri rozpore, neoverené ostáva neznáme.*  
> **Ústavný Invariant:** `VITAL_MAX_HP == 6` | $\neg\text{Valid}(A) \implies \Delta S \equiv 0$ (Zero-Mutation on Rejection)  
> **Normatívny Rámec:** Krystal Sovereign Principle Pack (KSPP v1: P1-P10) | Dual-Mode Epistemic Architecture (EXPLORE $\neq$ VERIFY)

---

## 1. Úvod: Čo je inferovateľný pattern v Krystal Stack?

V konvenčnom softvérovom inžinierstve sa termínom *design pattern* často označuje pasívna štruktúrna šablóna (napr. Singleton, Factory, Observer). V prostredí **Krystal Stack Frameworku** má však pattern zásadne odlišný, exaktne formalizovaný epistemický a fyzikálny význam:

```
                               ┌──────────────────────────────────────────────┐
                               │           APRIÓRNY INVARIANT (I)             │
                               │  VITAL_MAX_HP == 6, det(J) == 1.0, mu < 0.78 │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
┌──────────────────────────────┐              ┌───────────────┐              ┌──────────────────────────────┐
│     VSTUPNÉ POZOROVANIE      │ ───────────► │   KAUZÁLNY    │ ───────────► │     INFEROVANÝ VÝSLEDOK      │
│  (Telemetria, Počet Tokenov, │              │  INFERENČNÝ   │              │  (Crossover Target, Shadow   │
│   di/dt, Fázový bod q, p)    │              │   OPERÁTOR    │              │   Window, Clamped V, Comm.)  │
└──────────────────────────────┘              └───────────────┘              └──────────────────────────────┘
                                                      ▲
                                                      │
                               ┌──────────────────────┴───────────────────────┐
                               │       6-ZLOŽKOVÝ EVIDENČNÝ DÔKAZ             │
                               │  ⟨Charakteristika, Zdroj, Scope, Status,     │
                               │               Validita, Závislosti⟩          │
                               └──────────────────────────────────────────────┘
```

**Inferovateľný pattern** je deterministický vzťah, v ktorom:
1. Z čiastočných fyzikálnych pozorovaní, hardvérovej telemetrie alebo zmluvných invariantov systém **exaktne inferuje** (vyvodzuje) optimálnu akciu, rozvrhnutie prostriedkov, topologické zaradenie alebo stavovú mutáciu.
2. Inferencia nie je štatistickým odhadom, ale podlieha **11-článkovému kanonickému reťazcu overenia** a **6-zložkovej evidencii**:
   $$\text{Tvrdenie} \longrightarrow \text{Špecifikácia} \longrightarrow \text{Autorita} \longrightarrow \text{Invariant} \longrightarrow \text{Verifikácia} \longrightarrow \text{Validita} \longrightarrow \text{Meranie} \longrightarrow \text{Metrika} \longrightarrow \text{Falzifikátor} \longrightarrow \text{Verdikt} \longrightarrow \text{Rezíduum}$$
3. Každý inferovateľný pattern má definovaný **falzifikátor** (podmienku okamžitého vyvrátenia) a zaručuje, že pri neplatnosti vstupu alebo porušení invariantu nastáva $\Delta S \equiv 0$ (žiadna skrytá zmena autoritatívneho stavu).

---

## 2. Prehľad 8 Kanonických Učebnicových Patternov

| ID | Názov Patternu | Základný Matematický Zákon | Fyzikálne Ukotvenie v Kremíku | Falzifikátor |
|:---|:---|:---|:---|:---|
| **P1** | **Asymetrický fázový crossover** | $P^* = \left\lceil \frac{\tau_{\text{GPU}} - \tau_{\text{CPU}}}{\alpha_{\text{CPU}} - \alpha_{\text{GPU}}} \right\rceil$ | Intel Level Zero USM ($\tau \le 0.748\ \mu\text{s}$) | GPU decode prekoná CPU pri $B=1, P=1$ |
| **P2** | **Chronometrický VBlank tieň** | $\Phi_{\text{Scan}} \cdot \Phi_{\text{Comp}} = 0 \land \mu_{\text{UMA}} < 0.78$ | Intel Display Engine Raster Blanking ($\ge 3.5\text{ ms}$) | Tearing displeja alebo saturácia UMA $> 78\%$ |
| **P3** | **Symplektická leapfrog konzervácia** | $\det(\mathbf{J}) \equiv 1.000000000000$ | Terrain Manifold / Zero Numerical Gouging | $|\det(\mathbf{J}) - 1.0| > 10^{-12}$ |
| **P4** | **12-rámcová dwell hysterézia (P8)** | $M_k = \gamma M_{k-1} + (1-\gamma)T_k$, $N \ge 12$ | Tiger Lake PL1/PL2 DVFS Anti-Thrashing | Upscale commit pri počítadle $< 12$ rámcov |
| **P5** | **Hex-HSL diskrétna bijekcia** | $s(t) = 3t^2 - 2t^3$, $C^1$ spojitosť | Hexagonal D6 SoC Floorplan ($0\text{x}10$..$0\text{x}60$) | Nespojitosť $\frac{ds}{dt}$ na hranici sektora |
| **P6** | **Induktívny back-EMF clamping** | $V_{\text{ind}} = -L \frac{di}{dt}$, $V(t) \le V_0 + 0.15\text{ V}$ | Faraday-Lenz PDN Network ($L = 2.5\text{ nH}, Z_0 = 50\ \Omega$) | Napäťový prekmit $> 1.00\text{ V}$ ($0.85 + 0.15$) |
| **P7** | **Sub-50ns telemetrická mriežka** | $\tau = \frac{\Delta \text{TSC}}{f_{\text{TSC}}}$, $\eta = \frac{\tau_{\text{probe}}}{T_{\text{sample}}} \le 0.12\%$ | 8B `PulseFrame` s non-temporal store `MOVNTI` | Latencia sondy $> 50\text{ ns}$ alebo $\eta > 0.15\%$ |
| **P8** | **Dual-Mode stavová brána (ArtifactGate)** | $\text{Release}(x) \iff \text{Valid}(x) \land (HP \equiv 6)$ | Ústavná hranica jadra (Proposer $\neq$ Evaluator) | Commit stavu pri samocertifikácii |

---

## 3. Detailné Rozbory Jednotlivých Patternov

---

### Pattern 1: Asymetrická Fázová Crossover Inferencia (Prefill vs. Decode)

#### 1. Fyzikálno-architektonický mechanizmus
Veľké jazykové modely (LLM) majú pri behu na heterogénnom SoC dve diametrálne odlišné fázy:
1. **Prefill (Spracovanie promptu):** Masívne paralelný GEMM výpočet ($O(N)$ škálovanie voči dĺžke sekvencie). Táto fáza je **Compute-Bound** a saturuje SIMD jednotky grafického akcelerátora (Intel Iris Xe 80/96 EU).
2. **Decode (Autoregresívne generovanie tokenov):** Sekvenčné čítanie všetkých váh modelu po jednom tokene ($O(1)$ FLOP na váhu). Táto fáza je **Memory-Bandwidth-Bound** a trpí réžiou spúšťania GPU jadier. Hostiteľské CPU jadrá s priamym AVX2/AVX-512 dispatchom vykazujú nižšiu latenciu na jednotlivé kroky.

Crossover operátor inferuje presný prah počtu tokenov $P^*$, pri ktorom dochádza k zlomu:
$$T_{\text{GPU}}(P) = \alpha_{\text{GPU}} \cdot P + \tau_{\text{launch\_GPU}}$$
$$T_{\text{CPU}}(P) = \alpha_{\text{CPU}} \cdot P + \tau_{\text{dispatch\_CPU}}$$
$$P^* = \left\lceil \frac{\tau_{\text{launch\_GPU}} - \tau_{\text{dispatch\_CPU}}}{\alpha_{\text{CPU}} - \alpha_{\text{GPU}}} \right\rceil$$

```
Čas latencie (ms)
▲
│                       / T_CPU(P) (strmý rast kvôli pamäťovej priepustnosti)
│                      /
│                     /
│   T_GPU(P) ───────/─── (plytký rast vďaka masívnemu GPU GEMM)
│          │       /
│          │      /
│   tau_GPU│...../  ◄─── Prah crossoveru P* (napr. 15 tokenov)
│          │    /
│          │   /
│   tau_CPU│──/
│          └────────────────────────────────────────► Dĺžka sekvencie (P)
│             P < P* (CPU výhodnejšie) │ P >= P* (GPU výhodnejšie)
```

#### 2. Kód referenčnej inferencie
```python
from krystal_kernel.textbook_inferable_patterns import infer_asymmetric_phase_crossover

# Inferencia pre prefill fázu (512 tokenov)
result_prefill = infer_asymmetric_phase_crossover(prompt_tokens=512)
assert result_prefill.optimal_execution_target == "GPU_LEVEL_ZERO"
assert result_prefill.crossover_token_threshold == 15

# Inferencia pre single-token decode (1 token)
result_decode = infer_asymmetric_phase_crossover(prompt_tokens=1)
assert result_decode.optimal_execution_target == "CPU_NPU_HOST"
```

#### 3. 6-zložková evidencia
- **Charakteristika:** Prefill GPU paralelný GEMM crossover do CPU vektorového pamäťovo-limitovaného Decode.
- **Zdroj:** Analytický Roofline Model & Intel Level Zero USM telemetria (Report H04).
- **Scope:** Heterogénne SoC (Tiger Lake i5/i7, LPDDR4x-4266 dual-channel).
- **Evidenčný status:** `VERIFIED_EMPIRICAL`.
- **Validita:** Batch size $B = 1$, kontext $\le 4096$ tokenov, Level Zero USM zero-copy prenos.
- **Závislosti:** `Level Zero USM`, `LPDDR4x Dual Channel`, `VITAL_MAX_HP == 6`.

---

### Pattern 2: Chronometrický VBlank Tieňový Multiplexing

#### 1. Fyzikálno-architektonický mechanizmus
Zdieľaná zbernica UMA (Unified Memory Architecture) trpí extrémnou pamäťovou interferenciou, ak grafický scanout procesora displeja (eDP/HDMI) a ťažký neurónový výpočet čítajú RAM súčasne.
Zákon KSPP Principle P3 definuje chronometrické tieňové multiplexovanie:
$$\Phi_{\text{Scanout}}(t) \cdot \Phi_{\text{Compute}}(t) = 0 \quad \land \quad \mu_{\text{UMA}}(t) < 0.78$$

Pri obnovovacej frekvencii $60\text{ Hz}$ trvá jeden rámec $16.666\text{ ms}$. Ak aktívne vykresľovanie rasteru vyčerpá $12.80\text{ ms}$ a bezpečnostný guard band je $0.366\text{ ms}$, systém **inferuje bezkolízne tieňové okno**:
$$\tau_{\text{shadow}} = 16.666\text{ ms} - 12.80\text{ ms} - 0.366\text{ ms} = 3.500\text{ ms}$$

```
 Rámec (16.666 ms pri 60 Hz)
├──────────────────────────────────────┬───────────────┬───────────────┤
│    Aktívny Display Raster Scanout    │  Guard Band   │ VBlank Tieň   │
│           (12.800 ms)                │  (0.366 ms)   │  (3.500 ms)   │
│  [ZÁKAZ ŤAŽKÝCH VÝPOČTOV NA UMA]     │  [SYNC LOCK]  │ [KOMPÚT OK!]  │
└──────────────────────────────────────┴───────────────┴───────────────┘
                                                       ▲
                                                       │
                           Inferencia autorizácie výpočtu pri mu_UMA < 0.78
```

#### 2. Kód referenčnej inferencie
```python
from krystal_kernel.textbook_inferable_patterns import infer_chronometric_vblank_shadow

# Nominálny stav: 60 Hz, nízka záťaž zbernice UMA (42%)
res = infer_chronometric_vblank_shadow(refresh_rate_hz=60.0, uma_utilization=0.42)
assert res.shadow_window_authorized is True
assert res.allocated_shadow_ms >= 3.50

# Preťaženie UMA (82% > 78%) -> Inferencia okamžite blokuje tieňové okno
res_choke = infer_chronometric_vblank_shadow(refresh_rate_hz=60.0, uma_utilization=0.82)
assert res_choke.shadow_window_authorized is False
assert res_choke.allocated_shadow_ms == 0.0
```

---

### Pattern 3: Symplektická Leapfrog Konzervácia Fázového Priestoru

#### 1. Fyzikálno-architektonický mechanizmus
Bežné integračné schémy (Forward Euler, Runge-Kutta 4) v simulácii terénu a fyziky trpia numerickým unikaním energie (umelé tlmenie alebo nekontrolovaná divergencia/gouging). 
Symplektický integrátor typu Verlet / Leapfrog zachováva 2-formu $\omega = dq \wedge dp$ a determinant Jacobiánu je analyticky rovný 1:
$$\mathbf{J} = \begin{pmatrix} \frac{\partial q_{t+1}}{\partial q_t} & \frac{\partial q_{t+1}}{\partial p_t} \\ \frac{\partial p_{t+1}}{\partial q_t} & \frac{\partial p_{t+1}}{\partial p_t} \end{pmatrix}$$
$$\det(\mathbf{J}) \equiv 1.000000000000$$

Podmienka numerickej stability podľa Courant-Friedrichs-Lewy (CFL):
$$\Delta t < \frac{2}{\omega_0}$$

```
Fázový priestor (p vs q)
    p ▲
      │        ┌──────────┐
      │       /            \
      │      │   Plocha S1  │     Konzervácia objemu:
      │       \            /      Plocha S1 == Plocha S2
      │        └──────────┘       det(J) == 1.000000000000
      │              \            (Žiadna numerická disipácia!)
      │               ▼
      │             ┌──────────┐
      │            /  Plocha S2 \
      │           └──────────────┘
      └────────────────────────────────► q
```

#### 2. Kód referenčnej inferencie
```python
from krystal_kernel.textbook_inferable_patterns import infer_symplectic_leapfrog_step

res = infer_symplectic_leapfrog_step(q0=1.0, p0=0.0, dt=0.02, omega_0=1.5)
assert res.is_symplectic is True
assert abs(res.jacobian_determinant - 1.0) < 1e-12
```

---

### Pattern 4: 12-Rámcová Dwell Hysterézia a Merkle-Momentum (P8)

#### 1. Fyzikálno-architektonický mechanizmus
Pri dynamickom škálovaní frekvencie procesora (DVFS) a prideľovaní vrstiev spôsobuje okamžitá reakcia na šum tzv. *frekvenčný thrashing* (neustále prepínanie hore a dole), čo poškodzuje tepelnú stabilitu kremíka.
Zásada KSPP Principle P8 ukladá:
1. **Asymetria:** Rýchly zostup (Fast downscale) v jedinom kroku pri núdzovom prekročení rozpočtu.
2. **Hysterézia:** Vzostup (Upscale) vyžaduje zotrvanie v požadovanom stave minimálne $N_{\text{dwell}} \ge 12$ po sebe idúcich rámcov ($200\text{ ms}$ pri $60\text{ Hz}$).
3. **Rekurentný filter:**
   $$M_k = \gamma M_{k-1} + (1 - \gamma) T_k, \quad \gamma = 0.85$$

```
Požiadavka setpointu
▲
│    _   _   _   _                    Sustained Target (15 rámcov)
│   / \ / \ / \ / \                 ┌───────────────────────────
│  /   V   V   V   \                │
│ ─Noisy Thrashing──                │
│ [POTLAČENÉ COMMITOM: 0]           │ [COMMIT po 12 rámcoch!]
└───────────────────────────────────┴──────────────────────────► Čas (rámce)
                                    Dwell Counter: 1..12
```

#### 2. Kód referenčnej inferencie
```python
from krystal_kernel.textbook_inferable_patterns import infer_12frame_dwell_hysteresis

# 1. Šum (striedavé hodnoty) -> Žiadna zmena stavu
res_noise = infer_12frame_dwell_hysteresis(initial_state=1.0, target_stream=[1.0, 1.5] * 4)
assert res_noise.total_transitions_committed == 0

# 2. Udržiavaný vzostup (15 po sebe idúcich rámcov) -> Commit po 12 rámcoch
res_stable = infer_12frame_dwell_hysteresis(initial_state=1.0, target_stream=[2.0] * 15)
assert res_stable.total_transitions_committed == 1
assert res_stable.final_committed_state > 1.5
```

---

### Pattern 5: 3D Kartézska Hex-HSL Diskrétna Bijekcia a Hermiteov Smoothstep

#### 1. Fyzikálno-architektonický mechanizmus
Kontinuálny 3D priestor $(x, y, z) \in [-1, 1]^3$ je bižektívne transformovaný do diskrétnej 6-sektorovej dihedrálnej mriežky $D_6$ ($0\text{x}10$ až $0\text{x}60$). Pre zabezpečenie $C^1$ hladkosti na rozhraní sektorov sa aplikuje Hermiteov smoothstep polynóm:
$$s(u) = 3u^2 - 2u^3, \quad u \in [0, 1]$$
$$s'(0) = 0, \quad s'(1) = 0$$

Vzniká tak kompaktný 32-bitový farebný token `0xHHSSLLAA`:
$$\text{Token} = (\lfloor H \cdot 255 \rfloor \ll 24) \mid (\lfloor S \cdot 255 \rfloor \ll 16) \mid (\lfloor L \cdot 255 \rfloor \ll 8) \mid 0\text{xFF}$$

```
                0x30 (120°-180°)          0x20 (60°-120°)
                     \                  /
                      \   y ▲          /
                       \    │         /
                        \   │        /
                         \  │       /
                          \ │      /
   0x40 (180°-240°) ───────┼┼─────── 0x10 (0°-60°)
                          / │      \   ───► x
                         /  │       \
                        /   │        \
                       /    │         \
                      /     │          \
                     /                  \
                0x50 (240°-300°)          0x60 (300°-360°)
```

#### 2. Kód referenčnej inferencie
```python
from krystal_kernel.textbook_inferable_patterns import infer_hex_hsl_discrete_bijection

# Uhol 90 stupňov -> Sektor 0x20
res = infer_hex_hsl_discrete_bijection(x=0.0, y=1.0, z=0.5)
assert res.discrete_sector_id == 0x20
assert res.packed_color_hex.startswith("0x")
assert 0.0 <= res.hermite_smoothstep <= 1.0
```

---

### Pattern 6: UMA Induktívna Napäťová Regulácia a Back-EMF Clamping

#### 1. Fyzikálno-architektonický mechanizmus
Pri náhlom prechode jadier z úsporného C-stavu do plného zaťaženia vzniká vysoká strmosť prúdovej zmeny $\frac{di}{dt}$ (napr. $\Delta I = 15\text{ A}$ za $\Delta t = 200\text{ ps} \implies \frac{di}{dt} = 75\text{ A/ns}$).
Parazitná indukčnosť napájacej siete (Power Delivery Network - PDN) s $L = 2.5\text{ nH}$ vyvoláva podľa Faraday-Lenzovho zákona obrovskú indukovanú špičku:
$$V_{\text{ind}} = -L \frac{di}{dt} = -2.5 \cdot 10^{-9} \cdot 7.5 \cdot 10^{10} = -187.5\text{ V}$$

Bez aktívneho tlmenia dôjde k deštrukcii dielektrika tranzistorov. Riadiaci obvod s charakteristickou impedanciou $Z_0 = 50.0\ \Omega$ inferuje potrebné zopnutie tlmiviek a obmedzuje prekmit na povolených $\Delta V_{\max} \le 0.15\text{ V}$ nad nominálnych $0.85\text{ V}$ ($V_{\text{clamped}} \le 1.00\text{ V}$).
Časová konštanta ustálenia je:
$$\tau = \frac{L}{Z_0} = \frac{2.5\text{ nH}}{50\ \Omega} = 50.0\text{ ps} \implies 5\tau = 250.0\text{ ps} \quad (99.3\%\ \text{ustálenie})$$

```
Napätie V(t)
▲
│  187.5 V ── ── ── ── ── ── ── ── ── ── [Nechránený ráz bez tlmenia]
│
│   1.00 V ──┬─────────── [Kremíkový limit V_target + 0.15 V]
│            │\
│   0.85 V ──┼─\───────────────────────── [Nominálne napätie V_target]
│            │  \___ Ustálenie (5*tau = 250 ps)
└────────────┴──────────────────────────► Čas (ps)
             0   50  100  150  200  250
```

#### 2. Kód referenčnej inferencie
```python
from krystal_kernel.textbook_inferable_patterns import infer_uma_inductive_voltage_clamping

res = infer_uma_inductive_voltage_clamping(delta_current_amps=15.0, transition_time_ps=200.0)
assert res.raw_induced_voltage_volts == 187.5
assert res.clamped_voltage_volts <= 1.00
assert res.settling_time_ps == 250.0
assert res.overshoot_within_silicon_limit is True
```

---

### Pattern 7: Sub-50ns Pulzná Sekvenčná Telemetrická Mriežka

#### 1. Fyzikálno-architektonický mechanizmus
Sledovanie ultra-rýchlych mikrostavov na hostiteľskom procesore nesmie degradovať samotný meraný kód. Bežné I/O zápisy alebo alokácia pamäte generujú rušivý zásah pozorovateľa (Observer Effect).
Riešením je formátovanie 8-bajtového `PulseFrame` slova s priamym nestreamovaným zápisom (inštrukcia `MOVNTI`), ktorý obchádza L1/L2 cache linky:
```
Bit 63       56 55       48 47       40 39       32 31                             0
┌──────────────┬───────────┬───────────┬───────────┬─────────────────────────────────┐
│ Channel (8b) │Status (8b)│VitalHP(8b)│SeqNum (8b)│   Delta TSC Ticks (32b uint)    │
└──────────────┴───────────┴───────────┴───────────┴─────────────────────────────────┘
```
Režijné náklady sondy dosahujú iba $11.34\text{ ns}$. Pri intervale vzorkovania $10\ \mu\text{s}$ ($10000\text{ ns}$) je zásah do systému:
$$\eta = \frac{11.34\text{ ns}}{10000\text{ ns}} = 0.001134 = 0.1134\% \le 0.12\%$$

#### 2. Kód referenčnej inferencie
```python
from krystal_kernel.textbook_inferable_patterns import infer_sub50ns_pulse_telemetry

res = infer_sub50ns_pulse_telemetry(channel_id=0x42, delta_tsc_ticks=48, tsc_freq_ghz=2.4)
assert res.inferred_latency_ns == 20.0
assert res.observer_overhead_ratio < 0.0015
assert res.is_non_temporal_aligned is True
```

---

### Pattern 8: Dual-Mode Epistemic State Gate (ArtifactGate)

#### 1. Fyzikálno-architektonický mechanizmus
Základné epistemické pravidlo Antigravity zakazuje samocertifikáciu. Žiadny navrhovateľ (`PROPOSER`, napr. mód `EXPLORE`) nesmie sám autorizovať svoj výstup ako fakt.
Stavová brána implementuje ústavný kontrakt:
$$\text{Release}(x) \iff \text{ReceiptValid}(x) \land \text{AuthorityValid}(x) \land (\text{HP} \equiv 6) \land \text{FalsifierChecked}(x)$$
$$\neg \text{Release}(x) \implies \text{Quarantine}(x) \land \Delta S \equiv 0$$

```
    ┌───────────────────────────┐
    │     NÁVRH AKCIE (x)       │
    │  Navrhovateľ: EXPLORE     │
    └─────────────┬─────────────┘
                  │
                  ▼
    ┌───────────────────────────┐         Zlyhanie ktorejkoľvek podmienky
    │     ARTIFACT GATE         │ ────────────────────────────────────────────────┐
    │  - Proposer != Evaluator  │                                                 │
    │  - HMAC Receipt Platný    │                                                 │
    │  - VITAL_MAX_HP == 6      │                                                 │
    │  - Falzifikátor Preverený │                                                 │
    └─────────────┬─────────────┘                                                 │
                  │ VŠETKO PLATNÉ                                                 │
                  ▼                                                               ▼
    ┌───────────────────────────┐                                   ┌───────────────────────────┐
    │       COMMIT STAVU        │                                   │        KARANTÉNA          │
    │       S_{t+1} = S_t (+) x │                                   │     S_{t+1} == S_t        │
    │     Delta S = COMMITTED   │                                   │    Delta S == ZERO (0)    │
    └───────────────────────────┘                                   └───────────────────────────┘
```

#### 2. Kód referenčnej inferencie
```python
from krystal_kernel.textbook_inferable_patterns import infer_artifact_gate_commit

# 1. Neplatná akcia (pokus o samocertifikáciu: proposer == evaluator)
res_bad = infer_artifact_gate_commit(
    action_name="MUTATE_STATE",
    proposer_role="EXPLORE_AGENT",
    evaluator_role="EXPLORE_AGENT",  # CHYBA: Samocertifikácia!
    receipt_valid=True,
    authority_valid=True,
    falsifier_passed=True
)
assert res_bad.state_committed is False
assert res_bad.delta_s == "DELTA_S_ZERO"

# 2. Platná akcia s nezávislým evaluátorom
res_ok = infer_artifact_gate_commit(
    action_name="MUTATE_STATE",
    proposer_role="EXPLORE_AGENT",
    evaluator_role="VERIFY_ENGINE",  # Nezávislý audit
    receipt_valid=True,
    authority_valid=True,
    falsifier_passed=True
)
assert res_ok.state_committed is True
assert res_ok.delta_s == "COMMITTED"
```

---

## 4. Spustenie a Verifikácia v Riadiacom Centre

Všetkých 8 učebnicových patternov je plne integrovaných v jadre `krystal_kernel`, registrovaných v Theory Dependency Graph (`THEORY-TEXTBOOK-INFERABLE-PATTERNS`) a overiteľných cez automatizovaný testovací suite:

```bash
# 1. Spustenie dedikovanej testovacej sady (9/9 unit testov)
python -m unittest tests/test_textbook_inferable_patterns.py

# 2. Spustenie celého operačného riadiaceho centra a kompilátora
bash sovereign_operations_compiler.sh --all
```

---

## 5. Epistemický Súhrn a Záverečné Pravidlo

> **Pravidlo STOP:** Každý inferovateľný pattern musí zachovávať ústavnú rovnováhu. Ak v ktorejkoľvek inferenčnej vetve chýba kalibrovaný dôkaz, telemetrický signál zlyhá, alebo dôjde k pokusu o samocertifikáciu, operátor sa prepne do stavu `DELTA_S_ZERO`. Žiadny generovaný naratív nesmie nahradiť chýbajúcu fyzikálnu evidenciu.
