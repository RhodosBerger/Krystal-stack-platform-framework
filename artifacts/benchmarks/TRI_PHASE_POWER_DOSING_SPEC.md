# SPECIFICATION: TRI-PHASE POWER DOSING & PHASE-AWARE GPU SHARDING
# KRYSTAL STACK: 3-PHASE INVERTER CASCADE, NEUTRAL PROTECTION & GPU SHARDING

**Specification ID:** `SPEC-TRI-PHASE-POWER-DOSING-01`  
**Milestone:** Hardening & Autonomous Execution (Tri-Phase Hardware Hardening)  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  
**Infrastructure Target:**
- **3-Phase Inverter Cascade:** 3x Independent Monophase 5 kW Inverters on $L_1, L_2, L_3$
- **PSU Sharding:** 3x 2,000 W Titanium PSUs (PSU #1 $\to L_1$, PSU #2 $\to L_2$, PSU #3 $\to L_3$)
- **Compute Cluster:** 10x AMD GPU + 2x AMD EPYC (5.4 kW nominal peak)
  - Phase $L_1$: GPUs 0, 1, 2 + EPYC Core Complex 1 (max 1,800 W)
  - Phase $L_2$: GPUs 3, 4, 5 + EPYC Core Complex 2 (max 1,800 W)
  - Phase $L_3$: GPUs 6, 7, 8, 9 + Liquid Cooling Pumps & Radiators (max 1,800 W)
- **Thermal Incomer Monitoring:** 1-Wire / PT1000 sensors on terminals $L_1, L_2, L_3$ and Neutral ($N$)
- **Critical Threshold:** $T_{\text{alarm}} \ge 45.0^\circ\text{C}$ triggers immediate phase throttling
- **Thermal Heat Valve:** 3-Way motorized valve: 0 = Bojler TÚV ($< 65^\circ\text{C}$), 1 = External Dry-Cooler

---

## 1. Elektrická a termálna architektúra rozvádzača

```text
[ Verejná sieť 3x 230V/400V - 25A Hlavný Istič ]
      │ (CT Sondy: Prúd a teplota svoriek vzorkované každých 10 ms)
      ▼
[ KRYSTAL POWER SPINE (Silová chrbtica CYKY-J 5x10 mm²) ]
 ├──► Fáza L1 (max 23.9 A) ──► [ Menič #1 (5 kW) ] ──► PSU #1 (1 800 W) ──► GPUs 0, 1, 2 + EPYC #1
 ├──► Fáza L2 (max 23.9 A) ──► [ Menič #2 (5 kW) ] ──► PSU #2 (1 800 W) ──► GPUs 3, 4, 5 + EPYC #2
 ├──► Fáza L3 (max 23.9 A) ──► [ Menič #3 (5 kW) ] ──► PSU #3 (1 800 W) ──► GPUs 6, 7, 8, 9 + Pumpy
 └──► Nulový vodič (N) ─────► Zdvojený prierez 10 mm² s aktívnym posunom PWM spínania
```

### 1.1 Ochrana nulového vodiča ($N$) pred 3. harmonickou (150 Hz)
Spínané zdroje počítačov produkujú 3. harmonickú frekvenciu. V 3-fázovej sústave sa prúdy tretej harmonickej neodčítajú, ale sčítavajú v nulovom vodiči ($I_N = 3 \times I_{h3}$).
- **Architektonické opatrenie 1:** Zdvojený nulový vodič ($10\ \text{mm}^2$).
- **Architektonické opatrenie 2:** Fázovo posunuté spínanie meničov (Active Harmonic Cancellation): riadiace PWM meničov #1, #2, #3 bežia s posunom $120^\circ$ fázového uhla, čím eliminujú kumuláciu 150 Hz prúdov.

---

## 2. Riadiace algoritmy Krystal Jadra

### 2.1 Time-Sliced Power Dosing (Pulzné dávkovanie výkonu)
Pri dodávke energie do siete meniče nepracujú na 100 % konštantnom prúde, ktorý by kumulatívne zahrieval svorkovnice. Riadiaca slučka strieda výkonové okná (Duty-Cycle Dosing):
- **Okno T1 (napr. 5 s):** Fáza $L_1$ dodáva 100 % (23.5 A), fázy $L_2, L_3$ pracujú na 60 % (tepelné relaxačné okno).
- **Okno T2:** Záťaž preberá $L_2$, $L_1$ chladne.
- **Okno T3:** Záťaž preberá $L_3$, $L_2$ chladne.

### 2.2 Phase-Aware GPU Scheduling (Mapovanie úloh na fázy)
Každá skupina GPU má priradenú dynamickú prioritu podľa okamžitého zaťaženia danej fázy:
- **Priorita 1 (`AI_INFERENCE`):** $80 - 120\ \text{W}$ na kartu. Priradená k fáze s vysokým zaťažením domácnosti (napr. zapnutý sporák na $L_1$).
- **Priorita 2 (`ZKP_MATH`):** $180 - 220\ \text{W}$ na kartu. Stredná záťaž.
- **Priorita 3 (`CRYPTO_MINING`):** $350 - 450\ \text{W}$ na kartu. Povolí sa len na fázach, kde sú svorky studené ($< 40^\circ\text{C}$) a menič neposiela špičkový výkon do siete.

---

## 3. Binárny protokol: 128B Rámec s TriPhase Dosing rozšírením

```rust
#[repr(C, packed)]
pub struct TriPhaseDosingExtension {
    // 96..103: Teplota káblov a svoriek v rozvádzači (v 0.1 °C, napr. 385 = 38.5 °C)
    pub t_cable_l1_ddeg: i16,
    pub t_cable_l2_ddeg: i16,
    pub t_cable_l3_ddeg: i16,
    pub t_cable_neutral_ddeg: i16, // Kritické sledovanie nulového vodiča N

    // 104..109: Rozdelenie príkonu GPU podľa fáz (PSU #1, #2, #3)
    pub p_gpu_group_w: [u16; 3],   // Max 1 800 W na fázu (spolu 5 400 W)

    // 110..112: Priorita úloh na jednotlivých fázach
    // 0 = IDLE, 1 = AI_INFERENCE, 2 = ZKP_MATH, 3 = CRYPTO_MINING
    pub gpu_phase_priority: [u8; 3],

    // 113..115: Pulzné dávkovanie batérie (Duty Cycle 0..100 % pre Menič #1, #2, #3)
    pub inverter_dosing_duty_pct: [u8; 3],

    // 116..117: Stav 3-cestného ventilu a padding
    pub thermal_valve_state: u8,   // 0 = Bojler TÚV, 1 = Vonkajší Dry-Cooler
    pub _padding: u8,
}
```

Celkový layout 128-bajtového rámca:
- `0..57`: Hlavička a 3-fázová CT telemetria (58 B)
- `58..73`: Kryptografický SHA-256 state hash (16 B)
- `74..95`: L3 Grid Broker & Phased Array polia (22 B)
- `96..117`: TriPhase Dosing & GPU Sharding rozšírenie (22 B)
- `118..127`: Koncové zarovnanie (10 B)
$$\mathbf{58 + 16 + 22 + 22 + 10 = 128\ \text{Bajtov}}$$
