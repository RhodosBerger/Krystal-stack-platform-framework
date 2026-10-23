# SPECIFICATION: HOUSEHOLD-EDGE-128B
# KRYSTAL STACK: GRID-TO-HOUSEHOLD EDGE ARCHITECTURE & 128B INVERTER PROTOCOL

**Specification ID:** `SPEC-HOUSEHOLD-EDGE-128B-V01`  
**Milestone:** Hardening & Autonomous Execution (Household Edge Integration)  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  
**Target Infrastructure:** Slovak 3-Phase Residential Grid (3x 230V / 400V, 3x 25A Main Breaker = 17.25 kW max)  
**Core Components:**
- **CLU (Compute Logic Unit):** 5.4 kW Krystal Stack Cluster (10x GPU + 128-core EPYC) with liquid cooling heat exchanger
- **Asymmetric 3-Phase Hybrid Inverter:** Up to 15 kW bidirectional with independent phase regulation ($L_1, L_2, L_3$)
- **Battery Storage:** 20 kWh LiFePO4 (1C peak discharge, $\text{LCOS} \approx 0.038\ \text{EUR/kWh}$)
- **Thermal Storage:** 300L Domestic Hot Water (DHW / TÚV) storage cylinder with integrated heat coil
- **CT Telemetry:** Fast millisecond-level current transformers on main incomer ($L_1, L_2, L_3$)

---

## 1. 128-Bajtový Binárny Protokol (C / Rust Struct Parity)

Všetka telemetria medzi CT sondami, meničom, batériou a riadiacou jednotkou CLU prebieha cez **presne 128-bajtový naturally-aligned rámec** (Princíp P2 Krystal Stacku):

### 1.1 Rust Štruktúra (`krystal_core_ffi`)
```rust
#[repr(C, align(128))]
pub struct HouseholdEdgeFrame128 {
    // 0..7: Hlavička a Invariant
    pub magic: u32,               // 0x4B525953 ("KRYS")
    pub version: u8,             // 0x02
    pub vital_max_hp: u8,        // 0x06 (Constitutional Invariant)
    pub mode: u8,                // HouseholdEdgeMode (1..5)
    pub flags: u8,               // Bit 0: Breaker Warning, Bit 1: Island Mode, Bit 2: Zero-Export
    pub timestamp_us: u32,       // Mikrosekundy od epochy

    // 12..29: Trojfázová telemetria CT sond (Point of Common Coupling)
    pub i_grid_cma: [u16; 3],    // L1, L2, L3 prúd v stotinách ampéra (10 mA, napr. 2500 = 25.00 A)
    pub v_grid_dmv: [u16; 3],    // L1, L2, L3 napätie v desatinách voltu (100 mV, napr. 2300 = 230.0 V)
    pub cos_phi_bp: [i16; 3],    // L1, L2, L3 účinník v bázických bodoch (-10000..+10000)

    // 30..35: Príkazy asymetrického vstrekovania meniča
    pub p_inv_cmd_w: [i16; 3],   // Požadovaný výkon meniča na L1, L2, L3 (+ = vybíjanie, - = nabíjanie)

    // 36..43: Stav batériového úložiska (LFP)
    pub soc_bp: u16,             // State of Charge v stotinách percenta (0..10000 bp = 0..100.00 %)
    pub v_batt_dmv: u16,         // Napätie batérie v desatinách voltu (100 mV)
    pub i_batt_cma: i16,         // Prúd batérie v 10 mA (+ = nabíjanie, - = vybíjanie)
    pub t_batt_ddeg: i16,        // Teplota batérie v desatinách °C (0.1 °C)

    // 44..49: CLU Výpočet a termálna rekuperácia
    pub p_compute_w: u16,        // Aktuálny príkon CLU klastra (0..5400 W)
    pub q_thermal_dhw_w: u16,    // Teplo odvádzané do bojlera TÚV (W)
    pub eta_ppc_bp: u16,         // Koeficient PPC v bázických bodoch (10000 = 1.0000)

    // 50..57: Sieť a trh OKTE
    pub c_spot_cents_mwh: i32,   // Spotová cena elektriny v centoch/MWh (môže byť záporná)
    pub c_dist_cents_mwh: u16,   // Distribučný poplatok v centoch/MWh
    pub grid_freq_mhz_offset: u16, // Frekvencia siete ako offset od 49.000 Hz v mHz

    // 58..73: Kryptografický auditný odtlačok
    pub state_hash: [u8; 16],    // Prvých 16 bajtov SHA-256 z hlavičky

    // 74..95: L3 Grid Broker & Fleet Phased-Array Control (Rozšírenie _reserved)
    pub transformer_id: u32,        // ID lokálnej VN/NN trafostanice (22kV/0.4kV)
    pub phase_topology_mask: u8,   // Mapovanie fáz domu voči trafostanici (permutácia L1/L2/L3)
    pub battery_topology_mode: u8, // 0 = 48V/400V Parallel, 1 = 800V Series Grid-Boost
    pub q_reactive_cmd_var: [i16; 3], // Príkaz na kompenzáciu jalového výkonu (VAr) pre udržanie < 253.0 V
    pub _pad_vpp: u16,              // Zarovnanie na 4-bajtovú hranicu
    pub vpp_dispatch_target_w: i32, // Požadovaná agregovaná odchýlka od Brokera (+ dodávka / - odber)
    pub broker_contract_id: u32,    // ID aktívneho kontraktu podporných služieb (FCR / aFRR)
    pub _reserved_tail: [u8; 32],   // Zostávajúce zarovnanie do 128 bajtov (96..127)
}
```

---

## 2. Kľúčové algoritmy riadiacej jednotky (CLU)

### 2.1 Asymetrické vyrovnávanie fáz (Asymmetric Phase Balancing)
Menič reguluje každú fázu nezávisle:
$$P_{\text{inv}, L_i} = \min\left(P_{\text{home}, L_i},\ P_{\text{inv\_phase\_max}}\right)$$
Ak spotrebič na $L_1$ odoberá $3\,500\ \text{W}$ a fázy $L_2, L_3$ odoberajú len po $200\ \text{W}$, menič vstrekne do $L_1$ presne $3\,500\ \text{W}$ z batérie. Odber zo siete na $L_1$ zostáva na hodnote **$0\ \text{W}$** bez pretekania energie medzi fázami.

### 2.2 Dynamická ochrana hlavného ističa 3x25A (Dynamic Peak Shaving)
Limit pre každú fázu je $I_{\max} = 25.0\ \text{A}$ ($P_{\max, L_i} = 5\,750\ \text{W}$).
Ak v čase lacnej elektriny beží:
- Nabíjanie batérie: $3 \times 3\,333\ \text{W} = 10\,000\ \text{W}$ ($L_1: 3.33\ \text{kW}$)
- CLU výpočet: $3 \times 1\,800\ \text{W} = 5\,400\ \text{W}$ ($L_1: 1.80\ \text{kW}$)
- Súčet na $L_1$ pred zapnutím spotrebiča: $5.13\ \text{kW}$ ($22.3\ \text{A}$, bezpečné)
Ak používateľ zapne rýchlovarnú kanvicu na $L_1$ ($2\,200\ \text{W}$), súčet by dosiahol $7.33\ \text{kW}$ ($31.9\ \text{A} \implies$ istič by do 15 sekúnd vybavil).
**Zásah CLU (< 10 ms):**
1. CT sonda na $L_1$ deteguje nárazový prúd kanvice.
2. CLU okamžite zníži nabíjací prúd batérie na $L_1$ z $3.33\ \text{kW}$ na $1.13\ \text{kW}$ (alebo dočasne zníži odber CLU).
3. Celkový prúd na $L_1$ nepresiahne $24.0\ \text{A}$. Istič **nikdy nevybaví**.

### 2.3 Termodynamická kaskáda (Dvojité využitie 1 kWh)
Elektrická energia spotrebovaná CLU (5,4 kW) pri lacnej nočnej sadzbe sa neplytvá do vzduchu:
$$Q_{\text{DHW}} = \eta_{\text{exchanger}} \cdot P_{\text{compute}} \approx 0.92 \cdot 5\,400\ \text{W} \approx 4\,968\ \text{W}$$
Tento tepelný výkon ohreje 300L bojler vody z $15^\circ\text{C}$ na $55^\circ\text{C}$ za cca 2.8 hodiny. Domácnosť tak získava teplú úžitkovú vodu zadarmo ako vedľajší produkt AI inferencie / miningu.

---

## 3. Výpočet koeficientu PPC ($\eta_{PPC}$)

$$\eta_{PPC}(t) = \frac{V_{\text{compute}}(t) + V_{\text{home\_saved}}(t) + V_{\text{dhw\_saved}}(t)}{P_{\text{grid}}(t) \cdot (C_{\text{spot}}(t) + C_{\text{dist}}(t)) + P_{\text{batt\_dis}}(t) \cdot C_{\text{deg}}}$$

V nočnom sedle (pri cene $15\ \text{EUR/MWh}$): $\eta_{PPC} > 8.5$.  
V rannej špičke (pri cene $165\ \text{EUR/MWh}$): odber zo siete $P_{\text{grid}} = 0\ \text{W}$, $\eta_{PPC} \to \infty$ (čistá úspora z batérie).
