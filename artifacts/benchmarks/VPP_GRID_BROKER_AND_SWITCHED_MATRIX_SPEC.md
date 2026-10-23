# SPECIFICATION: VPP-GRID-BROKER-AND-SWITCHED-MATRIX
# KRYSTAL STACK: VIRTUAL POWER PLANT (VPP) BROKER & SWITCHED BATTERY MATRIX

**Specification ID:** `SPEC-VPP-BROKER-FLEET-01`  
**Milestone:** Hardening & Autonomous Execution (VPP & Substation Coordination)  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  
**Target Infrastructure:** 
- **Substation Level (L2/L3):** 22 kV / 0.4 kV, 630 kVA Oil/Dry Transformer supplying a feeder with 100 households
- **Feeder Limits:** Cable NAYY 4x150 mm² (length 800 m, $R \approx 0.206\ \Omega/\text{km}$, $X \approx 0.08\ \Omega/\text{km}$)
- **Voltage Standards:** EN 50160 ($230\ \text{V} \pm 10\% \implies V_{\max} = 253.0\ \text{V}$, $V_{\min} = 207.0\ \text{V}$)
- **Household Edge Level (L1):** 100x Krystal Edge Nodes (100x 20 kWh LFP = 2.0 MWh fleet, 100x 5.4 kW CLU = 540 kW flexible compute)
- **Switched Matrix:** SiC MOSFET Matrix reconfiguring between 48V parallel (normal cycling) and 800V DC bus (Grid Boost export)

---

## 1. Fyzikálno-matematický model distribučného vedenia a prepätia

### 1.1 Napäťový profil pozdĺž nízkonapäťového vývodu
Pre uzol $k \in \{1, \dots, N\}$ vo vzdialenosti $d_k$ od trafostanice je napätie $V_k$ dané zovšeobecnenou aproximáciou toku výkonu:

$$V_k = V_{\text{substation}} + \sum_{j=1}^{k} \frac{R_j \cdot P_{\text{net}, j} + X_j \cdot Q_{\text{net}, j}}{V_0}$$

Kde:
- $V_{\text{substation}} \approx 230.0\ \text{V}$ (sekundárne napätie transformátora)
- $P_{\text{net}, j} = P_{\text{export}, j} - P_{\text{home}, j} - P_{\text{compute}, j}$ (činný výkon vstrekovaný do siete v uzle $j$)
- $Q_{\text{net}, j}$ je jalový výkon meniča v uzle $j$ (induktívny $Q < 0$ znižuje napätie, kapacitný $Q > 0$ zvyšuje napätie)
- $R_j, X_j$ sú rezistancia a induktívna reaktancia úseku vedenia

### 1.2 Riziko prepätia ($V_k > 253.0\ \text{V}$)
Pri masívnom neorganizovanom exporte batérií do siete ($P_{\text{net}, j} > 0$) napätie na konci vedenia ($k \to 100$) prekročí **253.0 V**, čo spôsobí automatické odpojenie bežných striedačov podľa normy EN 50160.

### 1.3 Riadenie ako fázované pole (Phased Array Voltage Stabilization)
Krystal VPP Broker rieši optimalizačnú úlohu:

$$\max_{P_j, Q_j, P_{\text{compute}, j}} \sum_{j=1}^{N} \Pi_j \quad \text{s.t.} \quad \forall k: V_k \le 253.0\ \text{V} \land \sum_{j=1}^{N} P_j \le S_{\text{trafo\_max}}$$

1. **Selektívna dodávka činného výkonu ($P$):** Uzly blízko trafostanice ($k \le 40$) vstrekujú maximálny činný výkon $P$, kde je impedancia vedenia nízka.
2. **Jalová kompenzácia ($Q < 0$):** Uzly v strednej a koncovej časti vedenia vstrekujú induktívny jalový výkon ($\cos \varphi < 1.0$), čím kompenzujú $\Delta V$.
3. **CLU Tepelné stlačenie špičky:** Ak napätie na konci vedenia ($k \ge 80$) atakuje 252.0 V, broker zapne na koncových uzloch 5.4 kW CLU (ťažba/AI) s odvodom tepla do bojlera, čím lokálne zníži $P_{\text{net}}$ a okamžite stlačí napätie pod bezpečnú hranicu.

---

## 2. L1: Switched-Capacitor / SiC MOSFET Rekonfigurácia batérií

### 2.1 Režim 48V Paralelný (Štandardný cyklus)
- Batériové packy (napr. 16x 48V 100Ah modulov) sú paralelne prepojené.
- Bezpečné napätie pre bežnú prevádzku domu, minimalizácia stresu článkov, nabíjanie z domácej siete.

### 2.2 Režim 800V Sériový („GRID BOOST“ Masívny export)
- Pri broker signáli `GRID_BOOST` matica SiC MOSFET spínačov v čase $< 5\ \text{ms}$ prekonfiguruje packy do série: $16 \times 51.2\ \text{V} = 819.2\ \text{V}$ DC bus.
- **Dôsledok pre tepelné straty:**
  Pri prenose výkonu $P = 15\ \text{kW}$:
  $$I_{48\text{V}} = \frac{15\,000\ \text{W}}{48\ \text{V}} = 312.5\ \text{A} \implies P_{\text{loss}} = I^2 R = 97\,656 \cdot R$$
  $$I_{800\text{V}} = \frac{15\,000\ \text{W}}{800\ \text{V}} = 18.75\ \text{A} \implies P_{\text{loss}} = I^2 R = 351.5 \cdot R$$
  **Zníženie tepelných strát a úbytku napätia o faktor:**
  $$\frac{P_{\text{loss}, 48\text{V}}}{P_{\text{loss}, 800\text{V}}} = \left(\frac{800}{48}\right)^2 \approx 277.7\times$$
- Vysoké napätie umožňuje 3-fázovému striedaču pracovať pri optimálnom pracovnom bode bez potreby stratového boost-konvertora.

---

## 3. L2/L3: B2B Služby pre prevádzkovateľa prenosovej sústavy (SEPS)

Krystal Broker agreguje 100 uzlov do virtuálneho bloku:
- **Kapacita úložiska:** $100 \times 20\ \text{kWh} = 2.0\ \text{MWh}$
- **Okamžitý regulačný výkon (Nahor - aFRR+):** Až $+1.2\ \text{MW}$ (vybíjanie 800V batérií)
- **Okamžitý regulačný výkon (Nadol - aFRR-):** Až $-1.54\ \text{MW}$ ($100 \times 10\ \text{kW}$ nabíjanie + $100 \times 5.4\ \text{kW}$ CLU)
- **Rýchlosť odozvy:** $< 150\ \text{ms}$ (využitie XDP 128B rámcov predbieha bežné plynové a uhoľné elektrárne vyžadujúce minúty).
