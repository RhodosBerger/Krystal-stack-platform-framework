# EXPERIMENT SPECIFICATION: MOONSHOT-C
# PREDICTIVE GRID-AWARE ENERGY ARBITRAGE ENGINE & DYNAMIC COMPUTE BALANCER

**Specification ID:** `SPEC-MOONSHOT-C-GRID-ARBITRAGE-01`  
**Milestone:** Fáza Hardening & Autonomous Execution (Moonshot C)  
**Target Invariant:** `VITAL_MAX_HP == 6`  
**Constitutional Rule:** $\neg\text{Valid}(Action) \implies \Delta S = 0$  
**Hardware Cluster Target:** 10x AMD GPU + 128-core AMD EPYC Host Node  
**Cluster Power Limits:** $P_{\min} = 500.0\ \text{W}$ (Idle / Minimal AI) $\le P(t) \le P_{\max} = 5,400.0\ \text{W}$ (Full Load Peak)  
**Network Ingestion Target:** Sub-millisecond XDP/eBPF 128B Zero-Copy Contract Frames  

---

## 1. Motivácia a ekonomicko-fyzikálny rámec

V stredoeurópskom elektroenergetickom priestore (Slovenská prenosová sústava - SEPS, denný a vnútrodenný trh OKTE):
1. **Volatilita cien elektriny:** Vplyvom masívneho rozvoja solárnych a veterných zdrojov klesá spotová cena $P_{\text{spot}}$ počas slnečných a veterných špičiek do záporných hodnôt (až $-30\ \text{EUR}/\text{MWh}$), kým v čase rannej a večernej špičky prekračuje $150\ \text{EUR}/\text{MWh}$.
2. **Frekvenčná stabilizácia siete (FCR - Frequency Containment Reserve):** Synchrónna sieť pracuje na nominálnej frekvencii $f_0 = 50.000\ \text{Hz}$. Pri odchýlke $|\Delta f| > 0.020\ \text{Hz}$ SEPS a prevádzkovatelia sietí platia regulačný bonus subjektom, ktoré v reálnom čase (do stoviek milisekúnd) modulujú svoj odber a pomáhajú udržať rovnováhu.
3. **Statický odber je ekonomická smrť:** Prevádzkovať 5,4 kW výpočtový uzol pri fixnom odbere znamená obrovské straty pri vysokých cenách elektriny a nevyužitý zisk pri záporných cenách.

**Moonshot C transformuje Krystal Stack z pasívneho spotrebiteľa na aktívny virtuálny regulačný uzol**, ktorý autonómne a v sub-milisekundovom takte arbitruje medzi dvoma typmi záťaže:
- **Ťažba kryptomien (Crypto Mining / Hash Workload):** Vysoký odber ($P \to 5,400\ \text{W}$), nízka marža na jednotku energie, ale ideálna flexibilita na absorpciu prebytočnej energie (upward demand response).
- **Vysoko maržová AI inferencia (`MultiModelChatInferenceAdapter`):** Variabilný odber ($500\ \text{W} - 2,200\ \text{W}$), vysoká pridaná hodnota na token ($> 0.85\ \text{EUR}/\text{kWh}$ eq.), prioritizovaná pri drahých tarifách.

---

## 2. Matematický model kvadratickej arbitráže

Optimalizačný engine v každom riadiacom cykle rieši úlohu maximalizácie čistého zisku $\Pi(P, \alpha)$:

$$\max_{P \in [500, 5400], \alpha \in [0, 1]} \Pi(P, \alpha) = \Pi_{\text{workload}}(P, \alpha) + \text{Rev}_{\text{FCR}}(P, \Delta f) - \text{Cost}_{\text{grid}}(P, P_{\text{spot}}) - \text{Cost}_{\text{thermal}}(P, T_j, Z)$$

### 2.1 Zložky modelu
1. **Zisk zo záťaže:**
   $$\Pi_{\text{workload}}(P, \alpha) = \alpha \cdot \text{Rev}_{\text{mine}}(P) + (1 - \alpha) \cdot \text{Rev}_{\text{AI}}(P)$$
   kde:
   - $\text{Rev}_{\text{mine}}(P) = r_{\text{mine}} \cdot P \quad [r_{\text{mine}} = 0.045\times 10^{-3}\ \text{EUR}/\text{Wh}]$
   - $\text{Rev}_{\text{AI}}(P) = R_{\text{max\_AI}} \cdot \left(1 - \exp\left(-\frac{P - P_{\min}}{1200.0}\right)\right)$

2. **Cena odberu zo siete (OKTE):**
   $$\text{Cost}_{\text{grid}}(P, P_{\text{spot}}) = P \cdot \frac{P_{\text{spot}}}{1\,000\,000}\quad [\text{EUR}/\text{hodina}]$$
   *Poznámka:* Pri $P_{\text{spot}} < 0$ je $\text{Cost}_{\text{grid}} < 0$, čiže systém dostáva zaplatené za odber energie!

3. **Bonus za primárnu reguláciu frekvencie (FCR):**
   $$\text{Rev}_{\text{FCR}}(P, \Delta f) = C_{\text{FCR}} \cdot \text{clamp}\left(\frac{\Delta f - \Delta f_{\text{deadband}}}{\Delta f_{\max} - \Delta f_{\text{deadband}}}, -1.0, 1.0\right) \cdot P$$
   - Ak $\Delta f > +0.020\ \text{Hz}$ (nadfrekvencia $\implies$ prebytok v sieti), systém navyšuje odber na $5,400\ \text{W}$ a inkasuje FCR bonus.
   - Ak $\Delta f < -0.020\ \text{Hz}$ (podfrekvencia $\implies$ deficit v sieti), systém zhadzuje záťaž na $500\ \text{W}$ a inkasuje poplatok za odľahčenie siete.

4. **Termálna amortizácia (Princíp P7):**
   Prekročenie nominálnej teploty kremíka ($T_j > 75^\circ\text{C}$) alebo vysoká termálna impedancia $Z = R + jX$ penalizuje zisk podľa Arrheniovho zákona starnutia.

---

## 3. Diskrétne prevádzkové režimy (Workload Modes)

| Režim | Názov | Podmienka aktivácie | Cieľový príkon $P^*$ | Pomer Mining : AI |
| :---: | :--- | :--- | :---: | :---: |
| **M-1** | `CRYPTO_MINING` | $P_{\text{spot}} \le 20.0\ \text{EUR}/\text{MWh} \lor \Delta f \ge +0.05\ \text{Hz}$ | $4,500\ \text{W} - 5,400\ \text{W}$ | $1.0 : 0.0$ |
| **M-2** | `HIGH_MARGIN_AI_INFERENCE` | $P_{\text{spot}} \ge 110.0\ \text{EUR}/\text{MWh} \lor \Delta f \le -0.05\ \text{Hz}$ | $500\ \text{W} - 1,800\ \text{W}$ | $0.0 : 1.0$ |
| **M-3** | `DYNAMIC_HYBRID_ARBITRAGE` | $20.0 < P_{\text{spot}} < 110.0\ \text{EUR}/\text{MWh} \land |\Delta f| < 0.05\ \text{Hz}$ | Kvadratické optimum | $\alpha^* : (1 - \alpha^*)$ |

---

## 4. XDP 128B sieťový paket energetickej telemetrie

Každý prichádzajúci sieťový rámec s telemetriou z inteligentného elektromera / OKTE API je zarovnaný na **128 bajtov** (Princíp P2):

```text
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       Magic: 0x4B525953 ("KRYS")      | Version (u8)  | VITAL(0x06)  |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 Timestamp Delta (uint32_t microsec)           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|      Grid Frequency (u32, mHz)        | Frequency Dev (i16, mHz)|Mode|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         OKTE Spot Price (int32_t, cents / MWh)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Target Power (u16, W) | Alpha Ratio (u16, bp) | Reserved (28B)|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Padding to 128B                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

---

## 5. Invarianty a falzifikačné kritériá

1. **Konštitučný invariant:** `VITAL_MAX_HP == 6` v každom výstupnom stave a telemetrickom pulze.
2. **Výkonový strop:** Príkon nesmie nikdy prekročiť $5,400.0\ \text{W}$ ani klesnúť pod $500.0\ \text{W}$.
3. **Termálne predradenie:** Ak kremíková telemetria hlási $T_j > 82^\circ\text{C}$ (`SILICON_THERMAL_THROTTLE`), ekonomický cieľ $5,400\ \text{W}$ je okamžite prepísaný núdzovým termálnym stropom ($\le 2,800\ \text{W}$) bez ohľadu na záporné ceny elektriny ($\Delta S = 0$ na riskantné akcie).
4. **Deterministický prechod:** Zhodné vstupy $(P_{\text{spot}}, \Delta f, T_j)$ musia vygenerovať identický 128B hash a stavové preklopenie.
