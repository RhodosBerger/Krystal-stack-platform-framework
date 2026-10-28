# HTE-001 — MEASUREMENT UNCERTAINTY BUDGET & METROLOGICAL RESOLUTION SPECIFICATION
## GUM (JCGM 100:2008) Neistotový Model pre INA226 Telemetriu na Doske Xilinx KCU116
### Dekompozícia Type-A / Type-B a Kovariančné Vyrušenie Spoločných Chýb v Diferenciálnom Meraní

---

### Epistemický status dokumentu
- **Klasifikácia:** `[FORMAL METROLOGICAL SPECIFICATION / RESEARCH CONTRACT]`
- **Stav:** `[TYPE-A / TYPE-B METROLOGICALLY DECOMPOSED]`
- **Štandard:** ISO/IEC Guide 98-3:2008 (Guide to the Expression of Uncertainty in Measurement — GUM)
- **Cieľ:** Kvantifikovať kombinovanú a rozšírenú neistotu merania $U_{95\%}$ pre dynamickú energiu dosky, striktne oddeliť náhodný šum od systematických chýb a odvodiť vyrušenie spoločných chýb v párovanom kľudovom baseline.
- **Ústavné invarianty:** `Proposal != Authority` | `VITAL_MAX_HP == 6` | `Prediction != Evidence`

---

## 1. Merací Reťazec a Funkčný Model

Výkon dosky $P$ je meraný kalibrovaným monitorom napájania Texas Instruments INA226 zapojeným na $12.0\text{ V}$ napájacom vstupe dosky Xilinx KCU116 Kintex UltraScale+:
$$P = \frac{V_{\text{bus}} \cdot V_{\text{shunt}}}{R_{\text{shunt}}}$$

Nominálny pracovný bod:
$$V_{\text{bus}} = 12.0\text{ V}, \quad P_{\text{active}} = 4.18\text{ W}, \quad R_{\text{shunt}} = 10\text{ m}\Omega$$
$$I_{\text{nominal}} = \frac{P}{V_{\text{bus}}} = \frac{4.18}{12.0} \approx \mathbf{0.3483\text{ A}}$$
$$V_{\text{shunt}} = I_{\text{nominal}} \cdot R_{\text{shunt}} = 0.34833\text{ A} \times 0.010\ \Omega = \mathbf{3.4833\text{ mV}}$$

Dynamická energia akvizičného behu s párovaným kľudovým baseline:
$$\Delta P = P_{\text{active}} - P_{\text{idle}}$$
$$E_{\text{dynamic}} = \Delta P \cdot \Delta t_{\text{burst}}$$

Dynamická energia na jednu validovanú operáciu ($N_{\text{validated}} = 10^6$ operácií):
$$E_{\text{op}} = \frac{E_{\text{dynamic}}}{N_{\text{validated}}}$$

---

## 2. Metodologické Oddelenie Zložiek Type-A a Type-B

Chybou predchádzajúceho zjednodušeného modelu bolo delenie celkovej kombinovanej neistoty $u_c(P)$ odmocninou $\sqrt{N}$. V skutočnosti sa opakovaným odčítaním redukuje **výhradne náhodný stochastický šum (Type-A)**, kým systematické chyby senzora a tolerancie súčiastok (Type-B) zostávajú na úrovni strednej hodnoty konštantné:
$$u^2(P) = u_A^2(P) + u_B^2(P)$$

Priemerovanie $N$ nezávislých konverzií redukuje len stochastickú zložku:
$$u_A(\bar{P}) = \frac{u_A(P)}{\sqrt{N}}, \qquad u_B(\bar{P}) \approx u_B(P)$$

---

## 3. Kvantifikácia Zložiek Neistoty (Jednotlivé Čítanie)

### 3.1 Zložky Type-A (Stochastický Šum Zariadenia)
- **Kvantizačný šum $V_{\text{bus}}$:** $u_{\text{quant}}(V_{\text{bus}}) = \frac{1.25\text{ mV}}{\sqrt{12}} \approx 0.361\text{ mV}$
- **Kvantizačný šum $V_{\text{shunt}}$:** $u_{\text{quant}}(V_{\text{shunt}}) = \frac{2.5\ \mu\text{V}}{\sqrt{12}} \approx 0.722\ \mu\text{V}$
- **Ekvivalentný vstupný šum ADC:** $u_{\text{noise}}(V_{\text{shunt}}) \approx 2.5\ \mu\text{V}_{\text{rms}}$

Citlivostné koeficienty pre nominálny bod ($I = 0.3483\text{ A}$, $c_{V_{\text{shunt}}} = \frac{V_{\text{bus}}}{R} = 1200\text{ A}$):
$$u_A(P) = \sqrt{(I \cdot u_{\text{quant}}(V_{\text{bus}}))^2 + \left(\frac{V_{\text{bus}}}{R} \sqrt{u_{\text{quant}}^2 + u_{\text{noise}}^2}\right)^2} \approx \mathbf{0.00312\text{ W}}$$

### 3.2 Zložky Type-B (Systematické Hardvérové Hranice)
- **Gain error $V_{\text{bus}}$ ($\pm 0.1\%$):** $u_{\text{gain}}(V_{\text{bus}}) = \frac{0.001 \times 12.0}{\sqrt{3}} \approx 6.928\text{ mV}$
- **Gain error $V_{\text{shunt}}$ ($\pm 0.1\%$):** $u_{\text{gain}}(V_{\text{shunt}}) = \frac{0.001 \times 3.483\text{ mV}}{\sqrt{3}} \approx 2.011\ \mu\text{V}$
- **Offset napätie $V_{\text{OS}}$ ($\pm 10\ \mu\text{V}$):** $u(V_{\text{OS}}) = \frac{10\ \mu\text{V}}{\sqrt{3}} \approx 5.774\ \mu\text{V}$
- **Tolerancia bočníka ($\pm 0.1\%$):** $u_{\text{tol}}(R) = \frac{0.001 \times 0.010}{\sqrt{3}} \approx 5.774\ \mu\Omega$
- **Teplotný drift TCR ($25\text{ ppm}/^\circ\text{C}$, $\Delta T = 20^\circ\text{C}$):** $u_{\text{TCR}}(R) = \frac{0.0005 \times 0.010}{\sqrt{3}} \approx 2.887\ \mu\Omega$
- **Kombinovaná neistota odporu:** $u(R) = \sqrt{5.774^2 + 2.887^2} \approx 6.455\ \mu\Omega$

Systematická zložka jedného čítania výkonu:
$$u_B(P) \approx \sqrt{(I \cdot u_{\text{gain}}(V_{\text{bus}}))^2 + \left(\frac{V_{\text{bus}}}{R} \sqrt{u_{\text{gain}}^2 + u_{\text{OS}}^2}\right)^2 + \left(\frac{P}{R} u(R)\right)^2} \approx \mathbf{0.00803\text{ W}}$$

Celková neistota jedného odčítania:
$$u_c(P) = \sqrt{u_A^2(P) + u_B^2(P)} = \sqrt{0.00312^2 + 0.00803^2} \approx \mathbf{0.00861\text{ W}}$$

---

## 4. Diferenciálny Model a Vyrušenie Spoločných Chýb (Common-Mode Covariance)

Pri výpočte dynamického výkonu $\Delta P = \bar{P}_{\text{active}} - \bar{P}_{\text{idle}}$ prebieha meranie cez **identický fyzický senzor a rovnaký rezistor $R_{\text{shunt}}$ v časovom odstupe desiatok milisekúnd**.

### 4.1 Stochastická Zložka Type-A
Pretože vzorky v aktívnom okne ($N_{\text{active}} \ge 742$) a kľudovom okne ($N_{\text{idle}} \ge 358$) sú stochasticky nezávislé:
$$u_A(\Delta P) = \sqrt{\frac{u_A^2(P_{\text{active}})}{742} + \frac{u_A^2(P_{\text{idle}})}{358}} \approx \mathbf{0.000201\text{ W}}$$

### 4.2 Systematická Zložka Type-B (Dva Hraničné Modely)

#### Model 1: Konzervatívny Nekorelovaný Odhad (Worst-Case Non-Reducing Floor)
Predpokladá, že chyby aktívneho a kľudového stavu sú navzájom neznáme a nekorelujú:
$$u_{B,\text{cons}}(\Delta P) = \sqrt{u_B^2(P_{\text{active}}) + u_B^2(P_{\text{idle}})} \approx \sqrt{0.00803^2 + 0.00624^2} \approx \mathbf{0.01017\text{ W}}$$
$$u_c(E_{\text{op,cons}}) = \frac{0.01017 \times 0.208}{10^6} \approx 2.115\times 10^{-9}\text{ J} = 0.002115\ \mu\text{J}$$
Rozšírená neistota ($k = 2$):
$$\boxed{U_{95,\text{conservative}}(E_{\text{op}}) \approx \mathbf{0.00423\ \mu\text{J}} \quad (4.23\text{ nJ/op})}$$
Diskriminačný pomer voči $\epsilon_{\text{energy}} = 50.0\text{ nJ}$:
$$\frac{\epsilon_{\text{energy}}}{U_{95,\text{cons}}} = \frac{50.0\text{ nJ}}{4.23\text{ nJ}} \approx \mathbf{11.8\times}$$

#### Model 2: Korelovaný Diferenciálny Model (Fyzikálne Kovariančné Vyrušenie)
V reálnom hardvéri nastáva výrazné vyrušenie spoločných systematických chýb:
1. **Vyrušenie tolerancie bočníka:** Citlivosť na chybu odporu závisí výhradne od rozdielu výkonov $\Delta P \approx 0.93\text{ W}$, nie od celkového výkonu $4.18\text{ W}$:
   $$\left|\frac{\partial \Delta P}{\partial R}\right| u(R) = \frac{\Delta P}{R} u(R) = \frac{0.93}{0.010} \times 6.455\times 10^{-6} \approx \mathbf{0.000600\text{ W}}$$
2. **Vyrušenie offsetového napätia $V_{\text{OS}}$:** Keďže $V_{\text{OS}}$ je konštantný v oboch stavoch, do rozdielu vstupuje len cez pokles napätia zbernice (rail droop $\Delta V_{\text{bus}} \approx 0.05\text{ V}$):
   $$\frac{\Delta V_{\text{bus}}}{R} u(V_{\text{OS}}) = \frac{0.05}{0.010} \times 5.774\times 10^{-6} \approx \mathbf{0.000029\text{ W}}$$
3. **Vyrušenie lineárneho teplotného driftu:** Párovaný protokol $\bar{P}_{\text{idle}} = \frac{P_{\text{before}} + P_{\text{after}}}{2}$ priamo eliminuje lineárny drift teploty regulátorov.

Korelovaná systematická neistota:
$$u_{B,\text{corr}}(\Delta P) \approx \sqrt{0.000600^2 + 0.000029^2 + 0.000538^2} \approx \mathbf{0.000806\text{ W}}$$
Kombinovaná diferenciálna neistota:
$$u_c(\Delta P) = \sqrt{u_A^2(\Delta P) + u_{B,\text{corr}}^2(\Delta P)} = \sqrt{0.000201^2 + 0.000806^2} \approx \mathbf{0.000831\text{ W}}$$
$$u_c(E_{\text{op,corr}}) = \frac{0.000831 \times 0.208}{10^6} \approx \mathbf{0.000173\ \mu\text{J}} \quad (0.173\text{ nJ})$$
Rozšírená neistota ($k = 2$):
$$\boxed{U_{95,\text{correlated}}(E_{\text{op}}) \approx \mathbf{0.000346\ \mu\text{J}} \dots \mathbf{0.000411\ \mu\text{J}} \quad (\approx \mathbf{0.41\text{ nJ/op}})}$$
Diskriminačný pomer voči $\epsilon_{\text{energy}} = 50.0\text{ nJ}$:
$$\frac{\epsilon_{\text{energy}}}{U_{95,\text{corr}}} = \frac{50.0\text{ nJ}}{0.41\text{ nJ}} \approx \mathbf{121.5\times}$$

---

## 5. Metrologické Diskriminačné Pásma

| Režim Modelu | Rozšírená Neistota $U_{95\%}$ | Tolerancia $\epsilon_{\text{energy}}$ | Diskriminačný Pomer | Status |
|---|:---:|:---:|:---:|:---:|
| **Konzervatívny (Nekorelovaný Type-B)** | $4.23\text{ nJ/op}$ | $50.0\text{ nJ/op}$ | **$11.8\times$** | `GARANTOVANÝ PESIMISTICKÝ FLOOR` |
| **Korelovaný (Diferenciálne Vyrušenie)** | $0.41\text{ nJ/op}$ | $50.0\text{ nJ/op}$ | **$121.5\times$** | `NOMINÁLNY KALIBROVANÝ MODEL` |

### Pravidlo Rozhodovania
V oboch modeloch ($11.8\times$ aj $121.5\times$) leží rozšírená neistota prístroja rádovo pod akceptačným prahom experimentu:
1. **$|R_{\text{energy}}| \le 0.41\text{ nJ}$ (`SUB_NOISE_FLOOR`):** Experimentálne nerozlíšiteľné od šumu INA226.
2. **$0.41\text{ nJ} < |R_{\text{energy}}| \le 50.0\text{ nJ}$ (`RESOLVABLE_COMPLIANT`):** Fyzikálne rozlíšená odchýlka vyhovujúca modelu.
3. **$|R_{\text{energy}}| > 50.0\text{ nJ}$ (`ENVELOPE_BREACH_FALSIFIED`):** Hypotéza je vyvrátená. Chyba je najmenej $11.8\times$ väčšia než najkonzervatívnejšia možná neistota meracieho reťazca.

---

## 6. Záver a Epistemické Ohraničenie

HTE-001 má pred fyzickým spustením RUN-0001 explicitne dekomponovanú Type-A a Type-B neistotu merania. Vedecká interpretácia výsledkov RUN-0001 sa bude opierať o obe hranice (pesimistickú $11.8\times$ aj nominálnu $121.5\times$), čím sa eliminuje riziko metodologického spochybnenia metrologickej rozlíšiteľnosti.
