# HTE-001 — MEASUREMENT UNCERTAINTY BUDGET & METROLOGICAL RESOLUTION SPECIFICATION
## GUM (JCGM 100:2008) Neistotový Model pre INA226 Telemetriu na Doske Xilinx KCU116

---

### Epistemický status dokumentu
- **Klasifikácia:** `[FORMAL METROLOGICAL SPECIFICATION / RESEARCH CONTRACT]`
- **Štandard:** ISO/IEC Guide 98-3:2008 (Guide to the Expression of Uncertainty in Measurement — GUM)
- **Cieľ:** Kvantifikovať kombinovanú a rozšírenú neistotu merania $U_{95\%}$ pre dynamickú energiu dosky a stanoviť diskriminačnú hranicu šumu pre validáciu hypotézy $H_1$.
- **Ústavné invarianty:** `Proposal != Authority` | `VITAL_MAX_HP == 6` | `Prediction != Evidence`

---

## 1. Merací Reťazec a Funkčný Model

Výkon dosky $P$ je meraný kalibrovaným monitorom napájania Texas Instruments INA226 zapojeným na $12.0\text{ V}$ napájacom vstupe dosky Xilinx KCU116 Kintex UltraScale+:
$$P = \frac{V_{\text{bus}} \cdot V_{\text{shunt}}}{R_{\text{shunt}}}$$

Dynamická energia akvizičného behu s párovaným kľudovým baseline:
$$E_{\text{dynamic}} = (P_{\text{active}} - P_{\text{idle}}) \cdot \Delta t_{\text{burst}}$$

Dynamická energia na jednu validovanú operáciu:
$$E_{\text{op}} = \frac{E_{\text{dynamic}}}{N_{\text{validated}}}$$

---

## 2. Kvantifikácia Zložiek Neistoty (Typ B Podľa Špecifikácie Výrobcu)

Podľa GUM sú komponenty bez priamych štatistických opakovaní modelované pravouhlým rovnomerným rozdelením s deličom $\sqrt{3}$ alebo $\sqrt{12}$ (pre kvantizačný šum):

| Parameter | Znak | Nominálna Hodnota | Neistotový Model | Štandardná Neistota $u(x)$ |
|---|---|---|---|---|
| **Zbernicové napätie (Gain)** | $V_{\text{bus,gain}}$ | $12.0\text{ V}$ | Obdĺžnikové $\pm 0.1\%$ | $\frac{0.001 \times 12.0}{\sqrt{3}} \approx 6.928\text{ mV}$ |
| **Zbernicové napätie (LSB)** | $V_{\text{bus,quant}}$ | $1.25\text{ mV}$ | Obdĺžnikové $\pm \frac{\text{LSB}}{2}$ | $\frac{1.25\text{ mV}}{\sqrt{12}} \approx 0.361\text{ mV}$ |
| **Kombinované $u(V_{\text{bus}})$** | $u(V_{\text{bus}})$ | $12.0\text{ V}$ | Kvadratický súčet | $\sqrt{6.928^2 + 0.361^2} \approx 6.938\text{ mV}$ |
| **Napätie bočníka (Gain)** | $V_{\text{shunt,gain}}$ | $4.18\text{ mV}$ (@ 4.18 W) | Obdĺžnikové $\pm 0.1\%$ | $\frac{0.001 \times 4.18\text{ mV}}{\sqrt{3}} \approx 2.413\ \mu\text{V}$ |
| **Napätie bočníka (Offset)** | $V_{\text{OS}}$ | $0.0\ \mu\text{V}$ | Obdĺžnikové $\pm 10\ \mu\text{V}$ | $\frac{10\ \mu\text{V}}{\sqrt{3}} \approx 5.774\ \mu\text{V}$ |
| **Napätie bočníka (LSB)** | $V_{\text{shunt,quant}}$ | $2.5\ \mu\text{V}$ | Obdĺžnikové $\pm \frac{\text{LSB}}{2}$ | $\frac{2.5\ \mu\text{V}}{\sqrt{12}} \approx 0.722\ \mu\text{V}$ |
| **Kombinované $u(V_{\text{shunt}})$** | $u(V_{\text{shunt}})$ | $4.18\text{ mV}$ | Kvadratický súčet | $\sqrt{2.413^2 + 5.774^2 + 0.722^2} \approx 6.295\ \mu\text{V}$ |
| **Bočníkový rezistor (Tolerancia)** | $R_{\text{tol}}$ | $10\text{ m}\Omega$ | Obdĺžnikové $\pm 0.1\%$ | $\frac{0.001 \times 0.010}{\sqrt{3}} \approx 5.774\ \mu\Omega$ |
| **Bočníkový rezistor (TCR)** | $R_{\text{TCR}}$ | $25\text{ ppm}/^\circ\text{C}$ | $\Delta T = 20^\circ\text{C} \implies \pm 0.05\%$ | $\frac{0.0005 \times 0.010}{\sqrt{3}} \approx 2.887\ \mu\Omega$ |
| **Kombinované $u(R_{\text{shunt}})$** | $u(R_{\text{shunt}})$ | $10\text{ m}\Omega$ | Kvadratický súčet | $\sqrt{5.774^2 + 2.887^2} \approx 6.455\ \mu\Omega$ |
| **Časová základňa oscilátora** | $\Delta t$ | $208\text{ ms}$ | Stabilita $5\text{ ppm}$ | $\frac{5 \times 10^{-6} \times 0.208}{\sqrt{3}} \approx 0.60\text{ ns}$ |

---

## 3. Citlivostné Koeficienty a Zložená Neistota Výkonu $u_c(P)$

Parciálne derivácie funkcie výkonu:
$$c_{V_{\text{bus}}} = \frac{\partial P}{\partial V_{\text{bus}}} = \frac{V_{\text{shunt}}}{R_{\text{shunt}}} = I \approx 0.3483\text{ A}$$
$$c_{V_{\text{shunt}}} = \frac{\partial P}{\partial V_{\text{shunt}}} = \frac{V_{\text{bus}}}{R_{\text{shunt}}} = \frac{12.0}{0.010} = 1200.0\text{ A}$$
$$c_{R} = \frac{\partial P}{\partial R_{\text{shunt}}} = -\frac{V_{\text{bus}} \cdot V_{\text{shunt}}}{R_{\text{shunt}}^2} = -\frac{P}{R_{\text{shunt}}} \approx -418.0\text{ W}/\Omega$$

Zložená disperzia výkonu:
$$u_c^2(P) = \left(c_{V_{\text{bus}}} u(V_{\text{bus}})\right)^2 + \left(c_{V_{\text{shunt}}} u(V_{\text{shunt}})\right)^2 + \left(c_R u(R)\right)^2$$
Číselné príspevky pre $P_{\text{active}} = 4.18\text{ W}$:
- Vplyv $V_{\text{bus}}$: $(0.3483 \times 0.006938)^2 \approx 5.84 \times 10^{-6}\text{ W}^2$
- Vplyv $V_{\text{shunt}}$: $(1200.0 \times 6.295 \times 10^{-6})^2 \approx 5.71 \times 10^{-5}\text{ W}^2$
- Vplyv $R_{\text{shunt}}$: $(-418.0 \times 6.455 \times 10^{-6})^2 \approx 7.28 \times 10^{-6}\text{ W}^2$

Kombinovaná štandardná neistota jedného odčítania výkonu:
$$u_c(P) = \sqrt{5.84 \times 10^{-6} + 5.71 \times 10^{-5} + 7.28 \times 10^{-6}} \approx \mathbf{0.00838\text{ W}} \quad (0.20\%)$$

---

## 4. Propagácia do Dynamickej Energie a Štatistická Redukcia Vzoriek

Vďaka batch zosilneniu ($N_{\text{ops}} = 10^6$ operácií, $\Delta t_{\text{burst}} \ge 200\text{ ms}$) získa merací systém $N_{\text{active}} \ge 700$ nezávislých ADC konverzií a $N_{\text{idle}} \ge 350$ kľudových konverzií.

Neistota odhadu priemernej úrovne výkonu klesá s odmocninou počtu nezávislých konverzií:
$$u(\bar{P}_{\text{active}}) = \frac{u_c(P_{\text{active}})}{\sqrt{N_{\text{active}}}} = \frac{0.00838}{\sqrt{742}} \approx 0.000308\text{ W}$$
$$u(\bar{P}_{\text{idle}}) = \frac{u_c(P_{\text{idle}})}{\sqrt{N_{\text{idle}}}} = \frac{0.00652}{\sqrt{358}} \approx 0.000344\text{ W}$$

Neistota diferencie $\Delta P = \bar{P}_{\text{active}} - \bar{P}_{\text{idle}}$:
$$u(\Delta P) = \sqrt{u^2(\bar{P}_{\text{active}}) + u^2(\bar{P}_{\text{idle}})} \approx \sqrt{0.000308^2 + 0.000344^2} \approx \mathbf{0.000462\text{ W}}$$

Kombinovaná neistota dynamickej energie na validovanú operáciu:
$$u_c(E_{\text{op}}) = \frac{u(\Delta P) \cdot \Delta t_{\text{burst}}}{N_{\text{validated}}} = \frac{0.000462 \times 0.208}{10^6} \approx \mathbf{9.61 \times 10^{-11}\text{ J}} = \mathbf{0.000096\ \mu\text{J}}$$

Rozšírená neistota merania ($k = 2$, konfidenčná hladina 95%):
$$\boxed{U_{95\%}(E_{\text{op}}) = 2 \times u_c(E_{\text{op}}) \approx \mathbf{0.000212\ \mu\text{J}} \quad (0.212\text{ nJ})}$$

---

## 5. Metrologické Diskriminačné Pravidlo (Metrological Discrimination Rule)

Preregistrovaná tolerancia energetického rezídua v HTE-001 je:
$$\epsilon_{\text{energy}} = 0.050\ \mu\text{J} \quad (50.0\text{ nJ})$$

Pomer tolerančnej obálky k rozšírenej neistote merania:
$$\frac{\epsilon_{\text{energy}}}{U_{95\%}(E_{\text{op}})} = \frac{0.050\ \mu\text{J}}{0.000212\ \mu\text{J}} \approx \mathbf{236\times}$$

Metrologické pravidlo triedenia nameraného rezídua $R_{\text{energy}}$:
1. **Pásmo šumu merania (Sub-Noise Region):**
   $$|R_{\text{energy}}| \le U_{95\%} \; (0.212\text{ nJ})$$
   Odchýlka od modelu je experimentálne nerozlíšiteľná od metrologického šumu INA226. Model je plne v súlade.
2. **Pásmo fyzikálne rozlíšiteľnej zhody (Resolvable Compliant):**
   $$U_{95\%} < |R_{\text{energy}}| \le \epsilon_{\text{energy}} \; (50.0\text{ nJ})$$
   Odchýlka je reálna fyzikálna vlastnosť hardvéru (nie merací šum), ale stále spĺňa preregistrovanú tolerančnú obálku.
3. **Pásmo falzifikácie modelu (Envelope Breach):**
   $$|R_{\text{energy}}| > \epsilon_{\text{energy}} \; (50.0\text{ nJ})$$
   Model je nezvratne falzifikovaný. Chybu nemožno pripísať neistote merania, pretože tolerančné pásmo je $236\times$ širšie než neistota prístroja.

---

## 6. Záver pre Vedeckú Publikáciu

Výsledok tohto neistotového rozpočtu dokazuje, že experimentálny dizajn HTE-001 spĺňa najprísnejšie metrologické štandardy. Prípadné zlyhanie hypotézy $H_1$ na reálnom FPGA čipe nebude môcť byť spochybnené ako „artefakt nepresného lacného I2C senzora“.
