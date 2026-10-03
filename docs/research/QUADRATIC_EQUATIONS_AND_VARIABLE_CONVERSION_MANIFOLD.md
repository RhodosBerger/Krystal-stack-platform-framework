# Krystal-Stack Research: Kvadratické Rovnice & Prevod Premenných cez Latentné $x$

**Dátum:** 3. Október 2026  
**Autor:** Krystal-Stack Architectural & Mathematical Working Group  
**Kľúčový Invariant:** `VITAL_MAX_HP = 6` (Pravidlo maximálneho zdravia a celistvosti systému)  
**Zlatý Pomer:** $\phi = 1.61803398875$, $\phi^{-1} \approx 0.61803398875$ (Aristotelovská rovnováha)

---

## 1. Úvod a Motivácia

V heterogénnom distribuovanom výpočtovom prostredí ako **Krystal-Stack** (zahŕňajúcom Vulkan GPU compute, Intel Iris Xe 96 EUs, NPU DirectML, pamäťové zbernice L1/L2 SRAM, Host DDR VRAM, NVMe swap, ekonomický AMM engine a hernú logiku Poslední Kmen) operujú jednotlivé subsystémy v **nekompatibilných fyzikálnych a virtuálnych jednotkách**:

- **Pamäťový subsystém:** latencia v nanosekundách ($ns$), alokácia pamäte v megabajtoch ($MB$).
- **Grafický & Výpočtový engine:** taktovacia frekvencia v megahertzoch ($MHz$), priepustnosť v $GB/s$.
- **Termodynamika & Fázový priestor:** celková energia Hamiltoniánu $H \in [0.05, 1.20]$, informačná entropia $S$ v bitoch.
- **Ekonomický AMM engine:** trhová cena komodít v kreditoch, bonding curves.
- **Herná a vitálna logika:** životná energia $HP \in [0, 6]$ pod striktným invariantom $VITAL\_MAX\_HP = 6$.
- **Procedurálna geometria:** nadmorská výška $Z$ v metroch, paraboloidy a alchymistické SDF polia.

Priame lineárne prevody medzi týmito doménami zlyhávajú, pretože nereflektujú **nelineárny nárast zaťaženia**, saturačné odpory zberníc, tepelné obmedzenia (thermal throttling) a akceleráciu cien.

Riešením je **Kvadratický Manifold**, v ktorom každá doména komunikuje cez normalizovanú centrálnu premennú **$x \in [0.0, 1.0]$** — ktorá predstavuje *predmet skúmania* (latentný stav systému).

---

## 2. Matematická Formulácia Kvadratického Prevodníka

### 2.1 Priame Mapovanie (Forward Projection)

Každá platformová doména $i$ je popísaná parabolickou funkciou závislou na latentnom stave $x$:

$$V_i(x) = a_i x^2 + b_i x + c_i$$

Kde:
- **$a_i$ (Kvadratický koeficient):** vyjadruje zakrivenie, nelineárny odpor, zrýchlenie alebo tepelné škrtenie.
  - Ak $a_i > 0$: konvexná krivka (napr. latencia RAM rastie kvadraticky pri zapĺňaní bufferov).
  - Ak $a_i < 0$: konkávna krivka (napr. takt GPU klesá pri vysokom zaťažení z dôvodu TDP limitu; životy $HP$ ubúdajú zrýchlene pri kritickom zranení).
- **$b_i$ (Lineárny koeficient):** základná miera zmeny (drift, lineárny sklon).
- **$c_i$ (Základný posun / Offset):** hodnota veličiny v pokojovom stave ($x = 0$), napr. základná latencia L1 cache ($5.2\text{ ns}$) alebo plné zdravie ($6.0\text{ HP}$).

---

### 2.2 Inverzná Transformácia (Extrakcia Latentného $x$ cez Kvadratickú Rovnicu)

Ak pozorujeme konkrétnu hodnotu veličiny $V_i$ v ľubovoľnom subsystéme (napr. zbernica vykazuje latenciu $V_{mem} = 45\text{ ns}$), extrakcia latentného parametra $x$ prebieha vyriešením kvadratickej rovnice:

$$a_i x^2 + b_i x + (c_i - V_i) = 0$$

Definujeme efektívny absolútny člen $c'_{i} = c_i - V_i$. Diskriminant rovnice je:

$$\Delta_i = b_i^2 - 4 a_i (c_i - V_i)$$

#### Analýza Koreňov a Fyzikálny Význam:

1. **Reálny režim ($\Delta_i \ge 0$):**
   $$x_{1, 2} = \frac{-b_i \pm \sqrt{\Delta_i}}{2 a_i}$$
   Vyberá sa koreň ležiaci vo fyzikálne platnom intervale $x \in [0.0, 1.0]$.
2. **Kritický / Limitný režim ($\Delta_i < 0$):**
   Systém sa dostal mimo bežnú prevádzkovú obálku. Výpočet sa premieta na vrchol paraboly (apex):
   $$x_{apex} = -\frac{b_i}{2 a_i}$$
   Komplexná zložka reprezentuje fázový posun alebo reaktívnu záťaž zbernice.
3. **Degenerovaný lineárny prípad ($a_i = 0$):**
   $$x = \frac{V_i - c_i}{b_i}$$

---

### 2.3 Krížový Prevod Medzi Ľubovoľnými Doménami ($A \to x \to B$)

Prevod medzi zdrojovou doménou $A$ a cieľovou doménou $B$ prebieha cez dvojkrokový mostík:

```
[ Pozorovaná Veličina V_A ]
            │
            ▼ (Kvadratický riešiteľ: a_A·x² + b_A·x + c'_A = 0)
    [ Latentné x ]  <─── "Predmet Skúmania" (Normalizovaný stav)
            │
            ▼ (Priame vyhodnotenie: V_B = a_B·x² + b_B·x + c_B)
[ Cieľová Veličina V_B ]
```

Týmto mechanizmom je možné okamžite prepočítať napr.:
- **Pamäťovú latenciu $45\text{ ns}$** $\to x \approx 0.601 \to$ **Alokáciu VRAM $\approx 4296\text{ MB}$** $\to$ **Zostávajúce HP $= 3.84\text{ HP}$**.

---

## 3. Katalóg Kanonických Kvadratických Domén v Krystal-Stack

| Doména ID | Názov Veličiny | Jednotka | Koeficienty $(a, b, c)$ | Prevádzkový Rozsah | Fyzikálna Interpretácia |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `memory_latency_ns` | Latencia L1/L2 & DDR | $\text{ns}$ | $(85.0, 15.0, 5.2)$ | $[5.0, 120.0]$ | Kvadratický nárast latencie pri zahltení zbernice |
| `vram_allocation_mb`| Zdieľaná Host DDR VRAM | $\text{MB}$ | $(3200.0, 4800.0, 256.0)$ | $[256.0, 8256.0]$ | Alokácia textúr a geometrie pre Vulkan Iris Xe |
| `gpu_clock_mhz`     | Takt Intel Iris Xe EUs | $\text{MHz}$ | $(-400.0, 1050.0, 800.0)$| $[800.0, 1450.0]$ | Tepelné škrtenie (TDP limit) pri maximálnom zaťažení |
| `hamiltonian_h`     | Celková Energia Systému| $\text{energy}$ | $(0.85, 0.20, 0.05)$ | $[0.05, 1.20]$ | Harmonický oscilátor ($\frac{1}{2} k x^2$) fázového priestoru |
| `market_price_credits`| AMM Bonding Curve | $\text{cr}$ | $(120.0, 30.0, 10.0)$ | $[10.0, 160.0]$ | Kvadratické ocenenie výpočtových kvót bez hyperinflácie |
| `vital_hp`          | Životná Energia Kmeňa  | $\text{HP}$ | $(-3.5, -1.5, 6.0)$ | $[0.0, 6.0]$ | **Platformový Invariant: 6 Max HP** |
| `terrain_altitude_m`| Výška Terénu (SDF)     | $\text{m}$ | $(1200.0, 600.0, 50.0)$ | $[50.0, 1850.0]$ | Kvadratický paraboloid pre procedurálne pohoria |
| `system_entropy_s`  | Informačná Entropia    | $\text{bits}$ | $(0.45, 0.55, 0.02)$ | $[0.02, 1.02]$ | Miera neurčitosti a fragmentácie pamäte |

---

## 4. Zlatá Stredná Cesta (Aristotelovská Rovnováha pri $x = \phi^{-1}$)

Keď je latentný parameter nastavený na **Aristotelovskú Zlatú strednú cestu** ($x = \phi^{-1} \approx 0.61803398875$):
- **Pamäťová latencia:** $\approx 46.9\text{ ns}$ (stabilná prevádzka bez kolízií).
- **Zdieľaná VRAM:** $\approx 4446\text{ MB}$ (vyvážené vyrovnávacie pamäte).
- **Takt GPU Iris Xe:** $\approx 1296\text{ MHz}$ (vysoký výkon bez tepelného throttlingu).
- **Hamiltonián $H$:** $\approx 0.498$ (optimálna energetická entropia).
- **Životná energia:** $HP = 3.73\text{ HP}$ (stabilný vitálny stav pod limitom $6\text{ HP}$).

---

## 5. Implementácia a Tripartite Parita

1. **Python Jadro:** [quadratic_variable_transformer.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/economic_engine/quadratic_variable_transformer.py)
   - Dataclass `QuadraticDomain`, engine `QuadraticVariableTransformer`, singleton `GLOBAL_QUADRATIC_TRANSFORMER`.
2. **Janet DSL Modul:** [quadratic_variable_transformer.janet](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/quadratic_variable_transformer.janet)
   - Funkcie `evaluate-quadratic-v`, `solve-quadratic-x`, `convert-variable-via-x`.
3. **REST API Endpointy:** [krystal_engine_core.py](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/krystal_engine_core.py)
   - `GET /api/quadratic/domains`
   - `GET /api/quadratic/golden-state`
   - `POST /api/quadratic/solve-x`
   - `POST /api/quadratic/convert`
   - `POST /api/quadratic/convert-all`
4. **Interaktívne Štúdio:** [quadratic_transformer_studio.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/quadratic_transformer_studio.html)
   - Dve živé plátna (Parabola s diskriminantom a orbitálny flux premenných), interaktívna sonda a priama prevodná rúra.
5. **Krystal WebOS Integrácia:** [krystal_webos_desktop.html](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/static/krystal_webos_desktop.html)
   - Možnosť otvoriť kvadratický prevodník v plávajúcom okne priamo z plochy alebo štart menu.
