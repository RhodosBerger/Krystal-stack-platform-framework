# Warhammer Mobility, Locational Algebra, Checkpoints & Crafting Economy Framework

## 1. Architektonický a Matematický Prehľad

Tento systém implementuje hlbokú syntézu pravidiel stolového wargamingu (inšpirovaného Warhammer 40,000 / Kill Team), trojrozmernej hexagonálnej lokačnej algebry pre výškovú výhodu a wardové pole, modulárneho craftingu s limitmi kombinácií afixov a podvojného účtovníctva, a strategických checkpoint vlajok s Line of Supply (LoS) overením.

```
+---------------------------------------------------------------------------------------------------+
|                                     KRYSTAL-STACK GAME ENGINE                                      |
+---------------------------------------------------------------------------------------------------+
|  1. Štatistický Model       | 2. Crafting Engine       | 3. Lokačná Algebra  | 4. Warhammer Pohyb |
|  - D6 / 2D6 Distribúcie     | - Base Templates         | - 3D Hex (q, r, h)  | - Normal Move (M)  |
|  - S vs T Wound Matica      | - Polarity Compatibility | - Zhora Bonus       | - Advance (+D6)    |
|  - Sv vs AP + Invulnerable  | - Affix Capacity Caps    | - Zospodu Penaliz.  | - Charge (2D6)     |
|  - Monte Carlo & Analytika  | - Double-Entry Ledger    | - Ward Bubble Field | - Fights First     |
+---------------------------------------------------------------------------------------------------+
|                               5. Checkpoint Vlajky & Line of Supply                               |
|   - Zone of Control (ZoC)   - BFS Sieťové Spojenie     - Body Víťazstva (VP) - Respawn Kotva      |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Štatistické Modely a Pravdepodobnosti

### 2.1 Základné D6 a 2D6 Distribúcie
Pravdepodobnosť hodu na D6 väčšieho alebo rovného požadovanej hodnote $X \in \{1, \dots, 6\}$:
$$P(\text{D6} \ge X) = \frac{\max(0, \min(6, 7 - X))}{6}$$

Pre hod 2D6 (využívaný najmä pri Charge sekvenciách a Morálke):
$$P(\text{2D6} = k) = \frac{6 - |k - 7|}{36}, \quad k \in \{2, \dots, 12\}$$
$$P(\text{2D6} \ge X) = \sum_{k=X}^{12} P(\text{2D6} = k)$$

### 2.2 Warhammer Sila ($S$) vs Odolnosť ($T$) Wound Matica
Po úspešnom zásahu (Weapon/Ballistic Skill) sa porovnáva Sila útočníka so Odolnosťou obrancu:
* Ak $S \ge 2T \implies$ zranenie na **2+** ($P = \frac{5}{6}$)
* Ak $S > T \implies$ zranenie na **3+** ($P = \frac{4}{6}$)
* Ak $S = T \implies$ zranenie na **4+** ($P = \frac{3}{6}$)
* Ak $S < T \implies$ zranenie na **5+** ($P = \frac{2}{6}$)
* Ak $S \le \lfloor T/2 \rfloor \implies$ zranenie na **6+** ($P = \frac{1}{6}$)

### 2.3 Brnenie ($Sv$), Armor Penetration ($AP$) a Nezraniteľný Štít ($Invuln$)
Efektívny hod na záchranu (Armor Save):
$$\text{Modified Save} = Sv - AP$$
Ak má obranca nezraniteľný štít ($Invuln$), vyberá sa vždy priaznivejšia hodnota:
$$\text{Effective Threshold} = \min(Sv - AP, Invuln)$$
Pravdepodobnosť, že záchranný hod **zlyhá** (cieľ dostane poškodenie):
$$P(\text{Fail Save}) = 1.0 - P(\text{D6} \ge \text{Effective Threshold})$$

### 2.4 Analytická Očakávaná Hodnota $E[D]$ a Rozptyl
Konverzný koeficient jedného útoku:
$$p_{\text{conv}} = P(\text{Hit}) \cdot P(\text{Wound}) \cdot P(\text{Fail Save})$$
Pre počet útokov $A$ a poškodenie $D$ na nepreukázaný záchranný hod:
$$E[D] = A \cdot p_{\text{conv}} \cdot D$$
$$\operatorname{Var}(D) = A \cdot p_{\text{conv}} \cdot (1 - p_{\text{conv}}) \cdot D^2$$
$$\sigma(D) = \sqrt{\operatorname{Var}(D)}$$

---

## 3. Herná Ekonomika a Crafting Systém s Limitmi

### 3.1 Base Templates
1. **Kryštálová Čepeľ (`crystal_blade`)**: Slot `weapon_main`, +3 Útok, +1 AP, Base Power 5. Vyžaduje 3 Manu, 2 Kryštály.
2. **Čadičová Pavéza (`basalt_shield`)**: Slot `armor_chest`, +1 Sv, +1 Toughness, 5+ Invulnerable Save, Base Power 4. Vyžaduje 2 Manu, 1 Kryštál.
3. **Toxická Kadidelnica (`toxic_censer`)**: Slot `weapon_offhand`, +2 Sila, Toxic Miasma, Base Power 4. Vyžaduje 2 Manu, 2 Slizy.
4. **Druidská Palica (`druid_staff`)**: Slot `weapon_twohand`, +1 Range, +2 Mana regen, Base Power 5. Vyžaduje 3 Manu, 2 Jantáry.
5. **Aéterový Ward žiarič (`aether_ward_emitter`)**: Slot `accessory_relic`, +6 Ward kapacita, +1 Ward regen, Base Power 6. Vyžaduje 4 Manu, 3 Kryštály.

### 3.2 Limity a Pravidlá Kombinácií
* **Kapacitný limit afixov podľa vzácnosti**:
  - `Common`: 0 afixov (čistá šablóna)
  - `Magic`: max 2 afixy (max 1 prefix, max 1 suffix)
  - `Rare`: max 3 afixy (max 2 prefixy, max 2 suffixy)
  - `Epic`: max 4 afixy (max 2 prefixy, max 2 suffixy)
  - `Legendary`: max 5 afixov (max 3 prefixy, max 3 suffixy)
* **Pravidlo Elementárnej Polarity (Polarity Collision Rule)**:
  - Kryštálová esencia (`crystal`) a Toxická kyselina (`toxic`) sú vzájomne nestabilné.
  - Pokus o skombinovanie kryštálového a toxického afixu (alebo šablóny a afixu) bez stabilizačného katalyzátora (`amber_catalyst`) spôsobí zlyhanie craftingu a vyvolá `CraftingInstabilityError`.
* **Power Budget Cap**:
  - Každá vzácnosť má striktný strop na celkový súčet atribútov, čo zabraňuje creepu sily (Common: 6, Magic: 12, Rare: 18, Epic: 24, Legendary: 32).
* **Integrácia Podvojného Účtovníctva (Double-Entry Ledger)**:
  - Suroviny sú odpočítavané v atomickej transakcii s debetom zdrojov hráča a kreditom hodnoty výbavy.
  - Nedostatok surovín vyvolá okamžité odmietnutie požiadavky s HTTP 400 bez narušenia rovnováhy.

---

## 4. Vlajky ako Checkpointy a Zásobovacia Línia (Line of Supply)

### 4.1 Zone of Control (ZoC) a Obsadzovanie
* Vlajky sú umiestnené v strategických hexoch (`Nexus: [0, 0]`, `North: [0, -1]`, `South: [0, 1]`).
* Každá vlajka má rádius kontroly ($r = 1$ hex).
* Kontrola sa vyhodnocuje porovnaním prítomnosti jednotiek v dosahu:
  - Ak je v dosahu iba hráč $\implies$ progres obsadzovania $+25\%$ pre hráča.
  - Ak sú v dosahu obe strany $\implies$ checkpoint je v stave **Contested** (sporný), zisk bodov je pozastavený.

### 4.2 Overenie Zásobovacej Línie (Line of Supply - LoS)
* Pomocou algoritmu Breadth-First Search (BFS) sa overuje, či medzi kontrolovanou vlajkou a domovskou základňou kmeňa existuje neprerušená cesta priateľských alebo neutrálnych hexov.
* Odrezaná vlajka (obkľúčená nepriateľom) stráca schopnosť generovať Victory Points (VP) a nemôže slúžiť ako bod pre oživenie (respawn anchor).

---

## 5. Veže s Lokačnou Algebrou a Wardová Bublina

### 5.1 Trojrozmerná Hexagonálna Algebra
Každý uzol má súradnice:
$$\mathbf{P} = (q, r, h)$$
kde $(q, r)$ sú axiálne súradnice hexu a $h$ je nadmorská výška v metroch. Euklidovská vzdialenosť v 3D priestore:
$$d_{2D} = \text{HexDistance}((q_1, r_1), (q_2, r_2))$$
$$d_{3D} = \sqrt{(d_{2D} \cdot 1.732)^2 + (\Delta h)^2}$$
kde $\Delta h = h_{\text{attacker}} - h_{\text{defender}}$.

### 5.2 Zhora Bonus (Elevated High Ground)
Pri streľbe z výšky ($\Delta h \ge 1.5\text{ m}$):
* **Dosah**: $+\min(3, \lfloor \Delta h \rfloor)$ hexov
* **Presnosť (Hit modifier)**: $+1$ k hodu na zásah (napr. z 3+ na 2+)
* **Prieraznosť (AP modifier)**: $+1$ (napr. z AP-1 na AP-2)
* **Poškodenie**: $\times 1.5$ ($+50\%$ multiplikátor)

### 5.3 Zospodu na Vežu Útok (Low Ground Assault Penalty)
Pri útoku zdola nahor na vežu ($\Delta h \le -1.5\text{ m}$):
* **Dosah**: $-\min(2, \lfloor |\Delta h| \rfloor)$ hexov
* **Presnosť (Hit modifier)**: $-1$ k hodu na zásah
* **Krytie veže (Tower Cover)**: Obranca na veži získava $+2$ k Armor Save
* **Poškodenie**: $\times 0.75$ ($-25\%$ penalizácia z dôvodu nepriaznivého balistického uhla)

### 5.4 Wardové Pole Veže (Tower Ward Bubble)
* Veža generuje aéterické silové pole s vlastným bazénom energie (základ: 6 bodov).
* Prichádzajúce poškodenie najskôr pohlcuje Ward. Až po vyčerpaní Wardu prechádza zostávajúce poškodenie na telo hrdinu či konštrukcie.
* Ward pasívne regeneruje $+2$ body za kolo, pokiaľ nebol v predchádzajúcom kole úplne preťažený.

---

## 6. Warhammer Pravidlá Pohyblivosti a Spôsoby Pohybu

Systém rozlišuje 4 základné módy pohybu prevzaté z pravidiel wargamingu:

| Mód Pohybu | Max Vzdialenosť | Streľba / Schopnosti | Útok (Charge) | Popis a Špecifiká |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Move** | $M$ (základná hodnota) | Áno (normálne) | Áno | Štandardný taktický manéver |
| **Advance** | $M + \text{D6}$ bonus | Iba so zbraňou typu *Assault* | **Nie** | Rýchly šprint; obetuje možnosť útoku za mobilitu |
| **Charge** | Test 2D6 ($2\text{D6} \ge 2 \times d$) | Počas pohybu nie | **Fights First** | Zápasový nájazd do melee zóny; pri úspechu bojuje ako prvý |
| **Fall Back** | $M$ | Nie | Nie | Ústup z tesného kontaktu (engagement range) |

* **Terrain Penalties**:
  - Priechodný terén: Náklad 1.0 MP
  - Ťažký terén (bahno, suť): Náklad 1.5 MP
  - Nebezpečný terén (kyselina, kryštálové ostne): Náklad 2.0 MP + test nebezpečenstva (hod 1 na D6 spôsobí 1 Mortal Wound)
* **Keyword FLYING**:
  - Jednotky so schopnosťou lietania ignorujú penalizácie pozemného terénu a vertikálne výškové rozdiely pri prechode prekážok.
