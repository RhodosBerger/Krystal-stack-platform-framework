# Hĺbkový Rozbor Predikčnej Logiky: Temporal Parallax & Mirroring
*Metodológia pre prediktívne plánovanie v prostredí Gamesa/KrystalStack*

**Dátum:** 14. Január 2026
**Modul:** Predictive Mirror Framework

---

## 1. Koncept: Časová Paralaxa (Temporal Parallax)
V bežnom renderingu je paralaxa posun v priestore. V našom systéme zavádzame **posun v čase**.
*   **Teória:** Ak poznáme vektor zmeny vstupu (napr. rýchlosť pohybu myši $\vec{v}$ a akceleráciu $\vec{a}$), vieme vypočítať polohu kamery v čase $t+1$.
*   **Aplikácia:** Kým GPU vykresľuje snímku $t_0$, naše TPU (alebo CPU vlákno) už počíta rovnicu pre $t_1$.

## 2. Predikčný Horizont a "Tieňová Pamäť"
Systém udržiava dva stavy pamäte:
1.  **Real State (L1):** To, čo užívateľ vidí teraz.
2.  **Shadow State (Shadow RAM):** To, čo systém *predpokladá*, že užívateľ uvidí o 50ms.

### 2.1 Matematika Predikcie (Zjednodušený Kalmanov Filter)
Na odhad budúceho stavu používame vážený priemer histórie a aktuálneho trendu:

$$ Future(t+1) = Current(t) + (\vec{v}_{avg} \times \Delta t) + (\vec{a}_{inst} \times \frac{\Delta t^2}{2}) $$ 

*   $\vec{v}_{avg}$: Priemerná rýchlosť za posledných 10 cyklov (hladkosť).
*   $\vec{a}_{inst}$: Okamžité zrýchlenie (reakcia na "trhnutie").

## 3. Heuristika Spotreby Zdrojo (Resource Futures)
Okrem polohy predikujeme aj **Spotrebu Wattov**.
*   Ak predikčný model vidí, že kamera smeruje do oblasti s vysokou hustotou rovníc (napr. komplexný fraktál), systém **vopred** zvýši napätie na jadrách (Pre-emptive Voltage Injection), aby eliminoval "Vdroop" (pokles napätia) pri náraze záťaže.

## 4. Vizualizácia: Kužeľ Neistoty (Cone of Uncertainty)
V rozhraní budeme vizualizovať predikciu ako kužeľ.
*   **Úzky kužeľ:** Systém si je istý (lineárny pohyb). "Hard Pre-fetch".
*   **Široký kužeľ:** Systém váha (chaotický pohyb). "Broad Pre-fetch" (načítanie viacerých možných vetiev).

---

*Tento dokument slúži ako základ pre `predictive_mirror_framework.py`.*
