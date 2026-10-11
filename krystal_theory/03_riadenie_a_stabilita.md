# 03 — Riadenie a stabilita

*Modeluje:* prahové prepínanie v `classify` (bez pamäte) a jeho spätnú väzbu na teplotu čipu.
*Posledná kontrola voči kódu:* 2026-10-09 — **kód zatiaľ hysteréziu nemá.**

Triáž je z pohľadu teórie riadenia **spínaný regulátor** (relé): stav systému
$\theta$ určuje režim $q \in \{0, 1\}$ (1 = REG) a režim spätne ovplyvňuje príkon, a tým $\theta$.

---

## 3.1 Prepínanie pri prahu bez pamäte (chattering)

**AXIÓM A3.1.** Vzorky termálneho tlaku $\theta_k$ sú nezávislé a rovnako rozdelené,
$p = \Pr[\theta_k \ge \theta^\*]$.

**DEFINÍCIA 3.1.** Bezpamäťový režim: $q_k = \mathbf{1}[\theta_k \ge \theta^\*]$ (aktuálny kód).

### VETA 3.1 (Frekvencia prepínania)
Za A3.1 je pravdepodobnosť zmeny režimu medzi susednými vzorkami $2p(1-p)$, s maximom
$1/2$ pri $p = 1/2$.

*Dôkaz.* $q_k$ a $q_{k+1}$ sú nezávislé Bernoulliho premenné s parametrom $p$.
$\Pr[q_k \ne q_{k+1}] = p(1-p) + (1-p)p = 2p(1-p)$. Funkcia $2p(1-p)$ je konkávna kvadratická
funkcia s vrcholom v $p = 1/2$ a hodnotou $1/2$. $\blacksquare$

**Dôsledok.** Ak sa priemerná teplota usadí pri prahu (čo je pri reléovej spätnej väzbe
typické, pozri 3.3), každá druhá vzorka v priemere prepne režim. Každé prepnutie mení
pamäťovú politiku (`LEGACY_SIMD_BUFFER` ↔ iné) a hĺbku špekulácie 0 ↔ 4.
A3.1 je silný predpoklad (reálny šum je autokorelovaný); hypotéza H-R1 v
[04](04_hypotezy_a_protokoly.md) ho testuje.

---

## 3.2 Hysterézia

**DEFINÍCIA 3.2 (Schmittov spúšťač).** Pre šírku pásma $h \ge 0$:

$$
q_{k+1} =
\begin{cases}
1 & \text{ak } \theta_{k+1} \ge \theta^\* \\
0 & \text{ak } \theta_{k+1} < \theta^\* - h \\
q_k & \text{inak.}
\end{cases}
$$

Aktuálny kód zodpovedá $h = 0$.

**AXIÓM A3.2 (ohraničený šum).** $\theta_k = \bar\theta + n_k$, kde $\bar\theta$ je konštanta a $|n_k| \le \delta$.

### VETA 3.2 (Postačujúca šírka hysterézie)
Za A3.2 a pri $h \ge 2\delta$ nastane za celý časový horizont **najviac jedna** zmena režimu.

*Dôkaz.* Vstup do REG vyžaduje vzorku $\theta_k \ge \theta^\*$, teda $\bar\theta + \delta \ge \theta^\*$, t. j.
$\bar\theta \ge \theta^\* - \delta$. Výstup z REG vyžaduje vzorku $\theta_j < \theta^\* - h$, teda
$\bar\theta - \delta < \theta^\* - h$, t. j. $\bar\theta < \theta^\* - h + \delta$. Aby nastali dve zmeny (vstup aj výstup,
v ľubovoľnom poradí), musia platiť obe: $\theta^\* - \delta \le \bar\theta < \theta^\* - h + \delta$, čo je
neprázdne len ak $h < 2\delta$. Pri $h \ge 2\delta$ je teda možná najviac jedna zmena. $\blacksquare$

### DÔSLEDOK 3.3 ($h = 0$ nič negarantuje)
Pri $h = 0$ a $\delta > 0$ je interval $[\theta^\* - \delta,\ \theta^\* + \delta)$ neprázdny a pre každé $\bar\theta$ v ňom
sú prípustné postupnosti šumu, pri ktorých režim prepína v každom kroku.

*Dôkaz.* Pre $\bar\theta$ v intervale existujú prípustné $n^+, n^-$ s $\bar\theta + n^+ \ge \theta^\*$ a $\bar\theta + n^- < \theta^\*$;
striedanie $n^+, n^-, n^+, \dots$ prepína v každom kroku. $\blacksquare$

→ Inžiniersky cieľ **G1** (hysterézia so šírkou $h \ge 2\hat\delta$, kde $\hat\delta$ je namerané).

---

## 3.3 Termálny reléový cyklus

**AXIÓM A3.3 (tepelný model 1. rádu).**
$C\,\dot T = P(q) - \dfrac{T - T_{\mathrm{amb}}}{R_{\mathrm{th}}}$, s $\tau = R_{\mathrm{th}} C$ a príkonom
$P(1) = P_L$ (REG), $P(0) = P_H$, $P_L < P_H$. Ustálené teploty:
$T_L = T_{\mathrm{amb}} + R_{\mathrm{th}} P_L$, $T_H = T_{\mathrm{amb}} + R_{\mathrm{th}} P_H$.
Režim sa riadi definíciou 3.2 bez šumu. Teploty sú v rovnakých jednotkách ako $\theta^\*$.

**LEMA 3.4.** Pri konštantnom režime je $T(t) = T_\infty + (T_0 - T_\infty) e^{-t/\tau}$ a čas prechodu
z $T_0$ do $T_1$ (oba na tej istej strane $T_\infty$, $T_1$ bližšie k $T_\infty$) je
$t = \tau \ln \dfrac{T_0 - T_\infty}{T_1 - T_\infty}$.
*(Štandardné riešenie lineárnej ODR 1. rádu.)*

### VETA 3.5 (Absorpcia v núdzovom režime)
Po vstupe do REG pri $T = \theta^\*$ systém režim REG **nikdy neopustí** práve vtedy, keď $T_L \ge \theta^\* - h$.

*Dôkaz.* V REG platí $T(t) \to T_L$ monotónne. (⇐) Ak $T_L \ge \theta^\* - h$: pri $T_L < \theta^\*$ teplota klesá
k $T_L$, ale $T(t) > T_L \ge \theta^\* - h$ pre všetky konečné $t$; pri $T_L \ge \theta^\*$ neklesá vôbec. Podmienka
výstupu $T < \theta^\* - h$ teda nikdy nenastane. (⇒) Ak $T_L < \theta^\* - h$, teplota podľa lemy 3.4
dosiahne $\theta^\* - h$ za konečný čas $t_c$ a režim sa prepne. $\blacksquare$

**Interpretácia.** Ak núdzová politika neznižuje príkon dostatočne, systém uviazne v REG
(trvalý „legacy“ režim s nulovou špekuláciou). Ide o podmienku na *hardvér a chladenie*, nie na kód.

### VETA 3.6 (Limitný cyklus a jeho perióda)
Ak $T_L < \theta^\* - h$ a $T_H > \theta^\*$, systém konverguje k periodickému cyklu medzi $\theta^\* - h$ a $\theta^\*$ s
časom chladenia a ohrevu

$$
t_c = \tau \ln\frac{\theta^\* - T_L}{\theta^\* - h - T_L}, \qquad
t_h = \tau \ln\frac{T_H - \theta^\* + h}{T_H - \theta^\*},
$$

periódou $t_c + t_h$ a podielom času v REG $\;D = t_c / (t_c + t_h)$.

*Dôkaz.* Chladenie v REG z $\theta^\*$ do $\theta^\* - h$ k asymptote $T_L < \theta^\* - h$: lema 3.4 s $T_0 = \theta^\*$, $T_1 = \theta^\* - h$,
$T_\infty = T_L$. Ohrev mimo REG z $\theta^\* - h$ do $\theta^\*$ k asymptote $T_H > \theta^\*$: lema 3.4 s $T_0 = \theta^\* - h$,
$T_1 = \theta^\*$, $T_\infty = T_H$ (zlomok sa prepíše vynásobením čitateľa aj menovateľa $-1$). Obe hodnoty
sú konečné a kladné, keďže argumenty logaritmov sú $> 1$. Po prvom dosiahnutí $\theta^\*$ sa postupnosť
opakuje, cyklus je teda periodický. $\blacksquare$

**Poznámka.** Pri $h \to 0^+$ platí $t_c, t_h \to 0$: perióda klesá k nule a frekvencia prepínania
rastie nad všetky medze. To je spojitá obdoba vety 3.1 a ďalší argument pre $h > 0$. V diskrétnom čase
je frekvencia zhora ohraničená vzorkovacou frekvenciou.

---

## 3.4 Čo táto kapitola nehovorí

- Nehovorí nič o reálnych hodnotách $R_{\mathrm{th}}, \tau, P_L, P_H$ pre i5-1135G7: **nie sú namerané.**
- Neuvažuje DVFS ani PL1/PL2 regulátor firmvéru, ktorý pracuje paralelne s touto triážou
  (dva regulátory na jednej veličine môžu interagovať — pozri [P4](OTVORENE_PROBLEMY.md#p4-interakcia-s-firmvérovým-regulátorom)).
- `thermal_pressure_pct` je v kóde vstup; triáž príkon priamo neriadi. A3.3 predpokladá,
  že zmena politiky zmení príkon. To je hypotéza H-R2.
