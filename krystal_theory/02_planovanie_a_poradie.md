# 02 — Plánovanie a poradie

*Modeluje:* `schedule_ui_processes` (stabilné triedenie podľa $r$) a tabuľku `TRIAGE_POLICY`.
*Posledná kontrola voči kódu:* 2026-10-09.

---

## 2.1 Striktné poradie

**DEFINÍCIA 2.1 (Porušenie urgentného poradia).** Pre zoznam úloh $L = (a_1,\dots,a_n)$ a jeho
zoradenie $S(L)$ nastane porušenie, ak v $S(L)$ stojí úloha s $T(x_a) \ne \mathrm{REG}$
pred úlohou s $T(x_b) = \mathrm{REG}$.

### VETA 2.1 (Injektívny rank vylučuje porušenie)
Ak je $r$ injektívna a $r(\mathrm{REG}) = \min_{m\in M} r(m)$, potom pre každý vstupný zoznam $L$
triedenie podľa kľúča $r \circ T$ neporuší urgentné poradie.

*Dôkaz.* Pre $m \ne \mathrm{REG}$ z injektívnosti a minimality vyplýva $r(m) > r(\mathrm{REG})$.
Zoradený výstup je neklesajúci v kľúči, takže každá úloha s kľúčom $r(\mathrm{REG})$ predchádza
každej úlohe s väčším kľúčom. $\blacksquare$

### VETA 2.2 (Pôvodná politika poradie porušovala)
Pri pôvodnej politike $r(\mathrm{REG}) = r(\mathrm{PROJ}) = 1$ existuje zoznam s porušením.

*Dôkaz (protipríklad).* $L = (a, b)$, kde $x_a = (0{,}1;\ 1;\ 50) \mapsto \mathrm{PROJ}$ a
$x_b = (0{,}1;\ 1;\ 99) \mapsto \mathrm{REG}$. Kľúče sú rovnaké (1, 1). Pythonovské `list.sort`
je stabilné, takže zachová vstupné poradie: $S(L) = (a, b)$, PROJ pred REG. $\blacksquare$

*Zamknuté testom:* `test_h1_emergency_sorted_first_regardless_of_input_order`.

**Poznámka k voľbe opravy.** Nový rank (REG 1, PROJ 2, COMP 3, SUB 4) je jediné injektívne
priradenie hodnôt z $\{1,2,3,4\}$, ktoré (i) dáva REG minimum a (ii) zachováva **všetky** striktné nerovnosti
pôvodnej politiky (PROJ < COMP < SUB). Overenie: pôvodné striktné páry boli
PROJ<COMP, PROJ<SUB, COMP<SUB, REG<COMP, REG<SUB; jediná rovnosť bola REG=PROJ.
Rozbiť ju v prospech REG si vynucuje (i), takže poradie je jednoznačné.

---

## 2.2 (Ne)monotónnosť priority

Prirodzené očakávanie je: *väčší stres ⇒ aspoň rovnaká urgentnosť*, t. j. $r \circ T$
nerastie v každej premennej. Táto sekcia ukazuje, že to platí len pre $\theta$.

### VETA 2.3 (Monotónnosť v teplote)
Pre pevné konečné $e, c$ je funkcia $\theta \mapsto r(T(e,c,\theta))$ nerastúca.

*Dôkaz.* Pre $\theta < \theta^\*$ je hodnota konštantná $k = r(T(e,c,\theta)) \in \{2,3,4\}$ (nezávisí od $\theta$).
Pre $\theta \ge \theta^\*$ je hodnota 1. Keďže $k \ge 2 > 1$, funkcia je schodovitá a nerastúca. $\blacksquare$

### VETA 2.4 (Nemonotónnosť v kontencii a entropii)
Pre $\theta < \theta^\*$:
1. ak $e < e^\*$, potom zvýšenie $c$ cez $c^\*$ **zvýši** rank z 2 (PROJ) na 3 (COMP);
2. ak $c < c^\*$, potom zvýšenie $e$ cez $e^\*$ **zvýši** rank z 2 (PROJ) na 4 (SUB).

*Dôkaz.* Priamo z definície 1.1 a tabuľky 1.2. $\blacksquare$

**Interpretácia.** Vyšší rank znamená nižšiu urgentnosť. Úloha, ktorej pribudne kontencia
pamäte alebo entropia, sa teda **posunie dozadu.** Je to konzistentné s politikou
*odhadzovania záťaže* (load shedding: drahé alebo konfliktné úlohy čakajú), ale nie s politikou
*urgentnosti*. Kód ani dokumentácia nehovorí, ktorá z nich je zamýšľaná. Pozri
[OTVORENE_PROBLEMY.md, P2](OTVORENE_PROBLEMY.md#p2-sémantika-ranku).

---

## 2.3 Smithovo pravidlo a implikované váhy

Aby rank nebol ľubovoľný, dá sa odvodiť z cieľovej funkcie. Klasický výsledok:

**VETA 2.5 (Smith, 1956; WSPT).** Pre jeden stroj bez preempcie a úlohy so spracovacími časmi
$p_j > 0$ a váhami $w_j > 0$ minimalizuje $\sum_j w_j C_j$ ($C_j$ = čas dokončenia) práve každé
poradie, ktoré je neklesajúce v $p_j / w_j$.
*(Prevzatá veta; dôkaz výmenným argumentom: výmena susedných úloh $i, j$ zmení cieľ o
$w_j p_i - w_i p_j$, čo je nezáporné práve vtedy, keď $p_i/w_i \le p_j/w_j$.)*

**AXIÓM A2.1 (modelový predpoklad).** Spracovací čas úlohy s mechanizmom $m$ je $p = \pi_m$
a váha závisí len od mechanizmu, $w = w_m$.

### VETA 2.6 (Implikované váhy aktuálnej politiky)
Za axiómu A2.1 je aktuálne poradie REG < PROJ < COMP < SUB optimálne podľa WSPT
**pre každú dávku, ktorá obsahuje všetky štyri triedy, práve vtedy**, keď (pri normalizácii $w_{\mathrm{PROJ}} = 1$):

$$
w_{\mathrm{REG}} \ge 0{,}1, \qquad w_{\mathrm{COMP}} \le 0{,}24, \qquad w_{\mathrm{SUB}} \le \tfrac{35}{12}\, w_{\mathrm{COMP}} \;(\le 0{,}7).
$$

*Dôkaz.* WSPT vyžaduje
$\frac{50}{w_{\mathrm{REG}}} \le \frac{500}{1} \le \frac{120}{w_{\mathrm{COMP}}} \le \frac{350}{w_{\mathrm{SUB}}}$.
Prvá nerovnosť ⇔ $w_{\mathrm{REG}} \ge 0{,}1$. Druhá ⇔ $w_{\mathrm{COMP}} \le 120/500 = 0{,}24$.
Tretia ⇔ $w_{\mathrm{SUB}} \le \frac{350}{120} w_{\mathrm{COMP}} = \frac{35}{12} w_{\mathrm{COMP}}$, a s druhou
$w_{\mathrm{SUB}} \le \frac{35}{12}\cdot 0{,}24 = 0{,}7$. Obrátene, ak nerovnosti platia, reťazec WSPT platí
a podľa vety 2.5 je poradie optimálne. Nutnosť: v zoradenej dávke so všetkými triedami susedia
bloky REG|PROJ, PROJ|COMP a COMP|SUB; ak by niektorá nerovnosť neplatila, výmena susedného páru na
hranici blokov by cieľ ostro zlepšila. $\blacksquare$

**Poznámka (dávka bez niektorých tried).** Ak niektorá trieda v dávke chýba, treba len nerovnosti medzi
triedami, ktoré sú v dávke susedné. Napr. dávka bez COMP vyžaduje iba $w_{\mathrm{REG}} \ge 0{,}1$ a
$500 \le 350 / w_{\mathrm{SUB}}$, t. j. $w_{\mathrm{SUB}} \le 0{,}7$; podmienka $w_{\mathrm{COMP}} \le 0{,}24$ sa vtedy neuplatní.

**Čo to znamená.** Ak má súčasné poradie minimalizovať vážené čakanie, musí autor považovať
úlohu pod kontenciou pamäte za **najviac 24 %** tak dôležitú ako nominálnu úlohu. Toto je
testovateľné dizajnové tvrdenie: buď ho autor potvrdí, alebo treba rank prepočítať z
explicitne zvolených váh. Pozri [P2](OTVORENE_PROBLEMY.md#p2-sémantika-ranku).

**Hranice platnosti.** Veta neplatí pri preempcii, viacerých jadrách, termínoch (deadlines)
ani pri $p_j$ nezávislom od $\pi_m$. Hodnoty $\pi_m$ sú parametre modelu, nie merania.
