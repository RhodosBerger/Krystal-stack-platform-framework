# 01 — Triážny operátor

*Modeluje:* `CortexProcessorWhispererMcpBridge.classify` a `TRIAGE_POLICY` v
[`psychological_coping_cortex_mcp_bridge.py`](../krystal_kernel/psychological_coping_cortex_mcp_bridge.py).
*Posledná kontrola voči kódu:* 2026-10-09.

---

## 1.1 Definícia operátora

**DEFINÍCIA 1.1 (Triážny operátor).** Pre $x = (e, c, \theta) \in X$:

$$
T(x) =
\begin{cases}
\mathrm{REG}  & \text{ak } x \in N \;\text{ alebo }\; \theta \ge \theta^\* \\
\mathrm{COMP} & \text{ak } x \in F,\ \theta < \theta^\*,\ c \ge c^\* \\
\mathrm{SUB}  & \text{ak } x \in F,\ \theta < \theta^\*,\ c < c^\*,\ e \ge e^\* \\
\mathrm{PROJ} & \text{ak } x \in F,\ \theta < \theta^\*,\ c < c^\*,\ e < e^\*
\end{cases}
$$

Ide o **rozhodovací zoznam s lexikografickou prioritou** $\theta \succ c \succ e$:
o výsledku rozhoduje prvá premenná, ktorá prekročí svoj prah.

**DEFINÍCIA 1.2 (Politika).** Funkcia $P : M \to \mathbb{N} \times \mathbb{R}_{>0} \times \{0..4\}$,
$P(m) = (r(m), \pi_m, \sigma(m))$:

| $m$ | $r(m)$ | $\pi_m$ [µs] | $\sigma(m)$ |
|---|---:|---:|---:|
| REG  | 1 |  50 | 0 |
| PROJ | 2 | 500 | 4 |
| COMP | 3 | 120 | 1 |
| SUB  | 4 | 350 | 2 |

Hodnoty $\pi_m$ a $\sigma(m)$ sú **PARAMETRE MODELU**: nemajú v repozitári meranie.

---

## 1.2 Základné vlastnosti

### VETA 1.1 (Totálnosť a determinizmus)
$T$ je definovaný na celom $X$ a je to funkcia (každý vstup má práve jeden výstup).

*Dôkaz.* Množiny
$R_{\mathrm{REG}} = N \cup \{x\in F : \theta \ge \theta^\*\}$,
$R_{\mathrm{COMP}} = \{x \in F: \theta<\theta^\*, c \ge c^\*\}$,
$R_{\mathrm{SUB}} = \{x\in F: \theta<\theta^\*, c<c^\*, e\ge e^\*\}$,
$R_{\mathrm{PROJ}} = \{x\in F: \theta<\theta^\*, c<c^\*, e<e^\*\}$
sú po dvoch disjunktné: každé dve sa líšia aspoň v jednej podmienke, ktorá je v jednej
množine splnená a v druhej negovaná. Ich zjednotenie je $X$: pre $x \in N$ platí $x \in R_{\mathrm{REG}}$;
pre $x \in F$ je každé porovnanie dobre definované a pravdivé alebo nepravdivé, takže
rozhodovací strom so 4 listami pokryje všetky prípady. Množiny teda tvoria rozklad $X$
a $T(x)$ je index bloku, do ktorého $x$ patrí. $\blacksquare$

**Poznámka (typová doména).** $X$ obsahuje len hodnoty typu `float`. Vstupy iného typu sa na $X$
zobrazia pred aplikáciou $T$ funkciou `_coerce_float`: `None` → predvolená hodnota (ak existuje), `bool`
→ `TypeError`, číslo mimo rozsahu `float` → `ValueError`. Odmietnuté vstupy teda do $T$ nevstúpia;
NaN a $\pm\infty$ vstúpia a vzťahuje sa na ne veta 1.3.

### VETA 1.2 (Dosiahnuteľná množina)
$M = T(X) = \{\mathrm{REG}, \mathrm{PROJ}, \mathrm{COMP}, \mathrm{SUB}\}$, teda
$\mathcal{M} \setminus M = \{\mathrm{REPRESSION}, \mathrm{RATIONALIZATION}, \mathrm{HYPERVIGILANCE}\}$.

*Dôkaz.* $T$ vracia len štyri konštanty z definície 1.1, takže $T(X) \subseteq M$. Každá je
dosiahnuteľná: $(0,0,90) \mapsto \mathrm{REG}$, $(0,3,50) \mapsto \mathrm{COMP}$,
$(0{,}9,0,50) \mapsto \mathrm{SUB}$, $(0,0,50) \mapsto \mathrm{PROJ}$. $\blacksquare$

*Zamknuté testom:* `test_h2_triage_reachable_set_is_documented`.
*Dôsledok pre dizajn:* tri mechanizmy z taxonómie triáž nikdy nevyberie. Pozri
[OTVORENE_PROBLEMY.md, P1](OTVORENE_PROBLEMY.md#p1-nedosiahnuteľné-mechanizmy).

### VETA 1.3 (Fail-safe)
Pre všetky $x \in N$: $T(x) = \mathrm{REG}$, $r(T(x)) = \min r = 1$ a $\sigma(T(x)) = 0$.

*Dôkaz.* Priamo z prvého riadku definície 1.1 a z tabuľky v definícii 1.2. $\blacksquare$

**Poznámka (pôvodná chyba H3).** Pred opravou kód testoval len $\theta \ge \theta^\*$, $c \ge c^\*$, $e \ge e^\*$.
Podľa konvencie IEEE 754 (pozri [00](00_notacia_a_konvencie.md#konvencia-porovnaní-s-nan)) sú pre NaN všetky
tri porovnania nepravdivé, takže NaN padol do `else` → PROJ s $\sigma = 4$: **najagresívnejšia
politika pre úplne neznámy stav.** Veta 1.3 je bezpečnostná vlastnosť, ktorú pôvodný kód
porušoval. *Zamknuté testom:* `test_h3_non_finite_inputs_fail_safe`.

### LEMA 1.4 (Záporné hodnoty)
Pre konečné vstupy $T(e,c,\theta) = T(\max(e,0), \max(c,0), \theta)$.

*Dôkaz.* Prahy $e^\* = 0{,}7$ aj $c^\* = 2$ sú kladné. Pre $e<0$ platí $e<e^\*$ aj $\max(e,0) = 0 < e^\*$,
takže porovnanie s prahom dáva rovnaký výsledok; rovnako pre $c$. $\blacksquare$

*Dôsledok:* záporné vstupy (fyzikálne neplatné) nemenia klasifikáciu, preto ich kód
nemusí orezávať. Chyba by vznikla až pri zápornom prahu.

---

## 1.3 Konzistencia s inštrukčnými hintami

**DEFINÍCIA 1.3.** Predikát špekulácie $W : \mathcal{M} \to \{0,1\}$ je hodnota
`speculative_branch_taken` z `whisper_speculative_instruction`.

### VETA 1.5 (Konzistencia špekulácie)
Pre všetky $m \in M$: $W(m) = 1 \iff \sigma(m) > 0$.

*Dôkaz (po oprave H9).* PROJ: kód vracia `True` a $\sigma = 4 > 0$. Pre ostatné $m \in M$ vetva `else`
počíta `speculative_branch_taken = σ(m) > 0` z tej istej tabuľky `TRIAGE_POLICY`. $\blacksquare$

*Pred opravou* platilo $\sigma(\mathrm{SUB}) = 2$, ale $W(\mathrm{SUB}) = 0$: triáž sľúbila špekuláciu,
ktorú hint zakázal. *Zamknuté testom:* `test_h9_whisper_agrees_with_triage_spec_depth`.

Pre $m \in \mathcal{M} \setminus M$ (napr. RATIONALIZATION) $\sigma$ nie je definovaná a veta o nich
nič netvrdí.
