# 00 — Značenie a konvencie

## Stavové štítky

Každé tvrdenie v podprojekte nesie práve jeden štítok:

| Štítok | Význam | Čo ho môže zmeniť |
|---|---|---|
| **DEFINÍCIA** | Zavedenie pojmu. Nie je pravdivá ani nepravdivá. | Zmena kódu, ktorý definícia modeluje |
| **AXIÓM** | Predpoklad prijatý bez dôkazu, explicitne vymenovaný. | Rozhodnutie autora |
| **VETA** | Dokázané tvrdenie *za uvedených predpokladov*. | Chyba v dôkaze alebo zmena kódu |
| **LEMA / DÔSLEDOK** | Pomocná veta / priamy dôsledok vety. | ako VETA |
| **HYPOTÉZA** | Empirické tvrdenie s protokolom a kritériom zamietnutia. | Meranie |
| **DOMNIENKA** | Matematické tvrdenie bez dôkazu. | Dôkaz alebo protipríklad |
| **PARAMETER MODELU** | Číslo zvolené do modelu, nie namerané. | Meranie (stane sa z neho údaj) |

Dôkaz končí symbolom $\blacksquare$. Ak dôkaz závisí od správania kódu, uvádza sa aj test,
ktorý toto správanie zamyká (napr. `test_h1_ranks_are_strict_total_order`).

## Značenie

| Symbol | Význam |
|---|---|
| $\overline{\mathbb{R}}$ | $\mathbb{R} \cup \{+\infty, -\infty, \mathrm{NaN}\}$ — hodnoty typu `float` v Pythone (IEEE 754) |
| $x = (e, c, \theta)$ | Stav úlohy: kontextová entropia $e$, kontencia pamäte $c$, termálny tlak $\theta$ (% z $T_{j,\max}$) |
| $X = \overline{\mathbb{R}}^3$ | Celý vstupný priestor (čo môže funkcia reálne dostať) |
| $F = \mathbb{R}^3$ | Konečné vstupy; $N = X \setminus F$ sú nekonečné vstupy alebo NaN |
| $e^\*, c^\*, \theta^\*$ | Prahy: $e^\* = 0{,}70$, $c^\* = 2{,}0$, $\theta^\* = 85{,}0$ |
| $\mathcal{M}$ | Množina všetkých 7 mechanizmov (enum `PsychologicalCopingMechanism`) |
| $M \subseteq \mathcal{M}$ | Mechanizmy, ktoré triáž dokáže vrátiť |
| $T : X \to M$ | Triážny operátor (`CortexProcessorWhispererMcpBridge.classify`) |
| $r : M \to \mathbb{N}$ | Poradie (rank), 1 = najurgentnejšie |
| $\sigma : M \to \{0,\dots,4\}$ | Hĺbka špekulácie |
| $\pi_m$ | Časový rozpočet mechanizmu $m$ v µs |

Skratky mechanizmov: REG = REGRESSION, PROJ = PROJECTION, COMP = COMPARTMENTALIZATION,
SUB = SUBLIMATION.

## Konvencia porovnaní s NaN

Podľa IEEE 754 je každé usporiadacie porovnanie s NaN nepravdivé: $\mathrm{NaN} \ge a$ aj
$\mathrm{NaN} < a$ sú `False`. Reťazec `if/elif/else` preto pošle NaN do vetvy `else`.
Toto je presne mechanizmus chyby H3 (pozri [01, Veta 1.3](01_triazny_operator.md#veta-13-fail-safe)).
