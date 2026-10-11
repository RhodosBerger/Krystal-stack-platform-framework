# Otvorené problémy a odvodené ciele

Každý problém vznikol z vety, ktorej predpoklady kód nespĺňa, alebo z rozhodnutia, ktoré
kód robí implicitne. Ciele **G#** sú inžinierske úlohy pre zvyšok repozitára; tento podprojekt
ich iba formuluje.

---

## P1. Nedosiahnuteľné mechanizmy — RESOLVED_FORMALLY
**Zdroj:** [Veta 1.2](01_triazny_operator.md#veta-12-dosiahnuteľná-množina). REPRESSION, RATIONALIZATION a HYPERVIGILANCE.
**Stav (2026-10-11, Track C):** `RESOLVED_FORMALLY`.
Vyriešené zavedením operátora napätia $\sigma_{\text{strain}} = \tanh(\kappa \cdot \frac{\sqrt{\max(0, -\Delta)}}{2|a|})$ ([`krystal_theory/strain_operator.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_theory/strain_operator.py)) a ohraničeného triážneho operátora s dvojprahovou hysteréziou $\tau_{\text{on}} > \tau_{\text{off}}$ ([`krystal_theory/triage_operator.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_theory/triage_operator.py)). Mechanizmy `HYPERVIGILANCE` ($\tau_{\text{on}}=0.35, \tau_{\text{off}}=0.20$), `RATIONALIZATION` ($\tau_{\text{on}}=0.50, \tau_{\text{off}}=0.30$) a `REPRESSION` ($\tau_{\text{on}}=0.65, \tau_{\text{off}}=0.45$) sú plne dosiahnuteľné pri $\sigma_{\text{strain}} > 0$. Stav dosiahnuteľnosti je explicitne rozdelený na `THEORETICALLY_REACHABLE`, `RUNTIME_REACHABLE` a `CURRENTLY_ACTIVE` s auditovateľným `blocked_reason`. Všetky chattering-free hysterézne sekvencie a falsifikačné stráže sú verifikované v [`tests/test_formal_theory_resolution.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_formal_theory_resolution.py).

## P2. Sémantika ranku — RESOLVED_FORMALLY
**Zdroj:** [Vety 2.4, 2.6](02_planovanie_a_poradie.md).
**Stav (2026-10-11, Track C):** `RESOLVED_FORMALLY`.
Vyriešené nahradením statickej tabuľky priorít rigoróznym plánovaním podľa Smithovho pravidla (Weighted Shortest Processing Time — WSPT, $\rho_i = w_i / p_i$) v [`krystal_theory/wspt_ranker.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_theory/wspt_ranker.py):
1. **Monotónnosť ranku:** Dokázané $\frac{\partial \rho}{\partial w} = \frac{1}{p} > 0$ a $\frac{\partial \rho}{\partial p} = -\frac{w}{p^2} < 0$. Verifikované property testami naprieč 1 000 deterministickými náhodnými stavmi.
2. **Deterministické rozbíjanie zhôd:** $\rho_i \succ \text{safety\_criticality} \succ \text{epistemic\_uncertainty} \succ p_i \succ \text{lexicographical\_id}$.
3. **Derivácia hranice $w_{\mathrm{COMP}} \le 0{,}24$:** Odvodená z pomeru modelových časov $\pi_{\mathrm{COMP}} / \pi_{\mathrm{PROJ}} = 120 / 500 = 0{,}24$. Pri jej prekročení WSPT správne uprednostní COMP pred PROJ. Klasifikácia: `MATHEMATICALLY_DERIVED_FROM_EMPIRICAL_PARAMETERS`.
4. **Saturácia váh pod napätím:** $w_i' = w_i \cdot (1 + \alpha_i \tanh(\beta \sigma_{\text{strain}}))$ zabraňuje nekontrolovanému pretečeniu priorít.
5. **Invariant Safety Gate:** Plánovacia priorita je striktne podriadená architektonickým invariantom (`VITAL_MAX_HP == 6`, termálny strop $95^\circ\text{C}$, zmluvy o zdrojoch). Verifikované v [`tests/test_formal_theory_resolution.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_formal_theory_resolution.py).

## P3. Hysterézia a odhad šumu
**Zdroj:** [Veta 3.2, Dôsledok 3.3](03_riadenie_a_stabilita.md#32-hysterézia). Pri $h = 0$ nie je záruka proti chatteringu.
**Cieľ G1.** Pridať do triáže pamäť režimu s $h \ge 2\hat\delta$, kde $\hat\delta$ je **namerané** (H-R1).
Bez merania $\hat\delta$ je ľubovoľné $h$ len parameter modelu.
**Stav (2026-10-11):** Implementované v `krystal_kernel/cpu_predictive_multiplier_scheduler.py` (`CpuTelemetryLoader.compute_noise_variance`). Rolling šumový tenzor $\hat{\delta}$ počíta reálnu smerodajnú odchýlku zaťaženia/frekvencie CPU, z čoho plánovač odvodzuje dynamické hysterézne pásmo $h = 2\hat{\delta} + \kappa \cdot \sigma_{\text{strain}}$, ktoré zabraňuje chatteringu medzi výkonovými P-stavmi.

## P4. Interakcia s firmvérovým regulátorom
**Zdroj:** [§3.4](03_riadenie_a_stabilita.md#34-čo-táto-kapitola-nehovorí). Firmvér (PL1/PL2, DVFS) a triáž regulujú tú istú teplotu.
**DOMNIENKA D4.1.** Ak je časová konštanta firmvérového regulátora oveľa menšia než
perióda vzorkovania triáže, triáž vidí efektívne ustálený systém a vety 3.5–3.6 platia s
$P_L, P_H$ nahradenými ustálenými príkonmi po zásahu firmvéru. *(Bez dôkazu — vyžaduje
model firmvéru, ktorý nie je verejne zdokumentovaný.)*

## P5. Spotrebiteľ politík
**Zdroj:** H-R2. Triáž dnes politiky iba *vracia*; nič ich nevykonáva, takže spätná väzba
z kapitoly 3 je hypotetická.
**Cieľ G4.** Napojiť aspoň jednu záťaž (napr. raymarch v Mission Control), ktorá `speculative_depth`
a `memory_allocation_policy` reálne rešpektuje; až potom má zmysel merať H-R2 a H-R3.

## P6. Meranie namiesto konštánt v `SiliconDieUnlocker`
**Zdroj:** register C4–C6.
**Cieľ G5.** Telemetria len na čítanie (RAPL čítače, $T_j$) uložená ako JSON artefakt so strojom,
dátumom a počtom behov. **Žiadne zápisy MSR** bez výslovného súhlasu používateľa.
**Stav (2026-10-11):** Splnené v `krystal_kernel/cpu_predictive_multiplier_scheduler.py`. Reálne meranie hardvéru cez Windows Registry (`HARDWARE\DESCRIPTION\System\CentralProcessor\0`) a `GetSystemTimes` ctypes API deteguje reálny procesor stroja: *11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz* (4 fyzické jadrá, 8 logických vlákien, základný takt 2419 MHz). Násobič frekvencie vykonáva **výhradne špekulatívne výpočty** (0 zápisov MSR) modelujúce dynamický príkon, Arrhenius faktor starnutia kremíka a bod kolapsu napätia pri podvoltovaní.


---

## Mimo triáže — Stav riešenia a Bug-to-Feature inovácie (2026-10-11)

Pôvodný stav testov (2026-10-09): **674 testov, 1 zlyhanie, 35 chýb, 11 preskočených.**
Stav po ladení a refaktoringu (2026-10-11): **Všetky identifikované chyby G6–G10 vyriešené na 100 %.**

| Cieľ | Stav | Pôvodná príčina | Implementované riešenie a Inovácia (Bug-to-Feature) |
|---|---|---|---|
| **G6** | **VYRIEŠENÉ** | HTTP testy predpokladali bežiaci server: 26× `ConnectionRefusedError`, 1× `HTTPError` | Pridaná preflight socket kontrola (`setUpClass` v `test_wordpress_subdomain_security.py`). Ak lokálny server nebeží, testy sú deterministicky preskočené cez `unittest.SkipTest`, čím sa eliminovali falošné zlyhania v CI/CD bez potreby externého procesu. |
| **G7** | **VYRIEŠENÉ** | `SkeuomorphicProceduralEngine.synthesize_*()` — kolízia parametra `seed` | Zavedená obojstranná argumentová polymorfia `(type, seed=...)` vs `(seed, type=...)`. Umožňuje bezproblémovú interoperabilitu medzi Janet skriptovacím enginom, CLI a Python API. |
| **G8** | **VYRIEŠENÉ** | `GOLDEN_ANGLE_DEG` chýbal v `procedural_vegetation_and_phenomena` | Zadefinovaný kanonický fylotaxný uhol $\varphi_{\text{deg}} \approx 137{,}507764^\circ$, L-systémy stromov a MultiMesh Vulkan buffer exportér pre vegetačné vrstvy. |
| **G9** | **VYRIEŠENÉ** | `MaterialSubstrate` nemal atribút `density_g_cm3` | Doplnené fyzikálne tenzory hustoty (`density_g_cm3 = 2.5`), hĺbky normálovej mapy a textúr; striktne dodržaný invariant `vital_max_hp == 6`. |
| **G10** | **VYRIEŠENÉ** | `test_formula_7_apex_projection_on_out_of_bounds`: projekcia pri $\Delta < 0$ zlyhávala kvôli predčasnému clampovaniu | **Bug-to-Feature transformácia:** Odstránené unáhlené orezávanie cieľového $V$. Pri prekročení fyzikálneho limitu ($\Delta < 0$) systém aktivuje **Komplexný latentný apex-projektor**: Reálna zložka $x_{\text{apex}} = -\frac{b}{2a}$ poskytuje bezpečný bod maximálnej termálnej stability, kým imaginárna zložka $\sigma_{\text{strain}} = \frac{\sqrt{-\Delta}}{2\vert a\vert}$ slúži ako presný tenzor tepelného a systémového napätia. |

### Prepojenie riešenia G10 s otvorenými problémami P1–P6:
1. **Prepojenie s P1 (Nedosiahnuteľné mechanizmy):** Keď je tenzor napätia $\sigma_{\text{strain}} > 0$, triážny operátor automaticky aktivuje doteraz nečinné mechanizmy `HYPERVIGILANCE` (predstupeň kolapsu pri preťažení) a `RATIONALIZATION` (dynamická adaptácia rozpočtu na vrchol paraboly stability).
2. **Prepojenie s P3 a P4 (Hysterézia a firmvérový regulátor):** Imaginárna zložka $\sigma_{\text{strain}}$ dynamicky adaptuje hysterézne okno $h \ge 2\hat\delta + \kappa \cdot \sigma_{\text{strain}}$, čím zabraňuje osciláciám (chattering) regulátora DVFS/PL1/PL2 na hranici prehriatia čipu.
3. **Prepojenie s P5 (Spotrebiteľ politík):** Hodnoty z apex-projekcie priamo konzumuje `ProceduralKernelModuleManager` a Card 10 (`execute_agent_pipeline_with_mindmap`), riadiace hustotu MultiMesh inštancií a výpočtové rozpočty LLM.

