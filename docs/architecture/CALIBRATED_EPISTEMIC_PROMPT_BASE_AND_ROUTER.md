# Kalibračná báza promptov a Router mikropromptov (V-PROMPT-ROUTER-01)
**Document ID:** `SPEC-EPISTEMIC-ROUTER-01`  
**Milestone:** `V-PROMPT-ROUTER-01`  
**Classification:** Epistemická kalibračná báza & Router architektúra  
**Constitutional Invariant:** `VITAL_MAX_HP == 6`  

---

## 1. Účel a architektonický invariant

Cieľom tejto špecifikácie je **zviazať divergentné generovanie (EXPLORE) prísnym epistemickým protokolom (VERIFY)** prostredníctvom modulárnej **kalibračnej bázy promptov** riadenej špecializovaným **Routerom**.

Namiesto jedného obrovského, monolitického systémového promptu, ktorý trpí degradáciou pozornosti (attention dilution) a generickým správaním, systém používa **malé, vysoko špecializované mikroprompty**, ktoré sa dynamicky vyberajú podľa typu úlohy:

```
                                  [ Úloha / Kontext ]
                                           |
                                           v
                          +----------------------------------+
                          |      EPISTEMIC PROMPT ROUTER     |
                          +----------------------------------+
                                           |
          +-------------------+------------+------------+--------------------+
          |                   |                         |                    |
          v                   v                         v                    v
    [ P_M0_M7 ]       [ P_CONFLICT ]             [ P_UNKNOWN ]        [ P_PROMOTION ]
    8 Konkurenčných   Nekompromisný spor         Formálne             Deterministická
    modelov (M0-M7)   (Žiadne priemery)          Neznámo              Brána overenia
```

---

## 2. Ústavný prompt (Constitutional Prompt Directive)

Ústavný prompt je fundamentálnou normou všetkých inferencií a generovaní:

> **Ústavná direktíva:**  
> 1. Generuj aj nekonvenčné, odvážne hypotézy, ale **nikdy si ich nepomýľ s overeným faktom**.  
> 2. Každému dôležitému tvrdeniu priraď 6-zložkovú evidenčnú kartu:  
>    $$\text{ClaimCard} = \langle \text{Trieda}, \text{Zdroj}, \text{Proveniencia}, \text{Rozsah}, \text{Evidencia}, \text{Proti-evidencia} \rangle$$  
> 3. **Zákaz neoprávneného vyvrátenia:** Neoverené tvrdenie (`EVIDENCE_INSUFFICIENT`) nie je vyvrátené (`CONTRADICTED` / `REFUTED`).  
> 4. **Zákaz konvenčnej dogmy:** Konvenčné tvrdenie (`CONVENTIONAL`) nie je automaticky platné (`VALID`).  
> 5. **Internet je zdrojom hypotéz, nie verdiktu:** Vždy mapuj rodokmeň a závislosti zdrojov.  
> 6. **Zákaz vyhladzovania priemerom:** Pri rozpore zdrojov nevyrovnávaj závery do kompromisného stredu; vytvor explicitný spor a navrhni rozlišujúci test.  
> 7. **Zákaz fabulácie príbehov:** Pri neznámom mechanizme založ uzol `UNKNOWN_MECHANISM` s definovaným chýbajúcim testom.  
> 8. **Neutrálna optimalizácia:** Neoptimalizuj výstup na súhlas (sykofancia) ani na odpor (antagonizmus), optimalizuj výhradne na evidenciu.  
> 9. **Povýšenie čaká na externý audit:** Povýšenie do lokálnej pravdy (`APPROVED_FACT`) podlieha nezávislému auditu (Anti-Self-Certification). Výstup označ statusom a neistotou, nie sebavedomou vetou.

---

## 3. Portfólio 8 konkurenčných modelov ($M_0$ až $M_7$)

Pre každú skúmanú otázku či úzke miesto systém povinne formuluje a udržiava **8 paralelných konkurenčných modelov**:

| Model | Názov paradigmy | Popis a mechanizmus | Príklad v runtime |
| :--- | :--- | :--- | :--- |
| **$M_0$** | **Konvenčný (Baseline)** | Štandardné procesné vysvetlenie v doméne | Vyčerpanie thread poolu, I/O čakanie |
| **$M_1$** | **Architektonický** | Hardvérové invarianty, štruktúry zberníc, pamäť | Cache line bouncing, UMA/NUMA bottleneck |
| **$M_2$** | **Interakčný** | Dynamická interferencia medzi procesmi | LLM burst vs. VSync scanout kolízia |
| **$M_3$** | **Adversariálny** | Najhorší možný scenár, útok, šum, jitter | Hostilný fuzzer, zámerný jitter payload |
| **$M_4$** | **Ortogonálny / Alternatívny rámec** | Skrytá mätúca premenná (Confounder) | Meranie CPU cyklov namiesto lock latencie |
| **$M_5$** | **Nulový model (Null)** | Jav je len šumom pozadia alebo predvoleným stavom | Bežná systémová fluktuácia, nulový efekt zmeny |
| **$M_6$** | **Nekonvenčný / Emergentný** | Nelineárny fázový prechod, nová kauzalita | Samočinné preklopenie queue režimu |
| **$M_7$** | **Redukčný / Degradovaný** | Núdzový fallback, operácia v degradovanom móde | Safe-mode fallback s polovičným rozlíšením |

Každý model $M_i$ ($i \in \{0, \dots, 7\}$) musí explicitne obsahovať:
1. `mechanizmus`: Algoritmická či fyzikálna príčina.
2. `predikcie`: Kvantitatívna predikcia s tolerančným pásmom.
3. `falzifikátor`: Presná nameraná hodnota, pri ktorej je model okamžite vyvrátený.

---

## 4. Riešenie rozporov bez priemerovania (`Conflict Resolution`)

Bežné LLM pri protichodných tvrdeniach vytvárajú kompromisný syntetický priemer („niektorí hovoria A, iní B, pravda je uprostred“). V inžinierstve je takýto priemer takmer vždy toxickou nepravdou.

* **Záväzné pravidlo:**
  $$\text{Evidence}(A) \land \text{Evidence}(\neg A) \Longrightarrow \text{CreateDispute}(\text{Node}_{\text{conflict}}) \land \text{DesignCDT}(\text{Test}_{\text{discrim}})$$
* Vzniká uzol `EpistemicConflictNode`, ktorý:
  1. Izoluje oba pramene dôkazov s ich provenienciou.
  2. Zakazuje vyhladenie priemerom.
  3. Navrhuje najlacnejší rozlišujúci test (CDT), ktorý spor deterministicky rozhodne.

---

## 5. Protokol neznámeho mechanizmu (`UNKNOWN_MECHANISM`)

Ak systém nedokáže jav vysvetliť cez modely $M_0$ až $M_7$ s dostatočnou evidenciou:
* **Zákaz fabulácie:** Prísne zakázané vymýšľať kauzálne príbehy.
* **Vytvorenie uzla `UnknownMechanismNode`:**
  - `symptom`: Presný empirický popis javu.
  - `missing_evidence`: Zoznam chýbajúcej telemetrie alebo nepreskúmaných registrov.
  - `next_probe`: Navrhnutá sonda pre zmenu stavu z `UNKNOWN` na testovateľnú hypotézu.

---

## 6. Špecializované mikroprompty v Routeri

Router riadi alokáciu týchto samostatných mikropromptov:

1. **`PROMPT_CONSTITUTIONAL_AUDIT`:** Overenie 6-zložkovej evidenčnej karty tvrdenia.
2. **`PROMPT_M0_M7_GENERATOR`:** Vygenerovanie úplného 8-modelového portfólia pre zadanú otázku.
3. **`PROMPT_SOURCE_DEPENDENCY_AUDIT`:** Mapovanie rodokmeňa a deduplikácia závislostí zdrojov.
4. **`PROMPT_DISPUTE_RESOLUTION`:** Založenie sporu namiesto priemerovania a návrh CDT.
5. **`PROMPT_UNKNOWN_MECHANISM`:** Formalizácia chýbajúceho poznania bez halucinácie.
6. **`PROMPT_STATUS_AND_UNCERTAINTY_FORMATTER`:** Prevod sebavedomej odpovede na formát so statusom a neistotou.
7. **`PROMPT_PROMOTION_AUDIT`:** Deterministická kontrola pred zápisom do lokálnej pravdy (`APPROVED_FACT`).

---

## 7. Constitutional Invariant

$$\text{assert } \text{VITAL\_MAX\_HP} == 6$$

Všetky štruktúry (`EpistemicClaimCard`, `ModelPortfolioM0ToM7`, `EpistemicConflictNode`, `UnknownMechanismNode`, `EpistemicPromptRouter`) striktne validujú `vital_max_hp == 6` v `__post_init__`. Porušenie vyvoláva `ContractViolation`.
