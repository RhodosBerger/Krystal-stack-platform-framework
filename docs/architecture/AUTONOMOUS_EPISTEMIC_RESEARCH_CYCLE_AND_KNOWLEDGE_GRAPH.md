# Autonómny výskumno-inžiniersky cyklus a Epistemický Knowledge Graph (V-AUTONOMOUS-RESEARCH-01)
**Document ID:** `SPEC-AUTONOMOUS-RESEARCH-01`  
**Milestone:** `V-AUTONOMOUS-RESEARCH-01`  
**Classification:** Autonómny výskumno-inžiniersky cyklus, Knowledge Graph & Modularity Gate  
**Global Invariant:** `GLOBAL_VITAL_MAX_HP == 6`  

---

## 1. Účel a základná výskumná doktrína

Tento dokument povyšuje Krystal Policy Engine z reaktívneho tunera na **autonómny výskumno-inžiniersky cyklus**.

Systém neslúži na slepý hon za syntetickými benchmarkovými číslami ani preťažovanie parametrov. Cieľom je **odhaľovanie nových kauzálnych mechanizmov**, ktoré štrukturálne zlepšujú správanie systému a prehlbujú poznanie o vzájomnej interakcii hardvéru a softvéru.

> ### Jadrová výskumná direktíva
> **„Nie zrýchli kód, ale spochybni predpoklady, nájdi neznáme a falzifikuj vlastné modely.“**

---

## 2. 8-kroková autonómna výskumná slučka (The 8-Stage Epistemic Research Loop)

Každý výskumný cyklus prebieha v uzavretej 8-fázovej sekvencii:

```
                  [ 1. FRESH SYSTEM RE-MAPPING ]
                   Zmapuj celý systém nanovo
                                |
                                v
                  [ 2. HYPOTHESIS PORTFOLIO + M0 ]
                   Generuj viacero hypotéz vrátane nulovej (M0)
                                |
                                v
                  [ 3. ISOLATED ACTUATION + ROLLBACK ]
                   Priraď kontrolnú plochu (Tier 1..9), falzifikátor a rollback
                                |
                                v
                  [ 4. EVOI CANDIDATE SELECTION ]
                   Vyber kandidáta podľa informačného zisku (EVOI), nie greedily
                                |
                                v
                  [ 5. TRANSACTIONAL ACTUATION & MEASURE ]
                   Aplikuj cez tenký adaptér a zmeraj pred a po
                                |
                                v
                  [ 6. ADVERSARIAL CRITIC AUDIT ]
                   Nezávislé spochybnenie (hľadanie confounderov a flukov)
                                |
                                v
                  [ 7. KNOWLEDGE GRAPH REVISION ]
                   Aktualizuj stavy uzlov v Epistemic Knowledge Graph
                                |
                                v
                  [ 8. MODULARITY GATE: MODULE VS HACK ]
                   Otázka: Je to čistý nový modul, alebo len ad-hoc hack?
```

### Podrobný popis fáz:
1. **Fresh System Re-Mapping:** V každom cykle sa nanovo prečíta kompletný stav CPU, pamäte, I/O, akcelerátorov a termálov, čím sa eliminuje predpojatý odhad z predchádzajúcich behov.
2. **Hypothesis Portfolio + $M_0$ (NULL Model):** Pre každú neznámu alebo pozorovaný jav sa povinne formuluje portfólio hypotéz, kde $M_0$ predpokladá, že jav je šum, chyba kalibrácie orakula alebo náhodná fluktuácia.
3. **Isolated Actuation + Rollback:** Každý kandidát má jasne definovanú kontrolnú plochu v hierarchii 1..9 (App Config až Boot param) a priradenú triedu rollbacku (`EXACT_STATE`, `CONFIG_RESTORE`, `DRAIN_AND_CONVERGE`, `IRREVERSIBLE_SIDE_EFFECT`).
4. **EVOI Candidate Selection:** Kandidát sa nevyberá iba podľa najvyššieho sľubovaného zrýchlenia ($\text{Gain}$), ale podľa **očakávaného informačného zisku (Expected Value of Information)**:
   $$\text{Priorita}_{\text{EVOI}} = \frac{\text{Informačný zisk (Diskrétna entropia)} \times \text{Falzifikačná hodnota}}{\text{Riziko} \times \text{Náklady}}$$
   Testuje sa to, čo najefektívnejšie vyradí chybné teórie a zúži priestor prípustných modelov.
5. **Transactional Actuation & Measure:** Vykoná sa izolovaná zmena s pred a po meraním v rámci domain-scoped zámku.
6. **Adversarial Critic Audit:** Nezávislý oponent (`AdversarialCritic`) aktívne pátra po dôvodoch zlyhania:
   - Nešlo o prechodnú fluktuáciu?
   - Nezlepšila sa latencia na úkor skrytého vyhladovania iného procesu (thermal trottling, memory leak)?
   - Nebolo testovacie orakulum triviálne splnené bez reálneho mechanizmu?
7. **Knowledge Graph Revision:** Zistenia sa zapíšu do epistemického orientovaného grafu so statusmi (`HYPOTHESIS`, `UNDER_TEST`, `FALSIFIED`, `APPROVED_MECHANISM`, `UNKNOWN_GAP`, `CONFLICT`).
8. **Modularity Gate („Nový modul vs. Hack“):**
   - Po ukončení cyklu prebehne architektonický audit:
     * *„Dá sa tento mechanizmus vyjadriť ako čistá, znovupoužiteľná abstrakcia v Krystal Stacku (nový modul)?“*
     * Ak ide iba o „magickú konštantu“ viazanú na jedno špecifické prostredie, klasifikuje sa ako **AD_HOC_HACK** a je odmietnutá alebo odoslaná na refaktorovanie do zovšeobecneného kontraktu.

---

## 3. Štruktúra Epistemického Knowledge Graphu

Uzly v grafe reprezentujú typy poznania:
* `MechanismNode`: Skúmaný alebo potvrdený kauzálny mechanizmus (napr. VSync Blanking Micro-Slicing).
* `HypothesisNode`: Aktuálne testované hypotézy s falzifikátormi.
* `ModelNode`: Konkurenčné modely (vrátane $M_0$ NULL).
* `UnknownGapNode`: Formalizované biele miesta v telemetrii (nezmapované javy).
* `ConflictNode`: Nevyriešené rozpory medzi štúdiami alebo meraniami.

Hrany nesú sémantické vzťahy:
* `SUPPORTS`, `WEAKENS`, `FALSIFIES`, `REPLACES`, `DEPENDS_ON`, `DISCRIMINATES_AGAINST`.

---

## 4. Invarianty a architektonické mantry

1. **`GLOBAL_VITAL_MAX_HP == 6`:** Nemenný globálny zmluvný invariant.
2. **Zákaz ad-hoc hackov:** Každá schválená zmena musí byť povýšená na first-class systémový modul, capability policy alebo workload policy.
3. **Zákaz priamej manipulácie hostiteľa výskumnou vrstvou:** Výskum navrhuje plány experimentov, ale hostiteľa smie meniť výhradne Policy Engine.

---

## 5. Milestone V-AUTONOMOUS-RESEARCH-02: Closed-Loop Physical Experimentation

Milestone V2 premieňa simulovaný experimentálny tok na skutočný **uzavretý výskumno-experimentálny cyklus** nad hostiteľským operačným systémom.

> ### Kľúčová doktrína V2 (Core Doctrine)
> **„Research may propose experiments, but only the Policy Engine may alter the host; knowledge may be revised only from attested experiment receipts.“**

### Cieľový tok riadenia (Closed-Loop Pipeline):

```
                   [ ResearchOrchestrator ]
                              |
                              v
                      [ SystemRemapper ]
             (Číta reálny OS/HW stav, CPU, pamäť, topológiu)
                              |
                              v
                    [ HypothesisPortfolio ]
      (Trojitý nulový model: M0_NO_EFFECT, M1_NOISE, M2_MEASUREMENT_ERROR)
                              |
                              v
                      [ ExperimentPlan ]
                (Čistý návrh zásahu bez aktuácie)
                              |
                              v
               [ InformationGainPriorityHeuristic ]
            (Prioritizácia podľa informačného zisku a falzifikácie)
                              |
                              v
              [ PolicyEngine.execute_transaction() ]
           (Jediná povolená cesta k zásahu cez tenké adaptéry)
                              |
                              v
                     [ ExperimentReceipt ]
        (Attestovaný výsledok: baseline, post, rollback, fingerprint)
                              |
                              v
                    [ Independent Critic ]
           (Izolovaný oponent s parametrizovanou critic policy)
                              |
                              v
                 [ KnowledgeRevisionReceipt ]
       (Append-only kryptograficky linkovaná revízna história grafu)
                              |
                              v
                [ Modularity Classification ]
  (4 úrovne: NEW_FIRST_CLASS_MODULE, CAPABILITY_SPECIFIC_POLICY,
              WORKLOAD_SPECIFIC_POLICY, AD_HOC_HACK)
```

### Kľúčové komponenty V2:

1. **`SystemRemapper` & `EnvironmentFingerprint`:**
   - Real-time audit topológie hostiteľa (platforma, jadro, CPU sety, pamäť, cgroups/JobObjects, rozhrania).
   - Generuje kryptograficky zakotvený `EnvironmentFingerprint` s časovou pečiatkou a SHA-256 odtlačkom.

2. **Diferencované nulové modely (`NullModelClass`):**
   - `M0_NO_EFFECT`: Zásah nespôsobuje žiadnu kauzálnu odchýlku od baseline stavu.
   - `M1_NOISE`: Nameraný rozdiel leží plne v stochastickom šume systému ($\le 2\sigma$).
   - `M2_MEASUREMENT_ERROR`: Nameraný rozdiel je telemetrický artefakt alebo zlyhanie validátora (napr. drift časovača, thermal drift confounder, pretečenie počítadla).

3. **`ExperimentPlan` vs `PolicyEngine.execute_transaction()`:**
   - `ExperimentPlan` je nemenný návrh (`SubsystemAction`s, cieľová metrika, očakávaný zisk, falzifikátor).
   - Výskumná vrstva nemá oprávnenie priamo zasahovať do subsystémov; zásah smie vykonať len `PolicyEngine`, ktorý riadi domain locky, `HardExclusionSafetyGate` a garantuje rollback.

4. **`ExperimentReceipt`:**
   - Formálny záznam o vykonanom zásahu nesúci presnú telemetriu pred a po, informáciu o aktivácii falzifikátora, zoznam unwound akcií a fingerprint prostredia.

5. **`KnowledgeRevisionReceipt` (Append-Only revízna história):**
   - Žiadny uzol grafu sa neprepisuje na mieste bez zachovania histórie. Každá revízia generuje nemenný `KnowledgeRevisionReceipt` naviazaný na predchádzajúci digest a prislúchajúci `ExperimentReceipt`.

6. **4-stupňová taxonómia modularity (`ModuleVsHackClassifier`):**
   - `NEW_FIRST_CLASS_MODULE`: Plne prenosný modul (`is_cross_platform_abstracted=True`), formálne rozhranie, invariantné mantinely, žiadne magické konštanty.
   - `CAPABILITY_SPECIFIC_POLICY`: Platformovo alebo hardvérovo viazaná konfigurácia (napr. Windows DWM EcoQoS vs Linux sysfs) s čistou abstrakciou capability providera.
   - `WORKLOAD_SPECIFIC_POLICY`: Špecifická adaptácia pre konkrétny pracovný režim (napr. LLM inferencia vs 120Hz VSync render).
   - `AD_HOC_HACK`: Spolieha sa na nekalibrované magické konštanty, postráda formálny kontrakt alebo porušuje invarianty.

---

## 6. Milestone V-AUTONOMOUS-RESEARCH-03: Grounded Host Actuation & Attested Measurement

Milestone V3 posúva systém od emulovaných a simulovaných experimentov k **reálnej, meranej fyzickej interakcii s hostiteľským OS**.

> ### Ústredný invariant V3 (Core Invariant)
> $$\text{KnowledgePromotion} \Longrightarrow \text{PhysicalMeasurementReceipt} \land \text{CanonicalDigestOK} \land \text{IndependentObservation}$$

### Kľúčové vylepšenia V3:

1. **Reálna aktuácia na kontrolných plochách OS (`GroundedCapabilityAdapter`):**
   - Žiadne púhe prepisovanie interných Python atribútov.
   - **Tier 2 (ENV_VARS):** Fyzické nastavenie premenných prostredia cez `os.environ` (napr. `OMP_NUM_THREADS`, `KRYSTAL_PERF_MODE`) s deterministickým LIFO rollbackom.
   - **Tier 4 (CGROUPS_OR_JOBOBJECTS / Affinity):** Reálna manipulácia afinity jadier (Windows `SetProcessAffinityMask` / Linux `sched_setaffinity`) a triedy priority procesu s obnovením pôvodnej masky.

2. **Nezávislý telemetrický vzorkovač (`HostTelemetrySampler`):**
   - Meranie baseline a post-change nie je dodávané umelo zo simulácie.
   - Sampler vykonáva kalibrovaný fyzický mikroworkload, meria nanosekundový wall-clock čas (`perf_counter_ns`), procesorový čas procesu (`process_time_ns`), a reálnu spotrebu pamäte (Windows `GetProcessMemoryInfo` WorkingSetSize / Linux `/proc/self/statm` RSS).

3. **Dvojvrstvová identita prostredia (Dual-Layer Environment Identity):**
   - **`EnvironmentIdentityDigest`:** Deterministický statický SHA-256 odtlačok nemenného hardvéru (platforma, release OS, počet CPU jadier, celková RAM, podporované tiery). Nemenný naprieč cyklami na tom istom stroji.
   - **`ObservationDigest`:** Časovo viazaný dynamický odtlačok okamžitého stavu v čase $t$ (voľná pamäť, reálny governor z `/sys` alebo power scheme, nanosekundový timestamp, prepojenie na identity digest).

4. **Kanonické kanonizované digestovanie (Canonical Digest Binding):**
   - `ExperimentReceipt.receipt_digest`: viaže deterministicky zoradené polia vrátane `actions_unwound`, čím garantuje dôkazné reťazenie rollbacku.
   - `KnowledgeRevisionReceipt.receipt_digest`: viaže explicitný text `reason`, `experiment_receipt_id`, predchádzajúci digest aj verzie uzla.

5. **Hlboká nemennosť snapshotov (Deep Immutability):**
   - `node_version_snapshots` uchováva hlboké kópie (`copy.deepcopy`) vrátane metadát, čím zabraňuje retroaktívnej mutácii histórie cez zdieľané referencie.

6. **Striktné vynútenie tripartitného portfólia nulových modelov:**
   - Každý cyklus musí povinne obsahovať celú trojicu:
     $$\{M_{0\text{-NO\_EFFECT}}, M_{1\text{-NOISE}}, M_{2\text{-MEASUREMENT\_ERROR}}\} \subseteq \text{Portfolio}$$
     Vynechanie ktoréhokoľvek z nich okamžite vyvoláva `EpistemicContractViolation`.

7. **Procesne/dôveryhodnostne izolovaný oponent (`IsolatedAdversarialCritic`):**
   - Oddelený hodnotiaci kontext s parametrizovanou vedou riadenou politikou (`AdversarialCriticPolicy`), ktorý nezdieľa stav s navrhovateľom ani orchestrátorom.


