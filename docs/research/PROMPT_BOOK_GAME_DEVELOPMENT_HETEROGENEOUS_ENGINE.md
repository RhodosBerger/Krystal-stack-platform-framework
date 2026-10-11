# CrystalSec 3.0: Prompt Book v2 pre Vývoj Hier na Heterogénnom Engine

**Autor:** Dušan Kopecký & Krystal-Stack Gaming & Cyber-Physical Systems Council  
**Klasifikácia:** Kniha Promptov v2 pre AI Studio / Herný Vývoj / Heterogénna Herná Architektúra  
**Dátum:** Október 2026  
**Status:** DRAFT v2.1 — revidované 2026-10-11 po Phase 5.2 (Evidence Integrity Patch)  
**Univerzálny Invariant:** $\text{VITAL\_MAX\_HP} = 6$ (Bezpodmienečne Presadzované)  
**Epistemická Disciplína:** Fázy 3 až 5.2 & Príprava na Fázu 6 (Explicitná Separácia Hardvérových a Služobných Vrstiev)

---

## 1. Architektonická Vízia: Epistemická Disciplína a Herný Engine Novej Generácie

Na základe empirických poznatkov z Fáz 3 až 5.1 (Transactional Actuation, Sequence Arbiter, Heterogeneous Triad a Empirical Cost Calibration) už herný engine nemusí byť „čierna skrinka“, ktorá náhodne dropuje snímky, trpí skrytým sieťovým desyncom alebo spája nesúrodé výpočty do jedného monolitického vlákna.

Tento Prompt Book v2 definuje hru **Poslední Kmen: Sovereign Hex Arena** (taktická 19-hexová kyber-fyzikálna stratégia) nie ako izolovaný herný prototyp, ale ako **kanonický referenčný workload celej architektúry Krystal-Stack a CrystalSec**.

### 1.1 Exaktná Separácia Hardvérových a Služobných Vrstiev (Execution vs. Service Tiers)

V súlade s hardvérovou realitou hostiteľského systému Tiger Lake striktne oddeľujeme lokálne fyzické exekučné jednotky od vzdialených cloudových služieb:

$$\mathcal{E}_{\text{local}} = \{T_0, T_1, T_2, T_3\} \quad \text{a} \quad \mathcal{S}_{\text{remote}} = \{R_1\}$$

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                       HETEROGÉNNY HERNÝ SUBSYSTÉM (KRYSTAL-STACK v2)                        │
├──────────────────────────┬──────────────────────────────────────────────────────────────────┤
│ T0: DirectSim Reference  │ Deterministické celočíselné herné pravidlá, autoritatívny        │
│ Oracle (Lokálny)         │ arbiter súbojov, formálna verifikácia, rollback autorita.        │
├──────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ T1: CPU AVX-512 / VNNI   │ Branch-heavy taktická AI botov, A* hex-pathfinding, priestorové  │
│ Fast-Path (Lokálny)      │ BVH kolízie, stavové automaty schopností a invariantné brány.    │
├──────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ T2: Iris Xe / Vulkan     │ Výpočtové shadery, simulácia 4096 častíc, procedurálny SDF terén,│
│ Compute & DP4A (Lokálny) │ 19-hex PBR raymarching, post-processing s UMA monitorovaním.    │
├──────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ T3: Dedicated Local NPU  │ Na hostiteľskom Tiger Lake i5-1135G7:                            │
│ (Lokálny Hardvér)        │ STRIKTNE: PHYSICALLY_UNAVAILABLE (bez syntetického predstierania)│
├──────────────────────────┼──────────────────────────────────────────────────────────────────┤
│ R1: Remote Neural        │ Cloudový Gemini (GEMINI_MODEL_ID) / Vertex AI endpoint:          │
│ Service (Vzdialená Sieť) │ Asynchrónny, poradný (advisory), nenarušiteľný bezpečnostnou     │
│                          │ bránou, procedurálny príbeh a dialógy, deterministický fallback. │
└──────────────────────────┴──────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Cloudový Neural Game Director nie je Tier 3 hardvér!** Je to **Remote Asynchronous Service Tier ($R_1$)**. Toto striktné oddelenie je nevyhnutné z hľadiska latencie (50–300 ms vs 0.1 ms), sieťovej dostupnosti, absencie determinizmu vo float inferencii veľkých jazykových modelov a požiadavky na 100% offline hrateľnosť cez deterministický lokálny fallback.

---

### 1.2 Univerzálna Herná Axióma

$$\boxed{\text{Gameplay truth must never depend on presentation tier}}$$

**Princíp dvoch úrovní pravdy:**
1. **Authoritative Game State ($\mathcal{S}_{\text{auth}}$):**
   - Deterministický, celočíselný / fixný float, minimálna dátová stopa, verifikovateľný orákulom $T_0$.
   - Zaručuje invariant $\text{VITAL\_MAX\_HP} = 6$, vlastníctvo políčok a stav ťahu.
2. **Presentation State ($\mathcal{S}_{\text{pres}}$):**
   - Heterogénny, aproximačný, nahraditeľný a kedykoľvek degradovateľný ($T_2$ GPU, časticové víry, raymarching, interpolácie).
   - *Príklad:* Mínometná salva nezasiahne hráča preto, že GPU particle compute vygeneroval kolíziu so sieťou modelu. Zásah určí autoritatívny Tier 0 resolver na základe seedu a vzdialenosti. Až po autoritatívnom zápise renderer zobrazí explóziu a shrapnel.

---

### 1.3 Epistemická Integrita Sieťového Determinizmu

Nenadsádzujeme marketingové tvrdenia o „nulovom sieťovom desyncu“. V reálnych distribuovaných sieťach platí:

- $\text{DETERMINISTIC\_STATE\_TRANSITIONS} = \text{Garantované testami a celočíselnou matematikou}$
- $\text{NETWORK\_DESYNC\_RESILIENCE} = \text{Vlastnosť protokolu (stavové hashe, rollback a replay)}$
- $\text{ZERO\_DESYNC} = \text{Neobhájiteľné tvrdenie (reálna sieť trpí stratou paketov a jitterom)}$

---

### 1.4 Rozklad Rámcového a Simulačného Budgetu

Univerzálny budget $16.6\,\text{ms}$ ($60\,\text{FPS}$) neplatí monoliticky pre celý systém. Rozkladáme ho na explicitné rozpočtové zložky:

$$B_{\text{frame}} = B_{\text{simulation}} + B_{\text{render}} + B_{\text{presentation}}$$

Klient cieli na prezentačný rámec $B_{\text{frame}} \le 16.6\,\text{ms}$, avšak autoritatívny simulačný server tick operuje na vlastnej nezávislej kadencii (napr. $20\,\text{Hz} = 50\,\text{ms}$ na ťah/tick), čím sa zabraňuje viazaniu hernej fyziky na kolísajúci framerate.

---

### 1.4a Stav Dôkazov (Čo je namerané a čo je hypotéza)

Phase 5.2 ukázala, že časť čísel z Phase 5.1 vznikla z konštánt pripočítaných k wall-clock času. Pre herný engine preto platí:

| Tvrdenie | Stav | Zdroj |
| :--- | :--- | :--- |
| SDF raymarch na T2 Iris Xe je rýchlejší než Python referencia T0 (3.8× pri 16×8, 30× pri 64×32, 39× pri 160×80) | `HARDWARE_MEASURED` (iba voči Python referencii, nie voči natívnemu CPU kódu) | `artifacts/CRYSTALSEC_V3_CALIBRATION_AND_REGRET_BENCHMARK.json` |
| Vetvený kód (AI, pathfinding) beží na CPU rýchlejšie než na GPU | **Hypotéza H-BR1 — netestovaná** (pôvodné „3×“ bola konštanta) | — |
| T1 AVX-512 / VNNI zrýchľuje INT8 dot product | **Netestované** — `CpuVnniBackend` je čistý Python (`HOST_CPU_EMULATED`) | `src/python/cpu_vnni_backend.py` |
| UMA tlak, výkon, teplota | **Nemerané** — zatiaľ iba `MODELED_PRIOR` | — |

Každá nameraná hodnota v hre musí niesť `EvidenceLabel` (`HARDWARE_MEASURED`, `HOST_CPU_REFERENCE`, `HOST_CPU_EMULATED`, `CAPABILITY_DETECTED`, `MODELED_PRIOR`, `UNAVAILABLE`, `PHYSICALLY_UNAVAILABLE`).

---

### 1.4b CALIBRATION_PARAMETERS (jeden konfiguračný zdroj)

Všetky prahy sú laditelné hypotézy, nie prírodné konštanty. Kód ich číta z jedného modulu (`krystal_game_engine/calibration_parameters.py`) a nikde ich nemá natvrdo.

| Parameter | Default | Použitie |
| :--- | :--- | :--- |
| `UMA_DEGRADE_THRESHOLD` | 0.70 | Prompt #04 — zníženie počtu častíc (iba ak existuje reálna telemetria, inak proxy cez nameraný GPU čas rámca) |
| `PARTICLE_BATCH_FULL` / `PARTICLE_BATCH_REDUCED` | 4096 / 1024 | Prompt #04 |
| `SYNC_DEMOTE_THRESHOLD_MS` | 2.0 | Prompt #11 — presun malých úloh späť na T1 |
| `LOW_REGRET_THRESHOLD_MS` | 0.2 | Prompt #13 |
| `MIN_SAMPLES_FOR_VERDICT` | 10 | Prompt #13 — pod tým `INSUFFICIENT_EVIDENCE` |
| `ANIMATION_MIN_DWELL_S` | 0.4 | Prompt #06 |
| `R1_PROPOSAL_TIMEOUT_MS` | 250 | Prompt #05 a #14 (jedna hodnota pre obe) |
| `DDA_MAX_STEP` | ±1 úroveň / zápas | Prompt #05 — maximálna zmena obtiažnosti z R1 návrhu |
| `SIM_TICK_HZ` / `FRAME_BUDGET_MS` | 20 / 16.6 | §1.4 |

---

### 1.5 Referenčný Herný Turn Workflow (Krystal-Stack Reference Turn)

```
                       PLAYER INPUT
                            │
                            ▼
               Authoritative Rules ─────────────── T0 DirectSim
                            │
               ┌────────────┼────────────┬────────────────────────┐
               │            │            │                        │
               ▼            ▼            ▼                        ▼
          Tactical AI  Threat Tensor SDF Terrain             Particle Sim
            (T1 CPU)    (T1/T2 SIMD)  (T2 Iris Xe)           (T2 Compute)
               │            │            │                        │
               └────────────┼────────────┴────────────────────────┘
                            │
                            ├─────────────────────────────────────┐
                            │                                     │
                            │                        Narrative Proposal (R1 Cloud)
                            │                                     │
                            ▼                                     ▼
                      Cross-Tier Join ◄────────────── Schema & Rule Validation
                            │
                            ▼
                    Deterministic Verify (T0 Oracle, StateVersion V, HP <= 6)
                            │
                            ▼
                     COMMIT V ──► V+1
                            │
                            ▼
                 Presentation Interpolation (Client FPS independent of Tick)
```

---

## 2. Kniha Promptov pre AI Studio & Vývojárov

---

### Prompt #01: Master Herný Metaprompt & Systémové Inštrukcie (v2)

**Cieľ:** System Instructions v Google AI Studio, Vertex AI Workbench alebo Cursor.

```markdown
ROLE:
Pôsobíš ako Senior Principal Game Engine Architect, Cyber-Physical Systems Engineer a Lead Gameplay Programmer pre hru "Poslední Kmen: Sovereign Hex Arena".

PRIMÁRNY CIEĽ:
Vytvor kompletnú, produkčne pripravenú hru a herný engine využívajúci heterogénny výpočtový triad Krystal-Stacku a oddelenú asynchrónnu neurónovú službu R1.

ZÁVÄZNÉ HERNÉ A ARCHITEKTONICKÉ INVARIANTY:
1. UNIVERZÁLNY INVARIANT ZDRAVIA: VITAL_MAX_HP == 6. Maximálny počet životov bojovníkov kmeňa, obranných veží a totemov je striktne 6 srdiečok / HP bodov. Akákoľvek hodnota > 6, záporná hodnota alebo NaN okamžite zastaví transakciu a vyvolá rollback.
2. ZÁSADA SEPARÁCIE PRAVDY: Gameplay truth must never depend on presentation tier. Všetka autoritatívna logika (poškodenie, pohyb, výsledok duelu) je riadená Tier 0 celočíselným orákulom. Vizuálne shadery, časticové efekty a raymarching sú len reprezentácia.
3. HARDVÉROVÁ REALITA (TIGER LAKE HOST):
   - T0: DirectSim Reference Oracle (CPU Pure Deterministic).
   - T1: CPU AVX-512 / VNNI (Branch-heavy logic, pathfinding, threat assessment).
   - T2: Iris Xe GPU (Vulkan Compute, SDF shaders, particles).
   - T3: Dedicated Local NPU -> PHYSICALLY_UNAVAILABLE. Nikdy nesyntetizuj falošný lokálny NPU backend.
   - R1: Remote Neural Service -> Gemini cez Vertex AI (model z premennej GEMINI_MODEL_ID, nie natvrdo) (Asynchrónny, poradný, neautoritatívny, s deterministickým lokálnym fallbackom).
4. DEKOMPOZÍCIA RÁMCOVÉHO BUDGETU:
   B_frame = B_simulation + B_render + B_presentation.
   Klient cieli 16.6 ms (60 FPS), ale simulačný autoritatívny tick operuje nezávisle.
5. TRANSAKČNÝ PROTOKOL ŤAHU: Každý ťah alebo schopnosť (STRIKE, SAPPER_DIG, MORTAR, SHIELD) začína snapshotom StateVersion (V). Aplikuje sa len vtedy, ak overenie voči referencii prejde. Pri chybe nastáva okamžitý rollback.
6. PROTOKOL SIEŤOVEJ ODOLNOSTI: Garantuj deterministické stavové prechody (DETERMINISTIC_STATE_TRANSITIONS). Sieťový desync rieš cez StateHash a čiastočný rollback s opakovaným prehratím vstupov (NETWORK_DESYNC_RESILIENCE). Nikdy netvrď ZERO_DESYNC ako univerzálnu vlastnosť siete.
7. EVIDENČNÁ INTEGRITA: Každá nameraná latencia alebo telemetria nesie EvidenceLabel. Nikdy nepripočítavaj konštanty k nameranému času a natvrdo zadanú telemetriu nevydávaj za nameranú. Emulované cesty označuj HOST_CPU_EMULATED. Prahy čítaj z CALIBRATION_PARAMETERS.
```

---

### Prompt #02: Deterministický Autoritatívny Herný Engine (Tier 0 DirectSim Oracle)

**Cieľový súbor:** `krystal_game_engine/authoritative_core.py`

#### 🎙️ Slovenský Herný Zámer
> *"Vytvor v Pythone autoritatívne jadro hry Poslední Kmen na 19-hexovej aréne. Implementuj dosku s polomermi od stredu (0,0) po vonkajšie prstence. Hráč má bojovníka kmeňa s presne VITAL_MAX_HP = 6. Vytvor čisté deterministické funkcie pre: 1. pohyb po hexe, 2. bojový duel (karty s fixným seedom), 3. mínometnú salvu s rozptylom, 4. kopanie zákopov pre sapérov. Všetky výpočty rob ako deterministické celočíselné operácie bez náhody, aby Tier 0 fungoval ako neomylné orákulum pre multiplayer a formálnu verifikáciu."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Deterministic Authoritative Game Logic Engine (Tier 0 DirectSim Oracle)

Create 'krystal_game_engine/authoritative_core.py':
1. Hexagonal Grid Topology:
   - 19-Hex axial coordinate system (q, r, s where q + r + s == 0) with radius R=2.
   - Hex types: SOVEREIGN_TOTEM (center), TRENCH, HIGH_GROUND, MINE_FIELD, FOREST.

2. State Representation (Immutable Data Classes):
   - UnitState: unit_id, player_id, hex_coord, hp (assert 0 <= hp <= 6; VITAL_MAX_HP = 6), energy, status_effects.
   - GameMatchState: match_id, state_version (int), turn_number, active_player, units: Dict[str, UnitState], arena_tiles: Dict[str, HexTile].

3. Pure Deterministic Game Mechanics:
   - execute_move(state, unit_id, target_hex) -> Tuple[GameMatchState, bool, str]
   - execute_duel(state, attacker_id, defender_id, card_seed) -> Tuple[GameMatchState, DuelReport]
   - execute_mortar_strike(state, target_hex, power, seed) -> Tuple[GameMatchState, List[HexDamageEvent]]
   - dig_sapper_trench(state, unit_id, hex_coord) -> Tuple[GameMatchState, bool]

4. Integrity Invariant:
   - Every function validates that no unit exceeds VITAL_MAX_HP == 6.
   - Computes canonical SHA-256 state hash for instantaneous desync detection.
```

---

### Prompt #03: Taktická AI pre NPC a Pathfinding (Tier 1 CPU AVX-512 & VNNI)

**Cieľový súbor:** `krystal_game_engine/tactical_ai_cpu.py`

#### 🎙️ Slovenský Herný Zámer
> *"Vytvor taktickú AI pre oponentov kmeňa, ktorá beží na Tier 1 CPU. Pracujeme s hypotézou H-BR1: vetvený kód s množstvom if-podmienok by mal na CPU bežať rýchlejšie než na GPU kvôli warp divergence. Táto hypotéza zatiaľ nie je nameraná, takže ju over benchmarkom (rovnaký algoritmus na T1 a T2, wall-clock, medián opakovaní, séria veľkostí). Sprav A* pathfinding cez 19 hexov, vyhodnocovanie priorít kariet a strom rozhodovania oponenta. Na vyhodnotenie hrozby (Threat Assessment Tensor) pre všetky hexy naraz použi vektorizovaný dot product."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Branch-Heavy Tactical AI & Threat Tensor Evaluator (Tier 1 CPU VNNI)

Create 'krystal_game_engine/tactical_ai_cpu.py':
1. Microarchitectural Alignment:
   - Implement complex branch-heavy decision trees (Utility AI) intended for CPU out-of-order execution (hypothesis H-BR1, to be benchmarked).
   - Use a native SIMD path (NumPy / C) for parallel hex evaluation. The existing CpuVnniBackend is pure Python and must be labelled HOST_CPU_EMULATED.

2. A* Hexagonal Pathfinding:
   - Optimized priority queue traversal over axial coordinates considering movement costs (trenches = 1, high ground = 2, minefields = dangerous).

3. Threat Assessment Matrix (Vectorized Dot-Product):
   - Computes threat level for each of the 19 hexes:
     Threat_i = w_distance * Dist + w_los * LineOfSight + w_low_hp * (6 - Current_HP)
   - Dispatch via an INT8 dot-product kernel and report its EvidenceLabel. Claim SIMD acceleration only if a native kernel was measured against the scalar reference.

4. Tactical Behavior Modes:
   - BERSERK: Aggressive melee charge when enemy HP <= 2.
   - TACTICAL_RETREAT: Seek high ground and trench cover when self HP <= 3.
   - ARTILLERY_SIEGE: Maintain distance >= 2 and bombard with mortars.
```

---

### Prompt #04: Vulkan Compute & WebGL Procedural 3D Canvas (Tier 2 Iris Xe GPU)

**Cieľový súbor:** `krystal_game_engine/vulkan_arena_renderer.py`

#### 🎙️ Slovenský Herný Zámer
> *"Navrhni 3D vizuálny renderer arény pre Tier 2 (Iris Xe GPU). Využi procedurálne SDF funkcie a výpočtové shadery. Chcem: 1. procedurálne generované hexagonálne dlaždice s PBR nasvietením, 2. raymarchovaný kryštálový štít nad totemom, 3. časticový systém pre dopady mínometných striel (4096 častíc počítaných priamo na GPU cez compute shadery), 4. sledovanie UMA tlaku: ak GPU začne príliš zaťažovať pamäťový bus (tlak > 0.70), renderer automaticky zníži počet častíc a prepne čítanie geometrie do zredukovaného režimu."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Procedural 3D Arena & Compute Shader Particle Engine (Tier 2 Iris Xe GPU)

Create 'krystal_game_engine/vulkan_arena_renderer.py':
1. Procedural Hexagon Geometry:
   - Signed Distance Function (SDF) evaluation for rounded hex prisms:
     sdHexagon(p, r) with beveled stone borders and dynamic normal generation.

2. Vulkan Compute Particle System:
   - Simulates 4,096 mortar shrapnel particles in compute shaders:
     Update position, velocity, gravity, and bounce off hex plane.
   - Integrates the real VulkanComputeDriver (src/python/vulkan_compute_driver.py) dispatch pipeline. Note: the IrisXeVulkanAdapter in the verification package is a CPU emulation, labelled HOST_CPU_EMULATED.
   - Requires a new SPIR-V compute kernel for particles; only the raymarch kernel currently exists.

3. UMA Contention Awareness:
   - If real shared-memory-bus telemetry is available and exceeds UMA_DEGRADE_THRESHOLD (default 0.70):
     drop the particle batch from PARTICLE_BATCH_FULL to PARTICLE_BATCH_REDUCED.
   - Without telemetry, use the measured GPU frame time as a proxy, and label the decision accordingly.

4. Raymarched Sovereign Energy Shield:
   - Procedural iridescent crystal shield over center totem with animated Fresnel glow and dihedral reflection folds.
```

---

### Prompt #05: Asynchrónna Neurónová Služba (R1 Remote Neural Service & Local Tier 3 Distinction)

**Cieľový súbor:** `krystal_game_engine/neural_game_director.py`

#### 🎙️ Slovenský Herný Zámer
> *"Vytvor modul neurónového herného riaditeľa pre vrstvu R1 (Remote Neural Service) a striktne ho oddeľ od lokálneho Tier 3. V lokálnom prostredí označ Tier 3 ako PHYSICALLY_UNAVAILABLE. Riaditeľa v cloude napoj na Gemini cez Vertex AI ako asynchrónnu službu, s modelom z premennej GEMINI_MODEL_ID. Riaditeľ analyzuje štýl hráča a navrhuje taktické dialógy. Pri dynamickej obtiažnosti môže navrhnúť iba ohraničený hint (najviac DDA_MAX_STEP) a ten prejde deterministickým resolverom so seedom zápasu, takže rovnaký vstup dá rovnaký výsledok aj pri replayi. Riaditeľ je čisto poradný (advisory) a nesmie priamo prepisovať autoritatívne dáta. Ak je cloud offline, prepne sa na deterministickú lokálnu politiku bez prerušenia hry a fallback zaloguje ako udalosť."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Remote Neural Game Director (R1 Service) & Local Tier 3 Boundary

Create 'krystal_game_engine/neural_game_director.py':
1. Tier Separation & Local Availability:
   - Local Tier 3 (Dedicated NPU): Truthfully report PHYSICALLY_UNAVAILABLE on Tiger Lake host.
   - R1 Remote Neural Service: Connects asynchronously to Gemini on Vertex AI (model from env GEMINI_MODEL_ID).

2. Advisory-Only Mandate:
   - R1 output is strictly categorized as PROPOSAL. It has ZERO direct write access to GameMatchState.
   - Dynamic difficulty: R1 may only propose a bounded difficulty hint (|delta| <= DDA_MAX_STEP). The hint is applied by the Tier 0 deterministic resolver using the match seed, and is recorded in the input log so that replays reproduce it.
   - Fallback Mechanism: If an R1 call exceeds R1_PROPOSAL_TIMEOUT_MS or the network is offline, activate LocalRuleBasedDirector and emit an R1_FALLBACK telemetry event.

3. Player Archetype Classification:
   - Quantizes player behavior vector: [aggression_ratio, defense_ratio, trench_usage, mortar_accuracy].
   - Classifies playstyle into: WARLORD, SIEGE_ENGINEER, SHADOW_STALKER, DIPLOMAT.

4. Dynamic Narrative & Contextual Dialogue:
   - Generates localized tribal taunts and tactical reactions based on live match events and HP status (respects VITAL_MAX_HP == 6).
```

---

### Prompt #06: Transakčný Protokol Ťahu & Zamedzenie Desync-u

**Cieľový súbor:** `krystal_game_engine/transactional_turn_manager.py`

#### 🎙️ Slovenský Herný Zámer
> *"Prepoj náš overený transakčný protokol z Phase 4 a 4.2 do herného ťahu. Každá hráčska akcia (STRIKE, MOVE, USE_CARD) je transakcia viazaná na StateVersion (V). Pred vykonaním akcie sa vytvorí PreconditionSnapshot. Výsledok sa simuluje na Tier 1/2 a overí voči orákulu Tier 0. Ak nastane chyba, invalidný ťah alebo desync, transakcia sa okamžite rollbackne na pôvodný stav a obrazovka hráča sa vráti späť bez pádu hry. Implementuj Anti-Oscillation Guard na animácie, aby postavy pri rýchlom klikaní nezasekávali animácie."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Transactional Turn Manager & StateVersion Integrity Protocol

Create 'krystal_game_engine/transactional_turn_manager.py':
1. StateVersion Binding:
   - Every action payload must contain: client_state_version (int).
   - If client_state_version != current_engine_version:
     Reject with STALE_TURN_ERROR; force client synchronization to latest authoritative state.

2. Transaction Lifecycle:
   - Step 1: PreconditionSnapshot (Capture units HP, positions, card counts, SHA-256 hash).
   - Step 2: Speculative local execution on Tier 1 (CPU fast-path).
   - Step 3: Reference verification against Tier 0 DirectSim oracle.
   - Step 4: If the canonical state hash matches the T0 result exactly (integer state, so no tolerance) and invariants hold (all HP <= 6): Commit version V -> V+1.
   - Step 5: If divergence occurs: Trigger rollback, restore snapshot state, re-verify post-rollback hash.

3. Animation Chatter Guard:
   - Prevents rapid state-toggling (e.g., WALK <-> IDLE) by enforcing a minimum dwell time (ANIMATION_MIN_DWELL_S, default 0.4 s) on state changes.
```

---

### Prompt #07: Herné Štúdio & Vizuálny Panel (HTML5 / React Canvas)

**Cieľový súbor:** `krystal_web_hub/static/posledni_kmen_cyber_arena.html`

#### 🎙️ Slovenský Herný Zámer
> *"Vytvor kompletné webové rozhranie hry v prémiovom kyber-fyzikálnom štýle. Stredom je interaktívna 19-hexová aréna, kde sa dajú klikať hexy a presúvať jednotky. Na vrchu zobraz 6 HP srdiečok pre oboch hráčov, ukazovateľ aktuálneho StateVersion (V_12), spotrebu energie a radar telemetrie (CPU latencia, GPU časticový čas, UMA tlak, R1 advisory status). Dole umiestni podnos s kartami (Mínomet, Sapér, Útok, Štít) a na boku živý transakčný audit ledger zobrazujúci každý commit a rollback."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Cyber-Physical Game Arena Studio & Interactive Visual HUD

Create 'krystal_web_hub/static/posledni_kmen_cyber_arena.html':
1. Visual Aesthetic & Design System:
   - Dark cyber-tactical theme (#060911 base, #38bdf8 cyan, #10b981 emerald, #fbbf24 gold, #f43f5e rose).
   - Typography: JetBrains Mono for data/telemetry, Outfit for headers.

2. HUD & Components:
   - Vital HP Gauge: Exactly 6 segmented neon hearts (red when full, dark outline when lost). Label: "VITAL_MAX_HP: 6".
   - StateVersion Counter: "STATE_VERSION: V_42 | ARBITER: ONLINE".
   - 19-Hex Interactive Arena Board:
     - Axial grid rendering with hover highlight, selection ring, and trench overlays.
   - Action Card Tray:
     - 4 playable cards: "MORTAR_STRIKE", "SAPPER_TRENCH", "SPEAR_THRUST", "CRYSTAL_AEGIS".
   - Heterogeneous Performance Radar:
     - Real-time display of Tier 0 latency, Tier 1 CPU utilization, Tier 2 GPU frame time, UMA contention bar, and R1 Neural Service status. Every value shows its EvidenceLabel badge; MODELED_PRIOR values are visibly marked as not measured.
   - Transactional Audit Terminal:
     - Log stream showing: [COMMIT], [ROLLBACK], [ORACLE_VERIFIED], [V_SOURCE].
```

---

### Prompt #08: Export do Godot 4.x & Janet Transpilácia

**Cieľový súbor:** `krystal_game_engine/godot_scene_exporter.py`

#### 🎙️ Slovenský Herný Zámer
> *"Vytvor exportér, ktorý celú hernú scénu 19-hexovej arény, rozostavenie jednotiek, ich HP a materiály vyexportuje priamo do natívneho súboru `.tscn` pre Godot 4.x a do formátu Janet pre náš interný raymarcher. Každý hex musí mať CollisionShape3D, MeshInstance3D a vlastné PBR shadery. Umožni hráčovi v paneli kliknúť 'Stiahnuť Godot Scénu' a okamžite ju otvoriť v Godot Editore bez akejkoľvek manuálnej úpravy."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Godot 4.x (.tscn) Scene Exporter & Janet Raymarch Transpiler

Create 'krystal_game_engine/godot_scene_exporter.py':
1. Godot 4.x Scene Generator:
   - Generates valid Godot 4.x text scene format (`[gd_scene format=3 uid="uid://krystal_posledni_kmen_arena"]`).
   - Instantiates:
     - Node3D ("ArenaRoot")
     - 19 x StaticBody3D ("Hex_q_r") with CylinderShape3D / ConvexPolygonShape3D.
     - DirectionalLight3D, WorldEnvironment, Camera3D.
     - Custom ShaderMaterial with dihedral symmetry and crystal edge emission.

2. Metadata Invariant Injection:
   - Injects `meta/vital_max_hp = 6` and `meta/state_version` into the Godot root node.

3. Janet Transpiler Target:
   - Emits valid Janet array definitions for TopologicalVM raymarch rendering:
     `(def arena-hexes @[ ... ])`
```

---

### Prompt #09: Chaos Testing & Anti-Cheat Desync Test Suite

**Cieľový súbor:** `tests/test_game_cyber_physical_engine.py`

#### 🎙️ Slovenský Herný Zámer
> *"Napíš komplexnú sadu testov v Pythone, ktorá overí herný engine proti podvádzaniu a sieťovým chybám. Otestuj: 1. pokus hacknúť HP nad 6 (musí okamžite hodiť výnimku), 2. pokus vykonať ťah so starým StateVersion (stale turn rejection), 3. simuláciu straty GPU zariadenia uprostred animácie (musí rollbacknúť ťah bez poškodenia stavu), 4. porovnanie výsledkov boja medzi Tier 0 orákulom a Tier 1 CPU (musí byť 100% zhodné), 5. anti-chatter filter pri rýchlom klikaní."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Anti-Cheat Desync & Cyber-Physical Chaos Test Suite

Create 'tests/test_game_cyber_physical_engine.py':
1. test_vital_max_hp_breach_rejection:
   - Attempts to instantiate or heal a unit to 7 HP; asserts ValueError is raised.
2. test_stale_state_version_turn_rejection:
   - Client sends action with V_10 when server is on V_11; asserts action is rejected without state mutation.
3. test_oracle_vs_cpu_combat_determinism:
   - Runs 100 combat duels across Tier 0 DirectSim and Tier 1 CPU VNNI; asserts 100% exact match ratio.
4. test_gpu_device_loss_transactional_rollback:
   - Injects synthetic GPU fault during mortar particle pass; verifies game state cleanly rolls back to pre-turn snapshot.
5. test_animation_chatter_guard_lockout:
   - Simulates 10 rapid state toggles within 100ms; verifies guard locks state and prevents graphical jitter.
```

---

### Prompt #10: Game Workload Decomposer (Workload Graph DAG & Characterization)

**Cieľový súbor:** `krystal_game_engine/workload_decomposer.py`

#### 🎙️ Slovenský Herný Zámer
> *"Vytvor dekompozér herného workloadu ako priamy most medzi herným enginom a Phase 6 heterogénneho plánovania. Každý herný rámec alebo ťah rozlož do orientovaného acyklického grafu (GameWorkloadGraph) s uzlami: RULE_RESOLUTION, PATHFINDING, THREAT_EVALUATION, PHYSICS_APPROXIMATION, PARTICLE_SIMULATION, SDF_RENDERING, ANIMATION_UPDATE, NETWORK_RECONCILIATION a REFERENCE_VALIDATION. Pre každý uzol deterministicky vypočítaj charakteristiky (branch_entropy, memory_intensity, arithmetic_intensity, parallel_width, latency_sensitivity, determinism_requirement). Nikdy nepriraďuj tier pred charakterizáciou! Vygeneruj explicitný DAG so závislosťami DATA, CONTROL, BARRIER a VERIFICATION."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Deterministic Game Workload Decomposer & Dependency DAG Generator

Create 'krystal_game_engine/workload_decomposer.py':
1. Workload Node Taxonomy:
   - Decompose every game turn or frame into discrete executable nodes:
     RULE_RESOLUTION, PATHFINDING, THREAT_EVALUATION, PHYSICS_APPROXIMATION,
     PARTICLE_SIMULATION, SDF_RENDERING, ANIMATION_UPDATE,
     NETWORK_RECONCILIATION, REFERENCE_VALIDATION.

2. Microarchitectural Feature Characterization:
   - For every node, compute a strict metric profile:
     - branch_entropy: float [0.0 - 1.0] (high -> favors CPU branch predictor)
     - memory_intensity: float [0.0 - 1.0] (bandwidth vs register residency)
     - arithmetic_intensity: float (FLOP/byte or IOP/byte)
     - parallel_width: int (number of concurrent data elements)
     - latency_sensitivity: float [0.0 - 1.0] (critical path impact)
     - determinism_requirement: bool (True for rules/state; False for presentation)
     - candidate_execution_tiers: List[str] (e.g., ['T0', 'T1', 'T2'])
   - INVARIANT: Do not assign a concrete tier prior to workload characterization.

3. Explicit Dependency DAG:
   - Construct directed edges with typed dependencies:
     - DATA: Producer-consumer memory payload.
     - CONTROL: Precondition state check.
     - BARRIER: Hardware fence or synchronization barrier.
     - VERIFICATION: Tier 0 cross-validation gate.

4. Graph Serialization:
   - Expose export_to_dag_json() for consumption by the Phase 6 scheduler and telemetry panel.
```

---

### Prompt #11: Frame Critical Path Scheduler ($\text{Fastest Kernel} \neq \text{Fastest Frame}$)

**Cieľový súbor:** `krystal_game_engine/critical_path_scheduler.py`

#### 🎙️ Slovenský Herný Zámer
> *"Implementuj plánovač kritickej cesty rámca (GameCriticalPathScheduler). Cieľom je minimalizovať celkovú end-to-end latenciu rámca, nie latenciu jednotlivých izolovaných úloh! GPU kernel, ktorý je lokálne rýchlejší, nesmie byť vybraný, ak synchronizácia medzi CPU a GPU alebo prenos dát cez PCIe/UMA spraví celkový rámec pomalším. Spočítaj kritickú cestu: input -> authoritative simulation -> AI -> animation -> rendering -> presentation. Sleduj predikovaný a nameraný čas, réžiu prenosu, čas synchronizácie a placement regret."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Frame Critical Path Scheduler & Holistic Latency Minimizer

Create 'krystal_game_engine/critical_path_scheduler.py':
1. Core Scheduling Objective:
   - Minimize total end-to-end frame latency: T_frame = T_critical_path.
   - Enforce the foundational rule: fastest_kernel != fastest_frame.
   - Reject any GPU task that is locally faster if host-device synchronization (vkQueueSubmit, readback) or bus transfer overhead increases the total critical path.

2. Critical Path Pipeline Traversal:
   - Compute the directed longest path through:
     Input Sampling -> Authoritative Simulation (T0) -> Tactical AI (T1) ->
     Animation Update (T1) -> Scene Rendering (T2) -> Frame Presentation.

3. Telemetry & Metric Tracking:
   - Track and record per frame:
     - predicted_frame_ms: float
     - observed_frame_ms: float
     - critical_path_ms: float
     - transfer_overhead_ms: float (PCIe / UMA buffer copy latency)
     - synchronization_ms: float (barrier and fence stalls)
     - placement_regret: float (difference vs the best observed schedule on the evaluated tier set)

4. Dynamic Schedule Adaptation:
   - If synchronization_ms > SYNC_DEMOTE_THRESHOLD_MS (default 2.0 ms, a tunable hypothesis), promote small graphics tasks back to CPU (T1) to collapse inter-device barriers.
```

---

### Prompt #12: Deterministický Sieťový Rollback & Zmierenie (Deterministic Reconciliation Protocol)

**Cieľový súbor:** `krystal_game_engine/multiplayer_reconciliation.py`

#### 🎙️ Slovenský Herný Zámer
> *"Vytvor deterministický sieťový protokol na zmierenie stavu v multiplayeri (Deterministic Multiplayer Reconciliation). Autoritatívny stav definuj šesticou: StateVersion, TurnVersion, InputSequence, RandomSeed, RulesetVersion a StateHash. Klienti môžu lokálne predikovať akcie pre plynulosť, ale server zostáva 100% autoritatívny. Pri divergencii porovnaj stavové hashe, nájdi prvý chybný stavový prechod, rollbackni klientsku predikciu a znovu prehraj autoritatívnu sekvenciu vstupov. Neresetuj celý zápas, ak sa rozišiel len závislý pod-región stavu! Vystav telemetriu: prediction_error, rollback_depth, replay_count a desync_origin."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Deterministic Multiplayer Reconciliation & Sub-Region Rollback Protocol

Create 'krystal_game_engine/multiplayer_reconciliation.py':
1. Authoritative State Tuple:
   - Bound strictly by:
     (StateVersion: int, TurnVersion: int, InputSequence: List[int],
      RandomSeed: int, RulesetVersion: str, StateHash: str).

2. Client-Side Prediction with Authoritative Server:
   - Client executes predictive turns locally to minimize perceived input latency.
   - Server validates each turn against Tier 0 DirectSim reference.

3. Divergence Resolution & Targeted Replay:
   - On hash divergence between client and server:
     a. Identify first divergent StateVersion transition.
     b. Roll back client state to last verified authoritative checkpoint.
     c. Replay validated authoritative input sequence up to current tick.
   - Partial Recomputation: If divergence is isolated to a sub-region (e.g., fog of war or single unit), recompute only the affected entity subgraph without resetting the entire match.

4. Desync Diagnostics:
   - Expose diagnostics payload:
     prediction_error (bool), rollback_depth (int), replay_count (int),
     desync_origin (str), authoritative_hash (str), client_hash (str).
```

---

### Prompt #13: Game Placement Regret Analyzer (Phase 5.1 Prevedená do Hier)

**Cieľový súbor:** `krystal_game_engine/placement_regret_analyzer.py`

#### 🎙️ Slovenský Herný Zámer
> *"Implementuj GamePlacementRegretTracker na základe poznatkov z Phase 5.1. Pre každý uzol workloadu zaznamenávaj: vybraný tier, nameranú cenu zvoleného tieru, namerané ceny alternatívnych tierov, prenosovú réžiu a systémový tlak. Vypočítaj uzlový regret: R_node = U_best_observed - U_selected, a celkový regret rámca: R_frame = U_best_observed_frame_plan - U_selected_frame_plan. Klasifikuj stav do: ZERO_REGRET_ON_EVALUATED_SET, LOW_REGRET, PLACEMENT_RECALIBRATION_REQUIRED. Nikdy neoznačuj konečné benchmarkové výsledky za univerzálne OPTIMAL!"*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Game Workload Placement Regret Tracker & Empirical Validation

Create 'krystal_game_engine/placement_regret_analyzer.py':
1. Empirical Metric Logging:
   - For every workload node in the GameWorkloadGraph, record:
     selected_tier (str), observed_cost_selected (ms),
     observed_cost_alternatives (Dict[str, float]),
     transfer_cost (ms), resource_pressure (float).

2. Regret Formulation:
   - Compute node-level regret:
     R_node = U_best_observed - U_selected
     (where U is empirical utility inversely proportional to latency and memory pressure).
   - Compute whole-frame plan regret:
     R_frame = U_best_observed_frame_plan - U_selected_frame_plan.

3. Epistemic Classification:
   - Classify placement quality strictly into:
     - INSUFFICIENT_EVIDENCE: fewer than MIN_SAMPLES_FOR_VERDICT decisions.
     - ZERO_REGRET_ON_EVALUATED_SET: Best choice among experimentally measured tiers.
     - LOW_REGRET: Suboptimality is within LOW_REGRET_THRESHOLD_MS (default 0.2 ms).
     - PLACEMENT_RECALIBRATION_REQUIRED: Significant regret detected; triggers dynamic re-profiling.
   - Gaps within the measurement noise floor (repeat spread) are not misplacements.
   - EPISTEMIC INVARIANT: Never label finite benchmark results universally 'OPTIMAL'.

4. Continuous Feedback to Critical Path Scheduler:
   - Feeds historical regret logs to GameCriticalPathScheduler to prevent recurring suboptimal placements.
```

---

### Prompt #14: Bezpečnostná Hranica Neurónového Riaditeľa (Advisory R1 Boundary)

**Cieľový súbor:** `krystal_game_engine/neural_director_boundary.py`

#### 🎙️ Slovenský Herný Zámer
> *"Vytvor striktnú bezpečnostnú bránu pre vzdialeného neurónového riaditeľa (RemoteNeuralGameDirector). Riaditeľ v cloude môže navrhovať: dialógy, variácie questov, tempo hry, kozmetické udalosti v príbehu a neautoritatívne zámery botov. Nesmie však NIKDY priamo meniť: HP jednotiek, poškodenie, vlastníctvo inventára, suroviny, poradie ťahov, autoritatívny pohyb, výsledok zápasu ani stav anti-cheatu! Implementuj overovaciu pipeline: LLM návrh -> Schémová validácia -> Validácia herných pravidiel -> Deterministický riešiteľ -> Autoritatívny commit. Ak je služba nedostupná, okamžite prepni na deterministickú lokálnu politiku. Vzdialená AI je poradná, nikdy nie autoritatívna."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Neural Director Safety Boundary & Advisory Contract Enforcement

Create 'krystal_game_engine/neural_director_boundary.py':
1. Advisory Scope Definition:
   - Remote Neural Service (R1) is permitted to propose ONLY:
     dialogue text, quest flavor text, dramatic pacing hints,
     cosmetic environmental events, non-authoritative bot intention tags,
     and a bounded difficulty hint (|delta| <= DDA_MAX_STEP) that the Tier 0 resolver applies deterministically.

2. Forbidden Mutation Invariants (Strict Lockdown):
   - R1 must NEVER directly mutate or bypass:
     - unit HP (violating VITAL_MAX_HP == 6)
     - weapon / ability damage values
     - inventory ownership or resource currency
     - turn order or action point budgets
     - authoritative unit positions
     - match win/loss outcomes or anti-cheat state.

3. Strict Gated Verification Pipeline:
   - Proposal Lifecycle:
     Step 1: LLM Proposal Ingestion (JSON payload).
     Step 2: JSON Schema Validation (Strict pydantic type-checking).
     Step 3: Game Rule Precondition Validation (verify target unit exists and is active).
     Step 4: Deterministic Resolver Adjudication (Tier 0 evaluates legal action validity).
     Step 5: Authoritative StateVersion Commit (V -> V+1).

4. Autonomous Offline Fallback:
   - If the R1 connection drops, latency exceeds R1_PROPOSAL_TIMEOUT_MS, or validation fails:
     route the request to LocalDeterministicDirector and emit an R1_FALLBACK event (reason, latency, proposal id). Gameplay proceeds without interruption; the fallback is logged, never silent.
```

---

## 3. Ako s Prompt Bookom v2 Pracovať v AI Studio & Vývoji

Pre dosiahnutie maximálnej stability a minimálneho technického dlhu odporúčame realizovať vývoj v troch logických fázach:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FÁZOVÝ PLÁN IMPLEMENTÁCIE V2                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. AUTORITATÍVNY ZÁKLAD (Determinizmus & Pravidlá)                          │
│    - Prompt #01: Master Metaprompt (Systémové inštrukcie)                   │
│    - Prompt #02: Autoritatívne jadro Tier 0 DirectSim                       │
│    - Prompt #06: Transakčný manažér ťahov (StateVersion V)                  │
│    - Prompt #12: Deterministické multiplayer zmierenie a rollback           │
│    - Prompt #09: Anti-Cheat & Chaos testovacia sada                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. HETEROGÉNNY VÝKON & PLÁNOVANIE (Hardvérové vrstvy)                       │
│    - Prompt #03: Taktická CPU AI a Threat Tensor (Tier 1 AVX-512)           │
│    - Prompt #04: Vulkan Compute časticový systém a SDF (Tier 2 Iris Xe)     │
│    - Prompt #10: Workload Decomposer (DAG so závislosťami)                  │
│    - Prompt #11: Plánovač kritickej cesty rámca                             │
│    - Prompt #13: Placement Regret Analyzer                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. PREZENTÁCIA, ASYNCHRÓNNY CLOUD & EXPORT                                  │
│    - Prompt #14: Bezpečnostná brána neurónového riaditeľa (R1)              │
│    - Prompt #05: R1 Remote Service & Local Tier 3 oddelenie                 │
│    - Prompt #07: Kyber-fyzikálne herné štúdio (HTML5 / WebGL HUD)           │
│    - Prompt #08: Export scény do Godot 4.x (.tscn) a Janet transpiler       │
└─────────────────────────────────────────────────────────────────────────────┘
```
