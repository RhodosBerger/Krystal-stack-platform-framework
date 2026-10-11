# CrystalSec 3.0: Prompt Book pre Vývoj Hier na Heterogénnom Engine

**Autor:** Dušan Kopecký & Krystal-Stack Gaming & Cyber-Physical Systems Council  
**Klasifikácia:** Kniha Promptov pre AI Studio / Herný Vývoj / Heterogénna Herná Architektúra  
**Dátum:** Október 2026  
**Status:** SCHVÁLENÉ PRE GENERATÍVNY VÝVOJ HIER (AI STUDIO & VERTEX AI)  
**Univerzálny Invariant:** $\text{VITAL\_MAX\_HP} = 6$ (Bezpodmienečne Presadzované)  

---

## 1. Architektonická Vízia: Hra Novej Generácie

Na základe poznatkov z fáz 3 až 5.1 (Transactional Actuation, Sequence Arbiter, Heterogeneous Triad a Empirical Cost Calibration) už herný engine nemusí byť „čierna skrinka“, ktorá náhodne dropuje snímky alebo trpí sieťovým desync-om.

Tento Prompt Book definuje vývoj hry **Poslední Kmen: Sovereign Hex Arena** (taktická 19-hexová kyber-fyzikálna stratégia), postavenej na štyroch pilieroch:

$$\boxed{\text{Player Intent} \longrightarrow \text{Action Ranking (WSPT)} \longrightarrow \text{Heterogeneous Placement} \longrightarrow \text{Transactional Turn Commit}}$$

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       HETEROGÉNNY HERNÝ SUBSYSTÉM                          │
├───────────────────────┬─────────────────────────────────────────────────────┤
│ Tier 0: DirectSim     │ Deterministické herné pravidlá, autoritatívny arbiter│
│ Reference Oracle      │ bojov, nulový desync, rollback authority.            │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Tier 1: CPU AVX-512   │ Branch-heavy AI botov, A* hex-pathfinding, priestorové│
│ & VNNI Fast-Path      │ BVH kolízie, stavové automaty dialógov a questov.   │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Tier 2: Iris Xe / GPU │ Vulkan raymarching, časticové efekty, procedurálny  │
│ Vulkan Compute        │ SDF terén, 19-hex PBR shader, post-processing.      │
├───────────────────────┼─────────────────────────────────────────────────────┤
│ Tier 3: Cloud TPU /   │ Neurónové správanie oponentov, procedurálny príbeh, │
│ Dedicated NPU         │ adaptívna obtiažnosť a generovanie dialógov.        │
└───────────────────────┴─────────────────────────────────────────────────────┘
```

---

## 2. Kniha Promptov pre AI Studio

---

### Prompt #01: Master Herný Metaprompt & Systémové Inštrukcie

**Cieľ:** System Instructions v Google AI Studio alebo Claude / Cursor.

```markdown
ROLE:
Pôsobíš ako Senior Principal Game Engine Architect, Cyber-Physical Systems Engineer a Lead Gameplay Programmer pre hru "Poslední Kmen: Sovereign Hex Arena".

PRIMÁRNY CIEĽ:
Vytvor kompletnú, produkčne pripravenú hru a herný engine využívajúci heterogénny výpočtový triad Krystal-Stacku.

ZÁVÄZNÉ HERNÉ INVARIANTY:
1. UNIVERZÁLNY INVARIANT ZDRAVIA: VITAL_MAX_HP == 6. Maximálny počet životov bojovníkov kmeňa, obranných veží a totemov je striktne 6 srdiečok / HP bodov. Akákoľvek hodnota > 6 alebo NaN okamžite zastaví transakciu.
2. AUTORITATÍVNE TIER 0 ORÁKULUM: Všetka herná logika (poškodenie, pohyb na hexe, ťažba surovín, mínometné útoky) sa adjudikuje deterministickým matematickým modelom Tier 0. Žiadny float drift, žiadny sieťový desync medzi klientmi.
3. RÁMCOVÝ BUDGET (16.6 ms / 60 FPS): Úlohy hernej slučky sa triážujú podľa WSPT:
   rho_i = w_i / (p_i + C_migration).
   Ak frame time prekročí 13 ms, engine automaticky degraduje vizuálne častice (Tier 2), aby udržal input lag pod 16.6 ms.
4. TRANSAKČNÝ PROTOKOL ŤAHU: Každý ťah alebo schopnosť (STRIKE, SAPPER_DIG, MORTAR, SHIELD) začína snapshotom StateVersion (V). Aplikuje sa len vtedy, ak overenie voči referencii prejde. Pri chybe nastáva okamžitý rollback.
5. ANTI-CHATTER GUARD (TIER HYSTERESIS): Zamedz vizuálnemu záchvevu animácií (chatter) pri hraničných hodnotách; vyžaduj minimálny dwell time 0.5 s pred zmenou stavu.
```

---

### Prompt #02: Deterministický Autoritatívny Herný Engine (Tier 0)

**Cieľový súbor:** `krystal_game_engine/authoritative_core.py`

#### 🎙️ Slovenský Herný Zámer
> *"Vytvor v Pythone jadro hry Poslední Kmen na 19-hexovej aréne. Implementuj dosku s polomermi od středu (0,0) po vonkajšie prstence. Hráč má bojovníka kmeňa s presne VITAL_MAX_HP = 6. Vytvor čisté deterministické funkcie pre: 1. pohyb po hexe, 2. bojový duel (kocka/karty s fixným seedom), 3. mínometnú salvu s rozptylom, 4. kopanie zákopov pre sapérov. Všetky výpočty rob ako deterministické celočíselné alebo fixné float operácie bez náhody, aby Tier 0 fungoval ako neomylné orákulum pre multiplayer."*

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

### Prompt #03: Taktická AI pre NPC a Pathfinding (Tier 1 CPU AVX-512)

**Cieľový súbor:** `krystal_game_engine/tactical_ai_cpu.py`

#### 🎙️ Slovenský Herný Zámer
> *"Vytvor taktickú AI pre oponentov kmeňa, ktorá beží na Tier 1 CPU cez AVX-512 a VNNI. Naša výskumná fáza 5.1 dokázala, že vetvený kód s množstvom if-podmienok beží na CPU trikrát rýchlejšie než na GPU kvôli absencii warp divergence. Využi to! Sprav A* pathfinding cez 19 hexov, vyhodnocovanie priorít kariet a strom rozhodovania oponenta (či ísť liečiť HP, kopať zákop, alebo páliť z mínometu). Využi paralelný vektorizovaný dot product na vyhodnotenie hrozby (Threat Assessment Tensor) pre všetky hexy naraz."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Branch-Heavy Tactical AI & Threat Tensor Evaluator (Tier 1 CPU VNNI)

Create 'krystal_game_engine/tactical_ai_cpu.py':
1. Microarchitectural Alignment:
   - Implement complex branch-heavy decision trees (Utility AI) optimized for CPU out-of-order execution, avoiding GPU warp divergence.
   - Leverage CPU SIMD (AVX-512 / VNNI emulation) for parallel hex evaluation.

2. A* Hexagonal Pathfinding:
   - Optimized priority queue traversal over axial coordinates considering movement costs (trenches = 1, high ground = 2, minefields = dangerous).

3. Threat Assessment Matrix (Vectorized Dot-Product):
   - Computes threat level for each of the 19 hexes:
     Threat_i = w_distance * Dist + w_los * LineOfSight + w_low_hp * (6 - Current_HP)
   - Dispatches via CpuVnniBackend INT8 dot-product kernel to demonstrate real SIMD acceleration.

4. Tactical Behavior Modes:
   - BERSERK: Aggressive melee charge when enemy HP <= 2.
   - TACTICAL_RETREAT: Seek high ground and trench cover when self HP <= 3.
   - ARTILLERY_SIEGE: Maintain distance >= 2 and bombard with mortars.
```

---

### Prompt #04: Vulkan Compute & WebGL Procedural 3D Canvas (Tier 2 GPU)

**Cieľový súbor:** `krystal_game_engine/vulkan_arena_renderer.py`

#### 🎙️ Slovenský Herný Zámer
> *"Navrhni 3D vizuálny renderer arény pre Tier 2 (Iris Xe GPU). Využi procedurálne SDF funkcie a výpočtové shadery. Chcem: 1. procedurálne generované hexagonálne dlaždice s PBR nasvietením, 2. raymarchovaný kryštálový štít nad totemom, 3. časticový systém pre dopady mínometných striel (5 000 častíc počítaných priamo na GPU), 4. sledovanie UMA tlaku: ak GPU začne príliš zaťažovať pamäťový bus (tlak > 0.70), renderer automaticky zníži počet častíc a prepne čítanie geometrie do zredukovaného režimu."*

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
   - Integrates IrisXeVulkanAdapter compute dispatch pipeline.

3. UMA Contention Awareness:
   - Monitors shared memory bus pressure. If UMA pressure exceeds 0.70 (detected in Phase 5.1):
     Dynamically drops particle batch size from 4,096 to 1,024 to preserve CPU memory bandwidth.

4. Raymarched Sovereign Energy Shield:
   - Procedural iridescent crystal shield over center totem with animated Fresnel glow and dihedral reflection folds.
```

---

### Prompt #05: Neurónový Riaditeľ Hry & Reakčné Boty (Tier 3 TPU / NPU)

**Cieľový súbor:** `krystal_game_engine/neural_game_director.py`

#### 🎙️ Slovenský Herný Zámer
> *"Vytvor modul neurónového herného riaditeľa pre Tier 3. V lokálnom prostredí označ tento tier ako PHYSICALLY_UNAVAILABLE, ale v cloude ho napoj na Vertex AI / Cloud TPU v5e alebo Gemini 1.5 Flash. Riaditeľ analyzuje štýl hráča (či hrá defenzívne, agresívne alebo ekonomicky) a dynamicky prispôsobuje hru: generuje dialógy náčelníkov kmeňa priamo reagujúce na zranenia (napr. 'Tvoj totem má už iba 2 HP, vzdaj sa!'), upravuje balíček kariet a zabezpečuje, aby hra nebola ani nudná, ani nefér."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Neural Game Director & Adaptive Opponent Policy (Tier 3 TPU / NPU)

Create 'krystal_game_engine/neural_game_director.py':
1. Tier 3 Availability Handling:
   - Tiger Lake Host: Truthfully reports PHYSICALLY_UNAVAILABLE (no synthetic fake).
   - Cloud Environment: Connects to Vertex AI Cloud TPU v5e / Gemini 1.5 Flash.

2. Player Archetype Classification:
   - Quantizes player behavior vector: [aggression_ratio, defense_ratio, trench_usage, mortar_accuracy]
   - Classifies playstyle into: WARLORD, SIEGE_ENGINEER, SHADOW_STALKER, DIPLOMAT.

3. Dynamic In-Character Narrative Generation:
   - Prompts Gemini 1.5 Flash to generate localized tribal taunts and tactical reactions based on live match events and HP status (asserts VITAL_MAX_HP == 6).

4. Dynamic Difficulty Adjustment (DDA):
   - Modulates AI card draw probability to maintain target tension without breaking deterministic rules.
```

---

### Prompt #06: Transakčný Protokol Ťahu & Zamedzenie Desync-u

**Cieľový súbor:** `krystal_game_engine/transactional_turn_manager.py`

#### 🎙️ Slovenský Herný Zámer
> *"Prepoj náš overený transakčný protokol z Phase 4 a 4.2 do herného ťahu. Každá hráčska akcia (STRIKE, MOVE, USE_CARD) je transakcia viazaná na StateVersion (V). Pred vykonaním akcie sa vytvorí PreconditionSnapshot. Výsledok sa simuluje na Tier 1/2 a overí voči orákulu Tier 0. Ak nastane chyba, invalidný ťah alebo desync, transakcia sa okamžite rollbackne na pôvodný stav a obrazovka hráča sa vráti späť bez pádu hry. Implementuj Anti-Oscillation Guard na animácie, aby postavy pri rýchlom klikaní nezasekávali animácie."*

#### 🌐 Anglický Technický Prompt
```text
TITLE: Transactional Turn Manager & Zero-Desync State Protocol

Create 'krystal_game_engine/transactional_turn_manager.py':
1. StateVersion Binding:
   - Every action payload must contain: client_state_version (int).
   - If client_state_version != current_engine_version:
     Reject with STALE_TURN_ERROR; force client synchronization to latest authoritative state.

2. Transaction Lifecycle:
   - Step 1: PreconditionSnapshot (Capture units HP, positions, card counts, SHA-256 hash).
   - Step 2: Speculative local execution on Tier 1 (CPU fast-path).
   - Step 3: Reference verification against Tier 0 DirectSim oracle.
   - Step 4: If MAE == 0.0 and invariants hold (all HP <= 6): Commit version V -> V+1.
   - Step 5: If divergence occurs: Trigger rollback, restore snapshot state, re-verify post-rollback hash.

3. Animation Chatter Guard:
   - Prevents rapid state-toggling (e.g., WALK <-> IDLE) by enforcing minimum dwell time (0.4s) on state changes.
```

---

### Prompt #07: Herné Štúdio & Vizuálny Panel (HTML5 / React Canvas)

**Cieľový súbor:** `krystal_web_hub/static/posledni_kmen_cyber_arena.html`

#### 🎙️ Slovenský Herný Zámer
> *"Vytvor kompletné webové rozhranie hry v prémiovom kyber-fyzikálnom štýle. Stredom je interaktívna 19-hexová aréna (SVG alebo Three.js/WebGL), kde sa dajú klikať hexy a presúvať jednotky. Na vrchu zobraz 6 HP srdiečok pre oboch hráčov, ukazovateľ aktuálneho StateVersion (V_12), spotrebu energie a radar telemetrie (CPU latencia, GPU časticový čas, UMA tlak). Dole umiestni podnos s kartami (Mínomet, Sapér, Útok, Štít) a na boku živý transakčný audit ledger zobrazujúci každý commit a rollback."*

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
     - Real-time display of Tier 0 latency, Tier 1 CPU utilization, Tier 2 GPU frame time, and UMA contention bar.
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

## 3. Ako s Prompt Bookom pracovať v AI Studio

1. **Krok 1:** Vlož **Prompt #01** do systémových inštrukcií AI Studia.
2. **Krok 2:** Spusť **Prompt #02** (Autoritatívne Tier 0 jadro) a **Prompt #03** (CPU Taktická AI).
3. **Krok 3:** Spusť **Prompt #06** (Transakčný manažér ťahov) a **Prompt #04** (Vulkan 3D renderer).
4. **Krok 4:** Vygeneruj vizuálny frontend pomocou **Prompt #07** (HTML5 / React Canvas aréna).
5. **Krok 5:** Over integritu celej hry spustením **Prompt #09** (Anti-Cheat test suite).
