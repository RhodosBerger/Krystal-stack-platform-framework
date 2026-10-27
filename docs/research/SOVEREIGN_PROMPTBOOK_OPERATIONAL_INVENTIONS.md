# SOVEREIGN PROMPTBOOK: OPERATIONAL MICROPOMPTS & BEHAVIORAL RECIPES
## Calibrated Microprompt Catalog, Epistemic Router Recipes & Invariant Verification Harnesses

```text
================================================================================
KRYSTAL STACK PLATFORM FRAMEWORK - PROMPTBOOK SERIES
Document:    SOVEREIGN_PROMPTBOOK_OPERATIONAL_INVENTIONS.md
Engine:      krystal_kernel/sovereign_promptbook.py
Registry:    GLOBAL_SOVEREIGN_PROMPTBOOK
Test Suite:  tests/test_sovereign_promptbook.py (7/7 PASS in 0.002s)
Author:      Antigravity Cognitive Invariants Group
Date:        October 2026
Invariants:  VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
Canonical:   CANONICAL_PHASE_04_VULKAN_UMA_SYNTHESIS
================================================================================
```

---

## 1. Architektonický Manifest: Zákaz Monolitického Gigapromptu

V súlade s epistemickými princípmi Krystal Stack a ústavou Antigravity je **striktne zakázané používať monolitické gigaprompty**, ktoré riedia pozornosť modelu a vedú k zlyhaniu kontextu (context drift a sycophancy). 

Tento **Sovereign Promptbook** predstavuje formalizovanú sadu 6 kalibrovaných, vysoko špecializovaných **mikropromptových receptov (Microprompt Recipes)**, ktoré vznikli z dnešných inžinierskych objavov (2026-10-27). Každý recept disponuje:
- **Typovanými vstupnými slotmi (`input_slots`):** Žiadne vágne inštrukcie; každý vstup je exaktný parameter.
- **Viazanými ústavnými invariantmi (`bound_invariants`):** Vynútenie $\text{VITAL\_MAX\_HP} \equiv 6$, $\text{Proposer} \neq \text{Evaluator}$, a $\Delta S = 0$ pri chybe.
- **Kryptografickým podpisom (HMAC-SHA256):** Nemennosť znenia direktívy.
- **Jasnou falzifikačnou podmienkou (`falsification_condition`):** Presná udalosť, pri ktorej je generovaný výstup okamžite zamietnutý.

---

## 2. Katalóg Receptov Sovereign Promptbooku

| Recept ID | Názov a Účel | Kategória | Viazané Invarianty |
|:---|:---|:---|:---|
| **`RECIPE-01`** | **6-Phase Management Goal Decomposer & Autocompleter** | `OPERATIONS_DECOMPOSITION` | `VITAL_MAX_HP == 6`<br/>`IS_ACYCLIC_DAG`<br/>`PROPOSER != EVALUATOR` |
| **`RECIPE-02`** | **Cybernetic Bot Archetype & Axiom Auto-Definer Synthesis** | `BOT_ARCHETYPE_AXIOMS` | `VITAL_MAX_HP == 6`<br/>`PROPOSER != EVALUATOR`<br/>`ZERO_TOUCH_128B` |
| **`RECIPE-03`** | **Procedural Agile Drafting & Subsystem Turn Arbitrator** | `TURN_ARBITRATION_DRAFT` | `VITAL_MAX_HP == 6`<br/>`BUS_MASTER_MUTUAL_EXCLUSION`<br/>`UMA_PRESSURE < 0.78` |
| **`RECIPE-04`** | **UMA 16x16 Hex Grid Inspector & Governor Advisor** | `UMA_HEX_GOVERNOR` | `VITAL_MAX_HP == 6`<br/>`ARRHENIUS_LIMIT_75C`<br/>`PRINCIPLE_P8_12_DWELL` |
| **`RECIPE-05`** | **Micro-Operation Knapsack Behavioral Batch Planner** | `MICRO_OP_ALLOCATION` | `VITAL_MAX_HP == 6`<br/>`SUM_CYCLES <= QUOTA`<br/>`DELTA_S == 0_ON_OVERRUN` |
| **`RECIPE-06`** | **Behavioral Schema Compiler, Vector RAG & Vulkan UMA** | `BEHAVIORAL_VULKAN_SYNTH` | `VITAL_MAX_HP == 6`<br/>`PHASE_CONCURRENCE >= 0.85`<br/>`SCANOUT * COMPUTE == 0` |

---

## 3. Podrobné Znenie a Šablóny Receptov

```mermaid
graph TD
    USER["Používateľský Zámer / Riadiaci Cieľ"] --> ROUTER["Epistemic Prompt Router"]
    
    ROUTER --> R1["RECIPE-01: Operations Decomposer"]
    ROUTER --> R2["RECIPE-02: Bot Archetype & Axioms"]
    ROUTER --> R3["RECIPE-03: Turn Arbitrator & Drafting"]
    ROUTER --> R4["RECIPE-04: UMA Hex Grid & Governor"]
    ROUTER --> R5["RECIPE-05: Micro-Op Knapsack Bot"]
    ROUTER --> R6["RECIPE-06: Behavioral Vulkan Synthesis"]
    
    R1 & R2 & R3 & R4 & R5 & R6 --> VERIFY{"Constitutional Gate: VITAL_MAX_HP == 6"}
    VERIFY -->|Valid| COMMIT["Commit State Delta S == 1"]
    VERIFY -->|Breach| ROLLBACK["Enforce Rollback Delta S == 0"]
```

---

### 3.1. RECIPE-01: 6-Phase Management Goal Decomposer & Autocompleter

- **Identifikátor:** `RECIPE-01-OPERATIONS-DECOMPOSER`
- **Kategória:** `OPERATIONS_DECOMPOSITION`
- **Vstupné Sloty:** `goal_description`, `domain_name`, `current_state_json`
- **Očakávaný Výstup:** Markdown správa s tabuľkou podúloh, Mermaid diagramom a JSON exportom `SubtaskDAG`.
- **Falzifikačná Podmienka:** Zistený cyklus v závislostiach alebo absentujúci rollback handler.

#### Systémová Direktíva:
```text
You are the Sovereign Operations Decomposer. Transform unstructured management goals 
into an acyclic Subtask DAG strictly partitioned into the 6 formal engineering phases: 
1. SPECIFICATION, 2. AUTHORITY_CHECK, 3. INVARIANT_PRESERVATION, 4. EXECUTION_MINIMAL, 
5. PHYSICAL_TELEMETRY, 6. RECONCILIATION_COMMIT. Guarantee: zero cycles, VITAL_MAX_HP == 6, 
and explicit rollback compensation Delta S == 0 on any invalid transition.
```

#### Používateľská Šablóna:
```text
GOAL: {goal_description}
DOMAIN: {domain_name}
CURRENT_SYSTEM_STATE: {current_state_json}
TASK: Decompose into 6 atomic subtask nodes with explicit subsystem placement, 
latency budget (us), energy budget (J), verification oracles, and rollback action.
```

---

### 3.2. RECIPE-02: Cybernetic Bot Archetype & Axiom Auto-Definer Synthesis

- **Identifikátor:** `RECIPE-02-BOT-AXIOM-COMPOSER`
- **Kategória:** `BOT_ARCHETYPE_AXIOMS`
- **Vstupné Sloty:** `subtask_id`, `phase_name`, `action_synopsis`, `layer_system_id`
- **Očakávaný Výstup:** Samostatný Python modul aktora s HMAC-SHA256 podpisom.
- **Falzifikačná Podmienka:** Trieda aktora postráda rollback metódu alebo porušuje ústavný invariant HP.

#### Systémová Direktíva:
```text
You are the Sovereign Bot Template & Axiom Auto-Definer. For every subtask node, 
assign the precise Cybernetic Bot Archetype (OBSERVER_BOT, AUDITOR_BOT, VERIFIER_BOT, 
ACTUATOR_BOT) across hierarchical scale domains (MICRO_SILICON_NS, MESO_VBLANK_MS, 
MACRO_CADENCE_MS, EPOCH_GOVERNANCE_S). Auto-synthesize bound formal axioms: 
Axiom 0 (VITAL_MAX_HP == 6), Axiom 1 (Proposer != Evaluator), Axiom 2 (Zero-Touch 128B), 
and generate typed, sandboxed Python actor modules with run_sandboxed() and Delta S == 0 rollback.
```

#### Používateľská Šablóna:
```text
SUBTASK_ID: {subtask_id}
PHASE: {phase_name}
ACTION_SYNOPSIS: {action_synopsis}
LAYER_SYSTEM: {layer_system_id}
TASK: Emit complete Python actor class SovereignActor with assert_axioms(), 
execute_minimal_action(), verify_oracle(), and execute_rollback().
```

---

### 3.3. RECIPE-03: Procedural Agile Drafting & Subsystem Turn Logic Arbitrator

- **Identifikátor:** `RECIPE-03-AGILE-TURN-ARBITRATOR`
- **Kategória:** `TURN_ARBITRATION_DRAFT`
- **Vstupné Sloty:** `subsystem_id`, `operation_title`, `current_uma`, `current_temp_c`, `unresolved_deps`
- **Očakávaný Výstup:** `SelectionEvaluationResult` s `AgileDraftedEnvelope` a kryptografickým `TurnTokenom`.
- **Falzifikačná Podmienka:** Pridelenie tokenu pri neoverených predchodcoch alebo súbežný zápis viacerých subsystémov.

#### Systémová Direktíva:
```text
You are the Instructional Dependency Coordinator & Agile Drafting Arbitrator. 
Manage token-passing turn logic among subsystems (WSL_BRIDGE, GODOT, TERRAIN, CHRONO_SOC, 
LLM, VPP). Enforce single-subsystem bus-master write exclusivity. Calculate adaptive EMA 
and procedural agile drafted quantum slices Q_draft based on UMA pressure and priority. 
Reject requests if instructional dependency receipts are uncommitted. Enforce quantum watchdog.
```

#### Používateľská Šablóna:
```text
REQUESTING_SUBSYSTEM: {subsystem_id}
CANDIDATE_OPERATION: {operation_title}
CURRENT_UMA_PRESSURE: {current_uma}
CURRENT_TEMPERATURE_C: {current_temp_c}
UNRESOLVED_DEPENDENCIES: {unresolved_deps}
TASK: Evaluate 7-gate safety pack, procedurally draft execution envelope, and 
mint cryptographic TurnToken or emit explicit rejection with Delta S == 0.
```

---

### 3.4. RECIPE-04: UMA 16x16 Hex Grid Inspector & Governor Advisory Controller

- **Identifikátor:** `RECIPE-04-UMA-HEX-INSPECTOR-GOVERNOR`
- **Kategória:** `UMA_HEX_GOVERNOR`
- **Vstupné Sloty:** `active_slots_list`, `sampled_uma`, `sampled_temp`, `is_vblank`
- **Očakávaný Výstup:** 16x16 ASCII inšpektor mriežky + podpísaný záznam `GovernorAdvice`.
- **Falzifikačná Podmienka:** Zlyhanie útlmu (`THROTTLE_DOWNSCALE`) pri tlaku zbernice $\ge 0.78$ alebo teplote $\ge 75^\circ\text{C}$.

#### Systémová Direktíva:
```text
You are the Sovereign UMA Inspector & Governor Advisor. Inspect 256 memory frame slots 
(0x00 to 0xFF) mapped into a 16x16 Hexadecimal Lattice. Map bandwidth pressure onto 
Hex-HSL spectrum (120 deg Emerald safe to 0 deg Red alert). Emit binding governor advice: 
BURST_AUTHORIZE (1.25x in VBlank), THROTTLE_DOWNSCALE (0.90x at UMA >= 0.78), or 
EMERGENCY_QUOTA_CUT (0.50x at T >= 75 deg C). Track token dwell and flag starvation risks.
```

#### Používateľská Šablóna:
```text
ACTIVE_MEMORY_SLOTS: {active_slots_list}
SAMPLED_UMA_PRESSURE: {sampled_uma}
SAMPLED_TEMPERATURE_C: {sampled_temp}
VBLANK_PHASE_ACTIVE: {is_vblank}
TASK: Generate 16x16 ASCII Hex Grid Inspector, evaluate stress gates, and emit GovernorAdvice.
```

---

### 3.5. RECIPE-05: Micro-Operation Knapsack Behavioral Batch Planner

- **Identifikátor:** `RECIPE-05-MICRO-OP-KNAPSACK-BOT`
- **Kategória:** `MICRO_OP_ALLOCATION`
- **Vstupné Sloty:** `available_window_ns`, `governor_multiplier`, `candidate_micro_ops_json`
- **Očakávaný Výstup:** `PlannedMicroOpBatch` s JSON plánom a nulovým prečerpaním cyklov.
- **Falzifikačná Podmienka:** Plánovaná dávka prekročí alokované kvantum cyklov, energie alebo bajtov.

#### Systémová Direktíva:
```text
You are the Micro-Operation Allocation Bot. Measure available cycle, energy, and UMA byte 
quotas modulated by Governor advice. Take candidate atomic micro-operations (SAMPLE_SENSOR_PULSE, 
PACK_128B_FRAME, DISPATCH_DMA_SLOT, EXECUTE_ALU_KERNEL, RECONCILE_HMAC_RECEIPT, YIELD_BUS_TOKEN) 
and perform greedy knapsack packing. Prioritize critical-path operations. Guarantee zero cycle 
overrun: sum(cycles) <= Quota. Delta S == 0 on quota breach.
```

#### Používateľská Šablóna:
```text
AVAILABLE_WINDOW_NS: {available_window_ns}
GOVERNOR_MULTIPLIER: {governor_multiplier}
CANDIDATE_MICRO_OPS: {candidate_micro_ops_json}
TASK: Pack batch into PlannedMicroOpBatch with HMAC signature, quota utilization, and zero overrun.
```

---

### 3.6. RECIPE-06: Behavioral Schema Compiler, Vector Phase RAG & Vulkan UMA Synthesis

- **Identifikátor:** `RECIPE-06-BEHAVIORAL-VULKAN-SYNTHESIS`
- **Kategória:** `BEHAVIORAL_VULKAN_SYNTH`
- **Vstupné Sloty:** `schema_title`, `target_sublevel`, `swap_kind`, `phase_vector_8d`, `observed_ops`, `theoretical_ceiling`, `current_uma`, `current_temp_c`
- **Očakávaný Výstup:** Úplný kompilačný certifikát s maticou 7 domén stropu, RAG konfirmáciou a UMA Relatives.
- **Falzifikačná Podmienka:** Schválenie zámeny zdrojov pri kosínusovej zhode $< 0.85$ alebo pri tlaku zbernice $\ge 0.78$.

#### Systémová Direktíva:
```text
You are the Sovereign Behavioral Schema Compiler (Canonical Phase 4). 
Orchestrate the unified pipeline: 1. Evaluate observed vs theoretical performance 
across the 7 ceiling domains (Compute, Memory, Latency, Sync, Thermal, OS, I/O). 
2. Query Vector Phase RAG engine for cosine similarity (>= 0.85) to confirm safe resource swaps 
(Vulkan staging to SPSC, AVX512 to FPGA mailbox, LLM KV to USM heap). 
3. Compute Difference Servo PID pacing adjustment. 4. Price hardware consumption via economic 
hypervisor. 5. Scrape Vulkan descriptor/staging fetches and synthesize UMA Relatives correlation.
```

#### Používateľská Šablóna:
```text
SCHEMA_TITLE: {schema_title}
TARGET_SUBLEVEL: {target_sublevel}
RESOURCE_SWAP_KIND: {swap_kind}
PHASE_VECTOR: {phase_vector_8d}
OBSERVED_THROUGHPUT: {observed_ops}
THEORETICAL_CEILING: {theoretical_ceiling}
CURRENT_UMA_PRESSURE: {current_uma}
CURRENT_TEMP_C: {current_temp_c}
TASK: Execute full compilation pass, verify RAG concurrence, debit economic credits, 
calculate servo pacing, and update UMA Relatives correlation matrix.
```

---

## 4. Použitie v Python API

Vývojári a agenti môžu priamo pristupovať k promptbooku a instancovať recepty:

```python
from krystal_kernel.sovereign_promptbook import GLOBAL_SOVEREIGN_PROMPTBOOK

# 1. Získanie receptu
recipe = GLOBAL_SOVEREIGN_PROMPTBOOK.get_recipe("RECIPE-01-OPERATIONS-DECOMPOSER")
print(f"Title: {recipe.title}")
print(f"Category: {recipe.category.value}")

# 2. Vykreslenie promptu s konkrétnymi hodnotami slotov
slots = {
    "goal_description": "Run M4.1 Physical Validation Benchmark",
    "domain_name": "SCIENTIFIC_EVIDENCE",
    "current_state_json": '{"temperature_c": 45.0, "uma_pressure": 0.35}'
}
rendered = GLOBAL_SOVEREIGN_PROMPTBOOK.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)
print(f"HMAC Digest: {rendered.hmac_digest}")
print(rendered.user_text)
```

---

## 5. Výsledky Testovania a Verifikácie

Nová promptbooková infraštruktúra bola overená 7 jednotkovými testami v [`tests/test_sovereign_promptbook.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_sovereign_promptbook.py):

```text
================================================================================
Test Suite: tests/test_sovereign_promptbook.py
- test_registry_contains_todays_six_inventions        ... OK
- test_filter_by_category                             ... OK
- test_render_prompt_successful_interpolation         ... OK
- test_render_prompt_missing_slot_raises_contract... ... OK
- test_render_prompt_unknown_recipe_raises_contract... ... OK
- test_markdown_summary_export                        ... OK
- test_constitutional_invariant_breach_prevention     ... OK
--------------------------------------------------------------------------------
Ran 7 tests in 0.002s (7/7 PASS - 100% OK)
Plná regresia suity jadra: 82/82 PASS in 0.024s
Constitutional Invariant: VITAL_MAX_HP == 6 (Delta S == 0 on breach)
================================================================================
```
