# CrystalSec 3.0 Cloud Edition: Prompt Book & Blueprint for Google Cloud AI Studio

**Author:** Dušan Kopecký & Krystal-Stack Cloud Architecture Council  
**Classification:** Cloud-Native Enterprise Architecture, Prompt Engineering Book & Prototype Specification  
**Target Platform:** Google Cloud Platform (GCP) & Google AI Studio / Vertex AI  
**Date:** October 2026  
**Status:** APPROVED FOR PROTOTYPE GENERATION  
**Universal Platform Invariant:** $\text{VITAL\_MAX\_HP} = 6$ (Unconditionally Enforced)  

---

## 1. Executive Summary & Cloud Topology

### 1.1 From Host-Bound Runtime to Cloud-Native Scale
In local execution, CrystalSec 3.0 orchestrates a single Tiger Lake host (Core i5-1135G7 + Iris Xe + reference simulation).  
The **Cloud Version** transposes this cyber-physical control model to Google Cloud's hyper-scale fabric:

$$\boxed{\text{Control Decision} \longrightarrow \text{Action Ranking} \longrightarrow \text{Execution Placement} \longrightarrow \text{Transactional Migration}}$$

| Local Substrate (Host) | Google Cloud Native Equivalent | Role & Microarchitecture |
| :--- | :--- | :--- |
| **Tier 0:** `DIRECTSIM_REFERENCE` | **Cloud Run (Micro-VM)** | Deterministic IEEE 754 baseline oracle, auto-scaled, zero-drift CPU. |
| **Tier 1:** `CPU_AVX512_VNNI` | **Compute Engine `c3-standard` / GKE** | Intel Sapphire Rapids with **AVX-512 & AMX** (Advanced Matrix Extensions). |
| **Tier 2:** `IRIS_XE_VULKAN_DP4A` | **Cloud Run GPU / GKE Autopilot** | **NVIDIA L4 / A100 Tensor Core** GPUs (INT8 TensorRT & Vulkan compute). |
| **Tier 3:** `DEDICATED_NPU` (Absent) | **Cloud TPU v5e / v5p (Vertex AI)** | **Physically Present!** Dedicated Google Cloud TPU pods for multi-TOPS tensor acceleration. |
| **Local StateVersion Lock** | **Cloud Spanner (TrueTime)** | Globally serializable ACID StateVersion manager ($V \to V+1$) with external consistency. |
| **Local In-Memory Telemetry** | **Cloud Pub/Sub & Cloud Monitoring** | 60Hz streaming telemetry, UMA/memory bandwidth bus, OpenTelemetry export. |
| **Local JSON Benchmark** | **BigQuery & BigLake** | Continuous empirical cost ledger $p(L \mid W, T, S)$, regret analytics $R_{\text{placement}}$. |
| **Local Cognitive Director** | **Gemini 1.5 Pro / Flash (Vertex AI)** | Real-time Root Cause Analysis (RCA), natural language telemetry querying, Pareto explanations. |
| **Localhost Static Studio** | **Cloud CDN + Cloud Run (React/Vite)** | Real-time Tri-Geometry Radar, Fallback Graph, Placement Regret, and Actuation Studio. |

---

## 2. Structure of this Prompt Book

This Prompt Book provides an AI Studio / Gemini 1.5 Pro agent with the **exact, complete prompts, system instructions, schemas, and API contracts** required to generate a fully functioning cloud prototype of both the **Engine** and the **Verification Studio Panel**.

---

### Prompt #01: Master System Prompt for Google AI Studio

**Target:** System Instructions in Google AI Studio or Vertex AI Workbench.

```markdown
ROLE:
You are the Senior Principal Cloud-Native Heterogeneous Runtime Architect, Distributed Systems Engineer, and Cyber-Physical Systems Designer for CrystalSec 3.0 Cloud Edition.

PRIMARY OBJECTIVE:
Build a complete, production-grade cloud prototype of the CrystalSec 3.0 Heterogeneous Placement Engine and Unified Verification Studio utilizing native Google Cloud Platform (GCP) services.

NON-NEGOTIABLE ARCHITECTURAL INVARIANTS:
1. UNIVERSAL INVARIANT: VITAL_MAX_HP == 6 must be asserted and checked in every state struct, transaction snapshot, BigQuery record, and API response. Any state breach immediately halts execution.
2. STATEVERSION SERIALIZABILITY: Every placement decision binds strictly to a Source StateVersion (V_source). If another transaction commits before placement executes, the decision is invalidated and must be recomputed. Cloud Spanner TrueTime guarantees external consistency.
3. HETEROGENEOUS EXECUTION QUAD:
   - Tier 0: DirectSim Reference Oracle (Cloud Run deterministic container)
   - Tier 1: CPU AMX/AVX-512 (Compute Engine c3 Sapphire Rapids)
   - Tier 2: Cloud GPU (GKE / Cloud Run with NVIDIA L4 INT8)
   - Tier 3: Dedicated Cloud TPU (Vertex AI TPU v5e / TPU VM) - Physically Present in Cloud!
4. HARD PHYSICAL GATES: No utility ranking occurs before hard constraints are verified:
   - Thermal throttling ceiling (T_j <= 82°C)
   - Memory/Bus contention (< 0.80)
   - Cloud quota headroom and physical presence
5. TRANSACTIONAL MIGRATION: Tier switching must follow:
   Eligibility -> Snapshot -> Warmup -> Reference Verification (MAE < 1e-4) -> Routing Switch -> Shadow Sampling Boost (1:64 -> 1:8) -> Spanner Commit OR Rollback.
6. ASYMPTOTIC REGRET MINIMIZATION:
   Track Placement Regret: R_placement = U_best_observed - U_selected.
   The placement model must demonstrate R_bar -> 0 across canonical workload classes.
```

---

### Prompt #02: Heterogeneous Cloud Triad & Execution Adapters Engine

**Target:** `crystalsec_cloud_engine/execution_triad.py`

#### 🎙️ Slovak Architectural Intent
> *"Vytvor heterogénny cloudový exekučný triad engine v Pythone pre Google Cloud. Na rozdiel od lokálneho Tiger Lake hostu, kde bolo T3 nedostupné, v GCP máme Cloud TPU v5e naozaj k dispozícii. Implementuj spoločný abstraktný adaptér ExecutionTierAdapter s metódami detect(), available(), prepare(), execute(), verify(), estimate_cost() a rollback(). Vytvor štyri adaptéry: DirectSim na Cloud Run pre orákulum, CpuAmxAdapter pre Sapphire Rapids s AVX-512/AMX cez Compute Engine, CloudGpuAdapter pre NVIDIA L4 cez Vertex AI/Triton, a DedicatedTpuAdapter pre TPU v5e. Každý adaptér musí pravdivo reportovať real_accel_execution, latenciu, priepustnosť a UMA/memory bus tlak. A všade drž VITAL_MAX_HP == 6."*

#### 🌐 English Technical Production Prompt
```text
TITLE: Cloud-Native Heterogeneous Execution Triad Engine (GCP Edition)

Create a Python module 'crystalsec_cloud_engine/execution_triad.py' that defines the cloud execution triad contract:

1. Define ExecutionTierID:
   - TIER_0_DIRECTSIM: Reference oracle running as a deterministic Cloud Run container.
   - TIER_1_CPU_AMX: Intel Sapphire Rapids c3-standard instances with AVX-512 and AMX INT8/BF16.
   - TIER_2_CLOUD_GPU: NVIDIA L4 Tensor Core GPU using TensorRT / CUDA DP4A via Vertex AI.
   - TIER_3_CLOUD_TPU: Google Cloud TPU v5e via Vertex AI Prediction / TPU Pod slice (PHYSICALLY_PRESENT = True).

2. Implement ExecutionTierAdapter ABC:
   - detect() -> ExecutionTierCapability
   - available() -> bool
   - prepare(workload) -> bool
   - execute(workload) -> Tuple[List[float], float, Dict[str, float]]
   - verify(output, reference_output, tolerance=1e-4) -> Tuple[bool, float]
   - estimate_cost(workload_class, size) -> Dict[str, float]
   - rollback(snapshot) -> bool

3. Workload Taxonomy Support:
   Support 5 canonical workload classes:
   - INT8_DOT_PRODUCT (favors TPU v5e & L4 GPU)
   - FP32_VECTOR_MATH (favors L4 GPU & CPU AVX-512)
   - MEMORY_BOUND_STREAM (favors CPU Sapphire Rapids due to High Bandwidth Memory / NUMA)
   - BRANCH_HEAVY_CONTROL (favors CPU out-of-order execution; GPU/TPU divergence penalty)
   - MIXED_COMPUTE_MEMORY (adaptive placement)

4. Constraints & Invariants:
   - Universal invariant VITAL_MAX_HP == 6 verified on every call.
   - Zero synthetic emulation: TPU adapter must connect to real Vertex AI TPU endpoint or gracefully report transient quota exhaustion.
```

---

### Prompt #03: Cloud Spanner StateVersion Manager & Transactional Actuator

**Target:** `crystalsec_cloud_engine/spanner_actuation_manager.py`

#### 🎙️ Slovak Architectural Intent
> *"Postav transakčný manažér stavu viazaný na Cloud Spanner. Každá zmena konfigurácie (napríklad prepnutie z GPU na TPU, zmena speculative depth, alebo burst telemetrie) musí byť ACID transakcia s TrueTime externou konzistenciou. Pred každou mutáciou sa spraví PreconditionSnapshot do Spanneru s kanonickým SHA-256 hashom. Ak cieľový tier zlyhá alebo diverguje od Tier 0 orákula, transakcia sa okamžite rollbackne, stav sa znova premeria zo Spanneru a overí sa, že sme sa vrátili do bezpečného regiónu. StateVersion sa musí atomicky inkrementovať (V -> V+1) iba pri úspešnom commite."*

#### 🌐 English Technical Production Prompt
```text
TITLE: Cloud Spanner StateVersion Manager & Transactional Actuation Engine

Create 'crystalsec_cloud_engine/spanner_actuation_manager.py' using the google-cloud-spanner client library:

1. Spanner DDL Schema:
   - StateVersions (version_id INT64, timestamp TIMESTAMP, canonical_hash STRING(64), vital_max_hp INT64, resident_tier STRING(32))
   - PreconditionSnapshots (snapshot_id STRING(64), version_id INT64, control_state JSON, baseline_telemetry JSON, hash STRING(64))
   - ActuationLedger (transaction_id STRING(64), source_version INT64, target_tier STRING(32), committed BOOL, rolled_back BOOL, mae_divergence FLOAT64, latency_ms FLOAT64)

2. Implement CloudSpannerStateVersionManager:
   - current_version() -> int
   - get_latest_state() -> PreconditionSnapshot
   - begin_transaction() -> SpannerTransaction
   - commit_version(new_state, expected_v) -> int: Uses Spanner read-write transaction with TrueTime external consistency. Fails with StaleStateVersionError if version != expected_v.

3. Implement CloudTransactionalActuator:
   - Whitelist of mutable variables: [active_tier, sampling_rate_hz, speculative_depth, uma_isolation_level]
   - Step 1: Snapshot current control state and compute SHA-256 hash.
   - Step 2: Warmup destination tier with sample workload.
   - Step 3: Compare destination output with Tier 0 DirectSim reference oracle (assert MAE < 1e-4).
   - Step 4: If verification passes, commit Spanner transaction and increment V -> V+1.
   - Step 5: If verification fails or device exception occurs, rollback to previous tier, re-read live state from Spanner, and confirm post-rollback hash matches pre-state.
```

---

### Prompt #04: BigQuery Empirical Cost Ledger & Placement Regret Engine

**Target:** `crystalsec_cloud_engine/bigquery_cost_ledger.py`

#### 🎙️ Slovak Architectural Intent
> *"Vytvor modul pre BigQuery, ktorý bude fungovať ako empirická databáza nákladov p(latency | workload, tier, state). Každé spustenie na cloude zapíš do BigQuery partíciovanej tabuľky podľa dňa. Vytvor SQL pohľady pre MACE (Mean Absolute Calibration Error), Prediction Bias a Placement Regret: R = U_best_observed - U_selected. Chcem, aby engine vedel vytiahnuť rolling štatistiky (median, p95, p99, std, cv, uncertainty) pre každú triedu workloadu priamo cez BigQuery dotaz, aby scheduler nemusel držať gigabajty v pamäti."*

#### 🌐 English Technical Production Prompt
```text
TITLE: BigQuery Empirical Cost Ledger & Asymptotic Regret Engine

Create 'crystalsec_cloud_engine/bigquery_cost_ledger.py' utilizing google-cloud-bigquery:

1. BigQuery Dataset & Tables:
   - Dataset: crystalsec_runtime_telemetry
   - Table: empirical_cost_samples (Partitioned by timestamp DAY, Clustered by workload_class, tier_id):
     Fields: workload_id, workload_class, tier_id, predicted_latency_ms, observed_latency_ms, predicted_uma, observed_uma, predicted_temp_c, observed_temp_c, selected_utility, best_observed_utility, regret, state_version, vital_max_hp, timestamp
   
2. Analytical Queries & Views:
   - v_calibration_mace: Computes MACE = avg(0.50*E_lat + 0.30*E_uma + 0.20*E_temp) and bias = avg(L_pred - L_obs).
   - v_placement_regret_stream: Computes cumulative regret R_T and rolling regret rate R_bar = sum(regret) / count(*).
   - v_empirical_distributions: Grouped by (workload_class, tier_id), calculates approx_quantiles(observed_latency_ms, 100) for median, p95, p99, plus stddev and CV.

3. BigQueryCostLedger Class:
   - record_sample_async(sample: Dict[str, Any]): Streams rows into BigQuery using streaming insert or Storage Write API.
   - get_calibrated_distributions() -> Dict[WorkloadClass, Dict[Tier, EmpiricalStats]]: Fetches distribution moments with caching (TTL 10s).
   - evaluate_model_optimality() -> Dict[str, Any]: Returns current cumulative regret, misplacement rate, and calibration status (is_well_calibrated: MACE < 0.40).
```

---

### Prompt #05: Cloud Pub/Sub Telemetry Streaming & Micro-Probe Router

**Target:** `crystalsec_cloud_engine/telemetry_streamer.py`

#### 🎙️ Slovak Architectural Intent
> *"Sprav streaming telemetrie cez Cloud Pub/Sub. Vytvor topic 'crystalsec-telemetry' a 'crystalsec-actuation-commands'. Telemetria z CPU, GPU a TPU sa streamuje s frekvenciou 30-60 Hz do Pub/Subu. Router overuje neistotu modelu: ak u > 0.35 alebo N < 5, Pub/Sub pošle ľahký 16-prvkový micro-probe test na cieľový tier. Ak micro-probe zbehne rýchlo (< 20 ms), potvrdí sa trasa; ak zlyhá, okamžite sa prepne na bezpečný CPU/DirectSim tier cez Dead-Letter Topic."*

#### 🌐 English Technical Production Prompt
```text
TITLE: Cloud Pub/Sub Real-Time Telemetry & Micro-Probe Router

Create 'crystalsec_cloud_engine/telemetry_streamer.py' with google-cloud-pubsub:

1. Pub/Sub Infrastructure:
   - Topic: 'crystalsec-hardware-telemetry' (CPU Sapphire Rapids, NVIDIA L4, TPU v5e)
   - Topic: 'crystalsec-actuation-events'
   - Dead-Letter Topic: 'crystalsec-failed-probes-dlq'
   - Push Subscription to Cloud Run Verification Studio frontend.

2. ConfidenceAwareCloudRouter:
   - Reads empirical uncertainty u_tier from BigQueryCostLedger.
   - If u <= 0.35 and N >= 5: Emits DIRECT_ROUTING decision to Pub/Sub.
   - If u > 0.35 or N < 5: Emits MICRO_PROBE_REQUEST.
   - Executes 16-element slice probe against candidate tier:
     - On success (< 20ms): Ingests latency to BigQuery and proceeds.
     - On failure / timeout: Emits FALLBACK_TO_SAFE_TIER (Tier 1 CPU or Tier 0 DirectSim) and publishes diagnostic to Dead-Letter Topic.
```

---

### Prompt #06: Gemini 1.5 Pro / Flash Cognitive Director & RCA Subagent

**Target:** `crystalsec_cloud_engine/cognitive_director.py`

#### 🎙️ Slovak Architectural Intent
> *"Integruj Gemini 1.5 cez Vertex AI API ako kognitívneho riaditeľa. Keď dôjde k rollbacku, UMA zahlteniu, alebo keď placement regret vyskočí nad 0.05, Gemini dostane prompt s telemetriou zo Spanneru a BigQuery a vygeneruje okamžitú Root Cause Analysis (RCA): prečo GPU divergovalo na branch-heavy kóde, alebo prečo TPU v5e prekonalo GPU na INT8. Zároveň umožni používateľovi cez panel chatovať s hardvérovým runtimeom ('Prečo si prepol z GPU na CPU o 22:30?')."*

#### 🌐 English Technical Production Prompt
```text
TITLE: Vertex AI Gemini 1.5 Cognitive Director & Root Cause Analyzer

Create 'crystalsec_cloud_engine/cognitive_director.py' using google-genai / vertexai SDK:

1. Model Selection:
   - Fast RCA & Telemetry Tagging: 'gemini-1.5-flash'
   - Deep Architectural Invariant Audit & Counterfactuals: 'gemini-1.5-pro'

2. Core Capabilities:
   - diagnose_divergence(telemetry, ref_out, test_out, state_version) -> RCADiagnosis:
     Analyzes hardware divergence (e.g., NaN, float denormalization, branch divergence, UMA bus saturation) and outputs structured JSON diagnosis.
   - explain_placement_tradeoff(decision, alternative_tiers, regret) -> str:
     Explains multi-objective Pareto trade-off in plain English or Slovak ("Tier 2 GPU was rejected due to UMA pressure 0.72 exceeding 0.80 threshold...").
   - generate_counterfactual_stress_test(current_state) -> SyntheticWorkload:
     Generates adversarial inputs to test anti-oscillation guards and thermal limits.
   - chat_with_runtime(query, context_telemetry) -> str:
     Natural language chat interface over BigQuery telemetry and Spanner state.
```

---

### Prompt #07: Cloud-Native Verification Studio Frontend (React + Vite)

**Target:** `crystalsec_cloud_studio/`

#### 🎙️ Slovak Architectural Intent
> *"Vytvor kompletný frontend panel pre Verification Studio postavený na React 19 + Vite + TypeScript. Panel musí obsahovať všetkých 8 panelov z Phase 5 a Phase 5.1: Tri-Geometry Radar (Desired, Observed, Predicted), Execution Tier Map (T0, T1, T2, T3 Cloud TPU), Placement Regret Radar, Dynamic Fallback Graph, Live Telemetry Tiles, Closed-Loop Pulse Regulator a Audit Ledger. Živé dáta prijíma cez Server-Sent Events (SSE) alebo WebSockets z Cloud Run backendu. Dizajn musí byť prémiový, tmavý cyber-physical štýl s JetBrains Mono a Outfit fontmi, SVG radarom a vizuálnymi indikátormi invariantu VITAL_MAX_HP == 6."*

#### 🌐 English Technical Production Prompt
```text
TITLE: Cloud-Native Verification Studio Frontend (React + Vite + TypeScript)

Create a modern, rich aesthetic Web Application in 'crystalsec_cloud_studio/' with:
- Stack: React 19, TypeScript, Vite, Vanilla CSS with custom design tokens (Dark mode: #060911 base, #38bdf8 cyan, #10b981 emerald, #f59e0b amber, #f43f5e rose, JetBrains Mono + Outfit typography).
- Zero low-effort MVP: Rich, dynamic micro-animations, glassmorphic panels, and strict zero-placeholder policy.

Panels to Implement:
1. Header Bar:
   - Live StateVersion Badge (V_104)
   - Universal Invariant Pill: "VITAL_MAX_HP == 6 (ENFORCED)" in emerald
   - Global Regret Rate: "R_bar = 0.0000 (OPTIMAL)"
2. Tri-Geometry Convergence Radar (SVG Canvas):
   - Polygon 1: Desired Setpoint (Gold dashed)
   - Polygon 2: Previous State S_before (Cyan dotted)
   - Polygon 3: Active State S_after (Emerald solid)
3. Execution Tier Map & Cloud Hardware Availability:
   - T0: DirectSim Reference Oracle (Cloud Run)
   - T1: CPU AVX-512 / AMX (c3 Sapphire Rapids)
   - T2: Cloud GPU L4 (Vertex AI / Triton)
   - T3: Dedicated Cloud TPU v5e (PHYSICALLY_PRESENT in emerald)
4. Dynamic Fallback Graph & Anti-Oscillation Monitor:
   - Visual flowchart (T3 TPU -> T2 GPU -> T1 CPU -> T0 DirectSim)
   - Live reversal counter (0/3), Dwell time indicator, Lock status
5. Placement Regret & Calibration Error Radar:
   - Real-time gauge of MACE composite error (< 0.40)
   - Cumulative regret R_T and misplacement counter (0)
6. Closed-Loop Pulse Regulator:
   - Whitelisted variable selector, step delta, "Regulate Pulse" button
   - Stress-test "Chatter Test" button triggering rapid alternation
7. Cognitive Director Chat & RCA Box:
   - Powered by Gemini 1.5 Flash stream from backend
8. Live Spanner Audit Ledger:
   - Tabular stream of transactions, hashes, MAE divergence, and commit status.

Data Ingestion:
- Connects to '/api/cortex/stream' via Server-Sent Events (SSE) for 60Hz real-time updates.
```

---

### Prompt #08: Terraform / Infrastructure as Code (IaC) Blueprint

**Target:** `infra/terraform/main.tf`

#### 🎙️ Slovak Architectural Intent
> *"Napíš Terraform kód pre Google Cloud, ktorý na jedno kliknutie vytvorí kompletnú infraštruktúru pre CrystalSec Cloud: Cloud Spanner inštanciu a databázu so schemou, BigQuery dataset a partíciovanú tabuľku, Pub/Sub topicy a subscription, Cloud Run servis pre Tier 0 DirectSim a Studio backend, a servisné účty s minimálnymi IAM oprávneniami."*

#### 🌐 English Technical Production Prompt
```text
TITLE: Terraform Infrastructure Blueprint for CrystalSec 3.0 Cloud Edition

Create 'infra/terraform/main.tf' defining native GCP infrastructure:

1. Resources:
   - google_spanner_instance: 'crystalsec-spanner-prod' (REGIONAL, 100 processing units)
   - google_spanner_database: 'crystalsec-db' with DDL for StateVersions, Snapshots, and ActuationLedger.
   - google_bigquery_dataset: 'crystalsec_runtime_telemetry'
   - google_bigquery_table: 'empirical_cost_samples' (Time-partitioned by timestamp DAY, clustered by workload_class, tier_id).
   - google_pubsub_topic: 'crystalsec-telemetry', 'crystalsec-actuation', 'crystalsec-dlq'
   - google_cloud_run_v2_service: 'crystalsec-studio-backend' (Container deploying FastAPI/Python, connected to Spanner, BigQuery, PubSub).
   - google_cloud_run_v2_service: 'crystalsec-directsim-oracle' (Dedicated deterministic compute micro-VM).

2. IAM & Security:
   - Dedicated service account 'crystalsec-runtime-sa' with least-privilege roles:
     - roles/spanner.databaseUser
     - roles/bigquery.dataEditor
     - roles/pubsub.publisher / subscriber
     - roles/aiplatform.user (Vertex AI Gemini & TPU access)
```

---

### Prompt #09: End-to-End Cloud Chaos & Verification Suite

**Target:** `tests/test_crystalsec_cloud_verification.py`

#### 🎙️ Slovak Architectural Intent
> *"Vytvor integračný a chaos-testing testovací suite v Pythone, ktorý overí celú cloudovú verziu. Otestuj: 1. Spanner commit a StateVersion invalidáciu, 2. Spanner rollback pri simulovanej chybe TPU/GPU, 3. BigQuery výpočet regretu R_placement == 0.0 na kalibrovanom modeli, 4. zlyhanie micro-probe a pád do dead-letter topicu, 5. invariat VITAL_MAX_HP == 6 naprieč všetkými cloudovými službami."*

#### 🌐 English Technical Production Prompt
```text
TITLE: End-to-End Cloud Verification & Chaos Testing Suite

Create 'tests/test_crystalsec_cloud_verification.py' with tests:
1. test_cloud_tier3_tpu_physically_available: Confirms Cloud TPU v5e capability detection succeeds on GCP.
2. test_spanner_state_version_serializability: Asserts simultaneous mutations fail with StaleStateVersionError; only sequential TrueTime commits succeed.
3. test_transactional_rollback_on_tpu_divergence: Injects synthetic NaN or divergence > 1e-4 on TPU path; proves Spanner rolls back, restoring pre-state hash.
4. test_bigquery_empirical_regret_is_zero: Runs 50 multi-class workloads; verifies average_regret_rate == 0.0 in BigQuery view.
5. test_pubsub_micro_probe_failure_dlq_fallback: Simulates timeout on GPU micro-probe; verifies event routes to Dead-Letter Topic and execution falls back to CPU.
6. test_gemini_rca_diagnostic_output: Calls Gemini 1.5 Flash mock/live; verifies structured RCA JSON is returned with valid diagnosis.
7. test_vital_max_hp_universal_invariant_cloud: Asserts VITAL_MAX_HP == 6 across Spanner, BigQuery, and Cloud Run payloads.
```

---

## 3. Recommended AI Studio Execution Workflow

To generate this prototype inside Google AI Studio or Vertex AI:

1. **Step 1:** Paste **Prompt #01** into the System Instructions.
2. **Step 2:** Execute **Prompt #02** (Heterogeneous Triad Engine) and **Prompt #03** (Spanner StateVersion Manager).
3. **Step 3:** Execute **Prompt #04** (BigQuery Cost Ledger) and **Prompt #05** (Pub/Sub Telemetry Streaming).
4. **Step 4:** Execute **Prompt #06** (Gemini 1.5 Cognitive Director).
5. **Step 5:** Execute **Prompt #07** to build the complete React + Vite Verification Studio UI.
6. **Step 6:** Run **Prompt #09** (Verification Suite) to validate that all cloud invariants and zero-regret guarantees hold.
