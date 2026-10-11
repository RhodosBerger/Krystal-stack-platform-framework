# CrystalSec 3.0 Cloud Edition: Prompt Book & Blueprint for Google Cloud AI Studio

**Author:** Dušan Kopecký & Krystal-Stack Cloud Architecture Council  
**Classification:** Cloud-Native Enterprise Architecture, Prompt Engineering Book & Prototype Specification  
**Target Platform:** Google Cloud Platform (GCP) & Google AI Studio / Vertex AI  
**Date:** October 2026  
**Status:** DRAFT v2 — revised 2026-10-11 after the Phase 5.2 Evidence Integrity Patch (T/R separation, no `OPTIMAL`, evidence labels)  
**Universal Platform Invariant:** $\text{VITAL\_MAX\_HP} = 6$ (Unconditionally Enforced)  

---

## 1. Executive Summary & Cloud Topology

### 1.1 From Host-Bound Runtime to Cloud-Native Scale
In local execution, CrystalSec 3.0 orchestrates a single Tiger Lake host (Core i5-1135G7 + Iris Xe + reference simulation).  
The **Cloud Version** transposes this cyber-physical control model to Google Cloud's hyper-scale fabric:

$$\boxed{\text{Control Decision} \longrightarrow \text{Action Ranking} \longrightarrow \text{Execution Placement} \longrightarrow \text{Transactional Migration}}$$

#### Local execution tiers (co-located on one execution node)
Placement tiers must share one memory domain, as T0–T2 do on Tiger Lake. Otherwise every "placement" also pays a network hop and the comparison with the local engine breaks. In the cloud, the execution node is **one G2 VM / GKE G2 node** (Intel Cascade Lake with AVX-512 VNNI, plus an NVIDIA L4).

| Local Substrate (Tiger Lake) | Cloud Execution Node Equivalent | Role & Microarchitecture |
| :--- | :--- | :--- |
| **T0:** `DIRECTSIM_REFERENCE` | Reference process on the G2 node | Deterministic IEEE 754 baseline oracle. |
| **T1:** `CPU_AVX512_VNNI` | Cascade Lake vCPUs of the same G2 node | AVX-512 VNNI INT8. (Alternative CPU-only profile: `c3-standard` Sapphire Rapids with AMX, without T2.) |
| **T2:** `IRIS_XE_VULKAN_DP4A` | NVIDIA L4 attached to the same node | INT8 via CUDA / TensorRT (or Vulkan compute). |
| **T3:** `DEDICATED_NPU` | **None** | `PHYSICALLY_UNAVAILABLE` — no local NPU is attached to the node. Never synthesize it. |

#### Remote asynchronous services (not placement tiers)
| ID | Google Cloud Service | Contract |
| :--- | :--- | :--- |
| **R1** | Gemini on Vertex AI (`GEMINI_MODEL_ID`, configurable) | Advisory only: RCA, explanations, chat. No write access to state or actuators. Timeout means a logged fallback. |
| **R2** | Vertex AI endpoint on Cloud TPU (optional) | Remote batch tensor service. Its latency includes a network RTT; it never runs on the frame or decision hot path. |

#### Control plane & data plane
| Local Substrate | Google Cloud Native Equivalent | Role |
| :--- | :--- | :--- |
| **Local StateVersion Lock** | **Cloud Spanner (TrueTime)** | Serializable ACID StateVersion ($V \to V+1$) for **migrations and actuations only**. Per-decision placement uses an in-process version counter, and only commits are mirrored to Spanner (a Spanner commit costs milliseconds). |
| **Local In-Memory Telemetry** | **Cloud Pub/Sub & Cloud Monitoring** | Telemetry fan-out and OpenTelemetry export. Pub/Sub is for events, not for synchronous probing. |
| **Local JSON Benchmark** | **BigQuery & BigLake** | Empirical cost ledger $p(L \mid W, T, S)$ and regret analytics $R_{\text{placement}}$, with evidence labels per row. |
| **Local RCA / explanation layer** | **R1 Gemini (Vertex AI)** | Root Cause Analysis, natural-language telemetry querying, Pareto explanations. |
| **Localhost Static Studio** | **Cloud CDN + Cloud Run (React/Vite)** | Tri-Geometry Radar, Fallback Graph, Placement Regret, Actuation Studio. |

### 1.2 Evidence labels (mandatory on every measurement)
Every latency or telemetry value in the API, BigQuery and the UI carries one label:
`HARDWARE_MEASURED` · `HOST_CPU_REFERENCE` · `HOST_CPU_EMULATED` · `CAPABILITY_DETECTED` · `MODELED_PRIOR` · `UNAVAILABLE` · `PHYSICALLY_UNAVAILABLE`.
A value may be labelled `HARDWARE_MEASURED` only if it is the wall-clock or counter reading of code that actually ran on that device. Nothing may be added to a measured value.

### 1.3 Shared calibration parameters
These are tunable hypotheses, not constants of nature. They must be read from a single config.

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `MIN_SAMPLES_FOR_VERDICT` | 10 | Below this, the regret and calibration verdict is `INSUFFICIENT_EVIDENCE` |
| `MACE_TOLERANCE` | 0.30 | Composite calibration error tolerance (latency only while telemetry is `MODELED_PRIOR`) |
| `LOW_REGRET_RATE` | 0.05 | Noise-adjusted regret rate below which the verdict is `LOW_REGRET` |
| `UNCERTAINTY_PROBE_THRESHOLD` / `MIN_SAMPLES_DIRECT` | 0.35 / 5 | Router probes when $u > 0.35$ or $N < 5$ |
| `MICRO_PROBE_BUDGET_MS` | 20 | In-process micro-probe budget |
| `MIGRATION_MAE_TOLERANCE` | 1e-4 | Destination vs T0 reference |
| `R1_TIMEOUT_MS` | 2000 | Advisory Gemini call timeout (never blocks the control loop) |

---

## 2. Structure of this Prompt Book

This Prompt Book gives an AI Studio / Gemini agent the prompts, system instructions, schemas and API contracts needed to generate a cloud prototype of both the **Engine** and the **Verification Studio Panel**. It does not guarantee results: every performance claim the prototype makes must be backed by evidence-labelled measurements.

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
3. LOCAL EXECUTION TIERS (co-located on one G2 execution node) vs REMOTE SERVICES:
   - T0: DirectSim Reference Oracle (deterministic process on the execution node)
   - T1: CPU AVX-512 VNNI (Cascade Lake vCPUs of the same node)
   - T2: NVIDIA L4 attached to the same node (INT8)
   - T3: Dedicated local NPU -> PHYSICALLY_UNAVAILABLE. Never synthesize it.
   - R1: Gemini on Vertex AI (model ID from env GEMINI_MODEL_ID) -> advisory, asynchronous, no write access.
   - R2: Optional Vertex AI TPU endpoint -> remote batch service; never on the decision hot path.
4. HARD PHYSICAL GATES: No utility ranking occurs before hard constraints are verified:
   - Thermal throttling ceiling (T_j <= 82°C) where a real sensor exists; otherwise the gate is reported as UNAVAILABLE, not as passed
   - Memory/Bus contention (< 0.80)
   - Device presence and StateVersion match
5. TRANSACTIONAL MIGRATION: Tier switching must follow:
   Eligibility -> Snapshot -> Warmup -> Reference Verification (MAE < 1e-4) -> Routing Switch -> Shadow Sampling Boost (1:64 -> 1:8) -> Commit (mirrored to Spanner) OR Rollback.
6. EVIDENCE INTEGRITY:
   - Every latency or telemetry value carries an EvidenceLabel (HARDWARE_MEASURED, HOST_CPU_REFERENCE, HOST_CPU_EMULATED, CAPABILITY_DETECTED, MODELED_PRIOR, UNAVAILABLE, PHYSICALLY_UNAVAILABLE).
   - Never add constants to measured latencies. Never hardcode telemetry and present it as measured.
   - Predict before observing (no leakage of the observation into the prediction).
7. PLACEMENT REGRET (finite-sample honesty):
   Track R_placement = U_best_observed - U_selected over the evaluated tier set, with a measurement noise floor.
   Allowed verdicts: INSUFFICIENT_EVIDENCE (< 10 decisions), ZERO_REGRET_ON_EVALUATED_SET, LOW_REGRET, PLACEMENT_RECALIBRATION_REQUIRED.
   Never emit or display "OPTIMAL".
```

---

### Prompt #02: Heterogeneous Cloud Triad & Execution Adapters Engine

**Target:** `crystalsec_cloud_engine/execution_triad.py`

#### 🎙️ Slovak Architectural Intent
> *"Vytvor heterogénny exekučný triad engine v Pythone pre jeden G2 exekučný uzol v Google Cloude. T0 (DirectSim), T1 (AVX-512 VNNI) a T2 (NVIDIA L4) bežia na tom istom uzle, aby placement neplatil sieťový hop. T3 lokálne NPU nie je k dispozícii, takže pravdivo reportuj PHYSICALLY_UNAVAILABLE. Cloud TPU nie je T3, je to vzdialená služba R2 mimo hot path. Implementuj spoločný abstraktný adaptér ExecutionTierAdapter s metódami detect(), available(), prepare(), execute(), verify(), estimate_cost() a rollback(). Každá nameraná hodnota musí mať EvidenceLabel. Emulované cesty označ HOST_CPU_EMULATED a nikdy k nameranej latencii nepripočítavaj konštanty. A všade drž VITAL_MAX_HP == 6."*

#### 🌐 English Technical Production Prompt
```text
TITLE: Cloud-Native Heterogeneous Execution Triad Engine (GCP Edition)

Create a Python module 'crystalsec_cloud_engine/execution_triad.py' that defines the cloud execution triad contract for ONE co-located G2 execution node:

1. Define ExecutionTierID (local, same memory domain):
   - TIER_0_DIRECTSIM: Deterministic reference oracle process on the node.
   - TIER_1_CPU_AVX512_VNNI: Cascade Lake vCPUs of the node (AVX-512 VNNI INT8), native code path (NumPy / C), not pure Python.
   - TIER_2_CLOUD_GPU: NVIDIA L4 attached to the node (CUDA / TensorRT INT8).
   - TIER_3_DEDICATED_NPU: PHYSICALLY_UNAVAILABLE (no NPU attached). The adapter exists only to report this truthfully.
   Define RemoteServiceID separately (never part of placement ranking):
   - R1_GEMINI: Vertex AI Gemini, model from env GEMINI_MODEL_ID.
   - R2_TPU_ENDPOINT: optional Vertex AI endpoint on Cloud TPU, asynchronous batch only.

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
   - INT8_DOT_PRODUCT (hypothesis: L4 GPU at large N, CPU VNNI at small N)
   - FP32_VECTOR_MATH (hypothesis: L4 GPU at large N)
   - MEMORY_BOUND_STREAM (hypothesis: CPU, due to PCIe transfer cost to the L4)
   - BRANCH_HEAVY_CONTROL (hypothesis H-BR1: CPU out-of-order; SIMT divergence penalty)
   - MIXED_COMPUTE_MEMORY (adaptive placement)
   These are hypotheses to be confirmed by a workload-size sweep, not routing rules.

4. Constraints & Invariants:
   - Universal invariant VITAL_MAX_HP == 6 verified on every call.
   - Every execute() returns its EvidenceLabel. Emulated paths are labelled HOST_CPU_EMULATED; HARDWARE_MEASURED only if the code ran on that device.
   - Measured latency = wall-clock (median of repeats, warm-up excluded). Nothing is added to it.
   - Include a mutation test: injecting a delay into an adapter must change its measured latency.
```

---

### Prompt #03: Cloud Spanner StateVersion Manager & Transactional Actuator

**Target:** `crystalsec_cloud_engine/spanner_actuation_manager.py`

#### 🎙️ Slovak Architectural Intent
> *"Postav transakčný manažér stavu viazaný na Cloud Spanner. Každá zmena konfigurácie (napríklad prepnutie z GPU na CPU, zmena speculative depth, alebo burst telemetrie) musí byť ACID transakcia s TrueTime externou konzistenciou. Pred každou mutáciou sa spraví PreconditionSnapshot do Spanneru s kanonickým SHA-256 hashom. Ak cieľový tier zlyhá alebo diverguje od Tier 0 orákula, transakcia sa okamžite rollbackne, stav sa znova premeria zo Spanneru a overí sa, že sme sa vrátili do bezpečného regiónu. StateVersion sa musí atomicky inkrementovať (V -> V+1) iba pri úspešnom commite."*

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
TITLE: BigQuery Empirical Cost Ledger & Evidence-Gated Regret Engine

Create 'crystalsec_cloud_engine/bigquery_cost_ledger.py' utilizing google-cloud-bigquery:

1. BigQuery Dataset & Tables:
   - Dataset: crystalsec_runtime_telemetry
   - Table: empirical_cost_samples (Partitioned by timestamp DAY, Clustered by workload_class, tier_id):
     Fields: workload_id, workload_class, workload_size, tier_id, predicted_latency_ms, observed_latency_ms, latency_evidence, predicted_uma, observed_uma, predicted_temp_c, observed_temp_c, telemetry_evidence, selected_utility, best_observed_utility, regret, within_noise_floor BOOL, state_version, vital_max_hp, timestamp
   
2. Analytical Queries & Views:
   - v_calibration_mace: Computes MACE = avg(0.50*E_lat + 0.30*E_uma + 0.20*E_temp) only over rows where telemetry_evidence = 'HARDWARE_MEASURED'; otherwise MACE = E_lat. Also bias = avg(L_pred - L_obs).
   - v_placement_regret_stream: Computes cumulative regret R_T, rolling regret rate R_bar = sum(regret) / count(*), noise-adjusted regret rate (excluding within_noise_floor rows), and selected vs best tier distributions.
   - v_empirical_distributions: Grouped by (workload_class, tier_id), calculates approx_quantiles(observed_latency_ms, 100) for median, p95, p99, plus stddev and CV.

3. BigQueryCostLedger Class:
   - record_sample_async(sample: Dict[str, Any]): Streams rows into BigQuery using streaming insert or Storage Write API.
   - get_calibrated_distributions() -> Dict[WorkloadClass, Dict[Tier, EmpiricalStats]]: Fetches distribution moments with caching (TTL 10s).
   - evaluate_placement_quality() -> Dict[str, Any]: Returns cumulative regret, noise-adjusted regret rate, misplacement rate, placement_quality (INSUFFICIENT_EVIDENCE | ZERO_REGRET_ON_EVALUATED_SET | LOW_REGRET | PLACEMENT_RECALIBRATION_REQUIRED, never OPTIMAL), calibration_status (MACE < MACE_TOLERANCE = 0.30), evaluated_tiers and a scope_note.
```

---

### Prompt #05: Cloud Pub/Sub Telemetry Streaming & Micro-Probe Router

**Target:** `crystalsec_cloud_engine/telemetry_streamer.py`

#### 🎙️ Slovak Architectural Intent
> *"Sprav streaming telemetrie cez Cloud Pub/Sub. Vytvor topic 'crystalsec-telemetry' a 'crystalsec-actuation-commands'. Telemetria z CPU a GPU exekučného uzla sa agreguje lokálne a do Pub/Subu sa posiela v dávkach (napr. 1–5 Hz), každá hodnota s EvidenceLabel. Router overuje neistotu modelu: ak u > 0.35 alebo N < 5, spustí ľahký 16-prvkový micro-probe priamo v procese na exekučnom uzle, nie cez Pub/Sub, pretože sieťový RTT by rozpočet 20 ms znehodnotil. Ak probe zlyhá, router prejde na bezpečný T1/T0 tier a do Dead-Letter Topicu publikuje iba diagnostickú udalosť."*

#### 🌐 English Technical Production Prompt
```text
TITLE: Cloud Pub/Sub Real-Time Telemetry & Micro-Probe Router

Create 'crystalsec_cloud_engine/telemetry_streamer.py' with google-cloud-pubsub:

1. Pub/Sub Infrastructure:
   - Topic: 'crystalsec-hardware-telemetry' (execution-node CPU + L4; batched 1–5 Hz; every value carries an EvidenceLabel)
   - Topic: 'crystalsec-actuation-events'
   - Dead-Letter Topic: 'crystalsec-failed-probes-dlq'
   - Push Subscription to the Cloud Run Studio BACKEND, which fans out to browsers via SSE.

2. ConfidenceAwareCloudRouter (runs in-process on the execution node):
   - Reads empirical uncertainty u_tier from a local cache, periodically refreshed from BigQueryCostLedger. It never queries BigQuery per decision.
   - Makes the prediction BEFORE executing, and records the observation afterwards.
   - If u <= 0.35 and N >= 5: DIRECT_ROUTING.
   - If u > 0.35 or N < 5: executes a real 16-element slice probe in-process on the candidate tier:
     - On success (< MICRO_PROBE_BUDGET_MS): records the probe latency (labelled) and proceeds.
     - On failure / timeout: FALLBACK_TO_SAFE_TIER (T1 or T0), and publishes the diagnostic event to the Dead-Letter Topic.
   - Publishes routing decisions as events to Pub/Sub (asynchronously, fire-and-forget).
```

---

### Prompt #06: R1 Gemini Cognitive Director & RCA Subagent (advisory)

**Target:** `crystalsec_cloud_engine/cognitive_director.py`

#### 🎙️ Slovak Architectural Intent
> *"Integruj Gemini cez Vertex AI ako službu R1. Model sa vyberá cez premennú GEMINI_MODEL_ID, nie natvrdo v kóde. Keď dôjde k rollbacku, UMA zahlteniu, alebo keď noise-adjusted placement regret presiahne LOW_REGRET_RATE, Gemini dostane prompt s telemetriou zo Spanneru a BigQuery (vrátane EvidenceLabel) a vygeneruje Root Cause Analysis. RCA je hypotéza, ktorá musí citovať konkrétne riadky telemetrie a nesmie tvrdiť nič, čo dáta nepodporujú. R1 nemá žiadny zápisový prístup k stavu ani k aktuátorom. Zároveň umožni používateľovi cez panel chatovať s runtimeom ('Prečo si prepol z GPU na CPU o 22:30?')."*

#### 🌐 English Technical Production Prompt
```text
TITLE: R1 Vertex AI Gemini Cognitive Director & Root Cause Analyzer (advisory only)

Create 'crystalsec_cloud_engine/cognitive_director.py' using the google-genai SDK (Vertex AI backend):

1. Model Selection (configuration, never hardcoded):
   - Fast RCA & Telemetry Tagging: env GEMINI_FAST_MODEL_ID
   - Deep Architectural Invariant Audit & Counterfactuals: env GEMINI_DEEP_MODEL_ID
   - Pick current model IDs from the Vertex AI model list at deploy time. Calls are asynchronous, with timeout R1_TIMEOUT_MS; on timeout, log an R1_FALLBACK event and return a rule-based summary.
   - R1 output is a PROPOSAL / EXPLANATION. It has zero write access to Spanner state or to the actuator whitelist.

2. Core Capabilities:
   - diagnose_divergence(telemetry, ref_out, test_out, state_version) -> RCADiagnosis:
     Analyzes hardware divergence (e.g., NaN, float denormalization, branch divergence, UMA bus saturation) and outputs structured JSON diagnosis.
   - explain_placement_tradeoff(decision, alternative_tiers, regret) -> str:
     Explains multi-objective Pareto trade-off in plain English or Slovak, citing only logged values and their evidence labels ("T2 was rejected: UMA pressure 0.84 [HARDWARE_MEASURED] exceeded the 0.80 gate...").
   - generate_counterfactual_stress_test(current_state) -> SyntheticWorkload:
     Generates adversarial inputs to test anti-oscillation guards and thermal limits.
   - chat_with_runtime(query, context_telemetry) -> str:
     Natural language chat interface over BigQuery telemetry and Spanner state.
```

---

### Prompt #07: Cloud-Native Verification Studio Frontend (React + Vite)

**Target:** `crystalsec_cloud_studio/`

#### 🎙️ Slovak Architectural Intent
> *"Vytvor kompletný frontend panel pre Verification Studio postavený na React 19 + Vite + TypeScript. Panel musí obsahovať všetkých 8 panelov z Phase 5 a Phase 5.1: Tri-Geometry Radar (Desired, Observed, Predicted), Execution Tier Map (T0, T1, T2, T3 = PHYSICALLY_UNAVAILABLE, plus R1/R2 remote services as a separate lane), Placement Regret Radar, Dynamic Fallback Graph, Live Telemetry Tiles, Closed-Loop Pulse Regulator a Audit Ledger. Živé dáta prijíma cez Server-Sent Events (SSE) alebo WebSockets z Cloud Run backendu. Dizajn musí byť prémiový, tmavý cyber-physical štýl s JetBrains Mono a Outfit fontmi, SVG radarom a vizuálnymi indikátormi invariantu VITAL_MAX_HP == 6."*

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
   - Global Regret Pill: live noise-adjusted R_bar + placement_quality verdict (INSUFFICIENT_EVIDENCE / ZERO_REGRET_ON_EVALUATED_SET / LOW_REGRET / PLACEMENT_RECALIBRATION_REQUIRED). The string "OPTIMAL" never appears.
2. Tri-Geometry Convergence Radar (SVG Canvas):
   - Polygon 1: Desired Setpoint (Gold dashed)
   - Polygon 2: Previous State S_before (Cyan dotted)
   - Polygon 3: Active State S_after (Emerald solid)
3. Execution Tier Map & Hardware Availability (each tile shows its EvidenceLabel badge):
   - T0: DirectSim Reference Oracle (execution node)
   - T1: CPU AVX-512 VNNI (G2 Cascade Lake vCPUs)
   - T2: NVIDIA L4 (attached to the same node)
   - T3: Dedicated NPU -> PHYSICALLY_UNAVAILABLE (grey, never emerald)
   - Separate "Remote Services" lane: R1 Gemini (status, p95 latency, fallback count), R2 TPU endpoint (optional)
4. Dynamic Fallback Graph & Anti-Oscillation Monitor:
   - Visual flowchart (T2 GPU -> T1 CPU -> T0 DirectSim); R1/R2 have their own timeout -> logged-fallback edge
   - Live reversal counter (0/3), Dwell time indicator, Lock status
5. Placement Regret & Calibration Error Radar:
   - Real-time gauge of MACE composite error vs MACE_TOLERANCE (0.30), with a note when telemetry is MODELED_PRIOR
   - Cumulative regret R_T, noise-adjusted rate, live misplacement counter, selected vs best tier histogram
6. Closed-Loop Pulse Regulator:
   - Whitelisted variable selector, step delta, "Regulate Pulse" button
   - Stress-test "Chatter Test" button triggering rapid alternation
7. Cognitive Director Chat & RCA Box:
   - Powered by the R1 Gemini stream from the backend (GEMINI_FAST_MODEL_ID); answers are labelled "advisory" and cite telemetry rows
8. Live Spanner Audit Ledger:
   - Tabular stream of transactions, hashes, MAE divergence, and commit status.

Data Ingestion:
- Connects to '/api/cortex/stream' via Server-Sent Events (SSE). UI refresh up to 60 fps is interpolated client-side; backend event rate 5–10 Hz.
```

---

### Prompt #08: Terraform / Infrastructure as Code (IaC) Blueprint

**Target:** `infra/terraform/main.tf`

#### 🎙️ Slovak Architectural Intent
> *"Napíš Terraform kód pre Google Cloud, ktorý na jedno kliknutie vytvorí kompletnú infraštruktúru pre CrystalSec Cloud: Cloud Spanner inštanciu a databázu so schemou, BigQuery dataset a partíciovanú tabuľku, Pub/Sub topicy a subscription, Cloud Run servis pre Studio backend, jeden G2 exekučný uzol s NVIDIA L4 (T0 + T1 + T2 spolu), rozpočet s billing alertmi a servisné účty s minimálnymi IAM oprávneniami."*

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
   - google_compute_instance (or GKE node pool): 'crystalsec-exec-node', machine type g2-standard-8 with 1x NVIDIA L4. Hosts T0, T1 and T2 co-located.
   - google_billing_budget: monthly budget with 50/90/100 % alerts (Spanner, GPU and TPU endpoints bill continuously while provisioned).

2. IAM & Security:
   - Dedicated service account 'crystalsec-runtime-sa' with least-privilege roles:
     - roles/spanner.databaseUser
     - roles/bigquery.dataEditor
     - roles/pubsub.publisher / subscriber
     - roles/aiplatform.user (R1 Gemini & optional R2 TPU endpoint access)
```

---

### Prompt #09: End-to-End Cloud Chaos & Verification Suite

**Target:** `tests/test_crystalsec_cloud_verification.py`

#### 🎙️ Slovak Architectural Intent
> *"Vytvor integračný a chaos-testing testovací suite v Pythone, ktorý overí celú cloudovú verziu. Otestuj: 1. Spanner commit a StateVersion invalidáciu, 2. Spanner rollback pri simulovanej chybe GPU, 3. že BigQuery verdikt regretu je správny (pod 10 rozhodnutí INSUFFICIENT_EVIDENCE, pri vloženej misplacement LOW_REGRET alebo RECALIBRATION, nikdy OPTIMAL), 4. zlyhanie micro-probe a pád do dead-letter topicu, 5. že T3 hlási PHYSICALLY_UNAVAILABLE a R1 timeout vedie k zalogovanému fallbacku, 6. invariant VITAL_MAX_HP == 6 naprieč všetkými cloudovými službami."*

#### 🌐 English Technical Production Prompt
```text
TITLE: End-to-End Cloud Verification & Chaos Testing Suite

Create 'tests/test_crystalsec_cloud_verification.py' with tests:
1. test_tier3_npu_reports_physically_unavailable: T3 reports PHYSICALLY_UNAVAILABLE and is never ranked; R1/R2 are never in the placement candidate set.
2. test_spanner_state_version_serializability: Asserts simultaneous mutations fail with StaleStateVersionError; only sequential TrueTime commits succeed.
3. test_transactional_rollback_on_gpu_divergence: Injects synthetic NaN or divergence > 1e-4 on the T2 path; proves rollback restores the pre-state hash.
4. test_regret_verdicts_are_evidence_gated: < 10 decisions -> INSUFFICIENT_EVIDENCE; an injected misplacement beyond the noise floor -> LOW_REGRET or PLACEMENT_RECALIBRATION_REQUIRED; no code path returns "OPTIMAL".
5. test_pubsub_micro_probe_failure_dlq_fallback: Simulates timeout on the GPU micro-probe; verifies the event routes to the Dead-Letter Topic and execution falls back to CPU.
6. test_r1_gemini_rca_contract: With a mock R1, verifies structured RCA JSON, citations of telemetry rows, no write access, and that a timeout produces a logged R1_FALLBACK.
7. test_evidence_labels_present_and_honest: Every latency/telemetry field has an EvidenceLabel; an emulated adapter can never emit HARDWARE_MEASURED; a mutation test (injected delay) changes the measured latency.
8. test_vital_max_hp_universal_invariant_cloud: Asserts VITAL_MAX_HP == 6 across Spanner, BigQuery, and Cloud Run payloads.
```

---

## 3. Recommended AI Studio Execution Workflow

To generate this prototype inside Google AI Studio or Vertex AI:

1. **Step 1:** Paste **Prompt #01** into the System Instructions.
2. **Step 2:** Execute **Prompt #02** (Heterogeneous Triad Engine) and **Prompt #03** (Spanner StateVersion Manager).
3. **Step 3:** Execute **Prompt #04** (BigQuery Cost Ledger) and **Prompt #05** (Pub/Sub Telemetry Streaming).
4. **Step 4:** Execute **Prompt #06** (R1 Gemini Cognitive Director, advisory).
5. **Step 5:** Execute **Prompt #07** to build the complete React + Vite Verification Studio UI.
6. **Step 6:** Execute **Prompt #08** (Terraform, including the billing budget) and deploy.
7. **Step 7:** Run **Prompt #09** (Verification Suite) to validate the invariants, the evidence labels and the honesty of the regret verdicts. Report the measured verdicts as they are; do not tune thresholds until a verdict looks good.
