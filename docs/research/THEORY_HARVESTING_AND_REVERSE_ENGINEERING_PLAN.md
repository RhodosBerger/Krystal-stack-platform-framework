# KRYSTAL-STACK: REVERSE ENGINEERING & THEORY HARVESTING PLAN
==============================================================================
Document ID: DOC-RESEARCH-THEORY-HARVEST-2026-V1
Milestone Association: V-THEORY-HARVEST-01 / V-MILESTONE-H08
Constitutional Invariant: `VITAL_MAX_HP == 6`
Normative Authority: `AGENTS.md`, `GEMINI.md`, `krystal_theory/REGISTER_TVRDENI.md`, `KSPP v1`

---

## 1. Executive Summary & Purpose

Across the Krystal Stack codebase, diverse engineering subsystems (formal triage operators, diagnostic rails, UMA inductive voltage governors, OpenVINO heterogeneous schedulers, Janet PEGs, procedural world compilers, and Blender-style mimicry SDF engines) embody implicit and explicit **theories, mathematical lemmas, and physical models**.

This document outlines the **Comprehensive Reverse Engineering and Theory Harvesting Plan**:
1. **Automated Discovery**: Systematically crawls and extracts every theoretical construct into the formal [`ExperimentalTheoryDependencyGraph`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/verification/theory_dependency_graph.py).
2. **Epistemic Stratification**: Enforces strict epistemic classification (`DOKÁZANÉ`, `NAMERANÉ`, `MODEL`, `NEOVERENE`, `OPRAVENÉ`, `STIAHNUTÉ`, `ZAMIETNUTÉ`) according to [`krystal_theory/REGISTER_TVRDENI.md`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_theory/REGISTER_TVRDENI.md).
3. **Closed-Loop Verification Binding**: Couples harvested theory dependencies directly into Gate 1.5 of the [`TwoPhaseReasoningComputingAlternator`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/two_phase_reasoning_computing_alternator.py), prohibiting any state mutation based on refuted (`ZAMIETNUTÉ`) or unverified (`STIAHNUTÉ`) hypotheses.

---

## 2. Theoretical Architecture: The 6 Target Module Clusters

```
                     ┌────────────────────────────────────────────────────────┐
                     │          KRYSTAL-STACK CODEBASE & SUBSYSTEMS           │
                     └──────────────────────────┬─────────────────────────────┘
                                                │
           ┌──────────────────────┬─────────────┴────────────┬──────────────────────┐
           ▼                      ▼                          ▼                      ▼
┌──────────────────────┐┌──────────────────────┐┌──────────────────────┐┌──────────────────────┐
│  CLUSTER 1: RAILS &  ││ CLUSTER 2: HARDWARE  ││  CLUSTER 3: DUAL-    ││ CLUSTER 4: PROCEDURAL│
│   EPISTEMIC TRIAGE   ││   ACCEL & UMA PDN    ││    PHASE ALTERNATOR  ││  MANIFOLDS & JANET   │
│ - Vety 1.1, 2.6, 3.2 ││ - Faraday-Lenz PDN   ││ - Proposer!=Evaluator││ - World manifolds   │
│ - Smith's WSPT Rule  ││ - Return Loss (RL)   ││ - Chronometric Shadow││ - Antigravity PEG    │
│ - Triage operator    ││ - Tiger Lake bypass  ││ - Gate 1.5 Theory DAG││ - Polynomial CSG     │
└──────────────────────┘└──────────────────────┘└──────────────────────┘└──────────────────────┘
           │                      │                          │                      │
           └──────────────────────┴─────────────┬────────────┴──────────────────────┘
                                                │ Automated AST & Pattern Crawl
                                                ▼
                     ┌────────────────────────────────────────────────────────┐
                     │         THEORY HARVESTING ENGINE & DEPENDENCY DAG      │
                     │  - krystal_kernel/verification/theory_harvesting_engine│
                     │  - krystal_kernel/verification/theory_dependency_graph │
                     └────────────────────────────────────────────────────────┘
```

### Cluster 1: Formal Triage & Scheduling Theory
* **Primary Modules**: [`krystal_theory/`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_theory/), [`krystal_kernel/rails/library.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/rails/library.py), [`krystal_kernel/rails/rail.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/rails/rail.py)
* **Harvested Theories & Lemmas**:
  - `Veta 1.1`: Totality and determinism of triage state mapping (`DOKÁZANÉ`).
  - `Veta 2.6`: Smith's Rule (WSPT) minimizing weighted waiting time $\sum w_i C_i$ (`MODEL`).
  - `Veta 3.1 & 3.2`: Schmitt-trigger asymmetric hysteresis eliminating high-frequency switching chatter (`DOKÁZANÉ`).
  - `Lema 5.1 & 5.2`: Sampling bounds and quantization noise under non-intrusive clocks (`DOKÁZANÉ`).
  - `Axiómy A3.1 & A3.2`: Gaussian i.i.d. and bounded uniform noise models (`MODEL`).

### Cluster 2: Hardware Acceleration & Inductive UMA PDN
* **Primary Modules**: [`krystal_kernel/uma_inductive_voltage_governor.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/uma_inductive_voltage_governor.py), [`krystal_kernel/llm_hw_acceleration_bridge.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/llm_hw_acceleration_bridge.py), [`krystal_kernel/silicon_telemetry_lexicon.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/silicon_telemetry_lexicon.py)
* **Harvested Theories**:
  - `THEORY-H07-FARADAY-LENZ-PDN`: Opposing EMF $V_{\text{emf}} = -L \frac{di}{dt}$ mitigating transient droop under LLM prefill bursts (`NAMERANÉ`).
  - `THEORY-H07-TRANSMISSION-LINE-Z0`: Complex impedance $Z(\omega) = R + j\omega L$ and return loss $RL = -20 \log_{10}(|\Gamma|)$ on $50.0\ \Omega$ baseline (`NAMERANÉ`).
  - `THEORY-LLM-HYBRID-PREFILL-DECODE`: Heterogeneous task split between Iris Xe GPU (80 EUs) and AVX-512 CPU (`NAMERANÉ`).
  - `THEORY-SILICON-ARRHENIUS-WEAR`: Thermal longevity acceleration factor $AF = \exp\left(\frac{E_a}{k_B} \left(\frac{1}{T_0} - \frac{1}{T}\right)\right)$ governing safe PL1 bypass (`NAMERANÉ`).

### Cluster 3: Two-Phase Epistemic Alternator Layer
* **Primary Modules**: [`krystal_kernel/two_phase_reasoning_computing_alternator.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/two_phase_reasoning_computing_alternator.py), [`krystal_kernel/custom_governor.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/custom_governor.py)
* **Harvested Theories**:
  - `THEORY-P3-CHRONOMETRIC-SHADOW`: Mutual exclusion between render scanout and compute/reasoning memory bursts ($\Phi_{\text{Scanout}} \cdot \Phi_{\text{Compute}} = 0 \land \mu_{\text{UMA}} < 0.78$) (`DOKÁZANÉ`).
  - `THEORY-DUAL-MODE-EPISTEMIC-ALTERNATOR`: Strict separation of Proposer (Phase 1) and Evaluator (Phase 2), enforcing $\Delta S = 0$ upon contract breach (`DOKÁZANÉ`).

### Cluster 4: OpenWorld Semantic Manifolds & Janet Language
* **Primary Modules**: [`openworld_engine/world_semantic_compiler.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/openworld_engine/world_semantic_compiler.py), [`openworld_engine/terrain_manifold.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/openworld_engine/terrain_manifold.py), [`krystal_janet/antigravity_peg.janet`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/antigravity_peg.janet)
* **Harvested Theories**:
  - `THEORY-WORLD-BIOME-PHASE-SPACE`: Mathematical mapping from natural language tokens (Slovak/English) to multi-dimensional terrain & climate phase spaces (`MODEL`).
  - `THEORY-JANET-PEG-GRAMMAR`: Formal parsing expression grammar (PEG) guaranteeing linear-time, deterministic token synthesis without backtracking ambiguities (`DOKÁZANÉ`).

### Cluster 5: Mimicry Engine & Procedural Signed Distance Fields
* **Primary Modules**: [`mimicry_engine/primitives.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/mimicry_engine/primitives.py), [`krystal_kernel/procedural_sdf_hw_bridge.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/procedural_sdf_hw_bridge.py), [`tests/test_mimicry_compositor.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_mimicry_compositor.py)
* **Harvested Theories**:
  - `THEORY-SDF-POLYNOMIAL-SMOOTH-CSG`: $C^1$-continuous polynomial blending $s_{\min}(a, b, k) = \min(a, b) - \frac{\max(k - |a - b|, 0)^2}{4k}$ preserving surface normal continuity (`DOKÁZANÉ`).
  - `THEORY-HAUSSMANN-URBAN-COMPOSITION`: Spatial geometry constraints for procedural building setbacks, cornices, and street corridors (`MODEL`).

### Cluster 6: Verification Runtime & Criteria Sets
* **Primary Modules**: [`krystal_kernel/verification/typed_dependency_runtime.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/verification/typed_dependency_runtime.py), [`krystal_kernel/verification/capability_registry.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/verification/capability_registry.py)
* **Harvested Theories**:
  - Strict computational origin separation: Physical Conversion vs Algebraic Derivation vs Latent Projection (`EpistemicState.DER_LATENT_PROJECTION` vs `OBS`) (`DOKÁZANÉ`).
  - Paravirtual capability validation: Prohibiting blind dispatch without active micro-probing (`CapabilityState.VERIFIED`) (`DOKÁZANÉ`).

---

## 3. Four-Phase Reverse Engineering Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE A: Automated Static AST & Regex Harvest                               │
│ - Run TheoryHarvestingEngine over all .py, .json, .janet, and .md files.   │
│ - Extract raw theory_ref, formula patterns, and theorem citations.          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE B: Topological Dependency Ordering & Cycle Elimination                │
│ - Insert all extracted nodes into ExperimentalTheoryDependencyGraph.        │
│ - Run DFS cycle detector (TheoryDependencyCycleError).                      │
│ - Trace recursive dependency closures and identify root axioms.             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE C: Epistemic Stratification & CDT Generation                          │
│ - Audit all nodes labeled MODEL or NEOVERENE.                              │
│ - Formulate the Cheapest Discriminating Test (CDT) for each node.           │
│ - Design hardware telemetry probes (MSR, RAPL, or microbenchmarks).         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE D: Runtime Closed-Loop Binding into Alternator Gate 1.5               │
│ - Enforce that proposals declare their theoretical dependencies.            │
│ - TwoPhaseVerifierGate rejects any proposal depending on ZAMIETNUTE/STIAHNUTÉ│
│ - Invariant VITAL_MAX_HP == 6 enforced unconditionally across all stages.   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Current Harvest Audit Metrics (Empirical Baseline)

The static analysis engine [`TheoryHarvestingEngine`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/verification/theory_harvesting_engine.py) harvested the following baseline:

* **Files Scanned**: 458 source files.
* **Harvested Theory Definitions**: 70 formal theory references.
* **Constitutional Invariant**: `VITAL_MAX_HP == 6` (100% compliance).

### Distribution by Epistemic Status
| Status | Count | Description / Action Required |
| :--- | :--- | :--- |
| `DOKÁZANÉ` | 58 | Mathematically proven theorems (Triage totality, Schmitt hysteresis, PEG parsing). |
| `NAMERANÉ` | 8 | Hardware-verified telemetry (Inductive choke back-EMF, UMA return loss, thermal bypass). |
| `MODEL` | 1 | Parameterized model (WSPT scheduling weights). Needs empirical trace calibration. |
| `NEOVERENE` | 3 | Unverified hypotheses awaiting hardware telemetry probes. |
| `OPRAVENÉ` | Historical | Revised erroneous formulations (e.g. C1 die parity, A2 mechanism set). |
| `STIAHNUTÉ` | Historical | Formally withdrawn dead-ends (e.g. B1 constant embedding saving). |

---

## 5. Execution Roadmap & Definition of Done (DoD)

### Milestone Step 1: Automated AST Extraction CI Integration
- Add `TheoryHarvestingEngine.harvest_all()` as a standard verification step in the nightly build certificate pipeline.
- Fail CI if any code references a `theory_ref` not present in [`ExperimentalTheoryDependencyGraph`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_kernel/verification/theory_dependency_graph.py).

### Milestone Step 2: CDT Execution for Remaining `MODEL` Nodes
- Execute Cheapest Discriminating Tests on WSPT weights in [`krystal_theory/wspt_ranker.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_theory/wspt_ranker.py).
- Measure actual execution times ($p_i$) vs estimated weights ($w_i$) on live CPU threads to promote status from `MODEL` to `NAMERANÉ`.

### Milestone Step 3: Closed-Loop Alternator Dynamic Injection
- Ensure all game commands dispatched via [`GameLanguageApiFetcher`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/economic_engine/game_language_api_fetcher.py) and terrain modifications pass through Gate 1.5 with validated theory dependencies.

### Definition of Done (DoD):
1. **Zero Unregistered Theories**: 100% of theoretical claims in docstrings and rail schemas must resolve to registered `TheoryNode` instances.
2. **Zero Refuted Leaks**: No state commit is permitted if any parent dependency in its theoretical DAG is `ZAMIETNUTÉ` or `STIAHNUTÉ`.
3. **Invariant Preservation**: `VITAL_MAX_HP == 6` holds with `ContractViolation` on breach ($\Delta S = 0$).
4. **All Tests Green**: All 2,061+ existing unit tests and new theory tests pass with 0 failures.
