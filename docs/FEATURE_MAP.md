# GAMESA / KrystalStack Feature Map

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              UNIFIED BRAIN                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐              │
│  │ Temporal Fusion  │  │    Attention     │  │   Predictive     │              │
│  │  (Multi-scale)   │──│   Mechanism      │──│    Coding        │              │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘              │
│           │                     │                     │                         │
│           └─────────────────────┴─────────────────────┘                         │
│                                 │                                               │
│  ┌──────────────────────────────┴───────────────────────────────────┐          │
│  │                    COGNITIVE ORCHESTRATOR                         │          │
│  │    + INVENTION ENGINE + EMERGENT SYSTEM + HOMEOSTATIC REG        │          │
│  └──────────────────────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        COGNITIVE ORCHESTRATOR                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   Safety     │  │  Economic    │  │    Rule      │  │ Metacog     │ │
│  │  Guardrails  │──│   Engine     │──│   Engine     │──│ Interface   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
│         │                 │                 │                 │         │
│         └─────────────────┴─────────────────┴─────────────────┘         │
│                                   │                                      │
│                    ┌──────────────┴──────────────┐                      │
│                    │     BASE COGNITIVE ENGINE    │                      │
│                    ├─────────────────────────────┤                      │
│                    │ • Control Theory (PID)      │                      │
│                    │ • Reinforcement Learning    │                      │
│                    │ • Bayesian Updating         │                      │
│                    │ • Statistical Mechanics     │                      │
│                    │ • Information Theory        │                      │
│                    │ • Evolutionary Algorithms   │                      │
│                    └─────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Crystal-Vino Cross-Forex Runtime

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| Protocol | `crystal_protocol.py` | WORKING | GAMESA_JSON_V1 messages |
| Socket Exchange | `crystal_socketd.py` | WORKING | Market tick engine |
| Trading Agents | `crystal_agents.py` | WORKING | GPU/CPU/NPU traders |
| Guardian/Hex | `guardian_hex.py` | WORKING | Order clearing engine |
| Main Daemon | `gamesad.py` | WORKING | Runtime orchestration |

**Commodities Trading:**
- COMPUTE_CREDITS
- THERMAL_HEADROOM
- PRECISION (FP32/FP16/INT8)
- LATENCY_BUDGET

---

## 2. Cross-Platform HAL

| Platform | File | Status | Accelerators |
|----------|------|--------|--------------|
| Intel | `platform_hal.py` | WORKING | XMX, GNA, AVX-512 |
| AMD | `platform_hal.py` | WORKING | WMMA, XDNA, Infinity Cache |
| ARM | `platform_hal.py` | WORKING | NEON, SVE, Ethos, big.LITTLE |

**Platform-Specific Features:**
```
Intel: Thread Director hints, XMX auto-switch, GNA offload
AMD:   CCD-aware affinity, WMMA precision, Precision Boost
ARM:   EAS hints, Ethos/Hexagon NPU, DVFS coordination
```

---

## 3. Cognitive Engine Components

### 3.1 Control Theory (PID + Amygdala)
```python
# Location: cognitive_engine.py:39-106
FeedbackController:
  - Proportional/Integral/Derivative gains
  - Amygdala factor for risk modulation
  - Sigmoid-based risk dampening
```

### 3.2 Reinforcement Learning
```python
# Location: cognitive_engine.py:112-217
TDLearner:
  - Temporal Difference updates
  - Prioritized experience replay
  - Epsilon-greedy action selection
  - Q-table with state discretization
```

### 3.3 Bayesian Updating
```python
# Location: cognitive_engine.py:223-300
BayesianTracker:
  - Kalman-like belief updates
  - Uncertainty quantification
  - Curiosity computation
  - Confidence intervals
```

### 3.4 Statistical Mechanics
```python
# Location: cognitive_engine.py:306-381
StatMechAllocator:
  - Boltzmann distribution allocation
  - Zone energy computation
  - Free energy calculation
  - System entropy tracking
```

### 3.5 Information Theory
```python
# Location: cognitive_engine.py:387-486
EntropyAnalyzer:
  - Empirical entropy computation
  - Anomaly scoring (z-score)
  - Mutual information estimation
```

### 3.6 Evolutionary Algorithms
```python
# Location: cognitive_engine.py:492-626
EvolutionaryOptimizer:
  - Tournament selection
  - Uniform crossover
  - Gaussian mutation
  - Multi-objective fitness
```

---

## 4. Economic Engine

| Feature | Location | Status |
|---------|----------|--------|
| ResourceBudgets | `cognitive_engine.py:775-781` | WORKING |
| ActionEconomicProfile | `cognitive_engine.py:785-791` | WORKING |
| Trade Validation | `cognitive_engine.py:820-830` | WORKING |
| Trade Execution | `cognitive_engine.py:832-850` | WORKING |

**Budget Currencies:**
- `cpu_budget_mw` (45W default)
- `gpu_budget_mw` (150W default)
- `thermal_headroom_c` (15C)
- `latency_budget_ms` (16.67ms = 60 FPS)

---

## 5. Low-Code Inference Rules

| Feature | Location | Status |
|---------|----------|--------|
| MicroInferenceRule | `cognitive_engine.py:872-880` | WORKING |
| RuleEngine | `cognitive_engine.py:883-930` | WORKING |
| SafetyTier enum | `cognitive_engine.py:864-868` | WORKING |

**Rule Format:**
```python
MicroInferenceRule(
    rule_id="thermal_throttle",
    condition="thermal_headroom < 5",
    action="reduce_power",
    priority=10,
    safety_tier=SafetyTier.STRICT,
    shadow_mode=False
)
```

---

## 6. Safety Guardrails

| Constraint | Check | Violation Action |
|------------|-------|------------------|
| thermal_critical | `thermal_headroom > 0` | emergency_throttle |
| power_limit | `power_draw < power_limit * 1.1` | reduce_power |
| memory_pressure | `memory_util < 0.95` | gc_trigger |
| latency_budget | `latency_ms < latency_target * 2` | reduce_quality |

---

## 7. Metacognitive Interface

| Feature | Location | Status |
|---------|----------|--------|
| Experience Logging | `cognitive_engine.py:1039-1042` | WORKING |
| Pattern Analysis | `cognitive_engine.py:1044-1066` | WORKING |
| Policy Proposals | `cognitive_engine.py:1068-1080` | WORKING |
| Rule Approval | `cognitive_engine.py:1082-1091` | WORKING |

---

## 8. Memory Management

| Component | File | Status |
|-----------|------|--------|
| TierManager | `memory_manager.py` | WORKING |
| PrefetchPredictor | `memory_manager.py` | WORKING |
| AdaptiveCompressor | `memory_manager.py` | WORKING |
| SwapManager | `memory_manager.py` | WORKING |
| MemoryBroker | `memory_manager.py` | WORKING |

**Memory Tiers:**
```
HOT → WARM → COLD → FROZEN → SWAPPED
 L1    L2     L3    Compressed  Disk
```

---

## 9. Platform Telemetry

| Platform | Metrics | File |
|----------|---------|------|
| Intel | XMX util, GNA status, Thread Director | `platform_telemetry.py` |
| AMD | CCD temps, Infinity Cache hits, XDNA | `platform_telemetry.py` |
| ARM | big.LITTLE state, NPU util, EAS hints | `platform_telemetry.py` |

---

## 10. Accelerator Manager

| Feature | File | Status |
|---------|------|--------|
| Workload Routing | `accelerator_manager.py` | WORKING |
| Precision Selection | `accelerator_manager.py` | WORKING |
| Thermal-aware Scheduling | `accelerator_manager.py` | WORKING |

**Workload Types:**
- INFERENCE, TRAINING, MATRIX_MULTIPLY
- CONVOLUTION, SPEECH, VISION
- GENERAL_COMPUTE, SHADER

---

## Integration Flow

```
Telemetry Input
      │
      ▼
┌─────────────────┐
│ Safety Validate │──► CRITICAL? ──► Emergency Action
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cognitive Engine│
│ (PID + RL +     │
│  Bayesian + ...) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Rule Evaluation │──► Override if triggered
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Economic Check  │──► Can't afford? ──► Throttle
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Execute Action  │
│ + Log Experience│
└─────────────────┘
```

---

## File Index

| File | Lines | Purpose |
|------|-------|---------|
| `cognitive_engine.py` | 1212 | Core cognitive components |
| `crystal_protocol.py` | ~200 | Protocol definitions |
| `crystal_socketd.py` | ~350 | Market exchange |
| `crystal_agents.py` | ~400 | Trading agents |
| `guardian_hex.py` | ~300 | Order clearing |
| `gamesad.py` | ~400 | Main daemon |
| `platform_hal.py` | ~500 | Hardware abstraction |
| `platform_telemetry.py` | ~350 | Telemetry collection |
| `accelerator_manager.py` | 414 | Workload routing |
| `gamesad_crossplatform.py` | 504 | Cross-platform daemon |
| `memory_manager.py` | ~600 | Memory tiering |

---

## Usage Example

```python
from cognitive_engine import create_cognitive_orchestrator

# Create orchestrator
orchestrator = create_cognitive_orchestrator()

# Process telemetry
result = orchestrator.process({
    "frametime_ms": 16.6,
    "thermal_headroom": 0.8,
    "cpu_util": 0.6,
    "gpu_util": 0.7,
    "power_draw": 120,
    "power_limit": 150,
    "memory_util": 0.5,
    "latency_ms": 15,
    "latency_target": 16.67
})

print(f"Action: {result['action']}")
print(f"Budgets: {result['economic']['budgets']}")
print(f"Violations: {result['safety']['violations']}")
```

---

## 12. Invention Engine (Novel Algorithms)

| Component | File | Status | Innovation |
|-----------|------|--------|------------|
| SuperpositionScheduler | `invention_engine.py` | WORKING | Quantum-inspired task states |
| SpikeTimingAllocator | `invention_engine.py` | WORKING | Neuromorphic STDP allocation |
| SwarmOptimizer | `invention_engine.py` | WORKING | Particle swarm optimization |
| CausalInferenceEngine | `invention_engine.py` | WORKING | Granger causality detection |
| HyperdimensionalEncoder | `invention_engine.py` | WORKING | 10,000D holographic memory |
| ReservoirComputer | `invention_engine.py` | WORKING | Echo state network prediction |

**Key Innovations:**
- Tasks exist in superposition until "measured"
- Spike-timing dependent plasticity for resource flow
- Causal graph discovery with root cause analysis
- Holographic associative memory

---

## 13. Unified Brain

| Component | File | Status | Function |
|-----------|------|--------|----------|
| TemporalFusion | `unified_brain.py` | WORKING | Multi-timescale harmonization |
| AttentionMechanism | `unified_brain.py` | WORKING | 4-head telemetry attention |
| PredictiveCoding | `unified_brain.py` | WORKING | Hierarchical prediction |
| HomeostaticRegulator | `unified_brain.py` | WORKING | Biological stability |
| UnifiedBrain | `unified_brain.py` | WORKING | Master integration |
| DistributedBrain | `unified_brain.py` | WORKING | Multi-node consensus |

**Timescales:**
```
micro (1ms)  → meso (10ms) → macro (100ms) → meta (1s)
```

---

## 14. Emergent System

| Component | File | Status | Behavior |
|-----------|------|--------|----------|
| CellularAutomata | `emergent_system.py` | WORKING | Pattern emergence |
| SelfOrganizingMap | `emergent_system.py` | WORKING | Kohonen clustering |
| AntColonyOptimizer | `emergent_system.py` | WORKING | Pheromone paths |
| StigmergicEnvironment | `emergent_system.py` | WORKING | Indirect coordination |
| CriticalityDetector | `emergent_system.py` | WORKING | Edge of chaos |
| MorphogeneticField | `emergent_system.py` | WORKING | Turing patterns |

**Self-Organization:**
- Cellular automata for pattern detection
- SOM maps state space to 2D topology
- Stigmergy enables indirect agent communication
- Criticality: power law exponent ~1.5-2.5, branching ratio ~1.0

---

## Full Integration Example

```python
from unified_brain import create_unified_brain

# Create master brain
brain = create_unified_brain()

# Process telemetry through all systems
result = brain.process({
    "frametime_ms": 16.6,
    "thermal_headroom": 0.8,
    "cpu_util": 0.6,
    "gpu_util": 0.7,
    "power_draw": 120,
    "temperature": 72.0,
    "fps": 60.0,
    "memory_util": 0.5
})

print(f"Decision: {result['decision']}")
print(f"Surprise: {result['surprise']}")
print(f"Stress: {result['stress']}")
print(f"Source: {result['decision']['source']}")
```

---

## Updated File Index

| File | Lines | Purpose |
|------|-------|---------|
| `cognitive_engine.py` | 1212 | Core cognitive + orchestrator |
| `invention_engine.py` | 909 | Novel algorithmic innovations |
| `unified_brain.py` | 587 | Master integration layer |
| `emergent_system.py` | 812 | Self-organizing behaviors |
| `crystal_protocol.py` | ~200 | Protocol definitions |
| `crystal_socketd.py` | ~350 | Market exchange |
| `crystal_agents.py` | ~400 | Trading agents |
| `guardian_hex.py` | ~300 | Order clearing |
| `platform_hal.py` | ~500 | Hardware abstraction |
| `memory_manager.py` | ~600 | Memory tiering |

**Total: ~7,000+ lines of innovative code**

---

## 15. Testing & Metrics

### Test Suite (`test_sanity.py`)

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestSafetyGuardrails | 5 | thermal, power, memory, latency, violation rate |
| TestRuleEngine | 4 | trigger, no-trigger, shadow-mode, priority |
| TestEconomicEngine | 4 | afford, over-budget, trade, replenish |
| TestCognitiveOrchestrator | 4 | normal, safety override, economic, metacog |
| TestInventionEngine | 4 | quantum, causal, HD encoder, full process |
| TestEmergentSystem | 4 | CA, SOM, criticality, full process |

**Run tests:**
```bash
cd src/python && python -m unittest test_sanity -v
```

### Metrics Logger (`metrics_logger.py`)

| Component | Function |
|-----------|----------|
| MetricsCollector | samples, counters, histograms, percentiles |
| CSVExporter | export samples.csv, summary.csv |
| BrainMetrics | latency/thermal/budget/violation hooks |
| FeatureFlags | prod/lab configuration |

**Usage:**
```python
from metrics_logger import get_metrics, FeatureFlags, CSVExporter

# Get KPI summary
kpi = get_metrics().get_kpi_summary()
print(kpi["latency"]["process"])  # {count, min, max, avg, p50, p95, p99}

# Export to CSV
exporter = CSVExporter("./metrics")
exporter.export_summary(get_metrics().collector)
```

### Feature Toggles

```python
from metrics_logger import FeatureFlags
from unified_brain import UnifiedBrain

# Production mode (invention/emergent OFF)
brain = UnifiedBrain(flags=FeatureFlags.production())

# Lab mode (all features ON)
brain = UnifiedBrain(flags=FeatureFlags.laboratory())
```

| Flag | Production | Laboratory |
|------|------------|------------|
| enable_invention_engine | OFF | ON |
| enable_emergent_system | OFF | ON |
| enable_distributed_brain | OFF | ON |
| enable_metrics_logging | ON | ON |
| enable_shadow_rules | OFF | ON |

---

## 16. Default Rules Bundle

| Rule ID | Condition | Action | Priority | Mode |
|---------|-----------|--------|----------|------|
| thermal_warning | `thermal_headroom < 8` | reduce_power | 8 | STRICT |
| thermal_critical | `thermal_headroom < 3` | emergency_throttle | 10 | STRICT |
| latency_cutback | `latency_ms > target * 1.5` | reduce_quality | 7 | STRICT |
| memory_demotion | `memory_util > 0.85` | demote_cold_pages | 5 | STRICT |
| experimental_boost | idle + headroom > 15 | opportunistic_boost | 3 | SHADOW |

**Install rules:**
```python
from test_sanity import get_default_rules, install_default_rules
from cognitive_engine import create_cognitive_orchestrator

orchestrator = create_cognitive_orchestrator()
install_default_rules(orchestrator)
```

---

## 17. KPI Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Process latency (p95) | < 5ms | `metrics.latency_process_ms` |
| Cognitive latency (p95) | < 2ms | `metrics.latency_cognitive_ms` |
| Thermal violations/min | < 2 | `metrics.violations_thermal_*` |
| Decision consistency | > 95% | action distribution entropy |
| Memory overhead | < 50MB | process RSS delta |
| CPU overhead | < 2% | idle CPU usage |
