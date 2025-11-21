# GAMESA / KrystalStack Feature Map

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
