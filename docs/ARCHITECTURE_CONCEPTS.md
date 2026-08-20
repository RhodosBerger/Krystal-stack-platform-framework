# GAMESA Architecture Concepts

## System Overview

GAMESA (Game-Aware Memory-Enhanced System Architecture) is a unified adaptive intelligence platform integrating six core subsystems:

```
                    ┌─────────────────────┐
                    │  GamesaFramework    │
                    │   (Orchestrator)    │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌───────────────┐
│   Cognitive   │    │   CrossForex    │    │     MAVB      │
│   Synthesis   │    │    Exchange     │    │  (3D Memory)  │
└───────────────┘    └─────────────────┘    └───────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌─────────────────┐    ┌───────────────┐
│  TPU Bridge   │    │  Thread Boost   │    │   Knowledge   │
│ (Acceleration)│    │   (Affinity)    │    │   Optimizer   │
└───────────────┘    └─────────────────┘    └───────────────┘
```

## Core Concepts

### 1. Cognitive Synthesis Engine

The "brain" of GAMESA with seven specialized domains:

| Domain | Function | Role |
|--------|----------|------|
| STRATEGIC | Long-term planning | Resource allocation |
| TACTICAL | Short-term execution | Immediate responses |
| CREATIVE | Novel solutions | Innovation generation |
| ANALYTICAL | Pattern recognition | Data interpretation |
| PROTECTIVE | Safety monitoring | Thermal/power guards |
| REFLECTIVE | Self-analysis | Learning from history |
| INTUITIVE | Heuristic decisions | Fast approximations |

**Key Components:**
- **Knowledge Graph**: Semantic network of concepts and relationships
- **Consciousness Stream**: Thought flow (observation → hypothesis → decision → reflection)
- **Insight Engine**: Cross-domain pattern synthesis
- **Meta-Cognition**: Thinking about thinking

### 2. Cross-Forex Exchange

Market-based resource trading system:

```
Order Types:  BUY_LIMIT | SELL_LIMIT | BUY_MARKET | SELL_MARKET

Commodities:  COMPUTE | MEMORY | BANDWIDTH | THERMAL_QUOTA | PRIORITY_TOKENS

Tier System:  LEGENDARY (top 5%)
              EPIC (top 20%)
              RARE (top 40%)
              COMMON (middle)
              INHIBITED (bottom 10%)
```

**Amygdala Thermal Guardian:**
- Monitors CPU/GPU temperature and power
- Triggers SAFE → ELEVATED → THROTTLE → EMERGENCY transitions
- Broadcasts thermal directives to all agents

### 3. MAVB (Memory-Augmented Virtual Bus)

3D voxel-based memory fabric:

```
Dimensions:
  X-axis: Memory tier (L1 → L2 → L3 → RAM → SSD → HDD → Network)
  Y-axis: Temporal slot (0-15, 16ms each = 256ms window)
  Z-axis: Compute depth (32 priority levels)

Layers:
  PCL (Physical Capture): Hardware memory mapping
  VAL (Virtual Allocation): Voxel state management
  ARL (Action Relay): Trade execution
  Guardian: Arbitration and fairness
```

### 4. TPU Bridge

AI workload acceleration routing:

```
Accelerators:
  IRIS_XE_GPU: General compute, inference
  GNA_2_0: Low-power speech/audio
  VPU_MOVIDIUS: Vision processing
  CPU_AVX512: Fallback compute

Workload Matching:
  INFERENCE → IRIS_XE_GPU (high throughput)
  TRAINING → CPU_AVX512 (flexibility)
  SPEECH → GNA_2_0 (power efficient)
  VISION → VPU_MOVIDIUS (specialized)
```

### 5. Thread Boost Layer

Core affinity and memory optimization:

```
Zone Configuration:
  grid_coverage: ((x_start, x_end), (y_start, y_end), (z_start, z_end))
  memory_mb: Dedicated allocation
  priority: REALTIME | HIGH | NORMAL | LOW | BACKGROUND

RPG Craft Presets:
  LEGENDARY_BURST: 4P+4E cores, 512MB, duration 2000ms
  EPIC_PERFORMANCE: 4P+2E cores, 256MB, duration 5000ms
  RARE_BALANCED: 2P+4E cores, 128MB, duration 10000ms
  COMMON_EFFICIENCY: 0P+4E cores, 64MB, duration 30000ms
```

### 6. Knowledge Optimizer

Learning and adaptation layer:

```
Components:
  PrioritizedReplayBuffer: TD-error weighted experience sampling
  DoubleQLearner: Reduced overestimation bias
  CuriosityModule: Intrinsic motivation through prediction error
  HierarchicalTimescale: Fast/medium/slow adaptation loops
  AttractorGuidedSearch: Knowledge-based navigation
```

## Data Flow

### Main Tick Cycle

```
1. Gather Telemetry
   └─► CPU/GPU util, temps, power, performance metrics

2. Update Thermal State
   └─► Exchange.update_thermal() → Amygdala response

3. Cognitive Cycle
   └─► Domain activation → Knowledge graph query → Insight generation

4. Knowledge Optimizer
   └─► observe() → decide() → action vector

5. Exchange Tick
   └─► Order matching → Trade execution → Ranking update

6. MAVB Tick
   └─► Voxel expiration → Resource rebalancing

7. Thread Boost Tick
   └─► Zone updates → Boost map application

8. TPU Inference (periodic)
   └─► Workload routing → Accelerator execution

9. Reward & Learn
   └─► compute_reward() → optimizer.reward() → policy update

10. State Transition
    └─► RUNNING | OPTIMIZING | THROTTLING | EMERGENCY
```

## Integration Patterns

### Agent Registration

```python
# Agents participate in Cross-Forex trading
framework.exchange.register_agent("COGNITIVE_AGENT", credits=10000)
framework.exchange.update_domain("COGNITIVE_AGENT", "STRATEGIC")

# Agents stored in framework.agents dict
framework.agents["COGNITIVE_AGENT"] = {"domain": "STRATEGIC", "active": True}
```

### Resource Trading

```python
# Submit trade via framework API
result = framework.submit_trade(
    agent_id="COGNITIVE_AGENT",
    commodity=Commodity.COMPUTE,
    order_type=OrderType.BUY_LIMIT,
    quantity=100,
    price=1.5
)
```

### Voxel Allocation

```python
# Request MAVB voxel for memory allocation
result = framework.request_voxel(
    agent_id="MEMORY_AGENT",
    voxel=(2, 5, 10),  # L3 cache, slot 5, depth 10
    resource=ResourceType.COMPUTE,
    bytes_req=1024
)
```

### Inference Routing

```python
# Run TPU-accelerated inference
result = framework.run_inference(
    workload=WorkloadType.INFERENCE,
    input_data={"tensor": [1.0, 2.0, 3.0]},
    domain="ANALYTICAL"
)
```

## State Transitions

```
INIT → STARTING → RUNNING ←──────────────────┐
                     │                        │
                     ▼                        │
              OPTIMIZING ────────────────────►│
                     │                        │
                     ▼                        │
              THROTTLING ────────────────────►│
                     │
                     ▼
               EMERGENCY → STOPPING → STOPPED
```

## Reward Computation

```python
reward = 0.0
reward += 0.3 * performance      # Performance contribution
reward += 0.2 if thermal_ok      # Thermal bonus
reward += 0.3 * cognitive_fitness # Cognitive contribution
reward += 0.2 if safe            # Safety bonus
return clamp(reward, 0, 1)
```

## Design Principles

1. **Zero External Dependencies**: All components use Python stdlib only
2. **Modular Integration**: Each subsystem works standalone or together
3. **Unified API**: observe() → decide() → reward() pattern throughout
4. **Thermal Awareness**: Every component respects thermal constraints
5. **Market Fairness**: Tier-based priority prevents resource starvation
6. **Self-Awareness**: Meta-cognition enables learning about learning

## File Structure

```
src/python/
├── krystal_sdk.py           # Core adaptive intelligence
├── knowledge_optimizer.py   # Learning enhancements
├── krystal_enhanced.py      # SDK + optimizer bridge
├── cognitive_synthesis.py   # 7-domain cognitive engine
├── mavb.py                  # 3D memory fabric
├── cross_forex.py           # Resource trading exchange
├── tpu_bridge.py            # AI acceleration routing
├── thread_boost.py          # Core affinity management
└── gamesa_framework.py      # Unified orchestrator
```
