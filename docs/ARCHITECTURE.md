# KrystalSDK Architecture

## Overview

KrystalSDK is a multi-layer adaptive intelligence system designed for real-time optimization across diverse domains: games, servers, ML pipelines, and IoT devices.

```
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                             │
│  Game Optimizer │ Server Autoscaler │ ML Tuner │ IoT Controller │
├─────────────────────────────────────────────────────────────────┤
│                    GENERATIVE PLATFORM                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐ │
│  │ Planner │ │ Coder   │ │ Critic  │ │Guardian │ │ Optimizer │ │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └─────┬─────┘ │
│       └───────────┴───────────┴───────────┴─────────────┘       │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    LLM CLIENT                             │   │
│  │  Local: Ollama │ LMStudio │ vLLM                          │   │
│  │  API: OpenAI │ Claude │ Gemini                            │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    UNIFIED SYSTEM (6 Levels)                     │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐ │
│  │Hardware │→│ Signal  │→│Learning │→│Predict  │→│ Emergence │ │
│  │ Level   │ │ Level   │ │ Level   │ │ Level   │ │  Level    │ │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───────────┘ │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  GENERATION LEVEL                         │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    KRYSTAL SDK CORE                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────────┐ │
│  │MicroLearner│ │ MicroPhase │ │ MicroSwarm │ │MicroController│ │
│  │ (TD-Learn) │ │ (Phases)   │ │   (PSO)    │ │    (PID)     │ │
│  └────────────┘ └────────────┘ └────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. KrystalSDK Core (`krystal_sdk.py`)

The minimal, zero-dependency brain providing:

#### MicroLearner (TD-Learning)
- Temporal Difference learning for value estimation
- Online weight updates with configurable learning rate
- Experience-based policy improvement

```python
class MicroLearner:
    def predict(self, state) -> float       # Estimate state value
    def update(state, reward, next_state)   # TD update
```

#### MicroPhase (Phase Transitions)
- Automatic exploration/exploitation balance
- Four phases: SOLID → LIQUID → GAS → PLASMA
- Temperature-based state transitions

```python
class MicroPhase:
    SOLID   # Low exploration, exploit known good
    LIQUID  # Balanced exploration
    GAS     # High exploration
    PLASMA  # Breakthrough/creative mode
```

#### MicroSwarm (Particle Swarm Optimization)
- Global optimization via swarm intelligence
- Personal and global best tracking
- Velocity-based position updates

#### MicroController (PID Control)
- Real-time feedback control
- Proportional-Integral-Derivative terms
- Anti-windup protection

---

### 2. Unified System (`unified_system.py`)

Six-level architecture from hardware to generation:

| Level | Component | Input | Output | Purpose |
|-------|-----------|-------|--------|---------|
| 0 | HardwareLevel | Sensors | Telemetry | Raw data collection |
| 1 | SignalLevel | Telemetry | Control signals | Signal processing |
| 2 | LearningLevel | State vector | Action, confidence | Decision making |
| 3 | PredictionLevel | Telemetry | Future states | Anticipation |
| 4 | EmergenceLevel | Telemetry, objective | Evolution, phase | Self-organization |
| 5 | GenerationLevel | Context | Presets | Configuration synthesis |

---

### 3. LLM Client (`llm_client.py`)

Unified interface for local and API LLM providers:

#### Local Providers (ProviderType.LOCAL)
- **Ollama**: Self-hosted models (llama2, mistral, etc.)
- **LM Studio**: Desktop LLM application
- **vLLM**: High-performance serving

#### API Providers (ProviderType.API)
- **OpenAI**: GPT-3.5, GPT-4
- **Anthropic**: Claude 3 family
- **Google**: Gemini 1.5

#### Provider Selection
```
Priority: Local first (lower latency, privacy)
1. Check OLLAMA_HOST → Ollama
2. Check LMSTUDIO_HOST → LM Studio
3. Check VLLM_HOST → vLLM
4. Check OPENAI_API_KEY → OpenAI
5. Check ANTHROPIC_API_KEY → Claude
6. Check GEMINI_API_KEY → Gemini
7. Fallback → Mock
```

---

### 4. Generative Platform (`generative_platform.py`)

Multi-agent LLM orchestration system:

#### Agents
| Agent | Role | Capabilities |
|-------|------|--------------|
| PlannerAgent | Task decomposition | Break complex tasks into steps |
| CoderAgent | Code generation | Write code using LLM |
| CriticAgent | Quality review | Score and critique artifacts |
| GuardianAgent | Safety | Policy enforcement, blocking |
| OptimizerAgent | Performance | KrystalSDK integration |

#### Pipeline Flow
```
Request → Planner → Coder → Critic → Guardian → [Approve/Reject]
                                         ↓
                              Admin Control Panel
```

---

### 5. Emergent Intelligence (`emergent_intelligence.py`)

Self-organization and emergence components:

#### AttractorLandscape
- Natural convergence to optimal states
- Basin of attraction dynamics
- Multi-attractor support

#### PhaseTransitionEngine
- Critical temperature detection
- Phase state management
- Exploration rate modulation

#### CollectiveIntelligence
- Swarm-based optimization
- Diversity maintenance
- Consensus formation

#### SynapseNetwork
- Hebbian learning between components
- Connection strength adaptation
- Path strength computation

---

### 6. Breakthrough Engine (`breakthrough_engine.py`)

Next-generation optimization concepts:

- **TemporalPredictor**: Predict future states, pre-execute
- **NeuralHardwareFabric**: Backprop through hardware settings
- **QuantumInspiredOptimizer**: Superposition states, annealing
- **SelfModifyingEngine**: Runtime code generation
- **SwarmIntelligence**: Distributed fleet learning

---

## Data Flow

### Observe → Decide → Reward Loop

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│  Observe │ ──► │  Decide  │ ──► │  Reward  │
│  (state) │     │ (action) │     │ (signal) │
└──────────┘     └──────────┘     └──────────┘
     ▲                                  │
     │                                  │
     └────────── Learning ◄─────────────┘
```

### Multi-Level Processing

```
Telemetry ──► Signal ──► Learning ──► Prediction
                             │              │
                             ▼              ▼
                        Emergence ◄─── Generation
                             │
                             ▼
                      Output Preset
```

---

## Configuration System

### Layered Configuration

```
Priority (highest to lowest):
1. Environment variables
2. Config file (YAML/TOML)
3. Programmatic settings
4. Defaults
```

### Key Configuration Options

```yaml
# System layers
hardware:
  enabled: true
  poll_interval: 100ms

learning:
  learning_rate: 0.1
  gamma: 0.95
  exploration_decay: 0.99

emergence:
  swarm_size: 10
  phase_threshold: 0.5

# LLM settings
llm:
  provider: auto
  timeout: 60
  retry_count: 3
```

---

## Integration Points

### Python Guardian ↔ C Core
```
unified_system.py → guardian_hooks.py → thread_boost_layer.c
```

### Python Guardian ↔ Rust Bot
```
unified_system.py → kernel_bridge.py → orchestration.rs
```

### LLM Client ↔ All Agents
```
llm_client.py ← generative_platform.py (agents)
             ← unified_system.py (hints)
             ← examples/*.py (demos)
```

---

## Performance Characteristics

| Component | Latency | Throughput | Memory |
|-----------|---------|------------|--------|
| KrystalSDK core | <1ms | 10k+ ops/sec | <10MB |
| Phase transitions | <0.1ms | 100k+ ops/sec | <1MB |
| Swarm optimization | ~10ms | 1k+ ops/sec | ~50MB |
| LLM (local) | 100-500ms | 1-10 req/sec | Varies |
| LLM (API) | 200-2000ms | Rate limited | N/A |

---

## Security Considerations

1. **API Keys**: Never hardcoded, loaded from environment
2. **Rate Limiting**: Built into LLM client
3. **Content Filtering**: GuardianAgent policies
4. **Audit Logging**: Admin control panel
5. **Circuit Breaker**: Retry with backoff

---

## See Also

- [Quick Start](QUICKSTART.md)
- [API Reference](API_REFERENCE.md)
- [Roadmap](ROADMAP.md)
- [Examples](../examples/)
