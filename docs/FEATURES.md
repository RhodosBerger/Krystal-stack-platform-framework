# KrystalSDK Features

## Core Engine

| Feature | Description | Status |
|---------|-------------|--------|
| **TD-Learning** | Online value estimation with temporal difference updates | Ready |
| **Phase Transitions** | SOLID/LIQUID/GAS/PLASMA exploration modes | Ready |
| **Swarm Optimization** | Particle swarm for global search | Ready |
| **PID Control** | Real-time feedback with anti-windup | Ready |

## LLM Integration

| Provider | Type | Auto-Detection |
|----------|------|----------------|
| Ollama | Local | `OLLAMA_HOST` |
| LM Studio | Local | `LMSTUDIO_HOST` |
| vLLM | Local | `VLLM_HOST` |
| OpenAI | API | `OPENAI_API_KEY` |
| Anthropic Claude | API | `ANTHROPIC_API_KEY` |
| Google Gemini | API | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |

**Key capabilities:**
- Local-first priority (privacy, low latency)
- Unified Response object across all providers
- Retry with exponential backoff
- Streaming support
- Token counting and metrics

## Multi-Agent Platform

| Agent | Role |
|-------|------|
| Planner | Task decomposition |
| Coder | LLM-powered code generation |
| Critic | Quality scoring and review |
| Guardian | Safety policy enforcement |
| Optimizer | KrystalSDK integration |

## 6-Level Architecture

```
Level 0: Hardware    → Telemetry collection
Level 1: Signal      → Signal processing
Level 2: Learning    → TD-based decisions
Level 3: Prediction  → Future state anticipation
Level 4: Emergence   → Self-organization
Level 5: Generation  → Configuration synthesis
```

## Emergent Intelligence

- **Attractor Landscapes**: Basin dynamics, multi-stability
- **Phase Engine**: Critical temperature detection
- **Collective Intelligence**: Swarm consensus
- **Synapse Network**: Hebbian connection adaptation

## Configuration

- YAML/TOML config files
- Environment variable overrides
- `LLM_PROVIDER` / `LLM_MODEL` explicit control
- Programmatic `KrystalConfig` / `LLMConfig`

## Web Dashboard

- REST API (`/api/metrics`, `/api/decide`, `/api/observe`)
- Real-time metrics display
- FastAPI or minimal HTTP server

## CLI Tools

```bash
python -m src.python.krystal_sdk health   # Component check
python -m src.python.krystal_sdk demo     # Run demo
python -m src.python.krystal_sdk bench    # Performance test
python -m src.python.llm_client status    # Provider status
python -m src.python.llm_client chat      # Quick chat
```

## Domain Presets

| Domain | Factory Function | Optimizes |
|--------|------------------|-----------|
| Gaming | `create_game_optimizer()` | FPS, thermal, quality |
| Server | `create_server_optimizer()` | CPU, latency, scaling |
| ML | `create_ml_optimizer()` | Loss, accuracy, hyperparams |
| IoT | `create_iot_optimizer()` | Battery, power, throughput |

## Performance

| Component | Latency | Throughput |
|-----------|---------|------------|
| Core SDK | <1ms | 10k+ ops/sec |
| Phase transitions | <0.1ms | 100k+ ops/sec |
| Swarm optimization | ~10ms | 1k+ ops/sec |
| LLM (local) | 100-500ms | 1-10 req/sec |
| LLM (API) | 200-2000ms | Rate limited |

## Zero Dependencies

Core `krystal_sdk.py` requires only Python stdlib - no numpy, no external packages. Optional integrations available for enhanced functionality.
