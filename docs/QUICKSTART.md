# KrystalSDK Quick Start Guide

## Installation

### Basic (Zero Dependencies)
```bash
git clone https://github.com/your-org/krystal-sdk
cd krystal-sdk
python -m src.python.krystal_sdk health
```

### Full Installation
```bash
pip install -e .[full]
# Includes: numpy, fastapi, uvicorn
```

### Development
```bash
pip install -e .[dev]
# Includes: pytest, black, mypy
```

---

## 5-Minute Tutorial

### 1. Basic Adaptive Loop

```python
from src.python.krystal_sdk import Krystal

# Create optimizer
k = Krystal()

# Run optimization loop
for i in range(100):
    # Observe current state
    k.observe({"metric_a": 0.5, "metric_b": 0.7})

    # Get recommended action
    action = k.decide()  # Returns [0.1, 0.8, 0.3, 0.5]

    # Apply action and compute reward
    reward = compute_your_reward(action)

    # Feed back to learner
    k.reward(reward)

print(k.get_metrics())
```

### 2. Pre-configured Optimizers

```python
from src.python.krystal_sdk import (
    create_game_optimizer,
    create_server_optimizer,
    create_ml_optimizer,
    create_iot_optimizer
)

# Game FPS/thermal optimization
game = create_game_optimizer()
game.observe({"fps": 0.8, "temp": 0.6, "power": 0.5})
settings = game.decide()

# Server autoscaling
server = create_server_optimizer()
server.observe({"cpu": 0.9, "latency": 0.3})
scale_action = server.decide()
```

### 3. LLM Integration

```python
from src.python.llm_client import LLMClient, Message

# Auto-detect provider (local first, then API)
client = LLMClient()

# Simple chat
response = client.chat("Explain machine learning")
print(response)

# With system prompt
messages = [
    Message("system", "You are a coding expert"),
    Message("user", "Write a Python sort function")
]
response = client.complete(messages)
print(response.content)
```

### 4. Web Dashboard

```bash
# Start dashboard
python -m src.python.web_dashboard

# Open http://localhost:8000
```

### 5. Generative Platform

```python
from src.python.generative_platform import create_generative_platform

platform = create_generative_platform()

# Generate code
artifact = platform.generate({
    "task": "create sorting algorithm",
    "language": "python"
}, actor="developer")

print(artifact.content)
print(f"Quality: {artifact.quality_score}")
```

---

## Configuration

### Environment Variables

```bash
# LLM Providers
export OLLAMA_HOST=http://localhost:11434
export OPENAI_API_KEY=sk-xxx
export ANTHROPIC_API_KEY=sk-ant-xxx
export GEMINI_API_KEY=xxx

# Override defaults
export LLM_PROVIDER=ollama
export LLM_MODEL=llama2
export LLM_MAX_TOKENS=2048
```

### Config File (config.yaml)

```yaml
llm:
  provider: "auto"
  model: "gpt-3.5-turbo"
  max_tokens: 1024
  temperature: 0.7

learning:
  enabled: true
  params:
    learning_rate: 0.1
    gamma: 0.95

emergence:
  enabled: true
  params:
    swarm_size: 10

log_level: "info"
```

---

## CLI Commands

```bash
# Health check
python -m src.python.krystal_sdk health
python -m src.python.krystal_sdk health --json

# Benchmark
python -m src.python.krystal_sdk bench

# Demo
python -m src.python.krystal_sdk demo --cycles 100

# LLM status
python -m src.python.llm_client status
python -m src.python.llm_client chat --prompt "Hello"

# Dashboard
python -m src.python.web_dashboard
```

---

## Next Steps

1. Read [Architecture Guide](ARCHITECTURE.md)
2. Explore [Examples](../examples/)
3. Check [API Reference](API_REFERENCE.md)
4. See [Roadmap](ROADMAP.md)
