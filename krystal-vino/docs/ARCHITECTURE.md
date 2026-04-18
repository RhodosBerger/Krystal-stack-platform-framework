# Krystal Vino - Architecture & Design Principles

## Overview

Krystal Vino is a next-generation hardware optimization platform built on **statistical governors** instead of traditional deterministic limits. This document outlines the architecture, design principles, and how it improves upon the original Krystal Stack Platform.

## Key Improvements Over Original Codebase

### 1. **Deterministic → Statistical Decision Making**

**Original Approach:**
- Hard limits: "Never exceed 100 RPM"
- Binary safety rules: on/off
- No adaptation to conditions

**New Approach:**
- Probabilistic limits: "Operate safely at 85-105 RPM with 95% confidence"
- Statistical models: Learn from observations
- Adaptive governors: Adjust based on material, tool, environmental conditions

### 2. **Monolithic → Modular Architecture**

**Original:**
- 100+ Python modules tightly coupled
- Complex orchestration with Shadow Council, Dopamine Engine, etc.
- Hard to reason about critical paths

**New:**
- Rust core: Fast, safe, measurable
- Python orchestration: High-level intelligence
- Clean separation: Real-time control vs. intelligence

### 3. **Performance & Safety**

**Original:**
- Python: 10-100ms decision latency
- Thread safety: Hard to reason about with GIL
- Memory: Garbage collection pauses unpredictable

**New:**
- Rust: <1ms decision latency
- Memory safety: Compile-time guarantees
- Real-time: No garbage collection pauses

## System Architecture

```
┌────────────────────────────────────────────────┐
│         Python Intelligence Layer               │
│  - LLM Router, Orchestrator, Knowledge Engine  │
│  - High-level decision making (10-100ms ok)   │
└────────────┬─────────────────────────────────┘
             │
             │ gRPC / Async Message Bus
             │
┌────────────▼─────────────────────────────────┐
│      Rust Real-Time Control Core               │
│  ┌──────────────────────────────────────────┐ │
│  │      Control Runtime                      │ │
│  │  - Governor Registry & Orchestration     │ │
│  │  - Decision Making (<1ms)                │ │
│  │  - Metrics Collection                    │ │
│  └──────────────────────────────────────────┘ │
│              ▲              ▲                  │
│              │              │                  │
│        ┌─────┴──┬───┬──────┴──┐               │
│        │         │   │         │               │
│    ┌───▼──┐ ┌────▼─┐ ▼──┐ ┌──▼────┐          │
│    │Speed │ │Vibr. │Temp│ │ Load  │  ...    │
│    │Gov   │ │Gov   │Gov │ │ Gov   │          │
│    └──────┘ └──────┘────┘ └───────┘          │
│      Statistical Governors (Pluggable)        │
└────────────────────────────────────────────────┘
             │
             │ HAL / CNC Interface
             │
┌────────────▼────────────────────────────────┐
│         Hardware (CNC Machines)              │
└─────────────────────────────────────────────┘
```

## Core Concepts

### Governors

A **Governor** is a trait-based safety control mechanism:

```rust
pub trait Governor: Send + Sync + Debug {
    fn decide(&mut self, input: &Value, context: &Value) -> Result<GovernorDecision>;
    fn observe(&mut self, metrics: &Value) -> Result<()>;
    // ... other methods
}
```

**Examples:**
- **Speed Governor**: Adapts safe RPM based on vibration, tool wear
- **Thermal Governor**: Reduces load when temperature rises
- **Load Governor**: Prevents overload based on motor current patterns
- **Vibration Governor**: Modifies feedrate based on oscillations

### Statistical Models

Each governor maintains:
- **Mean, variance, confidence intervals** of safe operation
- **Bayesian updates** as new safe data arrives
- **Adaptive thresholds** that adjust to conditions

Instead of:
```python
MAX_SPEED = 100  # Fixed forever
```

We have:
```rust
safe_speed = observed_mean - 2*std_dev  // Adapts to observed variance
confidence = bayesian_update(new_observation)
```

### Decision Quality

Decisions include:
- **Allowed**: Yes/No based on statistical safety
- **Confidence**: 0.0-1.0 (high = well-trained model, low = few observations)
- **Safe Limit**: Recommended maximum/minimum
- **Reason**: Human-readable explanation

## Design Principles

### 1. **Safety First**
- When in doubt, be conservative
- Fail-safe mode available
- Multiple governors vote (all must agree)

### 2. **Fast & Predictable**
- Rust guarantees no GC pauses
- Governors run in <1ms
- No allocations in hot path

### 3. **Observable**
- Every decision logged with tracing
- Metrics exposed for monitoring
- Explainability built-in

### 4. **Composable**
- Plug new governors without recompilation
- Governors can chain (A → B → C)
- Shared metrics foundation

### 5. **Learning from Data**
- Governors observe safe operation
- Statistical models improve over time
- Can detect anomalies (vibration, temperature)

## File Structure

```
krystal-vino/
├── core/                          # Rust core library
│   ├── src/
│   │   ├── lib.rs                 # Entry point
│   │   ├── governor/              # Governor trait & implementations
│   │   ├── metrics/               # Metrics collection
│   │   ├── runtime/               # Control runtime
│   │   ├── telemetry/             # Logging & observability
│   │   └── error.rs               # Error types
│   ├── examples/                  # Example governors
│   └── Cargo.toml
├── orchestrator/                  # Python orchestration layer
│   ├── main.py                    # FastAPI server
│   ├── governors.py               # Python governor interface
│   ├── inference.py               # OpenVINO integration
│   └── requirements.txt
├── extensions/                    # Custom governor implementations
│   ├── speed_governor/
│   ├── thermal_governor/
│   └── ...
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # This file
│   ├── DESIGN_PATTERNS.md         # Patterns & best practices
│   └── API.md                     # API reference
└── examples/                      # End-to-end examples
```

## Development Workflow

### Adding a New Governor

1. **Implement the Governor trait** in `core/src/governor/`
   ```rust
   impl Governor for MyGovernor {
       fn decide(&mut self, input, context) -> Result<GovernorDecision> { ... }
       // ...
   }
   ```

2. **Test locally** with `cargo test`

3. **Expose via Python** using pyo3 bindings

4. **Use in orchestrator:**
   ```python
   runtime.register_governor("my_gov", MyGovernor(...))
   decision = runtime.decide(metrics)
   ```

### Performance Targets

- **Decision latency**: <1ms per governor
- **Confidence at 100 samples**: >0.85
- **Memory overhead**: <100KB per governor
- **CPU usage**: <5% on single core for 1000Hz metrics

## Comparison Table

| Aspect | Original | Krystal Vino |
|--------|----------|--------------|
| Language | Python | Rust (core) + Python |
| Decision Speed | 10-100ms | <1ms |
| Safety Model | Deterministic | Statistical |
| Adaptability | None | Full |
| Thread Safety | Python GIL | Lock-free (mostly) |
| Extensibility | Classes | Trait system |
| Observability | Logging | Structured tracing |
| Testing | Unit tests | Property-based |

## Next Steps

1. Implement core governors (speed, thermal, load, vibration)
2. Build Python orchestration layer with gRPC bindings
3. Integrate with OpenVINO for inference
4. Real-world testing on CNC hardware
5. Performance profiling and optimization
