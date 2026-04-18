# Krystal Vino - Design Principles & Best Practices

## Core Design Principles

### 1. **Safety First, Performance Second**

All decisions should err on the side of caution when uncertain.

```rust
// ❌ Bad: Takes risk when confidence low
if vibration < threshold {
    allow_operation()
}

// ✓ Good: Conservative when uncertain
if confidence > 0.9 && vibration < adaptive_threshold {
    allow_operation()
}
```

**Principle**: In safety-critical code, conservative defaults are features, not bugs.

### 2. **Explicit Over Implicit**

Never hide important state or decisions in side effects.

```rust
// ❌ Bad: Side effect in observe()
fn observe(&mut self, metrics: &Value) -> Result<()> {
    self.state = if self.samples > 100 { Active } else { Learning };
    Ok(())
}

// ✓ Good: Explicit state management
fn observe(&mut self, metrics: &Value) -> Result<()> {
    self.samples += 1;
    Ok(())
}

fn check_readiness(&mut self) -> bool {
    if self.samples > 100 && self.state == Initializing {
        self.state = Active;
        true
    } else {
        false
    }
}
```

### 3. **Measurable Confidence, Not Binary**

Every decision includes a confidence score. Decisions are only as good as our understanding.

```rust
// ❌ Bad: No indication of decision quality
GovernorDecision::new(true, "Looks safe")

// ✓ Good: Confidence represents model quality
GovernorDecision::new(true, 0.87, "87% confident based on 234 observations")
```

**Implication**:
- Low confidence (<0.7): Recent reset, limited data, anomalies detected
- Medium confidence (0.7-0.9): Operating normally, good data
- High confidence (>0.9): Well-trained, stable operation

### 4. **Fail-Safe by Design**

When a governor fails or encounters unknown conditions, it should default to safe behavior.

```rust
// ❌ Bad: Panics on unexpected input
let vibration = input["vibration"].as_f64().unwrap();

// ✓ Good: Graceful degradation
let vibration = input.get("vibration")
    .and_then(|v| v.as_f64())
    .unwrap_or_else(|| {
        // No data = unknown = be conservative
        self.state = GovernorState::FailSafe;
        f64::NEG_INFINITY  // Reject operation
    });
```

### 5. **Observability Built In**

Never fight observability for code brevity.

```rust
// ✓ Good: Every decision is traceable
pub fn decide(&mut self, input: &Value, context: &Value) -> Result<GovernorDecision> {
    let metrics = self.extract_metrics(input)?;
    
    telemetry::log_decision(
        self.name(),
        allowed,
        confidence,
        &format!("Vibration: {:.2}g", metrics.vibration)
    );

    Ok(decision)
}
```

### 6. **Composability Through Interfaces**

Governors should compose freely, not fight each other.

```rust
// Good: Different governors make independent decisions
// Runtime checks that ALL governors agree before allowing action
let decisions = runtime.decide_safe(input, context)?;
let all_allowed = decisions.iter().all(|(_, d)| d.allowed);

if all_allowed {
    // Only proceed if speed governor AND vibration governor AND thermal governor agree
    execute_operation()
}
```

### 7. **Learning from Safe Data Only**

Observations should only update models when we know conditions were safe.

```rust
// ❌ Bad: Learns from potentially unsafe operation
fn observe(&mut self, metrics: &Value) -> Result<()> {
    self.update_model(metrics)
}

// ✓ Good: Only learns from validated safe operation
fn observe(&mut self, metrics: &Value, decision: &GovernorDecision) -> Result<()> {
    if decision.allowed && decision.confidence > 0.8 {
        self.update_model(metrics)
    }
    Ok(())
}
```

### 8. **Transparent Decisions**

Every decision must be explainable to operators.

```rust
GovernorDecision::new(false, 0.72, 
    format!("Vibration spike detected: {:.2}g (threshold: {:.2}g) with 72% confidence",
        observed_vibration, safe_threshold))
```

## Implementation Guidelines

### Governor Naming

Use descriptive, domain-specific names:
- ✓ `VibrationGovernor`, `ThermalGovernor`, `ToolWearGovernor`
- ❌ `SafetyCheck1`, `Limiter`, `Constraint`

### Error Handling

Always convert errors to decisions, never propagate:

```rust
// ❌ Bad: Propagates error
fn decide(&mut self, input: &Value, context: &Value) -> Result<GovernorDecision> {
    let speed = input.get("speed").and_then(|v| v.as_f64())?; // Error!
    Ok(GovernorDecision::new(true, 1.0, "ok"))
}

// ✓ Good: Errors become safe decisions
fn decide(&mut self, input: &Value, context: &Value) -> Result<GovernorDecision> {
    let speed = input.get("speed")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| Error::ControlDecisionFailed("Missing speed metric".into()))?;
    
    // If we get here, speed is valid
    Ok(GovernorDecision::new(speed < limit, confidence, reason))
}
```

### Testing

Every governor should have:
1. **Unit tests** for individual decisions
2. **Property-based tests** for edge cases
3. **Integration tests** with the runtime

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_operation() {
        let mut gov = VibrationGovernor::new(Config::default());
        let input = json!({"vibration": 0.3, "speed": 50});
        let decision = gov.decide(&input, &json!({})).unwrap();
        assert!(decision.allowed);
    }

    #[test]
    fn test_reject_high_vibration() {
        let mut gov = VibrationGovernor::new(Config::default());
        let input = json!({"vibration": 5.0, "speed": 100});
        let decision = gov.decide(&input, &json!({})).unwrap();
        assert!(!decision.allowed);
    }
}
```

### Configuration

Governors should be configurable but have sensible defaults:

```rust
let config = GovernorConfig::new("speed_governor")
    .with_threshold(0.9)          // Confidence threshold
    .with_safety_first(true)      // Conservative mode
    .with_params(json!({
        "max_speed": 200,
        "vibration_limit": 2.0,
        "learning_rate": 0.01,
    }));
```

## Performance Considerations

### Latency Budget
- **Decision making**: 500µs (80%)
- **Metric extraction**: 50µs (8%)
- **Logging**: 50µs (12%)

### Memory Budget
- **Per governor**: <100KB
- **Metrics buffer**: Configurable (default 10k samples)
- **Total**: Should scale linearly with governor count

### Concurrency

Use lock-free data structures where possible:

```rust
// ✓ Good: Thread-safe, no locks in hot path
pub struct MetricsCollector {
    metrics: Arc<DashMap<String, Vec<Metric>>>,
}

// Use interior mutability only when necessary
pub struct GovernorRuntime {
    governors: Arc<RwLock<GovernorRegistry>>,
}
```

## Common Patterns

### Pattern: Bounded History

Governors maintain limited history to prevent unbounded growth:

```rust
const MAX_SAMPLES: usize = 1000;

fn record_sample(&mut self, value: f64) {
    self.history.push(value);
    if self.history.len() > MAX_SAMPLES {
        self.history.remove(0);  // Keep most recent
    }
}
```

### Pattern: Confidence Decay

Confidence should decay if model isn't updated:

```rust
fn decide(&mut self, ...) -> Result<GovernorDecision> {
    let time_since_update = Utc::now() - self.last_update;
    let decay = if time_since_update > Duration::hours(24) {
        0.8  // Decay to 80% confidence
    } else {
        1.0
    };

    let effective_confidence = self.base_confidence * decay;
    Ok(GovernorDecision::new(allowed, effective_confidence, reason))
}
```

### Pattern: Multi-Modal Safety

Allow multiple independent safety criteria:

```rust
fn decide(&mut self, input: &Value, context: &Value) -> Result<GovernorDecision> {
    let speed_safe = check_speed_limits(input)?;
    let thermal_safe = check_temperature_limits(input)?;
    let vibration_safe = check_vibration_limits(input)?;

    let all_safe = speed_safe && thermal_safe && vibration_safe;
    let combined_confidence = (sc + tc + vc) / 3.0;

    Ok(GovernorDecision::new(all_safe, combined_confidence, reason))
}
```

## Anti-Patterns to Avoid

### ❌ Magic Numbers

```rust
// Bad
if confidence > 0.85 && vibration < 2.5 {  // Where did these come from?
    allow()
}

// Good
const MIN_CONFIDENCE: f64 = 0.9;
const VIBRATION_THRESHOLD_G: f64 = 2.0;  // Nameplate specification

if confidence > MIN_CONFIDENCE && vibration < VIBRATION_THRESHOLD_G {
    allow()
}
```

### ❌ Hidden State Changes

```rust
// Bad: observe() has side effects
fn observe(&mut self, metrics: &Value) -> Result<()> {
    self.state = if self.samples > 100 { Active } else { Learning };  // Hidden!
    self.update_model(metrics);
    Ok(())
}

// Good: Explicit state transitions
fn observe(&mut self, metrics: &Value) -> Result<()> {
    self.update_model(metrics);
    Ok(())
}

fn transition_if_ready(&mut self) -> Option<GovernorState> {
    if self.samples > 100 && self.state == Initializing {
        self.state = Active;
        Some(Active)
    } else {
        None
    }
}
```

### ❌ Panicking in Production

```rust
// Bad: Will crash the system
let speed = input["speed"].as_f64().unwrap();

// Good: Graceful error handling
let speed = input.get("speed")
    .and_then(|v| v.as_f64())
    .ok_or_else(|| Error::ControlDecisionFailed(...))?;
```

## Validation Checklist

Before deploying a new governor:

- [ ] **Safety**: Defaults to denial on uncertainty
- [ ] **Observable**: All decisions logged
- [ ] **Tested**: Unit + property-based + integration tests
- [ ] **Documented**: Purpose, parameters, thresholds explained
- [ ] **Performant**: <1ms decision latency
- [ ] **Composable**: Works with other governors
- [ ] **Tuneable**: Configurable without recompilation
- [ ] **Resilient**: Handles missing/corrupted input
