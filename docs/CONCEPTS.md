# KrystalSDK Core Concepts

## Overview

This document explains the theoretical foundations and core concepts that power KrystalSDK's adaptive intelligence system.

---

## 1. Temporal Difference Learning

### The Idea

TD-Learning enables the system to learn from experience without waiting for final outcomes. It updates predictions based on the difference between consecutive estimates.

### Mathematical Foundation

```
V(s) ← V(s) + α × [r + γ × V(s') - V(s)]
       ─────   ─────────────────────────
       learning        TD Error
         rate
```

Where:
- `V(s)` = Value estimate for current state
- `α` = Learning rate (how fast to adapt)
- `r` = Immediate reward received
- `γ` = Discount factor (importance of future rewards)
- `V(s')` = Value estimate for next state

### Why It Matters

Unlike Monte Carlo methods that wait for episode completion, TD-Learning:
- Updates after every step (real-time adaptation)
- Works with continuous tasks (no episode boundaries)
- Bootstraps from its own estimates (efficient learning)

### In KrystalSDK

```python
class MicroLearner:
    def update(self, state, reward, next_state):
        current_value = self.predict(state)
        next_value = self.predict(next_state)

        # TD Error: how wrong was our prediction?
        td_error = reward + self.gamma * next_value - current_value

        # Update weights toward better predictions
        self.weights += self.learning_rate * td_error * state

        return td_error  # Signal for meta-learning
```

---

## 2. Phase Transitions

### Physical Inspiration

Matter exists in phases: solid, liquid, gas, plasma. Transitions between phases happen at critical temperatures where small changes cause dramatic shifts in behavior.

```
     Energy
        │
        │          ┌─── Plasma (creative/chaotic)
        │      ┌───┘
        │  ┌───┘    ┌─── Gas (high exploration)
        │  │    ┌───┘
        ├──┼────┼─────── Liquid (balanced)
        │  │    │
        └──┴────┴─────── Solid (stable/exploiting)

        Temperature ────────────────────►
```

### Exploration vs Exploitation

The fundamental tradeoff in optimization:
- **Exploration**: Try new things, discover unknown regions
- **Exploitation**: Use what you know, maximize immediate reward

Phase transitions naturally balance this:

| Phase | Temperature | Exploration | Behavior |
|-------|-------------|-------------|----------|
| SOLID | Low | 5% | Exploit known good solutions |
| LIQUID | Medium | 30% | Balance exploration/exploitation |
| GAS | High | 70% | Broad search, high variance |
| PLASMA | Critical | 95% | Breakthrough mode, radical changes |

### Transition Dynamics

```python
class MicroPhase:
    def update(self, reward):
        # Temperature rises when stuck (low rewards)
        if reward < self.baseline:
            self.temperature += self.heat_rate
        else:
            # Cool down when finding good solutions
            self.temperature *= self.cool_rate

        # Phase determined by temperature
        if self.temperature > 0.9:
            return Phase.PLASMA
        elif self.temperature > 0.6:
            return Phase.GAS
        elif self.temperature > 0.3:
            return Phase.LIQUID
        else:
            return Phase.SOLID
```

### Critical Phenomena

At phase boundaries, the system exhibits:
- **Critical slowing down**: Longer correlation times
- **Diverging fluctuations**: Large behavioral variance
- **Scale invariance**: Similar patterns at all scales

These properties enable finding novel solutions unreachable by gradient methods.

---

## 3. Swarm Intelligence

### Biological Inspiration

Ant colonies, bird flocks, and fish schools achieve collective intelligence through simple local rules. No central control, yet complex adaptive behavior emerges.

### Particle Swarm Optimization (PSO)

Each particle represents a candidate solution:

```
┌────────────────────────────────────────────────────────┐
│                    Search Space                         │
│                                                         │
│     ○───→  Particle 1 (velocity toward best)           │
│         ↘                                               │
│           ★ Global Best                                 │
│         ↗                                               │
│     ○───→  Particle 2                                   │
│                                                         │
│     ○─→    Particle 3                                   │
│       ↘                                                 │
│         • Personal Best (particle 3)                    │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### Update Rules

```python
# Velocity update
velocity = (
    w * velocity +                    # Inertia (momentum)
    c1 * r1 * (personal_best - pos) + # Cognitive (memory)
    c2 * r2 * (global_best - pos)     # Social (swarm)
)

# Position update
position = position + velocity
```

Parameters:
- `w` = Inertia weight (0.4-0.9)
- `c1` = Cognitive coefficient (personal learning)
- `c2` = Social coefficient (swarm learning)
- `r1, r2` = Random factors (stochasticity)

### Emergent Properties

From simple rules, complex behavior emerges:
- **Diversity maintenance**: Particles explore different regions
- **Information sharing**: Good solutions propagate
- **Robustness**: Failure of individuals doesn't stop swarm
- **Parallel search**: Multiple candidates evaluated simultaneously

---

## 4. PID Control

### Control Theory Basics

PID (Proportional-Integral-Derivative) control maintains a variable at a setpoint by continuously adjusting based on error feedback.

```
Setpoint ──┬──► Error = Setpoint - Current
           │           │
           │     ┌─────┴─────┐
           │     │           │
           │     ▼           ▼
           │  ┌─────┐    ┌─────┐    ┌─────┐
           │  │  P  │    │  I  │    │  D  │
           │  │     │    │     │    │     │
           │  └──┬──┘    └──┬──┘    └──┬──┘
           │     │          │          │
           │     └────┬─────┴─────┬────┘
           │          │           │
           │          ▼           │
           │    ┌──────────┐      │
Current ◄──┴────│  Output  │◄─────┘
                └──────────┘
```

### Three Terms

```python
output = Kp * error + Ki * integral + Kd * derivative
```

| Term | Formula | Purpose | Tuning |
|------|---------|---------|--------|
| **P** (Proportional) | `Kp × error` | React to current error | Higher = faster, may overshoot |
| **I** (Integral) | `Ki × Σerror` | Eliminate steady-state error | Higher = eliminates offset, may oscillate |
| **D** (Derivative) | `Kd × Δerror` | Predict future error | Higher = dampens oscillations, noise sensitive |

### Anti-Windup

When output saturates (hits limits), integral term can accumulate excessively:

```python
class MicroController:
    def update(self, setpoint, current):
        error = setpoint - current

        # Anti-windup: clamp integral
        self.integral += error
        self.integral = clamp(self.integral, -self.max_integral, self.max_integral)

        derivative = error - self.last_error
        self.last_error = error

        return self.kp * error + self.ki * self.integral + self.kd * derivative
```

---

## 5. Attractor Dynamics

### Dynamical Systems View

The system's state evolves through a landscape of attractors - stable configurations that "pull" nearby states toward them.

```
Energy Landscape
        │
    ────┴─────────────────────────────
        \     /\      /\         /
         \   /  \    /  \       /
          \_/    \__/    \_____/
           ↑       ↑        ↑
          A1      A2       A3
       (local)  (local)  (global)

A1, A2 = Local attractors (local optima)
A3 = Global attractor (global optimum)
```

### Basin of Attraction

Each attractor has a "basin" - the set of starting points that eventually converge to it:

```python
class AttractorLandscape:
    def __init__(self):
        self.attractors = []  # Known stable points
        self.basins = {}      # Mapping of states to attractors

    def find_basin(self, state):
        """Which attractor will this state converge to?"""
        # Simulate forward until convergence
        while not self.is_stable(state):
            state = self.step(state)
        return self.nearest_attractor(state)
```

### Multi-Stability

Real systems have multiple attractors. The challenge is:
1. Finding all attractors (not just one local minimum)
2. Characterizing basins (where does each attractor "pull" from?)
3. Transitioning between attractors (escaping local minima)

Phase transitions enable basin hopping - high temperature allows escaping local attractors to find better ones.

---

## 6. Emergence

### Definition

Emergence: Complex system-level behavior arising from simple component interactions that cannot be predicted by analyzing components in isolation.

```
Component Level          System Level
     ○                      ┌────────┐
    /|\                     │ SWARM  │
     │       ───────►       │BEHAVIOR│
    / \                     └────────┘
  Simple                    Complex
   Rules                   Intelligence
```

### Examples in KrystalSDK

| Component Behavior | Emergent Phenomenon |
|-------------------|---------------------|
| TD updates toward reward | Policy improvement |
| Temperature → phase | Exploration schedule |
| Particles follow best | Global optimization |
| Connection strengthening | Memory formation |

### Measuring Emergence

```python
class EmergenceDetector:
    def measure(self, components, system):
        # Emergence = System complexity - Sum of component complexities
        component_entropy = sum(entropy(c) for c in components)
        system_entropy = entropy(system)

        # Positive = emergent behavior
        # Zero = reducible to components
        # Negative = suppressive interactions
        return system_entropy - component_entropy
```

---

## 7. Hierarchical Learning

### Multi-Timescale Adaptation

Different aspects adapt at different rates:

```
Timescale        What Adapts           Rate
─────────────────────────────────────────────
Milliseconds     PID output            Very fast
Seconds          TD value estimates    Fast
Minutes          Phase temperature     Medium
Hours            Swarm global best     Slow
Days             Attractor landscape   Very slow
```

### Why Hierarchy Matters

- **Fast loops** handle immediate disturbances
- **Slow loops** capture long-term patterns
- **Separation of concerns** prevents oscillations
- **Robustness** to different types of change

### Implementation Pattern

```python
class HierarchicalLearner:
    def __init__(self):
        self.fast = MicroController(kp=1.0)      # Reactive
        self.medium = MicroLearner(lr=0.1)       # Adaptive
        self.slow = MicroSwarm(particles=10)     # Exploratory

    def decide(self, state, setpoint):
        # Fast: immediate correction
        correction = self.fast.update(setpoint, state.current)

        # Medium: learned adjustment
        adjustment = self.medium.predict(state.features)

        # Slow: global guidance (occasional)
        if self.should_explore():
            guidance = self.slow.get_direction()
        else:
            guidance = 0

        return correction + adjustment + guidance
```

---

## 8. Information Flow

### Observe-Decide-Act Loop

```
           ┌──────────────────────────────────────┐
           │                                      │
           ▼                                      │
     ┌──────────┐                                 │
     │ OBSERVE  │ ── state vector ──┐             │
     └──────────┘                   │             │
                                    ▼             │
                              ┌──────────┐        │
                              │  DECIDE  │        │
                              └────┬─────┘        │
                                   │              │
                              action vector       │
                                   │              │
                                   ▼              │
                              ┌──────────┐        │
                              │   ACT    │────────┤
                              └──────────┘        │
                                   │              │
                                reward            │
                                   │              │
                                   ▼              │
                              ┌──────────┐        │
                              │  LEARN   │────────┘
                              └──────────┘
```

### State Representation

What makes a good state representation?
- **Markov property**: State contains all relevant history
- **Discriminability**: Different situations → different states
- **Smoothness**: Similar situations → similar states
- **Compact**: Low-dimensional for efficient learning

### Action Space Design

| Type | Example | Pros | Cons |
|------|---------|------|------|
| Discrete | {low, medium, high} | Simple, guaranteed coverage | Coarse control |
| Continuous | [0.0, 1.0] | Fine-grained | Harder optimization |
| Hybrid | Discrete + continuous params | Flexible | Complex |

---

## 9. Reward Shaping

### The Reward Hypothesis

All goals can be expressed as maximizing cumulative reward. But designing good rewards is challenging.

### Common Pitfalls

| Problem | Example | Solution |
|---------|---------|----------|
| Sparse reward | Win/lose only at end | Add intermediate rewards |
| Reward hacking | Exploit loopholes | Robust reward specification |
| Multi-objective | FPS vs temperature | Weighted sum or Pareto |
| Delayed reward | Action effects later | TD learning with γ |

### Reward Design Patterns

```python
# Simple weighted sum
reward = w1 * performance + w2 * efficiency - w3 * risk

# Threshold-based
reward = 1.0 if meets_target else -0.1

# Shaping (potential-based)
reward = actual_reward + γ * potential(next_state) - potential(state)

# Curiosity bonus
reward = actual_reward + η * novelty(state)
```

---

## 10. Convergence and Stability

### When Does Learning Converge?

TD-Learning converges under conditions:
1. **All states visited infinitely often** (exploration)
2. **Learning rate decreases appropriately** (α → 0, Σα = ∞)
3. **Bounded rewards** (no explosions)

### Stability Analysis

For PID control, stability requires:
- Proper gain tuning (Kp, Ki, Kd)
- System not too fast/slow for controller
- Sensor/actuator bandwidth sufficient

For swarm optimization:
- Velocity limits prevent explosion
- Inertia weight balances exploration/exploitation
- Sufficient diversity maintained

### Monitoring Convergence

```python
class ConvergenceMonitor:
    def __init__(self, window=100, threshold=0.01):
        self.history = []
        self.window = window
        self.threshold = threshold

    def update(self, metric):
        self.history.append(metric)
        if len(self.history) >= self.window:
            recent = self.history[-self.window:]
            variance = np.var(recent)
            return variance < self.threshold  # Converged?
        return False
```

---

## Concept Interactions

### How Concepts Work Together

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   TD-Learning ◄───────────────► Phase Transitions           │
│        │                              │                     │
│        │ value estimates              │ exploration rate    │
│        │                              │                     │
│        ▼                              ▼                     │
│   ┌─────────────────────────────────────────────┐          │
│   │              DECISION ENGINE                 │          │
│   └─────────────────────────────────────────────┘          │
│        ▲                              ▲                     │
│        │ global guidance              │ immediate control   │
│        │                              │                     │
│   Swarm Intelligence ◄───────────► PID Control              │
│                                                             │
│                    ▲         ▲                              │
│                    │         │                              │
│              Attractor    Emergence                         │
│              Dynamics     Detection                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Synergies

| Concept A | Concept B | Synergy |
|-----------|-----------|---------|
| TD-Learning | Phase | Temperature based on TD error magnitude |
| Swarm | Phase | High temperature → more particle diversity |
| PID | TD | TD provides setpoints, PID tracks them |
| Emergence | Swarm | Collective patterns detected and amplified |
| Attractor | Phase | Phase transitions enable attractor hopping |

---

## Summary

KrystalSDK integrates multiple optimization paradigms:

1. **TD-Learning**: Learn from experience, predict future value
2. **Phase Transitions**: Balance exploration and exploitation naturally
3. **Swarm Intelligence**: Parallel search, collective optimization
4. **PID Control**: Precise real-time feedback control
5. **Attractor Dynamics**: Stable configurations and basin analysis
6. **Emergence**: System-level intelligence from simple rules
7. **Hierarchical Learning**: Multi-timescale adaptation
8. **Information Flow**: Clean observe-decide-act architecture
9. **Reward Shaping**: Effective goal specification
10. **Convergence**: Stability and termination guarantees

Together, these concepts create an adaptive system that:
- Learns continuously from experience
- Automatically adjusts exploration strategy
- Finds globally optimal solutions
- Maintains stable operation
- Exhibits emergent intelligent behavior

---

## Further Reading

- Sutton & Barto: *Reinforcement Learning: An Introduction*
- Kennedy & Eberhart: *Swarm Intelligence*
- Strogatz: *Nonlinear Dynamics and Chaos*
- Kauffman: *The Origins of Order*
- Åström & Murray: *Feedback Systems*
