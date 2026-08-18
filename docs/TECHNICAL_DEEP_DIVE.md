# KrystalSDK Technical Deep Dive

## Table of Contents
1. [Learning Algorithms](#learning-algorithms)
2. [Phase Transition Mathematics](#phase-transition-mathematics)
3. [Swarm Intelligence Theory](#swarm-intelligence-theory)
4. [LLM Integration Patterns](#llm-integration-patterns)
5. [Performance Engineering](#performance-engineering)
6. [Distributed Systems Design](#distributed-systems-design)

---

## 1. Learning Algorithms

### 1.1 Temporal Difference Learning

KrystalSDK uses TD(0) learning as its core value estimation method. The update rule is:

```
V(s) ← V(s) + α[r + γV(s') - V(s)]
```

Where:
- `V(s)` = Value of current state
- `α` = Learning rate (default: 0.1)
- `r` = Immediate reward
- `γ` = Discount factor (default: 0.95)
- `V(s')` = Value of next state

#### Implementation Details

```python
class MicroLearner:
    def __init__(self, dim=8, lr=0.1, gamma=0.95):
        self.weights = [random.gauss(0, 0.1) for _ in range(dim)]
        self.lr = lr
        self.gamma = gamma

    def predict(self, state):
        """Linear function approximation: V(s) = w · s"""
        return sum(w * s for w, s in zip(self.weights, state))

    def update(self, state, reward, next_state):
        """TD(0) update with gradient descent."""
        current_value = self.predict(state)
        next_value = self.predict(next_state)
        td_target = reward + self.gamma * next_value
        td_error = td_target - current_value

        # Gradient update: w ← w + α * δ * ∇V(s)
        for i, s in enumerate(state):
            self.weights[i] += self.lr * td_error * s

        return td_error
```

#### Convergence Properties

Under certain conditions, TD(0) converges to the optimal value function:

1. **Robbins-Monro conditions**: Learning rate must satisfy:
   - Σα_t = ∞ (infinite exploration)
   - Σα_t² < ∞ (bounded variance)

2. **Markov property**: State transitions must be Markovian

3. **Bounded rewards**: Rewards must be bounded

In practice, we use a fixed learning rate for online adaptation, trading theoretical guarantees for responsiveness.

### 1.2 Experience Replay (Future Enhancement)

For improved sample efficiency, prioritized experience replay weights samples by TD error:

```
P(i) = (|δ_i| + ε)^α / Σ_j(|δ_j| + ε)^α
```

Where:
- `δ_i` = TD error of transition i
- `ε` = Small constant (prevents zero probability)
- `α` = Prioritization exponent (0 = uniform, 1 = full prioritization)

### 1.3 Policy Derivation

Actions are derived from the learned value function using a softmax policy:

```
π(a|s) = exp(Q(s,a)/τ) / Σ_a' exp(Q(s,a')/τ)
```

Where `τ` is temperature, controlled by the phase transition system.

---

## 2. Phase Transition Mathematics

### 2.1 Statistical Mechanics Foundation

The phase system is inspired by statistical mechanics, treating the optimization landscape as a thermodynamic system.

#### Free Energy Formulation

```
F = E - TS
```

Where:
- `F` = Free energy (to minimize)
- `E` = Internal energy (objective function value)
- `T` = Temperature (exploration parameter)
- `S` = Entropy (diversity measure)

#### Boltzmann Distribution

The probability of being in state `s` follows:

```
P(s) = exp(-E(s)/T) / Z
```

Where `Z = Σ_s exp(-E(s)/T)` is the partition function.

### 2.2 Phase Definitions

| Phase | Temperature Range | Behavior |
|-------|-------------------|----------|
| SOLID | T < 0.25 T_c | Frozen, exploit best known |
| LIQUID | 0.25 T_c ≤ T < T_c | Flowing, balanced exploration |
| GAS | T_c ≤ T < 1.5 T_c | Expanding, high exploration |
| PLASMA | T ≥ 1.5 T_c | Ionized, maximum creativity |

Where `T_c` is the critical temperature (default: 0.5).

### 2.3 Phase Transition Dynamics

```python
class MicroPhase:
    def __init__(self, critical_temp=0.5):
        self.temperature = 0.3
        self.critical = critical_temp
        self.phase = Phase.SOLID

    def update(self, gradient, stability):
        """
        Update temperature based on optimization landscape.

        gradient: |∇f(x)| - magnitude of objective gradient
        stability: 1/(1 + variance) - solution stability
        """
        # Temperature dynamics: exponential moving average
        self.temperature = 0.9 * self.temperature + 0.1 * gradient

        # Phase determination with hysteresis
        if self.temperature < self.critical * 0.5:
            self.phase = Phase.SOLID
        elif self.temperature < self.critical:
            self.phase = Phase.LIQUID
        elif self.temperature < self.critical * 1.5:
            self.phase = Phase.GAS
        else:
            self.phase = Phase.PLASMA

        return self.phase
```

### 2.4 Exploration Rate Function

```python
def exploration_rate(self):
    """Map phase to exploration probability."""
    rates = {
        Phase.SOLID: 0.05,   # 5% random actions
        Phase.LIQUID: 0.20,  # 20% exploration
        Phase.GAS: 0.50,     # 50% exploration
        Phase.PLASMA: 0.90   # 90% random search
    }
    return rates[self.phase]
```

### 2.5 Critical Phenomena

Near the critical temperature, the system exhibits:

1. **Diverging correlation length**: Small changes propagate further
2. **Critical slowing down**: Longer equilibration times
3. **Power law distributions**: Scale-free behavior

These properties are exploited for escaping local minima.

---

## 3. Swarm Intelligence Theory

### 3.1 Particle Swarm Optimization (PSO)

KrystalSDK implements PSO for global optimization:

```
v_i(t+1) = w·v_i(t) + c_1·r_1·(p_best_i - x_i) + c_2·r_2·(g_best - x_i)
x_i(t+1) = x_i(t) + v_i(t+1)
```

Where:
- `v_i` = Velocity of particle i
- `x_i` = Position of particle i
- `w` = Inertia weight (default: 0.7)
- `c_1` = Cognitive coefficient (default: 1.5)
- `c_2` = Social coefficient (default: 1.5)
- `r_1, r_2` = Random numbers in [0, 1]
- `p_best_i` = Personal best position
- `g_best` = Global best position

### 3.2 Implementation

```python
class MicroSwarm:
    def __init__(self, n_particles=5, dim=4):
        # Initialize random positions in [0, 1]^dim
        self.particles = [[random.random() for _ in range(dim)]
                          for _ in range(n_particles)]
        self.velocities = [[0.0] * dim for _ in range(n_particles)]
        self.best_positions = [p[:] for p in self.particles]
        self.best_scores = [-float('inf')] * n_particles
        self.global_best = self.particles[0][:]
        self.global_best_score = -float('inf')

    def step(self, objective):
        w, c1, c2 = 0.7, 1.5, 1.5

        for i, (p, v) in enumerate(zip(self.particles, self.velocities)):
            # Evaluate fitness
            score = objective(p)

            # Update personal best
            if score > self.best_scores[i]:
                self.best_scores[i] = score
                self.best_positions[i] = p[:]

            # Update global best
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best = p[:]

            # Update velocity and position
            for j in range(len(p)):
                r1, r2 = random.random(), random.random()
                v[j] = (w * v[j] +
                        c1 * r1 * (self.best_positions[i][j] - p[j]) +
                        c2 * r2 * (self.global_best[j] - p[j]))
                p[j] = max(0, min(1, p[j] + v[j]))  # Clamp to [0, 1]

        return self.global_best
```

### 3.3 Convergence Analysis

PSO converges when particles cluster around the global best. The convergence rate depends on:

1. **Inertia weight**: Lower w → faster convergence, less exploration
2. **Acceleration coefficients**: Balance personal vs social learning
3. **Swarm topology**: Fully connected vs ring vs random

### 3.4 Diversity Maintenance

To prevent premature convergence:

```python
def diversity(self):
    """Calculate swarm diversity (average pairwise distance)."""
    total = 0
    count = 0
    for i, p1 in enumerate(self.particles):
        for p2 in self.particles[i+1:]:
            dist = sum((a - b) ** 2 for a, b in zip(p1, p2)) ** 0.5
            total += dist
            count += 1
    return total / count if count > 0 else 0

def reinitialize_if_converged(self, threshold=0.01):
    """Reinitialize random particles if swarm converged."""
    if self.diversity() < threshold:
        for i in range(len(self.particles) // 2):
            self.particles[i] = [random.random() for _ in range(self.dim)]
            self.velocities[i] = [0.0] * self.dim
```

---

## 4. LLM Integration Patterns

### 4.1 Provider Abstraction

The LLM client uses the Strategy pattern for provider interchangeability:

```
                    ┌──────────────┐
                    │  LLMProvider │ (Abstract)
                    │  + complete()│
                    │  + stream()  │
                    │  + available │
                    └──────┬───────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
┌──────▼──────┐    ┌───────▼───────┐   ┌───────▼───────┐
│OllamaProvider│   │OpenAIProvider │   │GeminiProvider │
└─────────────┘    └───────────────┘   └───────────────┘
```

### 4.2 Request Flow

```
User Request
     │
     ▼
┌─────────────┐
│  LLMClient  │
│             │
│ 1. Config   │──── Environment / File / Explicit
│ 2. Provider │──── Auto-detect or specified
│ 3. Retry    │──── Exponential backoff
│ 4. Metrics  │──── Track tokens, latency
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Provider   │
│             │
│ 1. Format   │──── Convert to provider format
│ 2. HTTP     │──── Send request
│ 3. Parse    │──── Extract response
│ 4. Return   │──── Unified Response object
└─────────────┘
```

### 4.3 Message Format Conversion

Different providers have different message formats:

```python
# OpenAI format
{
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
    ]
}

# Anthropic format
{
    "model": "claude-3-haiku",
    "system": "...",
    "messages": [
        {"role": "user", "content": "..."}
    ]
}

# Gemini format
{
    "contents": [
        {"role": "user", "parts": [{"text": "..."}]}
    ],
    "systemInstruction": {"parts": [{"text": "..."}]}
}
```

### 4.4 Retry Strategy

```python
def complete_with_retry(self, messages, **kwargs):
    last_error = None

    for attempt in range(self.config.retry_count):
        try:
            return self.provider.complete(messages, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < self.config.retry_count - 1:
                # Exponential backoff: 1s, 2s, 4s, ...
                delay = self.config.retry_delay * (2 ** attempt)
                time.sleep(delay)

    return Response(content=f"Error: {last_error}")
```

### 4.5 Streaming Implementation

For real-time responses:

```python
def stream(self, messages, **kwargs):
    """Generator yielding response tokens."""
    # Ollama example
    data = {
        "model": self.config.model,
        "messages": messages,
        "stream": True
    }

    with urlopen(request) as response:
        for line in response:
            chunk = json.loads(line)
            if "message" in chunk:
                yield chunk["message"]["content"]
```

---

## 5. Performance Engineering

### 5.1 Memory Optimization

#### Object Pooling
```python
class StatePool:
    """Reuse State objects to reduce allocation."""

    def __init__(self, size=100):
        self.pool = [State() for _ in range(size)]
        self.available = list(range(size))

    def acquire(self):
        if self.available:
            idx = self.available.pop()
            return self.pool[idx]
        return State()  # Fallback to new allocation

    def release(self, state):
        state.clear()
        if len(self.available) < len(self.pool):
            self.available.append(self.pool.index(state))
```

#### Numpy Integration (Optional)
```python
# With numpy (if available)
import numpy as np

class FastLearner:
    def __init__(self, dim=8):
        self.weights = np.random.randn(dim) * 0.1

    def predict(self, state):
        return np.dot(self.weights, state)

    def update(self, state, reward, next_state):
        state = np.array(state)
        next_state = np.array(next_state)
        td_error = reward + self.gamma * self.predict(next_state) - self.predict(state)
        self.weights += self.lr * td_error * state
        return td_error
```

### 5.2 Computational Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Observe | O(d) | O(d) |
| Decide | O(d) | O(1) |
| Reward (TD update) | O(d) | O(1) |
| Swarm step | O(n·d) | O(n·d) |
| Phase update | O(h) | O(h) |

Where:
- d = state dimension
- n = number of particles
- h = history length

### 5.3 Benchmarking

```python
def benchmark(iterations=10000):
    k = Krystal()
    start = time.time()

    for _ in range(iterations):
        k.observe({"x": random.random()})
        k.decide()
        k.reward(random.random())

    elapsed = time.time() - start
    ops_per_sec = iterations / elapsed
    return ops_per_sec

# Target: >10,000 ops/sec on modern hardware
```

### 5.4 Profiling Results

Typical breakdown for a single cycle:
- Observe: 5%
- Decide: 60%
- Reward: 30%
- Overhead: 5%

Optimization targets:
1. Vectorize weight updates
2. Reduce object allocations
3. Use slots for dataclasses
4. Profile-guided inlining

---

## 6. Distributed Systems Design

### 6.1 Fleet Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    COORDINATOR                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
│  │   Registry  │  │  Aggregator │  │  Model Server   │ │
│  └─────────────┘  └─────────────┘  └─────────────────┘ │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼────┐    ┌─────▼────┐    ┌────▼────┐
    │ Agent 1 │    │ Agent 2  │    │ Agent N │
    │ (Edge)  │    │ (Cloud)  │    │ (Mobile)│
    └─────────┘    └──────────┘    └─────────┘
```

### 6.2 Federated Learning Protocol

```python
# Agent side
def local_train(agent, data, epochs):
    for _ in range(epochs):
        for sample in data:
            agent.observe(sample.state)
            agent.decide()
            agent.reward(sample.reward)
    return agent.learner.weights

def send_update(weights, coordinator):
    delta = weights - last_global_weights
    encrypted = differential_privacy(delta)
    coordinator.receive(encrypted)

# Coordinator side
def aggregate(updates):
    """FedAvg: weighted average of updates."""
    total_samples = sum(u.n_samples for u in updates)
    aggregated = np.zeros_like(updates[0].weights)

    for update in updates:
        weight = update.n_samples / total_samples
        aggregated += weight * update.weights

    return aggregated

def broadcast(global_weights, agents):
    for agent in agents:
        agent.update_weights(global_weights)
```

### 6.3 Consistency Models

| Model | Latency | Consistency | Use Case |
|-------|---------|-------------|----------|
| Strong | High | Full | Safety-critical |
| Eventual | Low | Delayed | Performance |
| Causal | Medium | Ordered | Dependent updates |

### 6.4 Failure Handling

```python
class ResilientAgent:
    def __init__(self):
        self.checkpoint_interval = 100
        self.last_checkpoint = None

    def checkpoint(self):
        """Save state for recovery."""
        self.last_checkpoint = {
            "weights": self.learner.weights.copy(),
            "cycle": self.cycle,
            "metrics": self.metrics.copy()
        }

    def recover(self):
        """Restore from checkpoint."""
        if self.last_checkpoint:
            self.learner.weights = self.last_checkpoint["weights"]
            self.cycle = self.last_checkpoint["cycle"]
            self.metrics = self.last_checkpoint["metrics"]

    def tick_with_checkpoint(self):
        try:
            result = self.tick()
            if self.cycle % self.checkpoint_interval == 0:
                self.checkpoint()
            return result
        except Exception as e:
            self.recover()
            raise
```

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| s | State vector |
| a | Action vector |
| r | Reward signal |
| V(s) | State value function |
| Q(s,a) | Action value function |
| π(a\|s) | Policy (action probability) |
| α | Learning rate |
| γ | Discount factor |
| τ | Temperature |
| T_c | Critical temperature |
| w | Inertia weight (PSO) |
| c_1, c_2 | Acceleration coefficients |

## Appendix B: Default Parameters

```python
DEFAULT_CONFIG = {
    # Learning
    "state_dim": 8,
    "action_dim": 4,
    "learning_rate": 0.1,
    "gamma": 0.95,

    # Phase
    "critical_temp": 0.5,
    "temp_decay": 0.9,

    # Swarm
    "n_particles": 5,
    "inertia": 0.7,
    "cognitive": 1.5,
    "social": 1.5,

    # PID
    "kp": 1.0,
    "ki": 0.1,
    "kd": 0.05,
}
```

## Appendix C: References

1. Sutton & Barto - Reinforcement Learning: An Introduction
2. Kennedy & Eberhart - Particle Swarm Optimization
3. Kirkpatrick et al. - Optimization by Simulated Annealing
4. McMahan et al. - Communication-Efficient Learning of Deep Networks
