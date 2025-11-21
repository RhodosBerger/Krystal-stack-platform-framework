# KrystalSDK Use Cases & Design Patterns

## Table of Contents
1. [Gaming & Graphics](#gaming--graphics)
2. [Cloud & Infrastructure](#cloud--infrastructure)
3. [Machine Learning Operations](#machine-learning-operations)
4. [IoT & Edge Computing](#iot--edge-computing)
5. [Design Patterns](#design-patterns)
6. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)

---

## Gaming & Graphics

### Use Case 1: Dynamic Quality Adjustment

**Problem**: Maintain stable FPS while maximizing visual quality across varying scenes.

**Solution**:
```python
from src.python.krystal_sdk import create_game_optimizer

optimizer = create_game_optimizer()

class GameLoop:
    def __init__(self):
        self.quality_settings = {
            "resolution_scale": 1.0,
            "shadow_quality": "high",
            "texture_quality": "high",
            "antialiasing": "TAA",
            "ray_tracing": True
        }

    def frame(self):
        # Get current metrics
        fps = self.get_fps()
        frame_time = self.get_frame_time()
        gpu_temp = self.get_gpu_temperature()
        gpu_util = self.get_gpu_utilization()

        # Normalize to 0-1 range
        optimizer.observe({
            "fps": fps / 144,           # Target: 144 FPS
            "frame_time": frame_time / 16.67,  # Target: 16.67ms
            "gpu_temp": gpu_temp / 90,  # Max: 90°C
            "gpu_util": gpu_util,
            "quality": self.current_quality_score()
        })

        # Get optimization recommendation
        action = optimizer.decide()

        # Apply settings
        self.quality_settings["resolution_scale"] = 0.5 + action[0] * 0.5  # 50-100%
        self.quality_settings["shadow_quality"] = self.map_quality(action[1])
        self.quality_settings["ray_tracing"] = action[2] > 0.5
        self.apply_settings()

        # Reward: high FPS + high quality - thermal penalty
        reward = (fps / 144) * 0.4 + self.current_quality_score() * 0.4
        if gpu_temp > 85:
            reward -= (gpu_temp - 85) * 0.1  # Thermal penalty
        optimizer.reward(reward)

    def map_quality(self, value):
        if value < 0.33:
            return "low"
        elif value < 0.67:
            return "medium"
        return "high"
```

### Use Case 2: Frame Pacing

**Problem**: Eliminate micro-stutters by predicting frame times.

**Solution**:
```python
class FramePacer:
    def __init__(self):
        self.optimizer = Krystal(KrystalConfig(state_dim=16, action_dim=2))
        self.frame_history = []

    def predict_and_pace(self):
        # Build state from frame history
        state = {f"frame_{i}": t for i, t in enumerate(self.frame_history[-8:])}
        state["variance"] = np.var(self.frame_history[-16:]) if len(self.frame_history) >= 16 else 0

        self.optimizer.observe(state)
        action = self.optimizer.decide()

        # Action[0]: Pre-render budget allocation
        # Action[1]: VSync timing adjustment
        pre_render_ms = 2 + action[0] * 6  # 2-8ms pre-render
        vsync_offset = (action[1] - 0.5) * 2  # -1 to +1 ms offset

        return pre_render_ms, vsync_offset

    def record_frame(self, frame_time):
        self.frame_history.append(frame_time)
        if len(self.frame_history) > 100:
            self.frame_history.pop(0)

        # Reward smooth frames
        if len(self.frame_history) >= 2:
            jitter = abs(self.frame_history[-1] - self.frame_history[-2])
            reward = 1.0 - min(1.0, jitter / 5.0)  # Penalize >5ms jitter
            self.optimizer.reward(reward)
```

### Use Case 3: DLSS/FSR Quality Selection

**Problem**: Automatically select upscaling quality based on scene complexity.

```python
class UpscalerOptimizer:
    MODES = ["native", "quality", "balanced", "performance", "ultra_performance"]

    def __init__(self):
        self.optimizer = Krystal()
        self.current_mode = "quality"

    def select_mode(self, scene_complexity, motion_vectors, gpu_headroom):
        self.optimizer.observe({
            "complexity": scene_complexity,
            "motion": motion_vectors,
            "headroom": gpu_headroom,
            "current_mode": self.MODES.index(self.current_mode) / 4
        })

        action = self.optimizer.decide()
        mode_idx = int(action[0] * 4.99)  # 0-4
        self.current_mode = self.MODES[mode_idx]

        return self.current_mode

    def feedback(self, image_quality_score, performance_gain):
        # Balance quality and performance
        reward = image_quality_score * 0.6 + performance_gain * 0.4
        self.optimizer.reward(reward)
```

---

## Cloud & Infrastructure

### Use Case 4: Kubernetes Pod Autoscaling

**Problem**: Scale pods based on complex metrics beyond CPU/memory.

```python
class K8sAutoscaler:
    def __init__(self, deployment_name, namespace="default"):
        self.deployment = deployment_name
        self.namespace = namespace
        self.optimizer = create_server_optimizer()
        self.min_replicas = 2
        self.max_replicas = 50

    def get_metrics(self):
        """Collect metrics from Prometheus/metrics-server."""
        return {
            "cpu_avg": self.query_prometheus("avg(container_cpu_usage)"),
            "memory_avg": self.query_prometheus("avg(container_memory_usage)"),
            "request_rate": self.query_prometheus("rate(http_requests_total[1m])"),
            "latency_p99": self.query_prometheus("histogram_quantile(0.99, ...)"),
            "error_rate": self.query_prometheus("rate(http_errors_total[1m])"),
            "queue_depth": self.query_prometheus("workqueue_depth")
        }

    def scale(self):
        metrics = self.get_metrics()

        self.optimizer.observe({
            "cpu": metrics["cpu_avg"],
            "memory": metrics["memory_avg"],
            "rps": min(1.0, metrics["request_rate"] / 10000),
            "latency": min(1.0, metrics["latency_p99"] / 500),
            "errors": metrics["error_rate"],
            "queue": min(1.0, metrics["queue_depth"] / 1000)
        })

        action = self.optimizer.decide()

        # Map action to replica count
        target_replicas = int(
            self.min_replicas +
            action[0] * (self.max_replicas - self.min_replicas)
        )

        # Apply scaling
        current = self.get_current_replicas()
        if target_replicas != current:
            self.set_replicas(target_replicas)

        # Reward: low latency, low cost, no errors
        cost_factor = target_replicas / self.max_replicas
        reward = (
            (1 - metrics["latency_p99"] / 500) * 0.4 +
            (1 - metrics["error_rate"]) * 0.3 +
            (1 - cost_factor) * 0.3
        )
        self.optimizer.reward(reward)

        return target_replicas
```

### Use Case 5: Database Query Optimization

**Problem**: Dynamically tune database parameters based on workload.

```python
class DatabaseTuner:
    def __init__(self, connection_string):
        self.db = connect(connection_string)
        self.optimizer = Krystal(KrystalConfig(state_dim=10, action_dim=6))

    def tune(self):
        # Collect database metrics
        stats = self.db.get_statistics()

        self.optimizer.observe({
            "cache_hit_ratio": stats["buffer_cache_hit_ratio"],
            "disk_reads": stats["disk_reads_per_sec"] / 10000,
            "connections": stats["active_connections"] / stats["max_connections"],
            "lock_waits": stats["lock_wait_time"] / 1000,
            "query_time_avg": stats["avg_query_time"] / 100,
            "deadlocks": stats["deadlocks_per_min"],
            "temp_tables": stats["temp_tables_created"] / 100,
            "sort_spills": stats["sort_merge_passes"] / 100
        })

        action = self.optimizer.decide()

        # Apply tuning
        new_params = {
            "shared_buffers": f"{int(1 + action[0] * 15)}GB",
            "work_mem": f"{int(64 + action[1] * 448)}MB",
            "effective_cache_size": f"{int(4 + action[2] * 28)}GB",
            "random_page_cost": 1.1 + action[3] * 2.9,
            "parallel_workers": int(action[4] * 8),
            "checkpoint_completion_target": 0.5 + action[5] * 0.4
        }

        self.apply_params(new_params)

        # Reward based on performance improvement
        new_stats = self.db.get_statistics()
        reward = (
            new_stats["buffer_cache_hit_ratio"] * 0.3 +
            (1 - new_stats["avg_query_time"] / 100) * 0.4 +
            (1 - new_stats["lock_wait_time"] / 1000) * 0.3
        )
        self.optimizer.reward(reward)
```

### Use Case 6: Load Balancer Weight Optimization

```python
class SmartLoadBalancer:
    def __init__(self, backends):
        self.backends = backends
        self.optimizer = Krystal(KrystalConfig(
            state_dim=len(backends) * 3,
            action_dim=len(backends)
        ))

    def get_weights(self):
        # Collect metrics from all backends
        state = {}
        for i, backend in enumerate(self.backends):
            metrics = backend.get_metrics()
            state[f"cpu_{i}"] = metrics["cpu"]
            state[f"latency_{i}"] = metrics["latency"] / 100
            state[f"errors_{i}"] = metrics["error_rate"]

        self.optimizer.observe(state)
        action = self.optimizer.decide()

        # Normalize weights to sum to 1
        total = sum(action)
        weights = [a / total for a in action]

        return dict(zip([b.name for b in self.backends], weights))

    def record_request(self, backend_name, latency, success):
        # Simple reward based on last request
        reward = 1.0 if success else 0.0
        reward -= min(0.5, latency / 200)  # Penalty for high latency
        self.optimizer.reward(reward)
```

---

## Machine Learning Operations

### Use Case 7: Hyperparameter Tuning

```python
class HyperparameterOptimizer:
    def __init__(self, model_fn, train_data, val_data):
        self.model_fn = model_fn
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = create_ml_optimizer()
        self.best_params = None
        self.best_score = -float('inf')

    def search(self, n_trials=50):
        for trial in range(n_trials):
            # Get hyperparameters from optimizer
            self.optimizer.observe({
                "trial": trial / n_trials,
                "best_so_far": self.best_score if self.best_score > -float('inf') else 0
            })

            action = self.optimizer.decide()

            # Map actions to hyperparameters
            params = {
                "learning_rate": 10 ** (-4 + action[0] * 3),  # 1e-4 to 1e-1
                "batch_size": int(2 ** (4 + action[1] * 4)),  # 16 to 256
                "dropout": action[2] * 0.5,  # 0 to 0.5
                "weight_decay": 10 ** (-6 + action[3] * 4)  # 1e-6 to 1e-2
            }

            # Train and evaluate
            model = self.model_fn(**params)
            train_loss = model.fit(self.train_data, epochs=10)
            val_score = model.evaluate(self.val_data)

            # Update best
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_params = params

            # Reward
            self.optimizer.reward(val_score)

        return self.best_params, self.best_score
```

### Use Case 8: Training Job Scheduling

```python
class TrainingScheduler:
    def __init__(self, gpu_cluster):
        self.cluster = gpu_cluster
        self.optimizer = Krystal()
        self.job_queue = []

    def schedule(self, jobs):
        """Schedule training jobs across GPU cluster."""
        state = {
            "queue_size": len(jobs) / 100,
            "gpu_util_avg": self.cluster.avg_utilization(),
            "memory_free": self.cluster.free_memory_ratio(),
            "jobs_running": len(self.cluster.running_jobs()) / self.cluster.capacity
        }

        for i, job in enumerate(jobs[:4]):  # Consider top 4 jobs
            state[f"job_{i}_size"] = job.estimated_memory / self.cluster.max_memory
            state[f"job_{i}_priority"] = job.priority / 10

        self.optimizer.observe(state)
        action = self.optimizer.decide()

        # Scheduling decisions
        decisions = []
        for i, job in enumerate(jobs[:4]):
            if action[i] > 0.5:  # Schedule this job
                gpu = self.select_gpu(job, action[i])
                if gpu:
                    decisions.append((job, gpu))
                    self.cluster.assign(job, gpu)

        return decisions

    def job_completed(self, job, duration, success):
        # Reward efficient scheduling
        expected_duration = job.estimated_duration
        efficiency = expected_duration / duration if success else 0
        self.optimizer.reward(efficiency)
```

### Use Case 9: Model Serving Optimization

```python
class ModelServingOptimizer:
    def __init__(self, model, batch_timeout_ms=10):
        self.model = model
        self.optimizer = Krystal()
        self.request_buffer = []
        self.batch_timeout = batch_timeout_ms

    def optimize_batch_size(self, current_qps, latency_p99, gpu_memory_used):
        self.optimizer.observe({
            "qps": current_qps / 1000,
            "latency": latency_p99 / 100,
            "memory": gpu_memory_used,
            "batch_size": len(self.request_buffer) / 32
        })

        action = self.optimizer.decide()

        # Determine optimal batch size and timeout
        max_batch = int(1 + action[0] * 63)  # 1-64
        timeout = 1 + action[1] * 49  # 1-50ms
        priority_threshold = action[2]  # High-priority bypass threshold

        return max_batch, timeout, priority_threshold

    def record_inference(self, batch_size, latency, throughput):
        # Reward: high throughput, low latency
        reward = throughput / 1000 * 0.5 + (1 - latency / 100) * 0.5
        self.optimizer.reward(reward)
```

---

## IoT & Edge Computing

### Use Case 10: Battery-Aware Processing

```python
class BatteryOptimizer:
    def __init__(self):
        self.optimizer = create_iot_optimizer()
        self.processing_modes = ["sleep", "low", "medium", "high", "boost"]

    def select_mode(self, battery_level, solar_input, pending_tasks, urgency):
        self.optimizer.observe({
            "battery": battery_level,
            "solar": solar_input,
            "tasks": min(1.0, pending_tasks / 100),
            "urgency": urgency
        })

        action = self.optimizer.decide()

        # Select processing mode
        mode_idx = int(action[0] * 4.99)
        mode = self.processing_modes[mode_idx]

        # Transmission power
        tx_power = 0 + action[1] * 20  # 0-20 dBm

        return mode, tx_power

    def record_outcome(self, tasks_completed, battery_delta, transmission_success):
        # Reward: complete tasks while preserving battery
        reward = (
            tasks_completed * 0.4 +
            (1 + battery_delta) * 0.3 +  # Positive if charging
            transmission_success * 0.3
        )
        self.optimizer.reward(reward)
```

### Use Case 11: Sensor Sampling Optimization

```python
class AdaptiveSampler:
    def __init__(self, sensors):
        self.sensors = sensors
        self.optimizer = Krystal(KrystalConfig(
            state_dim=len(sensors) * 2,
            action_dim=len(sensors)
        ))

    def get_sampling_rates(self, event_detected, battery_level):
        state = {"battery": battery_level, "event": float(event_detected)}

        for i, sensor in enumerate(self.sensors):
            state[f"variance_{i}"] = sensor.recent_variance()
            state[f"importance_{i}"] = sensor.importance_score()

        self.optimizer.observe(state)
        action = self.optimizer.decide()

        # Map to sampling intervals
        rates = {}
        for i, sensor in enumerate(self.sensors):
            # 100ms to 10s sampling interval
            interval = 0.1 * (10 ** (action[i] * 2))
            rates[sensor.name] = interval

        return rates

    def record_detection(self, detected_event, sampling_cost):
        reward = 1.0 if detected_event else 0.5
        reward -= sampling_cost * 0.1  # Energy cost
        self.optimizer.reward(reward)
```

---

## Design Patterns

### Pattern 1: Observer-Decide-Reward Loop

The fundamental pattern for all KrystalSDK applications:

```python
class StandardLoop:
    def __init__(self):
        self.optimizer = Krystal()

    def run(self):
        while True:
            # 1. OBSERVE - Collect current state
            state = self.collect_metrics()
            self.optimizer.observe(state)

            # 2. DECIDE - Get recommended action
            action = self.optimizer.decide()

            # 3. ACT - Apply the action
            result = self.apply_action(action)

            # 4. REWARD - Evaluate outcome
            reward = self.compute_reward(result)
            self.optimizer.reward(reward)

            # 5. Optional: Log, checkpoint, etc.
            self.log(state, action, reward)
```

### Pattern 2: Hierarchical Optimization

For complex systems with multiple time scales:

```python
class HierarchicalOptimizer:
    def __init__(self):
        # High-level: strategic decisions (minutes)
        self.strategic = Krystal(KrystalConfig(learning_rate=0.01))

        # Low-level: tactical decisions (seconds)
        self.tactical = Krystal(KrystalConfig(learning_rate=0.1))

    def decide(self, state):
        # Strategic layer sets goals
        self.strategic.observe(state)
        goals = self.strategic.decide()

        # Tactical layer achieves goals
        augmented_state = {**state, "goal": goals[0]}
        self.tactical.observe(augmented_state)
        action = self.tactical.decide()

        return action, goals

    def reward(self, tactical_reward, strategic_reward):
        self.tactical.reward(tactical_reward)
        # Strategic updates less frequently
        if self.should_update_strategic():
            self.strategic.reward(strategic_reward)
```

### Pattern 3: Ensemble Optimization

Combine multiple optimizers for robustness:

```python
class EnsembleOptimizer:
    def __init__(self, n_optimizers=3):
        self.optimizers = [Krystal() for _ in range(n_optimizers)]
        self.weights = [1/n_optimizers] * n_optimizers

    def observe(self, state):
        for opt in self.optimizers:
            opt.observe(state)

    def decide(self):
        actions = [opt.decide() for opt in self.optimizers]

        # Weighted average of actions
        ensemble_action = []
        for i in range(len(actions[0])):
            weighted = sum(w * a[i] for w, a in zip(self.weights, actions))
            ensemble_action.append(weighted)

        return ensemble_action

    def reward(self, reward):
        # Update all optimizers
        errors = []
        for opt in self.optimizers:
            error = opt.reward(reward)
            errors.append(abs(error))

        # Adjust weights based on prediction accuracy
        total_error = sum(errors) + 1e-6
        self.weights = [(total_error - e) / total_error for e in errors]
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
```

### Pattern 4: Contextual Bandit

For discrete action selection:

```python
class ContextualBandit:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.optimizer = Krystal(KrystalConfig(action_dim=n_actions))

    def select_action(self, context):
        self.optimizer.observe(context)
        action_probs = self.optimizer.decide()

        # Sample action based on probabilities
        action_probs = [max(0.01, p) for p in action_probs]  # Ensure exploration
        total = sum(action_probs)
        action_probs = [p / total for p in action_probs]

        action = random.choices(range(self.n_actions), weights=action_probs)[0]
        return action

    def update(self, reward):
        self.optimizer.reward(reward)
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Sparse Rewards

**Problem**: Only rewarding at episode end
```python
# BAD
for step in range(1000):
    action = optimizer.decide()
    # ... no reward ...
optimizer.reward(final_score)  # Only reward at end
```

**Solution**: Provide intermediate rewards
```python
# GOOD
for step in range(1000):
    action = optimizer.decide()
    intermediate_reward = compute_progress()
    optimizer.reward(intermediate_reward)
```

### Anti-Pattern 2: Unbounded State

**Problem**: Not normalizing inputs
```python
# BAD
optimizer.observe({
    "temperature": 75.5,  # Celsius, unbounded
    "requests": 50000     # Very large number
})
```

**Solution**: Normalize to [0, 1]
```python
# GOOD
optimizer.observe({
    "temperature": 75.5 / 100,  # Normalized
    "requests": min(1.0, 50000 / 100000)  # Capped and normalized
})
```

### Anti-Pattern 3: Action-Reward Mismatch

**Problem**: Rewarding things not related to actions
```python
# BAD - Weather affects reward but actions can't control it
reward = performance - weather_penalty
```

**Solution**: Only reward controllable outcomes
```python
# GOOD - Reward only what actions influence
reward = performance_given_conditions
```

### Anti-Pattern 4: Reward Hacking

**Problem**: Optimizer finds loopholes
```python
# BAD - Can be maximized by oscillating
reward = change_in_metric  # Positive for any change
```

**Solution**: Design robust rewards
```python
# GOOD - Reward absolute performance
reward = current_metric / target_metric
```

---

## Summary

KrystalSDK excels at problems with:
- Continuous state and action spaces
- Frequent feedback opportunities
- Non-stationary environments
- Multi-objective tradeoffs

Choose the right pattern based on:
- **Simple control**: Standard loop
- **Complex systems**: Hierarchical
- **Uncertainty**: Ensemble
- **Discrete choices**: Contextual bandit
