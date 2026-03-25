# Integration Guide - Experimental Kernel into Krystal Stack

## 🎯 Purpose

This guide explains how to insert the Experimental Kernel layer into the existing Krystal Stack architecture to enable better features and improved performance.

---

## 📐 Architecture Overview

The Experimental Kernel is designed as a **middleware layer** between:
- **High-level Python control plane** (Gamesa Cortex V2, Brain Cortex V3)
- **Low-level hardware acceleration** (Vulkan, OpenVINO, OpenCL)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PYTHON CONTROL PLANE                          │
│  (Gamesa Cortex V2, Brain Cortex V3, Advanced CNC Copilot)      │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│              EXPERIMENTAL KERNEL (Rust Layer)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │    UUID      │  │   KERNEL     │  │   ECONOMIC           │  │
│  │   TRACKER    │  │  SCHEDULER   │  │   GOVERNOR           │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   FEATURE    │  │    MODULE    │  │   HOT RELOAD         │  │
│  │  CONTAINER   │  │  REGISTRY    │  │   MANAGER            │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                  HARDWARE ACCELERATION LAYER                     │
│  (Vulkan Grid Engine, OpenVINO, OpenCL, FANUC FOCAS)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Step 1: Build and Install

```bash
cd /home/dusan/Documents/GitHub/Dev-contitional/gamesa_cortex_v2/experimental_kernel

# Build the kernel
./build.sh

# Or manually:
cargo build --release
maturin develop --release
```

---

## 🔧 Step 2: Integration with Gamesa Cortex V2

### Option A: Direct Integration (Recommended)

Modify `gamesa_cortex_v2/src/python_control_plane.py`:

```python
# Add at the top
from experimental_kernel import (
    KernelScheduler,
    UUIDTracker,
    EconomicGovernor,
    FeatureContainer
)

class EnhancedNPUCoordinator:
    def __init__(self):
        # Existing initialization
        self.original_coordinator = NPUCoordinator()
        
        # NEW: Experimental Kernel integration
        self.scheduler = KernelScheduler()
        self.tracker = UUIDTracker()
        self.governor = EconomicGovernor()
        self.features = FeatureContainer()
        
        # Initialize credit budgets for different resource domains
        self.scheduler.init_budget("npu", 500, 50, 100)
        self.scheduler.init_budget("gpu", 1000, 100, 50)
        self.scheduler.init_budget("cpu", 300, 30, 200)
        
        # Register safety features
        self.features.register_feature(
            "thermal_safety",
            "Thermal Safety Monitor",
            "1.0.0",
            priority=255,
            dependencies=[]
        )
        self.features.initialize_all()
    
    def dispatch_task(self, task_type, model, input_data, priority=100, deadline_ms=100):
        """Dispatch task through experimental kernel scheduler."""
        
        # Submit to scheduler with credit tracking
        uuid = self.scheduler.submit_task(
            task_type=task_type,
            deadline_ms=deadline_ms,
            priority=priority,
            credits_cost=20,
            domain="npu",
            tags=[model, "inference"]
        )
        
        if not uuid:
            # Task rejected due to insufficient credits
            return None
        
        # Schedule the task
        scheduled_task = self.scheduler.schedule_next()
        if not scheduled_task:
            return None
        
        # Start tracking
        self.tracker.start_task(uuid)
        
        try:
            # Execute through original coordinator
            result = self.original_coordinator.run_inference(model, input_data)
            
            # Complete tracking
            latency = self.tracker.complete_task(uuid)
            self.scheduler.complete_task(uuid)
            
            # Add telemetry
            if latency > deadline_ms * 1_000_000:
                print(f"⚠️  Deadline missed: {latency/1_000_000:.2f}ms > {deadline_ms}ms")
            
            return result
            
        except Exception as e:
            # Mark task as failed
            self.tracker.fail_task(uuid, str(e))
            raise
    
    def get_performance_metrics(self):
        """Get comprehensive performance metrics."""
        return {
            'scheduler_stats': self.scheduler.get_stats(),
            'latency_stats': self.tracker.get_latency_stats(),
            'budget_status': self.governor.get_all_budgets(),
        }
```

### Option B: Wrapper Pattern

Create a new module `gamesa_cortex_v2/experimental_wrapper.py`:

```python
"""
Experimental Kernel Wrapper for Gamesa Cortex V2

This wrapper provides a clean interface to the Experimental Kernel
without modifying existing code.
"""

from experimental_kernel import KernelScheduler, UUIDTracker, EconomicGovernor

class ExperimentalKernelWrapper:
    """Wrapper for integrating experimental kernel with existing code."""
    
    def __init__(self, wrapped_object):
        self.wrapped = wrapped_object
        self.scheduler = KernelScheduler()
        self.tracker = UUIDTracker()
        self.governor = EconomicGovernor()
        
        # Initialize default budgets
        self._init_default_budgets()
    
    def _init_default_budgets(self):
        """Initialize default credit budgets."""
        self.scheduler.init_budget("default", 1000, 100, 100)
    
    def execute_with_tracking(self, func, *args, **kwargs):
        """Execute a function with UUID tracking and scheduling."""
        
        # Extract metadata from kwargs
        task_type = kwargs.pop('task_type', 'generic')
        priority = kwargs.pop('priority', 100)
        deadline_ms = kwargs.pop('deadline_ms', 100)
        credits_cost = kwargs.pop('credits_cost', 10)
        domain = kwargs.pop('domain', 'default')
        
        # Submit to scheduler
        uuid = self.scheduler.submit_task(
            task_type=task_type,
            deadline_ms=deadline_ms,
            priority=priority,
            credits_cost=credits_cost,
            domain=domain
        )
        
        if not uuid:
            raise RuntimeError("Task rejected: insufficient credits")
        
        # Schedule and execute
        task = self.scheduler.schedule_next()
        self.tracker.start_task(uuid)
        
        try:
            result = func(*args, **kwargs)
            latency = self.tracker.complete_task(uuid)
            self.scheduler.complete_task(uuid)
            return result, latency
            
        except Exception as e:
            self.tracker.fail_task(uuid, str(e))
            raise
    
    def get_metrics(self):
        """Get performance metrics."""
        return {
            'scheduler': self.scheduler.get_stats(),
            'latency': self.tracker.get_latency_stats(),
        }
```

Usage:

```python
from gamesa_cortex_v2 import NPUCoordinator
from gamesa_cortex_v2.experimental_wrapper import ExperimentalKernelWrapper

# Wrap existing coordinator
coordinator = NPUCoordinator()
enhanced_coordinator = ExperimentalKernelWrapper(coordinator)

# Use with automatic tracking
result, latency = enhanced_coordinator.execute_with_tracking(
    coordinator.run_inference,
    model="resnet50",
    input_data=image,
    task_type="image_classification",
    priority=150,
    deadline_ms=50
)

print(f"Inference completed in {latency/1_000_000:.2f}ms")
```

---

## 🔧 Step 3: Integration with Brain Cortex V3

Modify `brain_cortex_v3/cortex_game.py`:

```python
# Add experimental kernel integration
from experimental_kernel import KernelScheduler, UUIDTracker

class CortexGameWithKernel:
    def __init__(self, arena):
        self.arena = arena
        self.scheduler = KernelScheduler()
        self.tracker = UUIDTracker()
        
        # Initialize budgets for different arenas
        self.scheduler.init_budget("inference", 500, 50, 100)
        self.scheduler.init_budget("vulkan", 1000, 100, 50)
        self.scheduler.init_budget("power", 300, 30, 200)
    
    def run_arena(self):
        """Run arena with experimental kernel scheduling."""
        
        # Submit inference task
        uuid = self.scheduler.submit_task(
            task_type=f"arena_{self.arena}",
            deadline_ms=100,
            priority=150,
            credits_cost=20,
            domain="inference"
        )
        
        # ... rest of arena logic ...
```

---

## 🔧 Step 4: Feature Insertion Framework

The Experimental Kernel provides a framework for inserting new features without modifying core code:

```python
from experimental_kernel import FeatureContainer, FeaturePipelineBuilder

# Create feature container
container = FeatureContainer()

# Register new features
container.register_feature(
    feature_id="adaptive_frequency_scaling",
    name="Adaptive Frequency Scaling",
    version="1.0.0",
    priority=180,
    dependencies=["thermal_monitor"]
)

container.register_feature(
    feature_id="predictive_load_balancing",
    name="Predictive Load Balancing",
    version="1.0.0",
    priority=160,
    dependencies=[]
)

# Build feature pipeline
pipeline = FeaturePipelineBuilder()
    .add_feature("adaptive_frequency_scaling")
    .add_feature("predictive_load_balancing")
    .with_context_data("cpu_cores", "8")
    .with_context_data("gpu_frequency", "1500")
    .build()

# Execute pipeline
results = pipeline.execute()
```

---

## 🔧 Step 5: Telemetry Integration

### Export to Kafka

```python
from kafka import KafkaProducer
import json
from experimental_kernel import KernelScheduler

scheduler = KernelScheduler()
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send metrics periodically
def send_telemetry():
    stats = scheduler.get_stats()
    producer.send('krystal-scheduler-metrics', stats)
    
    latency = scheduler.get_latency_stats()
    producer.send('krystal-latency-metrics', latency)
```

### Export to Snowflake

```python
import snowflake.connector
from datetime import datetime

def export_to_snowflake(scheduler, tracker):
    conn = snowflake.connector.connect(
        user='krystal_user',
        password='password',
        account='YOUR_ACCOUNT',
        warehouse='COMPUTE_WH',
        database='KRystal_METRICS',
        schema='PUBLIC'
    )
    
    cursor = conn.cursor()
    
    # Insert scheduler metrics
    stats = scheduler.get_stats()
    cursor.execute("""
        INSERT INTO scheduler_metrics 
        (timestamp, tasks_scheduled, tasks_completed, deadline_misses)
        VALUES (%s, %s, %s, %s)
    """, (
        datetime.now(),
        stats['tasks_scheduled'],
        stats['tasks_completed'],
        stats['tasks_deadline_missed']
    ))
    
    # Insert latency metrics
    latency = tracker.get_latency_stats()
    cursor.execute("""
        INSERT INTO latency_metrics
        (timestamp, p50_ns, p95_ns, p99_ns, avg_ns)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        datetime.now(),
        latency['p50_ns'],
        latency['p95_ns'],
        latency['p99_ns'],
        latency['avg_ns']
    ))
    
    conn.commit()
    cursor.close()
    conn.close()
```

---

## 📊 Performance Tuning

### Credit Budget Configuration

| Workload Type | GPU Budget | CPU Budget | NPU Budget | Replenish Rate |
|---------------|------------|------------|------------|----------------|
| AI Inference | 1000 | 300 | 500 | 100/50ms |
| Rendering | 2000 | 200 | 100 | 200/100ms |
| Mining | 1500 | 100 | 50 | 150/100ms |
| Mixed | 1000 | 500 | 300 | 100/75ms |

### Scheduling Mode Selection

| Mode | Use Case | Latency | Throughput | Power |
|------|----------|---------|------------|-------|
| Normal | General purpose | Medium | Medium | Medium |
| Real-Time | Safety-critical | **Low** | Low | High |
| Eco | Battery-powered | High | Low | **Low** |
| Performance | Compute-heavy | Low | **High** | High |

---

## 🧪 Testing

```python
import unittest
from experimental_kernel import KernelScheduler, UUIDTracker

class TestExperimentalKernel(unittest.TestCase):
    
    def test_scheduler_basic(self):
        scheduler = KernelScheduler()
        scheduler.init_budget("test", 1000, 100, 100)
        
        uuid = scheduler.submit_task(
            "test_task", 50, 150, 10, "test"
        )
        
        self.assertIsNotNone(uuid)
        
        task = scheduler.schedule_next()
        self.assertIsNotNone(task)
        
        scheduler.complete_task(task['uuid'])
        stats = scheduler.get_stats()
        
        self.assertEqual(stats['tasks_completed'], 1)
    
    def test_latency_tracking(self):
        tracker = UUIDTracker()
        uuid = tracker.create_task("test", 100, 10)
        
        tracker.start_task(uuid)
        import time
        time.sleep(0.01)  # 10ms
        latency = tracker.complete_task(uuid)
        
        self.assertGreater(latency, 10_000_000)  # > 10ms in ns

if __name__ == '__main__':
    unittest.main()
```

---

## 📝 Checklist

- [ ] Build Experimental Kernel: `./build.sh`
- [ ] Run demo: `python3 demo.py`
- [ ] Integrate with Gamesa Cortex V2
- [ ] Integrate with Brain Cortex V3
- [ ] Configure credit budgets for your workload
- [ ] Set up telemetry export (Kafka/Snowflake)
- [ ] Tune scheduling mode for your use case
- [ ] Register custom features
- [ ] Run performance benchmarks
- [ ] Deploy to production

---

## 🚀 Next Steps

1. **Review the API documentation** in `README.md`
2. **Run the demo** to understand the capabilities
3. **Start with wrapper pattern** for non-invasive integration
4. **Gradually migrate** to direct integration for better performance
5. **Add custom features** using the Feature Container
6. **Enable telemetry** for production monitoring

---

## 📞 Support

**Author**: Dušan Kopecký  
**Email**: dusan.kopecky0101@gmail.com  
**GitHub**: https://github.com/RhodosBerger/Krystal-stack-platform-framework

---

## 📄 License

Apache 2.0 - See LICENSE file
