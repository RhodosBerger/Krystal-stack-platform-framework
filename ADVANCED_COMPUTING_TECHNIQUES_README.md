# Advanced Computing Techniques: Evolutionary Mechanics & Sysbench Optimization

This module introduces a closed-loop optimization system where **Evolutionary Algorithms** (Genetic Algorithms) are used to evolve and enhance system mechanics, validated by **Sysbench** performance metrics.

## Components

### 1. Evolutionary Sysbench Optimizer (`evolutionary_sysbench_optimizer.py`)
This is the core orchestration script that demonstrates the "Advanced Computing Technique".
- **Genotype**: A vector of normalized values [0.0 - 1.0] representing:
  - Thread Count
  - Memory Block Size (Logarithmic scale)
  - File Block Size
  - CPU Prime Limits
- **Phenotype**: Concrete system configurations passed to `sysbench`.
- **Fitness Function**: A composite score maximizing:
  - CPU Events per Second / Latency
  - Memory Throughput (MB/s)
  - Stability (penalties for excessive contention)

### 2. Sysbench Integration (`sysbench_integration.py`)
Enhanced to support fine-grained mechanics control:
- **New Feature**: `run_memory_benchmark` now accepts `block_size`, allowing the optimizer to tune memory access patterns.
- **Simulation Mode**: Ensures the system functions even without native `sysbench` binaries, facilitating development and testing.

### 3. Invention Engine Optimization (`src/python/invention_engine.py`)
Optimized for low-latency decision making (~50% reduction):
- **Caching**: Memoization of result states.
- **Pruning**: Top-K filtering to remove invalid states before quantum superposition.
- **Incremental Updates**: Skipping heavy SNN computations when input deltas are small.

## How to Run

To witness the evolutionary optimization of system mechanics:

```bash
python evolutionary_sysbench_optimizer.py
```

Output will show the Genetic Algorithm evolving better configurations over generations:
```
--- Generation 1/5 ---
  Best Score: 12450.2
  Best Config: {'threads': 4, 'memory_block_size': '4K', ...}

...

--- Generation 5/5 ---
  Best Score: 18920.5
  Best Config: {'threads': 8, 'memory_block_size': '16K', ...}
```
