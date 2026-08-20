# Hybrid Combinatorics Engine
## The Main Feature: Rust-Safe Predictive C

This module (`hybrid_combinatorics_engine.py`) implements the core stability feature of the system: a **Hybrid Combinatorics Method**.

### Architecture Overview

The system divides responsibilities to maximize both safety and performance:

1.  **C Layer (The Predictor)**:
    *   **Responsibility**: Predictive Operation Fetch.
    *   **Method**: **Combinatorics**. It generates multiple potential "code paths" (combinations) for a given context (e.g., "Texture Load").
    *   **Output**: Raw, unverified C code snippets.

2.  **Rust Layer (The Guardian Scope)**:
    *   **Responsibility**: Safety & Performance Approval.
    *   **Method**: **Scoping**. It inspects the constructed C code *before* compilation.
    *   **Criteria**:
        *   **Memory Safety**: Are address accesses within the safe range (0-1024)?
        *   **Logic Safety**: Are resource alloc/dealloc pairs correct?
    *   **Output**: An "Approved" code snippet or a Rejection.

### The "Combinatorics Method"

The stability of the system comes from the fact that we don't just generate *one* path; we generate *many* (Combinatorics) and let the Rust Layer filter for the *best safe one*.

```
[Context: Load Data]
      |
      +---> [C Option 1: Fast but Risky] ----X (Rejected by Rust)
      |
      +---> [C Option 2: Slow but Safe]  -----> (Approved)
      |
      +---> [C Option 3: Fast & Safe]    -----> (Approved & Selected for Perf)
```

### Usage

```python
from hybrid_combinatorics_engine import HybridEngine

engine = HybridEngine()

# Runs the cycle:
# 1. C generates options
# 2. Rust scopes and filters
# 3. Approved code is executed
engine.run_combinatorial_cycle()
```
