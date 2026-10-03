# Krystal-Stack Janet Engine (Sub-Project)
========================================================================

**Sub-Project:** `krystal-janet-engine`  
**Language:** [Janet Lisp](https://janet-lang.org/)  
**Classification:** Functional Heterogeneous Engine Port  
**Repository Directory:** `krystal_janet/`

---

## 1. Overview & Architecture

The **Krystal-Stack Janet Engine** is a native port of the Krystal-Stack Platform Framework into **Janet Lisp**. It exploits Janet's lightweight footprint, first-class cooperative fibers, immutable tuples/structs, and native Parsing Expression Grammars (PEGs) to deliver a concurrent, zero-dependency computational engine.

```
                    ┌────────────────────────────────────────┐
                    │      krystal_janet/main.janet          │
                    │   (Unified Standalone CLI Runner)      │
                    └───────────────────┬────────────────────┘
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│ cyclic_organism.janet │   │neural_ascii_engine    │   │  topological_vm.janet │
│  Symplectic Verlet    │   │ 3D SDF Raymarching    │   │ Fiber Channel Queues  │
│  Phase Space Trajectory│   │ TrueColor VT-100 ANSI │   │ Priority Ring Buffers │
└──────────┬────────────┘   └───────────┬───────────┘   └───────────┬───────────┘
           │                            │                           │
           └────────────────────────────┼───────────────────────────┘
                                        ▼
                            ┌───────────────────────┐
                            │    governor.janet     │
                            │ Visual Entropy E_tot  │
                            │ Coherence & Backpress.│
                            └───────────────────────┘
```

---

## 2. Sub-Project Structure & Modules

| File | Purpose | Key Symbols & Exports |
| :--- | :--- | :--- |
| [`project.janet`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/project.janet) | JPM Project Manifest & Dependencies | Declares sources & executables |
| [`main.janet`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/main.janet) | Standalone CLI Engine Runner | `print-header`, `run-engine`, `main` |
| [`cyclic_organism.janet`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/cyclic_organism.janet) | Symplectic Hamiltonian Organism | `create-organism`, `symplectic-step`, `total-hamiltonian` |
| [`neural_ascii_engine.janet`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/neural_ascii_engine.janet) | 3D SDF Raymarching & TrueColor Rasterizer | `create-camera`, `raymarch`, `render-frame` |
| [`governor.janet`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/governor.janet) | Visual Entropy & Economic Governor | `create-governor`, `compute-entropy`, `update-budget` |
| [`topological_vm.janet`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/topological_vm.janet) | Fiber-based Pipeline & Ring Buffers | `create-topological-vm`, `alloc-queue`, `step-vm` |
| [`krystal_sdf.janet`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/krystal_sdf.janet) | Analytical Signed Distance Fields & CSG | `smin`, `sdf-torus`, `sdf-sphere`, `mod-twist-y` |
| [`antigravity_peg.janet`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/antigravity_peg.janet) | Multilingual PEG Natural Language Parser | `prompt-peg`, `parse-antigravity-prompt` |
| [`openworld_chunk.janet`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/openworld_chunk.janet) | Infinite Terrain & Procedural fBm Chunks | `fbm-terrain`, `create-chunk`, `stream-chunks` |
| [`janet_bridge.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/janet_bridge.py) | Python AST Validator & Standalone Emulator | `JanetValidator`, `JanetEngineRunner` |
| [`run_janet.bat`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/run_janet.bat) | Windows Batch Sub-project Launcher | Runs `janet` or falls back to bridge |
| [`run_janet.ps1`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_janet/run_janet.ps1) | PowerShell Sub-project Launcher | Native / Bridge fallback execution |

---

> **Verification status (2026-10-03).** These sources are bracket-balance checked (`janet_bridge.py`) and their symbols were audited against the official Janet Core API (invalid `math/max`, `math/min`, `float` were removed), but they have **not been executed on a real Janet interpreter** (`janet` is not installed on this host). Known open item: `governor.janet` uses a multi-byte `(chr "...")`, which is invalid in Janet. See `janet-runtime-verify` in [PROJECT_AUDIT_AND_COMPUTE_PATTERNS.md](../docs/PROJECT_AUDIT_AND_COMPUTE_PATTERNS.md).

## 3. Running the Janet Sub-Project

### Option A: Using the Launcher Scripts
```powershell
# In PowerShell:
cd krystal_janet
.\run_janet.ps1

# In Windows CMD:
cd krystal_janet
run_janet.bat
```

### Option B: Using Native Janet (if installed)
```bash
janet krystal_janet/main.janet
```

### Option C: Using the Python Bridge Emulator & Validator
```powershell
# Validate all Janet files (bracket balance, S-expression AST, exports):
python -m krystal_janet.janet_bridge

# Run headless test:
python -m krystal_janet.janet_bridge test

# Run live interactive animation:
python -m krystal_janet.janet_bridge run
```

---

## 4. Key Mathematical Formulations in Janet

### 4.1 Symplectic Velocity-Verlet Integration
Preserves phase-space volume $dq \wedge dp$ without numerical divergence:
```janet
# Half-step momentum
(for i 0 4 (put p i (+ (p i) (* 0.5 dt (f-total-t i)))))
# Full-step coordinate
(for i 0 4 (put q i (+ (q i) (* dt (/ (p i) (m i))))))
# Complete momentum
(for i 0 4 (put p i (+ (p i) (* 0.5 dt (f-total-next i)))))
```

### 4.2 Non-linear Damping & Visual Entropy Coupling
Damping force dynamically couples to real-time visual scene entropy:
$$\mathbf{F}_{\text{diss}}(p, E) = -(\gamma_0 + \beta E^2) \mathbf{p}$$
When visual entropy exceeds $0.70$, the cognitive state immediately transitions into $\Omega$ phase, actuating backpressure and throttling render steps.
