# Dev-contitional Platform

This repository is presented as 2 connected but independently positioned projects:

- **60%: Krystal Vino + GAMESA 3D Grid** (primary platform)
- **40%: FANUC RISE** (secondary industrial branch, FOCAS integration)

---

## 1) Krystal Vino + GAMESA 3D Grid (Primary Platform, 60%)

### What Krystal Vino Is
Krystal Vino is a performance orchestration layer on top of OpenVINO/oneAPI for personal computers.  
Its goal is to reduce latency and increase throughput through adaptive planning, telemetry, and runtime policy control.

Codebase: `openvino_oneapi_system/`

### Core Components
- **OpenVINO runtime layer**: inference with a safe fallback mode.
- **oneAPI/OpenMP tuning**: dynamic control of `ONEAPI_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENVINO_NUM_STREAMS`, `KMP_*`.
- **Economic planner + evolutionary tuner**: online switching between `defensive/balanced/aggressive` modes.
- **GAMESA 3D Grid**: logical 3D memory layer for data organization/swap behavior.
- **Delegated logging**: separate channels for `system`, `telemetry`, `planning`, `policy`, `inference`, `grid_update`.

### Proven Results (Linux Benchmark)
Source: `openvino_oneapi_system/logs/benchmark_latest.txt`

- **Latency improvement**: `66.01%`
- **Throughput improvement**: `234.59%`
- **Utility improvement**: `270.42%`
- **Sysbench improvement**: `99.55%`  
  Baseline: `2615.43 events/s` -> Adaptive: `5219.10 events/s`

### Quick Run
```bash
python3 openvino_oneapi_system/main.py --cycles 10 --interval 0.5
python3 openvino_oneapi_system/benchmark_linux.py --cycles 60
```

### Debian Package (Whole Package)
Generated package:
- `openvino_oneapi_system/dist/openvino-oneapi-system_1.1.0_amd64.deb`

Includes:
- CLI: `ovo-runtime`, `ovo-benchmark`
- Service unit: `openvino-oneapi-system.service`
- Config: `/etc/default/openvino-oneapi-system`

---

## 2) FANUC RISE (Secondary Branch, 40%)

### Project Characterization
FANUC RISE is an industrial CNC layer focused on operations, telemetry, and workflow automation.  
FOCAS is a **secondary integration layer**, not the primary product target.

Codebase: `advanced_cnc_copilot/`

### Scope
- CNC operator workflows and supervision
- API + UI for production monitoring
- FANUC telemetry bridge (mock/real mode based on environment)
- Extensible backend services for manufacturing analytics

### Role in the Overall Ecosystem
- Krystal Vino handles performance runtime orchestration and compute optimization.
- FANUC RISE handles industrial context, machine/data connectivity, and operator use.
- Together they form a pipeline: **performance core + industrial execution**.

---

## Repository Map
- `openvino_oneapi_system/` primary performance platform (OpenVINO, oneAPI, GAMESA 3D Grid)
- `advanced_cnc_copilot/` FANUC RISE industrial stack
- `docs/` additional technical materials

## Direction Note
The priority of this repository is Krystal Vino/GAMESA 3D Grid as the main platform for PC hardware and inference performance.  
FANUC RISE remains a separate, secondary domain branch for CNC integrations.

---

## Author & License
**Author**: Dušan Kopecký  
**Email**: dusan.kopecky0101@gmail.com  
**License**: Apache 2.0 (See `LICENSE` file)
