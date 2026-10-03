---
name: neural-ascii-engine-development
description: Workflow cheatsheet and procedures for developing, training, benchmarking, and operating the Krystal-Stack Neural ASCII Engine, NPU offload, and Localhost Mission Control hub.
---

# Neural ASCII Engine Development & Operation Skill

## 1. Quick Start / Running the Localhost Environment
The Krystal-Stack Localhost Mission Control Hub provides real-time procedural 3D raymarching, style switching, and entropic telemetry streaming.

### Launching the Hub:
```powershell
# Option A: One-click batch launcher
.\start_localhost.bat

# Option B: PowerShell launcher
.\start_localhost.ps1

# Option C: Direct Python
python start_localhost.py
```
Access the web dashboard at: `http://localhost:8080/`

### Validating the Environment:
```powershell
python verify_localhost_env.py
```

---

## 2. Core Architectural Components

### A. High-Speed Terminal Driver (`demo_neural_ascii_engine_win.py`)
- Employs Win32 Virtual Terminal Processing (`SetConsoleMode`) and atomic `WriteFile` kernel writes.
- Renders 3D Signed Distance Fields (SDFs) with real-time normal extraction and directional Sobel mapping (`|`, `/`, `-`, `\`).
- Computes spatial variance and temporal frame-to-frame delta to derive visual entropy.

### B. Localhost Mission Control Hub (`krystal_web_hub/`)
- `server.py`: Multi-threaded HTTP + Server-Sent Events (SSE) server.
  - Streams frames at `/api/stream` at 30–60 FPS.
  - Exposes REST controls at `/api/control` (styles: `CYBERPUNK`, `BLUEPRINT_EDGE`, `HIGH_FIDELITY`, `RAYMARCH_ANOMALY`, `MATRIX_RAIN`, `RETRO_CRT`).
  - Cognitive Scene Director at `/api/director`.
- `static/index.html`: Cyberpunk mission control dashboard with live ASCII canvas, real-time gauges, and interactive controls.

### C. NPU & Vulkan Compute Acceleration (`docs/research/`)
- Reference GLSL Compute Shader: `ascii_rasterizer.comp` in [GAME_ENGINE_TECHNIQUES_AND_PROCEDURAL_RENDERING.md](docs/research/GAME_ENGINE_TECHNIQUES_AND_PROCEDURAL_RENDERING.md).
- INT8 Neural Model Distillation & Loss functions in [TRAINING_AND_FINE_TUNING_PIPELINE.md](docs/research/TRAINING_AND_FINE_TUNING_PIPELINE.md).
- Heterogeneous triad architecture in [DEEP_RESEARCH_NEURAL_ASCII_NPU_ENGINE.md](docs/research/DEEP_RESEARCH_NEURAL_ASCII_NPU_ENGINE.md).

---

## 3. Visual Backpressure Regulation Pattern
When visual entropy exceeds the threshold ($E_{total} > 0.70$):
1. **Trigger Alert**: Signal backpressure state to Economic Governor.
2. **Throttle Compute**: Reduce background task scheduling, defer non-critical rendering passes.
3. **Graceful Style Degradation**: Step down from complex density blocks (`░▒▓█`) to lightweight directional vectors (`| / - \`) until coherence recovers ($E_{total} < 0.55$).
