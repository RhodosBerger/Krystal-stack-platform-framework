# Brain Cortex V3 — HW Inference Game Platform

**Target Hardware:** Intel Core i5-1135G7 (Tiger Lake, 11th Gen)
- Intel Iris Xe (80 EU) — iGPU
- Intel NPU 3720 (GNA) — Neural Processing Unit
- Intel DL Boost / VNNI — INT8 acceleration
- Ubuntu 22.04+ Custom Kernel Advisory Grid

**Framework:** Gamesa Cortex V3 / OpenVINO / OpenAPI

---

## What is Brain Cortex V3?

A terminal-based **hardware inference game** that uses your real CPU, iGPU, and NPU as players.
The game reads live Vulkan draw-call logs, transcribes them through the **Advice Grid**, and sends
optimized inference instructions to OpenVINO and OpenAPI endpoints.

You win by keeping the system alive, managed, and efficient — the longer you sustain target FPS
and inference scores without thermal throttling or budget collapse, the higher your score.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      BRAIN CORTEX V3                             │
│                                                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────────┐  │
│  │  VULKAN LOG │───▶│ ADVICE GRID  │───▶│   INSTRUCTION BUS  │  │
│  │  INTERCEPTOR│    │  (Mapper)    │    │  (OpenVINO/OpenAPI) │  │
│  └─────────────┘    └──────────────┘    └────────────────────┘  │
│         │                  │                      │              │
│  ┌──────▼──────────────────▼──────────────────────▼───────────┐ │
│  │                  CORTEX CONTROL PLANE                       │ │
│  │   Power Mode: [ECO|BALANCED|OVERDRIVE]                      │ │
│  │   Device: [CPU|GPU.iris_xe|NPU.3720]                        │ │
│  │   Budget: [credits remaining]                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                    ┌─────────▼──────────┐                        │
│                    │   THE GAME ENGINE  │                        │
│                    │ (Inference Arenas) │                        │
│                    └────────────────────┘                        │
└──────────────────────────────────────────────────────────────────┘
```

## Game Modes (Inference Arenas)

1. **ARENA_ALPHA** — CPU vs NPU: Who classifies images faster?
2. **ARENA_BETA** — Battery Endurance: Sustain 30 FPS on lowest watt budget
3. **ARENA_GAMMA** — Vulkan Transcription Race: Fastest Advice Grid mapping wins
4. **ARENA_DELTA** — Thermal Survival: Don't let the CPU throttle

## Quick Start

```bash
cd brain_cortex_v3
pip install -r requirements.txt
python3 cortex_game.py --arena ALPHA
python3 cortex_game.py --arena BETA --power eco
python3 cortex_game.py --arena gamma
```
