---
name: krystal-ux-wireframe-hud-patterns
description: Lightweight zero-WebGL SVG/CSS HUD wireframe, bot-perception projection, State A/B CSS custom-property switching, and skill-slot state-machine patterns for Krystal Stack interfaces.
---

# Krystal UX Wireframe & Modular HUD Pattern Skill

Use this skill whenever building or refactoring HUDs, wireframes, bot-perception views, or skill/card bars in Krystal Stack.

## 1. Constitutional UI & Authority Invariants
1. **Pure Projection (`Render(S) != S`)**: The HUD never mutates authoritative state locally; every skill slot (`[Q]/[W]/[E]/[R]`) emits a typed action proposal to `STEP_GATE_PROXY` / `ClashGame.apply`.
2. **Zero GPU Bloat Mode**: Always provide a pure inline-SVG + flat-CSS wireframe mode (`render_lightweight_bot_perception_hud` at `/bot_perception_hud`) requiring **0 WebGL contexts**, **0 `box-shadow`**, and **0 `backdrop-filter`** so GPU/UMA bandwidth stays `< 0.78`.
3. **5-Tier Navigation Spine**: Top bar displays `WORLD -> FACTION -> SYSTEM`, center displays the 37-hex SVG perception grid, side rails display `BOT INPUT TENSORS` vs `SHADOW VERIFIER PLAN`, and bottom bar binds `ACTION / SKILL SLOTS` + `TELEMETRY`.

## 2. State A / State B Editable Wireframe Tokens
Toggle `body.state-b` using only CSS custom properties (`--bg`, `--surface`, `--wire`, `--accent`, `--stroke`) so Before/After render-cost comparisons require zero DOM reconstruction.

## 3. Skill-Slot State Machine
Every HUD skill/card slot renders one of 4 deterministic states:
- `READY` (`usable_frames >= cost` and hardware gate unlocked)
- `INSUFFICIENT_FRAMES` (`budget < cost`)
- `HARDWARE_VETOED` (e.g. `GPU_ALLOCATE` when `uma >= 0.78`)
- `PRECONDITION_LOCKED` (e.g. `FREIVALDS_AUDIT` when target unit is not in `CONFLICT`)
