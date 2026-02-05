# Developer Editing Guide

This guide explains **how to safely edit code in FANUC RISE** while preserving architecture, safety, and documentation standards.

## 1) Working Agreement (before editing)

1. Identify the layer you are touching:
   - **Repository/Data layer**: storage access only.
   - **Service layer**: business logic and domain rules.
   - **Interface layer**: HTTP/WebSocket translation only.
   - **HAL layer**: hardware/simulator interaction.
2. Keep changes scoped to one concern per commit.
3. If behavior changes, update documentation in the same PR.
4. Safety logic must remain deterministic and explicit.

## 2) Standard Edit Workflow

### Step 1 — Locate ownership
- Confirm which module owns behavior before changing anything.
- Avoid adding logic to transport/controller files when it belongs in services.

### Step 2 — Make minimal delta
- Prefer focused edits over broad refactors.
- Preserve public contracts unless explicitly versioning them.

### Step 3 — Validate locally
- Syntax check changed language files.
- Run basic smoke checks for impacted entrypoints.
- Verify machine-scoped and fallback telemetry paths if touching dashboard/ws code.

### Step 4 — Document and annotate
- Update one of:
  - `README.md` (entry-level impact)
  - `SYSTEM_ARCHITECTURE.md` (flow/architecture impact)
  - `docs/TECHNICAL_REFERENCE.md` (contract/NFR impact)

## 3) Editing Rules by Layer

### Interface Layer (FastAPI/WS/UI)
- Keep handlers thin; no hidden business rules.
- Validate and normalize inputs at boundaries.
- Include reason codes for safety-relevant outcomes.

### Service Layer
- Keep deterministic policy checks explicit and testable.
- Separate AI proposal generation from safety decision logic.

### HAL Layer
- Use circuit breakers and bounded retries.
- Always provide simulator fallback path for non-hardware environments.

### Data Layer
- Keep schema evolution backward-compatible where possible.
- Prefer additive migration strategy for minor versions.

## 4) Pull Request Checklist for Editors

- [ ] Change is scoped and explained.
- [ ] Contracts impacted are documented.
- [ ] Safety model unchanged or explicitly strengthened.
- [ ] Bootstrap/runbook instructions still valid.
- [ ] Feature blueprint or roadmap mapping updated (if relevant).

## 5) Anti-Patterns to Avoid

- Embedding domain logic in route/controller files.
- Treating LLM confidence as a safety gate.
- Adding hardcoded hostnames/ports where runtime-derived config is needed.
- Changing websocket payload shape without documenting compatibility implications.

## 6) Recommended Commit Style

Use intent-first commit messages:
- `harden ws fallback for machine-scoped telemetry`
- `document api reason_code contract for auditor decisions`
- `refactor service-layer policy checks into deterministic validator`
