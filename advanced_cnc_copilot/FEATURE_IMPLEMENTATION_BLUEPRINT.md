# FANUC RISE Feature Implementation Blueprint

This document turns the project roadmaps into an executable delivery plan. It is meant to be used with:
- `ROADMAP_TO_ARCHITECTURE.md`
- `IMPLEMENTATION_STRATEGY_PHASE2.md`
- `cms/theories/PROJECT_STATE_AND_ROADMAP.md`
- `NEXT_STEPS_OVERVIEW.md`

## 0) Roadmap Source Crosswalk (MD -> Implementation)

| Source document | What it contributes | Implemented as track |
|---|---|---|
| `ROADMAP_TO_ARCHITECTURE.md` | Macro architecture phases and target platform shape | A, B, E |
| `IMPLEMENTATION_STRATEGY_PHASE2.md` | Mid-phase delivery and integration sequencing | B, C, D |
| `cms/theories/PROJECT_STATE_AND_ROADMAP.md` | Current-state inventory + quarterly objectives | A, C, E |
| `NEXT_STEPS_OVERVIEW.md` | Tactical next actions and commercialization flow | C, D, E |

This crosswalk exists so every roadmap statement is traceable to an engineering workstream.

## 1) Dependency Baseline (must pass first)

### Required runtime layers
1. **Python backend** (FastAPI + orchestration + simulator/HAL)
2. **Database** (PostgreSQL / TimescaleDB)
3. **Optional cache/bus** (Redis)
4. **Frontend clients** (React / Vue / dashboard HTML)

### One-command bootstrap
Use:
```bash
./tools/bootstrap_and_audit.sh
```

The script creates `.venv`, installs Python deps, installs Node deps in all frontend workspaces, and prints npm/pip diagnostics.

---

## 2) Feature Delivery Tracks (from MD roadmaps)

## Track A — Hardware + HAL Reliability
**Goal**: production-safe telemetry and command path.

### A1. FOCAS bridge hardening
- Implement circuit breaker retries + timeout budgets in HAL adapters.
- Add machine profile registry (`Fanuc`, `Siemens`, `Mock`) to avoid hardcoded assumptions.
- Acceptance:
  - Graceful degradation to simulator when hardware unavailable.
  - 0 unhandled exceptions on adapter disconnect/reconnect chaos test.

### A2. Latency pipeline
- Move high-frequency shared state to Redis streams or Arrow IPC.
- Add p50/p95/p99 telemetry pipeline timing metrics.
- Acceptance:
  - p95 ingestion-to-dashboard < 100ms in simulator load tests.

## Track B — Shadow Council Safety Governance
**Goal**: AI suggestions never bypass deterministic guardrails.

### B1. Auditor policy engine
- Encode hard constraints for load/vibration/thermal/curvature bounds.
- Expose policy decisions + reasoning traces over API.
- Acceptance:
  - Any violating proposal is blocked with explicit reasons.

### B2. Creator + Accountant integration
- Creator generates strategy candidates.
- Accountant scores economics/time/risk.
- Auditor performs final deterministic gate.
- Acceptance:
  - Decision packet contains proposal, economics score, and pass/fail rationale.

## Track C — Probability Canvas + Fleet UX
**Goal**: roadmap-aligned multi-machine operations.

### C1. Fleet switching
- Keep machine-specific websocket route as primary (`/ws/telemetry/{machine_id}`) with fallback.
- Add machine selector and persisted last-machine state.
- Acceptance:
  - Operator can switch among 3+ machines without page refresh.

### C2. Fleet health overview
- Add hub-level card metrics (status, load trend, alert count) per machine.
- Acceptance:
  - Hub view updates in near real-time and highlights critical nodes.

## Track D — Simulator-to-LLM Training Loop
**Goal**: improve model quality safely offline.

### D1. Scenario generation service
- Generate normal + fault scenarios (chatter, thermal drift, stall).
- Persist traces in consistent schema.

### D2. Dataset builder
- Build SFT examples and pairwise preference data.
- Include auditor verdicts and economics outcomes.

### D3. Shadow deployment gate
- Model can propose only; deterministic systems decide execution.
- Acceptance:
  - Safety violation rate and rejection rate dashboards available.

## Track E — Multi-site Cloud + ERP Integration
**Goal**: roadmap Q2/Q3 scalability.

### E1. Fleet registry + tenancy
- Add site and machine tenancy boundaries.
- Implement RBAC scoped by site/role.

### E2. Event-driven sync
- Broadcast relevant learnings across machines (with policy controls).
- Acceptance:
  - Cross-machine strategy propagation with audit trail.

---

## 3) Suggested Sprint Plan (12-week template)

### Sprint 1-2
- Dependency baseline, CI checks, API health hardening.
- Deliverables: repeatable local bootstrap, passing lints/checks.

### Sprint 3-4
- HAL resiliency + telemetry latency instrumentation.
- Deliverables: circuit breaker + latency dashboard.

### Sprint 5-6
- Shadow Council decision packet + deterministic policy traceability.
- Deliverables: pass/fail auditor API with reason codes.

### Sprint 7-8
- Fleet UX (machine selector, hub rollup metrics, alerts).
- Deliverables: live multi-node dashboard.

### Sprint 9-10
- Simulator scenario generation + dataset pipeline.
- Deliverables: exportable SFT/preference datasets.

### Sprint 11-12
- Shadow deployment and go/no-go gates for pilot.
- Deliverables: pilot checklist + rollback plan.

---

## 4) Engineering Definition of Done

A feature is done only if all conditions hold:
1. Unit/integration checks pass.
2. Feature has monitoring signals (health, latency, error rate).
3. Auditor safety constraints are applied where relevant.
4. Docs updated (README + architecture + API notes).
5. Simulator regression scenarios pass.

---

## 5) Immediate next actions

1. Run `./tools/bootstrap_and_audit.sh`.
2. Bring up backend and verify `/api/health` and websocket routes.
3. Implement Fleet selector persistence in dashboard UI.
4. Add Auditor reason-code schema and include in websocket payload.
5. Create simulator dataset export command for SFT + preference data.


## 6) Documentation Standards Used

This blueprint follows documentation conventions commonly used in mature industrial software projects:
- **Traceability**: each workstream references roadmap sources.
- **Acceptance criteria**: every feature track has measurable outcomes.
- **Operational readiness**: runability and fallback behavior are first-class requirements.
- **Safety by design**: deterministic controls are explicit and mandatory.
- **Lifecycle clarity**: includes implementation sequence, DoD, and immediate actions.

For low-level operational and contract details, see `docs/TECHNICAL_REFERENCE.md`.

Contributor workflow docs:
- `docs/DEVELOPER_EDITING_GUIDE.md`
- `docs/METHODOLOGY_OF_ANALOGY.md`
- `docs/CODER_LEXICON.md`

