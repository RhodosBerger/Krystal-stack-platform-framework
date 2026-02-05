# Technical Reference (Production-Style Documentation)

> This document follows common standards used in industrial software projects: system context, component contracts, operational runbook, non-functional requirements, and release gates.

## 1. System Context

FANUC RISE consists of five technical planes:
1. **Control plane**: APIs, auth, orchestration, RBAC.
2. **Data plane**: telemetry ingestion, storage, query and aggregation.
3. **Decision plane**: Shadow Council agents (Creator, Auditor, Accountant).
4. **Execution plane**: HAL/FOCAS bridge, simulator fallback.
5. **Experience plane**: dashboard/hub/fleet UX.

## 2. Core Interfaces (contracts)

### REST interfaces
- `GET /api/health`: liveness/readiness summary.
- `POST /api/manufacturing/request`: unified orchestration entrypoint.
- `POST /conduct`: creative scenario endpoint.
- `POST /optimize`: optimization endpoint with material context.

### WebSocket interfaces
- Primary machine-scoped stream: `/ws/telemetry/{machine_id}`.
- Global fallback stream: `/ws/telemetry`.

### Contract requirements
- Payloads must include `timestamp`, machine identity, and typed telemetry fields.
- Safety-related actions must include a machine-readable `reason_code`.
- Backward compatibility policy: additive changes only for minor versions.

## 3. Non-Functional Requirements (NFR)

### Reliability
- Circuit breaker in HAL path.
- Simulator fallback when hardware path is degraded.
- Explicit retry strategy for websocket reconnection.

### Performance
- Telemetry pipeline objective: p95 ingest->UI under 100ms in simulation.
- API p95 latency budget defined per endpoint class (read/write/control).

### Security
- RBAC-scoped access for operator/engineer/admin roles.
- Token-based authentication for control endpoints.
- Audit logs for all policy-overridden or blocked actions.

### Observability
- Health, error rate, and throughput per subsystem.
- Decision traces for AI + deterministic auditor outcomes.
- Alerting for telemetry stalls and repeated fallback events.

## 4. Safety Model (critical technical details)

Safety is deterministic-first:
1. Creator can propose only.
2. Auditor validates against hard physics/policy limits.
3. Accountant score is advisory and never bypasses constraints.
4. Execution only after deterministic pass.

**Important:** model confidence is not considered a safety signal; only policy conformance and measured machine state are valid safety gates.

## 5. Data and Training Pipeline

### Dataset minimum schema
- `intent_text`
- `machine_id`
- `telemetry_window`
- `candidate_actions`
- `auditor_verdict`
- `reason_codes`
- `execution_outcome`
- `economic_score`

### Training pipeline stages
1. Scenario generation (normal/fault/adversarial).
2. Supervised fine-tuning dataset assembly.
3. Preference/ranking dataset assembly.
4. Offline benchmark and safety scoring.
5. Shadow deployment with rejection telemetry.

## 6. Release Readiness Checklist

A release candidate is acceptable only if:
- Dependency bootstrap script passes in clean environment.
- API health/readiness checks pass.
- Websocket primary/fallback behavior is validated.
- Safety policy regression suite passes.
- Documentation updates are complete and versioned.

## 7. Recommended Documentation Set (for this repo)

- `README.md`: entry-level setup and links.
- `SYSTEM_ARCHITECTURE.md`: conceptual and flow architecture.
- `FEATURE_IMPLEMENTATION_BLUEPRINT.md`: delivery execution plan.
- `docs/TECHNICAL_REFERENCE.md` (this file): contracts, NFRs, runbook-grade guidance.
