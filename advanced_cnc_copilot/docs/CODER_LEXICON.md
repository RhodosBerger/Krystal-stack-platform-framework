# Coder Lexicon

A shared vocabulary for contributors. Use these terms consistently in code, docs, and PRs.

## A

- **Action Trace**: Ordered list of actions proposed/applied during a scenario.
- **Auditor Verdict**: Deterministic pass/fail decision on a proposed action set.

## B

- **Blueprint Track**: A feature workstream in `FEATURE_IMPLEMENTATION_BLUEPRINT.md`.
- **Boundary Validation**: Input normalization and schema checks at API/WS edges.

## C

- **Creator**: Agent that proposes strategies or mutations.
- **Contract Field**: A payload field with defined semantics and compatibility expectations.

## D

- **Decision Packet**: Structured unit containing proposal, scores, and reasoned decision.
- **Deterministic Safety Gate**: Hard constraint layer that blocks unsafe actions.

## E

- **Execution Plane**: HAL/hardware/simulator domain where commands become physical effects.

## F

- **Fallback Stream**: Global websocket stream (`/ws/telemetry`) used if machine-scoped stream is unavailable.

## H

- **HAL**: Hardware Abstraction Layer connecting software to CNC hardware/simulator.
- **Hub**: Dashboard entry view linking operational and fleet pages.

## I

- **Interface Layer**: HTTP/WebSocket layer; translation only, minimal logic.

## L

- **Lexicon Term**: Canonical project term expected to remain stable across docs.

## M

- **Machine-Scoped Stream**: WebSocket endpoint `/ws/telemetry/{machine_id}`.
- **Methodics**: Repeatable procedural method for implementing or validating a concept.

## N

- **NFR**: Non-Functional Requirement (reliability, performance, security, observability).

## O

- **Operational Readiness**: State where monitoring, safety, and runbook criteria are met.

## P

- **Preference Pair**: LLM training sample comparing good vs bad plan.
- **Primary Path**: Preferred runtime path before fallback logic.

## R

- **Reason Code**: Machine-readable explanation for pass/fail/block decisions.
- **Rollback Trigger**: Condition requiring reversion to safer prior behavior.

## S

- **Shadow Mode**: Recommendation-only deployment with no autonomous execution.
- **Shadow Council**: Creator + Auditor + Accountant decision framework.

## T

- **Telemetry Window**: Time-series slice used for decision and training.
- **Traceability Crosswalk**: Mapping from roadmap source docs to implementation tracks.

## U

- **Upgrade Safety**: Principle of additive/compatible changes in contracts.

## V

- **Validation Gate**: Any explicit checkpoint that must pass before progression.

## W

- **Workstream**: Cohesive implementation track with acceptance criteria.
