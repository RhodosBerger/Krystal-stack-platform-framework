# Methodology of Analogy

This project uses analogy as a **design reasoning method** (not as proof). The goal is to convert complex systems into understandable engineering decisions while keeping validation deterministic.

## 1) Why analogy is used

Manufacturing intelligence spans hardware, data systems, and decision engines. Analogy helps teams:
- communicate intent across disciplines,
- reason about architecture boundaries,
- teach concepts quickly,
- preserve a shared mental model.

## 2) Core rule: Analogy guides design, metrics validate design

- Analogy can propose a model.
- Tests, telemetry, and safety constraints accept or reject the model.

## 3) Reusable analogy patterns in this repo

### A) Nervous System Analogy
- Interface layer = sensory nerves.
- Service layer = brain processing.
- HAL = motor/physical actuation.
- Use when deciding where logic belongs.

### B) Shadow Council Analogy
- Creator = proposal engine.
- Auditor = deterministic law.
- Accountant = trade-off scorer.
- Use when implementing safe AI workflow.

### C) Economic Translation Analogy
- SaaS metrics (churn/CAC) mapped to manufacturing costs/risks.
- Use for optimization and reporting semantics.

## 4) Analogy-to-Implementation Protocol

1. **Name the analogy** (e.g., "nervous system").
2. **Map entities to concrete modules**.
3. **Define constraints that cannot be violated**.
4. **Define measurable outcomes**.
5. **Run simulation or controlled validation**.
6. **Accept/reject based on evidence**.

## 5) Failure modes and mitigations

- **Failure mode**: treating metaphor as implementation detail.
  - **Mitigation**: maintain explicit contracts and tests.
- **Failure mode**: analogy drift across teams.
  - **Mitigation**: use shared lexicon and architecture docs.
- **Failure mode**: narrative over safety.
  - **Mitigation**: deterministic policy layer always precedes execution.

## 6) Example conversion template

Use this template when introducing a new analogy:

- Analogy name:
- Problem it clarifies:
- Module mapping:
- Hard constraints:
- Success metrics:
- Validation plan:
- Rollback criteria:

This keeps analogy usage operational, measurable, and safe.
