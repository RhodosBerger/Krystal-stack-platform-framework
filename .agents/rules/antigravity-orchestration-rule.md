# Antigravity Orchestration Rule

This rule governs all work across the Krystal Stack Platform Framework.

## Execution Pipeline

For every substantial change, execute:

$$\text{REFERENCE} \longrightarrow \text{ANALYZE} \longrightarrow \text{PLAN} \longrightarrow \text{IMPLEMENT} \longrightarrow \text{RUN} \longrightarrow \text{VISUALLY COMPARE} \longrightarrow \text{TEST RULES} \longrightarrow \text{PROFILE} \longrightarrow \text{CORRECT}$$

1. **REFERENCE**: Identify existing reference implementations, specifications, and baseline screenshots or metrics.
2. **ANALYZE**: Examine current behaviors, state models, schemas, and potential failure modes.
3. **PLAN**: Lay out precise, observable verification steps and architectural guardrails before editing code.
4. **IMPLEMENT**: Make disciplined, minimal, cleanly-typed changes adhering to repo style and honesty standards.
5. **RUN**: Execute the code in the live target environment (Python, Godot, Janet, Web, or Java).
6. **VISUALLY COMPARE**: Evaluate output against expectations and reference screenshots/renders.
7. **TEST RULES**: Verify domain invariants, assertions, property tests, and boundary conditions.
8. **PROFILE**: Measure execution time, allocations, frame times, and resource footprint prior to architectural changes.
9. **CORRECT**: Rectify discrepancies found during verification before finalizing.

---

## Domain Directives

- **Graphical Work:**
  Always compare the current build against reference screenshots.

- **Gameplay Work:**
  Always verify the mathematical game domain independently of rendering.

- **Procedural Work:**
  Always test deterministic reproduction from seed.

- **Optimization:**
  Profile before changing architecture.

---

## Completion Invariant

> **Do not mark a stage complete because code exists.**  
> **Mark it complete only after observable verification.**
