# Next Week Development Plan (Execution Focus)

## Goal for the week
Ship one full vertical slice that improves real execution confidence: **auditor reason-code contract + fleet selector persistence + simulator dataset export command**.

## Scope (Week Window)

### 1) Auditor reason-code contract (Backend + Docs)
- Add standardized reason-code enum for policy decisions.
- Return reason codes in relevant API and websocket payloads.
- Update technical reference with schema examples.

**Definition of done**
- Reason codes appear in both REST decision responses and live telemetry/decision stream where applicable.
- Backward-compatible payload extension.

### 2) Fleet selector persistence (Frontend)
- Add explicit machine selector in dashboard UI.
- Persist selected machine in URL + local storage.
- Keep machine-scoped websocket as primary with fallback behavior.

**Definition of done**
- User can switch between 3 machines without full app restart.
- Reopening page restores previous machine selection.

### 3) Simulator dataset export command (Training Enabler)
- Add CLI/script to export scenario traces into training-ready schema.
- Include intent, telemetry window, proposal, auditor verdict, reason codes, and outcomes.

**Definition of done**
- Produces deterministic JSONL output with documented schema.
- At least one sample dataset generated via simulator mode.

## Day-by-Day Plan

### Day 1
- Finalize reason-code schema and update docs contract section.
- Add backend model fields and serialization paths.

### Day 2
- Wire reason codes into websocket/decision payload streams.
- Add compatibility guard for clients that ignore new fields.

### Day 3
- Implement fleet selector UI + persistence behavior.
- Validate switching and reconnect behavior.

### Day 4
- Build simulator dataset export command.
- Generate sample JSONL and verify schema.

### Day 5
- Integration smoke checks.
- Documentation update and release notes.
- Prepare demo script for stakeholder review.

## Weekly Risks

- API payload drift across backend/frontend versions.
- Dataset schema ambiguity if simulator traces vary by mode.
- Scope creep into full policy engine refactor.

## Risk Controls

- Keep payload changes additive.
- Freeze schema by mid-week and publish examples.
- Timebox changes to defined vertical slice only.
