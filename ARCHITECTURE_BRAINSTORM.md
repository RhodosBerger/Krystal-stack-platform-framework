# ARCHITECTURE_BRAINSTORM.md - Key Concepts Summary

This document outlines ambitious future architectural concepts for GAMESA/KrystalStack, focusing on a metacognitive training loop, a hardware-aware economic engine, and a low-code inference ecosystem. It bridges high-level AI concepts with low-level system control.

## 1. Metacognitive Interface: The Self-Reflecting Guardian (Log-Driven Self-Reflection)
-   **Purpose:** Enables the Cognitive Stream (LLM) to analyze its own performance, understand policy impact, and propose improvements with self-awareness, explanations, and confidence scores. This is "thinking about thinking."
-   **Mechanism:** Operates through structured log-driven analysis cycles:
    -   **Trigger:** Periodically, on significant performance events, or explicit request.
    -   **Data Aggregation:** Guardian queries `events.log` (JSONL time-series) and `ExperienceStore` (S, A, R tuples) for relevant time windows.
    -   **Summary Generation:** Aggregated data compacted into a structured summary for the LLM.
    -   **Introspective Prompting:** LLM answers metacognitive questions (e.g., "Which policies correlate with frametime changes?").
-   **LLM Answer Structure:** Must be machine-readable JSON (`PolicyProposal` schema) for Rust to parse, allowing automated validation, simulation, and activation. Includes fields for `proposal_id`, `proposal_type`, `target`, `suggested_value`, `justification`, `confidence`, `introspective_comment`, and `related_metrics`.
-   **Cognitive Analogies:** LLM acts as the "Prefrontal Cortex" evaluating strategies. "Sleep-like Consolidation" processes logs for deeper pattern recognition.

## 2. Economic Engine: Inner Resource Economy
-   **Purpose:** Introduces an internal budgeting system balancing "cost" (power, latency, thermal) vs. "benefit" (FPS, stability, comfort) and "risk" for resource allocation decisions.
-   **Currencies:** Minimal set of internal budgets derived from telemetry: `CPU_Budget` (milliwatts/headroom), `GPU_Budget` (milliwatts/thermal headroom), `Memory_Tier_Budget` (bandwidth/hot slots), `Thermal_Headroom`, `Latency_Budget`, `Time_Budget`.
-   **Action Economic Profile:** Each candidate action has an associated economic profile (estimated costs, expected payoffs, risks).
-   **LLM's Role:** Proposes and refines scoring mechanisms, adapting weights based on `OperatorProfile` (e.g., "gaming" vs. "production").
-   **Interaction:**
    -   **Deterministic Stream (Rust):** Receives `ResourceBudgets`, calculates "utility score" for candidate actions, `ActionGate` selects based on scores and safety guardrails. Logs outcomes.
    -   **Cognitive Stream (LLM/Python):** Refines `PolicyProposal`s and can adjust the Economic Engine's scoring function based on feedback.

## 3. Low-Code Inference Ecosystem
-   **Purpose:** Defines a safe, declarative rule format (`MicroInferenceRule`) that the LLM can generate and the Rust Deterministic Stream can execute.
-   **Proposed Rule Format (JSON Schema for `MicroInferenceRule`):** Allows LLM to express rules structured for Rust parsing. Includes `rule_id`, `version`, `source`, `safety_tier`, `shadow_mode`, `conditions` (metric, operator, value, logical_op), `actions` (action_type, params), `justification`.
-   **Example Rules:** Demonstrates rules for combat-heavy gaming (pinning to P-cores), long rendering workloads (cooldown, memory bias), idle/background streaming (powersave profile).
-   **Evaluation & Conflict Resolution:** Rust orchestrator evaluates conditions, Economic Engine scores candidate actions, conflicts resolved by scores and safety tiers.
-   **Support Mechanisms:**
    -   **Shadow Evaluation:** Rules with `shadow_mode: true` are evaluated hypothetically without execution, logged for Metacognitive analysis.
    -   **Versioning and Rollback:** Rules have `rule_id` and `version` for management.
    -   **Automated Deactivation:** Metacognitive Interface flags and quarantines rules consistently leading to negative rewards or safety violations.

## 4. Safety & Metacognitive Guardrails
-   **Purpose:** Hard constraints to ensure system stability, user comfort, and integrity, preventing AI from overriding critical limits.
-   **Hard Constraints (Deterministic Stream):** Max temperatures, min free RAM, anti-cheat/OS integrity zones (no process injection/code patching, restricted `sysfs`). User overrides and panic switches.
-   **Two-Layer Safety System:**
    -   **Static Checks:** LLM proposes rules with safety justifications; Rust `ActionGate` validates syntax, semantics, conflicts, and resource predictions.
    -   **Dynamic Checks:** Runtime monitors for guardrail breaches, triggering `emergency_cooldown` and providing feedback to the Metacognitive layer.
-   **Learning from Mistakes:** Metacognitive layer tracks `emergency_cooldown`s, negative rewards, and confidence vs. outcome to adjust its proposal patterns, penalizing risky patterns and rewarding conservative ones.
