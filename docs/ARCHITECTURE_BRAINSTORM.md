# ARCHITECTURE_BRAINSTORM.md - Key Concepts Summary

This document outlines ambitious future architectural concepts for GAMESA/KrystalStack, focusing on a metacognitive training loop, a hardware-aware economic engine, and a low-code inference ecosystem. It bridges high-level AI concepts with low-level system control.

## 1. Metacognitive Interface: The Self-Reflecting Guardian (Log-Driven Self-Reflection)

**Purpose:** Enables the Cognitive Stream (LLM) to analyze its own performance, understand policy impact, and propose improvements with self-awareness, explanations, and confidence scores. This is "thinking about thinking."

**Mechanism:** Operates through structured log-driven analysis cycles:
- **Trigger:** Periodically, on significant performance events, or explicit request.
- **Data Aggregation:** Guardian queries `events.log` (JSONL time-series) and `ExperienceStore` (S, A, R tuples) for relevant time windows.
- **Summary Generation:** Aggregated data compacted into a structured summary for the LLM.
- **Introspective Prompting:** LLM answers metacognitive questions (e.g., "Which policies correlate with frametime changes?").

**LLM Answer Structure:** Must be machine-readable JSON (`PolicyProposal` schema) for Rust to parse, allowing automated validation, simulation, and activation. Includes fields for `proposal_id`, `proposal_type`, `target`, `suggested_value`, `justification`, `confidence`, `introspective_comment`, and `related_metrics`.

**Cognitive Analogies:** LLM acts as the "Prefrontal Cortex" evaluating strategies. "Sleep-like Consolidation" processes logs for deeper pattern recognition.

## 2. Economic Engine: Inner Resource Economy

**Purpose:** Introduces an internal budgeting system balancing "cost" (power, latency, thermal) vs. "benefit" (FPS, stability, comfort) and "risk" for resource allocation decisions.

**Currencies:** Minimal set of internal budgets derived from telemetry:
- `CPU_Budget` (milliwatts/headroom)
- `GPU_Budget` (milliwatts/thermal headroom)
- `Memory_Tier_Budget` (bandwidth/hot slots)
- `Thermal_Headroom`
- `Latency_Budget`
- `Time_Budget`

**Action Economic Profile:** Each candidate action has an associated economic profile (estimated costs, expected payoffs, risks).

**LLM's Role:** Proposes and refines scoring mechanisms, adapting weights based on `OperatorProfile` (e.g., "gaming" vs. "production").

**Interaction:**
- **Deterministic Stream (Python/Rust):** Receives `ResourceBudgets`, calculates "utility score" for candidate actions, `ActionGate` selects based on scores and safety guardrails. Logs outcomes.
- **Cognitive Stream (LLM/Python):** Refines `PolicyProposal`s and can adjust the Economic Engine's scoring function based on feedback.

## 3. Low-Code Inference Ecosystem

**Purpose:** Defines a safe, declarative rule format (`MicroInferenceRule`) that the LLM can generate and the Python Deterministic Stream can execute.

**Proposed Rule Format (JSON Schema for `MicroInferenceRule`):**

```json
{
  "rule_id": "string (unique identifier)",
  "version": "semver (e.g., 1.0.0)",
  "source": "string (LLM, user, generated)",
  "safety_tier": "enum (STRICT, EXPERIMENTAL, DEBUG)",
  "shadow_mode": "boolean (evaluate but don't execute)",
  "conditions": [
    {
      "metric": "string (e.g., temperature, fps, cpu_util)",
      "operator": "enum (>, <, ==, !=, >=, <=)",
      "value": "number or string",
      "logical_op": "enum (AND, OR, optional)"
    }
  ],
  "actions": [
    {
      "action_type": "string (e.g., set_affinity, boost_gpu, reduce_power)",
      "params": {
        "key": "value"
      }
    }
  ],
  "justification": "string (human-readable explanation)",
  "confidence": "float 0-1"
}
```

**Example Rules:**

1. **Combat-heavy gaming** (pinning to P-cores):
```json
{
  "rule_id": "combat_p_core_affinity",
  "version": "1.0.0",
  "source": "LLM",
  "safety_tier": "STRICT",
  "shadow_mode": false,
  "conditions": [
    {"metric": "game_state", "operator": "==", "value": "combat", "logical_op": "AND"},
    {"metric": "thermal_headroom", "operator": ">", "value": 10}
  ],
  "actions": [
    {"action_type": "set_affinity", "params": {"cores": "0-3", "priority": "high"}}
  ],
  "justification": "Combat requires low latency, P-cores provide best single-thread performance",
  "confidence": 0.85
}
```

2. **Long rendering workloads**:
```json
{
  "rule_id": "render_cooldown_memory",
  "version": "1.0.0",
  "source": "LLM",
  "safety_tier": "EXPERIMENTAL",
  "shadow_mode": false,
  "conditions": [
    {"metric": "app_category", "operator": "==", "value": "creative", "logical_op": "AND"},
    {"metric": "cpu_util", "operator": ">", "value": 0.8, "logical_op": "AND"},
    {"metric": "duration", "operator": ">", "value": 300}
  ],
  "actions": [
    {"action_type": "schedule_cooldown", "params": {"interval": 600}},
    {"action_type": "memory_bias", "params": {"tier": "WARM"}}
  ],
  "justification": "Long renders benefit from thermal management and memory efficiency",
  "confidence": 0.72
}
```

3. **Idle/background streaming**:
```json
{
  "rule_id": "idle_powersave",
  "version": "1.0.0",
  "source": "LLM",
  "safety_tier": "STRICT",
  "shadow_mode": false,
  "conditions": [
    {"metric": "cpu_util", "operator": "<", "value": 0.15, "logical_op": "AND"},
    {"metric": "gpu_util", "operator": "<", "value": 0.1}
  ],
  "actions": [
    {"action_type": "apply_profile", "params": {"profile": "powersave"}}
  ],
  "justification": "Low utilization indicates idle state, powersave reduces energy waste",
  "confidence": 0.95
}
```

**Evaluation & Conflict Resolution:**
- Python orchestrator evaluates conditions
- Economic Engine scores candidate actions
- Conflicts resolved by scores and safety tiers (STRICT > EXPERIMENTAL > DEBUG)
- Shadow mode rules evaluated but not executed (logged for analysis)

**Support Mechanisms:**
- **Shadow Evaluation:** Rules with `shadow_mode: true` are evaluated hypothetically without execution, logged for Metacognitive analysis.
- **Versioning and Rollback:** Rules have `rule_id` and `version` for management.
- **Automated Deactivation:** Metacognitive Interface flags and quarantines rules consistently leading to negative rewards or safety violations.

## 4. Safety & Metacognitive Guardrails

**Purpose:** Hard constraints to ensure system stability, user comfort, and integrity, preventing AI from overriding critical limits.

**Hard Constraints (Deterministic Stream):**
- Max temperatures (thermal throttle - 5°C safety margin)
- Min free RAM (15% minimum)
- Anti-cheat/OS integrity zones (no process injection/code patching, restricted `sysfs`)
- User overrides and panic switches

**Two-Layer Safety System:**

1. **Static Checks (Pre-deployment):**
   - LLM proposes rules with safety justifications
   - Python `ActionGate` validates:
     - Syntax and schema compliance
     - Semantic correctness (valid metrics, actions)
     - Conflict detection with existing rules
     - Resource prediction within budgets

2. **Dynamic Checks (Runtime):**
   - Runtime monitors for guardrail breaches
   - Triggers `emergency_cooldown` on violations
   - Provides feedback to Metacognitive layer

**Learning from Mistakes:**
- Metacognitive layer tracks:
  - `emergency_cooldown` events
  - Negative rewards
  - Confidence vs. outcome correlation
- Adjusts proposal patterns:
  - Penalizes risky patterns
  - Rewards conservative, successful patterns
  - Updates confidence calibration

## 5. Modular LLM Integration Architecture

**Design Principles:**
- **Backend Agnosticism:** Support multiple LLM providers (OpenAI, Anthropic, local models)
- **Pluggable Tools:** Easy integration of external functionalities
- **Flexible I/O:** Handle diverse input/output modalities
- **Conversation Management:** Maintain context and memory

**Directory Structure:**
```
src/python/metacognitive/
├── llm_integrations/
│   ├── base_connector.py      # BaseLLMConnector abstract class
│   ├── openai_connector.py    # OpenAI API integration
│   ├── anthropic_connector.py # Anthropic Claude integration
│   └── local_connector.py     # Local model integration
├── tools/
│   ├── tool_registry.py       # ToolRegistry for managing tools
│   ├── calculator.py          # Precise math calculations
│   ├── web_search.py          # Real-time information retrieval
│   ├── telemetry_analyzer.py  # GAMESA-specific telemetry analysis
│   └── rule_validator.py      # Rule syntax/safety validation
├── bot_core.py                # ConversationManager, main orchestrator
├── metacognitive_engine.py    # Main metacognitive reasoning engine
└── event_logger.py            # JSONL structured event logging
```

**Key Components:**

1. **BaseLLMConnector** - Abstract interface for LLM backends
2. **ToolRegistry** - Dynamic tool registration and invocation
3. **ConversationManager** - Context and history management
4. **MetacognitiveEngine** - Orchestrates analysis, proposals, and learning
5. **EventLogger** - Structured JSONL logging for analysis

## Relationship to GAMESA (`gamesa_tweaker` project):

- This document provides a roadmap for the future evolution of our `UnifiedBrain` and `GamesaFramework`.
- The "Generative Engine" concept (AI returning policy proposals) is a direct step towards implementing a `MicroInferenceRule` system.
- Our existing rule engine and safety guardrails are the foundational building blocks for the Metacognitive Interface.
- The insights into safety and adaptive learning are crucial for making our system more robust and reliable.

## Implementation Phases:

### Phase 1: Foundation (Current - Wave 2)
- ✅ Basic MicroInferenceRule system
- ✅ Safety guardrails
- ✅ Telemetry collection
- ✅ Neural learning (policy network)

### Phase 2: Metacognitive Logging (Wave 3)
- JSONL event logger
- Experience store (S, A, R tuples)
- Log query interface
- Performance summary generation

### Phase 3: LLM Integration (Wave 3-4)
- Modular LLM connector architecture
- Tool registry and tool integration
- Conversation manager
- PolicyProposal generation

### Phase 4: Economic Engine Enhancement (Wave 4)
- Formal resource currencies
- Action cost/benefit/risk profiles
- Utility scoring
- Operator profiles

### Phase 5: Full Metacognitive Loop (Wave 4-5)
- Automated policy proposal
- Shadow evaluation
- Rule versioning and rollback
- Automated deactivation of harmful rules
- Confidence calibration

---

**Vision:** GAMESA evolves from a reactive rule-based optimizer to a self-aware, self-improving system that learns from experience, proposes optimizations, and continuously refines its strategies while maintaining strict safety guarantees.
