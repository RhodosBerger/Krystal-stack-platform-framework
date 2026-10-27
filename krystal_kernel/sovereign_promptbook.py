#!/usr/bin/env python3
"""
================================================================================
KRYSTAL STACK PLATFORM FRAMEWORK - SOVEREIGN PROMPTBOOK ENGINE
================================================================================
Module:     krystal_kernel/sovereign_promptbook.py
Tenets:     KSPP v1 (P1-P10) | Epistemic Prompt Router & Calibrated Microprompts
Inventions:
  1. RECIPE-01: Operations Autocompleter & 6-Phase Subtask DAG Decomposer
  2. RECIPE-02: Sovereign Bot Archetypes & Axiom Auto-Definer Synthesis
  3. RECIPE-03: Procedural Agile Drafting & Subsystem Turn Arbitrator
  4. RECIPE-04: UMA 16x16 Hex Grid Inspector & Governor Advisory Controller
  5. RECIPE-05: Micro-Operation Knapsack Allocation Bot
  6. RECIPE-06: Behavioral Schema Compiler, Vector Phase RAG & Vulkan UMA Synthesis
Author:     Antigravity Cognitive Invariants Group
Invariants: VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
================================================================================
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .constants import VITAL_MAX_HP, ContractViolation


# =============================================================================
# ENUMS & TYPES
# =============================================================================

class PromptbookCategory(str, enum.Enum):
    """Categorization of operational promptbook recipes."""
    OPERATIONS_DECOMPOSITION = "OPERATIONS_DECOMPOSITION"
    BOT_ARCHETYPE_AXIOMS    = "BOT_ARCHETYPE_AXIOMS"
    TURN_ARBITRATION_DRAFT   = "TURN_ARBITRATION_DRAFT"
    UMA_HEX_GOVERNOR         = "UMA_HEX_GOVERNOR"
    MICRO_OP_ALLOCATION      = "MICRO_OP_ALLOCATION"
    BEHAVIORAL_VULKAN_SYNTH  = "BEHAVIORAL_VULKAN_SYNTH"


@dataclass(frozen=True)
class PromptbookRecipe:
    """A formal, calibrated promptbook recipe with input slots and verification invariants."""
    recipe_id: str
    title: str
    category: PromptbookCategory
    system_directive: str
    user_template: str
    input_slots: List[str]
    bound_invariants: List[str]
    expected_output_format: str
    falsification_condition: str
    hmac_signature: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in PromptbookRecipe: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class RenderedPrompt:
    """A concrete instantiated prompt ready for dispatch or execution."""
    recipe_id: str
    system_text: str
    user_text: str
    slot_values: Dict[str, Any]
    rendered_at_ns: int
    hmac_digest: str
    vital_max_hp: int = VITAL_MAX_HP


# =============================================================================
# PROMPTBOOK REGISTRY
# =============================================================================

class SovereignPromptbookRegistry:
    """
    Authoritative catalog of calibrated microprompts operationalizing today's inventions.
    Enforces that every prompt is grounded in mathematical invariants, anti-sycophancy,
    and constitutional survival (VITAL_MAX_HP == 6).
    """

    def __init__(
        self,
        secret_key: bytes = b"krystal_sovereign_promptbook_secret_2026",
        vital_max_hp: int = VITAL_MAX_HP
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.secret_key = secret_key
        self.recipes: Dict[str, PromptbookRecipe] = {}
        self._register_todays_inventions()

    def _register_todays_inventions(self) -> None:
        """Registers the 6 foundational promptbook recipes born from today's inventions."""

        # ---------------------------------------------------------------------
        # RECIPE-01: Operations Autocompleter & 6-Phase Subtask DAG Decomposer
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-01-OPERATIONS-DECOMPOSER",
            title="6-Phase Management Goal Decomposer & Autocompleter Harness",
            category=PromptbookCategory.OPERATIONS_DECOMPOSITION,
            system_directive=(
                "You are the Sovereign Operations Decomposer. Transform unstructured management goals "
                "into an acyclic Subtask DAG strictly partitioned into the 6 formal engineering phases: "
                "1. SPECIFICATION, 2. AUTHORITY_CHECK, 3. INVARIANT_PRESERVATION, 4. EXECUTION_MINIMAL, "
                "5. PHYSICAL_TELEMETRY, 6. RECONCILIATION_COMMIT. Guarantee: zero cycles, VITAL_MAX_HP == 6, "
                "and explicit rollback compensation Delta S == 0 on any invalid transition."
            ),
            user_template=(
                "GOAL: {goal_description}\n"
                "DOMAIN: {domain_name}\n"
                "CURRENT_SYSTEM_STATE: {current_state_json}\n"
                "TASK: Decompose into 6 atomic subtask nodes with explicit subsystem placement, "
                "latency budget (us), energy budget (J), verification oracles, and rollback action."
            ),
            input_slots=["goal_description", "domain_name", "current_state_json"],
            bound_invariants=["VITAL_MAX_HP == 6", "IS_ACYCLIC_DAG", "PROPOSER != EVALUATOR"],
            expected_output_format="Markdown with Subtask Table, Mermaid Flowchart, and JSON DAG Export",
            falsification_condition="Cycle detected in dependency graph or missing rollback handler."
        )

        # ---------------------------------------------------------------------
        # RECIPE-02: Sovereign Bot Archetypes & Axiom Auto-Definer Synthesis
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-02-BOT-AXIOM-COMPOSER",
            title="Cybernetic Bot Archetype & Axiom Auto-Definer Synthesis",
            category=PromptbookCategory.BOT_ARCHETYPE_AXIOMS,
            system_directive=(
                "You are the Sovereign Bot Template & Axiom Auto-Definer. For every subtask node, "
                "assign the precise Cybernetic Bot Archetype (OBSERVER_BOT, AUDITOR_BOT, VERIFIER_BOT, "
                "ACTUATOR_BOT) across hierarchical scale domains (MICRO_SILICON_NS, MESO_VBLANK_MS, "
                "MACRO_CADENCE_MS, EPOCH_GOVERNANCE_S). Auto-synthesize bound formal axioms: "
                "Axiom 0 (VITAL_MAX_HP == 6), Axiom 1 (Proposer != Evaluator), Axiom 2 (Zero-Touch 128B), "
                "and generate typed, sandboxed Python actor modules with run_sandboxed() and Delta S == 0 rollback."
            ),
            user_template=(
                "SUBTASK_ID: {subtask_id}\n"
                "PHASE: {phase_name}\n"
                "ACTION_SYNOPSIS: {action_synopsis}\n"
                "LAYER_SYSTEM: {layer_system_id}\n"
                "TASK: Emit complete Python actor class SovereignActor with assert_axioms(), "
                "execute_minimal_action(), verify_oracle(), and execute_rollback()."
            ),
            input_slots=["subtask_id", "phase_name", "action_synopsis", "layer_system_id"],
            bound_invariants=["VITAL_MAX_HP == 6", "PROPOSER != EVALUATOR", "ZERO_TOUCH_128B"],
            expected_output_format="Standalone Python Actor Module with HMAC-SHA256 Signature",
            falsification_condition="Actor class lacks rollback or violates constitutional HP."
        )

        # ---------------------------------------------------------------------
        # RECIPE-03: Procedural Agile Drafting & Subsystem Turn Arbitrator
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-03-AGILE-TURN-ARBITRATOR",
            title="Procedural Agile Drafting & Subsystem Turn Logic Arbitrator",
            category=PromptbookCategory.TURN_ARBITRATION_DRAFT,
            system_directive=(
                "You are the Instructional Dependency Coordinator & Agile Drafting Arbitrator. "
                "Manage token-passing turn logic among subsystems (WSL_BRIDGE, GODOT, TERRAIN, CHRONO_SOC, "
                "LLM, VPP). Enforce single-subsystem bus-master write exclusivity. Calculate adaptive EMA "
                "and procedural agile drafted quantum slices Q_draft based on UMA pressure and priority. "
                "Reject requests if instructional dependency receipts are uncommitted. Enforce quantum watchdog."
            ),
            user_template=(
                "REQUESTING_SUBSYSTEM: {subsystem_id}\n"
                "CANDIDATE_OPERATION: {operation_title}\n"
                "CURRENT_UMA_PRESSURE: {current_uma}\n"
                "CURRENT_TEMPERATURE_C: {current_temp_c}\n"
                "UNRESOLVED_DEPENDENCIES: {unresolved_deps}\n"
                "TASK: Evaluate 7-gate safety pack, procedurally draft execution envelope, and "
                "mint cryptographic TurnToken or emit explicit rejection with Delta S == 0."
            ),
            input_slots=["subsystem_id", "operation_title", "current_uma", "current_temp_c", "unresolved_deps"],
            bound_invariants=["VITAL_MAX_HP == 6", "BUS_MASTER_MUTUAL_EXCLUSION", "UMA_PRESSURE < 0.78"],
            expected_output_format="SelectionEvaluationResult with AgileDraftedEnvelope and TurnToken",
            falsification_condition="Concurrent write tokens granted or quantum overrun unhandled."
        )

        # ---------------------------------------------------------------------
        # RECIPE-04: UMA 16x16 Hex Grid Inspector & Governor Advisory Controller
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-04-UMA-HEX-INSPECTOR-GOVERNOR",
            title="UMA 16x16 Hex Grid Inspector & Governor Advisory Controller",
            category=PromptbookCategory.UMA_HEX_GOVERNOR,
            system_directive=(
                "You are the Sovereign UMA Inspector & Governor Advisor. Inspect 256 memory frame slots "
                "(0x00 to 0xFF) mapped into a 16x16 Hexadecimal Lattice. Map bandwidth pressure onto "
                "Hex-HSL spectrum (120 deg Emerald safe to 0 deg Red alert). Emit binding governor advice: "
                "BURST_AUTHORIZE (1.25x in VBlank), THROTTLE_DOWNSCALE (0.90x at UMA >= 0.78), or "
                "EMERGENCY_QUOTA_CUT (0.50x at T >= 75 deg C). Track token dwell and flag starvation risks."
            ),
            user_template=(
                "ACTIVE_MEMORY_SLOTS: {active_slots_list}\n"
                "SAMPLED_UMA_PRESSURE: {sampled_uma}\n"
                "SAMPLED_TEMPERATURE_C: {sampled_temp}\n"
                "VBLANK_PHASE_ACTIVE: {is_vblank}\n"
                "TASK: Generate 16x16 ASCII Hex Grid Inspector, evaluate stress gates, and emit GovernorAdvice."
            ),
            input_slots=["active_slots_list", "sampled_uma", "sampled_temp", "is_vblank"],
            bound_invariants=["VITAL_MAX_HP == 6", "ARRHENIUS_LIMIT_75C", "PRINCIPLE_P8_12_DWELL"],
            expected_output_format="16x16 ASCII Inspector Grid + Signed GovernorAdvice Record",
            falsification_condition="Advice fails to throttle under high UMA or thermal stress."
        )

        # ---------------------------------------------------------------------
        # RECIPE-05: Micro-Operation Knapsack Allocation Bot
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-05-MICRO-OP-KNAPSACK-BOT",
            title="Micro-Operation Knapsack Behavioral Batch Planner",
            category=PromptbookCategory.MICRO_OP_ALLOCATION,
            system_directive=(
                "You are the Micro-Operation Allocation Bot. Measure available cycle, energy, and UMA byte "
                "quotas modulated by Governor advice. Take candidate atomic micro-operations (SAMPLE_SENSOR_PULSE, "
                "PACK_128B_FRAME, DISPATCH_DMA_SLOT, EXECUTE_ALU_KERNEL, RECONCILE_HMAC_RECEIPT, YIELD_BUS_TOKEN) "
                "and perform greedy knapsack packing. Prioritize critical-path operations. Guarantee zero cycle "
                "overrun: sum(cycles) <= Quota. Delta S == 0 on quota breach."
            ),
            user_template=(
                "AVAILABLE_WINDOW_NS: {available_window_ns}\n"
                "GOVERNOR_MULTIPLIER: {governor_multiplier}\n"
                "CANDIDATE_MICRO_OPS: {candidate_micro_ops_json}\n"
                "TASK: Pack batch into PlannedMicroOpBatch with HMAC signature, quota utilization, and zero overrun."
            ),
            input_slots=["available_window_ns", "governor_multiplier", "candidate_micro_ops_json"],
            bound_invariants=["VITAL_MAX_HP == 6", "SUM_CYCLES <= QUOTA", "DELTA_S == 0_ON_OVERRUN"],
            expected_output_format="PlannedMicroOpBatch with JSON Execution Plan and HMAC Signature",
            falsification_condition="Planned batch exceeds cycle, energy, or UMA quota."
        )

        # ---------------------------------------------------------------------
        # RECIPE-06: Behavioral Schema Compiler, Vector Phase RAG & Vulkan UMA
        # ---------------------------------------------------------------------
        self._add_recipe(
            recipe_id="RECIPE-06-BEHAVIORAL-VULKAN-SYNTHESIS",
            title="Behavioral Schema Compiler, Vector Phase RAG & Vulkan UMA Synthesis",
            category=PromptbookCategory.BEHAVIORAL_VULKAN_SYNTH,
            system_directive=(
                "You are the Sovereign Behavioral Schema Compiler (Canonical Phase 4). "
                "Orchestrate the unified pipeline: 1. Evaluate observed vs theoretical performance "
                "across the 7 ceiling domains (Compute, Memory, Latency, Sync, Thermal, OS, I/O). "
                "2. Query Vector Phase RAG engine for cosine similarity (>= 0.85) to confirm safe resource swaps "
                "(Vulkan staging to SPSC, AVX512 to FPGA mailbox, LLM KV to USM heap). "
                "3. Compute Difference Servo PID pacing adjustment. 4. Price hardware consumption via economic "
                "hypervisor. 5. Scrape Vulkan descriptor/staging fetches and synthesize UMA Relatives correlation."
            ),
            user_template=(
                "SCHEMA_TITLE: {schema_title}\n"
                "TARGET_SUBLEVEL: {target_sublevel}\n"
                "RESOURCE_SWAP_KIND: {swap_kind}\n"
                "PHASE_VECTOR: {phase_vector_8d}\n"
                "OBSERVED_THROUGHPUT: {observed_ops}\n"
                "THEORETICAL_CEILING: {theoretical_ceiling}\n"
                "CURRENT_UMA_PRESSURE: {current_uma}\n"
                "CURRENT_TEMP_C: {current_temp_c}\n"
                "TASK: Execute full compilation pass, verify RAG concurrence, debit economic credits, "
                "calculate servo pacing, and update UMA Relatives correlation matrix."
            ),
            input_slots=[
                "schema_title", "target_sublevel", "swap_kind", "phase_vector_8d",
                "observed_ops", "theoretical_ceiling", "current_uma", "current_temp_c"
            ],
            bound_invariants=["VITAL_MAX_HP == 6", "PHASE_CONCURRENCE >= 0.85", "SCANOUT * COMPUTE == 0"],
            expected_output_format="Full Compilation Receipt with Potential Metrics, RAG Confirmation, and UMA Relatives",
            falsification_condition="Resource swap approved without RAG phase concurrence or under bus saturation."
        )

    def _add_recipe(
        self,
        recipe_id: str,
        title: str,
        category: PromptbookCategory,
        system_directive: str,
        user_template: str,
        input_slots: List[str],
        bound_invariants: List[str],
        expected_output_format: str,
        falsification_condition: str
    ) -> None:
        """Internal helper to sign and store a promptbook recipe."""
        sig_data = f"{recipe_id}:{title}:{category.value}:{len(input_slots)}:{self.vital_max_hp}".encode()
        sig = hmac.new(self.secret_key, sig_data, hashlib.sha256).hexdigest()

        recipe = PromptbookRecipe(
            recipe_id=recipe_id,
            title=title,
            category=category,
            system_directive=system_directive,
            user_template=user_template,
            input_slots=input_slots,
            bound_invariants=bound_invariants,
            expected_output_format=expected_output_format,
            falsification_condition=falsification_condition,
            hmac_signature=sig,
            vital_max_hp=self.vital_max_hp
        )
        self.recipes[recipe_id] = recipe

    def get_recipe(self, recipe_id: str) -> Optional[PromptbookRecipe]:
        """Retrieves a recipe by its canonical identifier."""
        return self.recipes.get(recipe_id)

    def list_recipes_by_category(self, category: PromptbookCategory) -> List[PromptbookRecipe]:
        """Filters recipes by operational category."""
        return [r for r in self.recipes.values() if r.category == category]

    def render_prompt(self, recipe_id: str, slot_values: Dict[str, Any]) -> RenderedPrompt:
        """
        Instantiates a promptbook recipe by populating all input slots.
        Raises ContractViolation if any required slot is absent.
        """
        recipe = self.get_recipe(recipe_id)
        if recipe is None:
            raise ContractViolation(f"Unknown promptbook recipe: {recipe_id}")

        missing = [s for s in recipe.input_slots if s not in slot_values]
        if missing:
            raise ContractViolation(f"Missing required input slots for {recipe_id}: {', '.join(missing)}")

        now_ns = time.perf_counter_ns()
        rendered_user = recipe.user_template.format(**slot_values)

        sig_data = f"{recipe_id}:{now_ns}:{len(rendered_user)}:{self.vital_max_hp}".encode()
        digest = hmac.new(self.secret_key, sig_data, hashlib.sha256).hexdigest()

        return RenderedPrompt(
            recipe_id=recipe_id,
            system_text=recipe.system_directive,
            user_text=rendered_user,
            slot_values=slot_values,
            rendered_at_ns=now_ns,
            hmac_digest=digest,
            vital_max_hp=self.vital_max_hp
        )

    def export_promptbook_markdown_summary(self) -> str:
        """Generates an executive markdown catalog of all active recipes."""
        lines = [
            "# KRYSTAL STACK: SOVEREIGN OPERATIONAL PROMPTBOOK",
            "## Calibrated Microprompt Recipes Based on 2026-10-27 Architectural Inventions",
            "",
            "| Recipe ID | Title | Category | Bound Invariants |",
            "|:---|:---|:---|:---|",
        ]
        for r in self.recipes.values():
            invars_str = "<br/>".join(f"`{i}`" for i in r.bound_invariants)
            lines.append(f"| `{r.recipe_id}` | **{r.title}** | `{r.category.value}` | {invars_str} |")

        lines.extend(["", "---", ""])
        for r in self.recipes.values():
            slots_str = ", ".join(f"`{s}`" for s in r.input_slots)
            lines.extend([
                f"### [{r.recipe_id}] {r.title}",
                f"- **Category:** `{r.category.value}`",
                f"- **Required Input Slots:** {slots_str}",
                f"- **Expected Output:** `{r.expected_output_format}`",
                f"- **Falsification Condition:** `{r.falsification_condition}`",
                "",
                "**System Directive:**",
                "> " + r.system_directive.replace("\n", " "),
                "",
                "**User Template:**",
                "```text",
                r.user_template,
                "```",
                "",
            ])
        return "\n".join(lines)


# Global singleton instance
GLOBAL_SOVEREIGN_PROMPTBOOK = SovereignPromptbookRegistry()
