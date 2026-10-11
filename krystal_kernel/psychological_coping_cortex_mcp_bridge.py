#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: PSYCHOLOGICAL COPING CORTEX & MCP PROCESSOR-WHISPERER BRIDGE
==============================================================================
Module: krystal_kernel/psychological_coping_cortex_mcp_bridge.py
Description: Advanced cognitive triage and hardware actuation framework that
             replaces naive timestamp/energy task sorting with psychological
             coping mechanism analogies (Sublimation, Repression, Projection,
             Rationalization, Compartmentalization, Hypervigilance, Regression).

             Bridges the Cortex Integrity Engine, nanosecond Processor Whisperer,
             and Model Context Protocol (MCP) to:
               1. Triage UI execution order and tensor semantic synonym evaluation.
               2. Inject speculative instruction hints into 64-bit Janet bytecode.
               3. Unlock the full silicon die potential artificially restricted
                  by OEM marketing segmentation (proving i5 == i7 silicon die).

Platform Invariant: VITAL_MAX_HP = 6
Author: Dušan Kopecký & Krystal-Stack Architecture & Hardware Systems Council (2026)

Changes 2026-10-09 (BREAKING for consumers of the returned dicts):
  - priority_rank renumbered to a strict order: REGRESSION 1, PROJECTION 2 (was 1),
    COMPARTMENTALIZATION 3 (was 2), SUBLIMATION 4 (was 3).
  - primary_coping_mechanism is null when no token matches (was SUBLIMATION).
  - embedding_bandwidth_saved_pct is derived from model constants (was constant 78.5).
  - Silicon report: physical_die_parity_confirmed=False, boost is a clock-ratio upper
    bound (88.9 %, was 193.8 %), stock_dram_rapl_limited=None (unknown),
    unlocked_dram_rapl_active=False; new fields figure_status, msr_writes_performed,
    requested_pl1_watts, die_parity_note.
  - Numeric inputs: bool is rejected; null -> documented default; NaN/inf -> REGRESSION.
  Formal background: krystal_theory/ (01_triazny_operator.md, 02_planovanie_a_poradie.md).
==============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import math
import uuid
import struct
import json
import threading
from collections import deque
from enum import Enum, IntEnum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable

# NOTE: stdout/stderr UTF-8 reconfiguration lives in main(). A library module must
# not mutate process-wide streams on import (it would affect every importer,
# including stdio-transport MCP servers).

# Universal Architectural Invariant
VITAL_MAX_HP: int = 6


def _coerce_float(value: Any, name: str, default: Optional[float] = None) -> float:
    """
    Strict numeric coercion shared by every entry point.
      - None -> default (TypeError when no default exists).
      - bool -> TypeError (True would otherwise silently become 1.0).
      - Too large for a float -> ValueError (float() raises OverflowError, which callers
        catching (TypeError, ValueError) would otherwise miss).
      - NaN / inf pass through on purpose: triage treats them as unknown state
        (fail-safe REGRESSION), not as a type error.
    """
    if value is None:
        if default is None:
            raise TypeError(f"{name} is required")
        return float(default)
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, not bool")
    try:
        return float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is out of float range") from exc


# ============================================================================
# 1. PSYCHOLOGICAL COPING TAXONOMY & PRIORITY SYSTEM
# ============================================================================

class PsychologicalCopingMechanism(str, Enum):
    """
    Cognitive coping mechanisms adapted as computational resource allocation policies.
    Derived from clinical neurosis and ego defense psychology to manage system stress.
    """
    SUBLIMATION = "SUBLIMATION"
    # Transforming raw, chaotic input entropy (unstructured noise / sensory bursts)
    # into high-order structured artistic or procedural rendering (ASCII, KSYN bytecode).

    REPRESSION = "REPRESSION"
    # Active suppression of redundant, low-priority, or jittery background tasks.
    # Disables background SSD logging and diagnostic polling during critical frame bursts.

    PROJECTION = "PROJECTION"
    # Anticipating external reality by projecting internal speculative state outward.
    # Drives K-ISA speculative instruction execution, L1 prefetching, and 120Hz interpolation.

    RATIONALIZATION = "RATIONALIZATION"
    # Approximating heavy, complex, painful computational loads with structured,
    # loss-tolerant heuristics (INT8/INT4 quantization, Tile4 spatial compression).

    COMPARTMENTALIZATION = "COMPARTMENTALIZATION"
    # Strictly segregating conflicting mental/data processes into isolated chambers.
    # Isolates Game VRAM heaps and Local LLM token generators into disjoint UMA slabs.

    HYPERVIGILANCE = "HYPERVIGILANCE"
    # Chronic ultra-alert scanning for microscopic hazards.
    # Nanosecond thermal watchdog, monitoring junction temp (Tj) and Arrhenius aging (AF).

    REGRESSION = "REGRESSION"
    # Reverting to a known, primitive, bulletproof baseline when under acute trauma.
    # Emergency fallback to AVX2/SSE4.2 legacy SIMD pipelines if thermal fuse is approached.


@dataclass
class CopingPriorityDescriptor:
    """Triage assessment produced by the psychological coping protocol."""
    task_id: str
    task_name: str
    coping_mechanism: PsychologicalCopingMechanism
    priority_rank: int                 # 1 (Highest / Immediate) to 7 (Background / Postponed)
    allocated_time_budget_us: float    # Nanosecond/microsecond execution slice
    speculative_depth: int             # Speculative lookahead depth (0 to 4 words)
    memory_allocation_policy: str      # UMA Slab type
    semantic_clustering_enabled: bool  # Suppresses synonym search space in LLM
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["coping_mechanism"] = self.coping_mechanism.value
        return d


# ============================================================================
# 2. HARDWARE SILICON POTENTIAL & MARKETING BINNING UNLOCKER
# ============================================================================

@dataclass
class SiliconDieUnlockReport:
    """
    Modelled (NOT measured) power/clock envelope for the Tiger Lake-UP3 die.

    Scope of the "parity" statement: Core i5-1135G7 and Core i7-1165G7 are cut
    from the same Tiger Lake-UP3 die design, but the i5 ships with fused-off
    resources (80 vs 96 Xe EUs, 8 MB vs 12 MB L3). Fuses cannot be re-enabled
    through MSRs, so die-design parity does not imply performance parity.
    This module reads and writes no MSR; every figure is a model input.
    """
    physical_die_architecture: str
    marketing_sku_detected: str
    physical_die_parity_confirmed: bool
    stock_pl1_watts: float
    unlocked_pl1_watts: float
    stock_ring_bus_ghz: float
    unlocked_ring_bus_ghz: float
    stock_dram_rapl_limited: Optional[bool]   # None = unknown (never probed)
    unlocked_dram_rapl_active: bool
    stock_thermal_trip_c: float
    unlocked_safe_trip_c: float
    gpu_slice_undervolt_mv: float
    sustained_all_core_ghz_stock: float
    sustained_all_core_ghz_unlocked: float
    silicon_wear_af: float
    overall_die_efficiency_boost_pct: float
    vital_max_hp: int = VITAL_MAX_HP
    requested_pl1_watts: float = 0.0
    figure_status: str = "MODEL_UNVERIFIED"
    msr_writes_performed: bool = False
    die_parity_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SiliconDieUnlocker:
    """
    Builds a *modelled* envelope for Intel Tiger Lake-UP3 power/clock limits.
    It is a planning aid: it performs no MSR access and confirms nothing on hardware.
    """

    # Model bounds for a PL1 request (W). 15 W is the i5-1135G7 nominal TDP; 35 W is
    # the model ceiling, not a validated safe limit for any particular chassis.
    PL1_MODEL_MIN_W: float = 15.0
    PL1_MODEL_MAX_W: float = 35.0

    @classmethod
    def unlock_full_silicon_potential(cls, target_pl1_w: float = 32.0) -> SiliconDieUnlockReport:
        """
        Returns the envelope that the following changes would *target*. Nothing is written.
          - MSR 0x610 (PKG_POWER_LIMIT): PL1 15 W -> request, clamped to [15, 35] W.
          - MSR 0x620 (UNCORE_RATIO_LIMIT): ring/uncore target 2.8 GHz.
          - MSR 0x618 (DRAM_POWER_LIMIT): DRAM RAPL is typically not exposed on
            client SKUs -- unverified on this host.
          - DPTF passive trip 72 °C -> 85 °C (OEM policy, chassis-dependent).
          - GPU slice -40 mV undervolt (model input; stability unverified).

        Raises ValueError for a non-finite request.
        """
        requested = _coerce_float(target_pl1_w, "target_pl1_w")
        if not math.isfinite(requested):
            raise ValueError(f"target_pl1_w must be finite, got {target_pl1_w!r}")

        stock_pl1 = 15.0
        # Previously clamped to [28, 35], which silently turned a 10 W request into 28 W.
        unlocked_pl1 = min(cls.PL1_MODEL_MAX_W, max(cls.PL1_MODEL_MIN_W, requested))

        stock_ring = 1.8
        stock_clock = 1.8
        if unlocked_pl1 > stock_pl1:
            unlocked_ring = 2.8   # model target
            unlocked_clock = 3.4  # model target, not a measured sustained clock
        else:
            # No extra power budget requested -> the model must not predict a gain.
            unlocked_ring = stock_ring
            unlocked_clock = stock_clock

        af = 0.9118  # model input (Arrhenius factor), not derived in this function

        # Upper bound for a compute-bound workload at constant IPC: throughput scales at
        # most linearly with core clock. The old formula multiplied core clock by ring
        # clock (GHz x GHz), which has no throughput interpretation (it produced 193.8%).
        gain_pct = round((unlocked_clock / stock_clock - 1.0) * 100.0, 1)

        return SiliconDieUnlockReport(
            physical_die_architecture="Intel 10nm SuperFin Willow Cove (Gen12 GT2)",
            marketing_sku_detected="Intel Core i5-1135G7 (assumed, not detected)",
            physical_die_parity_confirmed=False,
            stock_pl1_watts=stock_pl1,
            unlocked_pl1_watts=unlocked_pl1,
            stock_ring_bus_ghz=stock_ring,
            unlocked_ring_bus_ghz=unlocked_ring,
            stock_dram_rapl_limited=None,     # unknown: DRAM RAPL was never probed
            unlocked_dram_rapl_active=False,  # nothing was written, so nothing is active
            stock_thermal_trip_c=72.0,  # OEM conservative trip
            unlocked_safe_trip_c=85.0,   # Intel physical specification max is 100°C
            gpu_slice_undervolt_mv=-40.0,
            sustained_all_core_ghz_stock=stock_clock,
            sustained_all_core_ghz_unlocked=unlocked_clock,
            silicon_wear_af=af,
            overall_die_efficiency_boost_pct=gain_pct,
            vital_max_hp=VITAL_MAX_HP,
            requested_pl1_watts=requested,
            figure_status="MODEL_UNVERIFIED",
            msr_writes_performed=False,
            die_parity_note=(
                "Same die design as i7-1165G7, but i5 has fused-off EUs (80 vs 96) and "
                "L3 (8 vs 12 MB); fuses are not MSR-reversible. No hardware was probed."
            ),
        )


# ============================================================================
# 3. SEMANTIC SYNONYM COMPILATION OPTIMIZER (LLM EMBEDDING ACCELERATOR)
# ============================================================================

class SemanticSynonymCompilationOptimizer:
    """
    Keyword classifier that maps query tokens onto coping buckets.

    The reported "bandwidth saved" is computed from two declared model constants
    (MODEL_VOCAB_SIZE, MODEL_CLUSTER_SIZE); it is not a measurement. Note that a real
    embedding lookup is one row gather per token -- vocabulary-size cost is paid in
    the output projection/softmax, so this model is an upper-bound illustration only.
    """

    MODEL_VOCAB_SIZE: int = 32000
    MODEL_CLUSTER_SIZE: int = 128
    _TOKEN_STRIP: str = ".,;:!?\"'()[]{}<>"
    # Only regular inflections extend a keyword ("rendering", "predicted", "isolated").
    # Free prefix matching made "mutex" -> "mute", "codec" -> "code", "blockchain" -> "block".
    _INFLECTION_SUFFIXES: Tuple[str, ...] = ("s", "es", "d", "ed", "ing")

    # Categorical coping semantic buckets
    COPING_SEMANTIC_BUCKETS: Dict[PsychologicalCopingMechanism, List[str]] = {
        PsychologicalCopingMechanism.SUBLIMATION: [
            "create", "render", "synthesize", "compose", "generate", "transform", "art", "code", "shader"
        ],
        PsychologicalCopingMechanism.REPRESSION: [
            # "quarantine" removed: it was also in COMPARTMENTALIZATION and double-counted.
            "suppress", "halt", "block", "ignore", "drop", "mute", "silence", "throttle"
        ],
        PsychologicalCopingMechanism.PROJECTION: [
            "predict", "forecast", "interpolate", "speculate", "prefetch", "anticipate", "lookahead"
        ],
        PsychologicalCopingMechanism.RATIONALIZATION: [
            "quantize", "compress", "approximate", "simplify", "downsample", "streamline", "prune"
        ],
        PsychologicalCopingMechanism.COMPARTMENTALIZATION: [
            "isolate", "sandbox", "partition", "slice", "segregate", "divide", "quarantine", "decouple"
        ],
        PsychologicalCopingMechanism.HYPERVIGILANCE: [
            "watch", "audit", "guard", "monitor", "telemetry", "assert", "verify", "fuse", "sentinel"
        ],
        PsychologicalCopingMechanism.REGRESSION: [
            "fallback", "revert", "baseline", "rollback", "recover", "safe_mode", "legacy", "simd"
        ]
    }

    @classmethod
    def _token_matches(cls, word: str, keyword: str) -> bool:
        """Exact match, or keyword + regular inflection suffix.
        Substring matching made "start" match "art"; free prefix matching made "mutex" match "mute"."""
        if word == keyword:
            return True
        return word.startswith(keyword) and word[len(keyword):] in cls._INFLECTION_SUFFIXES

    @classmethod
    def filter_and_cluster_synonyms(cls, raw_query: str) -> Dict[str, Any]:
        """
        Classifies query tokens into coping buckets.
        - primary_coping_mechanism is None when no token matches (no silent default).
        - Ties break deterministically by enum declaration order, not by word order.
        - embedding_bandwidth_saved_pct is derived from the model constants and only
          credits tokens that actually matched a bucket.
        """
        words = [w.strip(cls._TOKEN_STRIP) for w in (raw_query or "").lower().split()]
        words = [w for w in words if w]
        detected_categories: Dict[str, int] = {}
        matched_tokens = 0

        for w in words:
            hit = False
            for mechanism, keywords in cls.COPING_SEMANTIC_BUCKETS.items():
                if any(cls._token_matches(w, k) for k in keywords):
                    detected_categories[mechanism.value] = detected_categories.get(mechanism.value, 0) + 1
                    hit = True
            if hit:
                matched_tokens += 1

        enum_order = [m.value for m in PsychologicalCopingMechanism]
        primary_mechanism: Optional[str] = (
            min(detected_categories, key=lambda k: (-detected_categories[k], enum_order.index(k)))
            if detected_categories else None
        )

        unpruned = len(words) * cls.MODEL_VOCAB_SIZE
        pruned = ((len(words) - matched_tokens) * cls.MODEL_VOCAB_SIZE
                  + matched_tokens * cls.MODEL_CLUSTER_SIZE)
        saved_pct = round(100.0 * (1.0 - pruned / unpruned), 1) if unpruned else 0.0

        return {
            "primary_coping_mechanism": primary_mechanism,
            "detected_semantic_clusters": detected_categories,
            "matched_tokens": matched_tokens,
            "total_tokens": len(words),
            "unpruned_dictionary_lookups": unpruned,   # model, not measured
            "pruned_clustered_lookups": pruned,        # model, not measured
            "embedding_bandwidth_saved_pct": saved_pct,
            "figure_status": "MODEL_UNVERIFIED",
            "vital_max_hp": VITAL_MAX_HP
        }


# ============================================================================
# 4. CORTEX & PROCESSOR-WHISPERER MCP BRIDGE
# ============================================================================

class CortexProcessorWhispererMcpBridge:
    """
    Bridge connecting the Model Context Protocol (MCP) to:
      1. Cortex Integrity Engine (micro-architectural coherence).
      2. Processor Whisperer (nanosecond RAM ring buffer hints).
      3. Psychological Coping Protocol (task triage & UI prioritization).
    """

    # Classification thresholds (inputs at or above trigger the branch).
    THERMAL_EMERGENCY_PCT: float = 85.0
    MEMORY_CONTENTION_LIMIT: float = 2.0
    CONTEXT_ENTROPY_LIMIT: float = 0.70
    DEFAULT_THERMAL_PCT: float = 65.0

    # Single source of truth: mechanism -> (rank, time_budget_us, spec_depth, mem_policy, synonym_cluster).
    # Ranks are a strict total order (1 = most urgent). Previously REGRESSION (thermal
    # emergency) and PROJECTION (nominal) both had rank 1, so emergencies could be
    # scheduled behind nominal work. The relative order of all other pairs is unchanged.
    TRIAGE_POLICY: Dict[PsychologicalCopingMechanism, Tuple[int, float, int, str, bool]] = {
        PsychologicalCopingMechanism.REGRESSION:           (1,  50.0, 0, "LEGACY_SIMD_BUFFER", True),
        PsychologicalCopingMechanism.PROJECTION:           (2, 500.0, 4, "HOST_COHERENT_PREFETCH_RING", False),
        PsychologicalCopingMechanism.COMPARTMENTALIZATION: (3, 120.0, 1, "ISOLATED_UMA_PLACED_HEAP", True),
        PsychologicalCopingMechanism.SUBLIMATION:          (4, 350.0, 2, "PROCEDURAL_ASCII_SLAB", True),
    }
    _TRIAGE_POLICY = TRIAGE_POLICY  # backward-compatible alias
    HISTORY_LIMIT: int = 256

    def __init__(self):
        self.lock = threading.Lock()
        self.unlocker = SiliconDieUnlocker()
        self.synonym_optimizer = SemanticSynonymCompilationOptimizer()
        self._execution_history: deque = deque(maxlen=self.HISTORY_LIMIT)
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP must remain 6"

    def evaluate_and_schedule_task(
        self,
        task_name: str,
        context_entropy: float = 0.25,
        memory_contention: float = 1.0,
        thermal_pressure_pct: float = DEFAULT_THERMAL_PCT
    ) -> CopingPriorityDescriptor:
        """
        Evaluates task urgency using the psychological coping protocol
        and returns the authoritative scheduling and speculative hints.
        """
        task_id = f"COPING-{uuid.uuid4().hex[:6].upper()}"

        mechanism = self.classify(context_entropy, memory_contention, thermal_pressure_pct)
        rank, time_budget, spec_depth, mem_policy, synonym_cluster = self.TRIAGE_POLICY[mechanism]

        desc = CopingPriorityDescriptor(
            task_id=task_id,
            task_name=task_name,
            coping_mechanism=mechanism,
            priority_rank=rank,
            allocated_time_budget_us=time_budget,
            speculative_depth=spec_depth,
            memory_allocation_policy=mem_policy,
            semantic_clustering_enabled=synonym_cluster,
            vital_max_hp=VITAL_MAX_HP
        )

        with self.lock:
            self._execution_history.append(desc)  # deque(maxlen) evicts the oldest in O(1)

        return desc

    @classmethod
    def classify(
        cls,
        context_entropy: float,
        memory_contention: float,
        thermal_pressure_pct: float
    ) -> PsychologicalCopingMechanism:
        """
        Pure decision function (no side effects). Precedence: thermal > memory > entropy > nominal.
        Fail-safe: any non-finite input (NaN/inf) means the system state is unknown and is
        treated as the worst case (REGRESSION). Previously NaN failed every comparison and fell
        through to PROJECTION with maximum speculative depth.
        """
        values = (
            _coerce_float(context_entropy, "context_entropy"),
            _coerce_float(memory_contention, "memory_contention"),
            _coerce_float(thermal_pressure_pct, "thermal_pressure_pct"),
        )
        if not all(math.isfinite(v) for v in values):
            return PsychologicalCopingMechanism.REGRESSION
        entropy, contention, thermal = values
        if thermal >= cls.THERMAL_EMERGENCY_PCT:
            return PsychologicalCopingMechanism.REGRESSION
        if contention >= cls.MEMORY_CONTENTION_LIMIT:
            return PsychologicalCopingMechanism.COMPARTMENTALIZATION
        if entropy >= cls.CONTEXT_ENTROPY_LIMIT:
            return PsychologicalCopingMechanism.SUBLIMATION
        return PsychologicalCopingMechanism.PROJECTION

    def schedule_ui_processes(self, processes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Determines the optimal execution order for UI and render processes
        according to coping mechanism rankings.
        """
        sorted_procs = []
        for p in processes:
            name = p.get("name") or "UI-Task"
            entropy = _coerce_float(p.get("entropy"), "entropy", 0.20)
            contention = _coerce_float(p.get("contention"), "contention", 1.0)
            # Thermal was previously ignored here, so a 99 °C process was triaged as nominal.
            raw_thermal = p.get("thermal_pressure_pct")
            if raw_thermal is None:
                raw_thermal = p.get("thermal")
            thermal = _coerce_float(raw_thermal, "thermal", self.DEFAULT_THERMAL_PCT)
            triage = self.evaluate_and_schedule_task(name, entropy, contention, thermal)
            p_out = dict(p)
            p_out["coping_triage"] = triage.to_dict()
            p_out["priority_rank"] = triage.priority_rank
            sorted_procs.append(p_out)

        # Sort by priority rank (1 is highest)
        sorted_procs.sort(key=lambda x: x["priority_rank"])
        return sorted_procs

    def whisper_speculative_instruction(self, pc: int, coping: PsychologicalCopingMechanism) -> Dict[str, Any]:
        """
        Injects nanosecond branch and ISA hints for the Processor Whisperer.
        """
        if coping == PsychologicalCopingMechanism.PROJECTION:
            return {
                "pc": pc,
                "isa_recommendation": "AVX-512 VNNI (CPU) / DP4A (Xe GPU) INT8",
                "prefetch_opcode": "0x0C (OP_PREFETCH_L1_STREAM)",
                "speculative_branch_taken": True,
                "latency_penalty_ns": 0.38,
                "vital_max_hp": VITAL_MAX_HP
            }
        elif coping == PsychologicalCopingMechanism.RATIONALIZATION:
            return {
                "pc": pc,
                "isa_recommendation": "INT8 Quantized Vector / Tile4 Swizzle",
                "prefetch_opcode": "0xA1 (K_SPEC_PREFETCH_UMA)",
                "speculative_branch_taken": True,
                "latency_penalty_ns": 0.50,
                "vital_max_hp": VITAL_MAX_HP
            }
        else:
            # Speculation must agree with the triage policy's speculative depth.
            # Previously SUBLIMATION got spec_depth=2 from triage but no speculation here.
            policy = self.TRIAGE_POLICY.get(coping)
            spec_depth = policy[2] if policy else 0
            return {
                "pc": pc,
                "isa_recommendation": "AVX2 SIMD Unrolled",
                "prefetch_opcode": "0x0D (OP_BARRIER_L3_SYNC)",
                "speculative_branch_taken": spec_depth > 0,
                "latency_penalty_ns": 1.20,
                "vital_max_hp": VITAL_MAX_HP
            }

    # ─── MCP TOOL SCHEMAS & EXPOSURE ─────────────────────────────────────────

    def get_mcp_tools_manifest(self) -> List[Dict[str, Any]]:
        """Returns MCP tool definitions. MCP `tools/list` uses `inputSchema`; `parameters`
        is kept as an alias for existing callers."""
        tools = [
            {
                "name": "cortex_triage_coping_priority",
                "description": "Evaluates task priority using psychological coping mechanisms and allocates nanosecond execution slices.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_name": {"type": "string"},
                        "context_entropy": {"type": "number", "description": "Shannon entropy [0.0, 1.0]"},
                        "memory_contention": {"type": "number", "description": "DRAM bus thrashing index"},
                        "thermal_pressure_pct": {"type": "number", "description": "Junction temp percentage of TjMax"}
                    },
                    "required": ["task_name"]
                }
            },
            {
                "name": "cortex_cluster_synonym_embeddings",
                "description": "Classifies query tokens into discrete coping categories and reports a modelled (unverified) lookup reduction.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_text": {"type": "string"}
                    },
                    "required": ["query_text"]
                }
            },
            {
                "name": "silicon_unlock_potential",
                "description": "Returns a MODELLED (unverified, no MSR access) Tiger Lake power/clock envelope for a requested PL1.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_pl1_watts": {"type": "number", "default": 32.0}
                    }
                }
            }
        ]
        for tool in tools:
            tool["inputSchema"] = tool["parameters"]
        return tools

    @staticmethod
    def _arg_float(arguments: Dict[str, Any], key: str, default: float) -> float:
        """JSON null / missing -> default; bool and out-of-range numbers raise (caught by dispatch)."""
        return _coerce_float(arguments.get(key), key, default)

    def dispatch_mcp_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Dispatches an MCP tool call. Invalid arguments return an error dict instead of raising."""
        arguments = arguments or {}
        try:
            if tool_name == "cortex_triage_coping_priority":
                desc = self.evaluate_and_schedule_task(
                    task_name=arguments.get("task_name") or "AnonymousTask",
                    context_entropy=self._arg_float(arguments, "context_entropy", 0.25),
                    memory_contention=self._arg_float(arguments, "memory_contention", 1.0),
                    thermal_pressure_pct=self._arg_float(arguments, "thermal_pressure_pct", self.DEFAULT_THERMAL_PCT)
                )
                return desc.to_dict()
            elif tool_name == "cortex_cluster_synonym_embeddings":
                return self.synonym_optimizer.filter_and_cluster_synonyms(
                    str(arguments.get("query_text") or "")
                )
            elif tool_name == "silicon_unlock_potential":
                pl1 = self._arg_float(arguments, "target_pl1_watts", 32.0)
                return self.unlocker.unlock_full_silicon_potential(pl1).to_dict()
        except (TypeError, ValueError) as exc:
            return {"error": f"Invalid arguments for {tool_name}: {exc}", "vital_max_hp": VITAL_MAX_HP}
        return {"error": f"Unknown MCP tool: {tool_name}", "vital_max_hp": VITAL_MAX_HP}


# Global singleton instance
GLOBAL_COPING_CORTEX_MCP_BRIDGE = CortexProcessorWhispererMcpBridge()


# ============================================================================
# 5. CLI ENTRYPOINT
# ============================================================================

def main():
    import argparse
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(
        description="Krystal-Stack Psychological Coping Cortex & MCP Bridge CLI"
    )
    parser.add_argument("--unlock-silicon", action="store_true", help="Audit and unlock full physical silicon die potential")
    parser.add_argument("--coping-triage", type=str, default="", help="Evaluate task name under coping mechanism protocol")
    parser.add_argument("--cluster-synonyms", type=str, default="", help="Cluster LLM synonyms through coping taxonomy")
    parser.add_argument("--mcp-manifest", action="store_true", help="Print MCP tool definitions")

    args = parser.parse_args()
    bridge = GLOBAL_COPING_CORTEX_MCP_BRIDGE

    if args.mcp_manifest:
        print(json.dumps(bridge.get_mcp_tools_manifest(), indent=2))
        return

    if args.unlock_silicon:
        print("================================================================================")
        print("  MODELLED SILICON ENVELOPE (UNVERIFIED - NO MSR READ OR WRITTEN)")
        print("================================================================================")
        rep = bridge.unlocker.unlock_full_silicon_potential(32.0)
        print(f"  Physical Architecture:  {rep.physical_die_architecture}")
        print(f"  Detected SKU:           {rep.marketing_sku_detected}")
        print(f"  Die Parity Confirmed:   {rep.physical_die_parity_confirmed}")
        print(f"  Parity Note:            {rep.die_parity_note}")
        print("-" * 80)
        print(f"  PL1 Power Envelope:     {rep.stock_pl1_watts} W -> {rep.unlocked_pl1_watts} W (target for MSR 0x610; not written)")
        print(f"  Ring Bus / Uncore:      {rep.stock_ring_bus_ghz} GHz -> {rep.unlocked_ring_bus_ghz} GHz (target for MSR 0x620; not written)")
        print(f"  All-Core AVX-512 Clock: {rep.sustained_all_core_ghz_stock} GHz -> {rep.sustained_all_core_ghz_unlocked} GHz (model)")
        print(f"  DRAM RAPL Domain:       Unknown (never probed; typically not exposed on client SKUs)")
        print(f"  Thermal Trip Headroom:  {rep.stock_thermal_trip_c}°C (OEM) -> {rep.unlocked_safe_trip_c}°C (model target; chassis-dependent)")
        print(f"  GPU Slice Undervolt:    {rep.gpu_slice_undervolt_mv} mV (model input; stability unverified)")
        print(f"  Arrhenius Aging Factor: {rep.silicon_wear_af} (model input)")
        print(f"  Clock-Ratio Upper Bound:+{rep.overall_die_efficiency_boost_pct}% (compute-bound, constant IPC; {rep.figure_status})")
        print(f"  MSR Writes Performed:   {rep.msr_writes_performed}")
        print(f"  Invariant:              VITAL_MAX_HP = {rep.vital_max_hp}")
        print("================================================================================")
        return

    if args.cluster_synonyms:
        res = bridge.synonym_optimizer.filter_and_cluster_synonyms(args.cluster_synonyms)
        print(json.dumps(res, indent=2))
        return

    # Default triage demonstration
    task = args.coping_triage or "Procedural ASCII Rendering Pipeline"
    desc = bridge.evaluate_and_schedule_task(task, context_entropy=0.75, memory_contention=1.2)
    print("================================================================================")
    print("  PSYCHOLOGICAL COPING PROTOCOL: SYSTEMIC DATA & TASK TRIAGE")
    print("================================================================================")
    print(f"  Task ID:                {desc.task_id} ({desc.task_name})")
    print(f"  Coping Mechanism:       {desc.coping_mechanism.value}")
    print(f"  Priority Rank:          {desc.priority_rank} (1 is Highest / Realtime)")
    print(f"  Time Budget:            {desc.allocated_time_budget_us} us")
    print(f"  Speculative Depth:      {desc.speculative_depth} words lookahead")
    print(f"  Memory Policy:          {desc.memory_allocation_policy}")
    print(f"  Synonym Pruning:        {'Active (modelled saving, unverified)' if desc.semantic_clustering_enabled else 'Direct'}")
    print(f"  Invariant:              VITAL_MAX_HP = {desc.vital_max_hp}")
    print("================================================================================")


if __name__ == "__main__":
    main()
