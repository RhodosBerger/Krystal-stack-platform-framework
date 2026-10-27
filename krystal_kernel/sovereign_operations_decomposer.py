#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL STACK: SOVEREIGN OPERATIONS AUTOCOMPLETER & TASK DECOMPOSITION GENERATOR
==============================================================================
Module: krystal_kernel/sovereign_operations_decomposer.py
Author: Antigravity Cognitive Invariants & Operations Group (2026)

Constitutional Invariants:
    - VITAL_MAX_HP == 6 (Delta S == 0 on violation)
    - Invariant != Policy
    - Proposer != Evaluator (Authority Separation)
    - Zero-Touch Interposition (P1) & Universal 128B Frame (P2)
    - Two-Phase Actuation: If not Valid(Action) -> S_{t+1} == S_t

Purpose:
    1. Operations Autocompleter:
       Real-time fuzzy, prefix, semantic, and context-aware suggestion engine
       for sovereign management operations, diagnostic routines, hardware actions,
       and scientific evidence pipelines.
    2. Management Task Decomposition Generator:
       Automatically breaks down complex, monolithic management goals into an
       atomic, dependency-ordered, verifiable Directed Acyclic Graph (DAG) of
       sub-tasks across the 6 formal engineering phases:
         [1] SPECIFICATION
         [2] AUTHORITY_CHECK
         [3] INVARIANT_PRESERVATION
         [4] EXECUTION_MINIMAL
         [5] PHYSICAL_TELEMETRY
         [6] RECONCILIATION_COMMIT
    3. Production-ready CLI and programmatic API with HMAC-SHA256 integrity.
==============================================================================
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import hmac
import hashlib
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any, Set

from krystal_kernel.constants import (
    VITAL_MAX_HP,
    ContractViolation,
    CapabilityScope,
    UMA_SATURATION_THRESHOLD,
    ARRHENIUS_MAX_TEMP_C,
)


# ── 1. ENUMS AND DATA CONTRACTS ──────────────────────────────────────────────

class OperationDomain(str, Enum):
    """Domains of operational governance within the Krystal Stack."""
    SYSTEM_GOVERNANCE    = "SYSTEM_GOVERNANCE"     # Global invariants, shadow cycles, UMA bus
    HARDWARE_CONTROL     = "HARDWARE_CONTROL"      # FPGA coprocessor, action space A0-A6, microcode
    SCIENTIFIC_EVIDENCE  = "SCIENTIFIC_EVIDENCE"   # M4.1 benchmarks, VHL_HC estimators, ablation probes
    KERNEL_COMPILATION   = "KERNEL_COMPILATION"    # KSL modules, C11 ABI, 128B zero-copy frames
    TELEMETRY_DIAGNOSTIC = "TELEMETRY_DIAGNOSTIC"  # RAPL MSRs, INA226 shunts, thermal gradients


class SubtaskPhase(str, Enum):
    """The 6 formal engineering phases of an atomic subtask."""
    SPECIFICATION          = "SPECIFICATION"           # Phase 1: Define bounds, inputs, targets
    AUTHORITY_CHECK        = "AUTHORITY_CHECK"         # Phase 2: Verify capability, Proposer != Evaluator
    INVARIANT_PRESERVATION = "INVARIANT_PRESERVATION"  # Phase 3: Assert VITAL_MAX_HP == 6 & zero-touch
    EXECUTION_MINIMAL      = "EXECUTION_MINIMAL"       # Phase 4: Surgical isolated intervention
    PHYSICAL_TELEMETRY     = "PHYSICAL_TELEMETRY"      # Phase 5: Direct measurement & sensor audit
    RECONCILIATION_COMMIT  = "RECONCILIATION_COMMIT"   # Phase 6: Commit state or rollback (Delta S == 0)


class SubsystemPlacement(str, Enum):
    """Physical or logical target for execution placement."""
    HOST_CPU          = "HOST_CPU"
    FPGA_MAILBOX      = "FPGA_MAILBOX"
    DMA_SHM_RING      = "DMA_SHM_RING"
    SOVEREIGN_KERNEL  = "SOVEREIGN_KERNEL"
    TELEMETRY_INA226  = "TELEMETRY_INA226"
    HOST_RAPL_MSR     = "HOST_RAPL_MSR"
    CI_REGRESSION     = "CI_REGRESSION"


@dataclass
class OperationSuggestion:
    """Individual autocompletion proposal returned by the engine."""
    command: str
    display_title: str
    completion_suffix: str
    domain: OperationDomain
    score: float
    synopsis: str
    default_parameters: Dict[str, Any]
    estimated_latency_us: float
    authority_required: CapabilityScope
    invariant_gate: str
    vital_max_hp: int = VITAL_MAX_HP


@dataclass
class SubtaskNode:
    """An individual atomic subtask within the decomposed management DAG."""
    subtask_id: str
    title: str
    description: str
    phase: SubtaskPhase
    subsystem_placement: SubsystemPlacement
    depends_on: List[str]
    latency_budget_us: float
    energy_budget_j: float
    verification_oracle: str
    rollback_action: str
    code_snippet: str
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SubtaskDAG:
    """Topologically ordered, verifiable plan decomposing a management goal."""
    plan_id: str
    management_goal: str
    domain: OperationDomain
    subtasks: List[SubtaskNode]
    critical_path_latency_us: float
    total_energy_budget_j: float
    is_acyclic: bool
    creation_timestamp_ns: int = field(default_factory=time.time_ns)
    hmac_sha256: str = ""
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "management_goal": self.management_goal,
            "domain": self.domain.value,
            "subtasks": [t.to_dict() for t in self.subtasks],
            "critical_path_latency_us": self.critical_path_latency_us,
            "total_energy_budget_j": self.total_energy_budget_j,
            "is_acyclic": self.is_acyclic,
            "creation_timestamp_ns": self.creation_timestamp_ns,
            "hmac_sha256": self.hmac_sha256,
            "vital_max_hp": self.vital_max_hp,
        }

    def generate_mermaid_diagram(self) -> str:
        """Generates a GitHub-flavored Mermaid flowchart of the decomposed DAG."""
        lines = ["```mermaid", "graph TD"]
        for t in self.subtasks:
            sanitized_title = t.title.replace('"', "'")
            lines.append(f'    {t.subtask_id}["{t.subtask_id}: {sanitized_title}<br/>({t.phase.value})"]')
        for t in self.subtasks:
            for dep in t.depends_on:
                lines.append(f"    {dep} --> {t.subtask_id}")
        lines.append("```")
        return "\n".join(lines)


# ── 2. CANONICAL OPERATIONS CATALOG ──────────────────────────────────────────

CANONICAL_OPERATIONS_CATALOG: List[Dict[str, Any]] = [
    # SYSTEM_GOVERNANCE
    {
        "command": "audit_system_state",
        "title": "Audit Complete 8-Tuple System State (S,A,T,I,O,E,H,W)",
        "domain": OperationDomain.SYSTEM_GOVERNANCE,
        "keywords": ["audit", "system", "state", "tuple", "invariant", "hp", "verify"],
        "synopsis": "Formally inspects current state S against invariant set I (VITAL_MAX_HP == 6).",
        "default_parameters": {"check_registers": True, "strict_bounds": True},
        "latency_us": 12.0,
        "authority": CapabilityScope.CAN_OBSERVE | CapabilityScope.CAN_VERIFY,
        "invariant": "VITAL_MAX_HP == 6",
    },
    {
        "command": "reconcile_shadow_cycle",
        "title": "Reconcile Chronometric Shadow Multiplexing & UMA Bus Pressure",
        "domain": OperationDomain.SYSTEM_GOVERNANCE,
        "keywords": ["shadow", "cycle", "reconcile", "uma", "bus", "vblank", "timing"],
        "synopsis": "Ensures epistemic reasoning executes strictly in scanout shadows with mu_UMA < 0.78.",
        "default_parameters": {"max_shadow_budget_ms": 3.5, "uma_threshold": UMA_SATURATION_THRESHOLD},
        "latency_us": 20.0,
        "authority": CapabilityScope.CAN_AUTHORIZE | CapabilityScope.CAN_EXECUTE,
        "invariant": "ScanoutPhase * ComputePhase == 0 and mu_UMA < 0.78",
    },
    {
        "command": "enforce_thermal_budget",
        "title": "Enforce Thermal Headroom & Arrhenius Temperature Ceiling",
        "domain": OperationDomain.SYSTEM_GOVERNANCE,
        "keywords": ["thermal", "budget", "temperature", "arrhenius", "cooling", "headroom"],
        "synopsis": "Assesses package junction temperature and engages hysteresis dwell if T > 75C.",
        "default_parameters": {"target_temp_c": ARRHENIUS_MAX_TEMP_C, "hysteresis_dwell_frames": 12},
        "latency_us": 15.0,
        "authority": CapabilityScope.CAN_EXECUTE,
        "invariant": "T_junction <= 75.0 C",
    },

    # HARDWARE_CONTROL
    {
        "command": "dispatch_microcode_batch",
        "title": "Dispatch KCB-M4 Microcode Pipeline Batch via PCIe Mailbox",
        "domain": OperationDomain.HARDWARE_CONTROL,
        "keywords": ["dispatch", "microcode", "fpga", "kcb", "pipeline", "pcie", "mailbox"],
        "synopsis": "Transmits 128B contract frames to the FPGA coprocessor (208 ns core turnaround).",
        "default_parameters": {"channel": 0, "batch_size": 32, "timeout_us": 5.0},
        "latency_us": 2.85,
        "authority": CapabilityScope.CAN_EXECUTE,
        "invariant": "Zero-Copy 128B naturally aligned frame layout",
    },
    {
        "command": "tune_action_space_a0_a6",
        "title": "Tune Adaptive Action Space (A0..A6 Batch/Worker/Hint Parameters)",
        "domain": OperationDomain.HARDWARE_CONTROL,
        "keywords": ["tune", "action", "space", "batch", "worker", "hint", "governor", "a0", "a6"],
        "synopsis": "Applies voltage-isolated performance hints and batch dimension adaptations.",
        "default_parameters": {"action_id": "A2_BATCH_UP", "allow_voltage_writes": False},
        "latency_us": 1.20,
        "authority": CapabilityScope.CAN_PROPOSE | CapabilityScope.CAN_AUTHORIZE,
        "invariant": "Voltage writes forbidden (pure hint-based pacing)",
    },
    {
        "command": "ablate_pcie_latency",
        "title": "Run PCIe DMA Serialization & Direct-Bypass Ablation Probe",
        "domain": OperationDomain.HARDWARE_CONTROL,
        "keywords": ["ablate", "pcie", "latency", "dma", "bypass", "probe", "ablation"],
        "synopsis": "Executes controlled probe discriminating CE_pcie (9.5%) against zero-latency on-die baseline.",
        "default_parameters": {"probe_cycles": 100, "compare_with_shm": True},
        "latency_us": 45.0,
        "authority": CapabilityScope.CAN_VERIFY,
        "invariant": "Deterministic delta verification (Delta S == 0)",
    },

    # SCIENTIFIC_EVIDENCE
    {
        "command": "run_m4_1_rbd_benchmark",
        "title": "Execute KSAC M4.1 Randomized Block Design Benchmark (N=60 Trials)",
        "domain": OperationDomain.SCIENTIFIC_EVIDENCE,
        "keywords": ["benchmark", "m4_1", "rbd", "randomized", "blocks", "vhl", "trials"],
        "synopsis": "Runs 30 randomized blocks (B1 vs K paired) to compute VHL_HC without thermal bias.",
        "default_parameters": {"blocks": 30, "seed": 42, "duty_cycle": 0.10},
        "latency_us": 10000000.0,  # 10s per trial in real campaign
        "authority": CapabilityScope.CAN_AUTHORIZE | CapabilityScope.CAN_EXECUTE,
        "invariant": "InvariantViolationRate == 0 and InvalidCommitRate == 0",
    },
    {
        "command": "compute_vhl_hc_estimator",
        "title": "Compute Dual VHL Estimators (Pairwise Ratio vs Ratio of Means)",
        "domain": OperationDomain.SCIENTIFIC_EVIDENCE,
        "keywords": ["vhl", "estimator", "pairwise", "ratio", "means", "ci95", "student"],
        "synopsis": "Derives VHL_HC = 1.1516, 95% CI [1.1408, 1.1624], and verifies estimator delta <= 0.0003.",
        "default_parameters": {"confidence_pct": 95.0, "df": 29},
        "latency_us": 5.0,
        "authority": CapabilityScope.CAN_VERIFY,
        "invariant": "CI_95_lower > 1.10 for STRONG_PASS",
    },
    {
        "command": "generate_evidence_package",
        "title": "Export 17-Artifact M4_1_EVIDENCE Package & Sign HMAC-SHA256",
        "domain": OperationDomain.SCIENTIFIC_EVIDENCE,
        "keywords": ["evidence", "package", "export", "hmac", "sha256", "archive", "receipt"],
        "synopsis": "Generates complete auditable bundle with raw runs, telemetry, bitstream and git hashes.",
        "default_parameters": {"target_dir": "experiments/ksac_m4_1/M4_1_EVIDENCE"},
        "latency_us": 15000.0,
        "authority": CapabilityScope.CAN_AUTHORIZE,
        "invariant": "Full-Frame cryptographic HMAC-SHA256 integrity",
    },

    # KERNEL_COMPILATION
    {
        "command": "compile_ksl_module",
        "title": "Compile KSL (Krystal Systems Language) Module to C11 & KµASM",
        "domain": OperationDomain.KERNEL_COMPILATION,
        "keywords": ["compile", "ksl", "module", "c11", "kmuasm", "parser", "ir", "codegen"],
        "synopsis": "Transforms .ksl source code through 5-layer KIR to 128B aligned C11 and microcode.",
        "default_parameters": {"target_arch": "x86_64_avx512", "optimize_level": 3},
        "latency_us": 850.0,
        "authority": CapabilityScope.CAN_PROPOSE | CapabilityScope.CAN_EXECUTE,
        "invariant": "_Alignas(64) zero-copy memory contracts",
    },
    {
        "command": "bind_shm_ring_buffer",
        "title": "Bind Lock-Free SPSC Shared Memory Ring Buffer (/dev/shm)",
        "domain": OperationDomain.KERNEL_COMPILATION,
        "keywords": ["shm", "ring", "buffer", "spsc", "lockfree", "memory", "ipc"],
        "synopsis": "Allocates naturally aligned 128B slots for sub-microsecond zero-copy host-kernel IPC.",
        "default_parameters": {"ring_capacity": 1024, "slot_size_bytes": 128},
        "latency_us": 25.0,
        "authority": CapabilityScope.CAN_AUTHORIZE | CapabilityScope.CAN_EXECUTE,
        "invariant": "Universal 128B Lingua Franca (P2)",
    },

    # TELEMETRY_DIAGNOSTIC
    {
        "command": "sample_rapl_msr",
        "title": "Sample Intel RAPL Energy Registers (MSR_PKG / MSR_DRAM 0x611/0x618)",
        "domain": OperationDomain.TELEMETRY_DIAGNOSTIC,
        "keywords": ["rapl", "msr", "energy", "sample", "power", "pkg", "dram"],
        "synopsis": "Reads continuous host power draw at 1,000 Hz with 32-bit counter rollover detection.",
        "default_parameters": {"sample_interval_ms": 1, "track_rollover": True},
        "latency_us": 3.5,
        "authority": CapabilityScope.CAN_OBSERVE,
        "invariant": "Calibrated hardware counters (zero-touch probe)",
    },
    {
        "command": "poll_ina226_shunts",
        "title": "Poll TI INA226 Bidirectional Current Monitors (Carrier 12V/0.85V/1.8V)",
        "domain": OperationDomain.TELEMETRY_DIAGNOSTIC,
        "keywords": ["ina226", "shunt", "pmbus", "fpga", "voltage", "current", "telemetry"],
        "synopsis": "Measures physical FPGA power draw across 3 rails at 10 kHz with +-0.1% precision.",
        "default_parameters": {"sampling_rate_hz": 10000, "r_shunt_mohm": 10.0},
        "latency_us": 8.0,
        "authority": CapabilityScope.CAN_OBSERVE,
        "invariant": "Continuous gross integration: E_FPGA = integral P_FPGA(t) dt",
    },
]


# ── 3. SOVEREIGN OPERATIONS AUTOCOMPLETER ────────────────────────────────────

class SovereignOperationsAutocompleter:
    """
    Real-time fuzzy, prefix, and semantic autocompleter for sovereign operations.
    """

    def __init__(self, catalog: Optional[List[Dict[str, Any]]] = None) -> None:
        self.catalog = catalog or CANONICAL_OPERATIONS_CATALOG

    def autocomplete(
        self,
        query: str,
        current_state: Optional[Dict[str, Any]] = None,
        max_results: int = 5,
    ) -> List[OperationSuggestion]:
        """
        Suggests matching operations for a user-supplied prefix or natural query.
        Ranks results based on exact prefix, keyword match, and operational state relevance.
        """
        q = query.strip().lower()
        if not q:
            # Return top-ranked standard operations
            return [self._to_suggestion(entry, 1.0, "") for entry in self.catalog[:max_results]]

        state = current_state or {}
        thermal_load_high = state.get("temperature_c", 25.0) > 65.0
        bus_pressure_high = state.get("uma_pressure", 0.50) >= UMA_SATURATION_THRESHOLD

        candidates: List[Tuple[float, Dict[str, Any], str]] = []

        for entry in self.catalog:
            cmd = entry["command"].lower()
            title = entry["title"].lower()
            synopsis = entry["synopsis"].lower()
            keywords = [k.lower() for k in entry.get("keywords", [])]

            score = 0.0
            suffix = ""

            # 1. Exact command prefix (Highest priority)
            if cmd.startswith(q):
                score += 0.85 + (len(q) / len(cmd)) * 0.15
                suffix = cmd[len(q):]
            # 2. Command substring
            elif q in cmd:
                score += 0.65
            # 3. Keyword matches
            matched_keywords = sum(1 for kw in keywords if kw.startswith(q) or q in kw)
            if matched_keywords > 0:
                score += 0.25 * matched_keywords
            # 4. Title & synopsis search
            if q in title:
                score += 0.30
            if q in synopsis:
                score += 0.15

            # Context-sensitive boost
            if thermal_load_high and entry["domain"] == OperationDomain.SYSTEM_GOVERNANCE and "thermal" in cmd:
                score += 0.40
            if bus_pressure_high and "shadow" in cmd:
                score += 0.40

            if score > 0.10:
                normalized_score = min(1.0, round(score, 4))
                candidates.append((normalized_score, entry, suffix))

        # Sort descending by score
        candidates.sort(key=lambda x: x[0], reverse=True)

        results: List[OperationSuggestion] = []
        for score, entry, suffix in candidates[:max_results]:
            results.append(self._to_suggestion(entry, score, suffix))

        return results

    def _to_suggestion(self, entry: Dict[str, Any], score: float, suffix: str) -> OperationSuggestion:
        return OperationSuggestion(
            command=entry["command"],
            display_title=entry["title"],
            completion_suffix=suffix,
            domain=entry["domain"],
            score=score,
            synopsis=entry["synopsis"],
            default_parameters=entry.get("default_parameters", {}),
            estimated_latency_us=entry.get("latency_us", 10.0),
            authority_required=entry.get("authority", CapabilityScope.CAN_OBSERVE),
            invariant_gate=entry.get("invariant", "VITAL_MAX_HP == 6"),
            vital_max_hp=VITAL_MAX_HP,
        )


# ── 4. TASK DECOMPOSITION GENERATOR ──────────────────────────────────────────

class SovereignOperationsDecomposer:
    """
    Decomposes high-level management objectives and operations into an atomic,
    dependency-ordered, verifiable Directed Acyclic Graph (DAG) of subtasks.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        self.vital_max_hp = vital_max_hp
        if self.vital_max_hp != 6:
            raise ContractViolation(f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {self.vital_max_hp}")

    def decompose(
        self,
        management_goal: str,
        domain: Optional[OperationDomain] = None,
        custom_parameters: Optional[Dict[str, Any]] = None,
    ) -> SubtaskDAG:
        """
        Decomposes a management intent into an acyclic 6-phase subtask DAG.
        """
        goal_lower = management_goal.lower().strip()
        params = custom_parameters or {}

        # Detect domain if not provided
        detected_domain = domain or self._detect_domain(goal_lower)

        # Generate subtask sequence across the 6 formal engineering phases
        subtasks = self._generate_phase_subtasks(goal_lower, detected_domain, params)

        # Verify acyclic property & topological sorting
        is_acyclic = self._verify_dag_acyclicity(subtasks)
        if not is_acyclic:
            raise ContractViolation("Decomposed subtask dependency graph contains circular cycles.")

        # Compute critical path latency and total energy budget
        crit_latency = self._compute_critical_path_latency(subtasks)
        total_energy = round(sum(t.energy_budget_j for t in subtasks), 4)

        # Unique Plan ID
        plan_hash = hashlib.sha256(f"{management_goal}|{time.time_ns()}".encode("utf-8")).hexdigest()[:8]
        plan_id = f"PLAN-{detected_domain.value[:4]}-{plan_hash.upper()}"

        # Sign cryptographic HMAC-SHA256 receipt
        payload_to_sign = f"{plan_id}|{detected_domain.value}|{len(subtasks)}|{crit_latency}|{total_energy}|VITAL_{self.vital_max_hp}"
        sig = hmac.new(
            key=b"KRYSTAL-SOVEREIGN-OPERATIONS-PLAN-VITAL-6",
            msg=payload_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()

        return SubtaskDAG(
            plan_id=plan_id,
            management_goal=management_goal,
            domain=detected_domain,
            subtasks=subtasks,
            critical_path_latency_us=crit_latency,
            total_energy_budget_j=total_energy,
            is_acyclic=is_acyclic,
            hmac_sha256=sig,
            vital_max_hp=self.vital_max_hp,
        )

    def _detect_domain(self, text: str) -> OperationDomain:
        """Infers the most appropriate operation domain from natural intent text."""
        if any(w in text for w in ["benchmark", "m4_1", "vhl", "evidence", "ablation", "falsif"]):
            return OperationDomain.SCIENTIFIC_EVIDENCE
        if any(w in text for w in ["fpga", "microcode", "pcie", "mailbox", "kcb", "action", "hint"]):
            return OperationDomain.HARDWARE_CONTROL
        if any(w in text for w in ["compile", "ksl", "abi", "c11", "ring", "shm", "zero-copy"]):
            return OperationDomain.KERNEL_COMPILATION
        if any(w in text for w in ["rapl", "msr", "ina226", "telemetry", "sample", "monitor", "sensor"]):
            return OperationDomain.TELEMETRY_DIAGNOSTIC
        return OperationDomain.SYSTEM_GOVERNANCE

    def _generate_phase_subtasks(
        self,
        goal: str,
        domain: OperationDomain,
        params: Dict[str, Any],
    ) -> List[SubtaskNode]:
        """Synthesizes the concrete subtask DAG nodes according to domain semantics."""
        subtasks: List[SubtaskNode] = []

        if domain == OperationDomain.SCIENTIFIC_EVIDENCE:
            # 1. SPECIFICATION
            subtasks.append(SubtaskNode(
                subtask_id="ST-01-SPEC",
                title="Freeze Experimental Hypothesis and Randomization Seed",
                description="Lock H1: VHL_F > 1.0 vs H0: VHL_F <= 1.0 with 30 blocks and seed=42.",
                phase=SubtaskPhase.SPECIFICATION,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=[],
                latency_budget_us=10.0,
                energy_budget_j=0.0001,
                verification_oracle="protocol_frozen == True and seed == 42",
                rollback_action="reset_protocol_locks()",
                code_snippet="protocol.lock(version='2.0-FROZEN-TWO-GATE', seed=42)",
            ))
            # 2. AUTHORITY_CHECK
            subtasks.append(SubtaskNode(
                subtask_id="ST-02-AUTH",
                title="Verify Proposer != Evaluator Authority Separation",
                description="Confirm evaluation harness runs independently of host adaptive controller.",
                phase=SubtaskPhase.AUTHORITY_CHECK,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=["ST-01-SPEC"],
                latency_budget_us=5.0,
                energy_budget_j=0.0001,
                verification_oracle="proposer_id != evaluator_id and capability_checked == True",
                rollback_action="quarantine_unauthorized_proposer()",
                code_snippet="assert capability.has_flag(CapabilityScope.CAN_VERIFY)",
            ))
            # 3. INVARIANT_PRESERVATION
            subtasks.append(SubtaskNode(
                subtask_id="ST-03-INVAR",
                title="Assert Constitutional Invariants & Boundary Guards",
                description="Enforce VITAL_MAX_HP == 6, zero dropped requests, SLA latency ceiling < 20 ms.",
                phase=SubtaskPhase.INVARIANT_PRESERVATION,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=["ST-02-AUTH"],
                latency_budget_us=8.0,
                energy_budget_j=0.0001,
                verification_oracle="vital_max_hp == 6 and sla_threshold_ms == 20.0",
                rollback_action="abort_and_preserve_state()",
                code_snippet="assert VITAL_MAX_HP == 6, 'Invariant violation'",
            ))
            # 4. EXECUTION_MINIMAL
            subtasks.append(SubtaskNode(
                subtask_id="ST-04-EXEC",
                title="Execute Randomized Block Design Campaign (30 Paired Blocks)",
                description="Run 60 trials in alternating (B1->K or K->B1) order to cancel thermal drift.",
                phase=SubtaskPhase.EXECUTION_MINIMAL,
                subsystem_placement=SubsystemPlacement.CI_REGRESSION,
                depends_on=["ST-03-INVAR"],
                latency_budget_us=2500.0,
                energy_budget_j=12.5000,
                verification_oracle="len(completed_blocks) == 30 and invalid_commits == 0",
                rollback_action="purge_corrupted_runs()",
                code_snippet="runner.execute_validation_campaign(trial_count_per_mode=30, seed=42)",
            ))
            # 5. PHYSICAL_TELEMETRY
            subtasks.append(SubtaskNode(
                subtask_id="ST-05-TELEM",
                title="Acquire INA226 PMBus Power and RAPL MSR Energy Data",
                description="Compute both active control window (4.22 J) and gross board integral (23.1 J).",
                phase=SubtaskPhase.PHYSICAL_TELEMETRY,
                subsystem_placement=SubsystemPlacement.TELEMETRY_INA226,
                depends_on=["ST-04-EXEC"],
                latency_budget_us=45.0,
                energy_budget_j=0.0020,
                verification_oracle="e_control_j > 0.0 and e_gross_j >= e_control_j",
                rollback_action="invalidate_telemetry_receipt()",
                code_snippet="telemetry = poll_ina226_rail_integrals(duration_s=10.0)",
            ))
            # 6. RECONCILIATION_COMMIT
            subtasks.append(SubtaskNode(
                subtask_id="ST-06-COMMIT",
                title="Compute VHL_HC (1.1516), Sign HMAC-SHA256, and Export Package",
                description="Evaluate CI 95% [1.1408, 1.1624] and commit Gate M4.1-A complete / M4.1-B pending.",
                phase=SubtaskPhase.RECONCILIATION_COMMIT,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=["ST-05-TELEM"],
                latency_budget_us=120.0,
                energy_budget_j=0.0050,
                verification_oracle="ci_lower > 1.10 and hmac_valid == True",
                rollback_action="rollback_state_delta_zero()",
                code_snippet="runner.export_evidence_package(evidence_dir, agg, trials, seed=42)",
            ))

        elif domain == OperationDomain.HARDWARE_CONTROL:
            # 1. SPECIFICATION
            subtasks.append(SubtaskNode(
                subtask_id="ST-01-SPEC",
                title="Define Action Space Hints (A0..A6) and Latency Budget",
                description="Specify microcode ingress turnaround <= 208 ns and PCIe mailbox loop <= 2.85 us.",
                phase=SubtaskPhase.SPECIFICATION,
                subsystem_placement=SubsystemPlacement.FPGA_MAILBOX,
                depends_on=[],
                latency_budget_us=5.0,
                energy_budget_j=0.0001,
                verification_oracle="t_core_ns <= 208.0 and t_sys_us <= 2.85",
                rollback_action="reset_action_hint_descriptors()",
                code_snippet="action_env = ActionSpace(A0_HOLD, A1_BATCH_DOWN, A2_BATCH_UP)",
            ))
            # 2. AUTHORITY_CHECK
            subtasks.append(SubtaskNode(
                subtask_id="ST-02-AUTH",
                title="Check Capability Contract for Hardware Mailbox Access",
                description="Ensure calling context holds CAN_EXECUTE and voltage writes are disabled.",
                phase=SubtaskPhase.AUTHORITY_CHECK,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=["ST-01-SPEC"],
                latency_budget_us=4.0,
                energy_budget_j=0.0001,
                verification_oracle="voltage_writes_enabled == False and authorized == True",
                rollback_action="revoke_mailbox_token()",
                code_snippet="assert not config.allow_voltage_writes",
            ))
            # 3. INVARIANT_PRESERVATION
            subtasks.append(SubtaskNode(
                subtask_id="ST-03-INVAR",
                title="Verify 128B Natural Alignment & HMAC Ring Buffer Slot",
                description="Guarantee _Alignas(64) zero-copy frame layout before bus dispatch.",
                phase=SubtaskPhase.INVARIANT_PRESERVATION,
                subsystem_placement=SubsystemPlacement.DMA_SHM_RING,
                depends_on=["ST-02-AUTH"],
                latency_budget_us=6.0,
                energy_budget_j=0.0001,
                verification_oracle="sizeof(frame) == 128 and alignment == 64",
                rollback_action="zero_memory_frame_buffer()",
                code_snippet="assert sizeof(ZeroCopyContractFrame) == 128",
            ))
            # 4. EXECUTION_MINIMAL
            subtasks.append(SubtaskNode(
                subtask_id="ST-04-EXEC",
                title="Transmit Microcode Command Frame to FPGA DMA Mailbox",
                description="Write 128B frame over PCIe Gen3 x4 channel 0 without blocking host threads.",
                phase=SubtaskPhase.EXECUTION_MINIMAL,
                subsystem_placement=SubsystemPlacement.FPGA_MAILBOX,
                depends_on=["ST-03-INVAR"],
                latency_budget_us=2.85,
                energy_budget_j=0.0008,
                verification_oracle="dma_transfer_ack == True and parity_ok == True",
                rollback_action="dma_abort_and_clear_ring()",
                code_snippet="pcie_mailbox.send_frame(frame_ptr, channel=0)",
            ))
            # 5. PHYSICAL_TELEMETRY
            subtasks.append(SubtaskNode(
                subtask_id="ST-05-TELEM",
                title="Audit Ingress-to-Decision Latch Latency via Hardware Timer",
                description="Sample physical microsecond stamp to confirm t_core <= 208 ns.",
                phase=SubtaskPhase.PHYSICAL_TELEMETRY,
                subsystem_placement=SubsystemPlacement.FPGA_MAILBOX,
                depends_on=["ST-04-EXEC"],
                latency_budget_us=8.0,
                energy_budget_j=0.0002,
                verification_oracle="measured_t_core_ns <= 220.0",
                rollback_action="flag_pipeline_timing_hazard()",
                code_snippet="measured_ns = fpga_timer.read_ingress_delta()",
            ))
            # 6. RECONCILIATION_COMMIT
            subtasks.append(SubtaskNode(
                subtask_id="ST-06-COMMIT",
                title="Commit Pacing Action Hint to Host Runtime Scheduler",
                description="Apply batch pacing update; if timing or parity violated, enforce Delta S == 0.",
                phase=SubtaskPhase.RECONCILIATION_COMMIT,
                subsystem_placement=SubsystemPlacement.HOST_CPU,
                depends_on=["ST-05-TELEM"],
                latency_budget_us=12.0,
                energy_budget_j=0.0005,
                verification_oracle="scheduler_hint_acknowledged == True",
                rollback_action="restore_static_baseline_b1()",
                code_snippet="host_scheduler.apply_performance_hint(hint)",
            ))

        elif domain == OperationDomain.KERNEL_COMPILATION:
            # 1. SPECIFICATION
            subtasks.append(SubtaskNode(
                subtask_id="ST-01-SPEC",
                title="Parse KSL Source Code & Extract AST Representation",
                description="Lex and parse .ksl file into AST nodes with contracts, state, and constraints.",
                phase=SubtaskPhase.SPECIFICATION,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=[],
                latency_budget_us=150.0,
                energy_budget_j=0.0010,
                verification_oracle="ast_root is not None and syntax_errors == 0",
                rollback_action="clear_syntax_tree()",
                code_snippet="ast = ksl_parser.parse(source_code)",
            ))
            # 2. AUTHORITY_CHECK
            subtasks.append(SubtaskNode(
                subtask_id="ST-02-AUTH",
                title="Verify Type Soundness & Capability Contract Permissions",
                description="Run semantic type checker verifying state variables and placement constraints.",
                phase=SubtaskPhase.AUTHORITY_CHECK,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=["ST-01-SPEC"],
                latency_budget_us=120.0,
                energy_budget_j=0.0008,
                verification_oracle="type_checker.error_count == 0",
                rollback_action="abort_compilation_with_diagnostic()",
                code_snippet="ksl_type_checker.check(ast)",
            ))
            # 3. INVARIANT_PRESERVATION
            subtasks.append(SubtaskNode(
                subtask_id="ST-03-INVAR",
                title="Assert Zero-Copy 128B Memory Layout & Ring Buffers",
                description="Verify all compiled structs are strictly aligned to 64 bytes and sized to 128B.",
                phase=SubtaskPhase.INVARIANT_PRESERVATION,
                subsystem_placement=SubsystemPlacement.DMA_SHM_RING,
                depends_on=["ST-02-AUTH"],
                latency_budget_us=80.0,
                energy_budget_j=0.0005,
                verification_oracle="alignment_verified == True and no_heap_allocation == True",
                rollback_action="invalidate_compiled_abi()",
                code_snippet="assert kir.verify_zero_copy_invariants(ast)",
            ))
            # 4. EXECUTION_MINIMAL
            subtasks.append(SubtaskNode(
                subtask_id="ST-04-EXEC",
                title="Emit C11 Dual-Plane Source & KµASM Microcode Sidecar",
                description="Generate C11 header/implementation pair and binary microcode for KSAC-1.",
                phase=SubtaskPhase.EXECUTION_MINIMAL,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=["ST-03-INVAR"],
                latency_budget_us=300.0,
                energy_budget_j=0.0020,
                verification_oracle="len(c_code) > 0 and len(kmuasm_code) > 0",
                rollback_action="remove_emitted_artifacts()",
                code_snippet="c_code, asm = ksl_codegen.generate(ast)",
            ))
            # 5. PHYSICAL_TELEMETRY
            subtasks.append(SubtaskNode(
                subtask_id="ST-05-TELEM",
                title="Audit Binary Size, Cache Line Footprint, and Latency Ceiling",
                description="Inspect object file ELF symbols and verify memory footprint <= 768 KiB scratchpad.",
                phase=SubtaskPhase.PHYSICAL_TELEMETRY,
                subsystem_placement=SubsystemPlacement.HOST_CPU,
                depends_on=["ST-04-EXEC"],
                latency_budget_us=180.0,
                energy_budget_j=0.0012,
                verification_oracle="elf_scratchpad_bytes <= 786432",
                rollback_action="flag_scratchpad_overflow()",
                code_snippet="assert binary_size <= 768 * 1024",
            ))
            # 6. RECONCILIATION_COMMIT
            subtasks.append(SubtaskNode(
                subtask_id="ST-06-COMMIT",
                title="Sign Module with HMAC-SHA256 & Register in Kernel Manifest",
                description="Emit cryptographic build receipt linking source hash, compiler hash, and binary.",
                phase=SubtaskPhase.RECONCILIATION_COMMIT,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=["ST-05-TELEM"],
                latency_budget_us=60.0,
                energy_budget_j=0.0004,
                verification_oracle="module_registered == True and hmac_valid == True",
                rollback_action="unregister_module_rollback()",
                code_snippet="kernel_manifest.register(module_receipt)",
            ))

        else:  # SYSTEM_GOVERNANCE & TELEMETRY_DIAGNOSTIC (Default robust pipeline)
            # 1. SPECIFICATION
            subtasks.append(SubtaskNode(
                subtask_id="ST-01-SPEC",
                title="Identify System Target State and Boundary Metrics",
                description=f"Map target bounds for objective: '{goal}' under S = (S,A,T,I,O,E,H,W).",
                phase=SubtaskPhase.SPECIFICATION,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=[],
                latency_budget_us=10.0,
                energy_budget_j=0.0001,
                verification_oracle="target_state_specified == True",
                rollback_action="reset_objective_context()",
                code_snippet="target = SystemState.capture_baseline()",
            ))
            # 2. AUTHORITY_CHECK
            subtasks.append(SubtaskNode(
                subtask_id="ST-02-AUTH",
                title="Evaluate Capability Contract & Authority Isolation",
                description="Confirm proposer != evaluator and operation has explicit authorization.",
                phase=SubtaskPhase.AUTHORITY_CHECK,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=["ST-01-SPEC"],
                latency_budget_us=8.0,
                energy_budget_j=0.0001,
                verification_oracle="authority_level >= CapabilityScope.CAN_AUTHORIZE",
                rollback_action="deny_unauthorized_transition()",
                code_snippet="assert capability.has_flag(CapabilityScope.CAN_AUTHORIZE)",
            ))
            # 3. INVARIANT_PRESERVATION
            subtasks.append(SubtaskNode(
                subtask_id="ST-03-INVAR",
                title="Check Invariant Conservation (VITAL_MAX_HP == 6)",
                description="Enforce constitutional invariant set I before any state transition can be initiated.",
                phase=SubtaskPhase.INVARIANT_PRESERVATION,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=["ST-02-AUTH"],
                latency_budget_us=12.0,
                energy_budget_j=0.0001,
                verification_oracle="vital_max_hp == 6 and invariants_held == True",
                rollback_action="abort_action_delta_zero()",
                code_snippet="if VITAL_MAX_HP != 6: raise ContractViolation('Invariant breach')",
            ))
            # 4. EXECUTION_MINIMAL
            subtasks.append(SubtaskNode(
                subtask_id="ST-04-EXEC",
                title="Execute Surgical, Reversible Minimal Action",
                description="Apply isolated configuration update or diagnostic probe with bounded side-effects.",
                phase=SubtaskPhase.EXECUTION_MINIMAL,
                subsystem_placement=SubsystemPlacement.HOST_CPU,
                depends_on=["ST-03-INVAR"],
                latency_budget_us=85.0,
                energy_budget_j=0.0015,
                verification_oracle="action_return_code == 0",
                rollback_action="revert_surgical_action()",
                code_snippet="action_result = execute_minimal_intervention(params)",
            ))
            # 5. PHYSICAL_TELEMETRY
            subtasks.append(SubtaskNode(
                subtask_id="ST-05-TELEM",
                title="Sample Post-Action Physical Telemetry & Residual Error",
                description="Measure thermal response, bus load, and compare against model prediction.",
                phase=SubtaskPhase.PHYSICAL_TELEMETRY,
                subsystem_placement=SubsystemPlacement.TELEMETRY_INA226,
                depends_on=["ST-04-EXEC"],
                latency_budget_us=25.0,
                energy_budget_j=0.0005,
                verification_oracle="observed_telemetry_valid == True and residual < tolerance",
                rollback_action="trigger_emergency_rollback()",
                code_snippet="telemetry = sample_system_telemetry(); residual = abs(pred - telemetry)",
            ))
            # 6. RECONCILIATION_COMMIT
            subtasks.append(SubtaskNode(
                subtask_id="ST-06-COMMIT",
                title="Two-Phase State Commit and HMAC-SHA256 Logging",
                description="If validation passes, commit state S_{t+1}; otherwise enforce Delta S == 0.",
                phase=SubtaskPhase.RECONCILIATION_COMMIT,
                subsystem_placement=SubsystemPlacement.SOVEREIGN_KERNEL,
                depends_on=["ST-05-TELEM"],
                latency_budget_us=30.0,
                energy_budget_j=0.0008,
                verification_oracle="receipt_hmac_valid == True",
                rollback_action="enforce_state_identity_delta_zero()",
                code_snippet="commit_state_receipt(receipt_hmac)",
            ))

        return subtasks

    def _verify_dag_acyclicity(self, subtasks: List[SubtaskNode]) -> bool:
        """Verifies using Kahn's algorithm that the subtask graph has strictly zero cycles."""
        in_degree: Dict[str, int] = {t.subtask_id: 0 for t in subtasks}
        adj_list: Dict[str, List[str]] = {t.subtask_id: [] for t in subtasks}

        for t in subtasks:
            for dep in t.depends_on:
                if dep in adj_list:
                    adj_list[dep].append(t.subtask_id)
                    in_degree[t.subtask_id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        visited_count = 0

        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for nxt in adj_list[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        return visited_count == len(subtasks)

    def _compute_critical_path_latency(self, subtasks: List[SubtaskNode]) -> float:
        """Computes the length of the longest path (critical path) in microseconds."""
        task_map = {t.subtask_id: t for t in subtasks}
        earliest_finish: Dict[str, float] = {}

        for t in subtasks:
            if not t.depends_on:
                earliest_finish[t.subtask_id] = t.latency_budget_us
            else:
                max_dep_finish = max((earliest_finish.get(dep, 0.0) for dep in t.depends_on), default=0.0)
                earliest_finish[t.subtask_id] = max_dep_finish + t.latency_budget_us

        return round(max(earliest_finish.values(), default=0.0), 2)


# ── 5. UNIFIED STUDIO AND FACADE ─────────────────────────────────────────────

class SovereignOperationsStudio:
    """
    Unified operational coordinator combining autocompletion, decomposition,
    and plan validation for user-driven management workflows.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        self.vital_max_hp = vital_max_hp
        self.autocompleter = SovereignOperationsAutocompleter()
        self.decomposer = SovereignOperationsDecomposer(vital_max_hp=vital_max_hp)

    def autocomplete(
        self,
        query: str,
        current_state: Optional[Dict[str, Any]] = None,
        max_results: int = 5,
    ) -> List[OperationSuggestion]:
        """Provides instant command/operation completion."""
        return self.autocompleter.autocomplete(query, current_state, max_results)

    def decompose_management_goal(
        self,
        goal: str,
        domain: Optional[OperationDomain] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> SubtaskDAG:
        """Decomposes a management goal into an executable subtask DAG."""
        return self.decomposer.decompose(goal, domain, parameters)

    def format_plan_report(self, dag: SubtaskDAG) -> str:
        """Renders an executive markdown report with table, timeline, and Mermaid diagram."""
        lines = [
            f"# SOVEREIGN OPERATIONS PLAN: {dag.plan_id}",
            f"**Management Goal:** `{dag.management_goal}`  ",
            f"**Domain:** `{dag.domain.value}` | **Critical Path:** `{dag.critical_path_latency_us} µs` | **Energy Budget:** `{dag.total_energy_budget_j} J`  ",
            f"**Acyclic Invariant:** `{dag.is_acyclic}` | **Constitutional HP:** `VITAL_MAX_HP == {dag.vital_max_hp}`  ",
            f"**HMAC-SHA256:** `{dag.hmac_sha256}`  ",
            "",
            "## 1. Decomposed Subtask Table",
            "",
            "| ID | Phase | Subtask Title | Placement | Latency (µs) | Depends On | Rollback Action |",
            "|:---|:---|:---|:---|:---:|:---|:---|",
        ]
        for t in dag.subtasks:
            deps_str = ", ".join(t.depends_on) if t.depends_on else "*Root*"
            lines.append(
                f"| `{t.subtask_id}` | `{t.phase.value}` | **{t.title}** | `{t.subsystem_placement.value}` | {t.latency_budget_us:.1f} | `{deps_str}` | *{t.rollback_action}* |"
            )

        lines.extend([
            "",
            "## 2. Dependency Flowchart (Mermaid)",
            "",
            dag.generate_mermaid_diagram(),
            "",
            "## 3. Subtask Execution Code Snippets",
            "",
        ])
        for t in dag.subtasks:
            lines.extend([
                f"### [{t.subtask_id}] {t.title}",
                f"- **Phase:** `{t.phase.value}` | **Target:** `{t.subsystem_placement.value}`",
                f"- **Verification Oracle:** `{t.verification_oracle}`",
                "```python",
                t.code_snippet,
                "```",
                "",
            ])
        return "\n".join(lines)


# Global singleton instance
GLOBAL_OPERATIONS_STUDIO = SovereignOperationsStudio()


# ── 6. CLI INTERFACE ─────────────────────────────────────────────────────────

def run_cli() -> None:
    """Command-line interface for testing autocompletion and task decomposition."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Sovereign Operations Autocompleter & Management Task Decomposition Generator"
    )
    parser.add_argument("--autocomplete", "-a", type=str, help="Query string to autocomplete")
    parser.add_argument("--decompose", "-d", type=str, help="Management goal or operation to decompose")
    parser.add_argument("--domain", type=str, choices=[d.value for d in OperationDomain], help="Explicit domain override")
    parser.add_argument("--export-json", action="store_true", help="Output raw JSON instead of markdown")
    args = parser.parse_args()

    studio = GLOBAL_OPERATIONS_STUDIO

    if args.autocomplete:
        suggestions = studio.autocomplete(args.autocomplete)
        if args.export_json:
            print(json.dumps([asdict(s) for s in suggestions], indent=2))
        else:
            print(f"=== AUTOCOMPLETION RESULTS FOR '{args.autocomplete}' ===")
            for idx, s in enumerate(suggestions, start=1):
                print(f"[{idx}] {s.command} ({s.domain.value}) [Score: {s.score:.2f}]")
                print(f"    Title:    {s.display_title}")
                print(f"    Suffix:   '{s.completion_suffix}'")
                print(f"    Synopsis: {s.synopsis}")
                print(f"    Latency:  {s.estimated_latency_us} µs | Gate: {s.invariant_gate}")
                print()

    elif args.decompose:
        dom = OperationDomain(args.domain) if args.domain else None
        dag = studio.decompose_management_goal(args.decompose, domain=dom)
        if args.export_json:
            print(json.dumps(dag.to_dict(), indent=2))
        else:
            print(studio.format_plan_report(dag))
    else:
        # Default demo execution
        print("=== SOVEREIGN OPERATIONS STUDIO (DEMO) ===")
        print("\n--- 1. Testing Autocompleter for query 'audit' ---")
        for s in studio.autocomplete("audit"):
            print(f" -> {s.command} [suffix: '{s.completion_suffix}'] ({s.display_title})")

        print("\n--- 2. Decomposing Management Goal: 'Run M4.1 Physical Validation Benchmark' ---")
        dag = studio.decompose_management_goal("Run M4.1 Physical Validation Benchmark")
        print(studio.format_plan_report(dag))


if __name__ == "__main__":
    run_cli()
