"""
os_coordination_policy_engine.py - Non-Invasive OS Coordination Layer & Reversible Policy Engine.

Implements the normative tenets of Milestone V-NON-INVASIVE-OS-01 (ARCH-POLICY-ENGINE-2026-V1):
1. Never rewrite the OS kernel; coordinate across subsystems via stable, existing interfaces.
2. Discovery Phase across 10 OS subsystems: CPU, Memory, I/O, FS, Network, Accelerators, Thermal,
   cgroups, systemd, drivers.
3. 11-Tier Control Surface Hierarchy: App Config -> Env Vars -> systemd drop-ins -> cgroups ->
   sysfs/procfs -> sysctl -> IRQ affinity -> eBPF observation -> module params -> boot params -> kernel patch.
4. Transactional isolated execution: single change at a time ("anti-batch" invariant), measure before & after.
5. Triad of Separation: Telemetry observes, Policy proposes, Validation decides.
6. Mandatory LIFO Rollback Stack.
7. Constitutional Invariant: VITAL_MAX_HP == 6.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

VITAL_MAX_HP: int = 6


class EpistemicContractViolation(Exception):
    """Raised when an OS coordination invariant or constitutional rule is violated."""
    pass


class ControlSurfaceTier(enum.IntEnum):
    """
    11-Tier Control Surface Hierarchy ordered strictly by non-invasiveness.
    Lower number = preferred / less invasive.
    """
    APP_CONFIG = 1
    ENV_VARS = 2
    SYSTEMD_DROPINS = 3
    CGROUPS_V2 = 4
    SYSFS_PROCFS = 5
    SYSCTL = 6
    IRQ_AFFINITY = 7
    EBPF_OBSERVATION = 8
    MODULE_PARAMETERS = 9
    BOOT_PARAMETERS = 10
    KERNEL_PATCH = 11


class SubsystemType(str, enum.Enum):
    CPU = "CPU"
    MEMORY = "MEMORY"
    IO = "IO"
    FILESYSTEM = "FILESYSTEM"
    NETWORK = "NETWORK"
    ACCELERATOR = "ACCELERATOR"
    THERMAL = "THERMAL"
    CGROUPS = "CGROUPS"
    SYSTEMD = "SYSTEMD"
    DRIVER = "DRIVER"


class OptimizationStatus(str, enum.Enum):
    HYPOTHESIS = "hypothesis"
    TESTING = "testing"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    CONFLICT = "conflict"


@dataclass
class SubsystemOptimizationHypothesis:
    """
    Structured hypothesis for cross-subsystem optimization.
    Must include mechanism, target metric, expected gain, side effects, falsifier, and rollback.
    """
    hypothesis_id: str
    subsystem: SubsystemType
    control_surface: ControlSurfaceTier
    mechanism: str
    target_metric: str
    expected_gain: str
    side_effects: str
    falsifier: str
    reversibility: str
    rollback_payload: Dict[str, Any]
    status: OptimizationStatus = OptimizationStatus.HYPOTHESIS
    baseline_metric: Optional[float] = None
    post_change_metric: Optional[float] = None
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise EpistemicContractViolation(
                f"Constitutional violation: VITAL_MAX_HP must be {VITAL_MAX_HP}, got {self.vital_max_hp}"
            )
        if not self.falsifier:
            raise EpistemicContractViolation("Hypothesis requires an explicit falsifier.")
        if not self.rollback_payload:
            raise EpistemicContractViolation("Hypothesis requires an explicit rollback payload.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "subsystem": self.subsystem.value,
            "control_surface": self.control_surface.name,
            "tier": self.control_surface.value,
            "mechanism": self.mechanism,
            "target_metric": self.target_metric,
            "expected_gain": self.expected_gain,
            "side_effects": self.side_effects,
            "falsifier": self.falsifier,
            "reversibility": self.reversibility,
            "status": self.status.value,
            "baseline_metric": self.baseline_metric,
            "post_change_metric": self.post_change_metric,
        }


class ThinSubsystemAdapter:
    """
    Base class for thin adapters interacting with OS subsystems strictly through stable interfaces.
    """
    def __init__(self, subsystem: SubsystemType) -> None:
        self.subsystem = subsystem
        self.active_state: Dict[str, Any] = {}

    def apply(self, hypothesis: SubsystemOptimizationHypothesis) -> Dict[str, Any]:
        """Applies configuration change and records previous state for rollback."""
        raise NotImplementedError

    def rollback(self, rollback_payload: Dict[str, Any]) -> bool:
        """Rolls back configuration change to exact previous state."""
        raise NotImplementedError


class CPUAdapter(ThinSubsystemAdapter):
    def __init__(self) -> None:
        super().__init__(SubsystemType.CPU)
        self.active_state = {"cpuset": "0-7", "governor": "powersave", "epp": "balance_performance"}

    def apply(self, hypothesis: SubsystemOptimizationHypothesis) -> Dict[str, Any]:
        old_state = dict(self.active_state)
        # Simulated application through stable sysfs/cgroups interface
        if "cpuset" in hypothesis.rollback_payload:
            self.active_state["cpuset"] = "0-3"  # Pin to P-cores
        if "governor" in hypothesis.rollback_payload:
            self.active_state["governor"] = "performance"
        return old_state

    def rollback(self, rollback_payload: Dict[str, Any]) -> bool:
        self.active_state.update(rollback_payload)
        return True


class MemoryAdapter(ThinSubsystemAdapter):
    def __init__(self) -> None:
        super().__init__(SubsystemType.MEMORY)
        self.active_state = {"swappiness": 60, "dirty_ratio": 20, "thp": "madvise"}

    def apply(self, hypothesis: SubsystemOptimizationHypothesis) -> Dict[str, Any]:
        old_state = dict(self.active_state)
        if "swappiness" in hypothesis.rollback_payload:
            self.active_state["swappiness"] = 10
        if "dirty_ratio" in hypothesis.rollback_payload:
            self.active_state["dirty_ratio"] = 10
        return old_state

    def rollback(self, rollback_payload: Dict[str, Any]) -> bool:
        self.active_state.update(rollback_payload)
        return True


class IOAdapter(ThinSubsystemAdapter):
    def __init__(self) -> None:
        super().__init__(SubsystemType.IO)
        self.active_state = {"scheduler": "mq-deadline", "read_ahead_kb": 128}

    def apply(self, hypothesis: SubsystemOptimizationHypothesis) -> Dict[str, Any]:
        old_state = dict(self.active_state)
        if "scheduler" in hypothesis.rollback_payload:
            self.active_state["scheduler"] = "kyber"
        return old_state

    def rollback(self, rollback_payload: Dict[str, Any]) -> bool:
        self.active_state.update(rollback_payload)
        return True


class AcceleratorAdapter(ThinSubsystemAdapter):
    def __init__(self) -> None:
        super().__init__(SubsystemType.ACCELERATOR)
        self.active_state = {"priority_queue": "normal", "power_limit_watts": 28.0}

    def apply(self, hypothesis: SubsystemOptimizationHypothesis) -> Dict[str, Any]:
        old_state = dict(self.active_state)
        if "priority_queue" in hypothesis.rollback_payload:
            self.active_state["priority_queue"] = "high_priority_compute"
        return old_state

    def rollback(self, rollback_payload: Dict[str, Any]) -> bool:
        self.active_state.update(rollback_payload)
        return True


class ThermalAdapter(ThinSubsystemAdapter):
    def __init__(self) -> None:
        super().__init__(SubsystemType.THERMAL)
        self.active_state = {"pl1_limit_watts": 28.0, "pl2_limit_watts": 64.0, "tau_sec": 28}

    def apply(self, hypothesis: SubsystemOptimizationHypothesis) -> Dict[str, Any]:
        old_state = dict(self.active_state)
        if "pl1_limit_watts" in hypothesis.rollback_payload:
            self.active_state["pl1_limit_watts"] = 35.0
        return old_state

    def rollback(self, rollback_payload: Dict[str, Any]) -> bool:
        self.active_state.update(rollback_payload)
        return True


class TransactionalChangeStack:
    """
    Manages transactional isolation and LIFO rollback stack.
    Enforces rule: strictly 1 active uncommitted change at a time (anti-batch invariant).
    """
    def __init__(self) -> None:
        self._stack: List[Tuple[SubsystemOptimizationHypothesis, ThinSubsystemAdapter]] = []
        self._active_uncommitted: Optional[SubsystemOptimizationHypothesis] = None

    @property
    def has_uncommitted_change(self) -> bool:
        return self._active_uncommitted is not None

    def begin_transaction(
        self,
        hypothesis: SubsystemOptimizationHypothesis,
        adapter: ThinSubsystemAdapter
    ) -> None:
        if self.has_uncommitted_change:
            raise EpistemicContractViolation(
                f"Anti-Batch Violation: Cannot apply change '{hypothesis.hypothesis_id}' while "
                f"'{self._active_uncommitted.hypothesis_id}' is still in testing/uncommitted state."
            )
        hypothesis.status = OptimizationStatus.TESTING
        self._active_uncommitted = hypothesis
        # Apply change and record to stack
        adapter.apply(hypothesis)
        self._stack.append((hypothesis, adapter))

    def commit_transaction(self, hypothesis_id: str) -> None:
        if not self._active_uncommitted or self._active_uncommitted.hypothesis_id != hypothesis_id:
            raise EpistemicContractViolation(f"No active uncommitted transaction for '{hypothesis_id}'.")
        self._active_uncommitted.status = OptimizationStatus.SUPPORTED
        self._active_uncommitted = None

    def rollback_latest(self, reason: str) -> Dict[str, Any]:
        """Rolls back the latest active change."""
        if not self._stack:
            return {"rolled_back": False, "reason": "Stack empty"}

        hyp, adapter = self._stack.pop()
        adapter.rollback(hyp.rollback_payload)
        hyp.status = OptimizationStatus.REJECTED
        self._active_uncommitted = None
        return {
            "rolled_back": True,
            "hypothesis_id": hyp.hypothesis_id,
            "subsystem": hyp.subsystem.value,
            "reason": reason,
            "restored_state": hyp.rollback_payload,
        }

    def rollback_all(self, reason: str) -> List[Dict[str, Any]]:
        """LIFO rollback of all recorded changes."""
        receipts = []
        while self._stack:
            receipts.append(self.rollback_latest(reason))
        return receipts


class TelemetryObserver:
    """
    Pure observer: reads metrics, counter probes, jitter without changing configurations.
    """
    def __init__(self) -> None:
        self.telemetry_history: List[Dict[str, Any]] = []

    def observe_metric(self, metric_name: str, simulated_value: float) -> float:
        self.telemetry_history.append({"metric": metric_name, "value": simulated_value})
        return simulated_value


class ValidationGate:
    """
    Independent decision authority: Evaluator != Proposer.
    Compares baseline vs post-change metrics against falsifier and expected gain.
    """
    def __init__(self, evaluator_id: str = "independent_validation_gate") -> None:
        self.evaluator_id = evaluator_id

    def evaluate_change(
        self,
        hypothesis: SubsystemOptimizationHypothesis,
        proposer_id: str,
        baseline: float,
        post_change: float,
        falsifier_triggered: bool
    ) -> Dict[str, Any]:
        if self.evaluator_id == proposer_id:
            raise EpistemicContractViolation(
                f"Anti-Self-Certification Violation: Proposer '{proposer_id}' cannot evaluate own change."
            )

        hypothesis.baseline_metric = baseline
        hypothesis.post_change_metric = post_change

        if falsifier_triggered:
            return {
                "verdict": "REJECT_AND_ROLLBACK",
                "reason": f"Falsifier triggered: {hypothesis.falsifier}",
                "should_rollback": True,
            }

        # Check for empirical improvement
        delta = post_change - baseline
        # Assuming lower latency/jitter is better
        is_improvement = post_change < baseline
        if is_improvement:
            return {
                "verdict": "APPROVE_AND_COMMIT",
                "reason": f"Empirical improvement confirmed: baseline={baseline}, post={post_change}, delta={delta:.3f}",
                "should_rollback": False,
            }

        return {
            "verdict": "REJECT_AND_ROLLBACK",
            "reason": f"Regression or zero improvement: baseline={baseline}, post={post_change}",
            "should_rollback": True,
        }


class KrystalPolicyEngine:
    """
    Central Brain orchestrating the non-invasive OS coordination layer:
    - Discovery catalog across 10 subsystems.
    - Control Surface tier enforcement (prefer App Config over sysfs/sysctl).
    - Transactional execution (1 change at a time).
    - Telemetry observes, Policy proposes, Validation decides.
    """
    def __init__(self, proposer_id: str = "krystal_policy_engine_brain") -> None:
        self.proposer_id = proposer_id
        self.adapters: Dict[SubsystemType, ThinSubsystemAdapter] = {
            SubsystemType.CPU: CPUAdapter(),
            SubsystemType.MEMORY: MemoryAdapter(),
            SubsystemType.IO: IOAdapter(),
            SubsystemType.ACCELERATOR: AcceleratorAdapter(),
            SubsystemType.THERMAL: ThermalAdapter(),
        }
        self.stack = TransactionalChangeStack()
        self.observer = TelemetryObserver()
        self.validation_gate = ValidationGate(evaluator_id="independent_system_auditor")
        self.discovery_catalog: Dict[SubsystemType, Dict[str, Any]] = {}
        self.hypotheses: Dict[str, SubsystemOptimizationHypothesis] = {}

    def run_subsystem_discovery(self) -> Dict[SubsystemType, Dict[str, Any]]:
        """Pasívna fáza discovery pre všetky kľúčové subsystémy."""
        self.discovery_catalog = {
            SubsystemType.CPU: {"topology": "4P+4E", "scaling_governor": "powersave", "epp": "balance_performance"},
            SubsystemType.MEMORY: {"ram_gb": 32, "zram_active": True, "swappiness": 60, "thp": "madvise"},
            SubsystemType.IO: {"default_scheduler": "mq-deadline", "nvme_queue_depth": 64},
            SubsystemType.FILESYSTEM: {"mount_options": "rw,noatime,discard", "journal_commit_sec": 5},
            SubsystemType.NETWORK: {"tcp_congestion": "bbr", "rmem_max": 2097152},
            SubsystemType.ACCELERATOR: {"gpu": "Intel Iris Xe Graphics", "driver": "i915/xe", "compute_queues": 4},
            SubsystemType.THERMAL: {"pl1_w": 28.0, "pl2_w": 64.0, "tau_s": 28, "junction_temp_c": 54.0},
            SubsystemType.CGROUPS: {"version": "v2", "controllers": ["cpu", "memory", "io"]},
            SubsystemType.SYSTEMD: {"active_slices": ["app.slice", "system.slice", "user.slice"]},
            SubsystemType.DRIVER: {"hags_supported": True, "tdr_timeout_sec": 5},
        }
        return self.discovery_catalog

    def propose_hypothesis(
        self,
        subsystem: SubsystemType,
        control_surface: ControlSurfaceTier,
        mechanism: str,
        target_metric: str,
        expected_gain: str,
        side_effects: str,
        falsifier: str,
        reversibility: str,
        rollback_payload: Dict[str, Any],
    ) -> SubsystemOptimizationHypothesis:
        # Enforce that kernel patches are rejected unless explicit exceptional escalation
        if control_surface == ControlSurfaceTier.KERNEL_PATCH:
            raise EpistemicContractViolation(
                "Violation of Non-Invasive Invariant: Kernel patches are forbidden when userspace interfaces exist."
            )

        hyp_id = f"hyp_{subsystem.value.lower()}_{uuid.uuid4().hex[:6]}"
        hyp = SubsystemOptimizationHypothesis(
            hypothesis_id=hyp_id,
            subsystem=subsystem,
            control_surface=control_surface,
            mechanism=mechanism,
            target_metric=target_metric,
            expected_gain=expected_gain,
            side_effects=side_effects,
            falsifier=falsifier,
            reversibility=reversibility,
            rollback_payload=rollback_payload,
            vital_max_hp=VITAL_MAX_HP,
        )
        self.hypotheses[hyp_id] = hyp
        return hyp

    def execute_transactional_optimization_cycle(
        self,
        hypothesis: SubsystemOptimizationHypothesis,
        simulated_baseline: float,
        simulated_post_change: float,
        falsifier_triggered: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes the mandatory lifecycle:
        1. Measure Baseline.
        2. Begin Transaction & Apply Change via Adapter.
        3. Measure Post-Change.
        4. Independent Validation Gate Evaluation.
        5. Commit or Rollback.
        """
        if hypothesis.subsystem not in self.adapters:
            raise EpistemicContractViolation(f"No thin adapter implemented for subsystem {hypothesis.subsystem.value}")

        adapter = self.adapters[hypothesis.subsystem]

        # 1. Telemetry observes baseline
        baseline = self.observer.observe_metric(hypothesis.target_metric, simulated_baseline)

        # 2. Begin transaction (enforces anti-batch)
        self.stack.begin_transaction(hypothesis, adapter)

        # 3. Telemetry observes post-change
        post = self.observer.observe_metric(hypothesis.target_metric, simulated_post_change)

        # 4. Validation Gate decides
        verdict_receipt = self.validation_gate.evaluate_change(
            hypothesis=hypothesis,
            proposer_id=self.proposer_id,
            baseline=baseline,
            post_change=post,
            falsifier_triggered=falsifier_triggered,
        )

        # 5. Commit or Rollback
        if verdict_receipt["should_rollback"]:
            rollback_res = self.stack.rollback_latest(reason=verdict_receipt["reason"])
            return {
                "outcome": "ROLLED_BACK",
                "hypothesis": hypothesis.to_dict(),
                "verdict": verdict_receipt,
                "rollback": rollback_res,
            }
        else:
            self.stack.commit_transaction(hypothesis.hypothesis_id)
            return {
                "outcome": "COMMITTED",
                "hypothesis": hypothesis.to_dict(),
                "verdict": verdict_receipt,
            }
