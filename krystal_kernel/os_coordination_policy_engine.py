"""
os_coordination_policy_engine.py - Non-Invasive OS Coordination Layer & Reversible Policy Engine (V2).

Implements the refined normative tenets of Milestone V-NON-INVASIVE-OS-02:
1. One Policy Brain, thin capability-based adapters, host OS remains authority.
2. Anti-Batch Invariant reformulated as: "One unvalidated transaction at a time"
   (enables coordinated cross-subsystem hypotheses while preventing uncontrolled drift).
3. Explicit Rollback Classes:
   - EXACT_STATE: immediate discrete state restoration (cpuset, env vars, cgroups).
   - CONFIG_RESTORE: register/param restoration; physical state converges over dwell time (power limits, thermals, GPU clocks).
   - DRAIN_AND_CONVERGE: queue flush / buffer drain (dirty writeback, socket buffers).
   - IRREVERSIBLE_SIDE_EFFECT: parameter restored, but side-effect cannot be un-done (cache drops).
4. eBPF moved to Observability (separate from Actuation Control Surface Hierarchy).
5. Explicit separation of Linux and Windows via Capability Providers.
6. Component separation: PolicyProposer != ValidationEvaluator as distinct object components.
7. VITAL_MAX_HP maintained as a global repository invariant.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Global Repository Constitutional Invariant
GLOBAL_VITAL_MAX_HP: int = 6


class EpistemicContractViolation(Exception):
    """Raised when an OS coordination invariant or architectural contract is violated."""
    pass


class RollbackClass(str, enum.Enum):
    """
    Categorizes the physical and temporal nature of reversibility.
    """
    EXACT_STATE = "EXACT_STATE"
    CONFIG_RESTORE = "CONFIG_RESTORE"
    DRAIN_AND_CONVERGE = "DRAIN_AND_CONVERGE"
    IRREVERSIBLE_SIDE_EFFECT = "IRREVERSIBLE_SIDE_EFFECT"


class ControlSurfaceTier(enum.IntEnum):
    """
    10-Tier Control Surface Hierarchy strictly for Actuation.
    Ordered from least invasive to most invasive.
    eBPF is deliberately excluded here and placed in ObservabilityChannel.
    """
    APP_CONFIG = 1
    ENV_VARS = 2
    SYSTEMD_DROPINS = 3
    CGROUPS_OR_JOBOBJECTS = 4
    SYSFS_PROCFS = 5
    SYSCTL = 6
    IRQ_AFFINITY = 7
    MODULE_PARAMETERS = 8
    BOOT_PARAMETERS = 9
    KERNEL_PATCH = 10


class ObservabilityChannel(str, enum.Enum):
    """
    Channels reserved purely for passive non-intrusive observation and telemetry.
    """
    EBPF_TRACEPOINT = "EBPF_TRACEPOINT"
    ETW_WINDOWS = "ETW_WINDOWS"
    HW_PERF_COUNTERS = "HW_PERF_COUNTERS"
    PROCFS_POLL = "PROCFS_POLL"
    PDH_COUNTERS = "PDH_COUNTERS"


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


# ============================================================================
# Platform Capability Providers (Linux vs Windows Separation)
# ============================================================================

class OSCapabilityProvider:
    """Base capability provider defining OS-specific interface query methods."""
    def __init__(self, platform_name: str) -> None:
        self.platform_name = platform_name

    def supports_control_surface(self, tier: ControlSurfaceTier) -> bool:
        raise NotImplementedError

    def get_supported_tiers(self) -> Set[ControlSurfaceTier]:
        raise NotImplementedError

    def get_subsystem_interface(self, subsystem: SubsystemType) -> str:
        raise NotImplementedError


class LinuxCapabilityProvider(OSCapabilityProvider):
    def __init__(self) -> None:
        super().__init__("Linux")

    def supports_control_surface(self, tier: ControlSurfaceTier) -> bool:
        return tier in self.get_supported_tiers()

    def get_supported_tiers(self) -> Set[ControlSurfaceTier]:
        return {
            ControlSurfaceTier.APP_CONFIG,
            ControlSurfaceTier.ENV_VARS,
            ControlSurfaceTier.SYSTEMD_DROPINS,
            ControlSurfaceTier.CGROUPS_OR_JOBOBJECTS,
            ControlSurfaceTier.SYSFS_PROCFS,
            ControlSurfaceTier.SYSCTL,
            ControlSurfaceTier.IRQ_AFFINITY,
            ControlSurfaceTier.MODULE_PARAMETERS,
            ControlSurfaceTier.BOOT_PARAMETERS,
            # KERNEL_PATCH is supported by OS physically, but strictly prohibited by policy
        }

    def get_subsystem_interface(self, subsystem: SubsystemType) -> str:
        interfaces = {
            SubsystemType.CPU: "/sys/devices/system/cpu/cpufreq",
            SubsystemType.MEMORY: "/proc/sys/vm",
            SubsystemType.IO: "/sys/block/*/queue",
            SubsystemType.FILESYSTEM: "/proc/mounts",
            SubsystemType.NETWORK: "sysctl net.ipv4",
            SubsystemType.ACCELERATOR: "/dev/dri/renderD128",
            SubsystemType.THERMAL: "/sys/class/thermal",
            SubsystemType.CGROUPS: "/sys/fs/cgroup",
            SubsystemType.SYSTEMD: "systemctl set-property",
            SubsystemType.DRIVER: "/sys/module",
        }
        return interfaces.get(subsystem, "generic_linux_vfs")


class WindowsCapabilityProvider(OSCapabilityProvider):
    def __init__(self) -> None:
        super().__init__("Windows")

    def supports_control_surface(self, tier: ControlSurfaceTier) -> bool:
        return tier in self.get_supported_tiers()

    def get_supported_tiers(self) -> Set[ControlSurfaceTier]:
        return {
            ControlSurfaceTier.APP_CONFIG,
            ControlSurfaceTier.ENV_VARS,
            ControlSurfaceTier.CGROUPS_OR_JOBOBJECTS,  # Windows Job Objects
            ControlSurfaceTier.IRQ_AFFINITY,          # SetThreadSelectedCpuSets / SetProcessAffinityMask
            ControlSurfaceTier.BOOT_PARAMETERS,       # BCD / bootmgr
        }

    def get_subsystem_interface(self, subsystem: SubsystemType) -> str:
        interfaces = {
            SubsystemType.CPU: "Win32_SetThreadSelectedCpuSets / EcoQoS",
            SubsystemType.MEMORY: "VirtualAlloc2 / MEMORYSTATUSEX",
            SubsystemType.IO: "DeviceIoControl / SetFileInformationByHandle",
            SubsystemType.FILESYSTEM: "GetVolumeInformationByHandle",
            SubsystemType.NETWORK: "IP Helper API / Winsock IOCTL",
            SubsystemType.ACCELERATOR: "DXGI / Direct3D12 / LevelZero",
            SubsystemType.THERMAL: "IOCTL_THERMAL_QUERY_INFORMATION / DPTF",
            SubsystemType.CGROUPS: "Windows Job Objects (CpuRateControl / MemoryLimit)",
            SubsystemType.SYSTEMD: "Windows Services API (Advapi32)",
            SubsystemType.DRIVER: "Windows Registry Display Configuration",
        }
        return interfaces.get(subsystem, "win32_subsystem")


# ============================================================================
# Coordinated Transaction & Actions
# ============================================================================

@dataclass
class SubsystemAction:
    """
    An individual action within a coordinated transaction.
    """
    subsystem: SubsystemType
    control_surface: ControlSurfaceTier
    rollback_class: RollbackClass
    target_param: str
    target_value: Any
    rollback_payload: Dict[str, Any]

    def __post_init__(self) -> None:
        if not self.rollback_payload:
            raise EpistemicContractViolation(f"SubsystemAction on {self.subsystem.value} requires rollback_payload.")


@dataclass
class CoordinationTransaction:
    """
    Represents an atomic, cross-subsystem optimization hypothesis.
    Invariant: Only ONE unvalidated transaction may be active in TESTING state at any time.
    """
    transaction_id: str
    hypothesis_rationale: str
    actions: List[SubsystemAction]
    target_metric: str
    expected_gain: str
    side_effects: str
    falsifier: str
    status: OptimizationStatus = OptimizationStatus.HYPOTHESIS
    baseline_metric: Optional[float] = None
    post_change_metric: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.actions:
            raise EpistemicContractViolation("CoordinationTransaction must contain at least one action.")
        if not self.falsifier:
            raise EpistemicContractViolation("CoordinationTransaction requires an explicit falsifier.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "hypothesis": self.hypothesis_rationale,
            "action_count": len(self.actions),
            "subsystems": [a.subsystem.value for a in self.actions],
            "rollback_classes": [a.rollback_class.value for a in self.actions],
            "target_metric": self.target_metric,
            "expected_gain": self.expected_gain,
            "side_effects": self.side_effects,
            "falsifier": self.falsifier,
            "status": self.status.value,
            "baseline_metric": self.baseline_metric,
            "post_change_metric": self.post_change_metric,
        }


# ============================================================================
# Capability-Based Thin Adapters
# ============================================================================

class CapabilityBasedThinAdapter:
    """Base thin adapter bound to an OSCapabilityProvider."""
    def __init__(self, subsystem: SubsystemType, provider: OSCapabilityProvider) -> None:
        self.subsystem = subsystem
        self.provider = provider
        self.active_state: Dict[str, Any] = {}

    def apply_action(self, action: SubsystemAction) -> Dict[str, Any]:
        raise NotImplementedError

    def rollback_action(self, action: SubsystemAction) -> Dict[str, Any]:
        raise NotImplementedError


class CapabilityCPUAdapter(CapabilityBasedThinAdapter):
    def __init__(self, provider: OSCapabilityProvider) -> None:
        super().__init__(SubsystemType.CPU, provider)
        self.active_state = {"cpuset": "0-7", "governor": "powersave", "epp": "balance_performance"}

    def apply_action(self, action: SubsystemAction) -> Dict[str, Any]:
        old_state = dict(self.active_state)
        self.active_state[action.target_param] = action.target_value
        return old_state

    def rollback_action(self, action: SubsystemAction) -> Dict[str, Any]:
        self.active_state.update(action.rollback_payload)
        return {
            "subsystem": self.subsystem.value,
            "rollback_class": action.rollback_class.value,
            "restored_state": action.rollback_payload,
        }


class CapabilityMemoryAdapter(CapabilityBasedThinAdapter):
    def __init__(self, provider: OSCapabilityProvider) -> None:
        super().__init__(SubsystemType.MEMORY, provider)
        self.active_state = {"swappiness": 60, "dirty_ratio": 20, "thp": "madvise"}

    def apply_action(self, action: SubsystemAction) -> Dict[str, Any]:
        old_state = dict(self.active_state)
        self.active_state[action.target_param] = action.target_value
        return old_state

    def rollback_action(self, action: SubsystemAction) -> Dict[str, Any]:
        self.active_state.update(action.rollback_payload)
        return {
            "subsystem": self.subsystem.value,
            "rollback_class": action.rollback_class.value,
            "restored_state": action.rollback_payload,
        }


class CapabilityIOAdapter(CapabilityBasedThinAdapter):
    def __init__(self, provider: OSCapabilityProvider) -> None:
        super().__init__(SubsystemType.IO, provider)
        self.active_state = {"scheduler": "mq-deadline", "read_ahead_kb": 128}

    def apply_action(self, action: SubsystemAction) -> Dict[str, Any]:
        old_state = dict(self.active_state)
        self.active_state[action.target_param] = action.target_value
        return old_state

    def rollback_action(self, action: SubsystemAction) -> Dict[str, Any]:
        self.active_state.update(action.rollback_payload)
        return {
            "subsystem": self.subsystem.value,
            "rollback_class": action.rollback_class.value,
            "restored_state": action.rollback_payload,
        }


class CapabilityAcceleratorAdapter(CapabilityBasedThinAdapter):
    def __init__(self, provider: OSCapabilityProvider) -> None:
        super().__init__(SubsystemType.ACCELERATOR, provider)
        self.active_state = {"priority_queue": "normal", "power_limit_watts": 28.0}

    def apply_action(self, action: SubsystemAction) -> Dict[str, Any]:
        old_state = dict(self.active_state)
        self.active_state[action.target_param] = action.target_value
        return old_state

    def rollback_action(self, action: SubsystemAction) -> Dict[str, Any]:
        self.active_state.update(action.rollback_payload)
        return {
            "subsystem": self.subsystem.value,
            "rollback_class": action.rollback_class.value,
            "restored_state": action.rollback_payload,
        }


class CapabilityThermalAdapter(CapabilityBasedThinAdapter):
    def __init__(self, provider: OSCapabilityProvider) -> None:
        super().__init__(SubsystemType.THERMAL, provider)
        self.active_state = {"pl1_limit_watts": 28.0, "pl2_limit_watts": 64.0, "tau_sec": 28}

    def apply_action(self, action: SubsystemAction) -> Dict[str, Any]:
        old_state = dict(self.active_state)
        self.active_state[action.target_param] = action.target_value
        return old_state

    def rollback_action(self, action: SubsystemAction) -> Dict[str, Any]:
        self.active_state.update(action.rollback_payload)
        return {
            "subsystem": self.subsystem.value,
            "rollback_class": action.rollback_class.value,
            "restored_state": action.rollback_payload,
        }


# ============================================================================
# Transactional Change Stack: One Unvalidated Transaction At A Time
# ============================================================================

class TransactionalChangeStack:
    """
    Manages transactional isolation and LIFO rollback.
    Anti-Batch Invariant: Exactly ONE unvalidated transaction in TESTING state at any time.
    Supports atomic cross-subsystem bundles within that single transaction.
    """
    def __init__(self) -> None:
        self._history: List[CoordinationTransaction] = []
        self._active_unvalidated_transaction: Optional[CoordinationTransaction] = None
        self._active_adapters: Dict[SubsystemType, CapabilityBasedThinAdapter] = {}

    @property
    def has_unvalidated_transaction(self) -> bool:
        return self._active_unvalidated_transaction is not None

    def begin_transaction(
        self,
        transaction: CoordinationTransaction,
        adapters: Dict[SubsystemType, CapabilityBasedThinAdapter]
    ) -> None:
        if self.has_unvalidated_transaction:
            assert self._active_unvalidated_transaction is not None
            raise EpistemicContractViolation(
                f"Anti-Batch Invariant Violation: Cannot begin transaction '{transaction.transaction_id}' while "
                f"transaction '{self._active_unvalidated_transaction.transaction_id}' is still in unvalidated/testing state."
            )

        transaction.status = OptimizationStatus.TESTING
        self._active_unvalidated_transaction = transaction
        self._active_adapters = adapters

        # Atomically apply all actions within the transaction
        for action in transaction.actions:
            if action.subsystem not in adapters:
                raise EpistemicContractViolation(f"No adapter available for subsystem {action.subsystem.value}")
            adapters[action.subsystem].apply_action(action)

    def commit_transaction(self, transaction_id: str) -> None:
        if not self._active_unvalidated_transaction or self._active_unvalidated_transaction.transaction_id != transaction_id:
            raise EpistemicContractViolation(f"No active unvalidated transaction matching '{transaction_id}' to commit.")

        self._active_unvalidated_transaction.status = OptimizationStatus.SUPPORTED
        self._history.append(self._active_unvalidated_transaction)
        self._active_unvalidated_transaction = None

    def rollback_transaction(self, reason: str) -> Dict[str, Any]:
        """
        Rolls back the active unvalidated transaction in reverse (LIFO) action order.
        """
        if not self._active_unvalidated_transaction:
            return {"rolled_back": False, "reason": "No active transaction to rollback"}

        tx = self._active_unvalidated_transaction
        action_receipts = []

        # LIFO unwind across all actions
        for action in reversed(tx.actions):
            adapter = self._active_adapters[action.subsystem]
            rec = adapter.rollback_action(action)
            action_receipts.append(rec)

        tx.status = OptimizationStatus.REJECTED
        self._history.append(tx)
        self._active_unvalidated_transaction = None

        return {
            "rolled_back": True,
            "transaction_id": tx.transaction_id,
            "reason": reason,
            "actions_unwound": action_receipts,
        }


# ============================================================================
# Component Separation: Proposer != Evaluator
# ============================================================================

class PolicyProposer:
    """
    Component 1: Proposes coordinated cross-subsystem transactions.
    Has NO authority to evaluate or commit changes.
    """
    def __init__(self, proposer_id: str = "policy_brain_proposer") -> None:
        self.proposer_id = proposer_id

    def formulate_transaction(
        self,
        hypothesis_rationale: str,
        actions: List[SubsystemAction],
        target_metric: str,
        expected_gain: str,
        side_effects: str,
        falsifier: str,
    ) -> CoordinationTransaction:
        # Check against forbidden kernel patches
        for a in actions:
            if a.control_surface == ControlSurfaceTier.KERNEL_PATCH:
                raise EpistemicContractViolation(
                    "Violation of Non-Invasive Invariant: Kernel patches are forbidden when userspace actuation exists."
                )

        tx_id = f"tx_coord_{uuid.uuid4().hex[:6]}"
        return CoordinationTransaction(
            transaction_id=tx_id,
            hypothesis_rationale=hypothesis_rationale,
            actions=actions,
            target_metric=target_metric,
            expected_gain=expected_gain,
            side_effects=side_effects,
            falsifier=falsifier,
        )


class ValidationEvaluator:
    """
    Component 2: Completely independent audit component.
    Must NOT be the same instance as PolicyProposer.
    Compares baseline vs post-change metrics against the falsifier.
    """
    def __init__(self, evaluator_id: str = "independent_system_auditor") -> None:
        self.evaluator_id = evaluator_id

    def audit_transaction(
        self,
        transaction: CoordinationTransaction,
        proposer: PolicyProposer,
        baseline: float,
        post_change: float,
        falsifier_triggered: bool,
    ) -> Dict[str, Any]:
        # Physical component separation
        if self is proposer or self.evaluator_id == proposer.proposer_id:
            raise EpistemicContractViolation(
                f"Anti-Self-Certification Component Violation: Evaluator cannot be the same component as Proposer."
            )

        transaction.baseline_metric = baseline
        transaction.post_change_metric = post_change

        if falsifier_triggered:
            return {
                "verdict": "REJECT_AND_ROLLBACK",
                "reason": f"Falsifier triggered: {transaction.falsifier}",
                "should_rollback": True,
            }

        delta = post_change - baseline
        is_improvement = post_change < baseline
        if is_improvement:
            return {
                "verdict": "APPROVE_AND_COMMIT",
                "reason": f"Empirical improvement confirmed: baseline={baseline}, post={post_change}, delta={delta:.3f}",
                "should_rollback": False,
            }

        return {
            "verdict": "REJECT_AND_ROLLBACK",
            "reason": f"Regression or neutral outcome: baseline={baseline}, post={post_change}",
            "should_rollback": True,
        }


# ============================================================================
# Central Policy Engine
# ============================================================================

class KrystalPolicyEngine:
    """
    The orchestrating coordinator:
    - Hosts the Proposer and Evaluator as distinct components.
    - Uses platform-specific Capability Provider (Linux or Windows).
    - Maintains the TransactionalChangeStack (one unvalidated transaction at a time).
    """
    def __init__(
        self,
        provider: Optional[OSCapabilityProvider] = None,
        proposer: Optional[PolicyProposer] = None,
        evaluator: Optional[ValidationEvaluator] = None,
    ) -> None:
        self.provider = provider or WindowsCapabilityProvider()
        self.proposer = proposer or PolicyProposer(proposer_id="policy_brain_proposer")
        self.evaluator = evaluator or ValidationEvaluator(evaluator_id="independent_system_auditor")
        self.stack = TransactionalChangeStack()

        # Initialize thin capability adapters
        self.adapters: Dict[SubsystemType, CapabilityBasedThinAdapter] = {
            SubsystemType.CPU: CapabilityCPUAdapter(self.provider),
            SubsystemType.MEMORY: CapabilityMemoryAdapter(self.provider),
            SubsystemType.IO: CapabilityIOAdapter(self.provider),
            SubsystemType.ACCELERATOR: CapabilityAcceleratorAdapter(self.provider),
            SubsystemType.THERMAL: CapabilityThermalAdapter(self.provider),
        }

    def execute_transaction(
        self,
        transaction: CoordinationTransaction,
        simulated_baseline: float,
        simulated_post_change: float,
        falsifier_triggered: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes an atomic coordinated transaction:
        1. Measure Baseline.
        2. Begin Transaction (applies cross-subsystem actions atomically).
        3. Measure Post-Change.
        4. Independent ValidationEvaluator decides.
        5. Commit or LIFO Rollback.
        """
        # 1. Baseline
        baseline = simulated_baseline

        # 2. Begin transaction (enforces one unvalidated transaction invariant)
        self.stack.begin_transaction(transaction, self.adapters)

        # 3. Post-change
        post = simulated_post_change

        # 4. Evaluation
        audit_res = self.evaluator.audit_transaction(
            transaction=transaction,
            proposer=self.proposer,
            baseline=baseline,
            post_change=post,
            falsifier_triggered=falsifier_triggered,
        )

        # 5. Commit or Rollback
        if audit_res["should_rollback"]:
            rollback_res = self.stack.rollback_transaction(reason=audit_res["reason"])
            return {
                "outcome": "ROLLED_BACK",
                "transaction": transaction.to_dict(),
                "verdict": audit_res,
                "rollback": rollback_res,
            }
        else:
            self.stack.commit_transaction(transaction.transaction_id)
            return {
                "outcome": "COMMITTED",
                "transaction": transaction.to_dict(),
                "verdict": audit_res,
            }
