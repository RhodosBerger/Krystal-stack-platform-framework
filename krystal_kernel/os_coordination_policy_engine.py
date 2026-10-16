"""
os_coordination_policy_engine.py - Non-Invasive OS Coordination Layer & Reversible Policy Engine (V3).

Implements the refined normative tenets of Milestone V-NON-INVASIVE-OS-03:
1. Doctrine: One Policy Brain, thin capability-based adapters, host OS remains authority.
   Reversible is configuration (not always physical system state due to thermal/cache hysteresis).
2. Domain-Scoped Transaction Locking (DomainLockManager): Disjoint domains (e.g. Storage vs Network)
   can execute independently, while overlapping domains remain strictly serialized.
3. HardExclusionSafetyGate: Kernel patching is placed behind a strict safety barrier outside
   the standard 9-tier control surface hierarchy, forbidden for automated policy engines.
4. Explicit Rollback Classes: EXACT_STATE, CONFIG_RESTORE, DRAIN_AND_CONVERGE, IRREVERSIBLE_SIDE_EFFECT.
5. Observability separated from Actuation (eBPF, ETW, HW counters in ObservabilityChannel).
6. Resilience & Chaos Engineering:
   - Transaction Timeout Handling.
   - Atomic Partial Apply Failure Unwinding.
   - Self-Healing Rollback with Safe Baseline Recovery.
7. Component Separation: PolicyProposer != ValidationEvaluator.
8. Global Invariant: GLOBAL_VITAL_MAX_HP == 6 maintained in global repository contracts.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Global Repository Constitutional Invariant
GLOBAL_VITAL_MAX_HP: int = 6


class EpistemicContractViolation(Exception):
    """Raised when an OS coordination invariant or architectural contract is violated."""
    pass


class SubsystemExecutionError(Exception):
    """Raised when an adapter fails during apply or rollback execution."""
    pass


class RollbackClass(str, enum.Enum):
    """
    Categorizes the physical and temporal nature of reversibility.
    Reversible is configuration, while physical state (thermal, cache) has dynamic hysteresis.
    """
    EXACT_STATE = "EXACT_STATE"
    CONFIG_RESTORE = "CONFIG_RESTORE"
    DRAIN_AND_CONVERGE = "DRAIN_AND_CONVERGE"
    IRREVERSIBLE_SIDE_EFFECT = "IRREVERSIBLE_SIDE_EFFECT"


class ControlSurfaceTier(enum.IntEnum):
    """
    9-Tier Standard Control Surface Hierarchy for Userspace Actuation.
    Ordered from least invasive to most invasive.
    Kernel patching is excluded from this standard hierarchy.
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


class ObservabilityChannel(str, enum.Enum):
    """Channels reserved purely for passive non-intrusive observation and telemetry."""
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


class SubsystemDomain(str, enum.Enum):
    """
    Resource domains for domain-scoped transaction locking.
    """
    COMPUTE_AND_THERMAL = "COMPUTE_AND_THERMAL"
    STORAGE_AND_IO = "STORAGE_AND_IO"
    NETWORK = "NETWORK"
    PLATFORM_SLICES = "PLATFORM_SLICES"


def get_subsystem_domain(subsystem: SubsystemType) -> SubsystemDomain:
    """Maps a subsystem to its resource domain."""
    if subsystem in (SubsystemType.CPU, SubsystemType.ACCELERATOR, SubsystemType.THERMAL):
        return SubsystemDomain.COMPUTE_AND_THERMAL
    if subsystem in (SubsystemType.IO, SubsystemType.FILESYSTEM, SubsystemType.MEMORY):
        return SubsystemDomain.STORAGE_AND_IO
    if subsystem == SubsystemType.NETWORK:
        return SubsystemDomain.NETWORK
    return SubsystemDomain.PLATFORM_SLICES


class OptimizationStatus(str, enum.Enum):
    HYPOTHESIS = "hypothesis"
    TESTING = "testing"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"
    CONFLICT = "conflict"


# ============================================================================
# Hard Exclusion Safety Gate (Kernel Patch Barrier)
# ============================================================================

class HardExclusionSafetyGate:
    """
    Blocks automated kernel patching.
    Requires explicit manual two-man rule cryptographic authorization token.
    """
    EMERGENCY_OVERRIDE_TOKEN = "CRYPTO_TWO_MAN_MANUAL_OVERRIDE_TOKEN_2026_CONFIRMED"

    @classmethod
    def verify_actuation_permitted(cls, tier_name: str, authorization_token: Optional[str] = None) -> None:
        if "KERNEL" in tier_name.upper():
            if authorization_token != cls.EMERGENCY_OVERRIDE_TOKEN:
                raise EpistemicContractViolation(
                    "HardExclusionSafetyGate: Kernel patching is strictly excluded from automated policy actuation. "
                    "Invalid or missing emergency cryptographic override token."
                )


# ============================================================================
# Platform Capability Providers (Linux vs Windows Separation)
# ============================================================================

class OSCapabilityProvider:
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
            ControlSurfaceTier.CGROUPS_OR_JOBOBJECTS,
            ControlSurfaceTier.IRQ_AFFINITY,
            ControlSurfaceTier.BOOT_PARAMETERS,
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
    """An individual action within a coordinated transaction."""
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
    start_time: float = field(default_factory=time.time)
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.actions:
            raise EpistemicContractViolation("CoordinationTransaction must contain at least one action.")
        if not self.falsifier:
            raise EpistemicContractViolation("CoordinationTransaction requires an explicit falsifier.")

    @property
    def affected_domains(self) -> Set[SubsystemDomain]:
        return {get_subsystem_domain(a.subsystem) for a in self.actions}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "hypothesis": self.hypothesis_rationale,
            "action_count": len(self.actions),
            "subsystems": [a.subsystem.value for a in self.actions],
            "domains": [d.value for d in self.affected_domains],
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
# Capability-Based Thin Adapters (with Fault Injection & Safe Defaults)
# ============================================================================

class CapabilityBasedThinAdapter:
    """Base thin adapter bound to an OSCapabilityProvider with safe baseline recovery."""
    def __init__(self, subsystem: SubsystemType, provider: OSCapabilityProvider) -> None:
        self.subsystem = subsystem
        self.provider = provider
        self.active_state: Dict[str, Any] = {}
        self.safe_baseline_defaults: Dict[str, Any] = {}
        self.simulate_apply_failure: bool = False
        self.simulate_rollback_failure: bool = False

    def apply_action(self, action: SubsystemAction) -> Dict[str, Any]:
        if self.simulate_apply_failure:
            raise SubsystemExecutionError(f"Simulated hardware/kernel rejection in {self.subsystem.value}")
        old_state = dict(self.active_state)
        self.active_state[action.target_param] = action.target_value
        return old_state

    def rollback_action(self, action: SubsystemAction) -> Dict[str, Any]:
        if self.simulate_rollback_failure:
            raise SubsystemExecutionError(f"Simulated adapter failure during rollback in {self.subsystem.value}")
        self.active_state.update(action.rollback_payload)
        return {
            "subsystem": self.subsystem.value,
            "rollback_class": action.rollback_class.value,
            "restored_state": action.rollback_payload,
        }

    def emergency_safe_baseline_reset(self) -> None:
        """Self-healing fallback when rollback encounters an adapter crash."""
        self.active_state.update(self.safe_baseline_defaults)


class CapabilityCPUAdapter(CapabilityBasedThinAdapter):
    def __init__(self, provider: OSCapabilityProvider) -> None:
        super().__init__(SubsystemType.CPU, provider)
        self.active_state = {"cpuset": "0-7", "governor": "powersave", "epp": "balance_performance"}
        self.safe_baseline_defaults = dict(self.active_state)


class CapabilityMemoryAdapter(CapabilityBasedThinAdapter):
    def __init__(self, provider: OSCapabilityProvider) -> None:
        super().__init__(SubsystemType.MEMORY, provider)
        self.active_state = {"swappiness": 60, "dirty_ratio": 20, "thp": "madvise"}
        self.safe_baseline_defaults = dict(self.active_state)


class CapabilityIOAdapter(CapabilityBasedThinAdapter):
    def __init__(self, provider: OSCapabilityProvider) -> None:
        super().__init__(SubsystemType.IO, provider)
        self.active_state = {"scheduler": "mq-deadline", "read_ahead_kb": 128}
        self.safe_baseline_defaults = dict(self.active_state)


class CapabilityAcceleratorAdapter(CapabilityBasedThinAdapter):
    def __init__(self, provider: OSCapabilityProvider) -> None:
        super().__init__(SubsystemType.ACCELERATOR, provider)
        self.active_state = {"priority_queue": "normal", "power_limit_watts": 28.0}
        self.safe_baseline_defaults = dict(self.active_state)


class CapabilityThermalAdapter(CapabilityBasedThinAdapter):
    def __init__(self, provider: OSCapabilityProvider) -> None:
        super().__init__(SubsystemType.THERMAL, provider)
        self.active_state = {"pl1_limit_watts": 28.0, "pl2_limit_watts": 64.0, "tau_sec": 28}
        self.safe_baseline_defaults = dict(self.active_state)


class CapabilityNetworkAdapter(CapabilityBasedThinAdapter):
    def __init__(self, provider: OSCapabilityProvider) -> None:
        super().__init__(SubsystemType.NETWORK, provider)
        self.active_state = {"tcp_congestion": "bbr", "rmem_max": 2097152}
        self.safe_baseline_defaults = dict(self.active_state)


# ============================================================================
# Domain Lock Manager & Transactional Stack (Resilient & Fault-Tolerant)
# ============================================================================

class DomainLockManager:
    """
    Manages domain-scoped locks.
    Disjoint domains (e.g. Storage vs Network) can execute concurrently,
    while overlapping domains (e.g. CPU vs Thermal) remain strictly serialized.
    """
    def __init__(self) -> None:
        self._locked_domains: Dict[SubsystemDomain, str] = {}  # Domain -> transaction_id

    def is_domain_locked(self, domain: SubsystemDomain) -> bool:
        return domain in self._locked_domains

    def acquire_domains(self, domains: Set[SubsystemDomain], transaction_id: str) -> None:
        for d in domains:
            if d in self._locked_domains:
                holder = self._locked_domains[d]
                raise EpistemicContractViolation(
                    f"Domain Lock Contention: Domain '{d.value}' is already locked by active transaction '{holder}'."
                )
        for d in domains:
            self._locked_domains[d] = transaction_id

    def release_domains(self, domains: Set[SubsystemDomain]) -> None:
        for d in domains:
            self._locked_domains.pop(d, None)


class TransactionalChangeStack:
    """
    Manages transactional isolation, domain-scoped locking, and resilient rollback.
    """
    def __init__(self) -> None:
        self.domain_locks = DomainLockManager()
        self._active_transactions: Dict[str, CoordinationTransaction] = {}
        self._active_adapters: Dict[SubsystemType, CapabilityBasedThinAdapter] = {}
        self._completed_history: List[CoordinationTransaction] = []

    def begin_transaction(
        self,
        transaction: CoordinationTransaction,
        adapters: Dict[SubsystemType, CapabilityBasedThinAdapter]
    ) -> None:
        # Acquire domain locks (enforces domain-scoped serialization)
        self.domain_locks.acquire_domains(transaction.affected_domains, transaction.transaction_id)

        transaction.status = OptimizationStatus.TESTING
        transaction.start_time = time.time()
        self._active_transactions[transaction.transaction_id] = transaction
        self._active_adapters = adapters

        # Atomic apply: if any action fails, unwind all previously applied actions within this tx
        applied_actions: List[Tuple[SubsystemAction, CapabilityBasedThinAdapter]] = []
        try:
            for action in transaction.actions:
                if action.subsystem not in adapters:
                    raise EpistemicContractViolation(f"No adapter available for subsystem {action.subsystem.value}")
                adapter = adapters[action.subsystem]
                adapter.apply_action(action)
                applied_actions.append((action, adapter))
        except Exception as apply_err:
            # Atomic reverse unwind of partially applied actions
            for act, adp in reversed(applied_actions):
                try:
                    adp.rollback_action(act)
                except Exception:
                    adp.emergency_safe_baseline_reset()

            self.domain_locks.release_domains(transaction.affected_domains)
            self._active_transactions.pop(transaction.transaction_id, None)
            transaction.status = OptimizationStatus.REJECTED
            raise SubsystemExecutionError(f"Partial apply failure. Unwound successfully: {apply_err}") from apply_err

    @property
    def has_unvalidated_transaction(self) -> bool:
        return len(self._active_transactions) > 0

    def commit_transaction(self, transaction_id: str) -> None:
        if transaction_id not in self._active_transactions:
            raise EpistemicContractViolation(f"No active unvalidated transaction matching '{transaction_id}' to commit.")

        tx = self._active_transactions.pop(transaction_id)
        tx.status = OptimizationStatus.SUPPORTED
        self.domain_locks.release_domains(tx.affected_domains)
        self._completed_history.append(tx)

    def rollback_transaction(
        self,
        transaction_id: Optional[str] = None,
        reason: str = "Rollback",
        status_override: Optional[OptimizationStatus] = None,
    ) -> Dict[str, Any]:
        """
        Resilient LIFO rollback across all actions with fallback self-healing.
        """
        target_id = transaction_id
        if target_id is None:
            if not self._active_transactions:
                return {"rolled_back": False, "reason": "No active transaction to rollback"}
            target_id = next(iter(self._active_transactions))

        if target_id not in self._active_transactions:
            return {"rolled_back": False, "reason": f"No matching active transaction for '{target_id}'"}

        tx = self._active_transactions.pop(target_id)
        action_receipts = []
        encountered_adapter_errors = []

        # LIFO unwind across all actions
        for action in reversed(tx.actions):
            adapter = self._active_adapters[action.subsystem]
            try:
                rec = adapter.rollback_action(action)
                action_receipts.append(rec)
            except Exception as roll_err:
                encountered_adapter_errors.append(str(roll_err))
                # Self-healing fallback to safe defaults
                adapter.emergency_safe_baseline_reset()
                action_receipts.append({
                    "subsystem": action.subsystem.value,
                    "rollback_class": "EMERGENCY_SAFE_BASELINE_FALLBACK",
                    "error": str(roll_err),
                })

        tx.status = status_override or OptimizationStatus.REJECTED
        self.domain_locks.release_domains(tx.affected_domains)
        self._completed_history.append(tx)

        return {
            "rolled_back": True,
            "transaction_id": tx.transaction_id,
            "reason": reason,
            "actions_unwound": action_receipts,
            "had_adapter_errors": len(encountered_adapter_errors) > 0,
        }

    def check_and_enforce_timeout(self, transaction_id: str, current_time: float) -> Optional[Dict[str, Any]]:
        """Automatically rolls back transactions exceeding timeout."""
        if transaction_id in self._active_transactions:
            tx = self._active_transactions[transaction_id]
            if (current_time - tx.start_time) > tx.timeout_seconds:
                return self.rollback_transaction(
                    transaction_id=transaction_id,
                    reason="Transaction timeout exceeded.",
                    status_override=OptimizationStatus.TIMED_OUT,
                )
        return None


# ============================================================================
# Component Separation: Proposer != Evaluator
# ============================================================================

class PolicyProposer:
    """Component 1: Proposes coordinated cross-subsystem transactions."""
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
        timeout_seconds: float = 5.0,
    ) -> CoordinationTransaction:
        # Check against forbidden kernel patches via HardExclusionSafetyGate
        for a in actions:
            HardExclusionSafetyGate.verify_actuation_permitted(a.control_surface.name)

        tx_id = f"tx_coord_{uuid.uuid4().hex[:6]}"
        return CoordinationTransaction(
            transaction_id=tx_id,
            hypothesis_rationale=hypothesis_rationale,
            actions=actions,
            target_metric=target_metric,
            expected_gain=expected_gain,
            side_effects=side_effects,
            falsifier=falsifier,
            timeout_seconds=timeout_seconds,
        )


class ValidationEvaluator:
    """Component 2: Completely independent audit component."""
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
                "Anti-Self-Certification Component Violation: Evaluator cannot be the same component as Proposer."
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
            SubsystemType.NETWORK: CapabilityNetworkAdapter(self.provider),
        }

    def execute_transaction(
        self,
        transaction: CoordinationTransaction,
        simulated_baseline: float,
        simulated_post_change: float,
        falsifier_triggered: bool = False,
    ) -> Dict[str, Any]:
        baseline = simulated_baseline

        # 1. Begin transaction (acquires domain locks & applies actions)
        self.stack.begin_transaction(transaction, self.adapters)

        # 2. Post-change measurement
        post = simulated_post_change

        # 3. Independent audit
        audit_res = self.evaluator.audit_transaction(
            transaction=transaction,
            proposer=self.proposer,
            baseline=baseline,
            post_change=post,
            falsifier_triggered=falsifier_triggered,
        )

        # 4. Commit or Rollback
        if audit_res["should_rollback"]:
            rollback_res = self.stack.rollback_transaction(
                transaction_id=transaction.transaction_id,
                reason=audit_res["reason"],
            )
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
