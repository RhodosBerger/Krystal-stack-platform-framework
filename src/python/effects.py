"""
Effect and Capability System (Python)

Every component declares its effects and allowed capabilities,
enabling safe composition, auditability, and hot-reload.
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import time


class Effect(Enum):
    """Effect types that components can declare."""
    # I/O Effects
    READ_TELEMETRY = auto()
    WRITE_TELEMETRY = auto()
    READ_CONFIG = auto()
    WRITE_CONFIG = auto()
    FILE_READ = auto()
    FILE_WRITE = auto()
    NETWORK_READ = auto()
    NETWORK_WRITE = auto()

    # System Effects
    CPU_CONTROL = auto()
    GPU_CONTROL = auto()
    MEMORY_CONTROL = auto()
    THERMAL_CONTROL = auto()
    PROCESS_CONTROL = auto()

    # State Effects
    STATE_READ = auto()
    STATE_WRITE = auto()
    LOG_WRITE = auto()
    METRICS_EMIT = auto()

    # Scheduling Effects
    PRIORITY_CHANGE = auto()
    AFFINITY_CHANGE = auto()
    BOOST_CHANGE = auto()

    # Learning Effects
    EXPERIENCE_READ = auto()
    EXPERIENCE_WRITE = auto()
    POLICY_UPDATE = auto()

    # IPC Effects
    IPC_SEND = auto()
    IPC_RECEIVE = auto()
    DRIVER_CONTROL = auto()


@dataclass
class CapabilityConstraints:
    """Constraints for capability usage."""
    max_frequency_hz: Optional[float] = None
    max_magnitude: Optional[float] = None
    allowed_targets: Optional[List[str]] = None
    requires_approval: bool = False


@dataclass
class Capability:
    """Capability token granting permission for effects."""
    id: str
    effects: Set[Effect]
    constraints: CapabilityConstraints = field(default_factory=CapabilityConstraints)
    issued_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    revocable: bool = True


@dataclass
class EffectDeclaration:
    """Component effect declaration."""
    component_id: str
    required_effects: Set[Effect]
    optional_effects: Set[Effect] = field(default_factory=set)
    provided_effects: Set[Effect] = field(default_factory=set)
    invariants: List[str] = field(default_factory=list)


class EffectChecker:
    """Effect checker for safe composition."""

    def __init__(self):
        self.declarations: Dict[str, EffectDeclaration] = {}
        self.capabilities: Dict[str, List[Capability]] = {}
        self.audit_log: List[Dict[str, Any]] = []

    def register(self, decl: EffectDeclaration) -> None:
        """Register a component's effect declaration."""
        self.declarations[decl.component_id] = decl

    def grant_capability(self, component_id: str, cap: Capability) -> None:
        """Grant capability to a component."""
        if component_id not in self.capabilities:
            self.capabilities[component_id] = []
        self.capabilities[component_id].append(cap)

    def revoke_capability(self, component_id: str, cap_id: str) -> bool:
        """Revoke a capability."""
        if component_id in self.capabilities:
            original_len = len(self.capabilities[component_id])
            self.capabilities[component_id] = [
                c for c in self.capabilities[component_id]
                if c.id != cap_id or not c.revocable
            ]
            return len(self.capabilities[component_id]) < original_len
        return False

    def can_perform(self, component_id: str, effect: Effect) -> bool:
        """Check if component can perform effect."""
        caps = self.capabilities.get(component_id, [])
        now = time.time()
        return any(
            effect in c.effects and (c.expires_at is None or c.expires_at > now)
            for c in caps
        )

    def validate_composition(self, component_ids: List[str]) -> 'CompositionResult':
        """Validate composition of components."""
        errors = []
        warnings = []

        for comp_id in component_ids:
            decl = self.declarations.get(comp_id)
            if not decl:
                warnings.append(f"Component '{comp_id}' not registered")
                continue

            for effect in decl.required_effects:
                if not self.can_perform(comp_id, effect):
                    errors.append(
                        f"Component '{comp_id}' requires {effect.name} but lacks capability"
                    )

        return CompositionResult(
            errors=errors,
            warnings=warnings,
            valid=len(errors) == 0
        )

    def record_effect(self, component_id: str, effect: Effect, details: Optional[Dict] = None) -> None:
        """Record effect execution for audit."""
        self.audit_log.append({
            "timestamp": time.time(),
            "component": component_id,
            "effect": effect.name,
            "details": details or {},
        })

    def get_audit_trail(self, component_id: Optional[str] = None) -> List[Dict]:
        """Get audit trail, optionally filtered by component."""
        if component_id:
            return [e for e in self.audit_log if e["component"] == component_id]
        return self.audit_log.copy()


@dataclass
class CompositionResult:
    """Result of composition validation."""
    errors: List[str]
    warnings: List[str]
    valid: bool


# Guardian component declarations
GUARDIAN_DECLARATION = EffectDeclaration(
    component_id="guardian",
    required_effects={
        Effect.READ_TELEMETRY,
        Effect.STATE_READ,
        Effect.STATE_WRITE,
        Effect.LOG_WRITE,
        Effect.EXPERIENCE_READ,
        Effect.EXPERIENCE_WRITE,
    },
    optional_effects={
        Effect.NETWORK_READ,
        Effect.IPC_SEND,
        Effect.POLICY_UPDATE,
    },
    provided_effects={
        Effect.METRICS_EMIT,
    },
)

OPENVINO_DECLARATION = EffectDeclaration(
    component_id="openvino_bridge",
    required_effects={
        Effect.READ_CONFIG,
        Effect.GPU_CONTROL,
    },
    optional_effects={
        Effect.CPU_CONTROL,
    },
    provided_effects={
        Effect.METRICS_EMIT,
    },
)

THREAD_BOOST_DECLARATION = EffectDeclaration(
    component_id="thread_boost",
    required_effects={
        Effect.CPU_CONTROL,
        Effect.AFFINITY_CHANGE,
        Effect.PRIORITY_CHANGE,
    },
    optional_effects={
        Effect.MEMORY_CONTROL,
    },
)


def create_guardian_checker() -> EffectChecker:
    """Create effect checker with guardian capabilities."""
    checker = EffectChecker()

    # Register components
    checker.register(GUARDIAN_DECLARATION)
    checker.register(OPENVINO_DECLARATION)
    checker.register(THREAD_BOOST_DECLARATION)

    # Grant guardian capabilities
    checker.grant_capability("guardian", Capability(
        id="guardian-core",
        effects={
            Effect.READ_TELEMETRY, Effect.STATE_READ, Effect.STATE_WRITE,
            Effect.LOG_WRITE, Effect.EXPERIENCE_READ, Effect.EXPERIENCE_WRITE,
            Effect.METRICS_EMIT, Effect.IPC_SEND,
        },
    ))

    # Grant OpenVINO capabilities
    checker.grant_capability("openvino_bridge", Capability(
        id="openvino-core",
        effects={Effect.READ_CONFIG, Effect.GPU_CONTROL, Effect.CPU_CONTROL},
    ))

    # Grant thread boost capabilities
    checker.grant_capability("thread_boost", Capability(
        id="thread-boost-core",
        effects={
            Effect.CPU_CONTROL, Effect.AFFINITY_CHANGE,
            Effect.PRIORITY_CHANGE, Effect.MEMORY_CONTROL,
        },
    ))

    return checker
