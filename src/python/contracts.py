"""
Contracts and Proofs System (Python)

Pre/post/invariant contracts plus proof validators enforce correctness
at runtime, supporting automated validation and self-healing.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import functools


class CompareOp(Enum):
    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"


class ViolationAction(Enum):
    LOG = "log"
    WARN = "warn"
    ERROR = "error"
    ABORT = "abort"
    HEAL = "heal"


@dataclass
class Condition:
    """Contract condition."""
    id: str
    description: str
    check: Callable[[Dict[str, Any]], bool]
    on_violation: ViolationAction = ViolationAction.ERROR
    heal_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None


@dataclass
class Contract:
    """Contract definition for a function/component."""
    id: str
    name: str
    preconditions: List[Condition] = field(default_factory=list)
    postconditions: List[Condition] = field(default_factory=list)
    invariants: List[Condition] = field(default_factory=list)


@dataclass
class Violation:
    """Contract violation record."""
    condition_id: str
    description: str
    actual_value: Optional[str] = None
    expected: str = ""


@dataclass
class HealedViolation:
    """Healed violation record."""
    condition_id: str
    original_value: str
    healed_value: str
    strategy: str


@dataclass
class ValidationResult:
    """Contract validation result."""
    contract_id: str
    passed: bool
    violations: List[Violation] = field(default_factory=list)
    healed: List[HealedViolation] = field(default_factory=list)


class ContractValidator:
    """Contract validator with proof recording."""

    def __init__(self):
        self.contracts: Dict[str, Contract] = {}
        self.proof_log: List[Dict[str, Any]] = []

    def register(self, contract: Contract) -> None:
        """Register a contract."""
        self.contracts[contract.id] = contract

    def check_preconditions(self, contract_id: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate preconditions."""
        return self._validate(contract_id, context, "preconditions")

    def check_postconditions(self, contract_id: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate postconditions."""
        return self._validate(contract_id, context, "postconditions")

    def check_invariants(self, contract_id: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate invariants."""
        return self._validate(contract_id, context, "invariants")

    def _validate(
        self,
        contract_id: str,
        context: Dict[str, Any],
        phase: str
    ) -> ValidationResult:
        contract = self.contracts.get(contract_id)
        if not contract:
            return ValidationResult(
                contract_id=contract_id,
                passed=False,
                violations=[Violation(
                    condition_id="unknown",
                    description="Contract not found",
                    expected="registered contract"
                )]
            )

        conditions = getattr(contract, phase, [])
        violations = []
        healed = []

        for cond in conditions:
            try:
                passed = cond.check(context)
            except Exception as e:
                passed = False

            if not passed:
                if cond.on_violation == ViolationAction.HEAL and cond.heal_fn:
                    try:
                        healed_ctx = cond.heal_fn(context)
                        healed.append(HealedViolation(
                            condition_id=cond.id,
                            original_value=str(context),
                            healed_value=str(healed_ctx),
                            strategy="heal_fn"
                        ))
                        context.update(healed_ctx)
                    except Exception:
                        violations.append(Violation(
                            condition_id=cond.id,
                            description=cond.description,
                        ))
                else:
                    violations.append(Violation(
                        condition_id=cond.id,
                        description=cond.description,
                    ))

        result = ValidationResult(
            contract_id=contract_id,
            passed=len(violations) == 0,
            violations=violations,
            healed=healed,
        )

        # Record proof
        self.proof_log.append({
            "timestamp": time.time(),
            "contract_id": contract_id,
            "phase": phase,
            "passed": result.passed,
            "violation_count": len(violations),
        })

        return result

    def get_proof_log(self) -> List[Dict[str, Any]]:
        """Get proof audit log."""
        return self.proof_log.copy()


def contract(
    preconditions: Optional[List[Condition]] = None,
    postconditions: Optional[List[Condition]] = None,
    invariants: Optional[List[Condition]] = None,
):
    """Decorator to apply contracts to functions."""
    def decorator(fn: Callable) -> Callable:
        contract_id = f"{fn.__module__}.{fn.__name__}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            context = {"args": args, "kwargs": kwargs}

            # Check preconditions
            for cond in (preconditions or []):
                if not cond.check(context):
                    if cond.on_violation == ViolationAction.ABORT:
                        raise ContractViolationError(
                            f"Precondition failed: {cond.description}"
                        )

            # Execute function
            result = fn(*args, **kwargs)

            # Check postconditions
            context["result"] = result
            for cond in (postconditions or []):
                if not cond.check(context):
                    if cond.on_violation == ViolationAction.ABORT:
                        raise ContractViolationError(
                            f"Postcondition failed: {cond.description}"
                        )

            return result

        wrapper._contract_id = contract_id
        return wrapper

    return decorator


class ContractViolationError(Exception):
    """Raised when a contract is violated with ABORT action."""
    pass


# Predefined conditions for common checks
def range_check(field: str, min_val: float, max_val: float) -> Condition:
    """Create a range check condition."""
    return Condition(
        id=f"range_{field}",
        description=f"{field} must be in [{min_val}, {max_val}]",
        check=lambda ctx: min_val <= ctx.get(field, 0) <= max_val,
        on_violation=ViolationAction.HEAL,
        heal_fn=lambda ctx: {field: max(min_val, min(max_val, ctx.get(field, min_val)))},
    )


def not_null_check(field: str) -> Condition:
    """Create a not-null check condition."""
    return Condition(
        id=f"not_null_{field}",
        description=f"{field} must not be null",
        check=lambda ctx: ctx.get(field) is not None,
        on_violation=ViolationAction.ERROR,
    )


def compare_check(field: str, op: CompareOp, value: float) -> Condition:
    """Create a comparison check condition."""
    ops = {
        CompareOp.EQ: lambda a, b: abs(a - b) < 1e-10,
        CompareOp.NE: lambda a, b: abs(a - b) >= 1e-10,
        CompareOp.LT: lambda a, b: a < b,
        CompareOp.LE: lambda a, b: a <= b,
        CompareOp.GT: lambda a, b: a > b,
        CompareOp.GE: lambda a, b: a >= b,
    }
    return Condition(
        id=f"compare_{field}_{op.value}_{value}",
        description=f"{field} must be {op.value} {value}",
        check=lambda ctx: ops[op](ctx.get(field, 0), value),
        on_violation=ViolationAction.ERROR,
    )


# Standard contracts for GAMESA components
TELEMETRY_CONTRACT = Contract(
    id="telemetry_snapshot",
    name="Telemetry Snapshot Validation",
    invariants=[
        range_check("cpu_util", 0.0, 1.0),
        range_check("gpu_util", 0.0, 1.0),
        range_check("temp_cpu", 0, 120),
        range_check("temp_gpu", 0, 120),
        compare_check("frametime_ms", CompareOp.GT, 0),
    ],
)

DIRECTIVE_CONTRACT = Contract(
    id="directive_decision",
    name="Directive Decision Validation",
    preconditions=[
        not_null_check("action_type"),
    ],
    postconditions=[
        Condition(
            id="valid_confidence",
            description="Confidence must be in [0, 1]",
            check=lambda ctx: 0 <= ctx.get("result", {}).get("confidence", 0) <= 1,
        ),
    ],
)

SAFETY_CONTRACT = Contract(
    id="safety_check",
    name="Safety Guardrail Validation",
    invariants=[
        compare_check("temp_cpu", CompareOp.LT, 95),
        compare_check("temp_gpu", CompareOp.LT, 90),
    ],
)


def create_guardian_validator() -> ContractValidator:
    """Create validator with standard GAMESA contracts."""
    validator = ContractValidator()
    validator.register(TELEMETRY_CONTRACT)
    validator.register(DIRECTIVE_CONTRACT)
    validator.register(SAFETY_CONTRACT)
    return validator
