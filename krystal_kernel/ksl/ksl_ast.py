"""
==============================================================================
KSL (KRYSTAL SYSTEMS LANGUAGE) — ABSTRACT SYNTAX TREE (AST)
==============================================================================
Constitutional Invariant:
    - VITAL_MAX_HP == 6
    - Invariant != Policy
    - Epistemic Type Lattice: Unknown < Predicted < Derived < Measured < Validated
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


class ASTNode:
    """Base class for all KSL AST nodes."""
    line: int = 0
    column: int = 0


@dataclass
class TypeAnnotation(ASTNode):
    name: str  # e.g., "Tensor", "pressure", "RuntimeTelemetry"
    epistemic_wrapper: Optional[str] = None  # "Measured", "Predicted", "Validated", "Derived"
    generics: List[TypeAnnotation] = field(default_factory=list)

    def to_string(self) -> str:
        if self.epistemic_wrapper:
            inner = self.name if not self.generics else f"{self.name}<{', '.join(g.to_string() for g in self.generics)}>"
            return f"{self.epistemic_wrapper}<{inner}>"
        if self.generics:
            return f"{self.name}<{', '.join(g.to_string() for g in self.generics)}>"
        return self.name


@dataclass
class DeviceGroupDecl(ASTNode):
    name: str
    devices: List[str]


@dataclass
class ConstraintDecl(ASTNode):
    name: str
    target_field: str
    operator: str  # "<", "<=", ">", ">=", "=="
    limit_value: float


@dataclass
class StateField(ASTNode):
    name: str
    field_type: str  # "pressure", "temperature", "energy"


@dataclass
class StateDecl(ASTNode):
    name: str
    fields: List[StateField]


@dataclass
class TransitionDecl(ASTNode):
    name: str
    before_state: str
    action_type: str
    predict_state: str
    after_state: str


@dataclass
class RingDecl(ASTNode):
    name: str
    element_type: str
    capacity: int
    frame_size_bytes: int


@dataclass
class Parameter(ASTNode):
    name: str
    param_type: TypeAnnotation


@dataclass
class Expression(ASTNode):
    pass


@dataclass
class IdentifierExpr(Expression):
    name: str


@dataclass
class LiteralExpr(Expression):
    value: Any
    literal_type: str  # "int", "float", "string", "bool"
    unit: Optional[str] = None


@dataclass
class BinaryExpr(Expression):
    left: Expression
    operator: str
    right: Expression


@dataclass
class ObserveExpr(Expression):
    state_name: str


@dataclass
class AllocExpr(Expression):
    """Represents a heap allocation attempt (e.g. malloc, new) which violates zeroalloc."""
    alloc_type: str
    size_expr: Optional[Expression] = None


@dataclass
class ActionCandidate(ASTNode):
    action_kind: str  # "HOLD", "BATCH_CHANGE", "WORKERS_CHANGE", "ROUTE", "PERFORMANCE"
    target: Optional[str] = None
    delta: int = 0


@dataclass
class PredictExpr(Expression):
    candidate_ref: str
    objectives: List[str]  # e.g., ["minimize latency", "minimize energy", "maximize useful_work"]
    required_constraints: List[str]  # e.g., ["safe_thermal", "safe_memory"]


@dataclass
class Statement(ASTNode):
    pass


@dataclass
class LetStmt(Statement):
    name: str
    type_annotation: Optional[TypeAnnotation]
    initializer: Expression


@dataclass
class CandidatesBlock(Statement):
    actions: List[ActionCandidate]


@dataclass
class CommitStmt(Statement):
    plan_expr: Expression


@dataclass
class ValidateBlock(Statement):
    target: str  # "transition" or specific identifier
    conditions: List[BinaryExpr]


@dataclass
class ZeroAllocBlock(Statement):
    label: str  # e.g. "hotpath"
    body: List[Statement]


@dataclass
class FunctionDecl(ASTNode):
    name: str
    parameters: List[Parameter]
    return_type: Optional[TypeAnnotation]
    latency_bound_ms: Optional[float]
    placement: Optional[str]  # e.g. "auto(compute_cluster)" or "gpu0"
    required_constraints: List[str]
    effects: List[str]  # e.g. ["compute", "memory(read)", "gpu"]
    body: List[Statement]


@dataclass
class ProgramNode(ASTNode):
    device_groups: List[DeviceGroupDecl] = field(default_factory=list)
    constraints: List[ConstraintDecl] = field(default_factory=list)
    states: List[StateDecl] = field(default_factory=list)
    transitions: List[TransitionDecl] = field(default_factory=list)
    rings: List[RingDecl] = field(default_factory=list)
    functions: List[FunctionDecl] = field(default_factory=list)
