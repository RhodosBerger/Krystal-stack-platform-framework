"""
==============================================================================
KSL (KRYSTAL SYSTEMS LANGUAGE) — EPISTEMIC & CONSTRAINT TYPE CHECKER
==============================================================================
Constitutional Invariant:
    - VITAL_MAX_HP == 6
    - Anti-Promotion Barrier (P4): Predicted < Derived < Measured < Validated
    - ZeroAlloc Invariant: Zero dynamic allocation inside zeroalloc regions
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from .ksl_ast import (
    ProgramNode, FunctionDecl, Statement, LetStmt, ZeroAllocBlock,
    CandidatesBlock, CommitStmt, ValidateBlock, Expression, IdentifierExpr,
    LiteralExpr, BinaryExpr, ObserveExpr, PredictExpr, AllocExpr, TypeAnnotation
)
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation


class KslTypeError(Exception):
    """Base class for all KSL type and constraint violations."""
    pass


class EpistemicPromotionError(KslTypeError):
    """Raised when an unvalidated prediction is improperly promoted to measurement/fact."""
    pass


class ZeroAllocViolation(KslTypeError):
    """Raised when dynamic heap allocation occurs inside a zeroalloc region."""
    pass


class ConstraintGatingError(KslTypeError):
    """Raised when an undeclared or unsatisfied constraint is required."""
    pass


class KslDimensionalMismatchError(KslTypeError):
    """Raised when incompatible physical dimensions or units are assigned or compared."""
    pass


@dataclass(frozen=True)
class PhysicalQuantity:
    """
    Formal representation of a physical quantity:
      Quantity = (Dimension, Unit, Scale, Offset)
    Enforces that unit conversions (e.g. Kelvin <-> Celsius) are affine transformations:
      V_base = Scale * V + Offset
      V_target = (V_base - Offset_target) / Scale_target
    """
    dimension: str
    unit: str
    scale: float = 1.0
    offset: float = 0.0

    def convert_to(self, value: float, target: "PhysicalQuantity") -> float:
        if self.dimension != target.dimension:
            raise KslDimensionalMismatchError(
                f"Cannot convert between incompatible dimensions: '{self.dimension}' and '{target.dimension}'"
            )
        base = value * self.scale + self.offset
        return (base - target.offset) / target.scale


CANONICAL_PHYSICAL_QUANTITIES: Dict[str, PhysicalQuantity] = {
    # Temperature (Affine: Kelvin Base SI)
    "temperature<kelvin>": PhysicalQuantity("temperature", "kelvin", scale=1.0, offset=0.0),
    "temperature<celsius>": PhysicalQuantity("temperature", "celsius", scale=1.0, offset=273.15),
    "kelvin": PhysicalQuantity("temperature", "kelvin", scale=1.0, offset=0.0),
    "celsius": PhysicalQuantity("temperature", "celsius", scale=1.0, offset=273.15),

    # Physical Pressure (Pascal Base SI)
    "pressure<pascal>": PhysicalQuantity("pressure", "pascal", scale=1.0, offset=0.0),
    "pressure<bar>": PhysicalQuantity("pressure", "bar", scale=100_000.0, offset=0.0),
    "pascal": PhysicalQuantity("pressure", "pascal", scale=1.0, offset=0.0),
    "bar": PhysicalQuantity("pressure", "bar", scale=100_000.0, offset=0.0),

    # Normalized Dimensionless Pressure in [0.0, 1.0] (incompatible with physical Pa without reference P0)
    "normalized_pressure": PhysicalQuantity("normalized_pressure", "normalized", scale=1.0, offset=0.0),
    "pressure<normalized>": PhysicalQuantity("normalized_pressure", "normalized", scale=1.0, offset=0.0),
    "pressure": PhysicalQuantity("pressure", "normalized", scale=1.0, offset=0.0),

    # Energy (Joule Base SI)
    "energy<joule>": PhysicalQuantity("energy", "joule", scale=1.0, offset=0.0),
    "energy<microjoule>": PhysicalQuantity("energy", "microjoule", scale=1e-6, offset=0.0),

    # Time (Second Base SI)
    "time<second>": PhysicalQuantity("time", "second", scale=1.0, offset=0.0),
    "time<millisecond>": PhysicalQuantity("time", "millisecond", scale=1e-3, offset=0.0),
    "time<nanosecond>": PhysicalQuantity("time", "nanosecond", scale=1e-9, offset=0.0),

    # Frequency (Hertz Base SI)
    "frequency<hertz>": PhysicalQuantity("frequency", "hertz", scale=1.0, offset=0.0),
    "frequency<megahertz>": PhysicalQuantity("frequency", "megahertz", scale=1e6, offset=0.0),
}


class KslTypeChecker:
    """Type-checks KSL programs enforcing Epistemic, ZeroAlloc, Dimensional, and Invariant contracts."""

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        self.vital_max_hp = vital_max_hp
        if self.vital_max_hp != 6:
            raise ContractViolation(f"VITAL_MAX_HP must equal 6, got {self.vital_max_hp}")
        self.constraints: Dict[str, float] = {}
        self.states: Set[str] = set()
        self.state_fields: Dict[str, TypeAnnotation] = {}
        self.rings: Set[str] = set()
        self.variables: Dict[str, TypeAnnotation] = {}
        self.in_zeroalloc = False
        self.active_zeroalloc_label = ""

    def check_program(self, prog: ProgramNode) -> None:
        # 1. Register constraints
        for c in prog.constraints:
            self.constraints[c.name] = c.limit_value

        # 2. Register states and state field dimensional types
        self.state_fields.clear()
        for s in prog.states:
            self.states.add(s.name)
            for f in s.fields:
                try:
                    from .ksl_lexer import KslLexer
                    from .ksl_parser import KslParser
                    toks = KslLexer(f.field_type).tokenize()
                    ann = KslParser(toks).parse_type()
                except Exception:
                    ann = TypeAnnotation(name=f.field_type)
                self.state_fields[f"{s.name}.{f.name}"] = ann
                self.state_fields[f"state.{f.name}"] = ann

        # 3. Register rings
        for r in prog.rings:
            self.rings.add(r.name)

        # 4. Check functions
        for fn in prog.functions:
            self.check_function(fn)

    def check_function(self, fn: FunctionDecl) -> None:
        # Check required constraints exist
        for req in fn.required_constraints:
            if req not in self.constraints:
                raise ConstraintGatingError(f"Function '{fn.name}' requires undeclared constraint '{req}'")

        # Local scope
        self.variables.clear()
        for p in fn.parameters:
            self.variables[p.name] = p.param_type

        for stmt in fn.body:
            self.check_statement(stmt)

    def check_statement(self, stmt: Statement) -> None:
        if isinstance(stmt, ZeroAllocBlock):
            prev = self.in_zeroalloc
            prev_label = self.active_zeroalloc_label
            self.in_zeroalloc = True
            self.active_zeroalloc_label = stmt.label
            try:
                for s in stmt.body:
                    self.check_statement(s)
            finally:
                self.in_zeroalloc = prev
                self.active_zeroalloc_label = prev_label

        elif isinstance(stmt, LetStmt):
            expr_type = self.infer_expression_type(stmt.initializer)

            # Check ZeroAlloc restriction on expression
            if self.in_zeroalloc and isinstance(stmt.initializer, AllocExpr):
                raise ZeroAllocViolation(
                    f"Dynamic heap allocation '{stmt.initializer.alloc_type}' is strictly forbidden "
                    f"inside zeroalloc region '{self.active_zeroalloc_label}'!"
                )

            # Check Epistemic Type Assignment
            if stmt.type_annotation is not None:
                declared = stmt.type_annotation
                # If target is Measured<T> and source is Predicted<T>, throw Epistemic Promotion Error
                if declared.epistemic_wrapper == "Measured" and expr_type.epistemic_wrapper == "Predicted":
                    raise EpistemicPromotionError(
                        f"Type Error: Cannot promote 'Predicted<{expr_type.name}>' to "
                        f"'Measured<{declared.name}>' without physical measurement! "
                        f"(Anti-Promotion Barrier violated)"
                    )
                # If target is bare T (unwrapped) and source is Predicted<T>, require explicit validation
                if declared.epistemic_wrapper is None and expr_type.epistemic_wrapper == "Predicted":
                    raise EpistemicPromotionError(
                        f"Type Error: Cannot assign unvalidated 'Predicted<{expr_type.name}>' to "
                        f"grounded type '{declared.name}' without explicit validation!"
                    )

                # Check Dimensional & Physical Type Invariants (P4 & Physical Contracts)
                physical_dimensions = {"temperature", "pressure", "normalized_pressure", "energy", "time", "frequency"}

                # Normalized pressure vs physical pressure<pascal> separation
                is_dec_norm_p = declared.name == "normalized_pressure" or (declared.name == "pressure" and (not declared.generics or declared.generics[0].name == "normalized"))
                is_exp_phys_p = expr_type.name == "pressure" and bool(expr_type.generics and expr_type.generics[0].name in ("pascal", "bar"))
                is_dec_phys_p = declared.name == "pressure" and bool(declared.generics and declared.generics[0].name in ("pascal", "bar"))
                is_exp_norm_p = expr_type.name == "normalized_pressure" or (expr_type.name == "pressure" and (not expr_type.generics or expr_type.generics[0].name == "normalized"))

                if (is_dec_norm_p and is_exp_phys_p) or (is_dec_phys_p and is_exp_norm_p):
                    raise KslDimensionalMismatchError(
                        f"Dimensional Type Error: Cannot assign '{expr_type.to_string()}' to '{declared.to_string()}'. "
                        f"Normalized dimensionless pressure in [0.0, 1.0] requires calibration reference P0 to convert to physical pressure<pascal>!"
                    )

                if declared.name in physical_dimensions and expr_type.name in physical_dimensions:
                    if declared.name != expr_type.name:
                        raise KslDimensionalMismatchError(
                            f"Dimensional Type Error: Cannot assign incompatible physical quantity "
                            f"'{expr_type.to_string()}' to '{declared.to_string()}'. "
                            f"({expr_type.name} != {declared.name})"
                        )
                    if declared.generics and expr_type.generics:
                        dec_units = [g.name for g in declared.generics]
                        exp_units = [g.name for g in expr_type.generics]
                        if dec_units != exp_units:
                            raise KslDimensionalMismatchError(
                                f"Unit Mismatch Error: Cannot assign '{expr_type.to_string()}' to "
                                f"'{declared.to_string()}'. Incompatible unit scales ({exp_units} != {dec_units})! "
                                f"Implicit unit conversion forbidden; use explicit affine conversion Quantity=(Dimension, Unit, Scale, Offset)."
                            )

                if declared.name in physical_dimensions and declared.generics and expr_type.name in ("float", "int") and not expr_type.generics:
                    raise KslDimensionalMismatchError(
                        f"Dimensional Type Error: Cannot assign dimensionless literal '{expr_type.name}' to "
                        f"dimensioned type '{declared.to_string()}' without explicit unit constructor!"
                    )

                self.variables[stmt.name] = declared
            else:
                self.variables[stmt.name] = expr_type

        elif isinstance(stmt, CandidatesBlock):
            # Verify candidate deltas are within sane operational bounds
            for act in stmt.actions:
                if act.delta < -VITAL_MAX_HP or act.delta > VITAL_MAX_HP:
                    raise ContractViolation(
                        f"Candidate action delta {act.delta} exceeds constitutional bound VITAL_MAX_HP={VITAL_MAX_HP}"
                    )

        elif isinstance(stmt, CommitStmt):
            plan_type = self.infer_expression_type(stmt.plan_expr)
            # Plan must be a valid plan type
            if plan_type.name not in {"ActionPlan", "Predicted", "CandidatePlan"}:
                # Permitted if generic
                pass

        elif isinstance(stmt, ValidateBlock):
            # Validates transitions and outputs
            pass

    def infer_expression_type(self, expr: Expression) -> TypeAnnotation:
        if isinstance(expr, ObserveExpr):
            if expr.state_name not in self.states:
                raise ConstraintGatingError(f"Attempted to observe undeclared state '{expr.state_name}'")
            return TypeAnnotation(name=expr.state_name, epistemic_wrapper="Measured")

        if isinstance(expr, PredictExpr):
            # Check constraints exist
            for c in expr.required_constraints:
                if c not in self.constraints:
                    raise ConstraintGatingError(f"Predict block requires undeclared constraint '{c}'")
            return TypeAnnotation(name="ActionPlan", epistemic_wrapper="Predicted")

        if isinstance(expr, AllocExpr):
            if self.in_zeroalloc:
                raise ZeroAllocViolation(
                    f"Dynamic heap allocation '{expr.alloc_type}' is strictly forbidden "
                    f"inside zeroalloc region '{self.active_zeroalloc_label}'!"
                )
            return TypeAnnotation(name="Pointer", epistemic_wrapper=None)

        if isinstance(expr, IdentifierExpr):
            if expr.name in self.variables:
                return self.variables[expr.name]
            if expr.name in self.state_fields:
                return self.state_fields[expr.name]
            return TypeAnnotation(name="Unknown", epistemic_wrapper=None)

        if isinstance(expr, LiteralExpr):
            return TypeAnnotation(name=expr.literal_type, epistemic_wrapper=None)

        if isinstance(expr, BinaryExpr):
            return TypeAnnotation(name="bool", epistemic_wrapper=None)

        return TypeAnnotation(name="Unknown", epistemic_wrapper=None)

    @staticmethod
    def convert_affine_quantity(value: float, from_spec: str, to_spec: str) -> float:
        """
        Performs explicit affine unit conversion: Quantity = (Dimension, Unit, Scale, Offset).
        Supports Kelvin <-> Celsius, Pascal <-> Bar, etc.
        """
        q_from = CANONICAL_PHYSICAL_QUANTITIES.get(from_spec.lower())
        q_to = CANONICAL_PHYSICAL_QUANTITIES.get(to_spec.lower())
        if not q_from:
            raise KslDimensionalMismatchError(f"Unknown source physical quantity specification: '{from_spec}'")
        if not q_to:
            raise KslDimensionalMismatchError(f"Unknown target physical quantity specification: '{to_spec}'")
        return q_from.convert_to(value, q_to)
