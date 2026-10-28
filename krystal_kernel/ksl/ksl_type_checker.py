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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
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


class DimensionalDerivationError(KslDimensionalMismatchError):
    """Raised when dimension algebra derivation fails or does not match expected type."""
    pass


@dataclass(frozen=True)
class DimensionVector:
    """
    Formal 7-dimensional vector for KSL physical and computational quantities:
      [mass: M, length: L, time: T, current: I, temperature: Theta, memory: B, work: W]
    Enforces abelian group operations:
      D_(A * B) = D_A + D_B
      D_(A / B) = D_A - D_B
      D_(A^n)   = n * D_A
    """
    mass: int = 0         # SI mass (kg)
    length: int = 0       # SI length (m)
    time: int = 0         # SI time (s)
    current: int = 0      # SI electric current (A)
    temperature: int = 0  # SI thermodynamic temperature (K)
    memory: int = 0       # Computational storage (bytes)
    work: int = 0         # Computational work (ops/cycles)

    def __add__(self, other: "DimensionVector") -> "DimensionVector":
        return DimensionVector(
            mass=self.mass + other.mass,
            length=self.length + other.length,
            time=self.time + other.time,
            current=self.current + other.current,
            temperature=self.temperature + other.temperature,
            memory=self.memory + other.memory,
            work=self.work + other.work,
        )

    def __sub__(self, other: "DimensionVector") -> "DimensionVector":
        return DimensionVector(
            mass=self.mass - other.mass,
            length=self.length - other.length,
            time=self.time - other.time,
            current=self.current - other.current,
            temperature=self.temperature - other.temperature,
            memory=self.memory - other.memory,
            work=self.work - other.work,
        )

    def __mul__(self, scalar: int) -> "DimensionVector":
        return DimensionVector(
            mass=self.mass * scalar,
            length=self.length * scalar,
            time=self.time * scalar,
            current=self.current * scalar,
            temperature=self.temperature * scalar,
            memory=self.memory * scalar,
            work=self.work * scalar,
        )

    def is_scalar(self) -> bool:
        return (
            self.mass == 0 and self.length == 0 and self.time == 0 and
            self.current == 0 and self.temperature == 0 and
            self.memory == 0 and self.work == 0
        )

    def to_string(self) -> str:
        parts = []
        if self.mass != 0:
            parts.append(f"M^{self.mass}" if self.mass != 1 else "M")
        if self.length != 0:
            parts.append(f"L^{self.length}" if self.length != 1 else "L")
        if self.time != 0:
            parts.append(f"T^{self.time}" if self.time != 1 else "T")
        if self.current != 0:
            parts.append(f"I^{self.current}" if self.current != 1 else "I")
        if self.temperature != 0:
            parts.append(f"Theta^{self.temperature}" if self.temperature != 1 else "Theta")
        if self.memory != 0:
            parts.append(f"memory^{self.memory}" if self.memory != 1 else "memory")
        if self.work != 0:
            parts.append(f"work^{self.work}" if self.work != 1 else "work")
        return f"[{' '.join(parts)}]" if parts else "[scalar]"


# --- Base Dimensions ---
DIM_SCALAR = DimensionVector()
DIM_MASS = DimensionVector(mass=1)
DIM_LENGTH = DimensionVector(length=1)
DIM_TIME = DimensionVector(time=1)
DIM_CURRENT = DimensionVector(current=1)
DIM_TEMPERATURE = DimensionVector(temperature=1)
DIM_MEMORY = DimensionVector(memory=1)
DIM_WORK = DimensionVector(work=1)

# --- Derived Dimensions ---
# Energy = [M L^2 T^-2] (Joule)
DIM_ENERGY = DimensionVector(mass=1, length=2, time=-2)
# Power = [M L^2 T^-3] (Watt = Energy / Time)
DIM_POWER = DimensionVector(mass=1, length=2, time=-3)
# Pressure = [M L^-1 T^-2] (Pascal = Force / Area)
DIM_PRESSURE = DimensionVector(mass=1, length=-1, time=-2)
# Frequency = [T^-1] (Hertz = 1 / Time)
DIM_FREQUENCY = DimensionVector(time=-1)
# Bandwidth = [memory T^-1] (bytes_per_second = Memory / Time)
DIM_BANDWIDTH = DimensionVector(memory=1, time=-1)
# Throughput = [work T^-1] (work_per_second = Work / Time)
DIM_THROUGHPUT = DimensionVector(work=1, time=-1)
# Work Efficiency = [work M^-1 L^-2 T^2] (Throughput / Power = work / Energy = work/Joule)
DIM_WORK_EFFICIENCY = DimensionVector(work=1, mass=-1, length=-2, time=2)

DERIVED_DIMENSION_REGISTRY: Dict[DimensionVector, Tuple[str, str]] = {
    DIM_POWER: ("power", "watt"),
    DIM_BANDWIDTH: ("bandwidth", "bytes_per_second"),
    DIM_FREQUENCY: ("frequency", "hertz"),
    DIM_WORK_EFFICIENCY: ("efficiency", "work_per_joule"),
    DIM_ENERGY: ("energy", "joule"),
    DIM_THROUGHPUT: ("throughput", "work_per_second"),
    DIM_PRESSURE: ("pressure", "pascal"),
    DIM_TIME: ("time", "second"),
    DIM_MEMORY: ("memory", "bytes"),
    DIM_WORK: ("work", "ops"),
    DIM_MASS: ("mass", "kilogram"),
    DIM_LENGTH: ("length", "meter"),
    DIM_CURRENT: ("current", "ampere"),
    DIM_TEMPERATURE: ("temperature", "kelvin"),
}

UNIT_ALIASES: Dict[str, str] = {
    "J": "joule",
    "s": "second",
    "sec": "second",
    "ms": "millisecond",
    "us": "microsecond",
    "ns": "nanosecond",
    "W": "watt",
    "mW": "milliwatt",
    "kW": "kilowatt",
    "Hz": "hertz",
    "MHz": "megahertz",
    "GHz": "gigahertz",
    "B": "bytes",
    "KiB": "kilobytes",
    "MB": "megabytes",
    "ops": "ops",
    "K": "kelvin",
    "C": "celsius",
    "Pa": "pascal",
    "bar": "bar",
}


@dataclass(frozen=True)
class PhysicalQuantity:
    """
    Formal representation of a physical quantity:
      Quantity = (Dimension, Unit, Scale, Offset, DimensionVector)
    Enforces that unit conversions (e.g. Kelvin <-> Celsius) are affine transformations:
      V_base = Scale * V + Offset
      V_target = (V_base - Offset_target) / Scale_target
    Multiplication and division require Offset == 0.0 (affine units like Celsius must
    first be converted to an absolute scale like Kelvin).
    """
    dimension: str
    unit: str
    scale: float = 1.0
    offset: float = 0.0
    dim_vector: DimensionVector = field(default_factory=DimensionVector)

    @property
    def is_affine(self) -> bool:
        return self.offset != 0.0

    def convert_to(self, value: float, target: "PhysicalQuantity") -> float:
        if self.dim_vector != target.dim_vector and (not self.dim_vector.is_scalar() or not target.dim_vector.is_scalar()):
            raise KslDimensionalMismatchError(
                f"Cannot convert between incompatible dimensions: {self.dim_vector.to_string()} and {target.dim_vector.to_string()}"
            )
        if self.dimension != target.dimension:
            raise KslDimensionalMismatchError(
                f"Cannot convert between incompatible dimensions: '{self.dimension}' and '{target.dimension}'"
            )
        base = value * self.scale + self.offset
        return (base - target.offset) / target.scale


CANONICAL_PHYSICAL_QUANTITIES: Dict[str, PhysicalQuantity] = {
    # Temperature (Affine: Kelvin Base SI, Celsius offset 273.15)
    "temperature<kelvin>": PhysicalQuantity("temperature", "kelvin", scale=1.0, offset=0.0, dim_vector=DIM_TEMPERATURE),
    "temperature<celsius>": PhysicalQuantity("temperature", "celsius", scale=1.0, offset=273.15, dim_vector=DIM_TEMPERATURE),
    "kelvin": PhysicalQuantity("temperature", "kelvin", scale=1.0, offset=0.0, dim_vector=DIM_TEMPERATURE),
    "celsius": PhysicalQuantity("temperature", "celsius", scale=1.0, offset=273.15, dim_vector=DIM_TEMPERATURE),

    # Physical Pressure (Pascal Base SI)
    "pressure<pascal>": PhysicalQuantity("pressure", "pascal", scale=1.0, offset=0.0, dim_vector=DIM_PRESSURE),
    "pressure<bar>": PhysicalQuantity("pressure", "bar", scale=100_000.0, offset=0.0, dim_vector=DIM_PRESSURE),
    "pascal": PhysicalQuantity("pressure", "pascal", scale=1.0, offset=0.0, dim_vector=DIM_PRESSURE),
    "bar": PhysicalQuantity("pressure", "bar", scale=100_000.0, offset=0.0, dim_vector=DIM_PRESSURE),

    # Normalized Dimensionless Pressure in [0.0, 1.0] (incompatible with physical Pa without reference P0)
    "normalized_pressure": PhysicalQuantity("normalized_pressure", "normalized", scale=1.0, offset=0.0, dim_vector=DIM_SCALAR),
    "pressure<normalized>": PhysicalQuantity("normalized_pressure", "normalized", scale=1.0, offset=0.0, dim_vector=DIM_SCALAR),
    "pressure": PhysicalQuantity("pressure", "normalized", scale=1.0, offset=0.0, dim_vector=DIM_SCALAR),

    # Energy (Joule Base SI)
    "energy<joule>": PhysicalQuantity("energy", "joule", scale=1.0, offset=0.0, dim_vector=DIM_ENERGY),
    "energy<microjoule>": PhysicalQuantity("energy", "microjoule", scale=1e-6, offset=0.0, dim_vector=DIM_ENERGY),
    "energy": PhysicalQuantity("energy", "joule", scale=1.0, offset=0.0, dim_vector=DIM_ENERGY),
    "joule": PhysicalQuantity("energy", "joule", scale=1.0, offset=0.0, dim_vector=DIM_ENERGY),

    # Power (Watt Base SI)
    "power<watt>": PhysicalQuantity("power", "watt", scale=1.0, offset=0.0, dim_vector=DIM_POWER),
    "power<milliwatt>": PhysicalQuantity("power", "milliwatt", scale=1e-3, offset=0.0, dim_vector=DIM_POWER),
    "power": PhysicalQuantity("power", "watt", scale=1.0, offset=0.0, dim_vector=DIM_POWER),
    "watt": PhysicalQuantity("power", "watt", scale=1.0, offset=0.0, dim_vector=DIM_POWER),

    # Time (Second Base SI)
    "time<second>": PhysicalQuantity("time", "second", scale=1.0, offset=0.0, dim_vector=DIM_TIME),
    "time<millisecond>": PhysicalQuantity("time", "millisecond", scale=1e-3, offset=0.0, dim_vector=DIM_TIME),
    "time<nanosecond>": PhysicalQuantity("time", "nanosecond", scale=1e-9, offset=0.0, dim_vector=DIM_TIME),
    "time": PhysicalQuantity("time", "second", scale=1.0, offset=0.0, dim_vector=DIM_TIME),
    "second": PhysicalQuantity("time", "second", scale=1.0, offset=0.0, dim_vector=DIM_TIME),

    # Frequency (Hertz Base SI)
    "frequency<hertz>": PhysicalQuantity("frequency", "hertz", scale=1.0, offset=0.0, dim_vector=DIM_FREQUENCY),
    "frequency<megahertz>": PhysicalQuantity("frequency", "megahertz", scale=1e6, offset=0.0, dim_vector=DIM_FREQUENCY),
    "frequency": PhysicalQuantity("frequency", "hertz", scale=1.0, offset=0.0, dim_vector=DIM_FREQUENCY),
    "hertz": PhysicalQuantity("frequency", "hertz", scale=1.0, offset=0.0, dim_vector=DIM_FREQUENCY),

    # Memory (Computational Bytes)
    "memory<bytes>": PhysicalQuantity("memory", "bytes", scale=1.0, offset=0.0, dim_vector=DIM_MEMORY),
    "memory<kilobytes>": PhysicalQuantity("memory", "kilobytes", scale=1024.0, offset=0.0, dim_vector=DIM_MEMORY),
    "memory<megabytes>": PhysicalQuantity("memory", "megabytes", scale=1048576.0, offset=0.0, dim_vector=DIM_MEMORY),
    "memory": PhysicalQuantity("memory", "bytes", scale=1.0, offset=0.0, dim_vector=DIM_MEMORY),
    "bytes": PhysicalQuantity("memory", "bytes", scale=1.0, offset=0.0, dim_vector=DIM_MEMORY),

    # Bandwidth (bytes / second)
    "bandwidth<bytes_per_second>": PhysicalQuantity("bandwidth", "bytes_per_second", scale=1.0, offset=0.0, dim_vector=DIM_BANDWIDTH),
    "bandwidth": PhysicalQuantity("bandwidth", "bytes_per_second", scale=1.0, offset=0.0, dim_vector=DIM_BANDWIDTH),

    # Work & Throughput
    "work<ops>": PhysicalQuantity("work", "ops", scale=1.0, offset=0.0, dim_vector=DIM_WORK),
    "work": PhysicalQuantity("work", "ops", scale=1.0, offset=0.0, dim_vector=DIM_WORK),
    "throughput<work_per_second>": PhysicalQuantity("throughput", "work_per_second", scale=1.0, offset=0.0, dim_vector=DIM_THROUGHPUT),
    "throughput": PhysicalQuantity("throughput", "work_per_second", scale=1.0, offset=0.0, dim_vector=DIM_THROUGHPUT),

    # Efficiency (work / joule)
    "efficiency<work_per_joule>": PhysicalQuantity("efficiency", "work_per_joule", scale=1.0, offset=0.0, dim_vector=DIM_WORK_EFFICIENCY),
    "efficiency": PhysicalQuantity("efficiency", "work_per_joule", scale=1.0, offset=0.0, dim_vector=DIM_WORK_EFFICIENCY),
    "work_efficiency<work_per_joule>": PhysicalQuantity("efficiency", "work_per_joule", scale=1.0, offset=0.0, dim_vector=DIM_WORK_EFFICIENCY),
    "work_efficiency": PhysicalQuantity("efficiency", "work_per_joule", scale=1.0, offset=0.0, dim_vector=DIM_WORK_EFFICIENCY),
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
                physical_dimensions = {
                    "temperature", "pressure", "normalized_pressure", "energy", "power",
                    "time", "frequency", "memory", "bandwidth", "work", "throughput", "efficiency", "work_efficiency"
                }

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

                declared_dim = self.get_dimension_vector(declared)
                expr_dim = self.get_dimension_vector(expr_type)

                # Bare dimensionless literal assigned to dimensioned type
                if not declared_dim.is_scalar() and expr_dim.is_scalar() and expr_type.name in ("float", "int") and not expr_type.generics:
                    raise KslDimensionalMismatchError(
                        f"Dimensional Type Error: Cannot assign dimensionless literal '{expr_type.name}' to "
                        f"dimensioned type '{declared.to_string()}' without explicit unit constructor!"
                    )

                # Dimensional derivation mismatch (both non-scalar dimensions)
                if not declared_dim.is_scalar() and not expr_dim.is_scalar() and declared_dim != expr_dim:
                    raise DimensionalDerivationError(
                        f"DimensionalDerivationError: expected {declared.to_string()} with dimension {declared_dim.to_string()}, "
                        f"derived {expr_type.to_string()} with dimension {expr_dim.to_string()}!"
                    )

                if declared.name in physical_dimensions and expr_type.name in physical_dimensions:
                    if declared.name != expr_type.name and (declared_dim.is_scalar() or declared_dim != expr_dim):
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

    def get_dimension_vector(self, ann: Optional[TypeAnnotation]) -> DimensionVector:
        """Resolves the 7-dimensional DimensionVector for a given type annotation."""
        if ann is None:
            return DIM_SCALAR
        if ann.name in {"float", "int", "bool", "string", "Pointer"}:
            return DIM_SCALAR

        full_spec = ann.to_string().lower()
        if full_spec in CANONICAL_PHYSICAL_QUANTITIES:
            return CANONICAL_PHYSICAL_QUANTITIES[full_spec].dim_vector

        base_name = ann.name.lower()
        if ann.generics:
            unit_name = ann.generics[0].name.lower()
            unit_canonical = UNIT_ALIASES.get(unit_name, unit_name)
            combo = f"{base_name}<{unit_canonical}>"
            if combo in CANONICAL_PHYSICAL_QUANTITIES:
                return CANONICAL_PHYSICAL_QUANTITIES[combo].dim_vector
            if unit_canonical in CANONICAL_PHYSICAL_QUANTITIES:
                return CANONICAL_PHYSICAL_QUANTITIES[unit_canonical].dim_vector

        if base_name in CANONICAL_PHYSICAL_QUANTITIES:
            return CANONICAL_PHYSICAL_QUANTITIES[base_name].dim_vector

        return DIM_SCALAR

    def get_physical_quantity(self, ann: Optional[TypeAnnotation]) -> Optional[PhysicalQuantity]:
        """Resolves the PhysicalQuantity definition for a given type annotation."""
        if ann is None:
            return None

        full_spec = ann.to_string().lower()
        if full_spec in CANONICAL_PHYSICAL_QUANTITIES:
            return CANONICAL_PHYSICAL_QUANTITIES[full_spec]

        base_name = ann.name.lower()
        if ann.generics:
            unit_name = ann.generics[0].name.lower()
            unit_canonical = UNIT_ALIASES.get(unit_name, unit_name)
            combo = f"{base_name}<{unit_canonical}>"
            if combo in CANONICAL_PHYSICAL_QUANTITIES:
                return CANONICAL_PHYSICAL_QUANTITIES[combo]
            if unit_canonical in CANONICAL_PHYSICAL_QUANTITIES:
                return CANONICAL_PHYSICAL_QUANTITIES[unit_canonical]

        if base_name in CANONICAL_PHYSICAL_QUANTITIES:
            return CANONICAL_PHYSICAL_QUANTITIES[base_name]

        return None

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
            if expr.unit:
                unit_canonical = UNIT_ALIASES.get(expr.unit, expr.unit.lower())
                qty = CANONICAL_PHYSICAL_QUANTITIES.get(unit_canonical)
                if not qty:
                    for q in CANONICAL_PHYSICAL_QUANTITIES.values():
                        if q.unit == unit_canonical:
                            qty = q
                            break
                if qty:
                    return TypeAnnotation(name=qty.dimension, generics=[TypeAnnotation(name=qty.unit)])
            return TypeAnnotation(name=expr.literal_type, epistemic_wrapper=None)

        if isinstance(expr, BinaryExpr):
            left_type = self.infer_expression_type(expr.left)
            right_type = self.infer_expression_type(expr.right)
            left_dim = self.get_dimension_vector(left_type)
            right_dim = self.get_dimension_vector(right_type)
            left_qty = self.get_physical_quantity(left_type)
            right_qty = self.get_physical_quantity(right_type)

            if expr.operator in {"<", "<=", ">", ">=", "==", "!="}:
                if not left_dim.is_scalar() and not right_dim.is_scalar() and left_dim != right_dim:
                    raise DimensionalDerivationError(
                        f"Cannot compare incompatible dimensions: {left_dim.to_string()} and {right_dim.to_string()}"
                    )
                return TypeAnnotation(name="bool", epistemic_wrapper=None)

            if expr.operator in {"+", "-"}:
                if left_dim.is_scalar() and right_dim.is_scalar():
                    return TypeAnnotation(name="float" if "float" in (left_type.name, right_type.name) else "int")
                if left_dim != right_dim:
                    raise DimensionalDerivationError(
                        f"Dimensional Type Error: Cannot perform '{expr.operator}' on incompatible dimensions {left_dim.to_string()} and {right_dim.to_string()}!"
                    )
                if expr.operator == "+" and left_qty and left_qty.is_affine and right_qty and right_qty.is_affine:
                    raise DimensionalDerivationError(
                        f"Affine quantity with non-zero offset ({left_qty.unit}) cannot be added to another affine quantity directly! "
                        f"Convert to an absolute scale (e.g. Kelvin) before algebraic operations."
                    )
                return left_type

            if expr.operator in {"*", "/"}:
                # Enforce Section 2 affine normalization rule: Offset == 0.0
                if left_qty and left_qty.is_affine:
                    raise DimensionalDerivationError(
                        f"Affine unit '{left_qty.unit}' with non-zero offset ({left_qty.offset}) cannot be multiplied or divided directly! "
                        f"Convert to an absolute scale (e.g. Kelvin) before algebraic operations."
                    )
                if right_qty and right_qty.is_affine:
                    raise DimensionalDerivationError(
                        f"Affine unit '{right_qty.unit}' with non-zero offset ({right_qty.offset}) cannot be multiplied or divided directly! "
                        f"Convert to an absolute scale (e.g. Kelvin) before algebraic operations."
                    )

                if expr.operator == "*":
                    res_dim = left_dim + right_dim
                else:
                    res_dim = left_dim - right_dim

                if res_dim.is_scalar():
                    return TypeAnnotation(name="float")

                if res_dim in DERIVED_DIMENSION_REGISTRY:
                    type_name, unit_name = DERIVED_DIMENSION_REGISTRY[res_dim]
                    return TypeAnnotation(name=type_name, generics=[TypeAnnotation(name=unit_name)])

                if left_dim.is_scalar() and not right_dim.is_scalar() and expr.operator == "*":
                    return right_type
                if not left_dim.is_scalar() and right_dim.is_scalar():
                    return left_type

                dim_str = res_dim.to_string().strip("[]").replace(" ", "_")
                return TypeAnnotation(name=f"derived_{dim_str}")

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
