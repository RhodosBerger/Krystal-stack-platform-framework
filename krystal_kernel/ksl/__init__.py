"""
==============================================================================
KSL (KRYSTAL SYSTEMS LANGUAGE) — CORE COMPILER PACKAGE
==============================================================================
"""

from .ksl_ast import (
    ProgramNode, DeviceGroupDecl, ConstraintDecl, StateDecl, StateField,
    TransitionDecl, RingDecl, FunctionDecl, Parameter, TypeAnnotation,
    Statement, LetStmt, ZeroAllocBlock, CandidatesBlock, ActionCandidate,
    CommitStmt, ValidateBlock, Expression, IdentifierExpr, LiteralExpr,
    BinaryExpr, ObserveExpr, PredictExpr, AllocExpr
)
from .ksl_lexer import KslLexer, KslSyntaxError, Token
from .ksl_parser import KslParser
from .ksl_type_checker import (
    KslTypeChecker, KslTypeError, EpistemicPromotionError,
    ZeroAllocViolation, ConstraintGatingError, KslDimensionalMismatchError,
    DimensionalDerivationError, DimensionVector, DERIVED_DIMENSION_REGISTRY,
    PhysicalQuantity, CANONICAL_PHYSICAL_QUANTITIES, UNIT_ALIASES,
    DIM_SCALAR, DIM_ENERGY, DIM_POWER, DIM_TIME, DIM_FREQUENCY,
    DIM_MEMORY, DIM_BANDWIDTH, DIM_WORK, DIM_THROUGHPUT, DIM_WORK_EFFICIENCY
)
from .ksl_ir import (
    KslIntermediateRepresentation, KslIrLowerer,
    KirS_Instruction, KirC_ConstraintGate, KirT_TransitionNode,
    KirH_PlacementPlan, KirM_MachineInstruction
)
from .ksl_code_generator import KslCodeGenerator
from .ksl_compiler import KslCompiler, KslCompilationResult, GLOBAL_KSL_COMPILER
from .compiler_passes import (
    Ksl6PassCompiler,
    KslOperationAST,
    KslStatement,
    KslVariable,
    KirInstruction,
    pack_kuisa_64bit
)
from .ksl_ast_bridge import (
    KslOptimizationObjective,
    OptimizationStrategy,
    RuntimePolicyMetrics,
    MultiObjectiveConstraint,
    ParetoPolicySelector,
    GLOBAL_PARETO_POLICY_SELECTOR,
    MicroOpDAG,
    KslOperationalAstBridge,
    GLOBAL_KSL_OPERATIONAL_AST_BRIDGE,
)

__all__ = [
    "Ksl6PassCompiler",
    "KslOperationAST",
    "KslStatement",
    "KslVariable",
    "KirInstruction",
    "pack_kuisa_64bit",
    "ProgramNode",
    "DeviceGroupDecl",
    "ConstraintDecl",
    "StateDecl",
    "StateField",
    "TransitionDecl",
    "RingDecl",
    "FunctionDecl",
    "Parameter",
    "TypeAnnotation",
    "Statement",
    "LetStmt",
    "ZeroAllocBlock",
    "CandidatesBlock",
    "ActionCandidate",
    "CommitStmt",
    "ValidateBlock",
    "Expression",
    "IdentifierExpr",
    "LiteralExpr",
    "BinaryExpr",
    "ObserveExpr",
    "PredictExpr",
    "AllocExpr",
    "KslLexer",
    "KslSyntaxError",
    "Token",
    "KslParser",
    "KslTypeChecker",
    "KslTypeError",
    "EpistemicPromotionError",
    "ZeroAllocViolation",
    "ConstraintGatingError",
    "KslDimensionalMismatchError",
    "DimensionalDerivationError",
    "DimensionVector",
    "DERIVED_DIMENSION_REGISTRY",
    "UNIT_ALIASES",
    "DIM_SCALAR",
    "DIM_ENERGY",
    "DIM_POWER",
    "DIM_TIME",
    "DIM_FREQUENCY",
    "DIM_MEMORY",
    "DIM_BANDWIDTH",
    "DIM_WORK",
    "DIM_THROUGHPUT",
    "DIM_WORK_EFFICIENCY",
    "KslIntermediateRepresentation",
    "KslIrLowerer",
    "KirS_Instruction",
    "KirC_ConstraintGate",
    "KirT_TransitionNode",
    "KirH_PlacementPlan",
    "KirM_MachineInstruction",
    "KslCodeGenerator",
    "KslCompiler",
    "KslCompilationResult",
    "GLOBAL_KSL_COMPILER",
    "KslOptimizationObjective",
    "OptimizationStrategy",
    "RuntimePolicyMetrics",
    "MultiObjectiveConstraint",
    "ParetoPolicySelector",
    "GLOBAL_PARETO_POLICY_SELECTOR",
    "MicroOpDAG",
    "KslOperationalAstBridge",
    "GLOBAL_KSL_OPERATIONAL_AST_BRIDGE",
    "PhysicalQuantity",
    "CANONICAL_PHYSICAL_QUANTITIES",
]
