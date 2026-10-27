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
    ZeroAllocViolation, ConstraintGatingError
)
from .ksl_ir import (
    KslIntermediateRepresentation, KslIrLowerer,
    KirS_Instruction, KirC_ConstraintGate, KirT_TransitionNode,
    KirH_PlacementPlan, KirM_MachineInstruction
)
from .ksl_code_generator import KslCodeGenerator
from .ksl_compiler import KslCompiler, KslCompilationResult, GLOBAL_KSL_COMPILER

__all__ = [
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
]
