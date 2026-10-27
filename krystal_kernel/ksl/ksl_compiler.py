"""
==============================================================================
KSL (KRYSTAL SYSTEMS LANGUAGE) — COMPILER DRIVER
==============================================================================
Module: krystal_kernel/ksl/ksl_compiler.py
Constitutional Invariant:
    - VITAL_MAX_HP == 6
    - Invariant != Policy
    - Epistemic Type Safety: Anti-Promotion Barrier
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from .ksl_lexer import KslLexer, KslSyntaxError
from .ksl_parser import KslParser
from .ksl_type_checker import KslTypeChecker, KslTypeError, EpistemicPromotionError, ZeroAllocViolation
from .ksl_ir import KslIrLowerer, KslIntermediateRepresentation
from .ksl_code_generator import KslCodeGenerator
from .ksl_ast import ProgramNode


@dataclass
class KslCompilationResult:
    """Consolidated artifact result from compiling a KSL program."""
    success: bool
    c11_code: str
    kmuasm_code: str
    ast: ProgramNode
    ir: KslIntermediateRepresentation
    constraints_registered: int
    states_registered: int
    functions_compiled: int
    zeroalloc_verified: bool
    epistemic_safety_verified: bool
    error_message: Optional[str] = None
    vital_max_hp: int = VITAL_MAX_HP


class KslCompiler:
    """Top-level compiler driver for KSL (Krystal Systems Language) v0.1."""

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        self.vital_max_hp = vital_max_hp
        if self.vital_max_hp != 6:
            raise ContractViolation(f"VITAL_MAX_HP must equal 6, got {self.vital_max_hp}")
        self.type_checker = KslTypeChecker(vital_max_hp=vital_max_hp)
        self.ir_lowerer = KslIrLowerer()
        self.code_gen = KslCodeGenerator()

    def compile(self, source: str, mode: str = "adaptive") -> KslCompilationResult:
        """
        Compiles KSL source code through all 5 phases:
            1. Tokenize (Lexer)
            2. Parse (AST)
            3. Type Check & Invariant Verification (Semantic Analyzer)
            4. Lowering to 5-layer KIR
            5. Dual Code Generation (C11 + KµASM)
        """
        try:
            # 1. Lexical Analysis
            lexer = KslLexer(source)
            tokens = lexer.tokenize()

            # 2. Parsing into AST
            parser = KslParser(tokens)
            ast = parser.parse_program()

            # 3. Type Checking & Constitutional Verification
            self.type_checker.check_program(ast)

            # 4. Lower to 5-Layer KIR
            ir = self.ir_lowerer.lower(ast)

            # 5. Code Generation
            c11_code = self.code_gen.generate_c11(ast, ir)
            kmuasm_code = self.code_gen.generate_kmuasm(ast, ir)

            return KslCompilationResult(
                success=True,
                c11_code=c11_code,
                kmuasm_code=kmuasm_code,
                ast=ast,
                ir=ir,
                constraints_registered=len(ast.constraints),
                states_registered=len(ast.states),
                functions_compiled=len(ast.functions),
                zeroalloc_verified=True,
                epistemic_safety_verified=True,
                vital_max_hp=self.vital_max_hp
            )

        except (KslSyntaxError, KslTypeError, ContractViolation) as e:
            return KslCompilationResult(
                success=False,
                c11_code="",
                kmuasm_code="",
                ast=ProgramNode(),
                ir=KslIntermediateRepresentation(),
                constraints_registered=0,
                states_registered=0,
                functions_compiled=0,
                zeroalloc_verified=False,
                epistemic_safety_verified=False,
                error_message=str(e),
                vital_max_hp=self.vital_max_hp
            )


# Global Compiler Instance
GLOBAL_KSL_COMPILER = KslCompiler()
