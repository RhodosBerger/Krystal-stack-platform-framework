"""
==============================================================================
KSL (KRYSTAL SYSTEMS LANGUAGE) — 5-LAYER INTERMEDIATE REPRESENTATION (KIR)
==============================================================================
KIR Hierarchy:
    1. KIR-S (Semantic IR):     Computation, Types, and Control Flow
    2. KIR-C (Constraint IR):   Invariants, Gating, Safety Predicates
    3. KIR-T (Transition IR):   State Vectors, Action Tuples, Predictors
    4. KIR-H (Placement IR):    Hardware Topology, Placement, Ring Buffers
    5. KIR-M (Machine IR):      Backend Code Emission & Instructions
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .ksl_ast import ProgramNode, FunctionDecl, ActionCandidate


@dataclass
class KirS_Instruction:
    op: str
    args: List[str]


@dataclass
class KirC_ConstraintGate:
    name: str
    target_metric: str
    threshold: float
    is_hard_invariant: bool = True


@dataclass
class KirT_TransitionNode:
    name: str
    candidates: List[Dict[str, Any]]
    predict_objectives: List[str]
    gated_constraints: List[str]


@dataclass
class KirH_PlacementPlan:
    function_name: str
    selected_device: str  # "CPU", "GPU", "NPU", "KSAC"
    latency_budget_ms: float
    is_zeroalloc: bool
    ring_buffer: Optional[str] = None


@dataclass
class KirM_MachineInstruction:
    opcode: str  # e.g. "LDS", "VEC6", "PRED", "GATE", "COMM", "VAL"
    operands: List[str]


@dataclass
class KslIntermediateRepresentation:
    """Consolidated 5-layer KSL IR."""
    kir_s_instructions: List[KirS_Instruction] = field(default_factory=list)
    kir_c_constraints: List[KirC_ConstraintGate] = field(default_factory=list)
    kir_t_transitions: List[KirT_TransitionNode] = field(default_factory=list)
    kir_h_placements: List[KirH_PlacementPlan] = field(default_factory=list)
    kir_m_machine_code: List[KirM_MachineInstruction] = field(default_factory=list)


class KslIrLowerer:
    """Lowers a validated KSL AST into the 5-layer KIR representation."""

    def lower(self, prog: ProgramNode) -> KslIntermediateRepresentation:
        ir = KslIntermediateRepresentation()

        # 1. Lower KIR-C (Constraints)
        for c in prog.constraints:
            ir.kir_c_constraints.append(
                KirC_ConstraintGate(
                    name=c.name,
                    target_metric=c.target_field,
                    threshold=c.limit_value,
                    is_hard_invariant=True
                )
            )

        # 2. Lower KIR-H (Placement & Memory)
        for fn in prog.functions:
            placement = fn.placement or "CPU"
            if "auto" in placement:
                # In adaptive mode, resolve to KSAC/Heterogeneous orchestration
                selected = "KSAC_HETEROGENEOUS"
            elif "gpu" in placement:
                selected = "GPU"
            elif "npu" in placement:
                selected = "NPU"
            else:
                selected = "CPU"

            ring_name = prog.rings[0].name if prog.rings else None
            ir.kir_h_placements.append(
                KirH_PlacementPlan(
                    function_name=fn.name,
                    selected_device=selected,
                    latency_budget_ms=fn.latency_bound_ms or 20.0,
                    is_zeroalloc=True,
                    ring_buffer=ring_name
                )
            )

        # 3. Lower KIR-S, KIR-T, and KIR-M
        for fn in prog.functions:
            ir.kir_s_instructions.append(KirS_Instruction(op="FUNCTION_ENTRY", args=[fn.name]))

            for stmt in fn.body:
                self._lower_statement(stmt, ir)

            ir.kir_s_instructions.append(KirS_Instruction(op="FUNCTION_EXIT", args=[fn.name]))

        return ir

    def _lower_statement(self, stmt: Any, ir: KslIntermediateRepresentation) -> None:
        from .ksl_ast import ZeroAllocBlock, LetStmt, CandidatesBlock, CommitStmt, ValidateBlock, ObserveExpr, PredictExpr

        if isinstance(stmt, ZeroAllocBlock):
            ir.kir_s_instructions.append(KirS_Instruction(op="ZEROALLOC_ENTER", args=[stmt.label]))
            ir.kir_m_machine_code.append(KirM_MachineInstruction(opcode="ENTER_CRITICAL_ZEROALLOC", operands=[stmt.label]))
            for s in stmt.body:
                self._lower_statement(s, ir)
            ir.kir_s_instructions.append(KirS_Instruction(op="ZEROALLOC_EXIT", args=[stmt.label]))
            ir.kir_m_machine_code.append(KirM_MachineInstruction(opcode="EXIT_CRITICAL_ZEROALLOC", operands=[stmt.label]))

        elif isinstance(stmt, LetStmt):
            ir.kir_s_instructions.append(KirS_Instruction(op="ASSIGN", args=[stmt.name]))

            if isinstance(stmt.initializer, ObserveExpr):
                ir.kir_s_instructions.append(KirS_Instruction(op="OBSERVE_HARDWARE", args=[stmt.initializer.state_name]))
                ir.kir_m_machine_code.append(KirM_MachineInstruction(opcode="LDS_VEC6", operands=["R0", stmt.initializer.state_name]))

            elif isinstance(stmt.initializer, PredictExpr):
                ir.kir_t_transitions.append(
                    KirT_TransitionNode(
                        name=stmt.name,
                        candidates=[],
                        predict_objectives=stmt.initializer.objectives,
                        gated_constraints=stmt.initializer.required_constraints
                    )
                )
                ir.kir_m_machine_code.append(
                    KirM_MachineInstruction(opcode="PRED_MULTI_OBJ", operands=stmt.initializer.objectives)
                )

        elif isinstance(stmt, CandidatesBlock):
            actions_data = [{"kind": a.action_kind, "target": a.target, "delta": a.delta} for a in stmt.actions]
            ir.kir_s_instructions.append(KirS_Instruction(op="CANDIDATES_PROPOSED", args=[str(len(stmt.actions))]))
            for idx, a in enumerate(stmt.actions):
                ir.kir_m_machine_code.append(
                    KirM_MachineInstruction(opcode="SET_CANDIDATE", operands=[f"A{idx}", a.action_kind, str(a.delta)])
                )

        elif isinstance(stmt, CommitStmt):
            ir.kir_s_instructions.append(KirS_Instruction(op="COMMIT_GATED", args=[]))
            ir.kir_m_machine_code.append(KirM_MachineInstruction(opcode="GATE_AND_COMMIT", operands=["R_PLAN", "DELTA_S_ZERO_ON_BREACH"]))

        elif isinstance(stmt, ValidateBlock):
            ir.kir_s_instructions.append(KirS_Instruction(op="VALIDATE_RECEIPT", args=[stmt.target]))
            ir.kir_m_machine_code.append(KirM_MachineInstruction(opcode="SEAL_RECEIPT_HMAC", operands=[stmt.target]))
