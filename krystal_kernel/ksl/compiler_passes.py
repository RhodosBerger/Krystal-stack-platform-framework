"""
==============================================================================
KSL (KRYSTAL SYSTEMS LANGUAGE) — 6-PASS COMPILER & EPISTEMIC TYPE ENGINE
==============================================================================
Module: krystal_kernel/ksl/compiler_passes.py
Canonical Architecture:
    Pass 1: Epistemic Type Checker      (Provenance Lattice: Proposed -> Predicted -> Measured -> Validated)
    Pass 2: 3D Resource Budgeter        (Constrained Resource Accounting: Q = (t, m, e), CUD >= h_min)
    Pass 3: Authority & Reversibility   (Proposal != Authority, PreCommit for Irreversible)
    Pass 4: Temporal & Conflict Graph   (Temporal overlap, Computational Impedance Z_c, Gamma_c Analogy)
    Pass 5: Transition Verifier Gen     (AEP Closed Loop: Predicted vs Measured Residual)
    Pass 6: Machine Lowering            (KIR -> KµISA 64-bit Words -> KSAC / FPGA Mailbox)
==============================================================================
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple


# --- AST & REPREZENTÁCIA ZDROJOVÉHO KÓDU KSL ---

@dataclass
class KslVariable:
    name: str
    epistemic_wrapper: str  # "Proposed" | "Predicted" | "Measured" | "Validated" | "Const"
    inner_type: str         # "SystemState" | "Action" | "f32" | "u32"


@dataclass
class KslStatement:
    kind: str               # "OBSERVE" | "PREDICT" | "CANDIDATE" | "COMMIT" | "VALIDATE" | "ASSIGN"
    target_var: Optional[KslVariable] = None
    source_vars: List[str] = field(default_factory=list)
    expression: str = ""
    tolerance: Optional[float] = None


@dataclass
class KslOperationAST:
    name: str
    reversibility: str      # "reversible" | "compensatable" | "irreversible"
    domain_capability: str  # napr. "GPU_QUEUE_0", "CNC_SPINDLE_0", "FPGA_MAILBOX_0"
    lane: str               # "L0" | "L1" | "L2" | "L3"
    deadline_ns: float
    energy_budget_nj: float
    max_uma_pressure: float
    statements: List[KslStatement]
    compensation_block: Optional[str] = None


# --- VÝSTUPNÉ ŠTRUKTÚRY: KIR A KµISA ---

@dataclass
class KirInstruction:
    step_id: int
    kir_op: str
    epistemic_in: List[str]
    epistemic_out: Optional[str]
    cost_vector_ns_mem_nj: Tuple[int, float, int]
    guard_condition: str


def pack_kuisa_64bit(opcode_id: int, lane_id: int, target_reg: int, time_q_ns: int, energy_q_nj: int) -> str:
    """
    Zabalí KµISA inštrukciu do 64-bitového hex slova pre FPGA L0 Mailbox:
    [63:56] Opcode (8b) | [55:52] Lane (4b) | [51:40] Target/UMA (12b) | [39:20] Time ns (20b) | [19:0] Energy nJ (20b)
    """
    word = (
        ((opcode_id & 0xFF) << 56)
        | ((lane_id & 0x0F) << 52)
        | ((target_reg & 0xFFF) << 40)
        | ((time_q_ns & 0xFFFFF) << 20)
        | (energy_q_nj & 0xFFFFF)
    )
    return f"0x{word:016X}"


# --- 6-PRECHODOVÝ KOMPILÁTOR KSL ---

class Ksl6PassCompiler:
    """
    KSL 6-Pass Compiler: Translates declarative operations into verified KIR and KµISA 64-bit binary words.
    Enforces non-promotion, resource bounds, reversibility pre-commit, and computational impedance analogies.
    """
    OPCODE_MAP = {
        "ACQUIRE_TURN_TOKEN": 0x10,
        "SAMPLE_TELEMETRY_L3": 0x20,
        "EVAL_PREDICTOR_L4": 0x30,
        "CHECK_CONSTRAINTS_L2": 0x40,
        "COMMIT_CANDIDATE_L6": 0x50,
        "VERIFY_RESIDUAL_L8": 0x60,
        "EMIT_RECEIPT_RELEASE": 0x70,
    }
    LANE_MAP = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}

    def compile(self, ast: KslOperationAST) -> Dict[str, Any]:
        symbol_table: Dict[str, KslVariable] = {}
        pass_logs: Dict[str, str] = {}

        # ---------------------------------------------------------------------
        # PASS 1: Epistemic Type Checker (Mindset V: Proposed ↛ Measured ↛ Validated)
        # ---------------------------------------------------------------------
        has_validation = False
        for stmt in ast.statements:
            if stmt.target_var:
                symbol_table[stmt.target_var.name] = stmt.target_var

            if stmt.kind == "ASSIGN":
                if not stmt.source_vars:
                    raise TypeError("[Pass 1 Epistemic Error] ASSIGN statement requires a source variable.")
                src = symbol_table.get(stmt.source_vars[0])
                dst = stmt.target_var
                if src and dst and src.epistemic_wrapper != dst.epistemic_wrapper:
                    raise TypeError(
                        f"[Pass 1 Epistemic Error] Cannot assign {src.epistemic_wrapper}<{src.inner_type}> "
                        f"to {dst.epistemic_wrapper}<{dst.inner_type}> without explicit validation."
                    )

            elif stmt.kind == "PREDICT":
                if not stmt.source_vars:
                    raise TypeError("[Pass 1 Epistemic Error] 'predict' requires an input variable.")
                src = symbol_table.get(stmt.source_vars[0])
                if not src or src.epistemic_wrapper not in ("Measured", "Validated"):
                    raise TypeError("[Pass 1 Epistemic Error] 'predict' requires Measured<T> or Validated<T> input.")

            elif stmt.kind == "VALIDATE":
                if len(stmt.source_vars) < 2:
                    raise TypeError("[Pass 1 Epistemic Error] 'validate' requires (measured, predicted) arguments.")
                m_var = symbol_table.get(stmt.source_vars[0])
                p_var = symbol_table.get(stmt.source_vars[1])
                if not m_var or m_var.epistemic_wrapper != "Measured":
                    raise TypeError("[Pass 1 Epistemic Error] First arg of 'validate' must be Measured<T>.")
                if not p_var or p_var.epistemic_wrapper != "Predicted":
                    raise TypeError("[Pass 1 Epistemic Error] Second arg of 'validate' must be Predicted<T>.")
                has_validation = True

        pass_logs["Pass_1_Epistemic_Typing"] = "OK (Strict Provenance Lattice Verified)"

        # ---------------------------------------------------------------------
        # PASS 2: Constrained Resource Budgeting (Mindset I: Q = (t, m, e), CUD >= h_min)
        # ---------------------------------------------------------------------
        est_costs = {
            "OBSERVE": (50, 0.03, 12),
            "PREDICT": (120, 0.05, 35),
            "CANDIDATE": (80, 0.02, 18),
            "COMMIT": (600, 0.28, 260),
            "VALIDATE": (350, 0.08, 110),
            "ASSIGN": (10, 0.01, 2),
        }
        total_t = sum(est_costs.get(s.kind, (0, 0, 0))[0] for s in ast.statements)
        total_m = sum(est_costs.get(s.kind, (0, 0, 0))[1] for s in ast.statements)
        total_e = sum(est_costs.get(s.kind, (0, 0, 0))[2] for s in ast.statements)

        if total_t > ast.deadline_ns or total_e > ast.energy_budget_nj or total_m > ast.max_uma_pressure:
            raise ValueError(f"[Pass 2 Budget Breach] Est ({total_t}ns, {total_m:.2f}, {total_e}nJ) > Budget ({ast.deadline_ns}ns, {ast.max_uma_pressure}, {ast.energy_budget_nj}nJ)")

        cud_score = len(ast.statements) / max(1.0, total_e)
        pass_logs["Pass_2_Resource_Budgeting"] = (
            f"OK (t={total_t}/{int(ast.deadline_ns)}ns, m={total_m:.2f}/{ast.max_uma_pressure}, "
            f"e={total_e}/{int(ast.energy_budget_nj)}nJ, CUD={cud_score:.4f})"
        )

        # ---------------------------------------------------------------------
        # PASS 3: Authority & Reversibility Classification (Mindset IV: Proposal != Authority)
        # ---------------------------------------------------------------------
        if ast.reversibility == "compensatable" and not ast.compensation_block:
            raise ValueError("[Pass 3 Authority Error] Compensatable operation requires 'on_abort' block.")
        if ast.reversibility == "irreversible" and not has_validation:
            raise ValueError("[Pass 3 Authority Error] Irreversible operation requires PreCommit/PostCommit validation.")
        if not has_validation:
            raise ValueError("[Pass 3 Authority Error] Commit without validation receipt is forbidden.")

        pass_logs["Pass_3_Capability_Authority"] = (
            f"OK (Mode={ast.reversibility}, Domain={ast.domain_capability}, UnverifiedCommit=BLOCKED)"
        )

        # ---------------------------------------------------------------------
        # PASS 4: Temporal & Computational Impedance Scheduling (Mindset II & III)
        # ---------------------------------------------------------------------
        # Z_c = (QueueDepth * Latency) / EffectiveBandwidth
        # Gamma_c = (Z_load - Z_base) / (Z_load + Z_base)
        z_c = (1.0 * total_t) / max(0.01, (1.0 - ast.max_uma_pressure) * 1000.0)
        gamma_c = (z_c - 0.5) / (z_c + 0.5)
        pass_logs["Pass_4_Temporal_Scheduling"] = (
            f"OK (Lane={ast.lane}, ResourceSet={{{ast.domain_capability}}}, "
            f"Z_c={z_c:.3f}, Gamma_c={gamma_c:.3f} [COMPUTATIONAL-ANALOGY])"
        )

        # ---------------------------------------------------------------------
        # PASS 5: Transition Verification Synthesis (AEP Closed Loop)
        # ---------------------------------------------------------------------
        kir_stream: List[KirInstruction] = []
        kir_stream.append(KirInstruction(
            0, "ACQUIRE_TURN_TOKEN", [], "TurnToken", (40, 0.01, 10), f"uma.pressure < {ast.max_uma_pressure}"
        ))
        step = 1
        for s in ast.statements:
            c_vec = est_costs.get(s.kind, (50, 0.01, 10))
            if s.kind == "OBSERVE":
                t_inner = s.target_var.inner_type if s.target_var else "SystemState"
                kir_stream.append(KirInstruction(
                    step, "SAMPLE_TELEMETRY_L3", [], f"Measured<{t_inner}>", c_vec, "Delta_S_obs == 0"
                ))
            elif s.kind == "PREDICT":
                kir_stream.append(KirInstruction(
                    step, "EVAL_PREDICTOR_L4", ["Measured<SystemState>"], "Predicted<SystemState>", c_vec, "ModelCalibrated == True"
                ))
            elif s.kind == "CANDIDATE":
                kir_stream.append(KirInstruction(
                    step, "CHECK_CONSTRAINTS_L2", ["Predicted<SystemState>"], "CandidateProposal", c_vec, s.expression
                ))
            elif s.kind == "COMMIT":
                kir_stream.append(KirInstruction(
                    step, "COMMIT_CANDIDATE_L6", ["CandidateProposal", "TurnToken"], "AuthoritativeTransition", c_vec, "IPR == 1.0"
                ))
            elif s.kind == "VALIDATE":
                kir_stream.append(KirInstruction(
                    step, "VERIFY_RESIDUAL_L8", ["Measured<SystemState>", "Predicted<SystemState>"], "Validated<SystemState>", c_vec, f"|R_t| <= {s.tolerance}"
                ))
            step += 1

        kir_stream.append(KirInstruction(
            step, "EMIT_RECEIPT_RELEASE", ["Validated<SystemState>"], "AepReceipt", (60, 0.02, 15), "ReceiptCompleteness == 1.0"
        ))
        pass_logs["Pass_5_Transition_Verification"] = f"OK ({len(kir_stream)} KIR nodes synthesized)"

        # ---------------------------------------------------------------------
        # PASS 6: Machine Lowering (KIR -> KµISA 64-bit words)
        # ---------------------------------------------------------------------
        lane_num = self.LANE_MAP.get(ast.lane, 1)
        kuisa_words = []
        for k_inst in kir_stream:
            op_id = self.OPCODE_MAP.get(k_inst.kir_op, 0x00)
            t_ns, _, e_nj = k_inst.cost_vector_ns_mem_nj
            word_hex = pack_kuisa_64bit(op_id, lane_num, 0x04A, t_ns, e_nj)
            kuisa_words.append({
                "step": k_inst.step_id,
                "kir_op": k_inst.kir_op,
                "kuisa_64b": word_hex,
                "guard": k_inst.guard_condition
            })

        pass_logs["Pass_6_Machine_Lowering"] = f"OK ({len(kuisa_words)} KµISA 64-bit words emitted)"

        return {
            "operation": ast.name,
            "passes": pass_logs,
            "kuisa_binary_stream": kuisa_words
        }
