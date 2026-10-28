"""
==============================================================================
KSL (KRYSTAL SYSTEMS LANGUAGE) — 6-PASS COMPILER & EPISTEMIC TYPE ENGINE
==============================================================================
Module: krystal_kernel/ksl/compiler_passes.py
Canonical Architecture:
    Pass 1: Epistemic Type Checker      (Provenance Lattice: Proposed -> Predicted -> Measured -> Validated)
    Pass 2: 3D Resource Budgeter        (Constrained Resource Accounting: Q = (t, m, e), CUD >= h_min)
    Pass 3: Authority & Reversibility   (Proposal != Authority, Commit-After-Validate AEP Rule)
    Pass 4: Temporal & Conflict Graph   (Temporal overlap, Computational Impedance Z_c, Gamma_c Analogy)
    Pass 5: Transition Verifier Gen     (AEP Closed Loop: Execute -> Observe -> Validate -> Commit)
    Pass 6: Machine Lowering            (KIR -> KµISA 64-bit Words -> KSAC / FPGA Mailbox)
==============================================================================
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple


# --- EXPLICIT COMPILER EXCEPTIONS ---

class MachineEncodingOverflowError(ValueError):
    """Raised when an instruction field exceeds its allocated bit-width in KµISA."""
    pass


class UnknownOpcodeError(KeyError):
    """Raised when an unknown KIR operation cannot be mapped to a machine opcode (Fail-Closed)."""
    pass


class UnknownLaneError(KeyError):
    """Raised when an unknown execution lane is requested (Fail-Closed)."""
    pass


class EpistemicOrderingError(ValueError):
    """Raised when AEP execution sequence is violated (e.g., Commit before Validate)."""
    pass


# --- AST & REPREZENTÁCIA ZDROJOVÉHO KÓDU KSL ---

@dataclass
class KslVariable:
    name: str
    epistemic_wrapper: str  # "Proposed" | "Predicted" | "Measured" | "Validated" | "Const"
    inner_type: str         # "SystemState" | "Action" | "f32" | "u32"


@dataclass
class KslStatement:
    kind: str               # "OBSERVE" | "PREDICT" | "CANDIDATE" | "EXECUTE" | "VALIDATE" | "COMMIT" | "ASSIGN"
    target_var: Optional[KslVariable] = None
    source_vars: List[str] = field(default_factory=list)
    expression: str = ""
    tolerance: Optional[float] = None


@dataclass
class KslOperationAST:
    name: str
    reversibility: str      # "reversible" | "compensatable" | "irreversible"
    domain_capability: str  # e.g. "GPU_QUEUE_0", "CNC_SPINDLE_0", "FPGA_MAILBOX_0"
    lane: str               # "L0" | "L1" | "L2" | "L3"
    deadline_ns: float
    energy_budget_nj: float
    max_uma_pressure: float
    statements: List[KslStatement]
    useful_work_units: float = 1.0
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
    
    Fail-Closed Range Validation: raises MachineEncodingOverflowError on field overflow.
    """
    if not (0 <= opcode_id <= 0xFF):
        raise MachineEncodingOverflowError(f"Opcode ID {opcode_id} out of 8-bit range [0, 255]")
    if not (0 <= lane_id <= 0x0F):
        raise MachineEncodingOverflowError(f"Lane ID {lane_id} out of 4-bit range [0, 15]")
    if not (0 <= target_reg <= 0xFFF):
        raise MachineEncodingOverflowError(f"Target register {target_reg} out of 12-bit range [0, 4095]")
    if not (0 <= time_q_ns <= 0xFFFFF):
        raise MachineEncodingOverflowError(f"Time {time_q_ns} ns out of 20-bit range [0, 1048575]")
    if not (0 <= energy_q_nj <= 0xFFFFF):
        raise MachineEncodingOverflowError(f"Energy {energy_q_nj} nJ out of 20-bit range [0, 1048575]")

    word = (
        (opcode_id << 56)
        | (lane_id << 52)
        | (target_reg << 40)
        | (time_q_ns << 20)
        | energy_q_nj
    )
    return f"0x{word:016X}"


# --- 6-PRECHODOVÝ KOMPILÁTOR KSL ---

class Ksl6PassCompiler:
    """
    KSL 6-Pass Compiler: Translates declarative operations into verified KIR and KµISA 64-bit binary words.
    Enforces non-promotion, resource bounds, Commit-after-Validate dominance, and computational impedance analogies.
    """
    OPCODE_MAP = {
        "ACQUIRE_TURN_TOKEN": 0x10,
        "SAMPLE_TELEMETRY_L3": 0x20,
        "EVAL_PREDICTOR_L4": 0x30,
        "CHECK_CONSTRAINTS_L2": 0x40,
        "EXECUTE_CANDIDATE_L6": 0x50,
        "VERIFY_RESIDUAL_L8": 0x60,
        "COMMIT_VALIDATED_L8": 0x68,
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
        has_commit = False
        validation_step_idx = -1
        commit_step_idx = -1

        for idx, stmt in enumerate(ast.statements):
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
                validation_step_idx = idx

            elif stmt.kind == "COMMIT":
                has_commit = True
                commit_step_idx = idx

        # Control-Flow Dominance: Commit cannot precede Validation
        if has_commit and (not has_validation or commit_step_idx < validation_step_idx):
            raise EpistemicOrderingError(
                "[Pass 3 Authority Error] Commit cannot precede validation: "
                "AuthoritativeCommit requires prior Validated<T> receipt "
                "(AEP Contract Violation: Execute -> Observe -> Validate -> Commit)."
            )

        pass_logs["Pass_1_Epistemic_Typing"] = "OK (Strict Provenance Lattice Verified)"

        # ---------------------------------------------------------------------
        # PASS 2: Constrained Resource Budgeting (Mindset I: Q = (t, m, e), CUD >= h_min)
        # ---------------------------------------------------------------------
        est_costs = {
            "OBSERVE": (50, 0.03, 12),
            "PREDICT": (120, 0.05, 35),
            "CANDIDATE": (80, 0.02, 18),
            "EXECUTE": (500, 0.20, 200),
            "VALIDATE": (350, 0.08, 110),
            "COMMIT": (100, 0.08, 60),
            "ASSIGN": (10, 0.01, 2),
        }
        total_t = sum(est_costs.get(s.kind, (50, 0.01, 10))[0] for s in ast.statements)
        peak_m = max((est_costs.get(s.kind, (50, 0.01, 10))[1] for s in ast.statements), default=0.0)
        total_e = sum(est_costs.get(s.kind, (50, 0.01, 10))[2] for s in ast.statements)

        if total_t > ast.deadline_ns or total_e > ast.energy_budget_nj or peak_m > ast.max_uma_pressure:
            raise ValueError(
                f"[Pass 2 Budget Breach] Est (t={total_t}ns, peak_m={peak_m:.2f}, e={total_e}nJ) > "
                f"Budget (t={ast.deadline_ns}ns, m={ast.max_uma_pressure}, e={ast.energy_budget_nj}nJ)"
            )

        # CUD = Delta UsefulWork / Delta Energy
        useful_work = getattr(ast, "useful_work_units", float(len(ast.statements)))
        cud_score = useful_work / max(1.0, float(total_e))
        pass_logs["Pass_2_Resource_Budgeting"] = (
            f"OK (t={total_t}/{int(ast.deadline_ns)}ns, peak_m={peak_m:.2f}/{ast.max_uma_pressure}, "
            f"e={total_e}/{int(ast.energy_budget_nj)}nJ, CUD={cud_score:.6f})"
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
            f"OK (Mode={ast.reversibility}, Domain={ast.domain_capability}, "
            f"CommitAfterValidate=ENFORCED, UnverifiedCommit=BLOCKED)"
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
        # PASS 5: Transition Verification Synthesis (AEP Closed Loop: Execute -> Validate -> Commit)
        # ---------------------------------------------------------------------
        kir_stream: List[KirInstruction] = []
        kir_stream.append(KirInstruction(
            0, "ACQUIRE_TURN_TOKEN", [], "TurnToken", (40, 0.01, 10), f"uma.pressure < {ast.max_uma_pressure}"
        ))
        step = 1
        committed_in_stream = False

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
            elif s.kind == "EXECUTE":
                # Execute in candidate / uncommitted space (AEP Stage 2)
                kir_stream.append(KirInstruction(
                    step, "EXECUTE_CANDIDATE_L6", ["CandidateProposal", "TurnToken"], "ExecutionObservationCandidate", c_vec, "IPR == 1.0"
                ))
            elif s.kind == "VALIDATE":
                kir_stream.append(KirInstruction(
                    step, "VERIFY_RESIDUAL_L8", ["Measured<SystemState>", "Predicted<SystemState>"], "Validated<SystemState>", c_vec, f"|R_t| <= {s.tolerance}"
                ))
            elif s.kind == "COMMIT":
                # Authoritative commit committed ONLY after validation
                kir_stream.append(KirInstruction(
                    step, "COMMIT_VALIDATED_L8", ["Validated<SystemState>", "ExecutionObservationCandidate"], "AuthoritativeTransition", c_vec, "ResidualValid == True"
                ))
                committed_in_stream = True
            step += 1

        # If AST validated but did not have an explicit COMMIT statement, auto-synthesize commit
        if has_validation and not committed_in_stream:
            kir_stream.append(KirInstruction(
                step, "COMMIT_VALIDATED_L8", ["Validated<SystemState>"], "AuthoritativeTransition", (100, 0.08, 60), "ResidualValid == True"
            ))
            step += 1

        kir_stream.append(KirInstruction(
            step, "EMIT_RECEIPT_RELEASE", ["AuthoritativeTransition"], "AepReceipt", (60, 0.02, 15), "ReceiptCompleteness == 1.0"
        ))
        pass_logs["Pass_5_Transition_Verification"] = f"OK ({len(kir_stream)} KIR nodes synthesized, AEP Contract Enforced)"

        # ---------------------------------------------------------------------
        # PASS 6: Machine Lowering (KIR -> KµISA 64-bit words, Fail-Closed)
        # ---------------------------------------------------------------------
        if ast.lane not in self.LANE_MAP:
            raise UnknownLaneError(f"Unknown execution lane: '{ast.lane}'. Allowed: {list(self.LANE_MAP.keys())}")
        lane_num = self.LANE_MAP[ast.lane]

        kuisa_words = []
        for k_inst in kir_stream:
            if k_inst.kir_op not in self.OPCODE_MAP:
                raise UnknownOpcodeError(f"Unknown KIR operation: '{k_inst.kir_op}' cannot be lowered to KµISA.")
            op_id = self.OPCODE_MAP[k_inst.kir_op]
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
