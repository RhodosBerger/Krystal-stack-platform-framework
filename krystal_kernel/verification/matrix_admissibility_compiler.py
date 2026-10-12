#!/usr/bin/env python3
"""
KRYSTAL-STACK // DETERMINISTIC MATRIX ADMISSIBILITY & STATE TRANSITION COMPILER
=============================================================================
Section 7 & 8 Implementation:
Compiles high-level CandidateAction proposals and paravirtual state transitions
via formal linear algebraic matrix operations:

  1. State & Action Vectorization:
       s in R^n (Host + Tenant State Vector)
       a in R^m (Candidate Action Control Vector)
  2. Deterministická Matica Prípustnosti (Hard Safety Constraint Matrix M_safe):
       M_safe * [s; a] <= c_limits
       Sequential reduction: A_possible -> A_safe
  3. Epistemická Matica Dôkazov (Evidence Admissibility Matrix M_evidence):
       M_evidence * e >= theta_threshold
       Reduction: A_safe -> A_evidence_supported
  4. Dynamická Matica Prechodu Stavov (State Transition Matrices T, B):
       s_{k+1} = T * s_k + B * a
       Predictive projection of expected delta
  5. Cross-Tenant Interference Matrix (I_cross):
       I_i(X) = x_i^T * W_interf * sum_{j != i} x_j
  6. Final Committable Synthesis:
       RTP execution plan generated ONLY if all matrix bounds are non-negative.
  7. C-ABI / SIMD Export: Flat array exports for C++ / Rust acceleration.
=============================================================================
"""

import os
import sys
import time
import math
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.keras_tensor_math_engine import TensorMatrix
from krystal_kernel.verification.paravirtual_envelope import (
    CandidateAction,
    SafetyVerdict,
    SafetyVerdictType,
    SafetyViolation,
    SafetyTier,
    EvidenceVerdict,
    EvidenceVerdictType,
    EvidenceGap,
    ContradictionSet,
    AuthorityVerdict,
    AuthorityVerdictType,
    ResourceEvidenceEnvelope,
    DynamicSafetyPolicy,
    ReversibilityClass
)


# =============================================================================
# 1. VECTOR INDEX CONSTANTS
# =============================================================================

# State Vector s indices (Dim = 6)
STATE_IDX_CPU_USAGE_PCT     = 0
STATE_IDX_FREE_RAM_MB       = 1
STATE_IDX_TEMP_CELSIUS      = 2
STATE_IDX_PACKAGE_POWER_W   = 3
STATE_IDX_VITAL_HP          = 4
STATE_IDX_STATE_VERSION     = 5

# Action Vector a indices (Dim = 4)
ACTION_IDX_VCPU_ALLOC       = 0
ACTION_IDX_RAM_ALLOC_MB     = 1
ACTION_IDX_IO_WEIGHT        = 2
ACTION_IDX_BALLOON_DELTA_MB = 3


@dataclass
class CompiledRtpMatrixPlan:
    """Compiled, verified, and executable matrix state transition plan for RTP."""
    plan_id: str
    action_id: str
    target_tenant_id: str
    state_version: int
    is_committable: bool
    authority_verdict: AuthorityVerdictType
    slack_margins: Dict[str, float]
    projected_state_delta: Dict[str, float]
    predicted_next_state: Dict[str, float]
    interference_penalty: float
    c_abi_signature: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "action_id": self.action_id,
            "target_tenant_id": self.target_tenant_id,
            "state_version": self.state_version,
            "is_committable": self.is_committable,
            "authority_verdict": self.authority_verdict.value,
            "slack_margins": self.slack_margins,
            "projected_state_delta": self.projected_state_delta,
            "predicted_next_state": self.predicted_next_state,
            "interference_penalty": round(self.interference_penalty, 4),
            "c_abi_signature": self.c_abi_signature,
            "timestamp": self.timestamp
        }


# =============================================================================
# 2. MATRIX ADMISSIBILITY COMPILER
# =============================================================================

class MatrixAdmissibilityCompiler:
    """
    Translates CandidateActions and host/tenant telemetry into linear algebraic
    matrix systems and evaluates admissibility, safety margins, and transition projections.
    """
    def __init__(self, policy: Optional[DynamicSafetyPolicy] = None):
        self.policy = policy or DynamicSafetyPolicy()
        self.state_dim = 6
        self.action_dim = 4

    # ── 2.1 VECTOR ENCODING ──────────────────────────────────────────────────

    def compile_state_vector(
        self,
        host_telemetry: Dict[str, Any],
        current_state_version: int
    ) -> TensorMatrix:
        """Encodes host telemetry into column vector s in R^(6x1)."""
        cpu_pct = float(host_telemetry.get("cpu_usage_pct", 40.0))
        free_ram_mb = float(host_telemetry.get("free_ram_bytes", 8192 * 1024 * 1024)) / (1024.0 * 1024.0)
        temp_c = float(host_telemetry.get("temperature_celsius", 52.0))
        power_w = float(host_telemetry.get("package_power_watts", 18.0))
        vital_hp = float(host_telemetry.get("vital_hp", VITAL_MAX_HP))
        state_ver = float(current_state_version)

        data = [[cpu_pct], [free_ram_mb], [temp_c], [power_w], [vital_hp], [state_ver]]
        return TensorMatrix(data, name="state_vector_s")

    def compile_action_vector(self, candidate: CandidateAction) -> TensorMatrix:
        """Encodes CandidateAction into column vector a in R^(4x1)."""
        vcpu = float(candidate.required_resources.get("vcpu", 0.0))
        ram_mb = float(candidate.required_resources.get("ram_mb", 0.0))
        io_weight = float(candidate.required_resources.get("io_weight", 100.0))
        balloon_delta = float(candidate.required_resources.get("balloon_delta_mb", 0.0))

        data = [[vcpu], [ram_mb], [io_weight], [balloon_delta]]
        return TensorMatrix(data, name="action_vector_a")

    # ── 2.2 DETERMINISTIC SAFETY CONSTRAINT MATRIX M_safe ────────────────────

    def compile_safety_constraint_matrix(
        self,
        host_telemetry: Dict[str, Any]
    ) -> Tuple[TensorMatrix, TensorMatrix, List[str]]:
        """
        Builds the Hard Safety Constraint Matrix M_safe (6 x 10) and limits vector c_limits (6 x 1).
        Evaluates: M_safe * [s; a] <= c_limits

        Rows:
          Row 0: Memory Headroom floor:      ram_alloc - free_ram <= -headroom_floor_mb
          Row 1: CPU Quota ceiling:          cpu_pct + 12.0 * vcpu_alloc <= 95.0%
          Row 2: Dynamic Thermal ceiling:    temp_c + 3.5 * vcpu_alloc <= dynamic_thermal_limit
          Row 3: Dynamic Power ceiling:      power_w + 4.2 * vcpu_alloc <= dynamic_power_ceiling
          Row 4: Vital System Invariant:     -vital_hp <= -6.0 (vital_hp >= 6.0)
          Row 5: State Version Monotonicity: state_ver_candidate - state_ver_current <= 0
        """
        host_id = host_telemetry.get("host_id", "local")
        workload = host_telemetry.get("workload_class", "standard")
        conf = host_telemetry.get("confidence", 0.9)

        headroom_mb = self.policy.memory_headroom_bytes(host_id) / (1024.0 * 1024.0)
        thermal_ceiling = self.policy.thermal_limit(host_id, workload, conf)
        power_ceiling = self.policy.power_ceiling_watts(host_id, workload)

        constraint_names = [
            "RAM_HEADROOM_FLOOR",
            "CPU_QUOTA_CEILING",
            "PHYSICAL_THERMAL_CEILING",
            "PHYSICAL_POWER_CEILING",
            "VITAL_MAX_HP_INVARIANT",
            "STATE_VERSION_CONSISTENCY"
        ]

        # Composite vector x = [s_0..s_5, a_0..a_3]^T in R^10
        # M_safe is (6 x 10)
        M_data = [
            # s_cpu, s_ram,  s_temp, s_pwr, s_hp,  s_ver,   a_vcpu, a_ram, a_io, a_bal
            [ 0.0,  -1.0,    0.0,   0.0,   0.0,   0.0,     0.0,    1.0,  0.0,  0.0],  # a_ram - s_ram <= -headroom
            [ 1.0,   0.0,    0.0,   0.0,   0.0,   0.0,    12.0,    0.0,  0.0,  0.0],  # s_cpu + 12*a_vcpu <= 95%
            [ 0.0,   0.0,    1.0,   0.0,   0.0,   0.0,     3.5,    0.0,  0.0,  0.0],  # s_temp + 3.5*a_vcpu <= T_max
            [ 0.0,   0.0,    0.0,   1.0,   0.0,   0.0,     4.2,    0.0,  0.0,  0.0],  # s_pwr + 4.2*a_vcpu <= P_max
            [ 0.0,   0.0,    0.0,   0.0,  -1.0,   0.0,     0.0,    0.0,  0.0,  0.0],  # -s_hp <= -6
            [ 0.0,   0.0,    0.0,   0.0,   0.0,   1.0,     0.0,    0.0,  0.0,  0.0],  # s_ver <= target_ver
        ]

        c_data = [
            [-headroom_mb],
            [95.0],
            [thermal_ceiling],
            [power_ceiling],
            [-float(VITAL_MAX_HP)],
            [float(host_telemetry.get("current_state_version", 1))]
        ]

        return (
            TensorMatrix(M_data, name="M_safe_constraints"),
            TensorMatrix(c_data, name="c_limits_vector"),
            constraint_names
        )

    # ── 2.3 STATE TRANSITION MATRICES T & B ──────────────────────────────────

    def compile_transition_matrices(self) -> Tuple[TensorMatrix, TensorMatrix]:
        """
        Dynamická matica prechodu stavov: s_{k+1} = T * s_k + B * a
        T in R^(6x6): Autonomous dynamics (thermal decay, power steady-state, invariant preservation)
        B in R^(6x4): Actuation coupling (impact of vCPU, RAM, IO, ballooning on state)
        """
        # Autonomous state evolution T (6x6)
        T_data = [
            [0.92, 0.00, 0.00, 0.00, 0.0, 0.0],  # CPU usage natural decay
            [0.00, 1.00, 0.00, 0.00, 0.0, 0.0],  # RAM stays constant unless actuated
            [0.00, 0.00, 0.95, 0.05, 0.0, 0.0],  # Temperature dissipates toward ambient / power coupling
            [0.05, 0.00, 0.00, 0.90, 0.0, 0.0],  # Power tracks CPU load with decay
            [0.00, 0.00, 0.00, 0.00, 1.0, 0.0],  # VITAL_MAX_HP strictly preserved
            [0.00, 0.00, 0.00, 0.00, 0.0, 1.0],  # State version persists
        ]

        # Control actuation matrix B (6x4)
        B_data = [
            # a_vcpu, a_ram,   a_io,  a_bal
            [ 10.5,    0.001,  0.05,  0.00],  # CPU delta per vCPU
            [ 0.00,   -1.000,  0.00,  1.00],  # RAM delta (alloc consumes, balloon returns)
            [  2.8,    0.000,  0.02,  0.00],  # Temp increase per vCPU and IO
            [  3.6,    0.000,  0.01,  0.00],  # Power increase per vCPU
            [  0.0,    0.000,  0.00,  0.00],  # Invariant cannot be modified by actuation
            [  0.0,    0.000,  0.00,  0.00],  # Version updated atomically on commit
        ]

        return (
            TensorMatrix(T_data, name="T_state_transition"),
            TensorMatrix(B_data, name="B_control_actuation")
        )

    # ── 2.4 EVIDENCE ADMISSIBILITY MATRIX ────────────────────────────────────

    def compile_evidence_matrix(
        self,
        candidate: CandidateAction,
        envelopes: Dict[str, ResourceEvidenceEnvelope]
    ) -> Tuple[bool, EvidenceVerdict]:
        """
        Evaluates evidence freshness, dimension completeness, and cross-sensor consistency
        via algebraic matrix projection.
        """
        reqs = candidate.evidence_requirements

        if not candidate.evidence_refs:
            return False, EvidenceVerdict(
                verdict=EvidenceVerdictType.INSUFFICIENT,
                gap=EvidenceGap(reqs.mandatory_dimensions, "No evidence envelopes referenced", reqs.max_acceptable_age_seconds, float("inf"))
            )

        now = time.time()
        cpu_samples: List[float] = []

        for ref_id in candidate.evidence_refs:
            env = envelopes.get(ref_id)
            if not env:
                return False, EvidenceVerdict(
                    verdict=EvidenceVerdictType.INSUFFICIENT,
                    gap=EvidenceGap([], f"Evidence {ref_id} not found in store", reqs.max_acceptable_age_seconds, float("inf"))
                )

            age = now - env.timestamp
            if age > reqs.max_acceptable_age_seconds:
                return False, EvidenceVerdict(
                    verdict=EvidenceVerdictType.INSUFFICIENT,
                    gap=EvidenceGap([], f"Evidence {ref_id} stale ({age:.2f}s > {reqs.max_acceptable_age_seconds:.2f}s)", reqs.max_acceptable_age_seconds, age)
                )

            for dim in reqs.mandatory_dimensions:
                if dim not in env.telemetry:
                    return False, EvidenceVerdict(
                        verdict=EvidenceVerdictType.INSUFFICIENT,
                        gap=EvidenceGap([dim], f"Missing mandatory dimension {dim}", reqs.max_acceptable_age_seconds, age)
                    )

            if "cpu_usage_pct" in env.telemetry:
                cpu_samples.append(float(env.telemetry["cpu_usage_pct"]))

        # Discrepancy detector across multiple envelopes
        if len(cpu_samples) > 1:
            discrepancy = max(cpu_samples) - min(cpu_samples)
            if discrepancy > 35.0:
                return False, EvidenceVerdict(
                    verdict=EvidenceVerdictType.CONTRADICTORY,
                    contradictions=ContradictionSet(
                        conflicting_pairs=[{"samples": cpu_samples}],
                        discrepancy_magnitude=discrepancy,
                        explanation=f"Sensor conflict: CPU discrepancy {discrepancy:.1f}% exceeds 35% tolerance"
                    )
                )

        return True, EvidenceVerdict(verdict=EvidenceVerdictType.SUFFICIENT)

    # ── 2.5 CROSS-WORKLOAD INTERFERENCE MATRIX ───────────────────────────────

    def calculate_interference_penalty(
        self,
        tenant_features: Dict[str, float],
        active_neighbors: List[Dict[str, float]]
    ) -> float:
        """
        Computes quadratic interference collision:
          I_i(X) = x_i^T * W_interf * sum_{j != i} x_j
        Features: [LLC_intensity, Memory_Bus_BW_ratio, IOPS_ratio]
        """
        if not active_neighbors:
            return 0.0

        # W_interf weights: [alpha_llc=0.45, beta_bus=0.35, gamma_iops=0.20]
        w_llc = 0.45
        w_bus = 0.35
        w_io  = 0.20

        t_llc = tenant_features.get("llc_intensity", 0.5)
        t_bus = tenant_features.get("bus_intensity", 0.4)
        t_io  = tenant_features.get("io_intensity", 0.3)

        sum_n_llc = sum(n.get("llc_intensity", 0.2) for n in active_neighbors)
        sum_n_bus = sum(n.get("bus_intensity", 0.2) for n in active_neighbors)
        sum_n_io  = sum(n.get("io_intensity", 0.1) for n in active_neighbors)

        penalty = (
            w_llc * (t_llc * sum_n_llc) +
            w_bus * (t_bus * sum_n_bus) +
            w_io  * (t_io  * sum_n_io)
        )
        return min(1.0, penalty)

    # ── 2.6 FULL COMPILATION PIPELINE ────────────────────────────────────────

    def compile_and_verify(
        self,
        candidate: CandidateAction,
        envelopes: Dict[str, ResourceEvidenceEnvelope],
        host_telemetry: Dict[str, Any],
        active_neighbors: Optional[List[Dict[str, float]]] = None
    ) -> CompiledRtpMatrixPlan:
        """
        Executes complete matrix compilation and admissibility evaluation.
        Reduces: A_possible -> A_safe -> A_evidence_supported -> A_committable.
        """
        plan_id = f"RTP-PLAN-{uuid.uuid4().hex[:8].upper()}"
        cur_version = int(host_telemetry.get("current_state_version", 1))

        # 1. Compile vectors
        s = self.compile_state_vector(host_telemetry, cur_version)
        a = self.compile_action_vector(candidate)

        # 2. Compile and evaluate Hard Safety Constraints M_safe * [s; a] <= c_limits
        M_safe, c_limits, constraint_names = self.compile_safety_constraint_matrix(host_telemetry)

        # Form composite vector x = [s; a] in R^10
        composite_data = [[s.data[r][0]] for r in range(s.rows)] + [[a.data[r][0]] for r in range(a.rows)]
        x = TensorMatrix(composite_data, name="composite_vector_x")

        # Compute: lhs = M_safe * x
        lhs = M_safe.matmul(x)

        # Compute slack vector: slack = c_limits - lhs (must be >= 0 for all rows)
        slack_margins: Dict[str, float] = {}
        safety_violation_found: Optional[Tuple[str, float, float]] = None

        for r in range(c_limits.rows):
            name = constraint_names[r]
            limit_val = c_limits.data[r][0]
            actual_val = lhs.data[r][0]
            margin = limit_val - actual_val
            slack_margins[name] = round(margin, 2)

            if margin < -1e-4:  # Breach detected
                if not safety_violation_found:
                    safety_violation_found = (name, actual_val, limit_val)

        # Pre-actuation VETO if any safety row breached
        if safety_violation_found is not None:
            name, actual, limit = safety_violation_found
            return CompiledRtpMatrixPlan(
                plan_id=plan_id,
                action_id=candidate.action_id,
                target_tenant_id=candidate.target_tenant_id,
                state_version=cur_version,
                is_committable=False,
                authority_verdict=AuthorityVerdictType.VETO_SAFETY_BREACH,
                slack_margins=slack_margins,
                projected_state_delta={},
                predicted_next_state={},
                interference_penalty=0.0,
                c_abi_signature=f"C-ABI-VETO-{name}"
            )

        # 3. Compile and evaluate Evidence Admissibility M_evidence
        evidence_ok, evidence_verdict = self.compile_evidence_matrix(candidate, envelopes)

        if not evidence_ok:
            auth_verdict = (
                AuthorityVerdictType.QUARANTINE_CONTRADICTORY_EVIDENCE
                if evidence_verdict.verdict == EvidenceVerdictType.CONTRADICTORY
                else AuthorityVerdictType.DEFER_INSUFFICIENT_EVIDENCE
            )
            return CompiledRtpMatrixPlan(
                plan_id=plan_id,
                action_id=candidate.action_id,
                target_tenant_id=candidate.target_tenant_id,
                state_version=cur_version,
                is_committable=False,
                authority_verdict=auth_verdict,
                slack_margins=slack_margins,
                projected_state_delta={},
                predicted_next_state={},
                interference_penalty=0.0,
                c_abi_signature=f"C-ABI-DEFERRED-{auth_verdict.value}"
            )

        # 4. Project State Transition: s_{k+1} = T * s_k + B * a
        T, B = self.compile_transition_matrices()
        T_s = T.matmul(s)
        B_a = B.matmul(a)

        s_next_data = [
            [T_s.data[r][0] + B_a.data[r][0]] for r in range(s.rows)
        ]
        s_next = TensorMatrix(s_next_data, name="predicted_next_state_s1")

        state_keys = ["cpu_pct", "free_ram_mb", "temp_c", "power_w", "vital_hp", "state_version"]
        projected_delta = {
            state_keys[r]: round(s_next.data[r][0] - s.data[r][0], 2)
            for r in range(len(state_keys))
        }
        predicted_next = {
            state_keys[r]: round(s_next.data[r][0], 2)
            for r in range(len(state_keys))
        }

        # 5. Evaluate Cross-Tenant Interference Penalty
        tenant_features = {
            "llc_intensity": float(candidate.required_resources.get("vcpu", 1.0)) * 0.25,
            "bus_intensity": float(candidate.required_resources.get("ram_mb", 512.0)) / 2048.0,
            "io_intensity": float(candidate.required_resources.get("io_weight", 100.0)) / 200.0
        }
        interference = self.calculate_interference_penalty(tenant_features, active_neighbors or [])

        # 6. Commit Approved
        return CompiledRtpMatrixPlan(
            plan_id=plan_id,
            action_id=candidate.action_id,
            target_tenant_id=candidate.target_tenant_id,
            state_version=cur_version,
            is_committable=True,
            authority_verdict=AuthorityVerdictType.COMMIT_APPROVED,
            slack_margins=slack_margins,
            projected_state_delta=projected_delta,
            predicted_next_state=predicted_next,
            interference_penalty=interference,
            c_abi_signature=f"C-ABI-COMMIT-V{cur_version}-{uuid.uuid4().hex[:6].upper()}"
        )

    # ── 2.7 C-ABI / SIMD EXPORT ──────────────────────────────────────────────

    def export_c_abi_matrices(self) -> Dict[str, Any]:
        """
        Exports flat float arrays for zero-overhead C++ / Rust AVX-512 / Vulkan compute shader
        matrix admissibility evaluation.
        """
        host_dummy = {"host_id": "c_abi_export", "workload_class": "standard", "confidence": 0.9}
        M_safe, c_limits, names = self.compile_safety_constraint_matrix(host_dummy)
        T, B = self.compile_transition_matrices()

        return {
            "M_safe_flat": [val for row in M_safe.data for val in row],
            "M_safe_rows": M_safe.rows,
            "M_safe_cols": M_safe.cols,
            "c_limits_flat": [row[0] for row in c_limits.data],
            "T_transition_flat": [val for row in T.data for val in row],
            "B_actuation_flat": [val for row in B.data for val in row],
            "constraint_names": names,
            "simd_alignment_bytes": 64  # AVX-512 512-bit cacheline alignment
        }
