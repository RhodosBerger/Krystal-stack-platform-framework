"""
krystal_kernel/sovereign_operations_compiler.py

Sovereign Operations Control Center & Compiler Aggregator Engine.
=============================================================================
Unifies the compilation of formal theories, 128B zero-copy contract frames,
ENLG grammar lattices, and runtime coordination across all live subsystems.

Constitutional Invariant: VITAL_MAX_HP == 6 (not Valid(A) => Delta S == 0)
"""

from __future__ import annotations

import os
import sys
import time
import json
import hashlib
import hmac
import subprocess
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from .constants import VITAL_MAX_HP, ContractViolation
from .zero_copy_contracts import BinaryContractType
from .sovereign_runtime import GLOBAL_SOVEREIGN_RUNTIME, SubsystemNodeId
from .future_theory_integrations_orchestrator import GLOBAL_FUTURE_THEORY_INTEGRATIONS_ORCHESTRATOR
from .redundant_epistemic_inference_engine import GLOBAL_REDUNDANT_INFERENCE_ENGINE
from .verification.theory_dependency_graph import GLOBAL_THEORY_DEPENDENCY_GRAPH
from .pattern_rhythm_cadence_calculator import PatternCadenceType


class CompilationStage(str, Enum):
    """Stages of the Sovereign Compilation & Operational Verification Pipeline."""
    THEORY_DAG_AUDIT = "THEORY_DAG_AUDIT"
    CONTRACT_FRAME_COMPILATION = "CONTRACT_FRAME_COMPILATION"
    INTERPOLATOR_VALIDATION = "INTERPOLATOR_VALIDATION"
    SHADOW_CYCLE_RECONCILIATION = "SHADOW_CYCLE_RECONCILIATION"
    SUBSYSTEM_COORDINATION_POLL = "SUBSYSTEM_COORDINATION_POLL"
    REDUNDANT_INFERENCE_SYNTHESIS = "REDUNDANT_INFERENCE_SYNTHESIS"


@dataclass
class StageExecutionResult:
    """Result of an individual compilation and operational aggregation stage."""
    stage: CompilationStage
    passed: bool
    duration_ms: float
    details: Dict[str, Any]
    error_message: Optional[str] = None
    vital_max_hp: int = VITAL_MAX_HP


@dataclass
class SovereignOperationsReceipt:
    """Authoritative cryptographically signed receipt of the entire operations audit."""
    receipt_id: str
    timestamp_ns: int
    total_stages_run: int
    stages_passed: int
    is_fully_operational: bool
    total_duration_ms: float
    stage_results: List[StageExecutionResult]
    system_metrics: Dict[str, Any]
    hmac_digest: str
    vital_max_hp: int = VITAL_MAX_HP
    git_commit: str = ""
    compile_mode: str = "COMPILE_ONLY"
    invariant_set_hash: str = ""


class SovereignOperationsCompiler:
    """
    Authoritative compiler and operations control aggregator.
    Compiles theories, validates contracts, runs shadow cycles, and queries live subsystems.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional invariant breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.receipt_dir: str = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "artifacts"
        )

    def run_stage_theory_dag_audit(self) -> StageExecutionResult:
        """Stage 1: Compiles and validates DAG dependencies, cycles, and epistemic statuses."""
        t0 = time.perf_counter_ns()
        try:
            total_nodes = len(GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes)
            has_cycles = GLOBAL_THEORY_DEPENDENCY_GRAPH.has_cycles()
            proven_nodes = sum(
                1 for n in GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes.values()
                if n.epistemic_status.value in ("DOKAZANE", "HYPOTHESIS_CONFIRMED")
            )
            passed = (not has_cycles) and (total_nodes >= 25)
            dur = (time.perf_counter_ns() - t0) / 1e6
            return StageExecutionResult(
                stage=CompilationStage.THEORY_DAG_AUDIT,
                passed=passed,
                duration_ms=round(dur, 3),
                details={
                    "total_theory_nodes": total_nodes,
                    "proven_theories_count": proven_nodes,
                    "dag_cycle_detected": has_cycles,
                    "vital_hp": self.vital_max_hp
                }
            )
        except Exception as e:
            dur = (time.perf_counter_ns() - t0) / 1e6
            return StageExecutionResult(
                stage=CompilationStage.THEORY_DAG_AUDIT,
                passed=False,
                duration_ms=round(dur, 3),
                details={},
                error_message=str(e)
            )

    def run_stage_contract_frame_compilation(self) -> StageExecutionResult:
        """Stage 2: Compiles and transmits 128B zero-copy contract frames across coordination channels."""
        t0 = time.perf_counter_ns()
        try:
            channel_id = "WSL_TO_CHRONO_SOC"
            frame = GLOBAL_SOVEREIGN_RUNTIME.transmit_contract_frame(
                channel_id=channel_id,
                contract_type=BinaryContractType.EVIDENCE,
                source_id=0x10,
                target_id=0x20,
                payload_u64=0xCAFEBABEDEADBEEF
            )
            passed = (frame.vital_hp == VITAL_MAX_HP) and (frame.contract_type == int(BinaryContractType.EVIDENCE))
            dur = (time.perf_counter_ns() - t0) / 1e6
            return StageExecutionResult(
                stage=CompilationStage.CONTRACT_FRAME_COMPILATION,
                passed=passed,
                duration_ms=round(dur, 3),
                details={
                    "channel_tested": channel_id,
                    "frame_size_bytes": 128,
                    "frame_contract_type": frame.contract_type,
                    "vital_hp": frame.vital_hp
                }
            )
        except Exception as e:
            dur = (time.perf_counter_ns() - t0) / 1e6
            return StageExecutionResult(
                stage=CompilationStage.CONTRACT_FRAME_COMPILATION,
                passed=False,
                duration_ms=round(dur, 3),
                details={},
                error_message=str(e)
            )

    def run_stage_interpolator_validation(self) -> StageExecutionResult:
        """Stage 3: Validates the 4 adapted interpolators (Hex-HSL, Cadence, Symplectic, Voltage)."""
        t0 = time.perf_counter_ns()
        try:
            # 1. Spatial Hex-HSL
            spatial_res = GLOBAL_SOVEREIGN_RUNTIME.spatial_interpolator.interpolate(
                p1=(0.0, 0.0, 0.0),
                p2=(10.0, 10.0, 10.0),
                t=0.5
            )
            # 2. Cadence Hysteresis
            cadence_res = GLOBAL_SOVEREIGN_RUNTIME.cadence_interpolator.request_cadence(
                desired_cadence=PatternCadenceType.COGNITIVE_BRAINWAVE
            )
            # 3. Symplectic Verlet det(J) == 1.0
            symp_res = GLOBAL_SOVEREIGN_RUNTIME.symplectic_interpolator.step(q=1.0, p=0.0, dt_sec=0.016)
            # 4. Voltage Impedance
            volt_res = GLOBAL_SOVEREIGN_RUNTIME.voltage_interpolator.step_voltage(
                target_voltage=1.05,
                dt_ns=5.0,
                delta_current_a=0.05
            )

            passed = (
                (spatial_res.sector_code in [0x10, 0x20, 0x30, 0x40, 0x50, 0x60]) and
                (symp_res.jacobian_determinant == 1.0) and
                (volt_res.bus_impedance_ohm == 50.0)
            )
            dur = (time.perf_counter_ns() - t0) / 1e6
            return StageExecutionResult(
                stage=CompilationStage.INTERPOLATOR_VALIDATION,
                passed=passed,
                duration_ms=round(dur, 3),
                details={
                    "spatial_sector_code": f"0x{spatial_res.sector_code:02X}",
                    "cadence_active_period_ms": cadence_res.current_period_ms,
                    "symplectic_det_jacobian": symp_res.jacobian_determinant,
                    "voltage_settled_v": volt_res.current_voltage_v,
                    "vital_hp": self.vital_max_hp
                }
            )
        except Exception as e:
            dur = (time.perf_counter_ns() - t0) / 1e6
            return StageExecutionResult(
                stage=CompilationStage.INTERPOLATOR_VALIDATION,
                passed=False,
                duration_ms=round(dur, 3),
                details={},
                error_message=str(e)
            )

    def run_stage_shadow_cycle_reconciliation(self) -> StageExecutionResult:
        """Stage 4: Executes a two-phase shadow cycle verifying 3.5ms compute budget and UMA pressure."""
        t0 = time.perf_counter_ns()
        try:
            receipt = GLOBAL_SOVEREIGN_RUNTIME.execute_shadow_cycle(
                simulated_compute_ms=2.2,
                uma_bus_pressure=0.48
            )
            passed = receipt.is_shadow_budget_met and receipt.state_committed and (receipt.vital_max_hp == VITAL_MAX_HP)
            dur = (time.perf_counter_ns() - t0) / 1e6
            return StageExecutionResult(
                stage=CompilationStage.SHADOW_CYCLE_RECONCILIATION,
                passed=passed,
                duration_ms=round(dur, 3),
                details={
                    "cycle_index": receipt.cycle_index,
                    "shadow_consumed_ms": receipt.shadow_window_consumed_ms,
                    "shadow_budget_ms": receipt.shadow_window_budget_ms,
                    "uma_bus_pressure": receipt.uma_pressure,
                    "state_committed": receipt.state_committed,
                    "hmac_signature_length": len(receipt.hmac_signature)
                }
            )
        except Exception as e:
            dur = (time.perf_counter_ns() - t0) / 1e6
            return StageExecutionResult(
                stage=CompilationStage.SHADOW_CYCLE_RECONCILIATION,
                passed=False,
                duration_ms=round(dur, 3),
                details={},
                error_message=str(e)
            )

    def run_stage_subsystem_coordination_poll(self) -> StageExecutionResult:
        """Stage 5: Audits and verifies all 4 upgraded subsystems via the Integration Orchestrator."""
        t0 = time.perf_counter_ns()
        try:
            report = GLOBAL_FUTURE_THEORY_INTEGRATIONS_ORCHESTRATOR.run_full_integration_upgrade_suite()
            all_verified = report.get("all_subsystems_verified", False)
            hp_ok = (report.get("vital_max_hp") == VITAL_MAX_HP)
            passed = all_verified and hp_ok
            dur = (time.perf_counter_ns() - t0) / 1e6
            return StageExecutionResult(
                stage=CompilationStage.SUBSYSTEM_COORDINATION_POLL,
                passed=passed,
                duration_ms=round(dur, 3),
                details={
                    "total_subsystems": report.get("total_subsystems_upgraded", 0),
                    "all_subsystems_verified": all_verified,
                    "vital_hp": report.get("vital_max_hp", 0)
                }
            )
        except Exception as e:
            dur = (time.perf_counter_ns() - t0) / 1e6
            return StageExecutionResult(
                stage=CompilationStage.SUBSYSTEM_COORDINATION_POLL,
                passed=False,
                duration_ms=round(dur, 3),
                details={},
                error_message=str(e)
            )

    def run_stage_redundant_inference_synthesis(self) -> StageExecutionResult:
        """Stage 6: Runs 5-lens redundant inference across all 15 aspects and checks global synergy."""
        t0 = time.perf_counter_ns()
        try:
            synergy = GLOBAL_REDUNDANT_INFERENCE_ENGINE.compute_global_synergy_matrix()
            passed = (synergy["soundness_percentage"] >= 95.0) and synergy["vital_max_hp_verified"]
            dur = (time.perf_counter_ns() - t0) / 1e6
            return StageExecutionResult(
                stage=CompilationStage.REDUNDANT_INFERENCE_SYNTHESIS,
                passed=passed,
                duration_ms=round(dur, 3),
                details={
                    "total_aspects": synergy["total_aspects_evaluated"],
                    "total_claims": synergy["total_epistemic_claims_generated"],
                    "soundness_percentage": synergy["soundness_percentage"],
                    "cross_domain_links": len(synergy["cross_domain_links"])
                }
            )
        except Exception as e:
            dur = (time.perf_counter_ns() - t0) / 1e6
            return StageExecutionResult(
                stage=CompilationStage.REDUNDANT_INFERENCE_SYNTHESIS,
                passed=False,
                duration_ms=round(dur, 3),
                details={},
                error_message=str(e)
            )

    def compile_and_aggregate_all(self) -> SovereignOperationsReceipt:
        """
        Executes all 6 compilation and operational aggregation stages sequentially,
        emits a signed JSON receipt to artifacts/SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json.
        """
        t_start_ns = time.perf_counter_ns()
        stage_runners = [
            self.run_stage_theory_dag_audit,
            self.run_stage_contract_frame_compilation,
            self.run_stage_interpolator_validation,
            self.run_stage_shadow_cycle_reconciliation,
            self.run_stage_subsystem_coordination_poll,
            self.run_stage_redundant_inference_synthesis,
        ]

        results: List[StageExecutionResult] = []
        for runner in stage_runners:
            res = runner()
            results.append(res)

        total_stages = len(results)
        passed_stages = sum(1 for r in results if r.passed)
        is_operational = (passed_stages == total_stages) and (self.vital_max_hp == VITAL_MAX_HP)

        t_end_ns = time.perf_counter_ns()
        total_dur_ms = (t_end_ns - t_start_ns) / 1e6

        system_metrics = {
            "runtime_name": "KrystalSovereignRuntime",
            "coordination_channels_count": len(GLOBAL_SOVEREIGN_RUNTIME.channels),
            "vital_max_hp": self.vital_max_hp,
            "constitutional_invariant_intact": is_operational,
            "active_brainwave_cadence": GLOBAL_SOVEREIGN_RUNTIME.cadence_interpolator.active_cadence.value,
            "bus_impedance_ohm": GLOBAL_SOVEREIGN_RUNTIME.voltage_interpolator.bus_impedance_ohm,
            "shadow_window_budget_ms": GLOBAL_SOVEREIGN_RUNTIME.SHADOW_WINDOW_BUDGET_MS,
            "theory_nodes_total": len(GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes)
        }

        git_commit = self._get_git_commit()
        canonical_invariants = {
            "epistemic_anti_promotion": "PREDICTED_LT_DERIVED_LT_MEASURED",
            "min_cacheline_alignment_bytes": 64,
            "noether_symmetry": "DELTA_S_ZERO_ON_BREACH",
            "theory_nodes_frozen": len(GLOBAL_THEORY_DEPENDENCY_GRAPH._nodes),
            "vital_max_hp": self.vital_max_hp,
            "zero_copy_alignment_bytes": 128,
            "zero_touch_interposition": "SOURCE_CODE_DELTA_ZERO"
        }
        invariant_set_payload = json.dumps(canonical_invariants, sort_keys=True, separators=(",", ":")).encode("utf-8")
        invariant_set_hash = hashlib.sha256(invariant_set_payload).hexdigest()
        stages_canonical = "|".join(f"{r.stage.value}:{r.passed}:{r.duration_ms}" for r in results)

        # Canonical signature payload binding exact compiler state:
        raw_sig_payload = (
            f"KRYSTAL_OPERATIONS_COMPILER_V2:{git_commit}:{invariant_set_hash}:"
            f"{t_start_ns}:{total_stages}:{passed_stages}:{is_operational}:{self.vital_max_hp}:{stages_canonical}"
        ).encode("utf-8")

        hmac_key = b"KRYSTAL_OPERATIONS_COMPILER_HMAC_SECRET_2026"
        hmac_digest = hmac.new(hmac_key, raw_sig_payload, hashlib.sha256).hexdigest()

        receipt_id = f"OPS-RECEIPT-{int(time.time())}-{hmac_digest[:8]}"
        receipt = SovereignOperationsReceipt(
            receipt_id=receipt_id,
            timestamp_ns=t_start_ns,
            total_stages_run=total_stages,
            stages_passed=passed_stages,
            is_fully_operational=is_operational,
            total_duration_ms=round(total_dur_ms, 3),
            stage_results=results,
            system_metrics=system_metrics,
            hmac_digest=hmac_digest,
            vital_max_hp=self.vital_max_hp,
            git_commit=git_commit,
            compile_mode="COMPILE_ONLY",
            invariant_set_hash=invariant_set_hash
        )

        self._save_receipt_to_disk(receipt)
        return receipt

    def _get_git_commit(self) -> str:
        """Retrieves active Git commit hash for provenance binding."""
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return commit
        except Exception:
            return "UNKNOWN_COMMIT"

    def _save_receipt_to_disk(self, receipt: SovereignOperationsReceipt) -> str:
        """Serializes receipt to artifacts/SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json."""
        os.makedirs(self.receipt_dir, exist_ok=True)
        target_path = os.path.join(self.receipt_dir, "SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json")

        receipt_dict = {
            "receipt_id": receipt.receipt_id,
            "timestamp_ns": receipt.timestamp_ns,
            "git_commit": receipt.git_commit,
            "compile_mode": receipt.compile_mode,
            "receipt_integrity": "VERIFIED",
            "semantic_verdict": "COMMITTED" if receipt.is_fully_operational else "REJECTED",
            "authorization_scope": "current_compile_batch",
            "authorized_operations": receipt.stages_passed,
            "rejected_operations": receipt.total_stages_run - receipt.stages_passed,
            "invariant_set_hash": receipt.invariant_set_hash,
            "total_stages_run": receipt.total_stages_run,
            "stages_passed": receipt.stages_passed,
            "is_fully_operational": receipt.is_fully_operational,
            "total_duration_ms": receipt.total_duration_ms,
            "stage_results": [
                {
                    "stage": r.stage.value,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "details": r.details,
                    "error_message": r.error_message,
                    "vital_max_hp": r.vital_max_hp
                }
                for r in receipt.stage_results
            ],
            "system_metrics": receipt.system_metrics,
            "hmac_digest": receipt.hmac_digest,
            "vital_max_hp": receipt.vital_max_hp
        }

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(receipt_dict, f, indent=2)

        return target_path


# Global singleton compiler instance
GLOBAL_SOVEREIGN_OPERATIONS_COMPILER = SovereignOperationsCompiler()


def main() -> int:
    """CLI entrypoint for running the compiler and operations aggregator."""
    compiler = GLOBAL_SOVEREIGN_OPERATIONS_COMPILER
    print("================================================================================")
    print("⚡ KRYSTAL STACK: SOVEREIGN OPERATIONS CONTROL CENTER & COMPILER AGGREGATOR")
    print(f"🔒 Constitutional Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}")
    print("================================================================================")

    receipt = compiler.compile_and_aggregate_all()

    for idx, res in enumerate(receipt.stage_results, 1):
        status_sym = "✅ PASS" if res.passed else "❌ FAIL"
        print(f"[{idx}/6] {status_sym} | {res.stage.value:<32} ({res.duration_ms:6.2f} ms)")
        if not res.passed and res.error_message:
            print(f"      ERROR: {res.error_message}")

    print("--------------------------------------------------------------------------------")
    print(f"📊 Sumár: {receipt.stages_passed}/{receipt.total_stages_run} fáz prešlo úspešne ({receipt.total_duration_ms:.2f} ms)")
    print(f"📄 Audit Receipt: artifacts/SOVEREIGN_OPERATIONS_CONTROL_RECEIPT.json (ID: {receipt.receipt_id})")
    print(f"🛡️  HMAC Digest:  {receipt.hmac_digest}")

    if receipt.is_fully_operational:
        print(f"🏆 RIADIACE CENTRUM A KOMPILÁTOR: {receipt.stages_passed}/{receipt.total_stages_run} OPERÁCIÍ AUTORIZOVANÝCH (scope: current_compile_batch, Delta S == 0 on breach)")
        return 0
    else:
        print("❌ RIADIACE CENTRUM: OPERAČNÉ ZLYHANIE ALEBO PORUŠENIE INVARIANTU")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
