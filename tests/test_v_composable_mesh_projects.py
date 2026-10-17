"""
Krystal Stack Platform Framework - Composable Open-Source Mesh Verification Suite
================================================================================
Milestone: V-COMPOSABLE-MESH-01 (Verification of the 5 Domain Blueprints)
Constitutional Invariant: VITAL_MAX_HP == 6
"""

import time
import pytest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractHeader, ContractViolation, ReceiptContract
from krystal_kernel.epistemic_enlg_compiler import (
    EnlgRole,
    EnlgEvidenceClass,
    ArtifactGate,
)
from projects.sovereign_game_companion import (
    SovereignGameCompanionBridge,
    GameRenderFrameTelemetry,
)
from projects.zero_stutter_dcc_copilot import (
    ZeroStutterDccBridge,
    DccTaskResult,
)
from projects.self_tuning_workstation import (
    SelfTuningWorkstationEngine,
    HardwareTuningHypothesis,
    TuningEvaluationReport,
)
from projects.verified_robotics_twin import (
    VerifiedRoboticsTwinBridge,
    SensorObservation,
    RobotNavigationCommand,
)
from projects.low_latency_ide_mesh import (
    LowLatencyIdeMeshCoordinator,
    EditorFrameQoS,
    CodeAssistantSuggestion,
)


class TestVComposableMeshProjects:
    """End-to-end verification of all 5 Composable Mesh projects."""

    def test_01_vital_max_hp_across_all_projects(self):
        """Invariant: VITAL_MAX_HP == 6 must hold across all 5 projects."""
        assert VITAL_MAX_HP == 6

        # Negative test on telemetry dataclass
        with pytest.raises(ContractViolation):
            GameRenderFrameTelemetry(
                frame_index=1,
                render_duration_ms=6.0,
                uma_saturation=0.5,
                k_speculative_slots=8,
                upscaling_active=False,
                npc_tokens_generated=5,
                vital_max_hp=5,
            )

    def test_02_blueprint_1_game_companion_vblank_slicing_and_dp4a(self):
        """Blueprint 1: Tests 120 FPS VBlank slicing and DP4A upscaling at 78% UMA saturation."""
        gate = ArtifactGate()
        bridge = SovereignGameCompanionBridge(target_fps=120.0, artifact_gate=gate)

        # Frame 1: Low saturation (0.40) -> K_spec = 8, no upscaling
        telemetry_low, npc_text = bridge.process_render_frame(
            gpu_render_duration_ms=5.0,
            uma_bandwidth_saturation=0.40,
            npc_inference_generator=lambda: "NPC: Pozor na pravé krídlo!",
        )
        assert telemetry_low.k_speculative_slots == 8
        assert telemetry_low.upscaling_active is False
        assert npc_text is not None
        assert "NPC:" in npc_text

        # Frame 2: High saturation (0.82 >= 0.78) -> K_spec = 1, DP4A upscaling active
        telemetry_high, _ = bridge.process_render_frame(
            gpu_render_duration_ms=7.0,
            uma_bandwidth_saturation=0.82,
        )
        assert telemetry_high.k_speculative_slots == 1
        assert telemetry_high.upscaling_active is True

        # Deposit claim into ArtifactGate
        claim_id = bridge.submit_npc_claim_to_gate(
            claim_id="claim_001",
            claim_text="The citadel gate is unlocked",
            evidence=EnlgEvidenceClass.SIMULATED,
        )
        assert gate.quarantined_count == 1

    def test_03_blueprint_2_dcc_copilot_calorimeter_decouples_parasites(self):
        """Blueprint 2: Blender co-pilot calorimeter decouples runaway background tasks."""
        bridge = ZeroStutterDccBridge(max_task_joules=3.0)

        task_id = bridge.schedule_generative_task("geo_nodes_task", "Generate procedural crystal spires")

        # Slice 1: Normal generation (100 ms * 28 W = 2.8 J < 3.0 J limit, with entropy reduction)
        res1 = bridge.execute_generative_slice(
            task_id=task_id,
            execution_fn=lambda: {"vertices": 5000, "materials": 2},
            duration_ms=100.0,
            uma_bytes=1024 * 1024,
            entropy_reduction=1.0,
        )
        assert res1.completed is True
        assert res1.is_parasitic_decoupled is False
        assert res1.output_payload is not None

        # Slice 2: Runaway task (adds another 50 ms -> 4.2 J total > 3.0 J limit, zero entropy reduction)
        res2 = bridge.execute_generative_slice(
            task_id=task_id,
            execution_fn=lambda: {"vertices": 10000},
            duration_ms=50.0,
            uma_bytes=512,
            entropy_reduction=0.0,
        )
        assert res2.is_parasitic_decoupled is True
        assert res2.completed is False
        assert res2.output_payload is None

    def test_04_blueprint_3_self_tuning_workstation_safe_probing(self):
        """Blueprint 3: Autonomous workstation tuning commits gain and rejects regressions (Delta_S == 0)."""
        engine = SelfTuningWorkstationEngine()
        assert engine.current_state["pl1_watts"] == 28.0

        # Successful optimization hypothesis: 10% expected, 15% measured
        hyp_win = HardwareTuningHypothesis(
            hypothesis_id="hyp_pl1_32w",
            target_subsystem="pl1_watts",
            proposed_value=32.0,
            baseline_value=28.0,
            expected_latency_improvement_pct=10.0,
        )

        rep_win = engine.evaluate_and_actuate(
            hypothesis=hyp_win,
            probe_fn=lambda: 15.0,  # 15% improvement
        )
        assert rep_win.experiment_succeeded is True
        assert rep_win.measured_improvement_pct == 15.0
        assert rep_win.receipt_id is not None
        assert engine.current_state["pl1_watts"] == 32.0

        # Regressive hypothesis: 20% expected, only 5% measured -> strict Delta_S == 0
        hyp_regress = HardwareTuningHypothesis(
            hypothesis_id="hyp_pl1_45w",
            target_subsystem="pl1_watts",
            proposed_value=45.0,
            baseline_value=32.0,
            expected_latency_improvement_pct=20.0,
        )
        rep_regress = engine.evaluate_and_actuate(
            hypothesis=hyp_regress,
            probe_fn=lambda: 5.0,  # Below threshold
        )
        assert rep_regress.experiment_succeeded is True
        assert rep_regress.receipt_id is None
        # State unchanged!
        assert engine.current_state["pl1_watts"] == 32.0

    def test_05_blueprint_4_verified_robotics_sensor_conflict_bifurcation(self):
        """Blueprint 4: Disagreeing sensors trigger BifurcatedConflictRing and emergency stop."""
        bridge = VerifiedRoboticsTwinBridge()

        # Observation A: Camera simulator says obstacle is 10.0m away
        obs_cam = SensorObservation(
            sensor_name="sim_camera",
            obstacle_distance_meters=10.0,
            evidence_class=EnlgEvidenceClass.SIMULATED,
            timestamp_ns=time.time_ns(),
        )

        # Observation B: Physical LiDAR sensor says obstacle is 0.5m away (Severe discrepancy!)
        obs_lidar = SensorObservation(
            sensor_name="physical_lidar",
            obstacle_distance_meters=0.5,
            evidence_class=EnlgEvidenceClass.PHYSICAL_MEASURED,
            timestamp_ns=obs_cam.timestamp_ns,
        )

        cmd = bridge.process_dual_sensor_inputs(obs_cam, obs_lidar)
        assert cmd.is_conflict_superposition is True
        assert cmd.is_emergency_stop is True
        assert cmd.target_velocity_mps == 0.0

        # Resolve conflict via valid physical CDT receipt
        header = ContractHeader("rcpt_robot_cdt", 1, 1, 0x10, 1, 1000, "0" * 64)
        receipt = ReceiptContract(
            header=header,
            execution_digest="exec_lidar_ground_truth",
            applied_profile="ultra_saver",
            actuation_duration_ns=100_000,
            was_degraded_fallback=False,
        )
        conflict_id = f"conf_{obs_cam.timestamp_ns}"
        collapsed_frame = bridge.resolve_sensor_conflict(
            conflict_id=conflict_id,
            winning_branch_generation=2,  # LiDAR won
            cdt_receipt=receipt,
        )
        assert collapsed_frame.generation_id == 0
        assert collapsed_frame.vital_hp == 6

    def test_06_blueprint_5_low_latency_ide_mesh_qos_and_code_promotion(self):
        """Blueprint 5: IDE editor activity throttles compiler, code proposals quarantined."""
        coord = LowLatencyIdeMeshCoordinator()

        # 1. Idle editor -> no throttling
        qos_idle = coord.arbitrate_qos()
        assert qos_idle.editor_active_keystrokes is False
        assert qos_idle.compiler_throttle_pct == 0.0

        # 2. Active typing -> 80% throttle to preserve 120 Hz UI
        coord.record_user_keystroke()
        qos_active = coord.arbitrate_qos()
        assert qos_active.editor_active_keystrokes is True
        assert qos_active.compiler_throttle_pct == 80.0

        # 3. AI code suggestion enters quarantine
        sug = coord.process_ai_code_suggestion(
            suggestion_id="sug_fn_01",
            code_snippet="def fast_matmul(): pass",
            evidence=EnlgEvidenceClass.CODE_PRESENT,
            role=EnlgRole.PROPOSER,
        )
        assert sug.artifact_quarantined is True
        assert sug.is_verified_by_tests is False

        # 4. Proposer cannot self-certify release
        header = ContractHeader("rcpt_test_01", 1, 1, 0x10, 1, 1000, "0" * 64)
        receipt = ReceiptContract(
            header=header,
            execution_digest="exec_pytest_pass",
            applied_profile="balanced",
            actuation_duration_ns=200_000,
            was_degraded_fallback=False,
        )
        with pytest.raises(ContractViolation):
            coord.certify_code_with_test_receipt(
                suggestion_id="sug_fn_01",
                receipt=receipt,
                evaluator_role=EnlgRole.PROPOSER,
            )

        # 5. Independent VERIFIER releases the code
        released = coord.certify_code_with_test_receipt(
            suggestion_id="sug_fn_01",
            receipt=receipt,
            evaluator_role=EnlgRole.VERIFIER,
        )
        assert released is True
