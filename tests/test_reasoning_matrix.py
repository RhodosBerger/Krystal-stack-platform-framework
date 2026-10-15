#!/usr/bin/env python3
"""
Unit and integration tests for Epistemic Reasoning Matrix, Selection Funnel,
and Polymorphic Dynamic Workspace (V-PLAN-02).
"""

import unittest
from typing import List

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.reasoning_matrix import (
    ReasoningModeKind,
    CandidateAction,
    AdmissibleHypothesisFunnel,
    CausalDiagnosticWorkspaceComposer,
    WorkspaceNodeKind,
    WorldStateVector,
    HypothesisEnvelope,
    ObservationResidual,
    IEpistemicNode,
    CausalDiagnosticNode,
    DeterministicMemoryStabilizerNode,
    ShadowExplorationNode,
    EpistemicOrchestrationKernel,
)


class TestReasoningMatrix(unittest.TestCase):

    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6, "Platform invariant VITAL_MAX_HP == 6 must hold")
        self.funnel = AdmissibleHypothesisFunnel(max_uma_pressure=0.82, min_dwell_s=1.0)
        self.composer = CausalDiagnosticWorkspaceComposer()

    def _generate_candidate_pool(self, count: int = 32) -> List[CandidateAction]:
        """Generates a synthetic high-entropy pool of candidate actions."""
        profiles = ["ultra_saver", "saver_npu", "balanced_npu", "balanced_hybrid", "performance_gpu", "max_burst"]
        modes = list(ReasoningModeKind)
        pool: List[CandidateAction] = []

        for i in range(count):
            prof = profiles[i % len(profiles)]
            mode = modes[i % len(modes)]
            req_gpu = "gpu" in prof or "burst" in prof
            req_npu = "npu" in prof
            req_ov = (i % 3 == 0)

            # Create varying telemetry expectations and evidence levels
            p_latency = 5.0 + (i * 2.0)
            p_fps = 300.0 - (i * 5.0)
            p_power = 2.0 + (i * 1.2)
            p_uma = 0.20 + (i * 0.02)
            evidence = 0.50 + ((i % 5) * 0.12)  # Some >= 0.70, some below

            cand = CandidateAction(
                action_id=f"cand_{i:02d}",
                target_profile=prof,
                reasoning_mode=mode,
                proposed_by=f"Node_{mode.value}",
                predicted_latency_ms=p_latency,
                predicted_throughput_fps=p_fps,
                predicted_power_w=p_power,
                predicted_uma_pressure=p_uma,
                requires_gpu=req_gpu,
                requires_npu=req_npu,
                requires_openvino=req_ov,
                evidence_score=evidence,
                vital_max_hp=6,
            )
            pool.append(cand)
        return pool

    def test_01_vital_max_hp_invariant(self):
        pool = self._generate_candidate_pool(10)
        for c in pool:
            self.assertEqual(c.vital_max_hp, 6)

    def test_02_funnel_filtration_reduces_entropy(self):
        """Validates that 32 candidates are systematically pruned through the 5 stages."""
        pool = self._generate_candidate_pool(32)
        self.assertEqual(len(pool), 32)

        winner, tel = self.funnel.filter_and_select(
            pool,
            current_temp_c=72.0,
            current_uma_pressure=0.60,
            battery_pct=85.0,
            now_s=100.0,
        )

        self.assertIsNotNone(winner)
        self.assertEqual(tel.initial_candidates, 32)
        # Pruning progression: 32 -> s1 <= 32 -> s2 <= s1 -> s3 <= s2 -> s4 <= 2 -> winner == 1
        self.assertLessEqual(tel.after_capability_filter, 32)
        self.assertLessEqual(tel.after_safety_filter, tel.after_capability_filter)
        self.assertLessEqual(tel.after_evidence_filter, tel.after_safety_filter)
        self.assertLessEqual(tel.after_utility_scoring, 2)
        self.assertEqual(tel.authority_verdict, "AUTH_COMMIT_APPROVED")
        self.assertIn(winner.target_profile, ["ultra_saver", "saver_npu", "balanced_npu", "balanced_hybrid", "performance_gpu", "max_burst"])

    def test_03_zero_feasible_candidates_strictly_enforces_degraded_safe_fallback(self):
        """Total veto (A_commit == 0) strictly triggers AUTH_DEGRADED_SAFE_FALLBACK to ultra_saver."""
        # Feed candidates that all have zero evidence score or are unsafe
        bad_candidates = [
            CandidateAction(
                action_id="bad_1",
                target_profile="max_burst",
                reasoning_mode=ReasoningModeKind.LLM_CAUSAL_NODE,
                proposed_by="SpeculativePlanner",
                predicted_latency_ms=10.0,
                predicted_throughput_fps=500.0,
                predicted_power_w=35.0,
                predicted_uma_pressure=0.90,
                requires_gpu=True,
                requires_npu=False,
                requires_openvino=False,
                evidence_score=0.20,  # Below 0.70 evidence threshold
            )
        ]

        winner, tel = self.funnel.filter_and_select(
            bad_candidates,
            current_temp_c=98.0,  # Extreme thermal emergency
            current_uma_pressure=0.88,
            battery_pct=10.0,
            now_s=105.0,
        )

        self.assertEqual(tel.authority_verdict, "AUTH_DEGRADED_SAFE_FALLBACK")
        self.assertIsNotNone(winner)
        self.assertEqual(winner.target_profile, "ultra_saver")
        self.assertEqual(winner.vital_max_hp, 6)

    def test_04_dwell_hysteresis_suppresses_rapid_flapping(self):
        """Within the 1.0s dwell window, existing active profile is held unless bypassed."""
        pool = self._generate_candidate_pool(20)

        # First commit at t = 100.0
        w1, _ = self.funnel.filter_and_select(
            pool,
            current_temp_c=65.0,
            current_uma_pressure=0.50,
            battery_pct=90.0,
            now_s=100.0,
        )

        # Immediate follow-up at t = 100.3 (inside 1.0s dwell window)
        w2, _ = self.funnel.filter_and_select(
            pool,
            current_temp_c=65.5,
            current_uma_pressure=0.52,
            battery_pct=89.9,
            now_s=100.3,
        )

        # Should retain previous profile to suppress oscillation
        self.assertEqual(w1.target_profile, w2.target_profile)

    def test_05_causal_diagnostic_workspace_generates_coherent_multi_node_scene(self):
        """Verifies composition of dynamic micro-app workspace instead of chat text."""
        scene = self.composer.compose_diagnostic_scene(
            query="Prečo sa mi spomalila inference a čo mám zmeniť?",
            observed_uma_pressure=0.84,
            observed_latency_p99_ms=24.5,
            observed_temp_c=68.0,
        )

        self.assertEqual(scene.vital_max_hp, 6)
        self.assertEqual(scene.scene_id, "SCENE-DIAG-UMA-THRASHING")
        self.assertIn("UMA bus thrashing detected", scene.root_cause_summary)
        self.assertEqual(len(scene.nodes), 4)

        node_kinds = [n.kind for n in scene.nodes]
        self.assertIn(WorkspaceNodeKind.CAUSAL_DIAGNOSIS, node_kinds)
        self.assertIn(WorkspaceNodeKind.ACTION_CARD, node_kinds)
        self.assertIn(WorkspaceNodeKind.INTERACTIVE_SIMULATION, node_kinds)

    def test_06_polymorphic_node_properties_and_actions(self):
        """Ensures every workspace node exposes unified polymorphic contract."""
        scene = self.composer.compose_diagnostic_scene(
            query="Diagnostika stavu",
            observed_uma_pressure=0.85,
            observed_latency_p99_ms=30.0,
            observed_temp_c=70.0,
        )
        d = scene.to_dict()
        self.assertIsInstance(d["nodes"], list)
        for n in d["nodes"]:
            self.assertEqual(n["vital_max_hp"], 6)
            self.assertIn("node_id", n)
            self.assertIn("kind", n)
            self.assertIn("title", n)
            self.assertIn("summary", n)
            self.assertIn("properties", n)
            self.assertIn("actions", n)

    def test_07_inter_node_protocol_and_world_state_vector(self):
        """Verifies that all concrete nodes adhere to IEpistemicNode runtime protocol."""
        diag_node = CausalDiagnosticNode()
        mem_node = DeterministicMemoryStabilizerNode()
        shadow_node = ShadowExplorationNode()

        self.assertIsInstance(diag_node, IEpistemicNode)
        self.assertIsInstance(mem_node, IEpistemicNode)
        self.assertIsInstance(shadow_node, IEpistemicNode)

        state = WorldStateVector(
            state_timestamp_ns=1_000_000,
            sequence_id=42,
            cpu_utilization_pct=45.0,
            gpu_utilization_pct=80.0,
            uma_pressure=0.85,
            temperature_c=71.0,
            battery_pct=92.0,
            plugged=True,
            queue_depth=12,
            vital_max_hp=6,
        )
        self.assertEqual(state.vital_max_hp, 6)
        self.assertTrue(diag_node.can_evaluate(state))
        self.assertTrue(mem_node.can_evaluate(state))
        self.assertTrue(shadow_node.can_evaluate(state))

        hyps_diag = diag_node.propose(state)
        hyps_mem = mem_node.propose(state)
        hyps_shadow = shadow_node.propose(state)

        self.assertEqual(len(hyps_diag), 1)
        self.assertEqual(len(hyps_mem), 1)
        self.assertEqual(len(hyps_shadow), 1)

        self.assertEqual(hyps_diag[0].vital_max_hp, 6)
        self.assertEqual(hyps_mem[0].vital_max_hp, 6)
        self.assertEqual(hyps_shadow[0].vital_max_hp, 6)

    def test_08_epistemic_orchestration_kernel_multi_node_step(self):
        """End-to-end multi-node orchestration cycle through the kernel."""
        diag_node = CausalDiagnosticNode()
        mem_node = DeterministicMemoryStabilizerNode()
        shadow_node = ShadowExplorationNode()

        kernel = EpistemicOrchestrationKernel(
            funnel=self.funnel,
            nodes=[diag_node, mem_node, shadow_node],
        )

        state = WorldStateVector(
            state_timestamp_ns=2_000_000,
            sequence_id=101,
            cpu_utilization_pct=50.0,
            gpu_utilization_pct=85.0,
            uma_pressure=0.86,
            temperature_c=68.0,
            battery_pct=95.0,
            plugged=True,
            queue_depth=18,
            vital_max_hp=6,
        )

        winner, scene, telemetry = kernel.step(state, query="Prečo je vysoká latencia?", now_s=200.0)

        self.assertIsNotNone(winner)
        self.assertEqual(telemetry.authority_verdict, "AUTH_COMMIT_APPROVED")
        self.assertEqual(scene.vital_max_hp, 6)
        self.assertTrue(scene.is_shadow_available)
        self.assertGreaterEqual(len(scene.nodes), 3)

    def test_09_attested_feedback_loop_calibrates_node_weights(self):
        """Verifies that observation residuals adjust internal node calibration."""
        mem_node = DeterministicMemoryStabilizerNode()
        initial_scale = mem_node._calibration_scale

        residual = ObservationResidual(
            residual_id="res_001",
            hypothesis_id="node_alt_a_memory_hyp_stream_throttle",
            node_id="node_alt_a_memory",
            error_latency_ms=4.5,  # Measured latency was higher than predicted
            error_throughput_fps=-20.0,
            error_power_w=0.5,
            error_uma_pressure=0.04,
            real_duration_us=15700.0,
            proof_hash="SHA256-PROOF-TEST-RESIDUAL",
            vital_max_hp=6,
        )

        mem_node.feedback(residual)
        # Scale should increase because error_latency_ms > 0
        self.assertGreater(mem_node._calibration_scale, initial_scale)

    def test_10_constitutional_invariant_vital_max_hp_across_all_abi_types(self):
        """Strict verification of VITAL_MAX_HP == 6 across all inter-node types."""
        self.assertEqual(VITAL_MAX_HP, 6)
        with self.assertRaises(AssertionError):
            WorldStateVector(
                state_timestamp_ns=1, sequence_id=1, cpu_utilization_pct=10, gpu_utilization_pct=10,
                uma_pressure=0.5, temperature_c=50, battery_pct=100, plugged=True, queue_depth=0,
                vital_max_hp=5  # Violation!
            )


if __name__ == "__main__":
    unittest.main()
