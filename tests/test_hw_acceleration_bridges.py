#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: HARDWARE ACCELERATION BRIDGES (LLM & PROCEDURAL SDF)
==============================================================================
Module: tests/test_hw_acceleration_bridges.py
Constitutional Invariant: VITAL_MAX_HP == 6
Milestone: V-HW-ACCEL-BRIDGE-VERIFY-01
Normative References:
  - artifacts/benchmarks/H07_SILICON_TELEMETRY_AND_OPENVINO_LEXICON_SPEC.md
  - docs/research/BLENDER_CORE_MODIFIER_MIMICRY_ENGINE.md

Verifies:
  1. Sacred constitutional invariant VITAL_MAX_HP == 6 across all bridges.
  2. LlmHardwareAccelerationBridge silicon telemetry inspection, dynamic arbitration,
     Turbo boost (PL1 32W), and zero-copy 128B contract frame generation.
  3. ProceduralSdfHwAccelerationBridge chunked batch point evaluation, surface normal
     derivation, and accelerated raymarching viewport rasterization.
  4. H-07 Hex Lattice event code emission (0x10, 0x20, 0x40, 0x41, 0xFF).
  5. Integration with GameLanguageControllerBridge MIMICRY_SYNTHESIS command.
  6. Failure & falsification checks (ContractViolation raised on tampering).

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.silicon_telemetry_lexicon import (
    TelemetryEventCode,
    PulseSequenceLogger,
    OpenVinoSiliconPolicyArbiter,
)
from krystal_kernel.llm_hw_acceleration_bridge import (
    HardwareInferenceReceipt,
    LlmHardwareAccelerationBridge,
    GLOBAL_LLM_HW_BRIDGE,
)
from krystal_kernel.procedural_sdf_hw_bridge import (
    SdfEvaluationReceipt,
    ProceduralSdfHwAccelerationBridge,
    GLOBAL_PROCEDURAL_SDF_BRIDGE,
)
from mimicry_engine.primitives import sdf_sphere, sdf_box
from krystal_web_hub.economic_engine.game_language_api_fetcher import (
    GameLanguageApiFetcher,
    GameCommandType,
)
from wsl_bridge.bridge_daemon import WSLBridgeDaemon
from krystal_kernel.zero_copy_contracts import ZeroCopyContractFrame


class TestHardwareAccelerationBridges(unittest.TestCase):
    """Formal verification test suite for Krystal Stack HW acceleration bridges."""

    def setUp(self):
        self.logger = PulseSequenceLogger()
        self.arbiter = OpenVinoSiliconPolicyArbiter(logger=self.logger)
        self.llm_bridge = LlmHardwareAccelerationBridge(
            logger=self.logger,
            arbiter=self.arbiter,
            vital_max_hp=6
        )
        self.sdf_bridge = ProceduralSdfHwAccelerationBridge(
            logger=self.logger,
            vital_max_hp=6
        )

    def test_01_vital_max_hp_invariants(self):
        """Invariant VITAL_MAX_HP == 6 must strictly hold across all bridge instances."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.llm_bridge.vital_max_hp, 6)
        self.assertEqual(self.sdf_bridge.vital_max_hp, 6)

        # Falsification / negative test: tampered invariant must raise ContractViolation
        with self.assertRaises(ContractViolation):
            LlmHardwareAccelerationBridge(vital_max_hp=5)

        with self.assertRaises(ContractViolation):
            ProceduralSdfHwAccelerationBridge(vital_max_hp=7)

    def test_02_llm_silicon_conditions_inspection(self):
        """Validates real-time inspection of Tiger Lake silicon and UMA contention."""
        conditions = self.llm_bridge.inspect_silicon_conditions()
        self.assertIn("status", conditions)
        self.assertIn("temperature_c", conditions)
        self.assertIn("uma_transit_us", conditions)
        self.assertIn("recommended_topology", conditions)
        self.assertEqual(conditions["vital_max_hp"], 6)

    def test_03_llm_hardware_accelerated_dispatch(self):
        """Validates hardware inference dispatch with PL1 boost and H-07 hex lattice emission."""
        prompt = "Synthesize tactical defense turret at sector alpha"
        receipt = self.llm_bridge.dispatch_accelerated_inference(
            prompt=prompt,
            max_new_tokens=32,
            allow_turbo_boost=True
        )

        self.assertIsInstance(receipt, HardwareInferenceReceipt)
        self.assertEqual(receipt.vital_max_hp, 6)
        self.assertGreater(receipt.tokens_per_second, 10.0)
        self.assertGreater(receipt.effective_bandwidth_gbps, 20.0)
        self.assertTrue(len(receipt.hex_event_codes) >= 3)
        self.assertIn("0xFF:CONSTITUTIONAL_HEARTBEAT", receipt.hex_event_codes)
        self.assertTrue(len(receipt.provenance_hash) == 64)

        # Verify pulse sequence frames recorded in logger
        events = self.logger.read_all()
        self.assertGreaterEqual(len(events), 3)

    def test_04_llm_zero_copy_contract_frame(self):
        """Ensures generation of a valid 128-byte cache-line aligned binary contract frame."""
        receipt = self.llm_bridge.dispatch_accelerated_inference("Test prompt", 16)
        frame = self.llm_bridge.create_zero_copy_contract_frame(receipt, frame_seq=42)

        self.assertEqual(frame.vital_hp, 6)
        self.assertEqual(frame.sequence, 42)
        session_key = b"krystal_zero_copy_test_key_32b!"
        packed = frame.pack_authenticated(session_key=session_key)
        self.assertEqual(len(packed), 128)

        # Unpack verification
        unpacked = ZeroCopyContractFrame.unpack_and_verify(packed, session_key=session_key)
        self.assertEqual(unpacked.vital_hp, 6)
        self.assertEqual(unpacked.sequence, 42)
        self.assertEqual(int(unpacked.contract_type), 0x05)  # EXECUTION

    def test_05_procedural_sdf_batch_evaluation(self):
        """Tests SIMD chunked batch point evaluation against analytical SDF."""
        points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.5, 0.5, 0.5),
            (3.0, 3.0, 3.0),
        ] * 10  # 50 points

        distances, receipt = self.sdf_bridge.evaluate_points_batch(
            sdf_func=lambda p: sdf_sphere(p, r=1.0),
            points=points
        )

        self.assertEqual(len(distances), 50)
        self.assertIsInstance(receipt, SdfEvaluationReceipt)
        self.assertEqual(receipt.points_evaluated, 50)
        self.assertEqual(receipt.vital_max_hp, 6)
        self.assertAlmostEqual(distances[0], -1.0, places=4)  # Origin inside sphere of r=1 -> -1
        self.assertAlmostEqual(distances[1], 0.0, places=4)   # (1,0,0) on surface -> 0
        self.assertAlmostEqual(distances[2], 1.0, places=4)   # (0,2,0) outside by 1 -> +1

    def test_06_procedural_sdf_surface_normals(self):
        """Tests symmetric central difference normal calculation in batch."""
        hit_points = [
            (1.0, 0.0, 0.0),   # +X surface
            (-1.0, 0.0, 0.0),  # -X surface
            (0.0, 1.0, 0.0),   # +Y surface
            (0.0, 0.0, 1.0),   # +Z surface
        ]
        normals = self.sdf_bridge.compute_surface_normal_batch(
            sdf_func=lambda p: sdf_sphere(p, r=1.0),
            hit_points=hit_points
        )

        self.assertEqual(len(normals), 4)
        self.assertAlmostEqual(normals[0][0], 1.0, places=2)
        self.assertAlmostEqual(normals[1][0], -1.0, places=2)
        self.assertAlmostEqual(normals[2][1], 1.0, places=2)
        self.assertAlmostEqual(normals[3][2], 1.0, places=2)

    def test_07_procedural_sdf_accelerated_raymarcher(self):
        """Validates accelerated raymarcher viewport rasterization."""
        # Sphere of r=1.0 at origin, camera at (0, 0, 3) looking at (0, 0, 0)
        res = self.sdf_bridge.accelerated_raymarch_viewport(
            sdf_func=lambda p: sdf_sphere(p, r=1.0),
            width=32,
            height=32,
            camera_pos=(0.0, 0.0, 3.0),
            camera_target=(0.0, 0.0, 0.0),
            max_steps=32
        )

        self.assertEqual(res["width"], 32)
        self.assertEqual(res["height"], 32)
        self.assertEqual(res["total_rays"], 1024)
        self.assertGreater(res["hits_count"], 0)
        self.assertEqual(res["receipt"]["vital_max_hp"], 6)

    def test_08_game_language_controller_mimicry_synthesis(self):
        """Validates integration with Game Language Controller for MIMICRY_SYNTHESIS."""
        controller = GameLanguageApiFetcher()
        
        # Test Slovak prompt
        intent_sk = controller.parse_prompt("Vygeneruj mimikru veže na hex [3, -2]")
        self.assertEqual(intent_sk.command_type, GameCommandType.MIMICRY_SYNTHESIS)
        self.assertEqual(intent_sk.parameters["recipe_id"], "CYBER_TURRET_MK4")
        self.assertEqual(intent_sk.parameters["target_hex"], [3, -2])

        # Test English prompt
        intent_en = controller.parse_prompt("Spawn procedural walker robot at [1, 2]")
        self.assertEqual(intent_en.command_type, GameCommandType.MIMICRY_SYNTHESIS)
        self.assertEqual(intent_en.parameters["recipe_id"], "MECH_WALKER_BIPED")

        # Test execution
        exec_res = controller.execute_intent(intent_sk)
        self.assertEqual(exec_res["status"], "SUCCESS")
        self.assertIn("CYBER_TURRET_MK4", exec_res["feedback"])

    def test_09_wsl_bridge_hardware_telemetry(self):
        """Validates WSLBridgeDaemon hardware telemetry collection and H-07 event reporting."""
        daemon = WSLBridgeDaemon()
        self.assertEqual(daemon.vital_max_hp, 6)
        telem = daemon.collect_linux_telemetry()
        self.assertIn("subsystem", telem)
        self.assertIn("thermal_celsius", telem)
        self.assertEqual(telem["vital_max_hp"], 6)


if __name__ == "__main__":
    unittest.main()
