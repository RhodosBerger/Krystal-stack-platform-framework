#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE
BLENDER SPIRAL TENSOR DRIVER ENGINE & AUTOMATION ORCHESTRATOR
==============================================================================
Tests:
  1. Universal Invariants: VITAL_MAX_HP == 6, GOLDEN_RATIO, GOLDEN_ANGLE_RAD.
  2. Logarithmic Spiral Tensor Spatial Mapping (r, theta, z).
  3. Canonical Knowledge Nodes (Protix ML, Ohmic Law, Network Arbiter,
     Tectonic Matrix, Termix Accountant, Fibonacci Paradox Orchestrator).
  4. Attached Knowledge Sub-lists & Vital HP Weight Clamping [1, 6].
  5. Cycles of Cycles Driver Parameters & Trigonometric Expression Synthesis.
  6. Pure Python Frame Evaluation (Displacement, Yaw, Scale, Emission).
  7. Standalone Blender 4.x Script (`bpy`) Generation & Driver Validation.
  8. Dynamic Knowledge Node Registration.
==============================================================================
"""

import math
import unittest
from typing import Dict, Any

from krystal_kernel.blender_spiral_tensor_driver_engine import (
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    INV_GOLDEN_RATIO,
    GOLDEN_ANGLE_RAD,
    DomainPattern,
    CycleParameter,
    KnowledgeSubItem,
    SpiralTensorNode,
    SpiralTensorFrameState,
    BlenderSpiralTensorDriverEngine,
    GLOBAL_SPIRAL_TENSOR_ENGINE
)


class TestBlenderSpiralTensorDriverEngine(unittest.TestCase):

    def setUp(self):
        self.engine = BlenderSpiralTensorDriverEngine(
            base_radius=3.5,
            radius_expansion_rate=0.125,
            z_step=1.2,
            default_fps=24,
            total_frames=240
        )

    def test_universal_invariants(self):
        """Verify strict adherence to Krystal Stack universal platform invariants."""
        self.assertEqual(VITAL_MAX_HP, 6, "VITAL_MAX_HP must strictly be 6 across all engines")
        self.assertAlmostEqual(GOLDEN_RATIO, 1.61803398875, places=7)
        self.assertAlmostEqual(INV_GOLDEN_RATIO, 1.0 / 1.61803398875, places=7)
        self.assertAlmostEqual(GOLDEN_ANGLE_RAD, 2.399963229728653, places=6)

        # Check all nodes and sub-items enforce VITAL_MAX_HP == 6
        for node in self.engine.nodes:
            self.assertEqual(node.vital_max_hp_rule, 6)
            for sub in node.knowledge_sub_list:
                self.assertLessEqual(sub.vital_hp_weight, 6)
                self.assertGreaterEqual(sub.vital_hp_weight, 1)

    def test_logarithmic_spiral_tensor_coordinates(self):
        """Verify 3D spatial points follow the logarithmic golden spiral tensor."""
        coords_0 = self.engine.compute_spiral_position(0)
        self.assertAlmostEqual(coords_0[0], 3.5, places=2)
        self.assertAlmostEqual(coords_0[1], 0.0, places=2)
        self.assertAlmostEqual(coords_0[2], 0.0, places=2)

        coords_1 = self.engine.compute_spiral_position(1)
        r_1 = 3.5 * math.pow(GOLDEN_RATIO, 0.125)
        self.assertAlmostEqual(math.sqrt(coords_1[0]**2 + coords_1[1]**2), r_1, places=2)
        self.assertAlmostEqual(coords_1[2], 1.2, places=2)

    def test_canonical_knowledge_nodes_completeness(self):
        """Verify that all 6 patterns from previous prompts are synthesized."""
        domains = [n.domain for n in self.engine.nodes]
        self.assertIn(DomainPattern.PROTIX_ML, domains)
        self.assertIn(DomainPattern.OHMIC_PHYSICAL_LAW, domains)
        self.assertIn(DomainPattern.NETWORK_TASK_ARBITER, domains)
        self.assertIn(DomainPattern.TECTONIC_MATRIX, domains)
        self.assertIn(DomainPattern.TERMIX_ACCOUNTANT, domains)
        self.assertIn(DomainPattern.FIBONACCI_ORCHESTRATOR, domains)
        self.assertEqual(len(self.engine.nodes), 6)

    def test_attached_knowledge_sub_lists(self):
        """Verify attached sub-lists of knowledge for each node."""
        protix_node = next(n for n in self.engine.nodes if n.domain == DomainPattern.PROTIX_ML)
        self.assertGreaterEqual(len(protix_node.knowledge_sub_list), 3)
        self.assertTrue(any("sensor" in k.description.lower() for k in protix_node.knowledge_sub_list))

        ohmic_node = next(n for n in self.engine.nodes if n.domain == DomainPattern.OHMIC_PHYSICAL_LAW)
        self.assertTrue(any("v^2 / r" in k.formula_or_axiom.lower() for k in ohmic_node.knowledge_sub_list))
        self.assertTrue(any("discount" in k.topic.lower() or "cooling" in k.topic.lower() for k in ohmic_node.knowledge_sub_list))

        fib_node = next(n for n in self.engine.nodes if n.domain == DomainPattern.FIBONACCI_ORCHESTRATOR)
        self.assertTrue("Tic-Tac-Toe" in fib_node.title or "Counter-Polarity" in fib_node.title)

    def test_knowledge_sub_item_hp_clamping(self):
        """Verify KnowledgeSubItem clamps vital_hp_weight to [1, 6]."""
        sub_high = KnowledgeSubItem("t1", "Topic", "Desc", "Axiom", vital_hp_weight=99)
        self.assertEqual(sub_high.vital_hp_weight, 6)

        sub_low = KnowledgeSubItem("t2", "Topic", "Desc", "Axiom", vital_hp_weight=-5)
        self.assertEqual(sub_low.vital_hp_weight, 1)

    def test_cycles_of_cycles_driver_expression(self):
        """Verify driver mathematical expressions for cycles of cycles."""
        cfg = CycleParameter(
            primary_period=120,
            secondary_period=24,
            amplitude_primary=1.0,
            amplitude_secondary=0.3,
            phase_offset=0.5
        )
        expr = cfg.to_driver_expression(base_z=5.0)
        self.assertIn("5.000", expr)
        self.assertIn("sin((frame / 120.0) * 6.283185 + 0.500)", expr)
        self.assertIn("cos((frame / 24.0) * 6.283185)", expr)

        emit_expr = cfg.to_emission_expression(base_emission=2.0)
        self.assertIn("sin((frame / 24.0) * 6.283185)", emit_expr)

    def test_frame_evaluation_dynamics(self):
        """Verify dynamic evaluation of spiral nodes across frames."""
        f0 = self.engine.evaluate_frame(0)
        self.assertEqual(f0.frame, 0)
        self.assertEqual(f0.total_nodes, 6)
        self.assertEqual(f0.vital_max_hp_rule, 6)

        f48 = self.engine.evaluate_frame(48)
        self.assertEqual(f48.frame, 48)
        for n_id, state in f48.node_states.items():
            self.assertIn("x", state)
            self.assertIn("y", state)
            self.assertIn("z", state)
            self.assertIn("yaw", state)
            self.assertIn("scale", state)
            self.assertIn("emission_strength", state)
            self.assertEqual(state["vital_hp"], 6)

        # Check that different frames yield different animated values (non-static)
        self.assertNotEqual(
            f0.node_states["node_protix_ml"]["z"],
            f48.node_states["node_protix_ml"]["z"]
        )

    def test_blender_script_generation(self):
        """Verify generated Blender 4.x Python script format and driver integration."""
        script = self.engine.generate_blender_automation_script()

        # Blender imports
        self.assertIn("import bpy", script)
        self.assertIn("import math", script)

        # Scene configuration
        self.assertIn('scene.unit_settings.system = "METRIC"', script)
        self.assertIn("scene.render.fps = 24", script)
        self.assertIn("scene.frame_end = 240", script)

        # 3D curve and objects
        self.assertIn("Krystal_Spiral_Guide", script)
        self.assertIn("Krystal_Origin_Gizmo", script)
        self.assertIn("node_protix_ml_Beacon", script)
        self.assertIn("node_protix_ml_Title", script)

        # Drivers
        self.assertIn('.driver_add("location", 2).driver', script)
        self.assertIn('.driver_add("rotation_euler", 2).driver', script)
        self.assertIn('d_loc_z.type = "SCRIPTED"', script)
        self.assertIn("sin((frame /", script)

        # Invariant checks in script
        self.assertIn("VITAL_MAX_HP == 6", script)

    def test_dynamic_node_addition(self):
        """Verify dynamic addition of custom knowledge nodes along the spiral."""
        custom_sub = [
            KnowledgeSubItem("sub_cust_1", "Quantum Teleport", "Transfers qubit states", "Psi = alpha|0> + beta|1>", 6)
        ]
        new_node = self.engine.add_custom_knowledge_node(
            title="QUANTUM: Qubit Entanglement Matrix",
            domain=DomainPattern.PROTIX_ML,
            knowledge_sub_list=custom_sub,
            primary_period=100,
            secondary_period=20
        )
        self.assertEqual(len(self.engine.nodes), 7)
        self.assertEqual(new_node.spiral_index, 6)
        self.assertEqual(new_node.vital_max_hp_rule, 6)

        status = self.engine.get_engine_status()
        self.assertEqual(status["nodes_count"], 7)


if __name__ == "__main__":
    unittest.main()
