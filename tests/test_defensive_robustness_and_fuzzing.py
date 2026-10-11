#!/usr/bin/env python3
"""
Defensive Robustness & Boundary Fuzzing Test Suite for Krystal-Stack Framework.

Validates defensive posture, boundary checking, and fail-safe invariants against:
1. Adversarial prompt fuzzing (SQLi, command injection strings, buffer stress, NULL bytes).
2. Physical calculation anomalies (NaN, Inf, negative resistance, Ohm's law division by zero).
3. Pauli Exclusion Guard collision stress testing on layer [A_CENTER].
4. Conduct Code Bridge safety interlock and deterministic rescue cycles ([Z_RETRACT_4MM_M07]).
5. Offline / Local LLM integration gateway compatibility for internal research.
"""

import os
import sys
import math
import time
import unittest
from typing import Dict, Any, List

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.procedural_kernel_module_manager import (
    ProceduralKernelModuleManager,
    ProcessingCommandMindmap,
    GLOBAL_KERNEL_MODULE_MANAGER
)
from krystal_kernel.sovereign_htib_bridge import (
    SovereignHTIBBridge,
    PauliExclusionViolation,
    ConductCodeViolation,
    MIN_EFFECTIVE_RESISTANCE,
    CONFIDENCE_THRESHOLD,
    LAYER_CENTER,
    LAYER_AURA_RING
)


class TestDefensiveRobustnessAndFuzzing(unittest.TestCase):
    """Deep defensive validation and edge-case fuzzing suite."""

    def setUp(self):
        self.module_mgr = ProceduralKernelModuleManager()
        self.htib = SovereignHTIBBridge(map_id="FUZZ_RESEARCH_SECTOR")
        self.sample_blueprint = {
            "map_id": "FUZZ_RESEARCH_SECTOR",
            "hex_radius": 1.0,
            "cells": [
                {"q": 0, "r": 0, "biome": "CRYSTAL_CITADEL", "elevation": 0.0, "is_traversable": True},
                {"q": 1, "r": 0, "biome": "PLAINS", "elevation": 0.0, "is_traversable": True},
                {"q": 0, "r": 1, "biome": "VOLCANIC", "elevation": 0.5, "is_hazard": True},
                {"q": -1, "r": 0, "biome": "VOID_CORRUPTION", "elevation": -1.0, "is_hazard": True}
            ],
            "entities": [
                {"id": "sentinel_01", "name": "Aegis Sentinel", "q": 0, "r": 0, "layer": "A_CENTER", "hp": 6, "base_resistance": 2.0},
                {"id": "sentinel_halo", "name": "Energy Ring", "q": 0, "r": 0, "layer": "A_AURA_RING", "hp": 6}
            ]
        }
        self.htib.compile_blueprint(self.sample_blueprint)

    def test_adversarial_prompt_injection_fuzzing(self):
        """Fuzz kernel module manager with SQLi, shell metachars, NULL bytes and huge payloads."""
        adversarial_payloads = [
            "'; DROP TABLE kernel_modules; --",
            "$(rm -rf / || del /f /q C:\\*.*) && cat /etc/passwd",
            "vytvor mi nastroj \x00\x01\x02\x1b[31m WITH ESCAPE CODES",
            "<script>alert('xss_attack')</script><iframe src='javascript:void(0)'>",
            "A" * 25000,  # 25 KB buffer
            "🔥💥⚡💀" * 500,  # High-density UTF-8 multi-byte emoji flood
            "   \t\r\n   ",  # Pure whitespace
            "{\"__proto__\": {\"admin\": true}, \"constructor\": {\"prototype\": {}}}"  # Prototype pollution style payload
        ]

        for payload in adversarial_payloads:
            # Must execute without unhandled exception and return valid 8-node DAG
            mindmap = self.module_mgr.execute_agent_pipeline_with_mindmap(payload)
            self.assertIsInstance(mindmap, ProcessingCommandMindmap)
            self.assertEqual(len(mindmap.nodes), 8)
            self.assertEqual(len(mindmap.edges), 10)
            self.assertEqual(mindmap.vital_max_hp, VITAL_MAX_HP)

            # Node 1 (Input command) must have properly clamped label
            input_node = next(n for n in mindmap.nodes if n.id == "node_user_input")
            self.assertTrue(len(input_node.label) <= 60)
            self.assertEqual(input_node.vital_hp, VITAL_MAX_HP)

    def test_boundary_physical_fuzzing_ohms_law(self):
        """Fuzz Ohm's law state resolution with extreme floats (NaN, Inf, negative resistance)."""
        fuzz_voltages = [
            float("nan"),
            float("inf"),
            float("-inf"),
            -999999.0,
            1e18,
            "corrupted_voltage_string"
        ]

        for v in fuzz_voltages:
            instr = {
                "type": "APPLY_OHM_VOLTAGE",
                "target_id": "sentinel_01",
                "applied_voltage": v,
                "confidence": 0.95,
                "is_hazardous": False
            }
            res = self.htib.resolve_engine_state(instr)
            self.assertEqual(res["status"], "OK")
            self.assertFalse(math.isnan(res["effective_voltage"]))
            self.assertFalse(math.isinf(res["effective_voltage"]))
            self.assertGreaterEqual(res["effective_resistance"], MIN_EFFECTIVE_RESISTANCE)
            self.assertGreaterEqual(res["target_new_hp"], 0)
            self.assertLessEqual(res["target_new_hp"], VITAL_MAX_HP)

    def test_pauli_exclusion_guard_collision_stress(self):
        """Verify Pauli Exclusion Guard prevents overlapping entities on [A_CENTER]."""
        # 1. Blueprint compilation collision
        colliding_blueprint = {
            "map_id": "COLLISION_TEST",
            "cells": [{"q": 0, "r": 0, "biome": "PLAINS", "is_traversable": True}],
            "entities": [
                {"id": "unit_a", "q": 0, "r": 0, "layer": "A_CENTER", "hp": 6},
                {"id": "unit_b", "q": 0, "r": 0, "layer": "A_CENTER", "hp": 6}
            ]
        }
        with self.assertRaises(PauliExclusionViolation):
            SovereignHTIBBridge().compile_blueprint(colliding_blueprint)

        # 2. Dynamic movement collision
        # Add unit_b on (1, 0)
        self.htib.entities["unit_b"] = {
            "id": "unit_b", "name": "Vanguard B", "q": 1, "r": 0, "layer": LAYER_CENTER, "hp": 6,
            "base_resistance": 1.0, "conduct": {"is_active": True, "confidence": 1.0, "vital_hp": 6}
        }
        self.htib.center_occupancy["1,0"] = "unit_b"

        # Attempt to move unit_b to (0, 0) where sentinel_01 already resides
        move_instr = {
            "type": "MOVE_ENTITY",
            "source_id": "unit_b",
            "target_q": 0,
            "target_r": 0,
            "confidence": 1.0,
            "is_hazardous": False
        }
        res = self.htib.resolve_engine_state(move_instr)
        self.assertEqual(res["status"], "PAULI_VIOLATION")
        self.assertEqual(res["fallback_action"], "DEFLECT_PLACEMENT_TO_STANDBY")
        # Ensure sentinel_01 still occupies (0, 0)
        self.assertEqual(self.htib.center_occupancy["0,0"], "sentinel_01")
        self.assertEqual(self.htib.entities["unit_b"]["q"], 1)

    def test_hardware_safety_fault_interlock_fuzzing(self):
        """Verify Conduct Code Bridge immediately engages ISO 13849 fail-safe on fault injection."""
        mindmap = self.module_mgr.execute_agent_pipeline_with_mindmap(
            prompt="Spusti frézovanie vysokou rýchlosťou",
            hardware_safety_fault=True
        )

        # Verify Node 7 (Hardware Safety Interlock) state
        node7 = next(n for n in mindmap.nodes if n.id == "node_hardware_interlock")
        self.assertEqual(node7.status, "BLOCKED")
        self.assertTrue(node7.metrics["e_stop_engaged"])
        self.assertEqual(node7.metrics["emulsion_pressure_bar"], 0.0)
        self.assertEqual(node7.metrics["rescue_action"], "Z_RETRACT_4MM_M07")
        self.assertFalse(node7.conduct_flags["is_active"])
        self.assertTrue(node7.conduct_flags["is_hazardous"])
        self.assertIn("[!ACT]", node7.conduct_badge)
        self.assertIn("[HAZ]", node7.conduct_badge)

        # Verify Node 8 (Fleet Telemetry Aggregator) intercepted emergency broadcast
        node8 = next(n for n in mindmap.nodes if n.id == "node_fleet_telemetry")
        self.assertEqual(node8.metrics["fleet_spindles_online"], 16)
        self.assertTrue(node8.metrics["emergency_interlock_logged"])
        self.assertEqual(node8.metrics["taylor_arrhenius_rul_hours"], 3.8)

    def test_conduct_code_low_confidence_gate(self):
        """Verify Conduct Code Bridge drops unconfirmed low-confidence (<0.85) commands."""
        low_conf_instr = {
            "type": "MOVE_ENTITY",
            "source_id": "sentinel_01",
            "target_q": 1,
            "target_r": 0,
            "confidence": 0.62,  # Below CONFIDENCE_THRESHOLD (0.85)
            "is_hazardous": False
        }
        res = self.htib.resolve_engine_state(low_conf_instr)
        self.assertEqual(res["status"], "FALLBACK_TRIGGERED")
        self.assertEqual(res["fallback_action"], "FALLBACK_DEFENSIVE_GROUNDING")
        self.assertIn("low confidence", res["error"])
        self.assertEqual(res["conduct_badge"], "[!ACT] [LCK] [!CNF] [HAZ] [VIT 6HP]")

    def test_local_llm_gateway_interoperability(self):
        """Simulate feeding test vectors from an offline / local LLM into Krystal-Stack."""
        class MockLocalLLMResearchRunner:
            """Simulates an offline local LLM producing synthetic edge-case test vectors."""
            @staticmethod
            def generate_synthetic_payloads() -> List[Dict[str, Any]]:
                return [
                    {"prompt": "EXTREME_THERMAL_SURGE 5000C", "fault": True},
                    {"prompt": "HEX_WARP_ANOMALY q=9999 r=-9999", "fault": False},
                    {"prompt": "DIV_BY_ZERO_RESISTANCE_PROBE", "fault": False}
                ]

        runner = MockLocalLLMResearchRunner()
        payloads = runner.generate_synthetic_payloads()

        for p in payloads:
            mm = self.module_mgr.execute_agent_pipeline_with_mindmap(
                prompt=p["prompt"],
                hardware_safety_fault=p["fault"]
            )
            self.assertEqual(len(mm.nodes), 8)
            # Invariant: Vital Max HP must never exceed 6
            self.assertEqual(mm.vital_max_hp, 6)
            for n in mm.nodes:
                self.assertLessEqual(n.vital_hp, 6)


if __name__ == "__main__":
    unittest.main()
