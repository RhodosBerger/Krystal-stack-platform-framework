#!/usr/bin/env python3
"""
Unit tests for Sovereign Engine Geometric Compiler, Go DSC/Python HTIB bridge,
Ohm's Law physical resolution, Pauli Exclusion Guard, and packed binary buffer.
"""

import os
import sys
import math
import struct
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.sovereign_htib_bridge import (
    SovereignHTIBBridge,
    PauliExclusionViolation,
    ConductCodeViolation,
    axial_to_world,
    axial_distance,
    MIN_EFFECTIVE_RESISTANCE,
    CONFIDENCE_THRESHOLD,
    LAYER_BASE,
    LAYER_AURA_RING,
    LAYER_CENTER,
    LAYER_OVERHEAD,
    GLOBAL_SOVEREIGN_HTIB_BRIDGE
)


class TestSovereignGeometricCompilerAndDSC(unittest.TestCase):
    """Test suite validating Geometric Compiler, DSC physics, Pauli Guard & packed buffer."""

    def setUp(self):
        self.bridge = SovereignHTIBBridge(map_id="TEST_SOVEREIGN_SECTOR_01")
        self.sample_blueprint = {
            "map_id": "TEST_SOVEREIGN_SECTOR_01",
            "hex_radius": 1.0,
            "cells": [
                {"q": 0, "r": 0, "biome": "CRYSTAL_CITADEL", "elevation": 0.0, "is_traversable": True},
                {"q": 1, "r": 0, "biome": "PLAINS", "elevation": 0.0, "is_traversable": True},
                {"q": 0, "r": 1, "biome": "VOLCANIC", "elevation": 0.2, "is_hazard": True},
                {"q": -1, "r": 0, "biome": "CRYO_FROST", "elevation": 0.0, "is_traversable": True}
            ],
            "entities": [
                {"id": "citadel_turret", "name": "Citadel Aegis Turret", "q": 0, "r": 0, "layer": "A_CENTER", "hp": 6, "base_resistance": 2.0},
                {"id": "scout_drone", "name": "Vanguard Scout", "q": 1, "r": 0, "layer": "A_CENTER", "hp": 4, "base_resistance": 1.0},
                {"id": "targeting_halo", "name": "Targeting Ring", "q": 0, "r": 0, "layer": "A_AURA_RING", "hp": 6}
            ]
        }

    def test_blueprint_compilation_and_topology(self):
        """Verify blueprint compiles cells and entities with axial coordinates."""
        res = self.bridge.compile_blueprint(self.sample_blueprint)
        self.assertEqual(res["status"], "COMPILED")
        self.assertEqual(res["total_cells"], 4)
        self.assertEqual(res["total_entities"], 3)
        self.assertEqual(res["vital_max_hp"], VITAL_MAX_HP)

        # Verify axial distance
        d = axial_distance(0, 0, 1, 0)
        self.assertEqual(d, 1)
        d2 = axial_distance(0, 0, 0, 1)
        self.assertEqual(d2, 1)

        # Verify axial to world coordinates
        x, z = axial_to_world(0, 0, 1.0)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(z, 0.0)

        x1, z1 = axial_to_world(1, 0, 1.0)
        self.assertAlmostEqual(x1, math.sqrt(3.0))
        self.assertAlmostEqual(z1, 0.0)

    def test_pauli_exclusion_guard_on_layer_a_center(self):
        """
        Verify strict Pauli Exclusion Guard:
        No two solid entities can occupy the exact same hex on layer A_CENTER.
        """
        # Blueprint with collision on A_CENTER at (0, 0)
        invalid_bp = {
            "cells": [{"q": 0, "r": 0}],
            "entities": [
                {"id": "standee_1", "q": 0, "r": 0, "layer": "A_CENTER"},
                {"id": "standee_2", "q": 0, "r": 0, "layer": "A_CENTER"} # Collision!
            ]
        }
        with self.assertRaises(PauliExclusionViolation):
            self.bridge.compile_blueprint(invalid_bp)

        # Valid blueprint: entities occupy different layers at (0, 0)
        valid_bp = {
            "cells": [{"q": 0, "r": 0}],
            "entities": [
                {"id": "base_glyph", "q": 0, "r": 0, "layer": "A_BASE"},
                {"id": "aura_ring", "q": 0, "r": 0, "layer": "A_AURA_RING"},
                {"id": "standee_3", "q": 0, "r": 0, "layer": "A_CENTER"},
                {"id": "overhead_rune", "q": 0, "r": 0, "layer": "A_OVERHEAD"}
            ]
        }
        res = self.bridge.compile_blueprint(valid_bp)
        self.assertEqual(res["status"], "COMPILED")
        self.assertEqual(res["total_entities"], 4)

    def test_ohm_law_physics_and_zero_division_guard(self):
        """
        Verify Ohm's Law physics: I = V / (R_target + R_reactions)
        and zero-division protection (R_eff >= 0.1).
        """
        self.bridge.compile_blueprint(self.sample_blueprint)

        # Case 1: Standard Ohm's Law: V = 6.0, R = 2.0 -> I = 3.0, damage = 3, HP = 6 - 3 = 3
        instr = {
            "type": "APPLY_OHM_VOLTAGE",
            "target_id": "citadel_turret",
            "applied_voltage": 6.0,
            "confidence": 0.95
        }
        res = self.bridge.resolve_engine_state(instr)
        self.assertEqual(res["status"], "OK")
        self.assertAlmostEqual(res["calculated_current"], 3.0)
        self.assertEqual(res["damage_dealt"], 3)
        self.assertEqual(res["target_new_hp"], 3)

        # Case 2: Zero-division guard: Target base resistance set to 0.0
        self.bridge.entities["citadel_turret"]["base_resistance"] = 0.0
        self.bridge.entities["citadel_turret"]["hp"] = 6

        instr_zero = {
            "type": "APPLY_OHM_VOLTAGE",
            "target_id": "citadel_turret",
            "applied_voltage": 0.5,
            "confidence": 1.0
        }
        res_zero = self.bridge.resolve_engine_state(instr_zero)
        self.assertGreaterEqual(res_zero["effective_resistance"], MIN_EFFECTIVE_RESISTANCE)
        expected_current = 0.5 / MIN_EFFECTIVE_RESISTANCE # 5.0
        self.assertAlmostEqual(res_zero["calculated_current"], expected_current)

    def test_saturation_arithmetic_and_memory_integrity(self):
        """Verify damage saturation arithmetic prevents HP underflow below 0 and overflow above 6."""
        self.bridge.compile_blueprint(self.sample_blueprint)

        # Overkill voltage: V = 100.0, R = 1.0 -> damage = 100
        instr_overkill = {
            "type": "APPLY_OHM_VOLTAGE",
            "target_id": "scout_drone",
            "applied_voltage": 100.0,
            "confidence": 1.0
        }
        res = self.bridge.resolve_engine_state(instr_overkill)
        self.assertEqual(res["target_new_hp"], 0)
        self.assertFalse(self.bridge.entities["scout_drone"]["conduct"]["is_active"])

        # Check that neutralized unit vacated occupancy on A_CENTER
        self.assertNotIn("1,0", self.bridge.center_occupancy)

    def test_lifo_interrupt_stack(self):
        """
        Verify LIFO Interrupt Stack processes reaction resistance and counter-voltage
        in Last-In-First-Out order within the validity window.
        """
        self.bridge.compile_blueprint(self.sample_blueprint)

        # Register reaction 1 (Reactive Armor: +2.0 R)
        self.bridge.resolve_engine_state({
            "type": "REGISTER_INTERRUPT",
            "interrupt": {
                "id": "react_armor",
                "action_type": "REACTIVE_ARMOR",
                "resistance_delta": 2.0,
                "voltage_delta": 0.0,
                "valid_ticks": 5
            }
        })

        # Register reaction 2 (Grounding Field: +3.0 R, -1.0 V)
        self.bridge.resolve_engine_state({
            "type": "REGISTER_INTERRUPT",
            "interrupt": {
                "id": "ground_field",
                "action_type": "GROUNDING",
                "resistance_delta": 3.0,
                "voltage_delta": -1.0,
                "valid_ticks": 5
            }
        })

        # Target has base_resistance = 2.0.
        # Cumulative R = Base(2.0) + React1(2.0) + React2(3.0) = 7.0
        # Applied V = 15.0 - 1.0 = 14.0
        # Current I = 14.0 / 7.0 = 2.0 -> damage = 2 -> HP = 6 - 2 = 4
        instr = {
            "type": "APPLY_OHM_VOLTAGE",
            "target_id": "citadel_turret",
            "applied_voltage": 15.0,
            "confidence": 0.95
        }
        res = self.bridge.resolve_engine_state(instr)
        self.assertEqual(res["interrupts_processed"], 2)
        self.assertAlmostEqual(res["effective_resistance"], 7.0)
        self.assertAlmostEqual(res["effective_voltage"], 14.0)
        self.assertAlmostEqual(res["calculated_current"], 2.0)
        self.assertEqual(res["damage_dealt"], 2)
        self.assertEqual(res["target_new_hp"], 4)

    def test_binary_packed_buffer_layout(self):
        """
        Verify packed instance buffer satisfies 96 bytes (24 float32s) per instance:
        - 16 floats: 4x4 transform matrix
        - 4 floats: PBR color & roughness
        - 4 floats: custom shader flags & layer ID & conduct
        """
        self.bridge.compile_blueprint(self.sample_blueprint)
        buf = self.bridge.generate_packed_buffer(hex_radius=1.0)

        # 4 cells + 3 entities = 7 instances
        expected_bytes = 7 * 96
        self.assertEqual(len(buf), expected_bytes)

        # Inspect first cell at offset 0 (Citadel cell at q=0, r=0)
        m00 = struct.unpack_from("<f", buf, 0)[0]
        posX = struct.unpack_from("<f", buf, 48)[0]
        posZ = struct.unpack_from("<f", buf, 56)[0]
        self.assertAlmostEqual(m00, 1.0)
        self.assertAlmostEqual(posX, 0.0)
        self.assertAlmostEqual(posZ, 0.0)

        # Inspect layer_id at offset 80 (float 20)
        layer_id = struct.unpack_from("<f", buf, 80)[0]
        self.assertEqual(layer_id, LAYER_BASE)

    def test_conduct_code_bridge_and_fallback(self):
        """Verify Conduct Code Bridge redirects to safe fallback upon low confidence or hazard."""
        self.bridge.compile_blueprint(self.sample_blueprint)

        # Low confidence (< 0.85)
        instr_low_conf = {
            "type": "APPLY_OHM_VOLTAGE",
            "target_id": "citadel_turret",
            "applied_voltage": 10.0,
            "confidence": 0.65
        }
        res_low = self.bridge.resolve_engine_state(instr_low_conf)
        self.assertEqual(res_low["status"], "FALLBACK_TRIGGERED")
        self.assertEqual(res_low["fallback_action"], "FALLBACK_DEFENSIVE_GROUNDING")

        # Hazard detected
        instr_hazard = {
            "type": "APPLY_OHM_VOLTAGE",
            "target_id": "citadel_turret",
            "applied_voltage": 10.0,
            "confidence": 1.0,
            "is_hazardous": True
        }
        res_haz = self.bridge.resolve_engine_state(instr_hazard)
        self.assertEqual(res_haz["status"], "FALLBACK_TRIGGERED")

    def test_adaptive_render_telemetry_profiles(self):
        """Verify hardware profile selection based on FPS, CPU load, and temperature."""
        # Profile 1: Minimal Wireframe (Critical state: low FPS or high temp)
        p1 = self.bridge.evaluate_hardware_profile(fps=22.0, cpu_load=88.0, temp_c=82.0)
        self.assertEqual(p1, "MINIMAL_WIREFRAME")

        # Profile 2: Standard Living Standees (Balanced state)
        p2 = self.bridge.evaluate_hardware_profile(fps=45.0, cpu_load=60.0, temp_c=70.0)
        self.assertEqual(p2, "STANDARD_LIVING_STANDEES")

        # Profile 3: AAA Cinematic (High headroom state)
        p3 = self.bridge.evaluate_hardware_profile(fps=60.0, cpu_load=30.0, temp_c=55.0)
        self.assertEqual(p3, "AAA_CINEMATIC")


if __name__ == "__main__":
    unittest.main()
