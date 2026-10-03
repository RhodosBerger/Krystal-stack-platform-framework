# ==============================================================================
# KRYSTAL-STACK: UNIT & INTEGRATION TESTS FOR THEORETICAL FORMULAS & GODOT EXTENSIONS
# ==============================================================================

import unittest
import math
import json
from krystal_web_hub.economic_engine import (
    hex_riemannian_distance,
    hex_to_world_cartesian,
    calculate_ballistic_apex_height,
    sample_ballistic_bezier_hermite_curve,
    hex_prism_sdf,
    crystal_spire_sdf,
    polynomial_smooth_min,
    update_hex_biome_transition,
    fuse_cards,
    compact_godot_ast
)

class TestGodotTheoreticalFormulas(unittest.TestCase):

    # --------------------------------------------------------------------------
    # FORMULA 1: HEX-RIEMANNIAN METRIC TENSOR & AXIAL DISTANCE
    # --------------------------------------------------------------------------
    def test_hex_riemannian_distance(self):
        # Identity
        self.assertEqual(hex_riemannian_distance([0, 0], [0, 0]), 0)
        # Immediate adjacent (all 6 directions)
        for offset in [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]]:
            self.assertEqual(hex_riemannian_distance([0, 0], offset), 1)
        # Symmetry
        self.assertEqual(hex_riemannian_distance([1, 2], [-2, 3]), hex_riemannian_distance([-2, 3], [1, 2]))
        # Multi-hex step
        self.assertEqual(hex_riemannian_distance([0, 0], [2, 2]), 4)
        self.assertEqual(hex_riemannian_distance([0, 0], [0, 4]), 4)

    def test_hex_to_world_cartesian(self):
        x0, y0, z0 = hex_to_world_cartesian(0, 0, radius=1.5)
        self.assertEqual(x0, 0.0)
        self.assertEqual(z0, 0.0)
        self.assertAlmostEqual(y0, 0.0, places=2)

        x1, y1, z1 = hex_to_world_cartesian(1, 0, radius=1.5)
        self.assertGreater(x1, 0.0)
        self.assertEqual(z1, 0.0)

    # --------------------------------------------------------------------------
    # FORMULA 2: AERODYNAMIC BALLISTIC BEZIER-HERMITE TRAJECTORY
    # --------------------------------------------------------------------------
    def test_calculate_ballistic_apex_height(self):
        h_melee = calculate_ballistic_apex_height(1, attack_type="melee")
        self.assertEqual(h_melee, 0.50)

        h_self = calculate_ballistic_apex_height(0, attack_type="self")
        self.assertEqual(h_self, 0.20)

        h_ranged_1 = calculate_ballistic_apex_height(1, attack_type="ranged")
        h_ranged_3 = calculate_ballistic_apex_height(3, attack_type="ranged")
        h_ranged_5 = calculate_ballistic_apex_height(5, attack_type="ranged")

        self.assertGreater(h_ranged_3, h_ranged_1)
        self.assertLessEqual(h_ranged_5, 4.20)

    def test_sample_ballistic_bezier_hermite_curve(self):
        p0 = [0.0, 0.35, 2.0]
        p3 = [0.0, 0.20, -3.0]
        curve = sample_ballistic_bezier_hermite_curve(p0, p3, distance=3, attack_type="ranged", num_samples=16)

        self.assertIn("samples", curve)
        self.assertEqual(len(curve["samples"]), 17) # 0 to 16

        # Start and end positions match p0 and p3
        self.assertAlmostEqual(curve["samples"][0][0], p0[0], places=2)
        self.assertAlmostEqual(curve["samples"][0][1], p0[1], places=2)
        self.assertAlmostEqual(curve["samples"][0][2], p0[2], places=2)

        self.assertAlmostEqual(curve["samples"][-1][0], p3[0], places=2)
        self.assertAlmostEqual(curve["samples"][-1][1], p3[1], places=2)
        self.assertAlmostEqual(curve["samples"][-1][2], p3[2], places=2)

        # Midpoint apex is elevated
        mid_y = curve["samples"][8][1]
        self.assertGreater(mid_y, max(p0[1], p3[1]))

        # Impact tangent is normalized (unit vector)
        tangent = curve["impact_tangent"]
        t_len = math.sqrt(sum(t**2 for t in tangent))
        self.assertAlmostEqual(t_len, 1.0, places=3)

    # --------------------------------------------------------------------------
    # FORMULA 3: ANALYTIC SDF SOLID GEOMETRY & SMOOTH MINIMUM
    # --------------------------------------------------------------------------
    def test_hex_prism_sdf(self):
        # Center inside hex prism has negative distance
        d_center = hex_prism_sdf([0.0, 0.0, 0.0], radius=1.5, height=0.5)
        self.assertLess(d_center, 0.0)

        # Point far outside has positive distance
        d_outside = hex_prism_sdf([5.0, 2.0, 5.0], radius=1.5, height=0.5)
        self.assertGreater(d_outside, 0.0)

    def test_crystal_spire_sdf(self):
        d_center = crystal_spire_sdf([0.0, 0.0, 0.0], scale=1.2)
        self.assertLess(d_center, 0.0)

        d_outside = crystal_spire_sdf([2.0, 2.0, 2.0], scale=1.2)
        self.assertGreater(d_outside, 0.0)

    def test_polynomial_smooth_min(self):
        a, b = 2.0, 2.1
        smin_val = polynomial_smooth_min(a, b, k=0.5)
        # Smooth minimum must be slightly less than exact min(a, b)
        self.assertLess(smin_val, min(a, b))

    # --------------------------------------------------------------------------
    # FORMULA 4: DYNAMIC HEX TERRAFORMING & BIOME PHASE TRANSITION
    # --------------------------------------------------------------------------
    def test_biome_transition_dynamics(self):
        neutral_vector = (0.333, 0.333, 0.334)
        # Impact with crystal meteor at distance 0 (direct hit)
        result = update_hex_biome_transition(neutral_vector, spell_element="crystal", impact_distance=0)
        vec = result["updated_vector"]

        # Partition of unity: sum is approximately 1.0
        self.assertAlmostEqual(sum(vec), 1.0, places=2)
        # Crystal component increases significantly
        self.assertGreater(vec[0], neutral_vector[0])
        self.assertGreater(vec[0], vec[1])
        self.assertGreater(vec[0], vec[2])
        self.assertEqual(result["dominant_biome"], "crystal_peaks")

        # Toxic impact at distance 0
        toxic_res = update_hex_biome_transition(neutral_vector, spell_element="toxic", impact_distance=0)
        self.assertGreater(toxic_res["updated_vector"][1], neutral_vector[1])
        self.assertEqual(toxic_res["dominant_biome"], "toxic_marsh")

    # --------------------------------------------------------------------------
    # FORMULA 5: CARD FUSION TENSOR ALGEBRA
    # --------------------------------------------------------------------------
    def test_card_fusion_tensor(self):
        card1 = {
            "id": "crystal_meteor",
            "name": "Kryštálový Meteor",
            "tribe": "crystal",
            "cost": 3,
            "hp_delta": -2,
            "min_range": 1,
            "max_range": 4
        }
        card2 = {
            "id": "acid_slime",
            "name": "Kyslý Sliz",
            "tribe": "toxic",
            "cost": 1,
            "hp_delta": -1,
            "min_range": 1,
            "max_range": 3
        }

        fused = fuse_cards(card1, card2)
        self.assertEqual(fused["synergy_score"], 0.60) # Crystal x Toxic synergy
        self.assertIn("Fúzia", fused["name"])
        # Cost is discounted by synergy
        self.assertLessEqual(fused["cost"], card1["cost"] + card2["cost"])
        # Power is amplified
        self.assertLess(fused["hp_delta"], card1["hp_delta"])
        # Range envelope merges
        self.assertEqual(fused["min_range"], 1)
        self.assertEqual(fused["max_range"], 4)

    # --------------------------------------------------------------------------
    # FORMULA 6: GODOT SCENE AST COMPACTION
    # --------------------------------------------------------------------------
    def test_ast_compaction_ratio(self):
        sample_godot_tree = {
            "type": "Spatial",
            "name": "RootScene",
            "properties": {"position": [0, 0, 0]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "HexFloor",
                    "properties": {"position": [0, -0.2, 0], "mesh": "hex_tile.obj"},
                    "children": [
                        {"type": "Particles", "name": "Aura", "properties": {"position": [0, 0.5, 0]}},
                        {"type": "OmniLight", "name": "Glow", "properties": {"position": [0, 1.0, 0]}}
                    ]
                }
            ]
        }

        compaction = compact_godot_ast(sample_godot_tree)
        self.assertGreater(compaction["raw_size_bytes"], compaction["compact_size_bytes"])
        self.assertGreaterEqual(compaction["compression_ratio"], 1.5)
        self.assertEqual(compaction["token_count"], 4)
        self.assertIn("RootScene", compaction["compact_payload"])
        self.assertIn("HexFloor", compaction["compact_payload"])

if __name__ == '__main__':
    unittest.main()
