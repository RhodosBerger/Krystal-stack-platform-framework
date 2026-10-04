"""Unit tests for Procedural Vegetation Strata and Master Phenomena Engine.

Validates:
1. 6-Tier Botanical Stratification & VITAL_MAX_HP = 6 Invariant.
2. 10 Canonical Phenomena Specifications & VITAL_MAX_HP = 6.
3. Vogel Phyllotaxis generation (137.508° golden angle).
4. L-System fractal generation & 3D branch segments.
5. Master phenomena resonance simulation & energy flux.
6. Godot MultiMesh Transform3D buffer serialization (16 floats per instance).
"""

import unittest
import math
from krystal_web_hub.economic_engine.procedural_vegetation_and_phenomena import (
    ProceduralVegetationAndPhenomenaEngine,
    GLOBAL_VEGETATION_PHENOMENA_ENGINE,
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    GOLDEN_ANGLE_DEG,
    CANONICAL_STRATA,
    CANONICAL_PHENOMENA,
    BotanicalStratum,
    MasterPhenomenon,
    BotanicalInstance,
)


class TestProceduralVegetationAndPhenomena(unittest.TestCase):
    def setUp(self):
        self.engine = ProceduralVegetationAndPhenomenaEngine()

    def test_botanical_strata_invariants(self):
        """Verifies exactly 6 botanical strata with VITAL_MAX_HP = 6."""
        self.assertEqual(len(self.engine.strata), 6)
        for key, stratum in self.engine.strata.items():
            self.assertEqual(stratum.vital_max_hp, VITAL_MAX_HP)
            self.assertEqual(stratum.vital_max_hp, 6)
            self.assertGreater(stratum.height_range[1], stratum.height_range[0])
            self.assertGreaterEqual(stratum.light_affinity, 0.0)
            self.assertLessEqual(stratum.light_affinity, 1.0)

    def test_master_phenomena_invariants(self):
        """Verifies exactly 10 master phenomena with VITAL_MAX_HP = 6."""
        self.assertEqual(len(self.engine.phenomena), 10)
        for key, phenom in self.engine.phenomena.items():
            self.assertEqual(phenom.vital_max_hp, VITAL_MAX_HP)
            self.assertEqual(phenom.vital_max_hp, 6)
            self.assertGreater(phenom.frequency_hz, 0.0)
            self.assertGreater(phenom.wavelength_nm, 0.0)
            self.assertGreater(phenom.volumetric_density, 0.0)
            self.assertIn(phenom.category, [
                "atmospheric_optical", "electromagnetic_plasma",
                "biological_spore", "geological_telluric", "cosmic_occult"
            ])

    def test_vogel_phyllotaxis_generation(self):
        """Verifies golden ratio phyllotaxis coordinate generation."""
        instances = self.engine.generate_phyllotaxis_points(count=80, spread=2.0)
        self.assertEqual(len(instances), 80)
        for i, inst in enumerate(instances):
            self.assertEqual(inst.vital_max_hp, 6)
            self.assertEqual(inst.instance_id, i)
            self.assertIsInstance(inst.x, float)
            self.assertIsInstance(inst.z, float)
            self.assertIsInstance(inst.height, float)
            self.assertGreater(inst.scale_y, 0.0)
            # Distance from origin increases with sqrt(n)
            dist = math.sqrt(inst.x**2 + inst.z**2)
            if i > 0:
                expected_dist = 2.0 * math.sqrt(i)
                self.assertAlmostEqual(dist, expected_dist, delta=1e-4)

    def test_lsystem_fractal_generation(self):
        """Verifies L-System branch string derivation and segment interpretation."""
        data = self.engine.generate_lsystem_tree(iterations=3, length=3.0, angle_deg=25.0)
        self.assertIn("branch_string", data)
        self.assertIn("segments", data)
        self.assertGreater(len(data["branch_string"]), 20)
        self.assertGreater(len(data["segments"]), 5)
        self.assertEqual(data["vital_max_hp"], 6)

        # Check segments have valid start and end 3D coordinates
        for seg in data["segments"]:
            self.assertEqual(len(seg["start"]), 3)
            self.assertEqual(len(seg["end"]), 3)
            self.assertGreater(seg["radius"], 0.0)

    def test_phenomenon_resonance_simulation(self):
        """Verifies simulation of atmospheric and energetic resonance for all 10 phenomena."""
        res = self.engine.simulate_phenomenon_resonance(active_phenomena_keys=None, global_potency=1.2)
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertEqual(res["active_count"], 10)
        self.assertGreater(res["aggregate_energy_flux_w_m2"], 0.0)
        self.assertIn("aetheric_aurora_stream", res["phenomena_readouts"])
        self.assertIn("ball_lightning_vortex", res["phenomena_readouts"])

        aurora = res["phenomena_readouts"]["aetheric_aurora_stream"]
        self.assertEqual(aurora["category"], "atmospheric_optical")
        self.assertEqual(aurora["vital_max_hp"], 6)
        self.assertGreater(aurora["energy_flux_w_m2"], 0.0)

    def test_godot_multimesh_buffer_export(self):
        """Verifies serialization of Transform3D multi-mesh float buffer (16 floats per instance)."""
        buffer = self.engine.export_godot_multimesh_buffer(count=40)
        self.assertEqual(len(buffer), 40 * 16)
        for val in buffer:
            self.assertIsInstance(val, float)
            self.assertFalse(math.isnan(val))
            self.assertFalse(math.isinf(val))

    def test_state_manifest(self):
        """Verifies comprehensive state snapshot contains complete catalog."""
        manifest = self.engine.get_state_manifest()
        self.assertEqual(manifest["vital_max_hp_rule"], 6)
        self.assertEqual(len(manifest["strata"]), 6)
        self.assertEqual(len(manifest["phenomena"]), 10)
        self.assertIn("phyllotaxis_golden_angle_deg", manifest)
        self.assertAlmostEqual(manifest["phyllotaxis_golden_angle_deg"], GOLDEN_ANGLE_DEG, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
