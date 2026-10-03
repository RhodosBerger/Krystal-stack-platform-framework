import unittest
import os
import time

from krystal_web_hub.economic_engine.visual_phenomena_and_totem_anomalies import (
    VisualPhenomenonType,
    SpellProjectionType,
    SectorAnomalyType,
    TotemStatus,
    VisualPhenomenaEngine,
    SpellProjectionEngine,
    AnomalyDetectorSensorArray,
    SectorTotemManager
)
from krystal_janet.janet_bridge import JanetValidator


class TestVisualPhenomenaAndTotemAnomalies(unittest.TestCase):
    """
    Unit test suite verifying:
    1. Procedural Visual Phenomena Shaders & GPUParticles3D emitters
    2. Geometric Spell Projections (Conical AoE, Radial, Ballistic Beam)
    3. Spatial Anomaly Detector Array (Gradient Divergence, Ley-line Shifts)
    4. Sector Totem Manager (5 Sectors, Vital 6 Max HP Invariant, Overcharge, Attunement)
    5. Janet DSL validation for totem_anomalies_and_projections.janet
    """

    def setUp(self):
        self.phenomena_engine = VisualPhenomenaEngine()
        self.projection_engine = SpellProjectionEngine()
        self.detector = AnomalyDetectorSensorArray()
        self.totem_manager = SectorTotemManager()

    # ── 1. PROCEDURAL VISUAL PHENOMENA ────────────────────────────────────────
    def test_visual_phenomena_generation(self):
        # Chromatic Aberration Burst
        res_chroma = self.phenomena_engine.generate_phenomenon(
            phenomenon_type=VisualPhenomenonType.CHROMATIC_ABERRATION_BURST.value,
            coordinates=(1.0, 2.0, 0.5),
            intensity=0.9,
            resonance_hz=432.0
        )
        self.assertIn("godot_shader_uniforms", res_chroma)
        self.assertIn("distortion_amplitude", res_chroma["godot_shader_uniforms"])
        self.assertIn("chroma_shift_rgb", res_chroma["godot_shader_uniforms"])
        self.assertEqual(res_chroma["gpu_particles_spec"]["particle_type"], "optical_lens_dust")

        # St. Elmo's Plasma Discharge
        res_plasma = self.phenomena_engine.generate_phenomenon(
            phenomenon_type=VisualPhenomenonType.ST_ELMOS_PLASMA_DISCHARGE.value,
            coordinates=(0.0, 0.0, 3.0),
            intensity=0.8,
            resonance_hz=864.0
        )
        self.assertIn("plasma_lumens", res_plasma["godot_shader_uniforms"])
        self.assertEqual(res_plasma["gpu_particles_spec"]["emission_shape"], "totem_crown_emitter")

    def test_phenomena_catalog_completeness(self):
        catalog = self.phenomena_engine.get_all_phenomena_catalog()
        self.assertEqual(len(catalog), 5)
        for p_type in VisualPhenomenonType:
            self.assertIn(p_type.value, catalog)
            self.assertIn("godot_shader_uniforms", catalog[p_type.value])
            self.assertIn("gpu_particles_spec", catalog[p_type.value])

    # ── 2. SPELL PROJECTIONS & HIT-TESTING ─────────────────────────────────────
    def test_conical_aoe_projection(self):
        totems = [
            {"id": "totem_in_cone", "name": "Target Totem", "element": "crystal", "world_pos": (0.0, 5.0)},
            {"id": "totem_behind", "name": "Behind Totem", "element": "crystal", "world_pos": (0.0, -5.0)},
            {"id": "totem_out_of_range", "name": "Far Totem", "element": "crystal", "world_pos": (0.0, 25.0)}
        ]
        # Heading 90 deg = +Y direction
        proj = self.projection_engine.project_spell(
            spell_key="frost_crystal_nova",
            caster_origin=(0.0, 0.0),
            target_heading_deg=90.0,
            totems_in_sector=totems
        )
        self.assertEqual(proj["spell_key"], "frost_crystal_nova")
        self.assertEqual(proj["totems_affected_count"], 1)
        self.assertEqual(proj["affected_totems"][0]["totem_id"], "totem_in_cone")
        self.assertGreater(proj["affected_totems"][0]["hit_effectiveness"], 0.0)
        self.assertTrue(proj["affected_totems"][0]["affinity_matched"])

    def test_radial_blast_projection(self):
        totems = [
            {"id": "totem_close", "name": "Near Totem", "element": "toxic", "world_pos": (2.0, 2.0)},
            {"id": "totem_distant", "name": "Far Totem", "element": "toxic", "world_pos": (12.0, 12.0)}
        ]
        proj = self.projection_engine.project_spell(
            spell_key="toxic_miasma_eruption",
            caster_origin=(0.0, 0.0),
            target_heading_deg=0.0,
            totems_in_sector=totems
        )
        self.assertEqual(proj["totems_affected_count"], 1)
        self.assertEqual(proj["affected_totems"][0]["totem_id"], "totem_close")

    # ── 3. SPATIAL ANOMALY DETECTION ──────────────────────────────────────────
    def test_trigger_and_scan_anomaly(self):
        sector_id = "sector_north_crystal"
        totems = self.totem_manager.get_totems(sector_id)
        self.assertGreaterEqual(len(totems), 1)

        # Trigger Void Corruption Zone
        anom = self.detector.trigger_anomaly(
            sector_id=sector_id,
            anomaly_type=SectorAnomalyType.VOID_CORRUPTION_ZONE.value,
            epicenter_coords=(0.0, -8.66),
            magnitude=0.85,
            duration_sec=30.0
        )
        self.assertEqual(anom["sector_id"], sector_id)
        self.assertGreater(anom["frequency_drift_hz"], 30.0)
        self.assertGreater(anom["gradient_divergence"], 2.0)

        # Scan sector
        scan = self.detector.scan_sector(sector_id, totems)
        self.assertEqual(scan["anomalies_detected_count"], 1)
        reading = scan["sensor_readings"][0]
        self.assertEqual(reading["severity"], "CRITICAL_CATACLYSM")
        self.assertGreaterEqual(len(reading["threatened_totems"]), 1)
        self.assertLess(scan["sector_stability_index"], 80.0)

    # ── 4. SECTOR TOTEM MANAGER & VITAL INVARIANTS ─────────────────────────────
    def test_default_totems_and_vital_6_hp_invariant(self):
        all_totems = self.totem_manager.get_totems()
        self.assertEqual(len(all_totems), 5)
        for t in all_totems:
            self.assertEqual(t["max_hp"], 6, "Every totem must adhere to the 6 Max HP vital invariant!")
            self.assertLessEqual(t["hp"], 6)
            self.assertGreaterEqual(t["hp"], 1)
            self.assertIn("base_frequency_hz", t)
            self.assertIn("ward_shield", t)

    def test_spell_affinity_overcharge(self):
        totem_id = "totem_north_crystal"  # Element: crystal, base charge: 75%
        res = self.totem_manager.apply_spell_impact_to_totem(
            totem_id=totem_id,
            spell_element="crystal",
            potency=4.0,
            hit_factor=0.9
        )
        self.assertTrue(res["success"])
        self.assertTrue(res["is_aligned"])
        self.assertEqual(res["new_status"], TotemStatus.OVERCHARGED.value)
        self.assertGreaterEqual(res["new_resonance"], 85.0)

    def test_opposing_spell_ward_damage_and_vital_hp_guard(self):
        totem_id = "totem_north_crystal"
        # Opposing toxic spell against crystal totem
        res = self.totem_manager.apply_spell_impact_to_totem(
            totem_id=totem_id,
            spell_element="toxic",
            potency=10.0,
            hit_factor=1.0
        )
        self.assertTrue(res["success"])
        self.assertFalse(res["is_aligned"])
        totem = res["totem"]
        # Ward shield reduced to 0
        self.assertEqual(totem["ward_shield"], 0)
        # HP must never drop below 1 or violate 6 max HP
        self.assertGreaterEqual(totem["hp"], 1)
        self.assertLessEqual(totem["hp"], 6)

    def test_anomaly_corruption_and_attunement_restoration(self):
        totem_id = "totem_north_crystal"
        # Apply void corruption
        flux_res = self.totem_manager.apply_anomaly_flux_to_totem(
            totem_id=totem_id,
            anomaly_type=SectorAnomalyType.VOID_CORRUPTION_ZONE.value,
            magnitude=0.9
        )
        self.assertEqual(flux_res["totem_state"]["status"], TotemStatus.CORRUPTED.value)
        self.assertNotEqual(flux_res["totem_state"]["current_frequency_hz"], flux_res["totem_state"]["base_frequency_hz"])

        # Player channels energy to attune and cleanse
        attune_res = self.totem_manager.attune_totem(
            totem_id=totem_id,
            caster_tribe="crystal",
            channel_energy=30.0
        )
        self.assertTrue(attune_res["success"])
        self.assertEqual(attune_res["current_frequency_hz"], 432.0)
        self.assertIn(attune_res["status"], [TotemStatus.ATTUNED.value, TotemStatus.OVERCHARGED.value])
        self.assertLessEqual(attune_res["hp"], 6)

    # ── 5. JANET DSL VALIDATION ───────────────────────────────────────────────
    def test_janet_totem_anomalies_and_projections_syntax(self):
        janet_path = os.path.join(
            os.path.dirname(__file__), "..", "krystal_janet", "totem_anomalies_and_projections.janet"
        )
        self.assertTrue(os.path.exists(janet_path), f"Janet file not found: {janet_path}")
        val_res = JanetValidator.validate_file(janet_path)
        self.assertTrue(val_res["valid"], f"Janet validation failed: {val_res.get('error')}")
        self.assertGreaterEqual(val_res["line_count"], 40)


if __name__ == "__main__":
    unittest.main()
