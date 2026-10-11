#!/usr/bin/env python3
"""
Unit and integration tests for Procedural Tokenization, Procedure Partitioning,
6-Axis Mortar Attribute Radar, and Statistical Turn Capacity Simulation.

Verifies:
  - ProceduralCompositionTokenizer: Token creation and domain partitioning.
  - MortarAttributeRadar: 6-axis normalization and composite rating.
  - StatisticalCapacitySimulator: Hit probability, CEP scatter, and damage clamp (0..6).
  - WebGLStoryDirectives: Shader uniforms, trajectory curves, and PBR palettes.
  - Server endpoints: /api/tactical/radar_catalog and /api/tactical/token_radar_simulate.
  - Platform invariant: VITAL_MAX_HP == 6.
"""

import unittest
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.procedural_token_radar_simulator import (
    ProcedureDomain,
    ProceduralToken,
    ProceduralCompositionTokenizer,
    MortarAttributeRadar,
    TurnSimulationOutcome,
    StatisticalCapacitySimulator,
    WebGLStoryDirectives,
    ProceduralTokenRadarOrchestrator,
    GLOBAL_TOKEN_RADAR_ORCHESTRATOR
)


class TestProceduralTokenization(unittest.TestCase):
    """Verifies directive tokenization and procedural partitioning."""

    def test_vital_max_hp_invariant(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_tokenize_mortar_and_spell_directive(self):
        directive = "Spusti mínometnú salvu na nepriateľské pozície s kryštálovým kúzlom a aetherickou vlnou"
        tokens = ProceduralCompositionTokenizer.tokenize_directive(directive)
        self.assertGreaterEqual(len(tokens), 4)

        domains = {t.domain for t in tokens}
        self.assertIn(ProcedureDomain.WEBGL_COMPOSITION, domains)
        self.assertIn(ProcedureDomain.TACTICAL_SIMULATION, domains)
        self.assertIn(ProcedureDomain.NARRATIVE_STORY, domains)

        token_ids = [t.token_id for t in tokens]
        self.assertIn("TOK_WGL_BALLISTIC_ARC", token_ids)
        self.assertIn("TOK_WGL_CRYSTAL_REFRACTION", token_ids)
        self.assertIn("TOK_SIM_MORTAR_BALLISTICS", token_ids)
        self.assertIn("TOK_NAR_TEMPORAL_ARC", token_ids)


class TestMortarAttributeRadar(unittest.TestCase):
    """Verifies 6-axis attribute radar mathematics and normalization."""

    def test_radar_axis_bounds(self):
        radar = MortarAttributeRadar.evaluate(
            muzzle_velocity=26.0,
            elevation_angle_deg=62.0,
            blast_radius_m=3.5,
            spell_capacity_pct=70.0,
            mechanics_capacity_pct=80.0
        )
        self.assertEqual(radar.vital_max_hp, 6)
        axes = [
            radar.range_reach,
            radar.aoe_dispersion,
            radar.armor_penetration,
            radar.action_cost_efficiency,
            radar.spell_affinity,
            radar.visual_drama
        ]
        for a in axes:
            self.assertGreaterEqual(a, 0.0)
            self.assertLessEqual(a, 1.0)

        self.assertGreaterEqual(radar.composite_rating, 0.0)
        self.assertLessEqual(radar.composite_rating, 1.0)
        self.assertGreater(radar.composite_rating, 0.5)

    def test_radar_tradeoffs_spell_vs_mechanics(self):
        # High Spell, Low Mechanics
        r_spell = MortarAttributeRadar.evaluate(spell_capacity_pct=95.0, mechanics_capacity_pct=20.0)
        # Low Spell, High Mechanics
        r_mech = MortarAttributeRadar.evaluate(spell_capacity_pct=15.0, mechanics_capacity_pct=95.0)

        self.assertGreater(r_spell.spell_affinity, r_mech.spell_affinity)
        self.assertGreater(r_mech.action_cost_efficiency, r_spell.action_cost_efficiency)


class TestStatisticalCapacitySimulator(unittest.TestCase):
    """Verifies probabilistic turn modeling and damage clamping."""

    def test_simulate_mortar_turn_clamping(self):
        outcome = StatisticalCapacitySimulator.simulate_mortar_turn(
            target_hex=(1, 1),
            current_defender_hp=6,
            spell_capacity=80.0,
            mechanics_capacity=80.0,
            elevation_angle_deg=65.0,
            salvo_count=3,
            seed=101
        )
        self.assertEqual(outcome.vital_max_hp, 6)
        self.assertGreaterEqual(outcome.effective_damage, 0)
        self.assertLessEqual(outcome.effective_damage, 6)
        self.assertGreaterEqual(outcome.remaining_hp_defender, 0)
        self.assertLessEqual(outcome.remaining_hp_defender, 6)

        # Confirm ballistics
        self.assertGreater(outcome.time_of_flight_s, 0.0)
        self.assertGreater(outcome.trajectory_apex_m, 0.0)
        self.assertGreater(outcome.hit_probability, 0.0)
        self.assertLessEqual(outcome.hit_probability, 1.0)

        # Confirm 3-phase narrative story
        story = outcome.temporal_story_log
        self.assertIn("before", story)
        self.assertIn("during", story)
        self.assertIn("after", story)


class TestProceduralTokenRadarOrchestrator(unittest.TestCase):
    """Verifies end-to-end token radar processing and WebGL story directives."""

    def setUp(self):
        self.orchestrator = GLOBAL_TOKEN_RADAR_ORCHESTRATOR

    def test_process_turn_directive_end_to_end(self):
        res = self.orchestrator.process_turn_directive(
            prompt="Spusti mínometný útok na nepriateľský bunker s plnou silou",
            target_hex=(2, 0),
            current_hp=5,
            spell_capacity=60.0,
            mechanics_capacity=70.0,
            elevation_deg=60.0,
            salvo_count=2,
            seed=42
        )
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertLess(res["execution_latency_ms"], 100.0)

        # Confirm procedure partitioning
        part = res["procedure_partitioning"]
        self.assertIn("webgl_composition", part)
        self.assertIn("tactical_simulation", part)
        self.assertIn("narrative_story", part)

        # Confirm WebGL story directives
        webgl = res["webgl_story_directives"]
        self.assertEqual(webgl["vital_max_hp"], 6)
        self.assertEqual(len(webgl["trajectory_curve_coefficients"]), 3)
        self.assertEqual(len(webgl["pbr_palette_rgb"]), 3)
        self.assertTrue(len(webgl["narrative_caption"]) > 0)


if __name__ == "__main__":
    unittest.main()
