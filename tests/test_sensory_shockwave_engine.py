#!/usr/bin/env python3
"""
Unit and integration tests for Sensory Shockwave & Impulse Task Processing Engine.
Validates:
  1. Sensory modalities (Kinetic Overpressure, Acoustic, Seismic, Optical, Aetheric).
  2. Wave propagation delay and distance attenuation across hex metric.
  3. Sapper trench barrier dampening (80% overpressure deflection).
  4. Structured impulse task generation (Concussion, Suppression, Cognitive stress).
  5. Strict VITAL_MAX_HP = 6 vital invariant.
"""

import unittest
from krystal_web_hub.economic_engine.sensory_shockwave_engine import (
    SensoryModality,
    CognitiveStressState,
    SensoryWaveEmitter,
    ShockwavePropagationEngine,
    ImpulseTaskType,
    ImpulseTaskGenerator,
    SensoryShockwaveEngine,
    VITAL_MAX_HP
)


class TestSensoryShockwaveEngine(unittest.TestCase):

    def test_sensory_modalities_and_velocities(self):
        """Verifies that all 5 sensory modalities have defined wave propagation speeds."""
        velocities = SensoryWaveEmitter.VELOCITIES
        self.assertIn(SensoryModality.KINETIC_OVERPRESSURE.value, velocities)
        self.assertIn(SensoryModality.ACOUSTIC_BLAST.value, velocities)
        self.assertIn(SensoryModality.SEISMIC_TREMOR.value, velocities)
        self.assertIn(SensoryModality.OPTICAL_FLASH.value, velocities)
        self.assertIn(SensoryModality.AETHERIC_RESONANCE.value, velocities)

        # Kinetic blast wave must be supersonic (> 343 m/s)
        self.assertGreater(velocities[SensoryModality.KINETIC_OVERPRESSURE.value], 343.0)

    def test_wave_emission_and_hex_distance(self):
        """Verifies distance calculation between hex cells and shockwave emission."""
        wave = SensoryWaveEmitter.emit_shockwave(
            epicenter_hex=[0, 0],
            modality=SensoryModality.KINETIC_OVERPRESSURE,
            peak_intensity=120.0
        )
        self.assertEqual(wave["epicenter_hex"], [0, 0])
        self.assertEqual(wave["peak_intensity"], 120.0)

        # Distance to adjacent hex [1, 0]
        dist_adj = ShockwavePropagationEngine.calculate_hex_distance_meters([0, 0], [1, 0])
        self.assertAlmostEqual(dist_adj, 3.46, places=1)

        # Distance to 2 hexes away [2, 0]
        dist_2 = ShockwavePropagationEngine.calculate_hex_distance_meters([0, 0], [2, 0])
        self.assertAlmostEqual(dist_2, 6.92, places=1)

    def test_sapper_trench_barrier_deflection(self):
        """
        Verifies that a sapper trench deflects 80% of overpressure,
        dramatically reducing effective intensity compared to open ground.
        """
        wave = SensoryWaveEmitter.emit_shockwave(
            epicenter_hex=[0, 0],
            peak_intensity=100.0,
            blast_radius_m=8.0
        )

        # Target in open terrain (cover_density = 0.15)
        open_rec = ShockwavePropagationEngine.propagate_wave_to_target(
            wave=wave,
            target_hex=[0, 1],
            cover_density=0.15,
            is_trench=False
        )

        # Target in sapper trench (cover_density = 0.85, is_trench=True)
        trench_rec = ShockwavePropagationEngine.propagate_wave_to_target(
            wave=wave,
            target_hex=[0, 1],
            cover_density=0.85,
            is_trench=True
        )

        self.assertFalse(open_rec["is_sheltered_by_trench"])
        self.assertTrue(trench_rec["is_sheltered_by_trench"])
        self.assertEqual(trench_rec["trench_dampening_pct"], 80.0)
        self.assertLess(trench_rec["effective_intensity"], open_rec["effective_intensity"] * 0.25)

    def test_impulse_task_generation(self):
        """Verifies conversion of arriving wavefront into structured impulse tasks."""
        wave = SensoryWaveEmitter.emit_shockwave([0, 0], peak_intensity=100.0)
        reception = ShockwavePropagationEngine.propagate_wave_to_target(
            wave, [0, 0], cover_density=0.15, is_trench=False
        )
        combatant = {"unit_id": "test_hero", "name": "Bojovník", "current_hp": 6}

        tasks = ImpulseTaskGenerator.generate_impulse_tasks(combatant, reception, round_number=1)
        self.assertEqual(len(tasks), 4)

        task_types = [t["task_type"] for t in tasks]
        self.assertIn(ImpulseTaskType.CONCUSSION_STAGGER.value, task_types)
        self.assertIn(ImpulseTaskType.TACTICAL_SUPPRESSION.value, task_types)
        self.assertIn(ImpulseTaskType.COGNITIVE_OVERLOAD.value, task_types)
        self.assertIn(ImpulseTaskType.VITAL_DAMAGE_ABSORPTION.value, task_types)

        # Direct hit on open ground triggers concussion and suppression
        concussion = next(t for t in tasks if t["task_type"] == ImpulseTaskType.CONCUSSION_STAGGER.value)
        self.assertTrue(concussion["concussion_applied"])

    def test_tactical_detonation_cycle(self):
        """Verifies full end-to-end tactical detonation and impulse dispatch."""
        engine = SensoryShockwaveEngine()
        combatants = [
            {"unit_id": "p1", "name": "Entrenched Sapper", "current_hex": [0, -1], "current_hp": 6},
            {"unit_id": "o1", "name": "Exposed Witch", "current_hex": [0, 0], "current_hp": 6}
        ]
        cover_map = {"0_-1": 0.85, "0_0": 0.15}
        trenches = ["0_-1"]

        res = engine.trigger_tactical_detonation(
            epicenter_hex=[0, 0],
            combatants=combatants,
            cover_map=cover_map,
            trenches=trenches,
            round_number=1
        )

        self.assertEqual(res["status"], "PROCESSED")
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertEqual(len(res["evaluations"]), 2)

        # Entrenched sapper took 0 damage
        sapper_eval = next(e for e in res["evaluations"] if e["unit_id"] == "p1")
        self.assertEqual(sapper_eval["resulting_hp"], 6)

        # Exposed witch took damage
        witch_eval = next(e for e in res["evaluations"] if e["unit_id"] == "o1")
        self.assertLess(witch_eval["resulting_hp"], 6)


if __name__ == "__main__":
    unittest.main()
