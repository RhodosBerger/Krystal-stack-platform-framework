#!/usr/bin/env python3
"""
UNIT TESTS: Sovereign Invariant & Tactical Gameplay Mechanics Engine
===================================================================
Validates the 6 sovereign mechanics:
  1. VBlank Phase-Shift Evasion and UMA Back-EMF Collapse
  2. Symplectic Lens Curving (SO(2) rotation, kappa, det(J) == 1.0)
  3. 12-Frame Dwell Armor & Seismic Release (Merkle-Momentum)
  4. Vertigo Periscope Trench Targeting (Dolly Zoom zero-dispersion)
  5. Constitutional Quarantine & Zero-Mutation Tether (Delta S == 0, VITAL_MAX_HP == 6)
  6. Autonomic Hypervisor Combat Stance Switching
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP
from krystal_kernel.bot_diagnostic_and_tuning_toolkit import TuningProfile
from krystal_web_hub.economic_engine.sovereign_invariant_mechanics import (
    TacticalAttackType,
    VBlankPhaseShiftResolver,
    SymplecticLensCurvingResolver,
    DwellArmorMerkleAccumulator,
    VertigoPeriscopeDispersionResolver,
    ConstitutionalQuarantineResolver,
    AutonomicStanceResolver,
    SovereignInvariantGameplayEngine,
    GLOBAL_SOVEREIGN_INVARIANT_GAMEPLAY_ENGINE,
)


class TestSovereignInvariantMechanics(unittest.TestCase):

    def setUp(self):
        self.engine = SovereignInvariantGameplayEngine()

    def test_vblank_phase_shift_evasion(self):
        """Verifies phase-shift evasion under normal UMA and back-EMF collapse when saturated."""
        # Case A: Normal UMA load (mu_UMA = 0.50) -> 100% evasion of ranged projectile
        res_evade = VBlankPhaseShiftResolver.resolve_incoming_attack(
            attack_type=TacticalAttackType.RANGED_PROJECTILE,
            is_phase_shifted=True,
            uma_saturation_ratio=0.50
        )
        self.assertTrue(res_evade.evaded)
        self.assertFalse(res_evade.phase_collapsed)
        self.assertEqual(res_evade.back_emf_damage, 0)
        self.assertEqual(res_evade.vital_max_hp, 6)

        # Case B: UMA Bus Saturated (mu_UMA = 0.82 >= 0.78) -> Phase cloak collapses with Back-EMF shock
        res_collapse = VBlankPhaseShiftResolver.resolve_incoming_attack(
            attack_type=TacticalAttackType.RANGED_PROJECTILE,
            is_phase_shifted=True,
            uma_saturation_ratio=0.82
        )
        self.assertFalse(res_collapse.evaded)
        self.assertTrue(res_collapse.phase_collapsed)
        self.assertEqual(res_collapse.back_emf_damage, 1)

    def test_symplectic_lens_curving_and_conservation(self):
        """Verifies that curved trajectories preserve symplectic volume (det(J) == 1.0) and bypass LOS."""
        curved = SymplecticLensCurvingResolver.calculate_curved_shot(
            origin=(0, 0),
            target=(4, 0),
            lens_hex=(2, 2),
            curvature_power=1.5
        )
        self.assertTrue(curved.line_of_sight_bypassed)
        self.assertTrue(curved.is_symplectic_conserved)
        self.assertAlmostEqual(curved.jacobian_determinant, 1.0, places=12)
        self.assertGreater(curved.curvature_kappa, 0.0)
        self.assertEqual(len(curved.bent_path), 4)
        self.assertEqual(curved.vital_max_hp, 6)

    def test_dwell_armor_merkle_accumulation_and_seismic_release(self):
        """Verifies 12-frame dwell absorption via Merkle-Momentum and 13th-frame seismic release."""
        dwell = DwellArmorMerkleAccumulator(unit_id="WARRIOR_01", gamma=0.85)

        # Register hits at frames 0, 4, 8 (< 12 frames)
        s0 = dwell.register_hit(damage=2.0, frame_index=0)
        self.assertFalse(s0.is_release_ready)
        self.assertGreater(s0.accumulated_momentum, 0.0)

        s1 = dwell.register_hit(damage=3.0, frame_index=4)
        self.assertFalse(s1.is_release_ready)
        self.assertEqual(s1.hit_count_in_window, 2)

        # Register hit at frame 13 (>= 12 frames from frame 0)
        s2 = dwell.register_hit(damage=1.0, frame_index=13)
        self.assertTrue(s2.is_release_ready)

        # Trigger seismic release
        release = dwell.trigger_seismic_release()
        self.assertTrue(release.stuns_enemies)
        self.assertEqual(release.stun_duration_rounds, 1)
        self.assertEqual(release.blast_radius_hex, 1)
        self.assertGreater(release.released_damage, 0.0)
        self.assertEqual(release.vital_max_hp, 6)

        # Verify reset for next cycle
        self.assertEqual(dwell.state.accumulated_momentum, 0.0)
        self.assertFalse(dwell.state.is_release_ready)

    def test_vertigo_periscope_trench_targeting(self):
        """Verifies that Vertigo Dolly Zoom zeros out mortar dispersion and removes trench reduction."""
        # Case A: Target in sapper trench without Vertigo lock -> protected, dispersion 3.5 hex
        res_trench = VertigoPeriscopeDispersionResolver.resolve_trench_targeting(
            target_in_sapper_trench=True,
            is_vertigo_locked=False,
            distance_hexes=5
        )
        self.assertFalse(res_trench.target_uncovered)
        self.assertEqual(res_trench.calibrated_dispersion_sigma, 3.50)
        self.assertFalse(res_trench.critical_hit_granted)

        # Case B: Target in sapper trench with Vertigo lock -> uncovered, dispersion 0.0, critical hit
        res_vertigo = VertigoPeriscopeDispersionResolver.resolve_trench_targeting(
            target_in_sapper_trench=True,
            is_vertigo_locked=True,
            distance_hexes=5
        )
        self.assertTrue(res_vertigo.target_uncovered)
        self.assertEqual(res_vertigo.calibrated_dispersion_sigma, 0.00)
        self.assertTrue(res_vertigo.trench_damage_reduction_eliminated)
        self.assertTrue(res_vertigo.critical_hit_granted)

    def test_constitutional_quarantine_zero_mutation(self):
        """Verifies that invalid actions result strictly in Delta S == 0 and mana siphoning."""
        # Case A: Illegal move distance beyond maximum
        act_illegal = ConstitutionalQuarantineResolver.evaluate_action(
            current_hp=5,
            attempted_heal=0,
            attempted_move_distance=5,
            max_move_distance=2,
            mana_cost=3,
            available_mana=10
        )
        self.assertFalse(act_illegal.is_valid)
        self.assertEqual(act_illegal.status, "QUARANTINED_SAFETY_BREACH")
        self.assertEqual(act_illegal.mutation_delta_bytes, 0)
        self.assertEqual(act_illegal.delta_s, 0.0)
        self.assertEqual(act_illegal.mana_siphoned_to_soul_well, 3)
        self.assertEqual(act_illegal.vital_max_hp, 6)

        # Case B: Overhealing clamped to VITAL_MAX_HP == 6
        act_heal = ConstitutionalQuarantineResolver.evaluate_action(
            current_hp=5,
            attempted_heal=4,  # 5 + 4 = 9 > 6
            attempted_move_distance=1,
            max_move_distance=2,
            mana_cost=2,
            available_mana=10
        )
        self.assertTrue(act_heal.is_valid)
        self.assertEqual(act_heal.status, "COMMITTED_CLAMPED_TO_INVARIANT")
        self.assertEqual(act_heal.delta_s, 1.0)  # clamped to +1 HP
        self.assertEqual(act_heal.vital_max_hp, 6)

    def test_autonomic_combat_stances(self):
        """Verifies that all combat stances produce valid modifiers respecting invariant limits."""
        for prof in (TuningProfile.BALANCED, TuningProfile.ULTRA_PERFORMANCE,
                     TuningProfile.LOW_POWER_EFFICIENCY, TuningProfile.LOW_LATENCY_GAMING):
            mods = AutonomicStanceResolver.get_stance_modifiers(prof)
            self.assertIn("stance_name", mods)
            self.assertEqual(mods["vital_max_hp"], 6)

        berserk = AutonomicStanceResolver.get_stance_modifiers(TuningProfile.ULTRA_PERFORMANCE)
        self.assertEqual(berserk["pl1_power_w"], 32.0)
        self.assertEqual(berserk["attack_power_bonus_pct"], 25.0)

        meditation = AutonomicStanceResolver.get_stance_modifiers(TuningProfile.LOW_POWER_EFFICIENCY)
        self.assertEqual(meditation["pl1_power_w"], 15.0)
        self.assertEqual(meditation["passive_hp_regen_per_round"], 1)

    def test_master_engine_turn_execution(self):
        """Verifies master engine execution of tactical turns with HMAC certification."""
        turn = self.engine.execute_tactical_turn(
            actor_id="HERO_KRYSTAL_01",
            current_hp=4,
            attempted_action="MOVE_AND_CAST",
            mana_available=10,
            mana_cost=2,
            stance=TuningProfile.BALANCED
        )
        self.assertTrue(turn["action_valid"])
        self.assertEqual(turn["status"], "COMMITTED_SAFE")
        self.assertEqual(turn["vital_max_hp"], 6)
        self.assertEqual(len(turn["audit_hmac"]), 64)


if __name__ == "__main__":
    unittest.main()
