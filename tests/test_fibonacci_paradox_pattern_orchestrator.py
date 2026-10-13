#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE FOR FIBONACCI PARADOX PATTERN ORCHESTRATOR
==============================================================================
File: tests/test_fibonacci_paradox_pattern_orchestrator.py
Tests:
  - 3x3 Grid symbol placement and boundary validations
  - Classical 3-in-a-row winner detection
  - Tic-Tac-Toe draw paradox detection & Fibonacci tensor symmetry breaking
  - Phase transition event firing and procedural card generation
  - Adaptive counter-polarity mapping (toxic -> amber, fire -> mud, etc.)
  - Strict enforcement of universal invariant VITAL_MAX_HP == 6
  - Weight rotation and board auto-reset
==============================================================================
"""

import unittest
from krystal_kernel.fibonacci_paradox_pattern_orchestrator import (
    VITAL_MAX_HP,
    CellSymbol,
    GeneratedElementCard,
    PhaseTransitionEvent,
    FibonacciParadoxPatternOrchestrator,
    GLOBAL_PATTERN_ORCHESTRATOR
)


class TestFibonacciParadoxPatternOrchestrator(unittest.TestCase):
    """Test suite covering the Fibonacci & Tic-Tac-Toe Paradox Pattern Orchestrator."""

    def setUp(self):
        self.orchestrator = FibonacciParadoxPatternOrchestrator()

    def test_01_grid_placement_and_boundaries(self):
        """Verifies symbol placement and rejection of invalid or occupied coordinates."""
        self.assertTrue(self.orchestrator.place_symbol("X", 0, 0))
        self.assertEqual(self.orchestrator.grid[0][0], CellSymbol.PLAYER_X)

        # Disallow placing over existing symbol
        self.assertFalse(self.orchestrator.place_symbol("O", 0, 0))

        # Disallow out-of-bounds coordinates
        self.assertFalse(self.orchestrator.place_symbol("X", 3, 3))
        self.assertFalse(self.orchestrator.place_symbol("X", -1, 0))

    def test_02_classical_win_detection(self):
        """Verifies classic 3-in-a-row win terminates without paradox."""
        # Row win for X
        self.orchestrator.place_symbol("X", 1, 0)
        self.orchestrator.place_symbol("X", 1, 1)
        self.orchestrator.place_symbol("X", 1, 2)

        res = self.orchestrator.resolve_turn_or_paradox()
        self.assertEqual(res["status"], "CLASSICAL_WIN")
        self.assertEqual(res["winner"], "X")
        self.assertFalse(res["paradox_triggered"])

    def test_03_draw_paradox_and_fibonacci_symmetry_breaking(self):
        """Tests that a classic draw collapses into a Phase Transition with symmetry breaking."""
        # Fill board with non-winning draw layout:
        # X O X
        # X O O
        # O X X
        moves = [
            ("X", 0, 0), ("O", 0, 1), ("X", 0, 2),
            ("X", 1, 0), ("O", 1, 1), ("O", 1, 2),
            ("O", 2, 0), ("X", 2, 1), ("X", 2, 2)
        ]
        for player, r, c in moves:
            self.orchestrator.place_symbol(player, r, c)

        self.assertIsNone(self.orchestrator.check_classical_winner())
        self.assertTrue(self.orchestrator.is_board_full())

        res = self.orchestrator.resolve_turn_or_paradox(opponent_element_focus="toxic")

        self.assertEqual(res["status"], "PHASE_TRANSITION_RESOLVED")
        self.assertTrue(res["paradox_triggered"])
        self.assertIn("transition_event", res)

        event = res["transition_event"]
        self.assertTrue(event["symmetry_broken"])
        self.assertNotEqual(event["fibonacci_tensor_sum_x"], event["fibonacci_tensor_sum_o"])
        self.assertIn(event["winning_player"], ["PLAYER_X", "OPPONENT_O"])

        # Check that card was induced
        card = event["induced_element_card"]
        self.assertIsInstance(card, dict)
        self.assertIn("Fibonacci", card["card_name"])
        self.assertEqual(card["counter_target"], "toxic")
        self.assertLessEqual(card["damage"], VITAL_MAX_HP)
        self.assertEqual(card["vital_max_hp_rule"], VITAL_MAX_HP)

    def test_04_adaptive_counter_polarity_mappings(self):
        """Verifies adaptive element mapping against various opponent affinities."""
        # Against Fire
        card_fire = self.orchestrator.generate_counter_element_card(opponent_element="fire", fib_power_rank=2)
        self.assertEqual(card_fire.counter_target, "fire")
        self.assertEqual(card_fire.elemental_affinity, "acid_mud")

        # Against Druid
        card_druid = self.orchestrator.generate_counter_element_card(opponent_element="druid", fib_power_rank=3)
        self.assertEqual(card_druid.counter_target, "druid")
        self.assertEqual(card_druid.elemental_affinity, "basalt_combustion")

        # Against Crystal
        card_cryst = self.orchestrator.generate_counter_element_card(opponent_element="crystal", fib_power_rank=4)
        self.assertEqual(card_cryst.counter_target, "crystal")
        self.assertEqual(card_cryst.elemental_affinity, "druidic_bark")

    def test_05_strict_invariant_clamp_vital_max_hp_6(self):
        """Verifies generated damage and defense can NEVER exceed VITAL_MAX_HP == 6."""
        for rank in range(1, 15):
            card = self.orchestrator.generate_counter_element_card(opponent_element="toxic", fib_power_rank=rank)
            self.assertLessEqual(card.damage, VITAL_MAX_HP)
            self.assertLessEqual(card.defense, VITAL_MAX_HP)
            self.assertGreaterEqual(card.damage, 1)
            self.assertEqual(card.vital_max_hp_rule, VITAL_MAX_HP)

    def test_06_weight_rotation_on_reset(self):
        """Verifies that reset_board rotates the Fibonacci weights."""
        initial_weights = list(self.orchestrator.fib_weights)
        self.orchestrator.reset_board()
        rotated_weights = self.orchestrator.fib_weights

        self.assertNotEqual(initial_weights, rotated_weights)
        self.assertEqual(len(rotated_weights), 9)
        self.assertEqual(self.orchestrator.rotation_step, 1)

    def test_07_global_singleton(self):
        """Verifies GLOBAL_PATTERN_ORCHESTRATOR is available and initialized."""
        self.assertIsNotNone(GLOBAL_PATTERN_ORCHESTRATOR)
        status = GLOBAL_PATTERN_ORCHESTRATOR.get_orchestrator_status()
        self.assertEqual(status["subsystem"], "FIBONACCI_PARADOX_PATTERN_ORCHESTRATOR")
        self.assertEqual(status["vital_max_hp_rule"], VITAL_MAX_HP)


if __name__ == "__main__":
    unittest.main()
