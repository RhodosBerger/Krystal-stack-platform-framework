#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: MULTILOCK MATRIX AUTOCOMPLETER & SENSE CONFIRMATION ENGINE
==============================================================================
Test Target: krystal_kernel/multilock_matrix_autocompleter.py
Framework: unittest
Invariant: VITAL_MAX_HP == 6 strictly preserved across all states and branches.
==============================================================================
"""

import os
import sys
import unittest
import threading

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.multilock_matrix_autocompleter import (
    MatrixMultilockCoordinator,
    MatrixDomainLock,
    SenseConfirmationEngine,
    SyntaxPatternCloner,
    MultilockMatrixAutocompleter,
    GLOBAL_MULTILOCK_AUTOCOMPLETER
)


class TestMultilockMatrixAutocompleter(unittest.TestCase):
    """Rigorous tests for multilock concurrency, sense confirmation, and autocomplete maps."""

    def setUp(self):
        self.autocompleter = MultilockMatrixAutocompleter()
        self.coordinator = self.autocompleter.coordinator
        self.cloner = self.autocompleter.cloner

    def test_vital_max_hp_invariant(self):
        """Platform invariant VITAL_MAX_HP == 6 must hold universally."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_multilock_coordinator_concurrency_and_snapshot(self):
        """Coordinator must safely acquire/release all locks and provide atomic snapshot."""
        # Test atomic lock acquisition across threads
        def worker():
            self.coordinator.acquire_all()
            # Mutate state safely
            self.coordinator.state_a["spindle_rpm"] = 14000
            self.coordinator.release_all()

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=1.0)
        self.assertFalse(t.is_alive())

        snap = self.coordinator.get_status_snapshot()
        self.assertTrue(snap.lock_physical_active)
        self.assertTrue(snap.lock_software_active)
        self.assertGreater(snap.c_physical_frobenius_norm, 0.0)
        self.assertGreater(snap.c_software_frobenius_norm, 0.0)
        self.assertLessEqual(snap.a_discrete_spectral_bound, 2.5)
        self.assertEqual(snap.vital_max_hp, 6)

    def test_sense_confirmation_nominal_valid(self):
        """Nominal cooling and synthesis requests must be confirmed valid with high coherence."""
        state_a = {
            "name": "STATE_A_NOMINAL",
            "tip_temperature_c": 220.0,
            "spindle_rpm": 12000,
            "vital_hp": 6
        }

        # Cooling request
        sense_cool = SenseConfirmationEngine.evaluate_sense("zvýš chladenie kvapalinou", state_a)
        self.assertTrue(sense_cool.is_sense_confirmed)
        self.assertEqual(sense_cool.flow_validation_status, "CONFIRMED_VALID")
        self.assertGreaterEqual(sense_cool.coherence_score, 0.90)
        self.assertEqual(sense_cool.predicted_hp_state_b, 6)

        # Microarchitecture request
        sense_soft = SenseConfirmationEngine.evaluate_sense("optimalizuj IPC liniek AVX-512", state_a)
        self.assertTrue(sense_soft.is_sense_confirmed)
        self.assertEqual(sense_soft.flow_validation_status, "CONFIRMED_VALID")
        self.assertGreaterEqual(sense_soft.coherence_score, 0.90)

    def test_sense_confirmation_thermal_hazard_blocking(self):
        """Requesting high RPM when overheating (>600C) must be blocked with safety warning."""
        overheated_state = {
            "name": "STATE_A_OVERHEATED",
            "tip_temperature_c": 680.0,
            "spindle_rpm": 16000,
            "vital_hp": 3
        }

        sense = SenseConfirmationEngine.evaluate_sense("zvýš otáčky na 24000 RPM", overheated_state)
        self.assertFalse(sense.is_sense_confirmed)
        self.assertEqual(sense.flow_validation_status, "BLOCKED_HAZARD")
        self.assertTrue(sense.safety_intervention_required)
        self.assertLess(sense.coherence_score, 0.30)
        self.assertIn("ZAMMIETNUTÉ", sense.sense_explanation)

    def test_sense_confirmation_adaptive_ssv_throttling(self):
        """Moderate heat (450C-600C) with speed demand must be throttled with SSV modulation."""
        elevated_state = {
            "name": "STATE_A_ELEVATED",
            "tip_temperature_c": 490.0,
            "spindle_rpm": 12000,
            "vital_hp": 5
        }

        sense = SenseConfirmationEngine.evaluate_sense("nastav vyššie otáčky vretena faster", elevated_state)
        self.assertTrue(sense.is_sense_confirmed)
        self.assertEqual(sense.flow_validation_status, "THROTTLED_ADAPTIVE")
        self.assertTrue(sense.safety_intervention_required)
        self.assertIn("POTVRDENÉ S TLMENÍM", sense.sense_explanation)

    def test_syntax_pattern_cloner_matching_and_recording(self):
        """Cloner must return high-quality matches and store novel user patterns."""
        matches = self.cloner.find_matching_clones("vytvor mi karbid", max_results=3)
        self.assertGreater(len(matches), 0)
        self.assertTrue(any("karbid" in m.lower() for m in matches))

        # Record novel user syntax
        novel_input = "vytvor mi diamantový gravírovací hrot s rezonanciou 432 Hz"
        self.cloner.record_user_syntax(novel_input)
        self.assertIn(novel_input, self.cloner.cloned_syntax_templates)

        # Ensure novel syntax is now suggested
        new_matches = self.cloner.find_matching_clones("vytvor mi diamantový", max_results=2)
        self.assertEqual(new_matches[0], novel_input)

    def test_map_of_suggestions_structure(self):
        """Autocompleter must assemble a complete Map of Suggestions graph."""
        res = self.autocompleter.predict_autocompletions("vytvor mi karbid")
        self.assertEqual(res.prefix, "vytvor mi karbid")
        self.assertGreater(len(res.candidate_branches), 0)
        
        # Check top branch
        top = res.candidate_branches[0]
        self.assertIn("node_id", top.to_dict())
        self.assertIn("completion_suffix", top.to_dict())
        self.assertGreater(top.confidence, 0.50)
        self.assertLessEqual(top.vital_hp, 6)

        # Check Multilock status payload
        self.assertIn("c_physical_frobenius_norm", res.multilock_status)
        self.assertEqual(res.vital_max_hp, 6)


if __name__ == "__main__":
    unittest.main()
