#!/usr/bin/env python3
"""
==============================================================================
UNIT TEST SUITE: KRYSTAL CONFIGURATION TOOLCHAIN & PROFILE MANAGER
==============================================================================
Test Target: scripts/krystal_config_toolchain.py
Framework: unittest
Invariant: VITAL_MAX_HP == 6 strictly preserved across all profiles and exports.
==============================================================================
"""

import os
import sys
import unittest
import json

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from scripts.krystal_config_toolchain import (
    KrystalConfigToolchain,
    ProfileKind,
    CANONICAL_PROFILES,
    KrystalConfigurationProfile,
    VITAL_MAX_HP
)


class TestKrystalConfigToolchain(unittest.TestCase):
    """Rigorous tests for configuration presets, host inspection, export formats, and dry-run execution."""

    def test_invariant_vital_max_hp_across_all_profiles(self):
        """Platform invariant VITAL_MAX_HP == 6 must be preserved in all canonical profiles."""
        self.assertEqual(VITAL_MAX_HP, 6)
        profiles = KrystalConfigToolchain.list_profiles()
        self.assertGreaterEqual(len(profiles), 6)
        for prof in profiles:
            self.assertEqual(prof.vital_max_hp, 6)

    def test_host_inspection(self):
        """Host environment probing must return valid CPU, timer, and OS metadata."""
        host = KrystalConfigToolchain.inspect_host_environment()
        self.assertIn("os_name", host)
        self.assertIn("cpu_cores", host)
        self.assertGreaterEqual(host["cpu_cores"], 1)
        self.assertTrue(host["invariants_ok"])

    def test_profile_resolution(self):
        """Profiles must resolve by key or partial match."""
        rt = KrystalConfigToolchain.get_profile("REALTIME_LOW_LATENCY")
        self.assertEqual(rt.profile_id, ProfileKind.REALTIME_LOW_LATENCY)
        self.assertEqual(rt.target_cadence_ms, 1.0)
        self.assertEqual(rt.timer_resolution_ms, 1)

        bal = KrystalConfigToolchain.get_profile("balanced")
        self.assertEqual(bal.profile_id, ProfileKind.BALANCED_DEVELOPMENT)

    def test_export_env(self):
        """Profile must export valid .env format with required tokens."""
        prof = KrystalConfigToolchain.get_profile("REALTIME_LOW_LATENCY")
        env_str = KrystalConfigToolchain.export_env(prof)
        self.assertIn("KRYSTAL_PROFILE=REALTIME_LOW_LATENCY", env_str)
        self.assertIn("KRYSTAL_CADENCE_MS=1.0", env_str)
        self.assertIn("KRYSTAL_VITAL_MAX_HP=6", env_str)

    def test_export_compact_card(self):
        """Compact card must display ASCII borders and 6 HP hearts."""
        prof = KrystalConfigToolchain.get_profile("MISSION_CONTROL_STUDIO_KIOSK")
        card = KrystalConfigToolchain.export_compact_card(prof)
        self.assertIn("MISSION_CONTROL_STUDIO_KIOSK", card)
        self.assertIn("♥ ♥ ♥ ♥ ♥ ♥ (6/6 HP)", card)

    def test_validation_rules(self):
        """Validation must accept valid configurations and reject invalid bounds."""
        prof = KrystalConfigToolchain.get_profile("BALANCED_DEVELOPMENT")
        valid, errs = KrystalConfigToolchain.validate_configuration(prof.to_dict())
        self.assertTrue(valid)
        self.assertEqual(len(errs), 0)

        # Invalid HP
        bad_cfg = prof.to_dict()
        bad_cfg["vital_max_hp"] = 10
        valid_bad, errs_bad = KrystalConfigToolchain.validate_configuration(bad_cfg)
        self.assertFalse(valid_bad)
        self.assertTrue(any("vital_max_hp" in e for e in errs_bad))

        # Invalid cadence
        bad_cad = prof.to_dict()
        bad_cad["target_cadence_ms"] = -5.0
        valid_cad, errs_cad = KrystalConfigToolchain.validate_configuration(bad_cad)
        self.assertFalse(valid_cad)

    def test_dry_run_activation_steps(self):
        """Dry-run must simulate 9 formal atomic execution steps without error."""
        prof = KrystalConfigToolchain.get_profile("ECO_THERMAL_PRESERVATION")
        rep = KrystalConfigToolchain.dry_run_activation(prof)
        self.assertEqual(rep["status"], "DRY_RUN_PASSED")
        self.assertEqual(len(rep["activation_steps"]), 9)
        self.assertEqual(rep["vital_max_hp"], 6)


if __name__ == "__main__":
    unittest.main()
