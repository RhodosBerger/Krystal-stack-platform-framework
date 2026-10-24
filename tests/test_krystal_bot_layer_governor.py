"""
tests/test_krystal_bot_layer_governor.py

Unit test suite validating the SovereignLayerGovernor integration into krystal_bot.
Tests /layer commands, cheatsheets, tuning receipts, and natural language dispatch.
"""

import unittest
import os
import shutil
import tempfile
from krystal_kernel.constants import VITAL_MAX_HP
from krystal_bot.sovereign_layer_governor import (
    SOVEREIGN_LAYER_GOVERNOR,
    SovereignLayerGovernor
)
from krystal_bot.service import BotService


class TestKrystalBotLayerGovernor(unittest.TestCase):
    """Test suite validating layer governing skills and cheatsheets in krystal_bot."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.bot = BotService(root=self.tmp_dir, use_keras=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_list_variables_cheatsheet(self) -> None:
        """Verifies cheatsheet table rendering all 28 variables."""
        text = SOVEREIGN_LAYER_GOVERNOR.list_variables_cheatsheet()
        self.assertIn("PREHĽAD PREMENNÝCH VRSTIEV", text)
        self.assertIn("VAR_VITAL_MAX_HP", text)
        self.assertIn("VAR_VBLANK_SHADOW_BUDGET_MS", text)
        self.assertIn("VAR_PDN_INDUCTANCE_NH", text)
        self.assertIn("Celkom registrovaných riadiacich premenných: 28", text)

    def test_get_variable_details(self) -> None:
        """Verifies detailed output for an individual variable."""
        text = SOVEREIGN_LAYER_GOVERNOR.get_variable_details("VAR_VBLANK_SHADOW_BUDGET_MS")
        self.assertIn("DETAIL PREMENNEJ: VAR_VBLANK_SHADOW_BUDGET_MS", text)
        self.assertIn("L1_CHRONO_UMA", text)
        self.assertIn("3.5", text)
        self.assertIn("VITAL_MAX_HP == 6", text)

    def test_tune_variable_via_governor(self) -> None:
        """Verifies variable tuning commands (both valid and rejected)."""
        # Rejection: immutable variable
        res_imm = SOVEREIGN_LAYER_GOVERNOR.tune_variable("VAR_VITAL_MAX_HP", "7")
        self.assertIn("ODMIETNUTÉ ZÁKONOM", res_imm)

        # Rejection: out of bounds
        res_bounds = SOVEREIGN_LAYER_GOVERNOR.tune_variable("VAR_DWELL_FRAME_LIMIT", "2")
        self.assertIn("ODMIETNUTÉ ZÁKONOM", res_bounds)

        # Successful tuning
        res_ok = SOVEREIGN_LAYER_GOVERNOR.tune_variable("VAR_ALPHA_CADENCE_PERIOD_MS", "105")
        self.assertIn("PREMENNÁ ÚSPEŠNE UPRAVENÁ", res_ok)
        self.assertIn("105", res_ok)

    def test_optimal_operating_profile_cheatsheet(self) -> None:
        """Verifies Pareto-optimal operating profile output."""
        text = SOVEREIGN_LAYER_GOVERNOR.format_optimal_operating_cheatsheet("0.7")
        self.assertIn("OPTIMÁLNY MULTI-VRSTVOVÝ PREVÁDZKOVÝ BOD", text)
        self.assertIn("VBlank tieňové okno", text)
        self.assertIn("Saturácia zbernice UMA", text)
        self.assertIn("VITAL_MAX_HP == 6", text)

    def test_scales_cheatsheet(self) -> None:
        """Verifies 4-scale dimensioning cheatsheet output."""
        text = SOVEREIGN_LAYER_GOVERNOR.format_scales_cheatsheet()
        self.assertIn("ŠKÁLOVACIA DIMENZIA SOVEREIGN BOTA", text)
        self.assertIn("MICRO_SILICON_NS", text)
        self.assertIn("MESO_VBLANK_MS", text)
        self.assertIn("MACRO_CADENCE_MS", text)
        self.assertIn("EPOCH_GOVERNANCE_S", text)

    def test_bot_service_command_dispatch(self) -> None:
        """Verifies end-to-end bot message routing with /layer commands."""
        # 1. /layer list
        msg1 = self.bot.message("/layer list")
        self.assertTrue(msg1["ok"])
        self.assertEqual(msg1["skill"], "layer")
        self.assertIn("PREHĽAD PREMENNÝCH VRSTIEV", msg1["reply"])

        # 2. /layer get VAR_PDN_INDUCTANCE_NH
        msg2 = self.bot.message("/layer get VAR_PDN_INDUCTANCE_NH")
        self.assertTrue(msg2["ok"])
        self.assertIn("DETAIL PREMENNEJ", msg2["reply"])

        # 3. /layer optimal
        msg3 = self.bot.message("/layer optimal")
        self.assertTrue(msg3["ok"])
        self.assertIn("OPTIMÁLNY MULTI-VRSTVOVÝ PREVÁDZKOVÝ BOD", msg3["reply"])

        # 4. /layer scales
        msg4 = self.bot.message("/layer scales")
        self.assertTrue(msg4["ok"])
        self.assertIn("ŠKÁLOVACIA DIMENZIA SOVEREIGN BOTA", msg4["reply"])


if __name__ == "__main__":
    unittest.main()
