"""
Unit Tests for Greek Pantheon, Bohemian Alliances & Philosophical Memory Axioms
================================================================================
Validates:
1. Greek Deities and Bohemian Allies rosters.
2. Cross-Pantheon Pacts and synergy multipliers.
3. The 6 Philosophical Memory Axioms (Pythagoras, Heraclitus, Aristotle, Plato, Zeno, Epicurus).
4. Autonomous memory leveling simulation and Aristotelian Golden Mean convergence (0.618).
5. Strict adherence to the platform-wide 6 Max HP Vital Invariant.
"""

import unittest
from krystal_web_hub.economic_engine.greek_bohemia_pantheon_and_memory_axioms import (
    GreekBohemiaMemoryEngine,
    GreekDeity,
    BohemianPaganAlly,
    PantheonPact,
    PhilosophicalMemoryAxiom,
    GLOBAL_GREEK_BOHEMIA_ENGINE,
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    INV_GOLDEN_RATIO
)


class TestGreekBohemiaMemoryEngine(unittest.TestCase):

    def setUp(self):
        self.engine = GreekBohemiaMemoryEngine()

    def test_vital_max_hp_rule_is_strictly_six(self):
        """Ensures the 6 Max HP vital invariant is maintained across all entities."""
        self.assertEqual(VITAL_MAX_HP, 6)

        roster = self.engine.get_pantheon_roster()
        self.assertEqual(roster["vital_max_hp_rule"], 6)

        for deity in roster["greek_deities"]:
            self.assertEqual(deity["hp"], 6)
            self.assertEqual(deity["max_hp"], 6)

        for ally in roster["bohemian_allies"]:
            self.assertEqual(ally["hp"], 6)
            self.assertEqual(ally["max_hp"], 6)

        for pact in roster["pacts"]:
            self.assertEqual(pact["vital_hp"], 6)

        leveling = self.engine.simulate_autonomous_memory_leveling()
        self.assertEqual(leveling["vital_max_hp_rule"], 6)

    def test_greek_deities_roster(self):
        """Verifies canonical Olympian and Chthonic deities."""
        roster = self.engine.get_pantheon_roster()
        self.assertGreaterEqual(roster["greek_deities_count"], 8)
        
        deity_ids = [d["deity_id"] for d in roster["greek_deities"]]
        self.assertIn("zeus", deity_ids)
        self.assertIn("athena", deity_ids)
        self.assertIn("apollo", deity_ids)
        self.assertIn("hermes", deity_ids)
        self.assertIn("hephaestus", deity_ids)
        self.assertIn("poseidon", deity_ids)
        self.assertIn("hades", deity_ids)
        self.assertIn("ares", deity_ids)

    def test_bohemian_pagan_allies_roster(self):
        """Verifies Bohemian and Slavic pagan coalition leaders."""
        roster = self.engine.get_pantheon_roster()
        self.assertGreaterEqual(roster["bohemian_allies_count"], 8)

        ally_ids = [a["ally_id"] for a in roster["bohemian_allies"]]
        self.assertIn("perun", ally_ids)
        self.assertIn("libuse", ally_ids)
        self.assertIn("radegast", ally_ids)
        self.assertIn("veles", ally_ids)
        self.assertIn("kovar_krusnohor", ally_ids)
        self.assertIn("vodnik_vltava", ally_ids)
        self.assertIn("morana", ally_ids)
        self.assertIn("svantovit", ally_ids)

    def test_philosophical_memory_axioms_structure(self):
        """Validates the 6 ancient Greek philosophical memory strategies."""
        axioms = self.engine.get_philosophical_memory_axioms()
        self.assertEqual(axioms["axioms_count"], 6)

        ax_ids = [s["axiom_id"] for s in axioms["strategies"]]
        self.assertIn("pythagoras_harmonics", ax_ids)
        self.assertIn("heraclitus_flux", ax_ids)
        self.assertIn("aristotle_golden_mean", ax_ids)
        self.assertIn("plato_forms", ax_ids)
        self.assertIn("zeno_dichotomy", ax_ids)
        self.assertIn("epicurus_atomism", ax_ids)

    def test_autonomous_memory_leveling_simulation(self):
        """Validates convergence towards the Aristotelian Golden Mean (0.618)."""
        res = self.engine.simulate_autonomous_memory_leveling()
        self.assertEqual(res["status"], "MEMORY_AUTONOMOUSLY_LEVELLED")
        self.assertEqual(res["applied_axioms_count"], 6)
        self.assertGreater(res["combined_latency_reduction_percent"], 40.0)
        self.assertGreater(res["achieved_cache_hit_rate_percent"], 98.0)
        self.assertAlmostEqual(res["aristotelian_equilibrium_ratio"], INV_GOLDEN_RATIO, places=4)

        # Check memory pools leveling
        pools = res["rebalanced_memory_pools"]
        self.assertIn("l1_l2_sram", pools)
        self.assertIn("host_ddr_shared", pools)
        self.assertIn("nvme_ssd_swap", pools)
        self.assertIn("directml_tensor_ring", pools)

        for pool in pools.values():
            self.assertGreater(pool["golden_ratio_fit_percent"], 80.0)

    def test_form_new_pantheon_pact(self):
        """Tests forming a custom diplomatic pact between Greece and Bohemia."""
        pact_res = self.engine.form_new_pact(
            greek_deity_id="hermes",
            bohemian_ally_id="veles",
            pact_title="Pakt Rýchlych Ciest a Lesného Obchodu",
            lore="Hermes and Veles establish high-speed DMA conduits."
        )
        self.assertTrue(pact_res["success"])
        self.assertEqual(pact_res["pact"]["vital_max_hp"], 6)
        self.assertAlmostEqual(pact_res["pact"]["synergy_multiplier"], GOLDEN_RATIO, places=4)


if __name__ == '__main__':
    unittest.main()
