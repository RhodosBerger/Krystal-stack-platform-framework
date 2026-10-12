#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK // TEST SUITE: ELEMENTAL HYPERVISOR TELEMETRY MATRIX
==============================================================================
Tests:
  1. Universal Invariants: VITAL_MAX_HP == 6.
  2. All 5 Fundamental Elements + Compound Reactions.
  3. Hypervisor Ring Domain allocation (Ring -1, Ring 0, Ring 3).
  4. Telemetric benchmarking statistics (Speedups, Delas, Thermal drops).
  5. Elemental Polarity Reactions & Amber Catalyst arbitration.
  6. ASCII Periodic Table and Full JSON Manifest exports.
==============================================================================
"""

import unittest
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_kernel.elemental_hypervisor_telemetry_matrix import (
    VITAL_MAX_HP,
    ElementalPolarity,
    HypervisorRingDomain,
    ElementalBenchmarkStat,
    ElementalMapNode,
    ElementalMatrixSummary,
    ElementalHypervisorTelemetryMatrix,
    GLOBAL_ELEMENTAL_HYPERVISOR_MATRIX
)


class TestElementalHypervisorTelemetryMatrix(unittest.TestCase):

    def setUp(self):
        self.matrix = ElementalHypervisorTelemetryMatrix()

    def test_universal_invariants(self):
        """Verifies that VITAL_MAX_HP == 6 is strictly preserved everywhere."""
        self.assertEqual(VITAL_MAX_HP, 6)
        summary = self.matrix.get_system_summary()
        self.assertEqual(summary.vital_max_hp, 6)

        for elem in self.matrix.get_all_elements():
            self.assertEqual(elem.vital_max_hp, 6)

    def test_all_fundamental_elements_present(self):
        """Verifies that all 5 fundamental elemental polarities and compounds exist."""
        required = [
            ElementalPolarity.CRYSTAL,
            ElementalPolarity.BASALT_FIRE,
            ElementalPolarity.DRUID_BARK,
            ElementalPolarity.TOXIC_SLAG,
            ElementalPolarity.AETHER_QUANTUM,
            ElementalPolarity.AMBER_CATALYST,
            ElementalPolarity.COMBUSTION_SHOCKWAVE,
            ElementalPolarity.STEAM_QUENCH,
        ]
        for p in required:
            node = self.matrix.get_element(p)
            self.assertIsNotNone(node, f"Element {p} not registered in matrix")
            self.assertEqual(node.polarity, p)
            self.assertTrue(len(node.symbol) >= 2)
            self.assertEqual(len(node.polarity_vector), 5)
            self.assertGreater(len(node.key_metrics), 0)

    def test_ring_domains_assignment(self):
        """Verifies correct ring domain placement for architectural separation."""
        cr = self.matrix.get_element(ElementalPolarity.CRYSTAL)
        self.assertEqual(cr.ring_domain, HypervisorRingDomain.RING_HYPERVISOR_ROOT)

        bf = self.matrix.get_element(ElementalPolarity.BASALT_FIRE)
        self.assertEqual(bf.ring_domain, HypervisorRingDomain.RING_KERNEL_BAREMETAL)

        ae = self.matrix.get_element(ElementalPolarity.AETHER_QUANTUM)
        self.assertEqual(ae.ring_domain, HypervisorRingDomain.RING_GUEST_PARAVIRT)

        amb = self.matrix.get_element(ElementalPolarity.AMBER_CATALYST)
        self.assertEqual(amb.ring_domain, HypervisorRingDomain.COMPOUND_SYNTHESIS)

    def test_telemetric_statistical_speedups(self):
        """Verifies that measured speedups and improvements reflect kernel innovations."""
        # 1. Crystal DMA latency speedup > 30x
        cr = self.matrix.get_element(ElementalPolarity.CRYSTAL)
        lat_metric = next(m for m in cr.key_metrics if m.metric_id == "ipc_hypercall_latency")
        self.assertGreater(lat_metric.speedup_factor, 30.0)
        self.assertLess(lat_metric.krystal_value, 1.0)  # Under 1 ms

        # 2. Basalt/Fire junction temperature drop > 20°C
        bf = self.matrix.get_element(ElementalPolarity.BASALT_FIRE)
        temp_metric = next(m for m in bf.key_metrics if m.metric_id == "junction_temperature")
        self.assertLess(temp_metric.krystal_value, 55.0)
        self.assertGreater(temp_metric.legacy_value - temp_metric.krystal_value, 20.0)

        # 3. Aether codec serialization speedup > 80x
        ae = self.matrix.get_element(ElementalPolarity.AETHER_QUANTUM)
        codec_metric = next(m for m in ae.key_metrics if m.metric_id == "codec_serialization_time")
        self.assertGreater(codec_metric.speedup_factor, 80.0)
        self.assertLess(codec_metric.krystal_value, 1.0)  # Sub-microsecond

    def test_elemental_polarity_reaction_arbitration(self):
        """Verifies that Amber Catalyst resolves Crystal x Toxic clash, and reactions work."""
        # 1. Unstable clash without catalyst
        res_fail = self.matrix.evaluate_elemental_reaction(
            ElementalPolarity.CRYSTAL,
            ElementalPolarity.TOXIC_SLAG,
            has_amber_catalyst=False
        )
        self.assertFalse(res_fail["success"])
        self.assertEqual(res_fail["reaction_type"], "UNSTABLE_POLARITY_CLASH")
        self.assertEqual(res_fail["vital_hp"], 6)

        # 2. Stabilized with Amber Catalyst
        res_ok = self.matrix.evaluate_elemental_reaction(
            ElementalPolarity.CRYSTAL,
            ElementalPolarity.TOXIC_SLAG,
            has_amber_catalyst=True
        )
        self.assertTrue(res_ok["success"])
        self.assertEqual(res_ok["reaction_type"], "AMBER_CATALYZED_RESONANCE")
        self.assertEqual(res_ok["power_budget_multiplier"], 1.25)
        self.assertEqual(res_ok["vital_hp"], 6)

        # 3. Combustion Shockwave (Fire + Toxic)
        res_shock = self.matrix.evaluate_elemental_reaction(
            ElementalPolarity.BASALT_FIRE,
            ElementalPolarity.TOXIC_SLAG
        )
        self.assertTrue(res_shock["success"])
        self.assertEqual(res_shock["reaction_type"], "COMBUSTION_SHOCKWAVE")
        self.assertEqual(res_shock["vital_hp"], 6)

        # 4. Thermal Steam Quench (Fire + Crystal)
        res_quench = self.matrix.evaluate_elemental_reaction(
            ElementalPolarity.BASALT_FIRE,
            ElementalPolarity.CRYSTAL
        )
        self.assertTrue(res_quench["success"])
        self.assertEqual(res_quench["reaction_type"], "THERMAL_STEAM_QUENCH")
        self.assertGreater(res_quench["thermal_drop_c"], 10.0)
        self.assertEqual(res_quench["vital_hp"], 6)

    def test_manifest_and_ascii_exports(self):
        """Verifies JSON manifest and ASCII periodic table string representations."""
        ascii_table = self.matrix.export_periodic_table_ascii()
        self.assertIn("MAPA PRVKOV", ascii_table)
        self.assertIn("KRYŠTÁL", ascii_table)
        self.assertIn("BAZALT & OHEŇ", ascii_table)
        self.assertIn("VITAL_MAX_HP == 6", ascii_table)

        manifest = self.matrix.export_full_manifest()
        self.assertIn("summary", manifest)
        self.assertIn("elements", manifest)
        self.assertEqual(manifest["vital_max_hp_rule"], 6)
        self.assertEqual(len(manifest["elements"]), 8)


if __name__ == "__main__":
    unittest.main()
