#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK TEST SUITE: CHRONO-SYMPLECTIC PROCESSOR ARCHITECTURE
==============================================================================
Module: tests/test_chrono_symplectic_processor_architecture.py
Milestone: V-CHRONO-SYMPLECTIC-SOC-01
Constitutional Invariant: VITAL_MAX_HP == 6

Verifies:
  1. Constitutional Invariant VITAL_MAX_HP == 6 across the SoC and core clusters.
  2. Hexagonal 6-cluster floorplan topology (0x10 to 0x60) and 128B coherent ring.
  3. Total die area fraction conservation and TDP power envelope limit (28W).
  4. Four operational application spectra (Robotics, Local AI, Aerospace, Holography).
  5. Workload dispatch simulation, voltage droop clamping (<= 50 mV), and VBlank shadow.
  6. Theory DAG registration and dependency verification.
==============================================================================
"""

import unittest
from typing import Dict, Any, List

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.chrono_symplectic_processor_architecture import (
    CoreClusterRole,
    ApplicationSpectrumKind,
    ProcessorClusterSpec,
    ApplicationSpectrumDescriptor,
    WorkloadExecutionReceipt,
    ChronoSymplecticProcessorSoC,
    GLOBAL_CHRONO_SYMPLECTIC_SOC,
)
from krystal_kernel.verification.theory_dependency_graph import (
    GLOBAL_THEORY_DEPENDENCY_GRAPH,
    TheoryEpistemicStatus,
)


class TestChronoSymplecticProcessorArchitecture(unittest.TestCase):
    """Unit and empirical verification suite for Chrono-Symplectic SoC architecture."""

    def setUp(self) -> None:
        self.soc = GLOBAL_CHRONO_SYMPLECTIC_SOC

    # =========================================================================
    # 1. CONSTITUTIONAL INVARIANT TESTS
    # =========================================================================

    def test_constitutional_invariant_enforcement(self) -> None:
        """Verifies that vital_max_hp == 6 is strictly enforced and cannot be mutated."""
        self.assertEqual(self.soc.vital_max_hp, 6)

        # SoC constructor invariant breach
        with self.assertRaises(ContractViolation):
            ChronoSymplecticProcessorSoC(vital_max_hp=5)

        # Cluster spec constructor invariant breach
        with self.assertRaises(ContractViolation):
            ProcessorClusterSpec(
                cluster_index=0,
                sector_code="0x10",
                role=CoreClusterRole.CLUSTER_0x10_INDUCTIVE_PDN_GOVERNOR,
                frequency_ghz=2.4,
                cache_l2_kb=512,
                tdp_watts=3.5,
                die_area_fraction=0.12,
                primary_alu_kind="Choke ALU",
                vital_max_hp=7,
            )

        # Cluster index out of bounds
        with self.assertRaises(ContractViolation):
            ProcessorClusterSpec(
                cluster_index=6,  # Valid indices are 0..5
                sector_code="0x70",
                role=CoreClusterRole.CLUSTER_0x10_INDUCTIVE_PDN_GOVERNOR,
                frequency_ghz=2.4,
                cache_l2_kb=512,
                tdp_watts=3.5,
                die_area_fraction=0.12,
                primary_alu_kind="Choke ALU",
            )

    # =========================================================================
    # 2. 6-CLUSTER HEXAGONAL FLOORPLAN TESTS
    # =========================================================================

    def test_six_hexagonal_core_clusters(self) -> None:
        """Verifies the exact 6 constitutional core clusters and their specialization."""
        clusters = self.soc.list_clusters()
        self.assertEqual(len(clusters), 6)

        expected_sectors = ["0x10", "0x20", "0x30", "0x40", "0x50", "0x60"]
        for k, c in enumerate(clusters):
            self.assertEqual(c.cluster_index, k)
            self.assertEqual(c.sector_code, expected_sectors[k])
            self.assertEqual(c.vital_max_hp, 6)
            self.assertGreater(c.frequency_ghz, 1.0)
            self.assertGreater(c.tdp_watts, 0.0)

        # Verify cluster roles
        c_pdn = self.soc.get_cluster("0x10")
        self.assertIsNotNone(c_pdn)
        self.assertEqual(c_pdn.role, CoreClusterRole.CLUSTER_0x10_INDUCTIVE_PDN_GOVERNOR)

        c_crossbar = self.soc.get_cluster("0x20")
        self.assertEqual(c_crossbar.role, CoreClusterRole.CLUSTER_0x20_UMA_COHERENT_CROSSBAR)

        c_symp = self.soc.get_cluster("0x30")
        self.assertEqual(c_symp.role, CoreClusterRole.CLUSTER_0x30_SYMPLECTIC_HAMILTONIAN_ALU)

        c_alt = self.soc.get_cluster("0x40")
        self.assertEqual(c_alt.role, CoreClusterRole.CLUSTER_0x40_COGNITIVE_ALTERNATOR_ARBITER)

        c_sdf = self.soc.get_cluster("0x50")
        self.assertEqual(c_sdf.role, CoreClusterRole.CLUSTER_0x50_NEURAL_SDF_RAYMARCH_CORE)

        c_gate = self.soc.get_cluster("0x60")
        self.assertEqual(c_gate.role, CoreClusterRole.CLUSTER_0x60_CONSTITUTIONAL_HEARTBEAT_GATE)

    # =========================================================================
    # 3. DIE FLOORPLAN TOPOLOGY AND POWER TESTS
    # =========================================================================

    def test_die_floorplan_topology_verification(self) -> None:
        """Verifies floorplan geometry, die area sum, and power budget."""
        topo = self.soc.verify_die_floorplan_topology()
        self.assertEqual(topo["total_clusters"], 6)
        self.assertEqual(topo["vital_max_hp"], 6)
        self.assertAlmostEqual(topo["total_area_fraction"], 1.0, delta=1e-4)
        self.assertLessEqual(topo["total_tdp_watts"], 29.4)  # within 5% of 28W
        self.assertEqual(topo["central_ring_bandwidth_gbps"], 128.0)
        self.assertTrue(topo["is_topology_valid"])

    # =========================================================================
    # 4. FOUR APPLICATION SPECTRA TESTS
    # =========================================================================

    def test_four_application_spectra(self) -> None:
        """Verifies the 4 application spectra descriptors and educational metadata."""
        spectra = self.soc.list_spectra()
        self.assertEqual(len(spectra), 4)

        # 1. Edge Robotics
        s_robotics = self.soc.get_spectrum(ApplicationSpectrumKind.SPECTRUM_1_AUTONOMOUS_EDGE_ROBOTICS)
        self.assertIn("0x30", s_robotics.primary_cluster_affinities)
        self.assertEqual(s_robotics.latency_target_us, 15.0)
        self.assertIn("det(J) = 1.0", s_robotics.governing_equation)

        # 2. Local AI
        s_ai = self.soc.get_spectrum(ApplicationSpectrumKind.SPECTRUM_2_COGNITIVE_LOCAL_AI_COEXISTENCE)
        self.assertIn("0x40", s_ai.primary_cluster_affinities)
        self.assertIn("mu_UMA < 0.78", s_ai.governing_equation)

        # 3. Aerospace Inductive Power
        s_aero = self.soc.get_spectrum(ApplicationSpectrumKind.SPECTRUM_3_AEROSPACE_INDUCTIVE_POWER_GRID)
        self.assertIn("0x10", s_aero.primary_cluster_affinities)
        self.assertIn("Delta V_droop <= 50.0 mV", s_aero.governing_equation)

        # 4. Volumetric Holography
        s_holo = self.soc.get_spectrum(ApplicationSpectrumKind.SPECTRUM_4_VOLUMETRIC_HOLOGRAPHIC_3D_RENDERING)
        self.assertIn("0x50", s_holo.primary_cluster_affinities)
        self.assertEqual(s_holo.latency_target_us, 8.33)  # 120 FPS

    # =========================================================================
    # 5. WORKLOAD DISPATCH SIMULATION TESTS
    # =========================================================================

    def test_workload_dispatch_simulation(self) -> None:
        """Verifies workload execution receipts, voltage droop clamp, and VBlank compliance."""
        # Dispatch Robotics Workload
        rcpt_robotics = self.soc.dispatch_workload(
            workload_name="BipedStabilizationStep",
            spectrum_kind=ApplicationSpectrumKind.SPECTRUM_1_AUTONOMOUS_EDGE_ROBOTICS,
            payload_size_bytes=128,
        )
        self.assertTrue(rcpt_robotics.is_symplectic_conserved)
        self.assertTrue(rcpt_robotics.is_vblank_shadow_compliant)
        self.assertLessEqual(rcpt_robotics.max_voltage_droop_mv, 50.0)
        self.assertEqual(rcpt_robotics.vital_max_hp, 6)

        # Dispatch Local AI Coexistence Workload
        rcpt_ai = self.soc.dispatch_workload(
            workload_name="InterleavedTokenDecode",
            spectrum_kind=ApplicationSpectrumKind.SPECTRUM_2_COGNITIVE_LOCAL_AI_COEXISTENCE,
            payload_size_bytes=256,
        )
        self.assertTrue(rcpt_ai.is_vblank_shadow_compliant)
        self.assertLessEqual(rcpt_ai.max_voltage_droop_mv, 50.0)

        # Dispatch Aerospace Telemetry Workload
        rcpt_aero = self.soc.dispatch_workload(
            workload_name="SolarInverterSurgeTelemetry",
            spectrum_kind=ApplicationSpectrumKind.SPECTRUM_3_AEROSPACE_INDUCTIVE_POWER_GRID,
            payload_size_bytes=64,
        )
        self.assertLessEqual(rcpt_aero.max_voltage_droop_mv, 35.0)

        # Dispatch Holographic Raymarch Workload
        rcpt_holo = self.soc.dispatch_workload(
            workload_name="Gyroid120FpsRaymarchPass",
            spectrum_kind=ApplicationSpectrumKind.SPECTRUM_4_VOLUMETRIC_HOLOGRAPHIC_3D_RENDERING,
            payload_size_bytes=512,
        )
        self.assertTrue(rcpt_holo.is_vblank_shadow_compliant)
        self.assertGreater(rcpt_holo.elapsed_cycles, 0)

    # =========================================================================
    # 6. THEORY DAG REGISTRATION TESTS
    # =========================================================================

    def test_theory_dag_registration(self) -> None:
        """Verifies that THEORY-CHRONO-SYMPLECTIC-PROCESSOR-ARCHITECTURE is registered and valid in Theory DAG."""
        dag = GLOBAL_THEORY_DEPENDENCY_GRAPH
        node = dag.get_theory("THEORY-CHRONO-SYMPLECTIC-PROCESSOR-ARCHITECTURE")
        self.assertIsNotNone(node)
        self.assertEqual(node.epistemic_status, TheoryEpistemicStatus.DOKAZANE)
        self.assertIn("6-sector floorplan 0x10..0x60", node.assumptions)
        self.assertIn("THEORY-PATTERN-RHYTHM-CADENCE-COUPLING", node.parent_theories)
        self.assertIn("THEORY-H07-FARADAY-LENZ-PDN", node.parent_theories)
        self.assertIn("THEORY-P3-CHRONOMETRIC-SHADOW", node.parent_theories)


if __name__ == "__main__":
    unittest.main()
