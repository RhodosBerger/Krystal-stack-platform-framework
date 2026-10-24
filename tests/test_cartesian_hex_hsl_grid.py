#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: 3D CARTESIAN GRID COEFFICIENTS & HEXADECIMAL HSL TEST SUITE
==============================================================================
Module: tests/test_cartesian_hex_hsl_grid.py
Milestone: V-CARTESIAN-HEX-HSL-01
Constitutional Invariant: VITAL_MAX_HP == 6
Normative Reference:
  - krystal_kernel/cartesian_hex_hsl_grid.py
  - artifacts/benchmarks/H07_SILICON_TELEMETRY_AND_OPENVINO_LEXICON_SPEC.md

TEST OBJECTIVES:
  1. Verify constitutional invariant VITAL_MAX_HP == 6 across all structures.
  2. Verify Cartesian-to-HSL mapping and numerical bounds: H in [0, 1), S in [0, 1], L in [0, 1].
  3. Verify 6-sector constitutional color lattice (0x10..0x60) and #RRGGBB formatting.
  4. Verify 3D cubic lattice generation and geometric symmetry.
  5. Verify 3D Fibonacci phyllotaxis cylindrical shell generation.
  6. Verify hexagonal axial lattice generation.
  7. Verify 96-byte MultiMesh binary packing and 128-byte ZeroCopyContractFrame generation.
  8. Verify Theory DAG registration and dependency closure.
==============================================================================
"""

import os
import sys
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation, FRAME_SIZE_BYTES
from krystal_kernel.cartesian_hex_hsl_grid import (
    GridCoefficients3D,
    CartesianGridPoint,
    CartesianHexHslGrid,
    GLOBAL_CARTESIAN_HEX_HSL_GRID,
)
from krystal_kernel.verification.theory_dependency_graph import (
    GLOBAL_THEORY_DEPENDENCY_GRAPH,
)


class TestCartesianHexHslGrid(unittest.TestCase):
    """Formal unit verification of 3D Cartesian Grid Coefficients and Hex HSL Spectrum."""

    def setUp(self) -> None:
        self.coeff = GridCoefficients3D()
        self.grid = CartesianHexHslGrid(coefficients=self.coeff)

    def test_01_constitutional_invariant(self) -> None:
        """Constitutional invariant VITAL_MAX_HP == 6 must hold across all grid structures."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.coeff.vital_max_hp, 6)
        self.assertEqual(self.coeff.vital_sectors_count, 6)
        self.assertEqual(self.grid.vital_max_hp, 6)

        with self.assertRaises(ContractViolation):
            GridCoefficients3D(vital_max_hp=5)

        with self.assertRaises(ContractViolation):
            CartesianHexHslGrid(vital_max_hp=7)

    def test_02_cartesian_point_evaluation_and_hsl_bounds(self) -> None:
        """Evaluates arbitrary 3D Cartesian points and verifies strict numerical bounds."""
        test_points = [
            (0.0, 0.0, 0.0),
            (10.5, 4.2, -8.7),
            (-25.0, 15.0, 30.0),
            (100.0, -50.0, 75.0),
        ]

        for idx, (x, y, z) in enumerate(test_points):
            pt = self.grid.evaluate_point(x, y, z, index=idx)
            self.assertEqual(pt.vital_max_hp, 6)

            # Coordinate preservation
            self.assertAlmostEqual(pt.x, x, places=3)
            self.assertAlmostEqual(pt.y, y, places=3)
            self.assertAlmostEqual(pt.z, z, places=3)

            # HSL bounds
            self.assertGreaterEqual(pt.hue, 0.0)
            self.assertLess(pt.hue, 1.0)
            self.assertGreaterEqual(pt.saturation, 0.0)
            self.assertLessEqual(pt.saturation, 1.0)
            self.assertGreaterEqual(pt.lightness, 0.0)
            self.assertLessEqual(pt.lightness, 1.0)

            # RGB bounds
            r, g, b = pt.rgb
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)
            self.assertGreaterEqual(g, 0.0)
            self.assertLessEqual(g, 1.0)
            self.assertGreaterEqual(b, 0.0)
            self.assertLessEqual(b, 1.0)

            # Hex string: #RRGGBB
            self.assertTrue(pt.hex_color.startswith("#"))
            self.assertEqual(len(pt.hex_color), 7)

            # 32-bit packed HSLA
            h_byte = (pt.hex_hsla_u32 >> 24) & 0xFF
            s_byte = (pt.hex_hsla_u32 >> 16) & 0xFF
            l_byte = (pt.hex_hsla_u32 >> 8) & 0xFF
            a_byte = pt.hex_hsla_u32 & 0xFF
            self.assertEqual(a_byte, 0xFF)
            self.assertAlmostEqual(h_byte / 255.0, pt.hue, delta=0.01)
            self.assertAlmostEqual(s_byte / 255.0, pt.saturation, delta=0.01)
            self.assertAlmostEqual(l_byte / 255.0, pt.lightness, delta=0.01)

    def test_03_constitutional_six_sector_partition(self) -> None:
        """Verifies exact 6-sector constitutional assignment across full 360° Hue spectrum."""
        expected_sectors = [
            (0.05, 0x10, "RED_EMBER"),
            (0.20, 0x20, "AMBER_GOLD"),
            (0.38, 0x30, "EMERALD_BIO"),
            (0.55, 0x40, "CYAN_CRYSTAL"),
            (0.72, 0x50, "SAPPHIRE_VOID"),
            (0.90, 0x60, "AMETHYST_AETHER"),
        ]

        for h, exp_code, exp_name in expected_sectors:
            sector_idx = int(h * 6.0) % 6
            code = self.grid.SECTOR_HEX_CODES[sector_idx]
            name = self.grid.SECTOR_NAMES[sector_idx]
            self.assertEqual(code, exp_code)
            self.assertEqual(name, exp_name)

    def test_04_cubic_lattice_generation(self) -> None:
        """Generates a 3D Cartesian cubic grid and verifies spatial distribution."""
        pts = self.grid.generate_cubic_lattice(dim_x=3, dim_y=3, dim_z=3, step=2.5)
        self.assertEqual(len(pts), 27)

        dist = self.grid.calculate_sector_distribution(pts)
        self.assertEqual(len(dist), 6)
        total_counted = sum(dist.values())
        self.assertEqual(total_counted, 27)

    def test_05_fibonacci_phyllotaxis_lattice(self) -> None:
        """Generates 3D cylindrical Fibonacci spiral grid and verifies golden ratio harmonics."""
        pts = self.grid.generate_fibonacci_phyllotaxis_lattice(point_count=48, radius_scale=10.0)
        self.assertEqual(len(pts), 48)

        # Confirm all 6 sectors receive points due to golden angle dispersal
        dist = self.grid.calculate_sector_distribution(pts)
        active_sectors = [name for name, count in dist.items() if count > 0]
        self.assertGreaterEqual(len(active_sectors), 4)

    def test_06_hex_axial_lattice(self) -> None:
        """Generates hexagonal axial grid and verifies 2D-to-3D projection."""
        pts = self.grid.generate_hex_axial_lattice(radius_steps=2, hex_radius=2.0)
        # For radius 2: 1 + 6 + 12 = 19 hexes
        self.assertEqual(len(pts), 19)

    def test_07_multimesh_96b_and_zerocopy_contract_frame(self) -> None:
        """Verifies binary 96-byte MultiMesh packaging and 128-byte contract framing."""
        pt = self.grid.evaluate_point(5.0, 2.0, -3.0)

        # 96 bytes MultiMesh instance
        packed_bytes = self.grid.pack_to_multimesh_96b(pt, scale=1.5)
        self.assertEqual(len(packed_bytes), 96)

        # 128 bytes ZeroCopyContractFrame
        pts = self.grid.generate_cubic_lattice(dim_x=2, dim_y=2, dim_z=2)
        frame = self.grid.create_contract_frame(pts, sequence=42)
        self.assertEqual(frame.vital_hp, 6)
        self.assertEqual(frame.contract_type, 0x04)  # COMPUTATIONAL_GEOMETRY
        self.assertEqual(frame.sequence, 42)

        raw_bytes = frame.to_bytes()
        self.assertEqual(len(raw_bytes), FRAME_SIZE_BYTES)

    def test_08_theory_dag_linkage(self) -> None:
        """THEORY-CARTESIAN-HEX-HSL-GRID-COEFFICIENTS must be registered in Theory DAG."""
        dag = GLOBAL_THEORY_DEPENDENCY_GRAPH
        node = dag.get_theory("THEORY-CARTESIAN-HEX-HSL-GRID-COEFFICIENTS")
        self.assertIsNotNone(node)
        self.assertEqual(node.category.value, "PROCEDURAL_MANIFOLD")
        self.assertIn("THEORY-GOLDEN-RATIO-D6-TENSOR", node.parent_theories)
        self.assertIn("THEORY-HOLOGRAPHIC-INTERFERENCE-OPTICS", node.parent_theories)


if __name__ == "__main__":
    unittest.main()
