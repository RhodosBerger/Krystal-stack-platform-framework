#!/usr/bin/env python3
"""
Unit tests for Chained Convolutional Matrix Workloads, Boolean CRUD Database,
and Streaming Log Service.
Enforces constitutional invariant: assert VITAL_MAX_HP == 6.
"""

import unittest
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.convolutional_matrix_workloads import (
    BooleanMatrix,
    ConvolutionKernel,
    convolve_boolean_matrix,
    CANONICAL_CONVOLUTION_KERNELS,
    MatrixStreamingLogService,
    BooleanMatrixCrudDatabase,
    ChainedMatrixWorkloadPipeline,
    GLOBAL_BOOLEAN_CRUD_DATABASE,
    GLOBAL_CHAINED_MATRIX_WORKLOAD_PIPELINE
)


class TestConvolutionalMatrixWorkloads(unittest.TestCase):

    def setUp(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_boolean_matrix_basic_ops(self):
        mat = BooleanMatrix(rows=8, cols=8)
        self.assertEqual(mat.rows, 8)
        self.assertEqual(mat.cols, 8)
        self.assertEqual(mat.active_bit_count, 0)
        self.assertEqual(mat.density, 0.0)

        # Set and toggle bits
        mat.set(2, 3, True)
        self.assertTrue(mat.get(2, 3))
        self.assertEqual(mat.active_bit_count, 1)

        mat.toggle(2, 3)
        self.assertFalse(mat.get(2, 3))
        self.assertEqual(mat.active_bit_count, 0)

    def test_boolean_matrix_bitwise_operations(self):
        m1 = BooleanMatrix(rows=4, cols=4, initial_bits=[
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
            [False, False, True, True]
        ])
        m2 = BooleanMatrix(rows=4, cols=4, initial_bits=[
            [True, True, False, False],
            [False, True, True, False],
            [False, False, True, True],
            [True, False, False, True]
        ])

        m_and = m1.bitwise_and(m2)
        self.assertTrue(m_and.get(0, 0))
        self.assertFalse(m_and.get(0, 1))

        m_or = m1.bitwise_or(m2)
        self.assertTrue(m_or.get(0, 1))

        m_xor = m1.bitwise_xor(m2)
        self.assertFalse(m_xor.get(0, 0))
        self.assertTrue(m_xor.get(0, 1))

        m_not = m1.bitwise_not()
        self.assertFalse(m_not.get(0, 0))
        self.assertTrue(m_not.get(0, 1))

    def test_spatial_2d_convolution(self):
        # 8x8 matrix with a central 4x4 block
        grid = [[False for _ in range(8)] for _ in range(8)]
        for r in range(2, 6):
            for c in range(2, 6):
                grid[r][c] = True

        src = BooleanMatrix(rows=8, cols=8, initial_bits=grid)
        self.assertEqual(src.active_bit_count, 16)

        kernel = CANONICAL_CONVOLUTION_KERNELS["KERNEL_DIFFUSION_SMOOTHER_3X3"]
        convolved = convolve_boolean_matrix(src, kernel)
        self.assertEqual(convolved.rows, 8)
        self.assertEqual(convolved.cols, 8)
        # Smoothing preserves or diffuses bits
        self.assertGreater(convolved.active_bit_count, 0)

    def test_boolean_matrix_crud_database(self):
        db = BooleanMatrixCrudDatabase()
        mat = BooleanMatrix(rows=8, cols=8)
        mat.set(0, 0, True)
        mat.set(7, 7, True)

        # CREATE
        ent = db.create_entity(
            entity_id="TEST_ENT_01",
            name="Test Entity",
            matrix=mat,
            collection="unit_test",
            metadata={"priority": 1}
        )
        self.assertEqual(ent.entity_id, "TEST_ENT_01")
        self.assertEqual(ent.active_bits, 2)
        self.assertEqual(ent.vital_max_hp, 6)

        # READ
        retrieved = db.get_entity("TEST_ENT_01")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Test Entity")

        # CONVOLVE UPDATE
        updated = db.convolve_entity_update("TEST_ENT_01", "KERNEL_DIFFUSION_SMOOTHER_3X3")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.version, 2)

        # DELETE
        deleted = db.delete_entity("TEST_ENT_01")
        self.assertTrue(deleted)
        self.assertIsNone(db.get_entity("TEST_ENT_01"))

    def test_chained_pipeline_execution(self):
        pipeline = ChainedMatrixWorkloadPipeline()
        report = pipeline.execute_pipeline()
        self.assertEqual(len(report.stages), 6)
        self.assertEqual(report.vital_max_hp, 6)
        self.assertGreater(report.total_duration_us, 0.0)
        self.assertTrue(report.transaction_committed)


if __name__ == "__main__":
    unittest.main()
