#!/usr/bin/env python3
"""
Unit tests for Strand/Page Processors, NPU Life Action Puzzle Decoder, and Alternative Assignment Engine.
"""

import unittest
from krystal_kernel.strand_npu_action_puzzle_engine import (
    AlternativeAssignmentScheduler,
    NpuPuzzleDecoderAndInterpreter,
    GLOBAL_STRAND_NPU_ENGINE,
    LifeActionCellState,
    PuzzleActionType,
    ProcessorStrandStatus,
    VITAL_MAX_HP
)


class TestStrandNpuActionPuzzle(unittest.TestCase):
    def setUp(self):
        self.scheduler = AlternativeAssignmentScheduler()

    def test_canonical_strand_processors(self):
        self.assertEqual(len(self.scheduler.processors), 4)
        alpha = self.scheduler.processors["strand_page_0"]
        self.assertEqual(alpha.address_range, (0x0000, 0x1FFF))
        delta = self.scheduler.processors["strand_page_3"]
        self.assertEqual(delta.address_range, (0x6000, 0x7FFF))
        self.assertEqual(alpha.vital_max_hp, 6)

    def test_npu_bytecode_decoder(self):
        decoder = NpuPuzzleDecoderAndInterpreter()
        raw_hex = "0102040301020403"
        puzzle = decoder.decode_bytecode_to_puzzle(raw_hex, "test_puzzle_01")
        self.assertEqual(puzzle.puzzle_id, "test_puzzle_01")
        self.assertEqual(puzzle.grid_size, 8)
        self.assertEqual(len(puzzle.grid), 8)
        self.assertEqual(len(puzzle.grid[0]), 8)
        self.assertGreater(puzzle.count_neighbors(2, 2), 0)
        self.assertEqual(puzzle.vital_max_hp, 6)

    def test_npu_step_interpretation(self):
        decoder = NpuPuzzleDecoderAndInterpreter()
        puzzle = decoder.decode_bytecode_to_puzzle("0x0102040501020405", "test_puz_step")
        init_energy = puzzle.energy_level
        res = decoder.interpret_and_solve_step(puzzle)
        self.assertIn(res["chosen_action"], [a.value for a in PuzzleActionType])
        self.assertEqual(res["steps_taken"], 1)
        self.assertGreater(res["npu_inference_latency_us"], 0.0)
        self.assertEqual(res["vital_max_hp_rule"], 6)

    def test_alternative_assignment_on_strand_stall(self):
        rec = self.scheduler.trigger_strand_stall_and_reassign("strand_page_0")
        self.assertEqual(rec.source_strand_id, "strand_page_0")
        self.assertEqual(rec.target_execution_body, "NPU_COPROCESSOR")
        self.assertAlmostEqual(rec.pre_staged_latency_us, 0.08, places=2)
        self.assertAlmostEqual(rec.cold_swap_latency_us, 120.0, places=1)
        self.assertEqual(rec.effective_speedup_ratio, 1500.0)
        self.assertEqual(rec.vital_max_hp, 6)

        proc = self.scheduler.processors["strand_page_0"]
        self.assertEqual(proc.status, ProcessorStrandStatus.DIVERTED_TO_NPU)
        self.assertGreaterEqual(proc.reassignments_count, 1)

    def test_scheduler_status_and_global_singleton(self):
        status = GLOBAL_STRAND_NPU_ENGINE.get_status()
        self.assertIn("processors", status)
        self.assertEqual(len(status["processors"]), 4)
        self.assertGreaterEqual(status["active_puzzles_count"], 1)
        self.assertEqual(status["vital_max_hp"], 6)


if __name__ == '__main__':
    unittest.main()
