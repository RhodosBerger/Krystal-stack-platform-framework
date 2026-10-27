#!/usr/bin/env python3
"""
================================================================================
UNIT TEST: KSL DEMONSTRATION PIPELINE VERIFICATION
================================================================================
Module: tests/test_demo_ksl_pipeline.py
================================================================================
"""

import unittest
from demo_ksl_pipeline import run_ksl_demonstration


class TestDemoKslPipeline(unittest.TestCase):
    """Verifies that the KSL end-to-end demonstration runs cleanly with 100% verification."""

    def test_run_ksl_demonstration(self) -> None:
        res = run_ksl_demonstration()
        self.assertTrue(res["success"])
        self.assertTrue(res["compilation_result"].success)
        self.assertEqual(sorted(res["pareto_front"]), ["K", "S1"])
        self.assertTrue(res["dag_acyclic"])
        self.assertTrue(all(res["sva_results"].values()))
        self.assertTrue(res["receipt_id"].startswith("PROMPT-RECEIPT-"))


if __name__ == "__main__":
    unittest.main()
