import unittest

from krystal_kernel.verification.criteria_sets import (
    PIPELINE_OPERATIONS, default_pipeline_criteria)


class CriteriaSetTests(unittest.TestCase):
    def test_full_coverage(self):
        r = default_pipeline_criteria().evaluate("operations", PIPELINE_OPERATIONS)
        self.assertTrue(r.satisfied)
        self.assertEqual(r.coverage, 1.0)
        self.assertEqual(r.jaccard, 1.0)

    def test_partial_with_missing_and_extra(self):
        r = default_pipeline_criteria().evaluate("operations", {"add", "subtract", "foo"})
        self.assertFalse(r.satisfied)
        self.assertAlmostEqual(r.coverage, 2 / 8)
        self.assertAlmostEqual(r.jaccard, 2 / 9)
        self.assertEqual(r.extra, frozenset({"foo"}))
        self.assertIn("divide", r.missing)


if __name__ == "__main__":
    unittest.main()
