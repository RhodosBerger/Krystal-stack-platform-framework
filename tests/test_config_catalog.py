import unittest
from krystal_kernel.verification.config_catalog import (
    CATALOG, select_profile, list_by_category, estimated_runtime_h)


class ConfigCatalogTests(unittest.TestCase):
    def test_categories_present(self):
        for c in ("battery", "balanced", "performance"):
            self.assertTrue(list_by_category(c))

    def test_plugged_picks_max_perf(self):
        self.assertEqual(select_profile(80, plugged=True).name, "max_burst")

    def test_low_battery_picks_battery_profile(self):
        self.assertEqual(select_profile(10).category, "battery")

    def test_runtime_target_respected(self):
        p = select_profile(100, target_runtime_h=8, battery_wh=50)
        self.assertGreaterEqual(estimated_runtime_h(p, 50, 100), 8)

    def test_balanced_is_default(self):
        self.assertEqual(select_profile(70).category, "balanced")


if __name__ == "__main__":
    unittest.main()
