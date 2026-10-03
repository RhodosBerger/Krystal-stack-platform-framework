"""Unit tests for the project intelligence engine (priorities / expertise / patterns)."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "python"))

from project_intelligence import ProjectIntelligence  # noqa: E402


class TestProjectIntelligence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pi = ProjectIntelligence()
        cls.snap = cls.pi.snapshot(force=True)

    def test_overview_structure(self):
        o = self.pi.overview()
        for key in ("generated_at", "scan_ms", "repo", "toolchain", "open_priorities", "top3"):
            self.assertIn(key, o)
        self.assertLessEqual(len(o["top3"]), 3)

    def test_priorities_sorted_by_score_and_bounded(self):
        res = self.pi.priorities()
        scores = [p["score"] for p in res["priorities"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for s in scores:
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(s, 100)

    def test_limit(self):
        self.assertEqual(len(self.pi.priorities(limit=3)["priorities"]), 3)

    def test_priority_lookup(self):
        top = self.pi.priorities(limit=1)["priorities"][0]
        self.assertEqual(self.pi.priority(top["id"])["id"], top["id"])
        self.assertIsNone(self.pi.priority("does-not-exist"))

    def test_vulkan_priority_open_while_no_real_dispatch(self):
        driver = os.path.join(ROOT, "src", "python", "vulkan_compute_driver.py")
        with open(driver, "r", encoding="utf-8") as f:
            has_dispatch = "vkCmdDispatch" in f.read().replace('"""', "")
        ids = [p["id"] for p in self.pi.priorities()["priorities"]]
        if not has_dispatch:
            self.assertIn("vulkan-real-dispatch", ids)

    def test_expertise_all_and_filtered(self):
        everything = self.pi.expertise()
        self.assertEqual(everything["count"], len(everything["tracks"]))
        self.assertGreater(everything["count"], 0)
        scoped = self.pi.expertise("python-hotpath-vectorize")
        self.assertEqual(scoped["priority"]["id"], "python-hotpath-vectorize")
        self.assertTrue(scoped["required_tracks"])
        self.assertIsNone(self.pi.expertise("does-not-exist"))

    def test_staffing_signal_vocabulary(self):
        allowed = {"in_house_strong", "in_house_partial", "gap"}
        for t in self.pi.expertise()["tracks"]:
            self.assertIn(t["staffing_signal"], allowed)

    def test_patterns(self):
        res = self.pi.patterns()
        self.assertEqual(res["count"], len(res["patterns"]))
        self.assertGreaterEqual(res["count"], 5)

    def test_include_healthy_flag(self):
        res = self.pi.priorities(include_healthy=True)
        self.assertIn("healthy", res)

    def test_docstring_mention_is_not_an_implementation(self):
        import tempfile
        from project_intelligence import ScanContext
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "m.py"), "w", encoding="utf-8") as f:
                f.write('"""no vkCmdDispatch here"""\n# vkCmdDispatch in a comment\nx = 1\n')
            with open(os.path.join(d, "n.py"), "w", encoding="utf-8") as f:
                f.write("def go(vk):\n    vk.vkCmdDispatch(1, 1, 1)\n")
            ctx = ScanContext(d)
            self.assertNotIn("vkCmdDispatch", ctx.code_symbols("m.py"))
            self.assertIn("vkCmdDispatch", ctx.code_symbols("n.py"))

    def test_vulkan_subsystem_not_verified_without_dispatch(self):
        subs = {s["id"]: s for s in self.snap["repo"]["subsystems"]}
        driver = os.path.join(ROOT, "src", "python", "vulkan_compute_driver.py")
        with open(driver, "r", encoding="utf-8") as f:
            src = f.read()
        import ast
        called = any(isinstance(n, (ast.Name, ast.Attribute)) and getattr(n, "id", getattr(n, "attr", "")) == "vkCmdDispatch"
                     for n in ast.walk(ast.parse(src)))
        self.assertEqual(subs["vulkan-compute"]["runtime_verified"], called)

    def test_analyzer_is_not_its_own_pattern_evidence(self):
        for p in self.pi.patterns()["patterns"]:
            self.assertNotIn("src/python/project_intelligence.py", p["evidence"])


if __name__ == "__main__":
    unittest.main()
