"""
Unit Tests for Janet Lisp Engine Integration & Subproject Validation
====================================================================
Verifies that all Janet subproject source files are structurally valid,
S-expression trees can be tokenized and parsed, all core engine modules
export expected mathematical and pipeline definitions, and the Janet bridge
emulator executes without errors.
"""

import unittest
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from krystal_janet.janet_bridge import JanetValidator, JanetSExpressionParser, JanetEngineRunner
from openworld_engine.world_semantic_compiler import WorldSemanticCompiler

class TestJanetIntegration(unittest.TestCase):

    def setUp(self):
        self.janet_dir = os.path.join(REPO_ROOT, "krystal_janet")

    def test_01_janet_file_bracket_balance(self):
        target_files = [
            "project.janet",
            "krystal_sdf.janet",
            "cyclic_organism.janet",
            "neural_ascii_engine.janet",
            "governor.janet",
            "topological_vm.janet",
            "antigravity_peg.janet",
            "openworld_chunk.janet",
            "main.janet"
        ]
        for fname in target_files:
            fpath = os.path.join(self.janet_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"File missing: {fname}")
            res = JanetValidator.validate_file(fpath)
            self.assertTrue(res["valid"], f"Bracket imbalance in {fname}: {res.get('bracket_counts')}")
            self.assertGreater(res["line_count"], 5, f"File unexpectedly small: {fname}")

    def test_02_janet_sdf_definitions(self):
        fpath = os.path.join(self.janet_dir, "krystal_sdf.janet")
        res = JanetValidator.validate_file(fpath)
        defs = res["definitions"]
        expected = ["smin", "smax", "sdf-sphere", "sdf-box", "sdf-torus", "mod-mirror-x", "mod-twist-y"]
        for exp in expected:
            self.assertIn(exp, defs, f"Expected definition '{exp}' missing from krystal_sdf.janet")

    def test_03_janet_peg_definitions(self):
        fpath = os.path.join(self.janet_dir, "antigravity_peg.janet")
        res = JanetValidator.validate_file(fpath)
        defs = res["definitions"]
        expected = ["prompt-peg", "parse-antigravity-prompt", "ast->mathematical-ir"]
        for exp in expected:
            self.assertIn(exp, defs, f"Expected definition '{exp}' missing from antigravity_peg.janet")

    def test_04_janet_cyclic_organism_definitions(self):
        fpath = os.path.join(self.janet_dir, "cyclic_organism.janet")
        res = JanetValidator.validate_file(fpath)
        defs = res["definitions"]
        expected = [
            "create-organism",
            "potential-gradient",
            "dissipative-force",
            "total-hamiltonian",
            "lyapunov-stability-index",
            "symplectic-step",
            "generate-phase-trajectory"
        ]
        for exp in expected:
            self.assertIn(exp, defs, f"Expected definition '{exp}' missing from cyclic_organism.janet")

    def test_05_janet_neural_ascii_definitions(self):
        fpath = os.path.join(self.janet_dir, "neural_ascii_engine.janet")
        res = JanetValidator.validate_file(fpath)
        defs = res["definitions"]
        expected = ["create-camera", "scene-sdf", "calc-normal", "raymarch", "render-frame"]
        for exp in expected:
            self.assertIn(exp, defs, f"Expected definition '{exp}' missing from neural_ascii_engine.janet")

    def test_06_janet_governor_definitions(self):
        fpath = os.path.join(self.janet_dir, "governor.janet")
        res = JanetValidator.validate_file(fpath)
        defs = res["definitions"]
        expected = ["create-governor", "compute-entropy", "update-budget", "replenish-budget"]
        for exp in expected:
            self.assertIn(exp, defs, f"Expected definition '{exp}' missing from governor.janet")

    def test_07_janet_topological_vm_definitions(self):
        fpath = os.path.join(self.janet_dir, "topological_vm.janet")
        res = JanetValidator.validate_file(fpath)
        defs = res["definitions"]
        expected = ["create-ring-buffer", "ring-push", "ring-pop", "create-topological-vm", "alloc-queue", "step-vm"]
        for exp in expected:
            self.assertIn(exp, defs, f"Expected definition '{exp}' missing from topological_vm.janet")

    def test_08_janet_dsl_codegen_validity(self):
        compiler = WorldSemanticCompiler()
        spec = compiler.compile_natural_prompt("kaňon s lávou a 8 rekurzií")
        janet_code = compiler.to_janet_dsl(spec)

        tokens = JanetSExpressionParser.tokenize(janet_code)
        self.assertGreater(len(tokens), 20)

        # Validate bracket balance of generated Janet code
        counts = {"(": 0, ")": 0, "[": 0, "]": 0, "{": 0, "}": 0}
        for char in janet_code:
            if char in counts:
                counts[char] += 1
        self.assertEqual(counts["("], counts[")"])
        self.assertEqual(counts["["], counts["]"])
        self.assertEqual(counts["{"], counts["}"])

    def test_09_janet_bridge_runner_execution(self):
        res = JanetEngineRunner.run_subproject(frames=4, test_mode=True)
        self.assertEqual(res["status"], "OK")
        self.assertEqual(res["cycles"], 4)
        self.assertGreater(res["final_h"], 0.0)


if __name__ == "__main__":
    unittest.main()
