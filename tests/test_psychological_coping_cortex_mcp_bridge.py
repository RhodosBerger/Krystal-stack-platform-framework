#!/usr/bin/env python3
"""
Unit tests for Krystal Psychological Coping Cortex & MCP Processor-Whisperer Bridge.
Validates:
  - Psychological coping mechanism taxonomy and triage priority ordering.
  - Semantic synonym clustering (LLM vocabulary memory bandwidth savings).
  - Physical silicon die unlock report (Core i5 == Core i7 Willow Cove parity).
  - Processor Whisperer nanosecond instruction hints.
  - Model Context Protocol (MCP) manifest and tool dispatch.
  - Architectural invariant VITAL_MAX_HP == 6.
  - Regression tests for defects H1-H11 (see TestCopingBridgeRegressions).
"""

import subprocess
import sys
import unittest
import krystal_kernel.psychological_coping_cortex_mcp_bridge as bridge_module
from krystal_kernel.psychological_coping_cortex_mcp_bridge import (
    VITAL_MAX_HP,
    PsychologicalCopingMechanism,
    CopingPriorityDescriptor,
    SiliconDieUnlockReport,
    SiliconDieUnlocker,
    SemanticSynonymCompilationOptimizer,
    CortexProcessorWhispererMcpBridge,
    GLOBAL_COPING_CORTEX_MCP_BRIDGE
)

POLICY = CortexProcessorWhispererMcpBridge.TRIAGE_POLICY


class TestPsychologicalCopingCortexMcpBridge(unittest.TestCase):
    """Test suite for psychological coping triage, LLM synonym pruning, and silicon unlock."""

    def setUp(self):
        self.bridge = CortexProcessorWhispererMcpBridge()
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_coping_mechanisms_enum_and_constants(self):
        """Validates all 7 psychological coping mechanisms exist."""
        expected = {
            "SUBLIMATION",
            "REPRESSION",
            "PROJECTION",
            "RATIONALIZATION",
            "COMPARTMENTALIZATION",
            "HYPERVIGILANCE",
            "REGRESSION"
        }
        actual = {m.value for m in PsychologicalCopingMechanism}
        self.assertEqual(expected, actual)
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_coping_triage_decision_matrix(self):
        """Validates behavioral triage under various stress environments."""
        # 1. Thermal Emergency (>= 85°C) -> REGRESSION
        t1 = self.bridge.evaluate_and_schedule_task("EmergencyCooling", thermal_pressure_pct=88.0)
        self.assertEqual(t1.coping_mechanism, PsychologicalCopingMechanism.REGRESSION)
        self.assertEqual(t1.priority_rank, 1)
        self.assertEqual(t1.memory_allocation_policy, "LEGACY_SIMD_BUFFER")
        self.assertEqual(t1.vital_max_hp, 6)

        # 2. Memory Contention (>= 2.0) -> COMPARTMENTALIZATION
        t2 = self.bridge.evaluate_and_schedule_task("HeavyVramSlab", memory_contention=2.5, thermal_pressure_pct=50.0)
        self.assertEqual(t2.coping_mechanism, PsychologicalCopingMechanism.COMPARTMENTALIZATION)
        self.assertEqual(t2.priority_rank, 3)
        self.assertEqual(t2.memory_allocation_policy, "ISOLATED_UMA_PLACED_HEAP")

        # 3. High Context Entropy (>= 0.70) -> SUBLIMATION
        t3 = self.bridge.evaluate_and_schedule_task("AsciiRenderer", context_entropy=0.85, memory_contention=1.0, thermal_pressure_pct=50.0)
        self.assertEqual(t3.coping_mechanism, PsychologicalCopingMechanism.SUBLIMATION)
        self.assertEqual(t3.priority_rank, 4)
        self.assertEqual(t3.memory_allocation_policy, "PROCEDURAL_ASCII_SLAB")

        # 4. Nominal Cruising -> PROJECTION (rank 2: strictly after thermal emergency)
        t4 = self.bridge.evaluate_and_schedule_task("FastFrameInterpolation", context_entropy=0.20, memory_contention=1.0, thermal_pressure_pct=50.0)
        self.assertEqual(t4.coping_mechanism, PsychologicalCopingMechanism.PROJECTION)
        self.assertEqual(t4.priority_rank, 2)
        self.assertEqual(t4.speculative_depth, 4)
        self.assertEqual(t4.memory_allocation_policy, "HOST_COHERENT_PREFETCH_RING")

    def test_ui_process_scheduling(self):
        """Validates UI process queue sorting by priority rank."""
        processes = [
            {"name": "ComplexMeshSublimation", "entropy": 0.80, "contention": 1.0},
            {"name": "FastPredictiveProjection", "entropy": 0.10, "contention": 1.0},
            {"name": "HeavyUavBufferIsolation", "entropy": 0.30, "contention": 2.2},
        ]
        sorted_procs = self.bridge.schedule_ui_processes(processes)
        self.assertEqual(len(sorted_procs), 3)
        # Exact order, not just monotonicity (sort() alone guarantees monotonicity):
        # Projection (2) -> Isolation (3) -> Sublimation (4)
        self.assertEqual(
            [p["name"] for p in sorted_procs],
            ["FastPredictiveProjection", "HeavyUavBufferIsolation", "ComplexMeshSublimation"]
        )

    def test_semantic_synonym_clustering(self):
        """Validates semantic synonym clustering saves LLM embedding memory bandwidth."""
        text = "synthesize compose render forecast predict approximate simplify isolate"
        res = SemanticSynonymCompilationOptimizer.filter_and_cluster_synonyms(text)
        self.assertIn("primary_coping_mechanism", res)
        self.assertIn(res["primary_coping_mechanism"], ["SUBLIMATION", "PROJECTION", "RATIONALIZATION", "COMPARTMENTALIZATION"])
        self.assertGreaterEqual(res["embedding_bandwidth_saved_pct"], 75.0)
        self.assertEqual(res["vital_max_hp"], 6)

    def test_silicon_die_unlock_report(self):
        """Validates the modelled envelope is labelled honestly and computed consistently."""
        report = SiliconDieUnlocker.unlock_full_silicon_potential(32.0)
        # Nothing is probed, so nothing may be reported as confirmed.
        self.assertFalse(report.physical_die_parity_confirmed)
        self.assertFalse(report.msr_writes_performed)
        self.assertEqual(report.figure_status, "MODEL_UNVERIFIED")
        self.assertIn("fused", report.die_parity_note)
        self.assertEqual(report.stock_pl1_watts, 15.0)
        self.assertEqual(report.unlocked_pl1_watts, 32.0)
        self.assertEqual(report.requested_pl1_watts, 32.0)
        self.assertEqual(report.stock_ring_bus_ghz, 1.8)
        self.assertEqual(report.unlocked_ring_bus_ghz, 2.8)
        # Never probed / never written -> unknown / inactive (review F1).
        self.assertIsNone(report.stock_dram_rapl_limited)
        self.assertFalse(report.unlocked_dram_rapl_active)
        self.assertEqual(report.unlocked_safe_trip_c, 85.0)
        self.assertEqual(report.gpu_slice_undervolt_mv, -40.0)
        # Clock-ratio upper bound: 3.4 / 1.8 - 1 = 88.9 %
        self.assertAlmostEqual(report.overall_die_efficiency_boost_pct, 88.9, places=1)
        self.assertEqual(report.vital_max_hp, 6)

    def test_whisper_speculative_instruction(self):
        """Validates speculative instruction hints for Processor Whisperer."""
        # PROJECTION
        hint_proj = self.bridge.whisper_speculative_instruction(100, PsychologicalCopingMechanism.PROJECTION)
        self.assertEqual(hint_proj["pc"], 100)
        self.assertIn("AVX-512", hint_proj["isa_recommendation"])
        self.assertTrue(hint_proj["speculative_branch_taken"])
        self.assertEqual(hint_proj["vital_max_hp"], 6)

        # RATIONALIZATION
        hint_rat = self.bridge.whisper_speculative_instruction(200, PsychologicalCopingMechanism.RATIONALIZATION)
        self.assertEqual(hint_rat["pc"], 200)
        self.assertIn("Tile4", hint_rat["isa_recommendation"])
        self.assertTrue(hint_rat["speculative_branch_taken"])

        # REGRESSION
        hint_reg = self.bridge.whisper_speculative_instruction(300, PsychologicalCopingMechanism.REGRESSION)
        self.assertEqual(hint_reg["pc"], 300)
        self.assertIn("AVX2", hint_reg["isa_recommendation"])
        self.assertFalse(hint_reg["speculative_branch_taken"])

    def test_mcp_manifest_and_dispatch(self):
        """Validates Model Context Protocol tools registration and execution."""
        manifest = self.bridge.get_mcp_tools_manifest()
        tool_names = [t["name"] for t in manifest]
        self.assertIn("cortex_triage_coping_priority", tool_names)
        self.assertIn("cortex_cluster_synonym_embeddings", tool_names)
        self.assertIn("silicon_unlock_potential", tool_names)

        # Test tool 1
        res1 = self.bridge.dispatch_mcp_tool(
            "cortex_triage_coping_priority",
            {"task_name": "AsciiDitherRender", "context_entropy": 0.8}
        )
        self.assertEqual(res1["coping_mechanism"], "SUBLIMATION")
        self.assertEqual(res1["vital_max_hp"], 6)

        # Test tool 2
        res2 = self.bridge.dispatch_mcp_tool(
            "cortex_cluster_synonym_embeddings",
            {"query_text": "create render art code"}
        )
        self.assertEqual(res2["primary_coping_mechanism"], "SUBLIMATION")

        # Test tool 3
        res3 = self.bridge.dispatch_mcp_tool(
            "silicon_unlock_potential",
            {"target_pl1_watts": 30.0}
        )
        self.assertFalse(res3["physical_die_parity_confirmed"])
        self.assertEqual(res3["figure_status"], "MODEL_UNVERIFIED")
        self.assertEqual(res3["vital_max_hp"], 6)

        # Test unknown tool
        res_err = self.bridge.dispatch_mcp_tool("unknown_tool", {})
        self.assertIn("error", res_err)
        self.assertEqual(res_err["vital_max_hp"], 6)


class TestCopingBridgeRegressions(unittest.TestCase):
    """One test per defect confirmed by the 2026-10-09 debugging pass (H1-H11)."""

    def setUp(self):
        self.bridge = CortexProcessorWhispererMcpBridge()

    def test_h1_ranks_are_strict_total_order(self):
        """H1: a thermal emergency must never tie with nominal work."""
        ranks = [policy[0] for policy in POLICY.values()]
        self.assertEqual(len(ranks), len(set(ranks)))
        emerg = self.bridge.evaluate_and_schedule_task("E", thermal_pressure_pct=95.0)
        nominal = self.bridge.evaluate_and_schedule_task("N", context_entropy=0.1)
        self.assertLess(emerg.priority_rank, nominal.priority_rank)

    def test_h1_emergency_sorted_first_regardless_of_input_order(self):
        procs = [{"name": "nominal", "entropy": 0.1}, {"name": "hot", "thermal": 99.0}]
        out = self.bridge.schedule_ui_processes(procs)
        self.assertEqual(out[0]["name"], "hot")

    def test_h2_triage_reachable_set_is_documented(self):
        """H2: triage can only emit the four mechanisms in _TRIAGE_POLICY."""
        seen = {
            self.bridge.classify(e, c, t).value
            for t in (0, 84.9, 85, 100) for c in (0, 1.99, 2.0) for e in (0, 0.69, 0.7, 1.0)
        }
        expected = {m.value for m in POLICY}
        self.assertEqual(seen, expected)

    def test_h3_non_finite_inputs_fail_safe(self):
        """H3: NaN/inf must map to REGRESSION (no speculation), not PROJECTION depth 4."""
        for bad in (float("nan"), float("inf"), float("-inf")):
            for args in ((bad, 1.0, 50.0), (0.1, bad, 50.0), (0.1, 1.0, bad)):
                desc = self.bridge.evaluate_and_schedule_task("x", *args)
                self.assertEqual(desc.coping_mechanism, PsychologicalCopingMechanism.REGRESSION, args)
                self.assertEqual(desc.speculative_depth, 0)

    def test_h4_ui_scheduler_honours_thermal(self):
        out = self.bridge.schedule_ui_processes([{"name": "hot", "thermal": 99.0, "entropy": 0.1}])
        self.assertEqual(out[0]["coping_triage"]["coping_mechanism"], "REGRESSION")
        out = self.bridge.schedule_ui_processes([{"name": "hot", "thermal_pressure_pct": 90.0}])
        self.assertEqual(out[0]["coping_triage"]["coping_mechanism"], "REGRESSION")

    def test_h5_no_substring_false_positives(self):
        S = SemanticSynonymCompilationOptimizer
        # Substring and free-prefix false positives (review F3).
        for q in ("start smart particle", "decode", "hello world",
                  "mutex", "codec", "dropdown", "blockchain", "dividend", "guardian"):
            res = S.filter_and_cluster_synonyms(q)
            self.assertEqual(res["detected_semantic_clusters"], {}, q)
            self.assertIsNone(res["primary_coping_mechanism"], q)
        # Regular inflections still match.
        self.assertEqual(S.filter_and_cluster_synonyms("rendering")["primary_coping_mechanism"], "SUBLIMATION")
        self.assertEqual(S.filter_and_cluster_synonyms("Predicted,")["primary_coping_mechanism"], "PROJECTION")
        self.assertEqual(S.filter_and_cluster_synonyms("isolated")["primary_coping_mechanism"], "COMPARTMENTALIZATION")

    def test_h5_buckets_are_disjoint(self):
        seen = {}
        for mech, words in SemanticSynonymCompilationOptimizer.COPING_SEMANTIC_BUCKETS.items():
            for w in words:
                self.assertNotIn(w, seen, f"{w!r} in both {seen.get(w)} and {mech.value}")
                seen[w] = mech.value

    def test_h6_saving_is_derived_not_constant(self):
        S = SemanticSynonymCompilationOptimizer
        self.assertEqual(S.filter_and_cluster_synonyms("")["embedding_bandwidth_saved_pct"], 0.0)
        self.assertEqual(S.filter_and_cluster_synonyms("hello world")["embedding_bandwidth_saved_pct"], 0.0)
        res = S.filter_and_cluster_synonyms("render hello")
        expected = 100.0 * (1.0 - res["pruned_clustered_lookups"] / res["unpruned_dictionary_lookups"])
        self.assertAlmostEqual(res["embedding_bandwidth_saved_pct"], round(expected, 1))
        self.assertEqual(res["figure_status"], "MODEL_UNVERIFIED")

    def test_h7_tie_break_independent_of_word_order(self):
        S = SemanticSynonymCompilationOptimizer
        a = S.filter_and_cluster_synonyms("predict render")["primary_coping_mechanism"]
        b = S.filter_and_cluster_synonyms("render predict")["primary_coping_mechanism"]
        self.assertEqual(a, b)

    def test_h8_pl1_request_is_respected_and_no_gain_without_budget(self):
        low = SiliconDieUnlocker.unlock_full_silicon_potential(10.0)
        self.assertEqual(low.unlocked_pl1_watts, 15.0)      # clamped to stock, not 28
        self.assertEqual(low.overall_die_efficiency_boost_pct, 0.0)
        self.assertEqual(SiliconDieUnlocker.unlock_full_silicon_potential(20.0).unlocked_pl1_watts, 20.0)
        self.assertEqual(SiliconDieUnlocker.unlock_full_silicon_potential(50.0).unlocked_pl1_watts, 35.0)
        with self.assertRaises(ValueError):
            SiliconDieUnlocker.unlock_full_silicon_potential(float("nan"))

    def test_h9_whisper_agrees_with_triage_spec_depth(self):
        for mech, policy in POLICY.items():
            hint = self.bridge.whisper_speculative_instruction(0, mech)
            self.assertEqual(hint["speculative_branch_taken"], policy[2] > 0, mech.value)

    def test_h10_manifest_has_mcp_input_schema(self):
        for tool in self.bridge.get_mcp_tools_manifest():
            self.assertIn("inputSchema", tool)
            self.assertEqual(tool["inputSchema"]["type"], "object")

    def test_h11_dispatch_null_and_invalid_args(self):
        res = self.bridge.dispatch_mcp_tool(
            "cortex_triage_coping_priority", {"task_name": None, "context_entropy": None})
        self.assertNotIn("error", res)
        self.assertEqual(res["task_name"], "AnonymousTask")
        res = self.bridge.dispatch_mcp_tool("cortex_triage_coping_priority", {"context_entropy": "abc"})
        self.assertIn("error", res)
        res = self.bridge.dispatch_mcp_tool("silicon_unlock_potential", None)
        self.assertNotIn("error", res)
        res = self.bridge.dispatch_mcp_tool("cortex_cluster_synonym_embeddings", {"query_text": None})
        self.assertIsNone(res["primary_coping_mechanism"])


class TestCopingBridgeReviewFollowUps(unittest.TestCase):
    """Tests added by the 2026-10-09 code review (T1, T2, F4, F5, F7)."""

    def setUp(self):
        self.bridge = CortexProcessorWhispererMcpBridge()

    def test_t1_threshold_boundaries_are_inclusive(self):
        C = CortexProcessorWhispererMcpBridge
        M = PsychologicalCopingMechanism
        self.assertEqual(C.classify(0.0, 0.0, 85.0), M.REGRESSION)
        self.assertEqual(C.classify(0.0, 0.0, 84.999999), M.PROJECTION)
        self.assertEqual(C.classify(0.0, 2.0, 50.0), M.COMPARTMENTALIZATION)
        self.assertEqual(C.classify(0.0, 1.999999, 50.0), M.PROJECTION)
        self.assertEqual(C.classify(0.70, 0.0, 50.0), M.SUBLIMATION)
        self.assertEqual(C.classify(0.699999, 0.0, 50.0), M.PROJECTION)
        # Precedence: thermal > memory > entropy.
        self.assertEqual(C.classify(1.0, 5.0, 85.0), M.REGRESSION)
        self.assertEqual(C.classify(1.0, 5.0, 50.0), M.COMPARTMENTALIZATION)

    def test_t2_import_does_not_reconfigure_stdout(self):
        """Loads the module file in a fresh interpreter (bypassing the package __init__, which
        imports unrelated modules) with a probe stdout, and counts reconfigure() calls."""
        code = (
            "import importlib.util, sys\n"
            "calls = []\n"
            "class Probe:\n"
            "    encoding = 'probe'\n"
            "    def reconfigure(self, **kw): calls.append(kw)\n"
            "    def write(self, s): return len(s)\n"
            "    def flush(self): pass\n"
            "real_out, real_err = sys.stdout, sys.stderr\n"
            "sys.stdout, sys.stderr = Probe(), Probe()\n"
            "try:\n"
            f"    spec = importlib.util.spec_from_file_location('coping_probe', {bridge_module.__file__!r})\n"
            "    mod = importlib.util.module_from_spec(spec)\n"
            "    sys.modules['coping_probe'] = mod\n"
            "    spec.loader.exec_module(mod)\n"
            "finally:\n"
            "    sys.stdout, sys.stderr = real_out, real_err\n"
            "print(len(calls))\n"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "0")

    def test_f7_bool_is_rejected(self):
        with self.assertRaises(TypeError):
            CortexProcessorWhispererMcpBridge.classify(True, 1.0, 50.0)
        res = self.bridge.dispatch_mcp_tool("cortex_triage_coping_priority", {"thermal_pressure_pct": True})
        self.assertIn("error", res)

    def test_f4_overflow_returns_error_dict(self):
        res = self.bridge.dispatch_mcp_tool("cortex_triage_coping_priority", {"context_entropy": 10 ** 400})
        self.assertIn("error", res)
        res = self.bridge.dispatch_mcp_tool("silicon_unlock_potential", {"target_pl1_watts": 10 ** 400})
        self.assertIn("error", res)

    def test_f5_ui_scheduler_treats_null_as_default(self):
        out = self.bridge.schedule_ui_processes(
            [{"name": None, "entropy": None, "contention": None, "thermal": None}])
        self.assertEqual(out[0]["coping_triage"]["coping_mechanism"], "PROJECTION")
        self.assertEqual(out[0]["coping_triage"]["task_name"], "UI-Task")
        # thermal_pressure_pct: null falls back to the "thermal" key, not to the default.
        out = self.bridge.schedule_ui_processes([{"thermal_pressure_pct": None, "thermal": 95.0}])
        self.assertEqual(out[0]["coping_triage"]["coping_mechanism"], "REGRESSION")

    def test_history_is_bounded(self):
        for i in range(CortexProcessorWhispererMcpBridge.HISTORY_LIMIT + 10):
            self.bridge.evaluate_and_schedule_task(str(i))
        self.assertEqual(len(self.bridge._execution_history), CortexProcessorWhispererMcpBridge.HISTORY_LIMIT)


if __name__ == "__main__":
    unittest.main()
