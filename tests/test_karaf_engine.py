# ==============================================================================
# KRYSTAL-STACK: UNIT TESTS PRE KARAF (Krystal Adaptive Runtime Algorithm Fabric)
# ==============================================================================
# Verifies:
#   1. Invariant assert VITAL_MAX_HP == 6
#   2. Canonical YAML Template Loading & Contract Validation (9 templates)
#   3. Adaptive Resource Envelope Solver (ARE_SOLVER_V2)
#   4. Worker Elasticity Gradient (WORKER_GRADIENT_V1)
#   5. Cascade Hardware Placement Optimizer
#   6. Thermal Predictive Governor
#   7. UMA Contention Predictor
#   8. Cascade Critical-Path Optimizer
#   9. Epistemic Confidence Decay
#  10. Interference Correlation Matrix
#  11. Runtime Hypothesis Miner
#  12. KarafFabric Master Execution Cycle
#  13. Hybrid MCP Protocol Integration (krystal-karaf-evaluate)
# ==============================================================================

import unittest
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf_engine import (
    CanonicalTemplateLoader,
    AdaptiveResourceEnvelopeSolver,
    WorkerElasticityGradient,
    CascadeHardwarePlacementOptimizer,
    ThermalPredictiveGovernor,
    UmaContentionPredictor,
    CascadeCriticalPathOptimizer,
    EpistemicConfidenceDecay,
    InterferenceCorrelationMatrix,
    RuntimeHypothesisMiner,
    KarafFabric,
    GLOBAL_KARAF_FABRIC
)
from krystal_web_hub.hybrid_mcp_antigravity_connector import GLOBAL_HYBRID_MCP_CONNECTOR


class TestKarafEngine(unittest.TestCase):
    def setUp(self):
        self.karaf = GLOBAL_KARAF_FABRIC
        self.mcp_conn = GLOBAL_HYBRID_MCP_CONNECTOR

    # ─── 1. INVARIANT TESTS ──────────────────────────────────────────────────

    def test_01_vital_max_hp_invariant(self):
        """Strictly asserts that platform-wide invariant VITAL_MAX_HP is 6."""
        self.assertEqual(VITAL_MAX_HP, 6, "Inviolable system invariant: VITAL_MAX_HP must equal 6")

    # ─── 2. CANONICAL YAML TEMPLATE LOADER ───────────────────────────────────

    def test_02_load_all_canonical_templates(self):
        """Verifies loading of all 9 canonical algorithm templates from krystal_kernel/templates/."""
        expected_templates = [
            "resource_envelope",
            "worker_elasticity",
            "cascade_placement",
            "thermal_predictor",
            "uma_contention",
            "critical_path",
            "evidence_decay",
            "interference_matrix",
            "hypothesis_miner"
        ]
        self.assertEqual(len(self.karaf.templates), 9)
        for t_name in expected_templates:
            self.assertIn(t_name, self.karaf.templates)
            tpl = self.karaf.templates[t_name]
            self.assertIn("algorithm_template", tpl)
            algo = tpl["algorithm_template"]
            self.assertIn("identity", algo)
            self.assertIn("epistemic", algo)
            self.assertIn("inputs", algo)
            self.assertIn("objective", algo)
            self.assertIn("constraints", algo)

    # ─── 3. ADAPTIVE RESOURCE ENVELOPE SOLVER ────────────────────────────────

    def test_03_adaptive_resource_envelope_solver(self):
        """Verifies ARE_SOLVER_V2 optimization under normal and thermal stress conditions."""
        # 1. Normal conditions
        res_normal = AdaptiveResourceEnvelopeSolver.solve_envelope(
            cpu_util=35.0, mem_pressure=0.30, gpu_load=50.0,
            thermal_headroom_c=40.0, cascade_memory_mb=2048.0
        )
        self.assertEqual(res_normal["algorithm"], "ARE_SOLVER_V2")
        self.assertGreaterEqual(res_normal["optimal_workers"], 1)
        self.assertLessEqual(res_normal["optimal_workers"], 8)
        self.assertIn(res_normal["optimal_pl1_watts"], [15.0, 25.0, 28.0, 35.0])
        self.assertEqual(res_normal["vital_max_hp"], 6)

        # 2. Thermal stress conditions -> should reduce PL1 and/or workers
        res_stress = AdaptiveResourceEnvelopeSolver.solve_envelope(
            cpu_util=90.0, mem_pressure=0.85, gpu_load=95.0,
            thermal_headroom_c=3.0, cascade_memory_mb=7000.0
        )
        self.assertLessEqual(res_stress["optimal_pl1_watts"], 28.0)

    # ─── 4. WORKER ELASTICITY GRADIENT ───────────────────────────────────────

    def test_04_worker_elasticity_gradient(self):
        """Verifies WORKER_GRADIENT_V1 evaluating Gw = Delta Utility / Delta Workers."""
        # Low load -> scaling up should help
        res_low = WorkerElasticityGradient.evaluate_elasticity(
            current_workers=2, p99_latency_ms=15.0, llc_miss_rate=0.04,
            context_switch_rate=120.0, memory_bandwidth_util=25.0
        )
        self.assertEqual(res_low["algorithm"], "WORKER_GRADIENT_V1")
        self.assertIn(res_low["decision"], ["SCALE_UP", "KEEP_TOPOLOGY"])

        # High contention -> saturation or shrink
        res_contention = WorkerElasticityGradient.evaluate_elasticity(
            current_workers=8, p99_latency_ms=180.0, llc_miss_rate=0.45,
            context_switch_rate=3500.0, memory_bandwidth_util=92.0
        )
        self.assertIn(res_contention["decision"], ["SHRINK_WORKERS", "KEEP_TOPOLOGY"])
        self.assertEqual(res_contention["vital_max_hp"], 6)

    # ─── 5. CASCADE HARDWARE PLACEMENT OPTIMIZER ─────────────────────────────

    def test_05_cascade_hardware_placement_optimizer(self):
        """Verifies dynamic hardware placement and displacement upon GPU load / UMA pressure."""
        # 1. Normal state: Reasoner on GPU
        normal_p = CascadeHardwarePlacementOptimizer.optimize_placement(
            gpu_load_pct=40.0, uma_pressure=0.35, thermal_headroom_c=35.0
        )
        self.assertEqual(normal_p["active_placements"]["verifier"], "CPU_SIMD")  # Strict determinism
        self.assertEqual(normal_p["active_placements"]["reasoner"], "OPENVINO_GPU_IRIS_XE")

        # 2. GPU Overload: Reasoner displaced to NPU or CPU
        gpu_overload = CascadeHardwarePlacementOptimizer.optimize_placement(
            gpu_load_pct=94.0, uma_pressure=0.40, thermal_headroom_c=25.0, npu_available=True
        )
        self.assertEqual(gpu_overload["active_placements"]["reasoner"], "DIRECTML_NPU")
        self.assertGreater(len(gpu_overload["displacements"]), 0)

        # 3. UMA Pressure High: Coder set to lazy load
        uma_high = CascadeHardwarePlacementOptimizer.optimize_placement(
            gpu_load_pct=40.0, uma_pressure=0.88, thermal_headroom_c=30.0
        )
        self.assertEqual(uma_high["active_placements"]["coder"], "LAZY_LOAD_PAGED_CPU")

    # ─── 6. THERMAL PREDICTIVE GOVERNOR ──────────────────────────────────────

    def test_06_thermal_predictive_governor(self):
        """Verifies multi-scale thermal projections (100ms, 500ms, 2000ms) and breach alarms."""
        # 1. Cool idle state -> allow burst
        cool = ThermalPredictiveGovernor.predict_and_govern(
            current_temp_c=48.0, power_watts=15.0, cpu_load=20.0, freq_ghz=2.4
        )
        self.assertEqual(cool["thermal_risk"], "NOMINAL_SAFE")
        self.assertEqual(cool["governor_action"], "ALLOW_TURBO_BURST")

        # 2. Approaching thermal ceiling -> early predictive reaction
        hot = ThermalPredictiveGovernor.predict_and_govern(
            current_temp_c=88.0, power_watts=35.0, cpu_load=98.0, freq_ghz=4.2
        )
        self.assertIn(hot["thermal_risk"], ["ELEVATED_PRED_RISK", "CRITICAL_PRED_BREACH"])
        self.assertIn(hot["governor_action"], ["REDUCE_PARALLELISM_MINUS_ONE", "MIGRATE_OPTIONAL_NODES_AND_DROP_FREQ"])
        self.assertTrue(hot["safety_gate_compliant"])

    # ─── 7. UMA CONTENTION PREDICTOR ─────────────────────────────────────────

    def test_07_uma_contention_predictor(self):
        """Verifies UMA pressure metric and stall risk evaluation."""
        uma = UmaContentionPredictor.evaluate_uma(
            ram_used_mb=14500.0, ram_total_mb=16384.0,
            gpu_uma_alloc_mb=4096.0, dirty_pages_mb=850.0,
            swap_rate_pages=45.0, mem_bandwidth_pct=88.0
        )
        self.assertEqual(uma["algorithm"], "UMA_CONTENTION_PREDICTOR_V1")
        self.assertGreater(uma["der_uma_pressure"], 0.5)
        self.assertGreater(uma["pred_uma_stall_risk"], 0.5)
        self.assertGreater(len(uma["mitigation_actions"]), 0)
        self.assertEqual(uma["vital_max_hp"], 6)

    # ─── 8. CASCADE CRITICAL-PATH OPTIMIZER ──────────────────────────────────

    def test_08_cascade_critical_path_optimizer(self):
        """Verifies DAG longest-path topological sort and bottleneck isolation."""
        nodes = [
            {"id": "classifier", "latency_ms": 6.0},
            {"id": "rag", "latency_ms": 10.0},
            {"id": "reasoner", "latency_ms": 42.0},
            {"id": "verifier", "latency_ms": 4.0}
        ]
        edges = [
            ("classifier", "rag"),
            ("rag", "reasoner"),
            ("reasoner", "verifier")
        ]
        crit = CascadeCriticalPathOptimizer.find_and_optimize_critical_path(nodes, edges)
        self.assertEqual(crit["algorithm"], "DAG_CRITICAL_PATH_V1")
        self.assertEqual(crit["end_to_end_latency_ms"], 62.0)
        self.assertEqual(crit["primary_optimization_target"], "reasoner")
        self.assertEqual(crit["vital_max_hp"], 6)

    # ─── 9. EPISTEMIC CONFIDENCE DECAY ───────────────────────────────────────

    def test_09_epistemic_confidence_decay(self):
        """Verifies exponential decay C(t) = C0 * e^(-lambda * t) and expiration states."""
        # CPU_LOAD: half-life = 100ms
        # At 100ms: confidence should be ~0.50 * initial
        decay_100ms = EpistemicConfidenceDecay.evaluate_decay("CPU_LOAD", 1.0, 100.0)
        self.assertAlmostEqual(decay_100ms["current_confidence"], 0.50, places=2)
        self.assertEqual(decay_100ms["status"], "CONFIDENCE_DEGRADED")

        # At 500ms (5 half-lives) -> TMP_EXPIRED, AUTH_DEFERRED
        decay_500ms = EpistemicConfidenceDecay.evaluate_decay("CPU_LOAD", 1.0, 500.0)
        self.assertLess(decay_500ms["current_confidence"], 0.35)
        self.assertEqual(decay_500ms["status"], "TMP_EXPIRED")
        self.assertEqual(decay_500ms["authority_state"], "AUTH_DEFERRED")

        # HARDWARE_TOPOLOGY: half-life = 1 hour (3.6e6 ms)
        decay_topo = EpistemicConfidenceDecay.evaluate_decay("HARDWARE_TOPOLOGY", 1.0, 1000.0)
        self.assertGreater(decay_topo["current_confidence"], 0.99)
        self.assertEqual(decay_topo["status"], "CONFIDENCE_FRESH")

    # ─── 10. INTERFERENCE CORRELATION MATRIX ─────────────────────────────────

    def test_10_interference_correlation_matrix(self):
        """Verifies 5x5 sliding window correlation matrix across CPU, RAM, GPU, NPU, IO."""
        matrix_calc = InterferenceCorrelationMatrix(window_size=64)
        for i in range(16):
            # Simulate coupled RAM and GPU usage
            val = i * 5.0
            matrix_calc.record_sample(cpu=20.0 + i, ram=val, gpu=val * 1.1, npu=10.0, io=5.0)

        res = matrix_calc.compute_matrix()
        self.assertEqual(res["algorithm"], "INTERFERENCE_MATRIX_V1")
        self.assertEqual(len(res["correlation_matrix"]), 5)
        self.assertEqual(res["vital_max_hp"], 6)

    # ─── 11. RUNTIME HYPOTHESIS MINER ────────────────────────────────────────

    def test_11_runtime_hypothesis_miner(self):
        """Verifies autonomous law discovery and commit to platform HypothesisLedger."""
        miner = RuntimeHypothesisMiner()
        worker_rep = {"elasticity_state": "SATURATION", "current_workers": 8, "recommended_workers": 4}
        therm_rep = {"thermal_risk": "NOMINAL_SAFE"}
        uma_rep = {"pred_uma_stall_risk": 0.85, "der_uma_pressure": 0.78}

        mined = miner.mine_observations(worker_rep, therm_rep, uma_rep)
        self.assertGreaterEqual(len(mined), 2)
        # Check hypothesis registration in ledger
        rules = miner.ledger.get_active_corroborated_rules()
        self.assertTrue(any("RULE_LIMIT_WORKERS" in r for r in rules))

    # ─── 12. MASTER KARAF FABRIC CYCLE ───────────────────────────────────────

    def test_12_master_karaf_fabric_cycle(self):
        """Verifies end-to-end execution of KarafFabric.execute_fabric_cycle()."""
        cycle = self.karaf.execute_fabric_cycle()
        self.assertEqual(cycle["status"], "KARAF_CYCLE_COMPLETE")
        self.assertIn("1_adaptive_resource_envelope", cycle)
        self.assertIn("2_worker_elasticity", cycle)
        self.assertIn("3_cascade_placement", cycle)
        self.assertIn("4_thermal_prediction", cycle)
        self.assertIn("5_uma_contention", cycle)
        self.assertIn("6_cascade_critical_path", cycle)
        self.assertIn("7_evidence_decay", cycle)
        self.assertIn("8_interference_matrix", cycle)
        self.assertIn("9_mined_hypotheses", cycle)
        self.assertEqual(cycle["vital_max_hp"], 6)

    # ─── 13. MCP PROTOCOL INTEGRATION ────────────────────────────────────────

    def test_13_mcp_krystal_karaf_evaluate_tool(self):
        """Verifies krystal-karaf-evaluate tool execution through the Hybrid MCP Connector."""
        mcp_res = self.mcp_conn.execute_tool("krystal-karaf-evaluate", {
            "snapshot": {
                "cpu_util": 55.0, "workers": 4, "mem_pressure": 0.50,
                "ram_used_mb": 8000.0, "ram_total_mb": 16384.0,
                "gpu_load": 70.0, "gpu_uma_mb": 2500.0, "dirty_mb": 200.0,
                "swap_rate": 0.0, "mem_bw_pct": 60.0, "temp_c": 60.0,
                "power_w": 25.0, "freq_ghz": 3.0, "headroom_c": 35.0,
                "p99_latency_ms": 35.0, "llc_miss_rate": 0.15, "cs_rate": 600.0
            }
        })
        self.assertEqual(mcp_res["status"], "KARAF_CYCLE_COMPLETE")
        self.assertEqual(mcp_res["vital_max_hp"], 6)

    def test_14_mcp_catalog_enumeration(self):
        """Verifies that krystal-karaf-evaluate is exposed in tools/list (16 tools total)."""
        tools = self.mcp_conn.mcp_tool_schemas
        tool_names = [t["name"] for t in tools]
        self.assertGreaterEqual(len(tools), 16)
        self.assertIn("krystal-karaf-evaluate", tool_names)
        self.assertIn("krystal-llm-train-epoch", tool_names)
        self.assertIn("krystal-llm-rag-benchmark", tool_names)


if __name__ == "__main__":
    unittest.main()
