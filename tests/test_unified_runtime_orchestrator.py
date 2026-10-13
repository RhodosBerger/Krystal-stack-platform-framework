# ==============================================================================
# KRYSTAL-STACK: UNIT TESTS PRE ZJEDNOTENÝ EPISTEMICKÝ META-ORCHESTRÁTOR
# ==============================================================================
# Verifies the exact 5-step closed-loop lifecycle:
#   1. context = runtime.inspect_and_resolve(request)
#   2. plan = runtime.compose(request=request, cascade=cascade_plan, context=context)
#   3. shadow = runtime.execute_shadow(plan)
#   4. comparison = runtime.compare(expected=plan, observed=shadow.telemetry)
#   5. runtime.update_hypotheses(comparison)
#
# Asserts the 5 Cardinal Dimensions: Model, Backend, Worker, Runtime, Evidence.
# Constitutional Invariant: assert VITAL_MAX_HP == 6
# ==============================================================================

import unittest
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.runtime.unified_orchestrator import (
    EpistemicMetaOrchestrator,
    RuntimeResolutionContext,
    ComposedExecutionPlan,
    ShadowExecutionArtifact,
    EpistemicComparisonReport,
    HypothesisUpdateRecord,
    GLOBAL_EPISTEMIC_META_ORCHESTRATOR
)
from krystal_web_hub.hybrid_mcp_antigravity_connector import GLOBAL_HYBRID_MCP_CONNECTOR


class TestUnifiedRuntimeOrchestrator(unittest.TestCase):
    def setUp(self):
        self.runtime = GLOBAL_EPISTEMIC_META_ORCHESTRATOR
        self.mcp_conn = GLOBAL_HYBRID_MCP_CONNECTOR

    # ─── 1. INVARIANT TESTS ──────────────────────────────────────────────────

    def test_01_vital_max_hp_invariant(self):
        """Strictly asserts that platform-wide invariant VITAL_MAX_HP is 6."""
        self.assertEqual(VITAL_MAX_HP, 6, "Inviolable system invariant: VITAL_MAX_HP must equal 6")

    # ─── 2. STEP-BY-STEP CANONICAL LIFECYCLE ─────────────────────────────────

    def test_02_step_1_inspect_and_resolve(self):
        """Verifies Step 1: inspect_and_resolve collects hardware, profile, thermal, and cognitive strain."""
        req = {
            "request_id": "req-t01",
            "prompt": "Generuj 3D procedurálnu scénu pre Godot 4 s dihedrálnym zrkadlením",
            "profile": "balanced_copilot"
        }
        ctx = self.runtime.inspect_and_resolve(req)

        self.assertIsInstance(ctx, RuntimeResolutionContext)
        self.assertEqual(ctx.request_id, "req-t01")
        self.assertEqual(ctx.config_profile, "balanced_copilot")
        self.assertIn("gpu_device", ctx.hardware)
        self.assertIn("thermal_state", ctx.thermal_state)
        self.assertGreater(ctx.cognitive_strain["prompt_complexity_omega"], 0.0)
        self.assertEqual(ctx.vital_max_hp, 6)

    def test_03_step_2_compose_cardinal_dimensions(self):
        """Verifies Step 2: compose resolves Model, Backend, Worker, Runtime, and Evidence modes."""
        req = {
            "request_id": "req-t02",
            "prompt": "Jednoduchý test pre lokálny model",
            "profile": "offline_hermetic"
        }
        ctx = self.runtime.inspect_and_resolve(req)
        plan = self.runtime.compose(request=req, cascade=None, context=ctx)

        self.assertIsInstance(plan, ComposedExecutionPlan)
        # 1. Model: offline_hermetic should pick slm_local_1b
        self.assertEqual(plan.model, "slm_local_1b")
        # 2. Backend
        self.assertIn(plan.backend, ["DIRECTML_NPU", "OPENVINO_GPU_IRIS_XE", "CPU_AVX_THREADED"])
        # 3. Worker Profile
        self.assertIn("worker_threads", plan.worker_profile)
        self.assertIn("queue_policy", plan.worker_profile)
        # 4. Runtime Profile
        self.assertIn(plan.runtime_profile, ["ULTRA_LOW_LATENCY", "BALANCED_THROUGHPUT", "SUSTAINED_THERMAL"])
        # 5. Evidence Mode
        self.assertEqual(plan.evidence_mode, "FORMAL_GATE_WITH_SHADOW_COW")
        # Expected Telemetry
        self.assertGreater(plan.expected_telemetry["expected_latency_ms"], 0.0)
        self.assertEqual(plan.vital_max_hp, 6)

    def test_04_step_3_execute_shadow(self):
        """Verifies Step 3: execute_shadow runs in non-actuating isolation and captures telemetry."""
        req = {"request_id": "req-t03", "prompt": "Shadow trial"}
        plan = self.runtime.compose(request=req, context=self.runtime.inspect_and_resolve(req))
        shadow = self.runtime.execute_shadow(plan)

        self.assertIsInstance(shadow, ShadowExecutionArtifact)
        self.assertTrue(shadow.executed_in_isolation)
        self.assertIn("latency_ms", shadow.telemetry)
        self.assertIn("memory_mb", shadow.telemetry)
        self.assertIn("thermal_delta_c", shadow.telemetry)
        self.assertEqual(shadow.shadow_output["status"], "SHADOW_TRIAL_COMPLETE")
        self.assertEqual(shadow.vital_max_hp, 6)

    def test_05_step_4_compare(self):
        """Verifies Step 4: compare calculates expected vs observed telemetry deltas and verdict."""
        req = {"request_id": "req-t04", "prompt": "Comparison benchmark"}
        plan = self.runtime.compose(request=req, context=self.runtime.inspect_and_resolve(req))
        shadow = self.runtime.execute_shadow(plan)
        cmp = self.runtime.compare(expected=plan, observed=shadow.telemetry)

        self.assertIsInstance(cmp, EpistemicComparisonReport)
        self.assertIn("delta_latency_ms", cmp.deltas)
        self.assertIn("latency_error_pct", cmp.relative_errors)
        self.assertIn(cmp.verdict, ["AUTHORITY_APPROVED", "CONDITIONAL_APPROVAL_WITH_CALIBRATION", "REJECTED_FALSIFIED_REPLAN_REQUIRED"])
        self.assertTrue(cmp.invariants_verified)
        self.assertEqual(cmp.vital_max_hp, 6)

    def test_06_step_5_update_hypotheses(self):
        """Verifies Step 5: update_hypotheses feeds comparison back into the ledger and grants greenlight."""
        req = {"request_id": "req-t05", "prompt": "Hypothesis update"}
        plan = self.runtime.compose(request=req, context=self.runtime.inspect_and_resolve(req))
        shadow = self.runtime.execute_shadow(plan)
        cmp = self.runtime.compare(expected=plan, observed=shadow.telemetry)
        upd = self.runtime.update_hypotheses(cmp)

        self.assertIsInstance(upd, HypothesisUpdateRecord)
        self.assertGreaterEqual(upd.hypotheses_updated, 1)
        self.assertGreaterEqual(upd.active_corroborated_count, 1)
        self.assertIsInstance(upd.final_execution_greenlight, bool)
        self.assertEqual(upd.vital_max_hp, 6)

    # ─── 3. ALL-IN-ONE ORCHESTRATION FACADE ──────────────────────────────────

    def test_07_orchestrate_full_cycle(self):
        """Verifies high-level orchestrate_full_cycle running all 5 steps synchronously."""
        res = self.runtime.orchestrate_full_cycle({
            "request_id": "req-facade-01",
            "prompt": "Orchestruj kompletný kaskádový inferenčný graf",
            "profile": "max_compute_npu"
        })
        self.assertEqual(res["status"], "ORCHESTRATION_CYCLE_COMPLETE")
        self.assertIn("step_1_context", res)
        self.assertIn("step_2_plan", res)
        self.assertIn("step_3_shadow", res)
        self.assertIn("step_4_comparison", res)
        self.assertIn("step_5_update", res)
        self.assertEqual(res["vital_max_hp"], 6)

    # ─── 4. MCP PROTOCOL INTEGRATION ─────────────────────────────────────────

    def test_08_mcp_krystal_unified_orchestrate_tool(self):
        """Verifies krystal-unified-orchestrate invocation through the Hybrid MCP Connector."""
        mcp_res = self.mcp_conn.execute_tool("krystal-unified-orchestrate", {
            "request": {
                "request_id": "mcp-req-99",
                "prompt": "MCP vyhodnotenie cez zjednotený orchestrátor",
                "profile": "balanced_copilot"
            }
        })
        self.assertEqual(mcp_res["status"], "ORCHESTRATION_CYCLE_COMPLETE")
        self.assertEqual(mcp_res["step_2_plan"]["evidence_mode"], "FORMAL_GATE_WITH_SHADOW_COW")
        self.assertEqual(mcp_res["vital_max_hp"], 6)

    def test_09_mcp_catalog_enumeration(self):
        """Verifies that krystal-unified-orchestrate is exposed in tools/list (15 tools total)."""
        tools = self.mcp_conn.mcp_tool_schemas
        tool_names = [t["name"] for t in tools]
        self.assertGreaterEqual(len(tools), 15)
        self.assertIn("krystal-unified-orchestrate", tool_names)
        self.assertIn("krystal-karaf-evaluate", tool_names)
        self.assertIn("krystal-llm-train-epoch", tool_names)
        self.assertIn("krystal-llm-rag-benchmark", tool_names)

    # ─── 5. KARAF DECLARATIVE TEMPLATES INTEGRATION ──────────────────────────

    def test_10_karaf_templates_working_in_orchestration_cycle(self):
        """
        Verifies that resource_envelope.yaml, worker_elasticity.yaml, and cascade_placement.yaml
        actively calculate and parameterize the Epistemic Meta-Orchestrator.
        """
        req = {
            "request_id": "req-karaf-01",
            "prompt": "Generuj optimalizovaný kaskádový inferenčný plán pre neurónovú syntézu",
            "profile": "balanced_copilot"
        }
        res = self.runtime.orchestrate_full_cycle(req)
        plan = res["step_2_plan"]

        # Check KARAF Templates container
        self.assertIn("karaf_templates", plan)
        kt = plan["karaf_templates"]

        # 1. resource_envelope.yaml
        self.assertIn("resource_envelope", kt)
        are = kt["resource_envelope"]
        self.assertEqual(are["template"], "resource_envelope.yaml")
        self.assertEqual(are["algorithm"], "ARE_SOLVER_V2")
        self.assertGreaterEqual(are["optimal_workers"], 1)
        self.assertGreaterEqual(are["optimal_pl1_watts"], 10.0)

        # 2. worker_elasticity.yaml
        self.assertIn("worker_elasticity", kt)
        we = kt["worker_elasticity"]
        self.assertEqual(we["template"], "worker_elasticity.yaml")
        self.assertEqual(we["algorithm"], "WORKER_GRADIENT_V1")
        self.assertIn(we["decision"], ["SCALE_UP", "KEEP_TOPOLOGY", "SHRINK_WORKERS"])
        self.assertIn(we["elasticity_state"], ["ELASTIC_GROWTH", "SATURATION", "CONTENTION_AVOIDANCE"])

        # 3. cascade_placement.yaml
        self.assertIn("cascade_placement", kt)
        cp = kt["cascade_placement"]
        self.assertEqual(cp["template"], "cascade_placement.yaml")
        self.assertEqual(cp["algorithm"], "CASCADE_PLACEMENT_V2")
        # Constitutional mandate: verifier is ALWAYS CPU_SIMD for deterministic verification
        self.assertEqual(cp["active_placements"]["verifier"], "CPU_SIMD")

        # Worker Profile must be synthesized directly from the solvers
        wp = plan["worker_profile"]
        self.assertEqual(wp["worker_threads"], we["recommended_workers"])
        self.assertEqual(wp["power_limit_pl1_watts"], are["optimal_pl1_watts"])
        self.assertEqual(wp["elasticity_decision"], we["decision"])
        self.assertEqual(wp["node_placements"]["verifier"], "CPU_SIMD")

        # Invariant check
        self.assertEqual(res["vital_max_hp"], 6)

    def test_11_cascade_placement_deterministic_verifier_invariant(self):
        """Verifies that across all hardware conditions, verifier is strictly CPU_SIMD and VITAL_MAX_HP == 6."""
        for profile in ["low_latency_gaming", "balanced_copilot", "max_compute_npu", "offline_hermetic"]:
            req = {"request_id": f"req-inv-{profile}", "prompt": "Test verifier", "profile": profile}
            ctx = self.runtime.inspect_and_resolve(req)
            plan = self.runtime.compose(req, context=ctx)
            self.assertEqual(plan.worker_profile["node_placements"]["verifier"], "CPU_SIMD")
            self.assertEqual(plan.karaf_templates["cascade_placement"]["active_placements"]["verifier"], "CPU_SIMD")
            self.assertEqual(plan.vital_max_hp, 6)


if __name__ == "__main__":
    unittest.main()

