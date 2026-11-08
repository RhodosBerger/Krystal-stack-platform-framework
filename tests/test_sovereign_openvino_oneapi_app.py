"""
Unit and Integration Tests for SovereignOpenVinoOneApiApp (OpenVINO + oneAPI + XERJ Query-1 Chatbot Bridge).
"""

import os
import unittest

from krystal_kernel.sovereign_openvino_oneapi_app import (
    SovereignOpenVinoOneApiApp,
    VITAL_MAX_HP,
)


class TestSovereignOpenVinoOneApiApp(unittest.TestCase):
    def setUp(self) -> None:
        self.app = SovereignOpenVinoOneApiApp()

    def test_constitutional_invariant_and_ir_compilation(self) -> None:
        self.assertEqual(VITAL_MAX_HP, 6)
        status = self.app.get_hardware_and_api_status()
        self.assertEqual(status["status"], "ONLINE_VERIFIED")
        self.assertEqual(status["vital_max_hp"], 6)
        self.assertEqual(len(status["compiled_ir_digest"]), 64)
        self.assertIn("CPU", status["hardware"]["openvino_available_devices"])

    def test_chatbot_hybrid_query_and_hysteresis_transition(self) -> None:
        # 1. Green Zone (UMA 0.32) -> GPU_ZERO_COPY_FP16 with 512 token budget
        res_green = self.app.execute_hybrid_query(
            namespace="krystal_sovereign",
            prompt="Aké je zrýchlenie pre Vulkan Iris Xe?",
            uma_utilization=0.32,
            max_tokens=512,
            payload_bytes=32768,
        )
        self.assertEqual(res_green["query_response"]["verdict"], "MEASURED_WITH_SCOPE")
        self.assertEqual(res_green["governor_state"]["active_compute_path"], "GPU_ZERO_COPY_FP16")
        self.assertEqual(res_green["governor_state"]["allocated_token_budget"], 512)
        self.assertGreater(res_green["token_comparison"]["token_reduction_ratio"], 40.0)

        # 2. Orange Zone Spike (UMA 0.74) -> Immediate fallback to CPU_AVX2_INT8 + 12-frame hysteresis pin
        res_orange = self.app.execute_hybrid_query(
            namespace="krystal_sovereign",
            prompt="Porovnaj frame latency pre CPU Baseline vs Vulkan Iris Xe",
            uma_utilization=0.74,
            max_tokens=512,
            payload_bytes=4096,
        )
        self.assertEqual(res_orange["governor_state"]["active_compute_path"], "CPU_AVX2_INT8")
        self.assertEqual(res_orange["governor_state"]["allocated_token_budget"], 24)
        self.assertTrue(res_orange["governor_state"]["hysteresis_pinned_cpu"])
        self.assertEqual(res_orange["governor_state"]["hysteresis_dwell_counter"], 0)
        self.assertEqual(res_orange["oneapi_openvino_routing"]["selected_device"], "CPU")

        # 3. Subsequent Green frame (UMA 0.30) remains pinned to CPU due to 12-frame dwell hysteresis
        res_dwell = self.app.execute_hybrid_query(
            namespace="krystal_sovereign",
            prompt="What is the saved bandwidth for Zero Copy Ring?",
            uma_utilization=0.30,
            max_tokens=512,
            payload_bytes=32768,
        )
        self.assertEqual(res_dwell["governor_state"]["active_compute_path"], "CPU_AVX2_INT8")
        self.assertTrue(res_dwell["governor_state"]["hysteresis_pinned_cpu"])
        self.assertEqual(res_dwell["governor_state"]["hysteresis_dwell_counter"], 1)

    def test_concrete_model_parameter_benchmarks_and_frontend_file(self) -> None:
        bm = self.app.benchmark_concrete_models_on_hardware()
        self.assertEqual(bm["vital_max_hp"], 6)
        models = bm["models"]
        self.assertEqual(len(models), 5)
        param_scales = sorted([m["parameter_count_billions"] for m in models])
        self.assertEqual(param_scales, [0.05, 4.6, 4.65, 6.7, 12.0])

        res_ddes = self.app.execute_hybrid_query(
            namespace="krystal_sovereign",
            prompt="Porovnaj frame latency pre CPU Baseline vs Vulkan Iris Xe",
            selected_model_key="krystal/ddes-reasoner-v2",
            cognition_level=5,
            pattern_type="P3_CAUSAL_DIFFERENTIAL",
        )
        self.assertIn("DDES_HYBRID_FACT_AND_SENSE_MEASURED", res_ddes["evidence_classification"])
        self.assertIn("krystal/ddes-reasoner-v2", res_ddes["conversational_reply"])

        frontend_path = os.path.join(os.getcwd(), "frontend", "sovereign_openvino_oneapi_console.html")
        self.assertTrue(os.path.exists(frontend_path))
        with open(frontend_path, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn("KRYSTAL FLUID DESIGN MANUAL v2.0", html)
        self.assertIn("data-persona=", html)
        self.assertIn("btn-mode-ddes", html)

    def test_ddes_compositor_scraper_and_finetuning_export(self) -> None:
        state = self.app.compositor.get_compositor_state()
        self.assertEqual(state["vital_max_hp"], 6)
        self.assertEqual(len(state["precached_themes"]), 3)
        self.assertGreaterEqual(state["samples_count"], 3)

        synth = self.app.compositor.compose_hybrid_conclusion(
            prompt="Compare RAG prefill tokens vs Krystal AST fast-path on quantized GGUF",
            theme_id="theme_neurosymbolic_fastpath",
            cognition_level=5,
            pattern_type="P3_CAUSAL_DIFFERENTIAL",
            scrape_target="OpenVINO quantization research",
            invoke_live_llm=False,
        )
        self.assertEqual(synth["status"], "SYNTHESIZED_AND_STORED")
        self.assertEqual(synth["vital_max_hp"], 6)
        self.assertIn("conducted_neurosymbolic_facts", synth)
        self.assertIn("hybrid_sense_packet", synth)
        self.assertGreater(len(synth["conducted_neurosymbolic_facts"]["conducted_triples"]), 0)
        self.assertGreaterEqual(synth["hybrid_sense_packet"]["sense_coherence_score"], 0.8)
        node = synth["conclusion_node_sample"]
        self.assertEqual(node["cognition_level_code"], "L5_MULTI_SOURCE_SYNTHESIS")
        self.assertEqual(node["pattern_type"], "P3_CAUSAL_DIFFERENTIAL")
        self.assertEqual(len(node["node_digest"]), 64)

        jsonl = self.app.compositor.export_finetuning_jsonl()
        self.assertIn("NODE-DDES-0001", jsonl)
        self.assertIn("L5_MULTI_SOURCE_SYNTHESIS", jsonl)


if __name__ == "__main__":
    unittest.main()
