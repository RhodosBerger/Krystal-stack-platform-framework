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
        self.assertGreaterEqual(len(state["precached_themes"]), 4)
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

    def test_cv_quadratic_digital_twin_openvino_verification_loop(self) -> None:
        # 1. Nominal verification with automatic neurosymbolic quantization selection
        res = self.app.compositor.run_cv_digital_twin_verification_loop(
            cv_observations=[0.15, 0.27, None, 0.63, 0.87, None, 1.47, 1.83],
            residual_tolerance=0.02,
        )
        self.assertTrue(res["verified"])
        self.assertEqual(res["status"], "VERIFIED_COMMIT")
        self.assertEqual(res["delta_s"], 1)
        self.assertEqual(res["vital_max_hp"], 6)
        self.assertEqual(res["quadratic_scheme"]["reconstructed_indices"], [2, 5])
        self.assertLessEqual(res["openvino_numerical_validation"]["final_max_residual"], 0.02)
        self.assertIn("hardware_core_allocation", res)
        self.assertTrue(res["hardware_core_allocation"]["gpu_offload_enabled"])
        self.assertIn("P_CORE_SIMD", res["hardware_core_allocation"]["core_schemes"])
        self.assertIn("NPU_SYSTOLIC_ARRAY", res["hardware_core_allocation"]["core_schemes"])

        # 2. Forced coarse quantization with strict tolerance -> rejects with Delta S == 0
        res_reject = self.app.compositor.run_cv_digital_twin_verification_loop(
            cv_observations=[0.15, 0.27, None, 0.63, 0.87, None, 1.47, 1.83],
            residual_tolerance=0.001,
            force_coarse_quantization=True,
        )
        self.assertFalse(res_reject["verified"])
        self.assertEqual(res_reject["status"], "REJECTED_DELTA_S_ZERO")
        self.assertEqual(res_reject["delta_s"], 0)
        self.assertEqual(res_reject["pipeline_semaphores"]["SEM_05_OPENVINO_COMMIT_GATE"], "RED_ROLLBACK_DELTA_S_ZERO")

        # 3. Evaluate all 3 validation scheme datasets across pipeline semaphores
        suite = self.app.compositor.evaluate_validation_scheme_datasets()
        self.assertEqual(suite["datasets_evaluated"], 3)
        s1 = suite["results"]["SET_01_NOMINAL_SMOOTH_CV"]
        s2 = suite["results"]["SET_02_HIGH_CURVATURE_AUTOHEAL"]
        s3 = suite["results"]["SET_03_BEYOND_KNOWABLE_HORIZON"]
        self.assertEqual(s1["pipeline_semaphores"]["SEM_05_OPENVINO_COMMIT_GATE"], "GREEN_COMMIT_AUTHORIZED")
        self.assertEqual(s2["pipeline_semaphores"]["SEM_03_QUANTIZATION_GATE"], "YELLOW_AUTO_HEAL_UPGRADE")
        self.assertEqual(s2["delta_s"], 1)
        self.assertEqual(s3["status"], "BEYOND_KNOWABLE_LIMIT_UNKNOWN")
        self.assertEqual(s3["pipeline_semaphores"]["SEM_01_KNOWABILITY_GATE"], "RED_BEYOND_KNOWABLE_LIMIT")
        self.assertEqual(s3["delta_s"], 0)

    def test_backfill_joule_multiplier_vulkan_3d_hex_and_amplitude_blueprint(self) -> None:
        # 1. Retroactive preset templates & anti-empty guarantee
        tpl = self.app.compositor.get_retroactive_preset_templates()
        self.assertTrue(tpl["anti_empty_guarantee"])
        self.assertTrue(tpl["retroactive_adaptation_enabled"])
        self.assertIn("cv-vulkan-3d-joule", tpl["presets"])

        # 2. Reverse knowledge-gap summarizer & backfill flow
        bf = self.app.compositor.summarize_and_backfill_knowledge_gaps()
        self.assertEqual(bf["reverse_flow_status"], "COHERENT_BACKFILL_COMPLETE")
        self.assertGreaterEqual(bf["total_facts_across_namespaces"], 14)
        self.assertEqual(len(bf["reverse_flow_summaries"]), 4)

        # 3. Multi-task Joule Governor & Inference Multiplier
        jm = self.app.compositor.evaluate_joule_governor_inference_multiplier(
            joule_budget_j=15.0,
            concurrent_tasks=4,
        )
        self.assertEqual(jm["vital_max_hp"], 6)
        self.assertGreater(jm["peak_vipj_gain_ratio"], 10.0)
        self.assertEqual(len(jm["multiplier_scaling_table"]), 5)

        # 4. Vulkan 3D Hex Grid & 128B Speculative Discovery Arena
        vk = self.app.compositor.adapt_kernels_and_vulkan_3d_hex_grid(grid_dims=(4, 4, 4))
        self.assertEqual(vk["grid_dimensions"]["total_voxels"], 64)
        self.assertGreater(vk["breakthrough_candidates_count"], 0)
        self.assertTrue(vk["top_breakthrough_voxels"][0]["hex_voxel_descriptor"].startswith("0x06DF"))

        # 5. Amplitude-State Blueprint, Combinatorial Scheme Synthesizer & Future Prompts
        bp = self.app.compositor.synthesize_amplitude_state_blueprint_and_intermodule_templates()
        self.assertEqual(bp["vital_max_hp"], 6)
        combos = bp["amplitude_state_analogy"]["combinations_evaluated"]
        self.assertIn("CONSTRUCTIVE_RESONANCE_PROMOTED", combos[0]["interference_regime"])
        self.assertIn("DESTRUCTIVE_CANCELLATION_QUARANTINED", combos[1]["interference_regime"])
        self.assertEqual(len(bp["inter_module_templates"]), 5)
        self.assertEqual(len(bp["future_runtime_horizons"]), 4)

        # 6. Visual Erosion Encoder & Decoder + Derived Repository Flows
        ec = self.app.compositor.run_visual_erosion_codec_pipeline()
        self.assertEqual(ec["vital_max_hp"], 6)
        self.assertEqual(ec["status"], "EROSION_CODEC_VERIFIED")
        self.assertTrue(ec["encoder_output"]["hex_erosion_descriptor"].startswith("0x06E1"))
        self.assertGreater(ec["decoder_output"]["erosion_error_reduction_pct"], 90.0)
        self.assertEqual(ec["decoder_output"]["delta_s"], 1)
        self.assertEqual(len(ec["derived_repository_features"]), 4)


if __name__ == "__main__":
    unittest.main()
