#!/usr/bin/env python3
"""
================================================================================
TEST SUITE: POLYMORPHIC INGRESS, VISUAL GLYPH-STROKE ML COMPILER,
            INDO-EUROPEAN PHONOLOGICAL TYPOGRAPHY & GAMESA 3D HEX ENCODER
================================================================================
Targets:
  - krystal_kernel/polymorphic_ingress_adapter.py
  - krystal_kernel/visual_glyph_stroke_ml_compiler.py
Invariants:
  - VITAL_MAX_HP == 6
  - 128B ZeroCopyContractFrame
  - 56 Frozen Theory Graph Nodes
  - Delta S == 0 on corrupted/divergent input
================================================================================
"""

import pytest

from krystal_kernel.constants import VITAL_MAX_HP, FRAME_SIZE_BYTES, ContractViolation
from krystal_kernel.ksl.llm_topology_schemas import LlmInterconnectionTopology
from krystal_kernel.polymorphic_ingress_adapter import (
    SideCarOverflowArena,
    PolymorphicIngressAdapter,
    PolymorphicApiEndpointConnector,
)
from krystal_kernel.visual_glyph_stroke_ml_compiler import (
    SPRINT_MOTTO_SK,
    TensorSemaphoreColor,
    VisualGlyphStrokeCompiler,
    SyllabicPhonemeCompositionEngine,
    IndoEuropeanPhonologicalTypographyEngine,
    PhotographicSignalMemoryRegister,
    LlmLinguisticWordProblemValidator,
    AdaptiveKerasSpeculativeCore,
    GridPrefillBootstrapEngine,
    NeurolinguisticCharsetCompressionCodec,
    SlavicGermanicEtymologicalParser,
    SpeechCarrierGlyphSignalCore,
    OpenWeightLinguisticGraphInjector,
    HexRepresentationClass,
    Gamesa3DGridVulkanHexEncoder,
)


class TestPolymorphicIngressAndGlyphStrokePipeline:

    def test_01_vital_max_hp_invariant_enforcement(self):
        """Constitutional invariant VITAL_MAX_HP == 6 must hold across all classes."""
        assert VITAL_MAX_HP == 6
        arena = SideCarOverflowArena(max_slots=8, vital_max_hp=6)
        assert arena.vital_max_hp == 6

        for cls in (
            SideCarOverflowArena,
            PolymorphicIngressAdapter,
            VisualGlyphStrokeCompiler,
            SyllabicPhonemeCompositionEngine,
            IndoEuropeanPhonologicalTypographyEngine,
            PhotographicSignalMemoryRegister,
            LlmLinguisticWordProblemValidator,
            AdaptiveKerasSpeculativeCore,
            GridPrefillBootstrapEngine,
            NeurolinguisticCharsetCompressionCodec,
            SlavicGermanicEtymologicalParser,
            SpeechCarrierGlyphSignalCore,
            OpenWeightLinguisticGraphInjector,
            Gamesa3DGridVulkanHexEncoder,
        ):
            with pytest.raises(ContractViolation):
                cls(vital_max_hp=5)

    def test_02_unexpected_data_structures_to_128b_frame(self):
        """Arbitrary/unexpected structures must normalize into exact 128B frames."""
        adapter = PolymorphicIngressAdapter(arena_max_slots=16)

        unexpected_payloads = [
            {"sensor_tree": {"nested": {"latency_ms": 2.31, "uma_pressure": 0.48}}, "tags": ["x", "y"]},
            [1.85, 0.62, "jagged_array_element", {"sub": 99}],
            b"\x4b\x53\x59\x4e\x00\x01\x02\xff",
            "unstructured_raw_telemetry_string_payload",
        ]

        for item in unexpected_payloads:
            receipt = adapter.ingest_unexpected_structure(item)
            assert receipt.vital_max_hp == 6
            assert len(receipt.frame_128b_raw) == FRAME_SIZE_BYTES == 128
            assert receipt.state_delta_committed is True
            assert len(receipt.arena_sha256_hex) == 64

    def test_03_corrupted_nan_inf_safe_fallback_delta_s_zero(self):
        """NaN or Inf in unexpected structures must trigger AUTH_DEGRADED_SAFE_FALLBACK (Delta S = 0)."""
        adapter = PolymorphicIngressAdapter()
        bad_payload = {"gpu_render_ms": float("nan"), "uma_pressure": float("inf")}
        receipt = adapter.ingest_unexpected_structure(bad_payload)

        assert receipt.vital_max_hp == 6
        assert receipt.state_delta_committed is False
        assert receipt.verdict_code == "AUTH_DEGRADED_SAFE_FALLBACK"
        assert receipt.target_profile == "deterministic_safe"

    def test_04_unexpected_inference_burst_routing_cascade(self):
        """Verify dynamic topology switching across UMA bus pressure thresholds."""
        adapter = PolymorphicIngressAdapter()

        low = adapter.route_unexpected_inference_burst("b-low", 128, uma_bus_pressure=0.35)
        assert low.selected_topology == LlmInterconnectionTopology.SPECULATIVE_SHADOW_CASCADE
        assert low.vulkan_drs_scale_factor == 1.00

        div = adapter.route_unexpected_inference_burst(
            "b-div", 128, uma_bus_pressure=0.45, speculative_divergence_detected=True
        )
        assert div.selected_topology == LlmInterconnectionTopology.ADAPTIVE_IMPEDANCE_ROUTING

        high = adapter.route_unexpected_inference_burst("b-high", 512, uma_bus_pressure=0.85)
        assert high.selected_topology == LlmInterconnectionTopology.PYRAMIDAL_TIERED_KV_STREAMING
        assert high.fallback_triggered is True

    def test_05_visual_glyph_stroke_compiler_and_semaphore(self):
        """Verify 6D stroke decomposition (písmená -> znaky -> čiary) and Tensor Semaphore."""
        compiler = VisualGlyphStrokeCompiler()
        sim_vw = compiler.compute_glyph_cosine_similarity("V", "W")
        sim_vo = compiler.compute_glyph_cosine_similarity("V", "O")
        assert sim_vw > 0.95
        assert sim_vw > sim_vo

        green_eval = compiler.evaluate_text_stroke_tensor(SPRINT_MOTTO_SK, uma_bus_pressure=0.35)
        assert green_eval["tensor_semaphore"] == TensorSemaphoreColor.GREEN_SPECULATIVE_FLOW.value
        assert green_eval["recommended_drs_scale"] == 1.00

        red_eval = compiler.evaluate_text_stroke_tensor(SPRINT_MOTTO_SK, uma_bus_pressure=0.85)
        assert red_eval["tensor_semaphore"] == TensorSemaphoreColor.RED_BACKPRESSURE_QUARANTINE.value
        assert red_eval["recommended_drs_scale"] == 0.65

    def test_06_syllabic_subsets_and_adaptive_keras_branch_pruning(self):
        """Verify syllabic subset decomposition and deductive branch pruning."""
        syl_engine = SyllabicPhonemeCompositionEngine()
        comp = syl_engine.compose_syllabic_subsets(SPRINT_MOTTO_SK, grid_cols=8, grid_rows=8)
        assert comp["vital_max_hp"] == 6
        assert comp["total_syllabic_subsets"] >= 15
        assert len(comp["spatial_composition_elements"]) > 0

        core = AdaptiveKerasSpeculativeCore()
        res = core.evaluate_and_prune_deductive_branches(
            operation_id="op-spec-01",
            text_payload=SPRINT_MOTTO_SK,
            candidate_branches=[
                {"branch_id": "valid_branch_A", "weight_confidence": 0.91, "entropy_reduction": 0.42},
                {"branch_id": "weak_branch_B", "weight_confidence": 0.20, "entropy_reduction": 0.10},
                {"branch_id": "parasite_branch_C", "weight_confidence": 0.85, "entropy_reduction": 0.0},
            ],
            uma_bus_pressure=0.40,
        )
        assert res.vital_max_hp == 6
        assert res.surviving_branches == ("valid_branch_A",)
        assert "weak_branch_B" in res.pruned_branches
        assert "parasite_branch_C" in res.pruned_branches

    def test_07_indo_european_typography_photographic_memory_and_word_problems(self):
        """Verify Indo-European phonological typography, Photographic Memory, and LLM word problems."""
        validator = LlmLinguisticWordProblemValidator()
        wp_res = validator.validate_word_problem(
            problem_id="wp-glyph-01",
            premise_text_sk="Písmená a znaky sú čiary, z ktorých je kryštál a tvar zložený.",
            cognate_parallel_text="Crystal shape and glyph stroke form the grid.",
            expected_semantic_anchor="MEANING_MORPHOLOGICAL_LINE_COMPOSITION",
        )
        assert wp_res["construct_valid"] is True
        assert wp_res["verdict"] == "VERIFIED_WITH_SCOPE"
        assert wp_res["vital_max_hp"] == 6

    def test_08_utf8_compression_slavic_germanic_etymology_and_open_weights(self):
        """Verify UTF-8/CP1250/6-bit compression, Slavic-Germanic etymology, and 56-node graph injection."""
        codec = NeurolinguisticCharsetCompressionCodec()
        enc = codec.encode_and_analyze_compression("Čeština, slovenčina a nemčina: zámok pamäte")
        assert enc["utf8_byte_length"] > enc["cp1250_byte_length"]
        assert enc["krystal_6bit_packed_bytes"] < enc["cp1250_byte_length"]
        assert enc["compression_gain_vs_utf8"] > 1.3

        parser = SlavicGermanicEtymologicalParser()
        parsed = parser.parse_etymological_collocations(
            ["zámok pamäte", "zámek paměti", "Speicherschloss", "cieľ systému", "Systemziel"]
        )
        assert "STRATUM_LOCK_CASTLE_MEMORY" in parsed["matched_etymological_strata"]
        assert "STRATUM_TARGET_GOAL_BOUNDARY" in parsed["matched_etymological_strata"]
        assert parsed["signal_continuity_core_score"] > 0.40

        injector = OpenWeightLinguisticGraphInjector()
        synth = injector.inject_weights_and_couple_reasoning("XERJ_UNIVERSAL_MORPHOLOGY_V1")
        assert synth.preserved_theory_nodes == 56
        assert synth.state_delta_committed is True
        assert synth.coupled_reasoning_equilibrium > 0.50

    def test_09_gamesa_3d_grid_vulkan_hex_classes_and_all_api_endpoints(self):
        """Verify Gamesa 3D Grid hexadecimal classes (0x10..0x80), ML dataset, and all 11 API endpoints."""
        encoder = Gamesa3DGridVulkanHexEncoder()
        ds = encoder.build_framework_ml_training_dataset()
        assert ds["hex_class_count"] == 8
        assert ds["x_feature_tensor_shape"] == (8, 8)
        assert ds["y_label_tensor_shape"] == (8, 8)
        assert ds["encoded_samples"][0]["ksyn_hex_packet"].startswith("4b53594e")

        connector = PolymorphicApiEndpointConnector()
        assert len(connector.SUPPORTED_ENDPOINTS) == 24

        r_glyph = connector.dispatch("POST", "/api/v1/ml/glyph_stroke_compile", {"text": SPRINT_MOTTO_SK})
        assert r_glyph["status_code"] == 200
        assert r_glyph["vital_max_hp"] == 6

        r_boot = connector.dispatch("POST", "/api/v1/cache/bootstrap_preset", {"grid_cols": 16, "grid_rows": 16})
        assert r_boot["status_code"] == 200
        assert r_boot["preloaded_photographic_clusters"] >= 4

        r_wp = connector.dispatch("POST", "/api/v1/linguistics/word_problem_validate", {})
        assert r_wp["status_code"] == 200
        assert r_wp["construct_valid"] is True

        r_etym = connector.dispatch("POST", "/api/v1/linguistics/etymological_parse", {})
        assert r_etym["status_code"] == 200
        assert len(r_etym["matched_etymological_strata"]) >= 1

        r_graph = connector.dispatch("POST", "/api/v1/graph/inject_open_weights", {})
        assert r_graph["status_code"] == 200
        assert r_graph["preserved_theory_nodes"] == 56

        r_hex = connector.dispatch("POST", "/api/v1/gamesa_grid/hex_dataset", {})
        assert r_hex["status_code"] == 200
        assert r_hex["hex_class_count"] == 8

        r_phi = connector.dispatch("POST", "/api/v1/typography/phi_charset", {"characters": "KRYSTAL"})
        assert r_phi["status_code"] == 200
        assert abs(r_phi["phi_constant"] - 1.61803399) < 1e-6
        assert abs(r_phi["cap_to_x_height_ratio"] - 1.618034) < 1e-4

        r_voice = connector.dispatch(
            "POST",
            "/api/v1/voice/geometric_glyph_synth",
            {"text": "A B S", "vocal_fold_f0_hz": 132.0},
        )
        assert r_voice["status_code"] == 200
        frames = r_voice["acoustic_frames"]
        assert frames[0]["voicing_category"] == "VOWEL_HARMONIC_PERIODIC"
        assert frames[0]["vocal_fold_f0_hz"] > 130.0
        assert frames[1]["voicing_category"] == "CONSONANT_VOICED_HYBRID"
        assert frames[1]["vocal_fold_f0_hz"] > 120.0
        assert frames[2]["voicing_category"] == "CONSONANT_VOICELESS_NOISE"
        assert frames[2]["vocal_fold_f0_hz"] == 0.0
        assert frames[2]["aperiodic_noise_gain"] >= 0.75

    def test_10_chromatic_hex_scales_lip_reading_autocomplete_and_reversible_provenance(self):
        """
        Verify C minor to Db/D major hex/dB scales, Audacity DSP morphing,
        lip-reading & mimic muscle tonus, onomatopoeic Python autocomplete,
        and reversible signal provenance across code structures.
        """
        connector = PolymorphicApiEndpointConnector()

        r_scale = connector.dispatch("POST", "/api/v1/acoustics/chromatic_hex_scale", {})
        assert r_scale["status_code"] == 200
        assert r_scale["vital_max_hp"] == 6
        assert "0xC0" in r_scale["c_minor_scale_hex"]
        assert "0xDA" in r_scale["db_major_scale_hex"]
        assert "0xDC" in r_scale["d_major_scale_hex"]
        assert r_scale["epistemic_triad"]["poznatelnost_cognizability"] > 0.90

        r_lip = connector.dispatch(
            "POST",
            "/api/v1/cv/lip_reading_glottal",
            {"spoken_text": "PBM AEO", "rhetorical_preset": "DEMOSTHENIC_ICTUS_STRIKE"},
        )
        assert r_lip["status_code"] == 200
        assert r_lip["vital_max_hp"] == 6
        assert r_lip["bidirectional_tone_to_charset_verified"] is True
        assert len(r_lip["viseme_glottal_frames"]) == 6

        r_whisper = connector.dispatch(
            "POST",
            "/api/v1/autocomplete/onomatopoeic_whisper",
            {"prefix_token": "šum_svist"},
        )
        assert r_whisper["status_code"] == 200
        assert r_whisper["detected_zvukomalba_class"] == "SIBILANT_FRICTION_ZVUKOMALBA"
        assert len(r_whisper["ranked_python_completions"]) >= 3

        r_prov = connector.dispatch(
            "POST",
            "/api/v1/provenance/reversible_trace",
            {"signal_payload": "Kryštál a čiary", "hex_tone_id": "0xC2"},
        )
        assert r_prov["status_code"] == 200
        assert r_prov["vital_max_hp"] == 6
        assert r_prov["reverse_provenance_audit"]["bijective_reversibility_verified"] is True
        assert (
            r_prov["chip_and_memory_telemetry"]["memory_throughput_increase"][
                "6bit_glyph_packing_bandwidth_gain_pct"
            ]
            == 33.33
        )

        r_joule = connector.dispatch("POST", "/api/v1/evolution/retrospective_joule_audit", {})
        assert r_joule["status_code"] == 200
        assert r_joule["vital_max_hp"] == 6
        assert len(r_joule["omitted_topological_bridges"]) == 4
        assert len(r_joule["module_parameter_scaling_matrix"]) == 4
        assert r_joule["joule_amplitude_boost_telemetry"]["reclaimed_joule_reserve_per_op"] > 0.65
        assert r_joule["joule_amplitude_boost_telemetry"]["unlocked_amplitude_boost_db"] > 10.0

        r_bezier = connector.dispatch(
            "POST",
            "/api/v1/perception/bezier_isa_morph",
            {"source_cluster": "zámok pamäte", "target_cluster": "Speicherschloss", "steps": 5},
        )
        assert r_bezier["status_code"] == 200
        assert r_bezier["vital_max_hp"] == 6
        assert len(r_bezier["bezier_morph_trajectory"]) == 5
        assert len(r_bezier["llm_adapter_instruction_set"]) == 8
        assert len(r_bezier["perceptual_parameter_queries"]) == 5
        mem = r_bezier["coherent_text_bezier_memory_estimates"]
        assert mem["per_cluster_bytes"]["clusters_per_128b_contract_frame"] == 6
        assert mem["scale_estimates"][0]["coherent_photographic_o1_kb"] < 1.0

        r_mcp_ncp = connector.dispatch(
            "POST",
            "/api/v1/mcp_ncp/control_plane",
            {"mcp_tool_name": "krystal_glyph_bezier_isa_morph", "uma_bus_pressure": 0.34},
        )
        assert r_mcp_ncp["status_code"] == 200
        assert r_mcp_ncp["vital_max_hp"] == 6
        assert r_mcp_ncp["mcp_registered_tools_count"] == 5
        assert r_mcp_ncp["active_ncp_silicon_node"]["max_draft_tokens"] == 28

        r_synth = connector.dispatch(
            "POST",
            "/api/v1/runtime/hex_coherent_synthesis",
            {"coherent_text": "Písmená sú symboly a znaky sú čiary.", "hex_tone_id": "0xC4"},
        )
        assert r_synth["status_code"] == 200
        assert r_synth["vital_max_hp"] == 6
        assert r_synth["synthesis_operations_count"] == 6
        assert r_synth["coherent_hex_units"][0]["hex_geometry_record"].startswith("0x4B53_")

        r_trig = connector.dispatch(
            "POST",
            "/api/v1/math/kernel_trigonometric_eval",
            {"angle_deg": 58.282525, "glottal_f0_hz": 261.63},
        )
        assert r_trig["status_code"] == 200
        assert r_trig["vital_max_hp"] == 6
        assert r_trig["pythagorean_invariant_preserved"] is True
        assert abs(r_trig["pythagorean_identity_sum"] - 1.0) <= 1e-4
        assert len(r_trig["chebyshev_harmonics_t1_to_t4"]) == 4

        r_webapp = connector.dispatch("GET", "/api/v1/webapp/unified_knowledge_kernel", {})
        assert r_webapp["status_code"] == 200
        assert r_webapp["vital_max_hp"] == 6
        assert r_webapp["webapp_backend_status"] == "UNIFIED_KNOWLEDGE_KERNEL_ONLINE"
        assert len(r_webapp["kernel_trigonometry_and_deductions"]["architectural_shift_deductions"]) == 4
        risc_cmp = r_webapp["hex_coherent_runtime"]["average_config_and_iris_xe_uma_audit"][
            "risc_arm_bytecode_matrix_benchmarks"
        ]["architecture_comparison"]
        assert "Intel_Xeon_Scalable_AMX_INT8_8Ch_DDR5_HBM2e" in risc_cmp
        xeon_cfg = risc_cmp["Intel_Xeon_Scalable_AMX_INT8_8Ch_DDR5_HBM2e"]
        assert xeon_cfg["total_matrix_cycles"] == 4
        assert xeon_cfg["parallel_128b_frames_per_amx_pass"] == 64

        from krystal_kernel.visual_glyph_stroke_ml_compiler import KernelGovernedTrigonometricHarmonicEngine
        hyst = KernelGovernedTrigonometricHarmonicEngine().evaluate_model_pi_and_b_across_12_frame_hysteresis()
        assert hyst["vital_max_hp"] == 6
        assert hyst["hysteresis_window_frames"] == 12
        assert len(hyst["frames_trace"]) == 13
        assert hyst["frames_trace"][6]["phase_pi"] == "6*pi/6"
        assert hyst["frames_trace"][6]["active_draft_tokens"] == 12
        assert hyst["frames_trace"][12]["phase_pi"] == "12*pi/6"
        assert hyst["frames_trace"][12]["active_draft_tokens"] == 28
        assert hyst["frames_trace"][12]["model_b_pinned_cpu"] is False








