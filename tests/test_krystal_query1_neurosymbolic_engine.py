#!/usr/bin/env python3
"""
================================================================================
UNIT TESTS: KRYSTAL QUERY-1 NEUROSYMBOLIC CPU ENGINE & RECURRENT HYSTERESIS
================================================================================
Module:     tests/test_krystal_query1_neurosymbolic_engine.py
Author:     Dušan Kopecký & Antigravity Cognitive Invariants Group (2026)
Invariants: VITAL_MAX_HP == 6 | Proposer != Evaluator | Delta S == 0 on breach
================================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.epistemic_enlg_compiler import EnlgVerdict
from krystal_kernel.krystal_query1_neurosymbolic_engine import (
    QueryOpcode,
    CompiledQueryProgram,
    QueryProgramInstruction,
    KrystalQuery1Engine,
    build_krystal_sovereign_fact_engine,
)
from krystal_kernel.slovak_llm_adapter import SlovakLLMAdapter, ModelSpec
from krystal_kernel.llm_inference_governor import AepInferenceGovernor, GovernorMode


class TestKrystalQuery1NeurosymbolicEngine(unittest.TestCase):
    """Verifies XERJ Query-1 style neurosymbolic compilation, O(1) fact updates, and UNKNOWN/CONFLICT gates."""

    def setUp(self) -> None:
        self.engine = build_krystal_sovereign_fact_engine()

    def test_status_endpoint_and_constitutional_invariant(self) -> None:
        """GET /_facts returns loaded state and VITAL_MAX_HP == 6."""
        status = self.engine.get_status()
        self.assertEqual(status["state"], "loaded")
        self.assertEqual(status["vital_max_hp"], VITAL_MAX_HP)
        self.assertIn("krystal_sovereign", status["namespaces"])
        self.assertGreaterEqual(status["total_facts"], 10)

    def test_direct_lookup_slovak_and_english_queries(self) -> None:
        """Prompts compile into OP_LOOKUP programs and return exact facts with SHA-256 digests."""
        r1 = self.engine.ask("krystal_sovereign", "Aké je zrýchlenie pre Vulkan Iris Xe?")
        self.assertEqual(r1.status_label, "VERIFIED_FACT")
        self.assertEqual(r1.verdict, EnlgVerdict.MEASURED_WITH_SCOPE)
        self.assertEqual(r1.answer_value, 20.35)
        self.assertEqual(r1.unit, "x")
        self.assertTrue(r1.dual_seed_consensus)
        self.assertEqual(r1.program_hash_seed_a, r1.program_hash_seed_b)
        self.assertEqual(len(r1.supporting_digests[0]), 64)

        r2 = self.engine.ask("krystal_sovereign", "What is the saved bandwidth for Zero Copy Ring?")
        self.assertEqual(r2.status_label, "VERIFIED_FACT")
        self.assertEqual(r2.answer_value, 497.66)
        self.assertEqual(r2.unit, "MB/s")

    def test_compare_operator_across_two_subjects(self) -> None:
        """Comparison prompt compiles into OP_COMPARE and computes exact delta/ratio on CPU."""
        r = self.engine.ask("krystal_sovereign", "Porovnaj frame latency pre CPU Baseline vs Vulkan Iris Xe")
        self.assertEqual(r.status_label, "VERIFIED_FACT")
        self.assertEqual(r.compiled_program[0]["opcode"], QueryOpcode.OP_COMPARE.value)
        self.assertEqual(r.answer_value["CPU_BASELINE"], 46.99)
        self.assertEqual(r.answer_value["VULKAN_IRIS_XE"], 2.31)
        self.assertAlmostEqual(r.answer_value["ratio"], 20.342, places=2)

    def test_count_closed_set_with_subject_ids(self) -> None:
        """Counting query over a closed set with valid subject_ids returns exact count."""
        r = self.engine.ask("krystal_sovereign", "Koľko máme hardware accelerator zariadení?")
        self.assertEqual(r.status_label, "VERIFIED_FACT")
        self.assertEqual(r.compiled_program[0]["opcode"], QueryOpcode.OP_COUNT_SET.value)
        self.assertEqual(r.answer_value, 3)

    def test_count_unclosed_set_or_missing_subject_id_returns_unknown(self) -> None:
        """Strict XERJ rc.101 rule: unclosed sets or records missing subject_id return UNKNOWN."""
        self.engine.bulk_ingest(
            namespace="krystal_sovereign",
            facts=[
                {
                    "subject_id": "NODE_1",
                    "predicate": "open_sensor_reading",
                    "predicate_aliases": ["open sensor reading"],
                    "object_value": 42.0,
                    "is_closed_set": False,
                }
            ],
        )
        r = self.engine.ask("krystal_sovereign", "How many open sensor reading items exist?")
        self.assertEqual(r.status_label, "UNKNOWN")
        self.assertEqual(r.verdict, EnlgVerdict.UNKNOWN)
        self.assertIsNone(r.answer_value)

    def test_contradictory_facts_return_conflict_with_homotopic_branches(self) -> None:
        """When two stored facts disagree on (subject_id, predicate), engine returns CONFLICT (no guessing)."""
        self.engine.bulk_ingest(
            namespace="krystal_sovereign",
            facts=[
                {
                    "fact_id": "FACT-CONFLICT-B",
                    "subject_id": "TRI_PHASE_DOSING",
                    "predicate": "reaction_latency_ms",
                    "object_value": 9.95,
                    "unit": "ms",
                    "source_uri": "file:///uncalibrated_external_claim.md",
                }
            ],
            replace_existing=False,
        )
        r = self.engine.ask("krystal_sovereign", "Aká je reakčná latencia pre Tri Phase Dosing?")
        self.assertEqual(r.status_label, "CONFLICT")
        self.assertEqual(r.verdict, EnlgVerdict.CONFLICT)
        self.assertIsNone(r.answer_value)
        self.assertEqual(len(r.conflict_branches), 2)

    def test_zero_second_fact_update_without_reembedding(self) -> None:
        """Updating a fact with replace_existing=True updates immediately in O(1) without re-indexing."""
        res = self.engine.bulk_ingest(
            namespace="krystal_sovereign",
            facts=[
                {
                    "fact_id": "FACT-TRI-PHASE-LATENCY-02",
                    "subject_id": "TRI_PHASE_DOSING",
                    "predicate": "reaction_latency_ms",
                    "object_value": 0.025,
                    "unit": "ms",
                }
            ],
            replace_existing=True,
        )
        self.assertLess(res["ingestion_latency_us"], 5000.0)
        r = self.engine.ask("krystal_sovereign", "Aká je reakčná latencia pre Tri Phase Dosing?")
        self.assertEqual(r.status_label, "VERIFIED_FACT")
        self.assertEqual(r.answer_value, 0.025)

    def test_dual_seed_program_mismatch_returns_unknown(self) -> None:
        """If Seed A and Seed B compile different programs, execution is blocked and returns UNKNOWN."""
        divergent_prog_b = CompiledQueryProgram(
            program_id="DIVERGENT-B",
            seed_id=2,
            instructions=(
                QueryProgramInstruction(
                    opcode=QueryOpcode.OP_COUNT_SET,
                    namespace="krystal_sovereign",
                    target_predicate="vital_max_hp",
                ),
            ),
        )
        r = self.engine.ask(
            "krystal_sovereign",
            "Aký je ústavný invariant pre Krystal Stack?",
            override_program_seed_b=divergent_prog_b,
        )
        self.assertFalse(r.dual_seed_consensus)
        self.assertEqual(r.status_label, "UNKNOWN")
        self.assertEqual(r.verdict, EnlgVerdict.UNKNOWN)


class TestRecurrentInferenceHysteresisAndMomentum(unittest.TestCase):
    """Verifies KSPP Principle 8 12-frame dwell hysteresis and O(1) Merkle-Momentum in AepInferenceGovernor."""

    def test_12_frame_hysteresis_prevents_cpu_gpu_oscillation(self) -> None:
        adapter = SlovakLLMAdapter(ModelSpec(parameter_count_b=3.0, quantization="Q4_K_M"))
        gov = AepInferenceGovernor(
            mode=GovernorMode.ADAPTIVE_GOVERNOR,
            adapter=adapter,
            enable_hysteresis=True,
            hysteresis_dwell_frames=12,
        )

        # 1. Trigger Orange Zone (0.75) -> immediate downscale to CPU and pin
        ok, receipt, _ = gov.dispatch_inference("Špička zbernice", max_tokens=32, target_device="GPU", current_bus_pressure=0.75)
        self.assertTrue(ok)
        self.assertEqual(receipt.device, "CPU")
        self.assertTrue(gov.current_state.hysteresis_pinned_cpu)
        self.assertEqual(gov.current_state.hysteresis_dwell_counter, 0)

        # 2. Next 11 frames in Green Zone (0.20) must remain pinned to CPU (anti-oscillation dwell)
        for step in range(1, 12):
            ok, receipt, _ = gov.dispatch_inference(f"Dwell krok {step}", max_tokens=32, target_device="GPU", current_bus_pressure=0.20)
            self.assertTrue(ok)
            self.assertEqual(receipt.device, "CPU")
            self.assertTrue(gov.current_state.hysteresis_pinned_cpu)
            self.assertEqual(gov.current_state.hysteresis_dwell_counter, step)

        # 3. 12th low-pressure frame releases the CPU pin and restores nominal routing
        ok, receipt, _ = gov.dispatch_inference("Dwell krok 12 - uvoľnenie", max_tokens=32, target_device="GPU", current_bus_pressure=0.20)
        self.assertTrue(ok)
        self.assertFalse(gov.current_state.hysteresis_pinned_cpu)
        self.assertEqual(gov.current_state.hysteresis_dwell_counter, 0)
        self.assertGreater(gov.current_state.merkle_momentum, 0.0)
        self.assertEqual(len(gov.current_state.recurrent_state_digest), 64)


if __name__ == "__main__":
    unittest.main()
