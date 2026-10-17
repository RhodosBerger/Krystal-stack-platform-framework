# ==============================================================================
# KRYSTAL-STACK: UNIT TESTS PRE RAG-CONDITIONED LLM TRAINING FABRIC
# ==============================================================================
# Verifies:
#   1. Invariant assert VITAL_MAX_HP == 6
#   2. RAG Model Configuration Vector & Encoding (8D)
#   3. RAG-Conditioned FiLM & Elastic LoRA Adapter Network (rank 4 / 8 / 16)
#   4. Multi-Task LLM Architecture (Causal LM, Citation, Hazard, Abstain)
#   5. Chronological Split & Featurization (ml-best-practices)
#   6. Multi-Task Training Objective & Loss Formulation
#   7. Training Epoch with Gradient Updates & Metrics
#   8. Benchmark across RAG Configurations with 95% Bootstrapped CIs
#   9. MCP Protocol Integration (krystal-llm-train-epoch, krystal-llm-rag-benchmark)
#  10. Full MCP Tool Catalog Enumeration (18 tools total)
# ==============================================================================

import unittest
import sys
import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

RAG_TRAINER_IMPORT_ERROR = None
try:
    import numpy.random  # Probe C-extension mtrand under Windows Application Control
    from krystal_kernel.krystal_rag_llm_trainer import (
        RagRetrievalMode,
        RagModelConfiguration,
        CANONICAL_RAG_PRESETS,
        RagConditionedAdapterNetwork,
        KrystalRagConditionedLlm,
        KrystalTrainingSample,
        KrystalTrainingDatasetBuilder,
        KrystalRagTrainingFabric,
        LlmRagPerformanceBenchmark,
        GLOBAL_RAG_LLM_TRAINER
    )
except (ImportError, OSError) as e:
    RAG_TRAINER_IMPORT_ERROR = str(e)

from krystal_web_hub.hybrid_mcp_antigravity_connector import GLOBAL_HYBRID_MCP_CONNECTOR


class TestKrystalRagLlmTrainer(unittest.TestCase):
    def setUp(self):
        if RAG_TRAINER_IMPORT_ERROR is not None:
            self.skipTest(f"SKIP_ENVIRONMENT_BLOCKED: Windows WDAC / AppLocker policy restricted isolated C-extension: {RAG_TRAINER_IMPORT_ERROR}")
        self.trainer = GLOBAL_RAG_LLM_TRAINER
        self.mcp_conn = GLOBAL_HYBRID_MCP_CONNECTOR

    # ─── 1. INVARIANT TESTS ──────────────────────────────────────────────────

    def test_01_vital_max_hp_invariant(self):
        """Strictly asserts that platform-wide invariant VITAL_MAX_HP is 6."""
        self.assertEqual(VITAL_MAX_HP, 6, "Inviolable system invariant: VITAL_MAX_HP must equal 6")

    # ─── 2. RAG CONFIGURATION VECTOR ENCODING ────────────────────────────────

    def test_02_rag_configuration_vector_encoding(self):
        """Verifies 8-dimensional RAG feature vector encoding across retrieval modes."""
        hybrid_cfg = CANONICAL_RAG_PRESETS["high_fidelity_hybrid"]
        vec = hybrid_cfg.to_feature_vector()
        self.assertEqual(len(vec), 8)
        self.assertEqual(vec[0], 0.0)
        self.assertEqual(vec[1], 0.0)
        self.assertEqual(vec[2], 1.0)  # Hybrid one-hot
        self.assertAlmostEqual(vec[3], 4.0 / 8.0)  # top_k=4 -> 0.5

        # Empty fallback configuration
        empty_cfg = CANONICAL_RAG_PRESETS["degraded_offline_empty"]
        empty_vec = empty_cfg.to_feature_vector()
        self.assertEqual(empty_vec[0], 0.0)
        self.assertEqual(empty_vec[1], 0.0)
        self.assertEqual(empty_vec[2], 0.0)  # empty mode
        self.assertEqual(empty_vec[3], 0.0)  # top_k=0

    # ─── 3. ELASTIC LoRA & FiLM ADAPTER ──────────────────────────────────────

    def test_03_elastic_lora_and_film_adapter(self):
        """Verifies elastic rank scaling (r=4 for hybrid, r=16 for empty) and FiLM parameters."""
        adapter = RagConditionedAdapterNetwork(rag_dim=8, hidden_dim=64, max_rank=16, seed=42)
        dummy_x = np.ones(64, dtype=np.float32)

        # 1. High-Fidelity Hybrid (top-k >= 4) -> effective_rank = 4
        hybrid_vec = CANONICAL_RAG_PRESETS["high_fidelity_hybrid"].to_feature_vector()
        out_hybrid, meta_hybrid = adapter.forward(dummy_x, hybrid_vec)
        self.assertEqual(meta_hybrid["effective_rank"], 4)
        self.assertEqual(meta_hybrid["vital_max_hp"], 6)
        self.assertEqual(len(out_hybrid), 64)

        # 2. Degraded Empty (top-k = 0) -> effective_rank = 16
        empty_vec = CANONICAL_RAG_PRESETS["degraded_offline_empty"].to_feature_vector()
        out_empty, meta_empty = adapter.forward(dummy_x, empty_vec)
        self.assertEqual(meta_empty["effective_rank"], 16)
        self.assertEqual(meta_empty["vital_max_hp"], 6)

    # ─── 4. MULTI-TASK LLM ARCHITECTURE ──────────────────────────────────────

    def test_04_multi_task_llm_architecture(self):
        """Verifies multi-task forward pass: causal LM, citation fidelity, hazard, abstain."""
        model = KrystalRagConditionedLlm(vocab_size=128, hidden_dim=64, seed=42)
        tokens = np.array([12, 45, 67, 89], dtype=np.int32)
        rag_vec = CANONICAL_RAG_PRESETS["high_fidelity_hybrid"].to_feature_vector()

        out = model.forward(tokens, rag_vec)
        self.assertIn("lm_logits", out)
        self.assertIn("cite_probs", out)
        self.assertIn("hazard_prob", out)
        self.assertIn("abstain_prob", out)
        self.assertEqual(len(out["lm_logits"]), 128)
        self.assertEqual(len(out["cite_probs"]), 4)
        self.assertTrue(0.0 <= out["hazard_prob"] <= 1.0)
        self.assertTrue(0.0 <= out["abstain_prob"] <= 1.0)
        self.assertEqual(out["vital_max_hp"], 6)

    # ─── 5. DATASET GENERATION & CHRONOLOGICAL SPLIT ─────────────────────────

    def test_05_chronological_split_best_practices(self):
        """Verifies chronological dataset splitting (70% train, 15% val, 15% test)."""
        corpus = KrystalTrainingDatasetBuilder.generate_corpus(num_samples=100, seed=42)
        train_s, val_s, test_s = KrystalTrainingDatasetBuilder.chronological_split(corpus)

        self.assertEqual(len(train_s), 70)
        self.assertEqual(len(val_s), 15)
        self.assertEqual(len(test_s), 15)

        # Monotonic timestamp assertion
        self.assertLessEqual(train_s[-1].timestamp_epoch, val_s[0].timestamp_epoch)
        self.assertLessEqual(val_s[-1].timestamp_epoch, test_s[0].timestamp_epoch)

    # ─── 6. MULTI-TASK LOSS CALCULATION ──────────────────────────────────────

    def test_06_multi_task_loss_calculation(self):
        """Verifies weighted multi-task loss breakdown."""
        fabric = KrystalRagTrainingFabric(lr=0.01)
        sample = KrystalTrainingSample(
            sample_id="s-test-01",
            tokens=[1, 2, 3],
            target_next_token=5,
            rag_config_name="high_fidelity_hybrid",
            ground_truth_citation_mask=[1.0, 0.0, 1.0, 0.0],
            ground_truth_hazard=0.0,
            ground_truth_abstain=0.0,
            domain="KERNEL_ARCHITECTURE",
            timestamp_epoch=1000.0
        )
        rag_vec = CANONICAL_RAG_PRESETS["high_fidelity_hybrid"].to_feature_vector()
        out = fabric.model.forward(np.array(sample.tokens, dtype=np.int32), rag_vec)

        total_loss, breakdown = fabric.compute_loss(out, sample)
        self.assertGreater(total_loss, 0.0)
        self.assertIn("lm_loss", breakdown)
        self.assertIn("cite_loss", breakdown)
        self.assertIn("hazard_loss", breakdown)
        self.assertIn("abstain_loss", breakdown)

    # ─── 7. TRAINING EPOCH EXECUTION ─────────────────────────────────────────

    def test_07_training_epoch_execution(self):
        """Verifies training epoch execution, weight update, and metrics generation."""
        corpus = KrystalTrainingDatasetBuilder.generate_corpus(num_samples=40, seed=42)
        train_s, val_s, _ = KrystalTrainingDatasetBuilder.chronological_split(corpus)

        fabric = KrystalRagTrainingFabric(lr=0.01)
        res = fabric.train_epoch(train_s, val_s, epoch=1)

        self.assertEqual(res["epoch"], 1)
        self.assertIn("train_loss", res)
        self.assertIn("val_loss", res)
        self.assertIn("val_perplexity", res)
        self.assertIn("val_citation_f1", res)
        self.assertTrue(res["invariant_verified"])
        self.assertEqual(res["vital_max_hp"], 6)

    # ─── 8. BENCHMARK ACROSS RAG CONFIGURATIONS ──────────────────────────────

    def test_08_benchmark_across_rag_configurations(self):
        """Verifies comparative evaluation across all 4 RAG presets with bootstrapped 95% CIs."""
        corpus = KrystalTrainingDatasetBuilder.generate_corpus(num_samples=30, seed=42)
        _, _, test_s = KrystalTrainingDatasetBuilder.chronological_split(corpus)

        model = KrystalRagConditionedLlm(vocab_size=128, hidden_dim=64, seed=42)
        bench = LlmRagPerformanceBenchmark.evaluate_model_across_rag_configurations(model, test_s)

        self.assertIn("comparison_report", bench)
        rep = bench["comparison_report"]
        self.assertIn("high_fidelity_hybrid", rep)
        self.assertIn("low_latency_dense", rep)
        self.assertIn("hermetic_bm25", rep)
        self.assertIn("degraded_offline_empty", rep)

        # High fidelity hybrid should have rank 4
        self.assertEqual(rep["high_fidelity_hybrid"]["effective_rank"], 4)
        # Degraded offline should have rank 16
        self.assertEqual(rep["degraded_offline_empty"]["effective_rank"], 16)
        # Verify 95% CI tuple
        ci = rep["high_fidelity_hybrid"]["ci_95_perplexity"]
        self.assertEqual(len(ci), 2)
        self.assertLessEqual(ci[0], ci[1])
        self.assertEqual(bench["vital_max_hp"], 6)

    # ─── 9. MCP PROTOCOL TOOL INVOCATION ─────────────────────────────────────

    def test_09_mcp_krystal_llm_train_epoch_tool(self):
        """Verifies krystal-llm-train-epoch invocation through the Hybrid MCP Connector."""
        mcp_res = self.mcp_conn.execute_tool("krystal-llm-train-epoch", {
            "samples_count": 20,
            "epoch": 1,
            "learning_rate": 0.005
        })
        self.assertEqual(mcp_res["epoch"], 1)
        self.assertIn("train_loss", mcp_res)
        self.assertEqual(mcp_res["vital_max_hp"], 6)

    def test_10_mcp_krystal_llm_rag_benchmark_tool(self):
        """Verifies krystal-llm-rag-benchmark invocation through the Hybrid MCP Connector."""
        mcp_res = self.mcp_conn.execute_tool("krystal-llm-rag-benchmark", {
            "test_samples_count": 10
        })
        self.assertIn("comparison_report", mcp_res)
        self.assertEqual(mcp_res["vital_max_hp"], 6)

    # ─── 10. TOOL CATALOG ENUMERATION ────────────────────────────────────────

    def test_11_mcp_tool_catalog_enumeration(self):
        """Verifies that all 18 tools are registered and enumerate properly."""
        tools = self.mcp_conn.mcp_tool_schemas
        tool_names = [t["name"] for t in tools]
        self.assertGreaterEqual(len(tools), 18)
        self.assertIn("krystal-llm-train-epoch", tool_names)
        self.assertIn("krystal-llm-rag-benchmark", tool_names)


if __name__ == "__main__":
    unittest.main()
