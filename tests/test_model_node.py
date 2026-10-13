#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: MODEL NODE ABSTRACTION & SPECIALIZED NODES
==============================================================================
Module: tests/test_model_node.py
Description:
    Tests for ModelNode base class, ModelNodeSpec, all 9 NodeExecutionClasses,
    cost/latency/memory estimators, load/unload lifecycle, and concrete nodes:
    - IntentClassifierNode (LOCAL_SLM)
    - CoreRagNode (RAG_NODE, wrapping SmallModelCoreRagEngine)
    - ReasoningNode (REASONER)
    - CodingSpecialistNode (CODE_MODEL)
    - VerificationNode (VERIFIER)
    - FinalSynthesisNode (FINAL_SYNTHESIS)
    - EmbeddingModelNode (EMBEDDING_MODEL)
    - ToolNode (TOOL_NODE)
    - CustomExtensionNode (CUSTOM_EXTENSION)

Constitutional Invariant:
    assert VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from typing import Dict, Any

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade.model_node import (
    NodeExecutionClass,
    ModelNode,
    ModelNodeSpec,
    IntentClassifierNode,
    CoreRagNode,
    ReasoningNode,
    CodingSpecialistNode,
    VerificationNode,
    FinalSynthesisNode,
    EmbeddingModelNode,
    ToolNode,
    CustomExtensionNode,
)


class TestModelNode(unittest.TestCase):

    def test_01_vital_max_hp_invariant(self):
        """Universal platform invariant check."""
        self.assertEqual(VITAL_MAX_HP, 6)
        spec = ModelNodeSpec(
            node_id="test-spec",
            model_id="Test-SLM",
            role="CLASSIFIER",
            backend="CPU_SIMD",
            context_limit=2048,
            memory_budget_mb=512.0
        )
        self.assertEqual(spec.vital_max_hp, 6)

    def test_02_all_nine_execution_classes(self):
        """Verify all 9 canonical NodeExecutionClasses exist and have unique values."""
        expected_classes = {
            "LOCAL_SLM", "REASONER", "CODE_MODEL", "EMBEDDING_MODEL",
            "RAG_NODE", "VERIFIER", "TOOL_NODE", "FINAL_SYNTHESIS", "CUSTOM_EXTENSION"
        }
        actual_classes = {c.value for c in NodeExecutionClass}
        self.assertEqual(expected_classes, actual_classes)

    def test_03_intent_classifier_node(self):
        """Test IntentClassifierNode classification, latency estimation and inference."""
        node = IntentClassifierNode("node-cls")
        self.assertEqual(node.spec.execution_class, NodeExecutionClass.LOCAL_SLM.value)
        self.assertTrue(node.estimate_latency(100) > 0.0)
        self.assertTrue(node.estimate_memory(100) > 0.0)
        cost_info = node.estimate_cost(100)
        self.assertIn("estimated_cost", cost_info)
        self.assertTrue(cost_info["estimated_cost"] > 0.0)

        res = node.infer({"prompt": "Napíš Python script pre triedenie poľa"})
        self.assertEqual(res.status, "SUCCESS")
        self.assertIn("CODE_GENERATION", res.output)
        self.assertEqual(res.vital_max_hp, 6)

    def test_04_core_rag_node_delegation(self):
        """Verify CoreRagNode preserves SmallModelCoreRagEngine methods and signatures."""
        node = CoreRagNode("node-rag")
        self.assertEqual(node.spec.execution_class, NodeExecutionClass.RAG_NODE.value)

        # 1. retrieve_chunks
        ret = node.retrieve_chunks("VITAL_MAX_HP", top_k=3)
        self.assertIn("chunks", ret)
        self.assertIsInstance(ret["chunks"], list)

        # 2. synthesize_query
        sq = node.synthesize_query("VITAL_MAX_HP")
        self.assertIn("synthesized_answer", sq)

        # 3. _generate_small_model_answer
        ans = node._generate_small_model_answer("test query", [])
        self.assertIsInstance(ans, str)

        # 4. get_status and get_domains
        status = node.get_status()
        self.assertIn("vital_max_hp", status)
        domains = node.get_domains()
        self.assertTrue(len(domains) > 0)

        # 5. infer
        res = node.infer({"prompt": "Vysvetli invariant pamäte"})
        self.assertEqual(res.status, "SUCCESS")
        self.assertTrue(len(res.evidence_refs) > 0)

    def test_05_reasoning_and_coding_nodes(self):
        """Verify ReasoningNode and CodingSpecialistNode inference and estimates."""
        reasoner = ReasoningNode("node-reason")
        self.assertEqual(reasoner.spec.execution_class, NodeExecutionClass.REASONER.value)
        r_res = reasoner.infer({"prompt": "Analyzuj architektúru", "rag_context": "VITAL_MAX_HP == 6"})
        self.assertIn("<think>", r_res.output)

        coder = CodingSpecialistNode("node-code")
        self.assertEqual(coder.spec.execution_class, NodeExecutionClass.CODE_MODEL.value)
        c_res = coder.infer({"prompt": "def test():"})
        self.assertIn("def execute_verified_workload", c_res.output)

    def test_06_verification_and_synthesis_nodes(self):
        """Verify VerificationNode and FinalSynthesisNode."""
        verifier = VerificationNode("node-verif")
        self.assertEqual(verifier.spec.execution_class, NodeExecutionClass.VERIFIER.value)
        v_res = verifier.infer({"prompt": "Kód s assert VITAL_MAX_HP == 6", "reasoning": "Plne vyhovuje"})
        self.assertEqual(v_res.epistemic_verdict, "EPI_VALIDATED")

        synthesizer = FinalSynthesisNode("node-synth")
        self.assertEqual(synthesizer.spec.execution_class, NodeExecutionClass.FINAL_SYNTHESIS.value)
        s_res = synthesizer.infer({"prompt": "Otázka", "reasoning": "Odpoveď", "verification": "EPI_VALIDATED"})
        self.assertEqual(s_res.status, "SUCCESS")

    def test_07_embedding_tool_and_custom_nodes(self):
        """Verify EmbeddingModelNode, ToolNode and CustomExtensionNode."""
        emb = EmbeddingModelNode("node-emb")
        self.assertEqual(emb.spec.execution_class, NodeExecutionClass.EMBEDDING_MODEL.value)
        e_res = emb.infer({"prompt": "text to embed"})
        self.assertIn("EmbeddingDim: 64", e_res.output)

        tool = ToolNode("node-tool")
        self.assertEqual(tool.spec.execution_class, NodeExecutionClass.TOOL_NODE.value)
        t_res = tool.infer({"prompt": "git status"})
        self.assertIn("Tool 'git status' executed", t_res.output)

        cust = CustomExtensionNode("node-ext", extension_name="TestNpuPlugin")
        self.assertEqual(cust.spec.execution_class, NodeExecutionClass.CUSTOM_EXTENSION.value)
        cu_res = cust.infer({"prompt": "process extension"})
        self.assertIn("CUSTOM_EXT[TestNpuPlugin]", cu_res.output)

    def test_08_load_unload_lifecycle(self):
        """Verify load/unload lifecycle management."""
        node = IntentClassifierNode("node-lifecycle")
        self.assertFalse(node.loaded)
        self.assertTrue(node.load())
        self.assertTrue(node.loaded)
        self.assertTrue(node.unload())
        self.assertFalse(node.loaded)


if __name__ == "__main__":
    unittest.main()
