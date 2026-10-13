#!/usr/bin/env python3
"""
Unit tests for Small Model Core RAG Engine & UI System.
"""

import unittest
from krystal_kernel.small_model_core_rag_engine import (
    SmallModelCoreRagEngine,
    GLOBAL_SMALL_MODEL_RAG_ENGINE,
    CoreKnowledgeDomain,
    VITAL_MAX_HP
)


class TestSmallModelCoreRag(unittest.TestCase):
    def setUp(self):
        self.engine = SmallModelCoreRagEngine()

    def test_canonical_knowledge_seed(self):
        self.assertGreaterEqual(len(self.engine.knowledge_base), 8)
        self.assertIn("ck_kernel_invariant_01", self.engine.knowledge_base)
        self.assertIn("ck_strand_npu_memory_02", self.engine.knowledge_base)
        self.assertIn("ck_npu_action_puzzle_03", self.engine.knowledge_base)
        self.assertIn("ck_gpu_noise_cosine_04", self.engine.knowledge_base)

    def test_small_model_specification(self):
        spec = self.engine.model_spec
        self.assertEqual(spec.model_id, "Krystal-SLM-1.5B-CoreUI")
        self.assertEqual(spec.vram_footprint_mb, 896.0)
        self.assertEqual(spec.vital_max_hp, 6)

    def test_retrieval_ranking(self):
        # Query for strand processors
        hits = self.engine.retrieve_chunks("strandove procesory a pamät 0x7fff", top_k=3)
        self.assertGreater(len(hits), 0)
        top_chunk, score = hits[0]
        self.assertIn(top_chunk.domain, (CoreKnowledgeDomain.STRAND_NPU_MEMORY, CoreKnowledgeDomain.KERNEL_ARCHITECTURE))
        self.assertGreater(score, 0.1)

    def test_domain_filtering(self):
        hits = self.engine.retrieve_chunks(
            query="vulkan mrt render",
            top_k=2,
            domain_filter=CoreKnowledgeDomain.POLYGRAPHIC_DTP_3D
        )
        for chunk, _ in hits:
            self.assertEqual(chunk.domain, CoreKnowledgeDomain.POLYGRAPHIC_DTP_3D)

    def test_small_model_synthesis_and_grounding(self):
        res = self.engine.synthesize_query("ako funguje opencl opengl gpu dekoder s cosine")
        self.assertIn("Krystal-SLM-1.5B", res.synthesized_answer)
        self.assertGreater(len(res.citations), 0)
        self.assertGreater(res.factual_grounding_score, 0.0)
        self.assertEqual(res.vital_max_hp, 6)
        self.assertGreater(len(res.suggested_ui_actions), 0)

    def test_add_custom_core_knowledge(self):
        new_chunk = self.engine.add_custom_chunk(
            domain=CoreKnowledgeDomain.EPISTEMIC_AUTHORITY,
            title="Custom Reversible Barrier Protocol",
            content="Deterministický reverzibilný protokol brány umožňuje rollback stavov v O(1) čase.",
            tags=["barrier", "reversible"]
        )
        self.assertIn(new_chunk.chunk_id, self.engine.knowledge_base)
        hits = self.engine.retrieve_chunks("deterministický reverzibilný protokol brány", top_k=1)
        self.assertEqual(hits[0][0].chunk_id, new_chunk.chunk_id)

    def test_global_singleton_status(self):
        status = GLOBAL_SMALL_MODEL_RAG_ENGINE.get_status()
        self.assertIn("model_spec", status)
        self.assertIn("indexed_chunks_count", status)
        self.assertEqual(status["vital_max_hp"], 6)

        domains = GLOBAL_SMALL_MODEL_RAG_ENGINE.get_domains()
        self.assertEqual(len(domains), len(CoreKnowledgeDomain))


if __name__ == '__main__':
    unittest.main()
