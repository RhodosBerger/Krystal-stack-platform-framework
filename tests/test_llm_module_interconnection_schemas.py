#!/usr/bin/env python3
"""
================================================================================
KRYSTAL-STACK: LLM MODULE INTERCONNECTION SCHEMAS & POTENTIAL TEST SUITE
================================================================================
Module: tests/test_llm_module_interconnection_schemas.py
Milestone: KSL-LLM-TOPOLOGY-SCHEMAS-01
Constitutional Invariant: VITAL_MAX_HP == 6
Normative Reference: KSL_V02_OPERATIONAL_TRANSITION_ALGEBRA_SPEC.md, INFERENCE_WORKLOAD.md

Tests:
  1. Constitutional invariant VITAL_MAX_HP == 6 enforcement across nodes, edges, schemas, and registry.
  2. Registration and integrity of all 7 canonical LLM module interconnection schemas.
  3. Strict DAG acyclicity validation (Kahn's algorithm) and rejection of cyclic topologies.
  4. Workload potential evaluation (throughput, TTFT, TPOT, energy, UMA contention, context headroom).
  5. Multi-criteria Pareto ranking and dominance of decoupled/speculative topologies over S0 baseline.
  6. Mermaid diagram serialization with zero-copy annotations.
  7. Long-context scaling potential in pyramidal tiered KV streaming.
================================================================================
"""

import unittest
from typing import Dict, List

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.ksl.llm_topology_schemas import (
    LlmWorkloadStage,
    TargetDeviceBlock,
    InterconnectionChannelType,
    LlmInterconnectionTopology,
    LlmModuleNode,
    LlmModuleConnectionEdge,
    LlmWorkloadCharacteristics,
    LlmTopologyPotentialEstimate,
    LlmInterconnectionSchema,
    LlmTopologyRegistry,
    GLOBAL_LLM_TOPOLOGY_REGISTRY,
)


class TestLlmModuleInterconnectionSchemas(unittest.TestCase):
    """Formal unit verification of LLM module interconnection schemas."""

    def setUp(self) -> None:
        self.registry = GLOBAL_LLM_TOPOLOGY_REGISTRY

    def test_01_constitutional_invariant_enforcement(self) -> None:
        """Constitutional invariant VITAL_MAX_HP == 6 must be strictly enforced."""
        self.assertEqual(VITAL_MAX_HP, 6)
        self.assertEqual(self.registry.vital_max_hp, 6)

        # Breach in LlmModuleNode
        with self.assertRaises(ContractViolation):
            LlmModuleNode(
                node_id="bad_node",
                label="Bad Node",
                stage=LlmWorkloadStage.PROMPT_PREPROCESSING,
                target_device=TargetDeviceBlock.CPU_WILLOW_COVE_AVX512,
                memory_footprint_mb=10.0,
                peak_compute_gflops=10.0,
                bandwidth_demand_gbps=1.0,
                base_latency_us=50.0,
                vital_max_hp=5  # Breach
            )

        # Breach in LlmModuleConnectionEdge
        with self.assertRaises(ContractViolation):
            LlmModuleConnectionEdge(
                source_node_id="n1",
                target_node_id="n2",
                channel_type=InterconnectionChannelType.ZERO_COPY_UMA_RING,
                transfer_size_bytes=128,
                bandwidth_limit_gbps=45.0,
                latency_penalty_us=0.15,
                is_zero_copy=True,
                vital_max_hp=7  # Breach
            )

        # Breach in LlmTopologyRegistry
        with self.assertRaises(ContractViolation):
            LlmTopologyRegistry(vital_max_hp=4)

    def test_02_all_canonical_schemas_registered(self) -> None:
        """All 7 canonical topologies must be registered and well-formed."""
        schemas = self.registry.list_all_schemas()
        self.assertEqual(len(schemas), 7)

        expected_topologies = {
            LlmInterconnectionTopology.SEQUENTIAL_MONOLITHIC_S0,
            LlmInterconnectionTopology.DISAGGREGATED_PREFILL_DECODE,
            LlmInterconnectionTopology.SPECULATIVE_SHADOW_CASCADE,
            LlmInterconnectionTopology.PYRAMIDAL_TIERED_KV_STREAMING,
            LlmInterconnectionTopology.EPISTEMIC_ENSEMBLE_ARBITER,
            LlmInterconnectionTopology.HETEROGENEOUS_ASYMMETRIC_PIPELINE,
            LlmInterconnectionTopology.ADAPTIVE_IMPEDANCE_ROUTING,
        }
        registered_topologies = {s.topology for s in schemas}
        self.assertEqual(expected_topologies, registered_topologies)

        for s in schemas:
            self.assertTrue(s.is_acyclic)
            self.assertGreater(len(s.nodes), 0)
            self.assertGreater(len(s.edges), 0)
            self.assertEqual(len(s.topological_order), len(s.nodes))

    def test_03_dag_acyclicity_and_cycle_rejection(self) -> None:
        """Kahn's algorithm must validate DAGs and raise ContractViolation on cycles."""
        # Create a valid DAG
        n_a = LlmModuleNode("a", "A", LlmWorkloadStage.PROMPT_PREPROCESSING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 1.0, 1.0, 1.0, 10.0)
        n_b = LlmModuleNode("b", "B", LlmWorkloadStage.ATTENTION_CORE, TargetDeviceBlock.GPU_IRIS_XE_DP4A, 1.0, 1.0, 1.0, 10.0)
        e_ab = LlmModuleConnectionEdge("a", "b", InterconnectionChannelType.ZERO_COPY_UMA_RING, 128, 45.0, 0.1, True)

        valid_schema = LlmInterconnectionSchema(
            topology=LlmInterconnectionTopology.DISAGGREGATED_PREFILL_DECODE,
            name="Valid Test DAG",
            description="Testing acyclicity",
            nodes={"a": n_a, "b": n_b},
            edges=[e_ab]
        )
        self.assertTrue(valid_schema.is_acyclic)
        self.assertEqual(valid_schema.topological_order, ["a", "b"])

        # Create a cycle: a -> b -> a
        e_ba = LlmModuleConnectionEdge("b", "a", InterconnectionChannelType.ZERO_COPY_UMA_RING, 128, 45.0, 0.1, True)
        with self.assertRaises(ContractViolation) as ctx:
            LlmInterconnectionSchema(
                topology=LlmInterconnectionTopology.DISAGGREGATED_PREFILL_DECODE,
                name="Cyclic Schema",
                description="Testing cycle rejection",
                nodes={"a": n_a, "b": n_b},
                edges=[e_ab, e_ba]
            )
        self.assertIn("cycle detected", str(ctx.exception).lower())

    def test_04_workload_potential_evaluation(self) -> None:
        """Potential evaluation must return quantified metrics across all dimensions."""
        wl = LlmWorkloadCharacteristics(
            prompt_tokens=256,
            generation_tokens=128,
            model_params_billion=8.0,
            quantization_bits=8
        )
        estimates = self.registry.compare_all_topologies(workload=wl)
        self.assertEqual(len(estimates), 7)

        for est in estimates:
            self.assertGreater(est.effective_throughput_tokens_sec, 0.0)
            self.assertGreater(est.time_to_first_token_ms, 0.0)
            self.assertGreater(est.time_per_output_token_ms, 0.0)
            self.assertGreater(est.energy_efficiency_tokens_per_joule, 0.0)
            self.assertGreaterEqual(est.uma_contention_index, 0.0)
            self.assertLessEqual(est.uma_contention_index, 1.0)
            self.assertGreater(est.theoretical_roofline_utilization_pct, 0.0)
            self.assertGreaterEqual(est.epistemic_safety_score, 0.0)
            self.assertLessEqual(est.epistemic_safety_score, 1.0)
            self.assertEqual(est.vital_max_hp, 6)

    def test_05_pareto_dominance_over_baseline_s0(self) -> None:
        """Advanced decoupled and speculative topologies must Pareto-dominate S0 baseline."""
        wl = LlmWorkloadCharacteristics(prompt_tokens=128, generation_tokens=64)
        s0_schema = self.registry.get_schema(LlmInterconnectionTopology.SEQUENTIAL_MONOLITHIC_S0)
        spec_schema = self.registry.get_schema(LlmInterconnectionTopology.SPECULATIVE_SHADOW_CASCADE)
        disagg_schema = self.registry.get_schema(LlmInterconnectionTopology.DISAGGREGATED_PREFILL_DECODE)

        s0_est = s0_schema.evaluate_potential(wl)
        spec_est = spec_schema.evaluate_potential(wl)
        disagg_est = disagg_schema.evaluate_potential(wl)

        # Speculative Shadow Cascade must have significantly higher throughput and lower TTFT than S0
        self.assertGreater(spec_est.effective_throughput_tokens_sec, s0_est.effective_throughput_tokens_sec * 2.5)
        self.assertLess(spec_est.time_to_first_token_ms, s0_est.time_to_first_token_ms)
        self.assertLess(spec_est.uma_contention_index, s0_est.uma_contention_index)

        # Disaggregated Prefill-Decode must have lower TTFT and higher throughput than S0
        self.assertGreater(disagg_est.effective_throughput_tokens_sec, s0_est.effective_throughput_tokens_sec * 2.0)
        self.assertLess(disagg_est.time_to_first_token_ms, s0_est.time_to_first_token_ms)

        # Pareto frontier filtering
        frontier = self.registry.filter_pareto_optimal_topologies(wl, min_safety_score=0.80)
        frontier_topologies = {e.topology for e in frontier}
        # S0 has safety_score 0.70 < 0.80, so it must not be in the high-safety frontier
        self.assertNotIn(LlmInterconnectionTopology.SEQUENTIAL_MONOLITHIC_S0, frontier_topologies)

    def test_06_pyramidal_kv_context_scaling(self) -> None:
        """Pyramidal tiered KV streaming must deliver massive context headroom."""
        pyramidal = self.registry.get_schema(LlmInterconnectionTopology.PYRAMIDAL_TIERED_KV_STREAMING)
        s0 = self.registry.get_schema(LlmInterconnectionTopology.SEQUENTIAL_MONOLITHIC_S0)

        est_pyr = pyramidal.evaluate_potential()
        est_s0 = s0.evaluate_potential()

        self.assertEqual(est_pyr.max_sustainable_context_tokens, 65536)
        self.assertEqual(est_s0.max_sustainable_context_tokens, 4096)
        self.assertGreater(est_pyr.max_sustainable_context_tokens, est_s0.max_sustainable_context_tokens * 10)

    def test_07_epistemic_ensemble_arbiter_safety(self) -> None:
        """Epistemic ensemble arbiter must deliver the highest epistemic safety score."""
        arbiter = self.registry.get_schema(LlmInterconnectionTopology.EPISTEMIC_ENSEMBLE_ARBITER)
        est = arbiter.evaluate_potential()
        self.assertAlmostEqual(est.epistemic_safety_score, 0.99, places=2)

    def test_08_mermaid_diagram_rendering(self) -> None:
        """Mermaid graph generation must output valid diagram markdown with zero-copy tags."""
        disagg = self.registry.get_schema(LlmInterconnectionTopology.DISAGGREGATED_PREFILL_DECODE)
        mermaid = disagg.to_mermaid()

        self.assertTrue(mermaid.startswith("```mermaid\ngraph LR"))
        self.assertTrue(mermaid.endswith("```"))
        self.assertIn("n_prefill", mermaid)
        self.assertIn("n_decode", mermaid)
        self.assertIn("ZERO_COPY_UMA_RING", mermaid)
        self.assertIn("[ZeroCopy]", mermaid)


if __name__ == "__main__":
    unittest.main()
