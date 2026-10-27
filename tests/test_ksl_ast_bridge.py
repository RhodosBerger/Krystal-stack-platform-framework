#!/usr/bin/env python3
"""
================================================================================
UNIT TESTS: KSL OPERATIONAL AST LOWERING BRIDGE
================================================================================
Module:     tests/test_ksl_ast_bridge.py
Pipeline:   OperationalAstProposal -> KSL Constraint IR -> MicroOp DAG -> PlannedMicroOpBatch -> Chained Receipt
Author:     Antigravity Cognitive Invariants Group (2026)
Invariants: VITAL_MAX_HP == 6 | Proposal != Authority | Delta S == 0 on breach
================================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_promptbook import (
    SovereignPromptbookRegistry,
    FalsificationOracleContract,
)
from krystal_kernel.ksl.ksl_ast_bridge import (
    KslOperationalAstBridge,
    KslOptimizationObjective,
    MicroOpDAG,
    GLOBAL_KSL_OPERATIONAL_AST_BRIDGE,
)
from krystal_kernel.sovereign_uma_inspector_governor import (
    MicroOperation,
    MicroOpKind,
    CapabilityScope,
    SubsystemNodeId,
    MicroOperationAllocationBot,
    GLOBAL_MICRO_OP_BOT,
)


class TestKslOperationalAstBridge(unittest.TestCase):
    """Verifies end-to-end lowering from Operational AST to KSL IR and MicroOp DAG."""

    def setUp(self) -> None:
        self.registry = SovereignPromptbookRegistry(vital_max_hp=VITAL_MAX_HP)
        self.bridge = GLOBAL_KSL_OPERATIONAL_AST_BRIDGE
        self.bot = GLOBAL_MICRO_OP_BOT

    def test_lower_proposal_to_ksl_ir(self) -> None:
        """OperationalAstProposal correctly lowers to 5-layer KSL IR with constraint gates."""
        slots = {
            "goal_description": "Bridge Test Goal",
            "domain_name": "COMPILER_TEST",
            "current_state_json": "{}"
        }
        rendered = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)
        proposal = self.registry.compile_to_operational_ast(
            rendered_prompt=rendered,
            candidate_set=["DMA_STAGING_FLUSH", "AVX512_DESCRIPTOR", "RECONCILE_RECEIPT"],
            resource_budget={"Q_time_ns": 2_000_000.0, "Q_memory_bytes": 2048.0, "Q_energy_uj": 100.0}
        )

        ir = self.bridge.lower_proposal_to_ksl_ir(proposal, KslOptimizationObjective.THROUGHPUT)

        # 1. Constraints in KIR-C
        c_names = [c.name for c in ir.kir_c_constraints]
        self.assertIn("VITAL_MAX_HP == 6", c_names)
        self.assertIn("IS_ACYCLIC_DAG", c_names)
        self.assertTrue(any(name.startswith("INVARIANT_SET_HASH_") for name in c_names))
        self.assertTrue(any(name.startswith("ORACLE_") for name in c_names))

        # 2. Placement in KIR-H
        self.assertEqual(len(ir.kir_h_placements), 1)
        placement = ir.kir_h_placements[0]
        self.assertEqual(placement.selected_device, "KSAC_K")
        self.assertEqual(placement.latency_budget_ms, 2.0)
        self.assertTrue(placement.is_zeroalloc)

        # 3. Transitions in KIR-T
        self.assertEqual(len(ir.kir_t_transitions), 1)
        transition = ir.kir_t_transitions[0]
        self.assertEqual(len(transition.candidates), 3)
        self.assertIn("THROUGHPUT", transition.predict_objectives)

        # 4. Instructions in KIR-S & KIR-M
        s_ops = [inst.op for inst in ir.kir_s_instructions]
        m_opcodes = [inst.opcode for inst in ir.kir_m_machine_code]
        self.assertIn("ASSERT_INVARIANT", s_ops)
        self.assertIn("COMMIT_GATED", s_ops)
        self.assertIn("GATE_AND_COMMIT", m_opcodes)
        self.assertIn("SEAL_RECEIPT_HMAC", m_opcodes)

    def test_lower_ksl_ir_to_micro_op_dag(self) -> None:
        """KSL IR lowers to an acyclic MicroOpDAG with calculated critical path."""
        slots = {
            "goal_description": "DAG Lowering Test",
            "domain_name": "DAG_TEST",
            "current_state_json": "{}"
        }
        rendered = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)
        proposal = self.registry.compile_to_operational_ast(
            rendered_prompt=rendered,
            candidate_set=["DMA_STAGING_FLUSH", "AVX512_DESCRIPTOR", "RECONCILE_RECEIPT"],
            resource_budget={"Q_time_ns": 3_000_000.0, "Q_memory_bytes": 3072.0, "Q_energy_uj": 120.0}
        )

        ir, dag = self.bridge.compile_proposal_to_micro_op_dag(proposal)

        self.assertTrue(dag.is_acyclic)
        self.assertEqual(len(dag.operations), 3)
        self.assertEqual(len(dag.topological_order), 3)
        self.assertGreater(dag.total_estimated_cycle_ns, 0)
        self.assertGreater(dag.total_estimated_energy_uj, 0)
        self.assertGreater(dag.critical_path_latency_ns, 0)

        # Topological order check: each operation finishes after its dependencies
        topo_ops = dag.get_topological_operations()
        self.assertEqual(len(topo_ops), 3)

    def test_dag_cycle_detection_enforces_is_acyclic_invariant(self) -> None:
        """Circular dependency in MicroOpDAG must raise ContractViolation (IS_ACYCLIC_DAG)."""
        op1 = MicroOperation(
            micro_op_id="OP_A",
            kind=MicroOpKind.DISPATCH_DMA_SLOT,
            target_subsystem=SubsystemNodeId.WSL_BRIDGE,
            cycle_cost_ns=1000,
            energy_cost_uj=1.0,
            uma_footprint_bytes=128,
            depends_on=["OP_B"]  # Circular: A depends on B
        )
        op2 = MicroOperation(
            micro_op_id="OP_B",
            kind=MicroOpKind.EXECUTE_ALU_KERNEL,
            target_subsystem=SubsystemNodeId.CHRONO_SOC,
            cycle_cost_ns=1000,
            energy_cost_uj=1.0,
            uma_footprint_bytes=128,
            depends_on=["OP_A"]  # Circular: B depends on A
        )
        oracle = FalsificationOracleContract(
            predicate_id="IS_ACYCLIC_DAG",
            expected_result=True,
            evaluator_rule="Reject if cycle detected"
        )
        with self.assertRaises(ContractViolation) as cm:
            MicroOpDAG(
                dag_id="TEST-CYCLIC-DAG",
                operations={"OP_A": op1, "OP_B": op2},
                edges={"OP_A": ["OP_B"], "OP_B": ["OP_A"]},
                falsification_oracle=oracle
            )
        self.assertIn("IS_ACYCLIC_DAG invariant breach", str(cm.exception))

    def test_pack_with_knapsack_produces_zero_planned_overrun(self) -> None:
        """Knapsack admission over MicroOpDAG guarantees PlannedOverrunRate == 0.0 and RPE metrics."""
        slots = {
            "goal_description": "Knapsack Pack Test",
            "domain_name": "KNAPSACK_TEST",
            "current_state_json": "{}"
        }
        rendered = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)
        proposal = self.registry.compile_to_operational_ast(
            rendered_prompt=rendered,
            candidate_set=["DMA_STAGING_FLUSH", "AVX512_DESCRIPTOR", "RECONCILE_RECEIPT"],
            resource_budget={"Q_time_ns": 5_000_000.0, "Q_memory_bytes": 4096.0, "Q_energy_uj": 150.0}
        )

        _, dag = self.bridge.compile_proposal_to_micro_op_dag(proposal)
        batch = dag.pack_with_knapsack(bot=self.bot)

        self.assertEqual(batch.planned_overrun_rate, 0.0)
        self.assertTrue(batch.is_quota_respected)
        self.assertEqual(len(batch.operations), 3)

        # RPE vector
        rpe_t, rpe_e, rpe_m = batch.rpe_vector
        self.assertGreater(rpe_t, 0.0)
        self.assertLessEqual(rpe_t, 1.0)
        self.assertGreater(batch.resource_packing_balance, 0.0)

    def test_execute_end_to_end_batch_with_receipt(self) -> None:
        """Full pipeline: Proposal -> KSL IR -> MicroOp DAG -> Knapsack -> Chained Receipt."""
        slots = {
            "goal_description": "End to End Execution",
            "domain_name": "E2E_DOMAIN",
            "current_state_json": "{}"
        }
        rendered = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)
        proposal = self.registry.compile_to_operational_ast(
            rendered_prompt=rendered,
            candidate_set=["DMA_STAGING_FLUSH", "AVX512_DESCRIPTOR", "RECONCILE_RECEIPT"],
            resource_budget={"Q_time_ns": 4_000_000.0, "Q_memory_bytes": 4096.0, "Q_energy_uj": 120.0}
        )

        batch, receipt = self.bridge.execute_end_to_end_batch(
            proposal=proposal,
            rendered_prompt=rendered,
            registry=self.registry,
            planner=self.bot
        )

        self.assertEqual(batch.planned_overrun_rate, 0.0)
        self.assertEqual(receipt.recipe_id, rendered.recipe_id)
        self.assertEqual(receipt.execution_status, "COMMITTED")
        self.assertTrue(receipt.invariants_verified)
        self.assertEqual(len(receipt.hmac_receipt_signature), 64)
        self.assertEqual(receipt.sequence_number, 1)

    def test_falsification_oracle_failure_aborts_execution(self) -> None:
        """Triggered falsification oracle strictly aborts execution with ContractViolation (Delta S == 0)."""
        slots = {
            "goal_description": "Oracle Abort Test",
            "domain_name": "ORACLE_TEST",
            "current_state_json": "{}"
        }
        rendered = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)

        # Oracle that unconditionally fails (expected True, but returns False)
        failing_oracle = FalsificationOracleContract(
            predicate_id="MOCK_FAILURE_PREDICATE",
            expected_result=True,
            evaluator_rule="Must fail on assertion test",
            evaluator_predicate=lambda ctx: False  # Returns False -> triggers falsification
        )

        proposal = self.registry.compile_to_operational_ast(
            rendered_prompt=rendered,
            candidate_set=["DMA_OP_1"],
            falsification_oracle=failing_oracle
        )

        with self.assertRaises(ContractViolation) as cm:
            self.bridge.execute_end_to_end_batch(
                proposal=proposal,
                rendered_prompt=rendered,
                registry=self.registry,
                planner=self.bot
            )
        self.assertIn("Falsification Oracle triggered in MicroOp DAG", str(cm.exception))


    def test_pareto_front_pruning(self) -> None:
        """Verifies that filter_pareto_front prunes dominated policies S0 and S2, returning {S1, K}."""
        from krystal_kernel.ksl.ksl_ast_bridge import GLOBAL_PARETO_POLICY_SELECTOR
        canonical_cands = GLOBAL_PARETO_POLICY_SELECTOR.get_canonical_benchmark_policies()
        self.assertEqual(len(canonical_cands), 4)

        frontier = GLOBAL_PARETO_POLICY_SELECTOR.filter_pareto_front(canonical_cands)
        frontier_ids = [p.policy_id for p in frontier]

        # S0 is dominated by all; S2 is strictly dominated by K
        self.assertEqual(sorted(frontier_ids), ["K", "S1"])
        self.assertNotIn("S0", frontier_ids)
        self.assertNotIn("S2", frontier_ids)

    def test_pareto_selector_objectives(self) -> None:
        """Verifies multi-objective selection across THROUGHPUT, ENERGY, LATENCY, and BALANCED."""
        from krystal_kernel.ksl.ksl_ast_bridge import (
            GLOBAL_PARETO_POLICY_SELECTOR,
            OptimizationStrategy,
            MultiObjectiveConstraint,
        )
        selector = GLOBAL_PARETO_POLICY_SELECTOR

        # 1. Throughput maximization -> K
        p_thru, audit_thru = selector.select_policy(objective=KslOptimizationObjective.THROUGHPUT)
        self.assertEqual(p_thru.policy_id, "K")
        self.assertEqual(audit_thru["selected_policy"], "K")

        # 2. Energy efficiency maximization -> S1
        p_energy, audit_energy = selector.select_policy(objective=KslOptimizationObjective.ENERGY)
        self.assertEqual(p_energy.policy_id, "S1")
        self.assertEqual(audit_energy["selected_policy"], "S1")

        # 3. Latency minimization -> K
        p_lat, audit_lat = selector.select_policy(objective=KslOptimizationObjective.LATENCY)
        self.assertEqual(p_lat.policy_id, "K")
        self.assertEqual(audit_lat["selected_policy"], "K")

        # 4. Balanced with Epsilon-Constraint: requires high energy efficiency -> S1
        c_energy_priority = MultiObjectiveConstraint(min_useful_work_per_joule=16_000_000.0)
        p_bal_e, _ = selector.select_policy(
            objective=KslOptimizationObjective.BALANCED,
            strategy=OptimizationStrategy.EPSILON_CONSTRAINT,
            constraints=c_energy_priority
        )
        self.assertEqual(p_bal_e.policy_id, "S1")

        # 5. Balanced with Epsilon-Constraint: requires low latency -> K
        c_lat_priority = MultiObjectiveConstraint(max_latency_p95_ns=50_000.0)
        p_bal_l, _ = selector.select_policy(
            objective=KslOptimizationObjective.BALANCED,
            strategy=OptimizationStrategy.EPSILON_CONSTRAINT,
            constraints=c_lat_priority
        )
        self.assertEqual(p_bal_l.policy_id, "K")

    def test_ksl_bridge_lowering_with_pareto_tagging(self) -> None:
        """Verifies that lower_proposal_to_ksl_ir tags KIR-H and KIR-C with the selected Pareto policy."""
        slots = {
            "goal_description": "Pareto Tagging Test",
            "domain_name": "PARETO_TEST",
            "current_state_json": "{}"
        }
        rendered = self.registry.render_prompt("RECIPE-01-OPERATIONS-DECOMPOSER", slots)
        proposal = self.registry.compile_to_operational_ast(
            rendered_prompt=rendered,
            candidate_set=["DMA_STAGING", "ALU_KERNEL"]
        )

        # Lower under ENERGY objective -> should select S1
        ir_s1 = self.bridge.lower_proposal_to_ksl_ir(proposal, KslOptimizationObjective.ENERGY)
        c_names_s1 = [c.name for c in ir_s1.kir_c_constraints]
        self.assertIn("PARETO_POLICY_S1", c_names_s1)
        self.assertEqual(ir_s1.kir_h_placements[0].selected_device, "KSAC_S1")

        # Lower under THROUGHPUT objective -> should select K
        ir_k = self.bridge.lower_proposal_to_ksl_ir(proposal, KslOptimizationObjective.THROUGHPUT)
        c_names_k = [c.name for c in ir_k.kir_c_constraints]
        self.assertIn("PARETO_POLICY_K", c_names_k)
        self.assertEqual(ir_k.kir_h_placements[0].selected_device, "KSAC_K")


if __name__ == "__main__":
    unittest.main()

