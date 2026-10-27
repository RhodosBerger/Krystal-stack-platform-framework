#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: SOVEREIGN OPERATIONS AUTOCOMPLETER & TASK DECOMPOSITION GENERATOR
==============================================================================
Module: tests/test_sovereign_operations_decomposer.py
Author: Antigravity Cognitive Invariants Group (2026)
Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from typing import Dict, Any

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation, CapabilityScope
from krystal_kernel.sovereign_operations_decomposer import (
    OperationDomain,
    SubtaskPhase,
    SubsystemPlacement,
    OperationSuggestion,
    SubtaskNode,
    SubtaskDAG,
    SovereignOperationsAutocompleter,
    SovereignOperationsDecomposer,
    SovereignOperationsStudio,
    GLOBAL_OPERATIONS_STUDIO,
)


class TestSovereignOperationsAutocompleter(unittest.TestCase):
    """Verifies prefix, fuzzy, and context-sensitive autocompletion."""

    def setUp(self) -> None:
        self.autocompleter = SovereignOperationsAutocompleter()

    def test_exact_prefix_matching_and_suffix(self) -> None:
        """Querying 'audit' should match 'audit_system_state' with correct suffix."""
        results = self.autocompleter.autocomplete("audit")
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertEqual(top.command, "audit_system_state")
        self.assertEqual(top.completion_suffix, "_system_state")
        self.assertEqual(top.domain, OperationDomain.SYSTEM_GOVERNANCE)
        self.assertGreater(top.score, 0.8)

    def test_hardware_control_prefix(self) -> None:
        """Querying 'disp' should match 'dispatch_microcode_batch'."""
        results = self.autocompleter.autocomplete("disp")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].command, "dispatch_microcode_batch")
        self.assertEqual(results[0].completion_suffix, "atch_microcode_batch")
        self.assertEqual(results[0].domain, OperationDomain.HARDWARE_CONTROL)

    def test_empty_query_returns_top_catalog(self) -> None:
        """An empty query should return the top canonical operations."""
        results = self.autocompleter.autocomplete("", max_results=4)
        self.assertEqual(len(results), 4)

    def test_thermal_context_boosting(self) -> None:
        """High temperature context should boost thermal management suggestions."""
        state_cool = {"temperature_c": 30.0}
        state_hot = {"temperature_c": 82.0}

        cool_res = self.autocompleter.autocomplete("thermal", current_state=state_cool)
        hot_res = self.autocompleter.autocomplete("thermal", current_state=state_hot)

        self.assertGreater(len(cool_res), 0)
        self.assertGreater(len(hot_res), 0)
        self.assertGreaterEqual(hot_res[0].score, cool_res[0].score)

    def test_uma_pressure_context_boosting(self) -> None:
        """High UMA bus pressure should boost shadow cycle reconciliation."""
        state_high_uma = {"uma_pressure": 0.88}
        results = self.autocompleter.autocomplete("shadow", current_state=state_high_uma)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].command, "reconcile_shadow_cycle")
        self.assertGreater(results[0].score, 0.9)


class TestSovereignOperationsDecomposer(unittest.TestCase):
    """Verifies 6-phase task decomposition, DAG acyclicity, and invariants."""

    def setUp(self) -> None:
        self.decomposer = SovereignOperationsDecomposer(vital_max_hp=VITAL_MAX_HP)

    def test_scientific_evidence_decomposition(self) -> None:
        """Decomposing a benchmark goal yields 6 structured subtasks."""
        dag = self.decomposer.decompose("Run M4.1 Physical Validation Benchmark")
        self.assertEqual(dag.domain, OperationDomain.SCIENTIFIC_EVIDENCE)
        self.assertEqual(len(dag.subtasks), 6)
        self.assertTrue(dag.is_acyclic)
        self.assertGreater(dag.critical_path_latency_us, 0.0)
        self.assertGreater(dag.total_energy_budget_j, 0.0)

        # Check the 6 formal phases in sequence
        phases = [t.phase for t in dag.subtasks]
        self.assertEqual(phases, [
            SubtaskPhase.SPECIFICATION,
            SubtaskPhase.AUTHORITY_CHECK,
            SubtaskPhase.INVARIANT_PRESERVATION,
            SubtaskPhase.EXECUTION_MINIMAL,
            SubtaskPhase.PHYSICAL_TELEMETRY,
            SubtaskPhase.RECONCILIATION_COMMIT,
        ])

    def test_hardware_control_decomposition(self) -> None:
        """Decomposing an FPGA microcode goal yields mailbox and timing subtasks."""
        dag = self.decomposer.decompose("Dispatch KCB-M4 FPGA microcode pipeline")
        self.assertEqual(dag.domain, OperationDomain.HARDWARE_CONTROL)
        self.assertEqual(len(dag.subtasks), 6)
        self.assertTrue(dag.is_acyclic)
        # Check subsystem placement
        placements = [t.subsystem_placement for t in dag.subtasks]
        self.assertIn(SubsystemPlacement.FPGA_MAILBOX, placements)
        self.assertIn(SubsystemPlacement.DMA_SHM_RING, placements)

    def test_kernel_compilation_decomposition(self) -> None:
        """Decomposing a KSL compile goal targets AST, type checking, and C11 emission."""
        dag = self.decomposer.decompose("Compile KSL kernel module to zero-copy ABI")
        self.assertEqual(dag.domain, OperationDomain.KERNEL_COMPILATION)
        self.assertEqual(len(dag.subtasks), 6)
        self.assertTrue(dag.is_acyclic)
        self.assertIn("ST-01-SPEC", dag.subtasks[0].subtask_id)
        self.assertEqual(dag.subtasks[0].phase, SubtaskPhase.SPECIFICATION)

    def test_system_governance_decomposition(self) -> None:
        """Generic management intent routes to system governance with state capture."""
        dag = self.decomposer.decompose("Enforce thermal headroom and audit invariants")
        self.assertEqual(dag.domain, OperationDomain.SYSTEM_GOVERNANCE)
        self.assertEqual(len(dag.subtasks), 6)
        self.assertTrue(dag.is_acyclic)

    def test_constitutional_invariant_breach_prevention(self) -> None:
        """Initializing decomposer with VITAL_MAX_HP != 6 must raise ContractViolation."""
        with self.assertRaises(ContractViolation):
            SovereignOperationsDecomposer(vital_max_hp=5)

    def test_hmac_integrity_signature(self) -> None:
        """Generated DAG must have a valid non-empty HMAC-SHA256 signature."""
        dag = self.decomposer.decompose("Execute latency probe")
        self.assertTrue(bool(dag.hmac_sha256))
        self.assertEqual(len(dag.hmac_sha256), 64)

    def test_mermaid_generation(self) -> None:
        """generate_mermaid_diagram must output valid mermaid graph lines."""
        dag = self.decomposer.decompose("Ablate PCIe DMA latency")
        mermaid = dag.generate_mermaid_diagram()
        self.assertIn("```mermaid", mermaid)
        self.assertIn("graph TD", mermaid)
        self.assertIn("ST-01-SPEC", mermaid)
        self.assertIn("ST-06-COMMIT", mermaid)


class TestSovereignOperationsStudio(unittest.TestCase):
    """Verifies unified studio interface and report rendering."""

    def setUp(self) -> None:
        self.studio = GLOBAL_OPERATIONS_STUDIO

    def test_studio_autocomplete_and_decompose_flow(self) -> None:
        """Studio provides seamless query -> suggestion -> DAG decomposition."""
        suggestions = self.studio.autocomplete("bench")
        self.assertGreater(len(suggestions), 0)
        selected_cmd = suggestions[0].command

        dag = self.studio.decompose_management_goal(selected_cmd)
        self.assertIsNotNone(dag)
        self.assertEqual(len(dag.subtasks), 6)

        report = self.studio.format_plan_report(dag)
        self.assertIn("# SOVEREIGN OPERATIONS PLAN", report)
        self.assertIn("## 1. Decomposed Subtask & Bot Archetype Table", report)
        self.assertIn("## 2. Dependency Flowchart (Mermaid)", report)
        self.assertIn("## 3. Autonomous Bot Operation Source Modules (Compiled Snippets)", report)


if __name__ == "__main__":
    unittest.main()
