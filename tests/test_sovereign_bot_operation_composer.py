#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: SOVEREIGN BOT OPERATION COMPOSER & AXIOM AUTO-DEFINER
==============================================================================
Module: tests/test_sovereign_bot_operation_composer.py
Author: Antigravity Cognitive Invariants Group (2026)
Constitutional Invariant: VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
from typing import Dict, Any

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.sovereign_bot_scale_dimensioner import ScaleDomain, LayerSystemId
from krystal_kernel.sovereign_bot_operation_composer import (
    BotArchetype,
    OperationalAxiom,
    AxiomAutoDefiner,
    SovereignBotTemplateSystem,
    SafeOperationSourceComposer,
    GLOBAL_BOT_OPERATION_COMPOSER,
)


class TestAxiomAutoDefiner(unittest.TestCase):
    """Verifies dynamic definition of constitutional and operational axioms."""

    def setUp(self) -> None:
        self.definer = AxiomAutoDefiner(vital_max_hp=VITAL_MAX_HP)

    def test_constitutional_invariants_always_present(self) -> None:
        """Every subtask must include Axiom 0, 1, and 2."""
        axioms = self.definer.define_axioms_for_subtask(
            subtask_id="ST-01-SPEC",
            phase_name="SPECIFICATION",
            domain_name="SCIENTIFIC_EVIDENCE",
            subsystem="SOVEREIGN_KERNEL",
            latency_budget_us=10.0,
            energy_budget_j=0.0001,
        )
        self.assertGreaterEqual(len(axioms), 4)

        axiom_ids = [a.axiom_id for a in axioms]
        self.assertIn("AXIOM-00-VITAL-SURVIVAL", axiom_ids)
        self.assertIn("AXIOM-01-AUTHORITY-SEPARATION", axiom_ids)
        self.assertIn("AXIOM-02-ZERO-TOUCH-128B", axiom_ids)

        # Axiom 0 must be constitutional
        ax0 = next(a for a in axioms if a.axiom_id == "AXIOM-00-VITAL-SURVIVAL")
        self.assertTrue(ax0.is_constitutional)
        self.assertEqual(ax0.vital_max_hp, 6)

    def test_phase_specific_axioms(self) -> None:
        """Different phases must receive corresponding derived operational axioms."""
        phases_and_expected = [
            ("SPECIFICATION", "AXIOM-03-SPEC-FREEZE"),
            ("AUTHORITY_CHECK", "AXIOM-03-CAPABILITY-GATE"),
            ("INVARIANT_PRESERVATION", "AXIOM-03-SAFETY-CONTAINMENT"),
            ("EXECUTION_MINIMAL", "AXIOM-03-MINIMAL-INTERVENTION"),
            ("PHYSICAL_TELEMETRY", "AXIOM-03-PHYSICAL-GROUNDING"),
            ("RECONCILIATION_COMMIT", "AXIOM-03-TRANSACTIONAL-COMMIT"),
        ]

        for phase, expected_id in phases_and_expected:
            axioms = self.definer.define_axioms_for_subtask(
                subtask_id=f"ST-{phase}",
                phase_name=phase,
                domain_name="SYSTEM_GOVERNANCE",
                subsystem="HOST_CPU",
                latency_budget_us=50.0,
                energy_budget_j=0.001,
            )
            axiom_ids = [a.axiom_id for a in axioms]
            self.assertIn(expected_id, axiom_ids, f"Phase {phase} missing {expected_id}")

    def test_constitutional_invariant_breach(self) -> None:
        """Initializing with VITAL_MAX_HP != 6 must raise ContractViolation."""
        with self.assertRaises(ContractViolation):
            AxiomAutoDefiner(vital_max_hp=4)


class TestSovereignBotTemplateSystem(unittest.TestCase):
    """Verifies bot archetype mapping, scale domain classification, and layer IDs."""

    def setUp(self) -> None:
        self.templates = SovereignBotTemplateSystem()

    def test_archetype_mapping(self) -> None:
        """Checks phase-to-archetype mapping."""
        self.assertEqual(self.templates.select_archetype_for_phase("SPECIFICATION"), BotArchetype.PROPOSER_BOT)
        self.assertEqual(self.templates.select_archetype_for_phase("AUTHORITY_CHECK"), BotArchetype.VERIFIER_BOT)
        self.assertEqual(self.templates.select_archetype_for_phase("EXECUTION_MINIMAL"), BotArchetype.ACTUATOR_BOT)
        self.assertEqual(self.templates.select_archetype_for_phase("PHYSICAL_TELEMETRY"), BotArchetype.OBSERVER_BOT)
        self.assertEqual(self.templates.select_archetype_for_phase("RECONCILIATION_COMMIT"), BotArchetype.AUDITOR_BOT)

    def test_scale_domain_classification(self) -> None:
        """Turnaround latency determines hierarchical scaling domain."""
        self.assertEqual(self.templates.select_scale_domain_for_latency(0.04), ScaleDomain.MICRO_SILICON_NS)
        self.assertEqual(self.templates.select_scale_domain_for_latency(2.85), ScaleDomain.MESO_VBLANK_MS)
        self.assertEqual(self.templates.select_scale_domain_for_latency(150000.0), ScaleDomain.MACRO_CADENCE_MS)
        self.assertEqual(self.templates.select_scale_domain_for_latency(5000000.0), ScaleDomain.EPOCH_GOVERNANCE_S)

    def test_layer_system_mapping(self) -> None:
        """Placement targets map to the appropriate architectural layers."""
        self.assertEqual(self.templates.select_layer_system_for_subsystem("TELEMETRY_INA226"), LayerSystemId.L0_SILICON_PDN)
        self.assertEqual(self.templates.select_layer_system_for_subsystem("FPGA_MAILBOX"), LayerSystemId.L1_CHRONO_UMA)
        self.assertEqual(self.templates.select_layer_system_for_subsystem("DMA_SHM_RING"), LayerSystemId.L4_COORDINATION_FABRIC)
        self.assertEqual(self.templates.select_layer_system_for_subsystem("SOVEREIGN_KERNEL"), LayerSystemId.L5_CONSTITUTIONAL_GATE)


class TestSafeOperationSourceComposer(unittest.TestCase):
    """Verifies synthesis, sandboxing, and executable execution of compiled bot source modules."""

    def setUp(self) -> None:
        self.composer = SafeOperationSourceComposer(vital_max_hp=VITAL_MAX_HP)

    def test_compose_source_module_structure(self) -> None:
        """Synthesized code must be a valid Python module with class, axioms, and entrypoint."""
        compiled = self.composer.compose_source_for_subtask(
            subtask_id="ST-04-EXEC",
            title="Execute Minimal Action",
            phase_name="EXECUTION_MINIMAL",
            domain_name="HARDWARE_CONTROL",
            subsystem="FPGA_MAILBOX",
            latency_budget_us=2.85,
            energy_budget_j=0.0008,
            verification_oracle="action_executed == True",
            rollback_action="abort_dma_transfer()",
            action_payload="pcie_mailbox.send_frame()",
        )

        self.assertIsNotNone(compiled)
        self.assertEqual(compiled.bot_archetype, BotArchetype.ACTUATOR_BOT)
        self.assertEqual(compiled.scale_domain, ScaleDomain.MESO_VBLANK_MS)
        self.assertEqual(compiled.layer_system_id, LayerSystemId.L1_CHRONO_UMA)
        self.assertGreater(len(compiled.axioms), 0)
        self.assertTrue(bool(compiled.hmac_sha256))

        source = compiled.source_code
        self.assertIn("class SovereignSt04execActor:", source)
        self.assertIn("def run_st_04_exec_actor", source)
        self.assertIn("VITAL_MAX_HP", source)
        self.assertIn("ContractViolation", source)
        self.assertIn("assert_axioms", source)
        self.assertIn("execute_rollback", source)

    def test_compiled_module_live_execution(self) -> None:
        """The generated Python code must execute cleanly under exec() sandbox."""
        compiled = self.composer.compose_source_for_subtask(
            subtask_id="ST-01-SPEC",
            title="Freeze Hypothesis",
            phase_name="SPECIFICATION",
            domain_name="SCIENTIFIC_EVIDENCE",
            subsystem="SOVEREIGN_KERNEL",
            latency_budget_us=10.0,
            energy_budget_j=0.0001,
            verification_oracle="protocol_locked == True",
            rollback_action="reset_locks()",
            action_payload="protocol.lock()",
        )

        local_env: Dict[str, Any] = {}
        # Execute the compiled source in an isolated namespace
        exec(compiled.source_code, local_env)

        entry_fn = local_env.get(compiled.entrypoint_symbol)
        self.assertTrue(callable(entry_fn))

        # Run the actor
        receipt = entry_fn({"proposer_role": "PROPOSER", "evaluator_role": "VERIFIER", "temperature_c": 35.0})
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt.get("verdict"), "COMMIT")
        self.assertEqual(receipt.get("subtask_id"), "ST-01-SPEC")
        self.assertEqual(receipt.get("vital_max_hp"), 6)


if __name__ == "__main__":
    unittest.main()
