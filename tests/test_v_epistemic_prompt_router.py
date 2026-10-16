"""
Integration Test Suite for Epistemic Prompt Router & Calibrated Micro-Prompt Base
=================================================================================
Milestone: V-PROMPT-ROUTER-01
Constitutional Invariant: VITAL_MAX_HP == 6

Verifies:
  1. Inviolable constitutional invariant VITAL_MAX_HP == 6 across all router structures.
  2. 6-Tuple Claim Card: Class, Source, Provenance, Scope, Pro-Evidence, Counter-Evidence.
  3. Epistemic distinctions: EVIDENCE_INSUFFICIENT != REFUTED; CONVENTIONAL != VALID.
  4. 8-Model Portfolio (M0 through M7) completeness.
  5. Anti-Averaging Conflict Resolution: preserves dispute, bans averaging, demands discriminating test.
  6. UNKNOWN_MECHANISM protocol: explicit missing telemetry, zero confabulation allowed.
  7. Non-Sycophancy & Anti-Self-Certification output formatting (Status + Uncertainty).
  8. Specialized micro-prompt dynamic routing by TaskType (strictly avoiding monolithic prompt bloat).
"""

import pytest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.epistemic_provenance_logger import EpistemicVerdict
from krystal_kernel.epistemic_prompt_router import (
    EpistemicClaimClass,
    TaskType,
    EpistemicClaimCard,
    ModelM0ToM7Spec,
    ModelPortfolioM0ToM7,
    EpistemicConflictNode,
    UnknownMechanismNode,
    MicroPrompt,
    EpistemicPromptRouter,
)


class TestVEpistemicPromptRouter:

    def test_01_vital_max_hp_invariant(self):
        """Constitutional invariant VITAL_MAX_HP == 6 must be strictly preserved."""
        assert VITAL_MAX_HP == 6

        card = EpistemicClaimCard(
            claim_id="CLAIM-01",
            statement="DP4A lowers reconstruction time by 35%",
            claim_class=EpistemicClaimClass.MEASUREMENT,
            source="profiler/dp4a_bench",
            provenance_chain=["hardware_run_01"],
            scope_boundary="Intel Iris Xe / Arc graphics",
            pro_evidence=["psnr_41db_log"],
            counter_evidence=["older_gen_fallback_required"],
            vital_max_hp=6,
        )
        assert card.vital_max_hp == 6

        with pytest.raises(ContractViolation):
            EpistemicClaimCard(
                claim_id="CLAIM-BAD",
                statement="bad",
                claim_class=EpistemicClaimClass.HYPOTHESIS,
                source="src",
                provenance_chain=["p1"],
                scope_boundary="all",
                pro_evidence=[],
                counter_evidence=[],
                vital_max_hp=5,
            )

        with pytest.raises(ContractViolation):
            EpistemicConflictNode(
                conflict_id="CONF-BAD",
                disputed_claim="bad claim",
                thesis_evidence=["t1"],
                antithesis_evidence=["a1"],
                proposed_discriminating_test="test",
                vital_max_hp=7,
            )

        with pytest.raises(ContractViolation):
            EpistemicPromptRouter(vital_max_hp=4)

    def test_02_epistemic_claim_card_6tuple_and_distinctions(self):
        """
        Validates 6-tuple completeness and categorical rules:
        - EVIDENCE_INSUFFICIENT is not CONTRADICTED / REFUTED.
        - CONVENTIONAL is not automatically VALID.
        """
        card = EpistemicClaimCard(
            claim_id="CLAIM-ROUTER-02",
            statement="Micro-slicing LLM decode to VBlank eliminates frame stutter",
            claim_class=EpistemicClaimClass.TEST_RESULT,
            source="tests/test_v_krystal_nexus_accelerator.py",
            provenance_chain=["commit_e9eae83b", "vsync_trace_run_42"],
            scope_boundary="120Hz Displays on Unified Memory Architecture",
            pro_evidence=["Zero frame drops measured across 30 consecutive frames"],
            counter_evidence=["Display scanout latency increases if VBlank window is missed"],
        )

        assert card.statement.startswith("Micro-slicing")
        assert len(card.pro_evidence) == 1
        assert len(card.counter_evidence) == 1

        # Insufficient evidence vs Contradicted check
        assert card.validate_insufficient_vs_contradicted("EVIDENCE_INSUFFICIENT") is True
        with pytest.raises(ContractViolation):
            card.validate_insufficient_vs_contradicted("EVIDENCE_INSUFFICIENT_IS_REFUTED")

        # Conventional vs Valid check: conventional is not automatically valid
        assert card.validate_conventional_vs_valid(is_conventional=True, is_valid=False) is False
        assert card.validate_conventional_vs_valid(is_conventional=True, is_valid=True) is True

    def test_03_8_model_portfolio_m0_to_m7_completeness(self):
        """Validates that ModelPortfolioM0ToM7 requires all 8 models (M0..M7)."""
        models_dict = {
            "M0": ModelM0ToM7Spec("M0", "Conventional", "Thread contention in pool", "Queue depth > 8", "Queue depth <= 2"),
            "M1": ModelM0ToM7Spec("M1", "Architectural", "Cacheline false sharing", "L3 misses > 20%", "L3 misses < 5%"),
            "M2": ModelM0ToM7Spec("M2", "Interaction", "VSync scanout memory collision", "VBlank miss rate > 0", "Zero frame breaches"),
            "M3": ModelM0ToM7Spec("M3", "Adversarial", "Hostile high-frequency jitter fuzzer", "Jitter > 10ms", "Jitter < 0.5ms"),
            "M4": ModelM0ToM7Spec("M4", "Orthogonal", "Confounding timer preemption", "Clock skew > 1ms", "Zero clock skew"),
            "M5": ModelM0ToM7Spec("M5", "Null", "System background thermal fluctuation", "Effect delta == 0", "Effect delta > 15%"),
            "M6": ModelM0ToM7Spec("M6", "Unconventional", "Non-linear DRAM bus phase lock", "Resonance peak at 120Hz", "Flat spectrum"),
            "M7": ModelM0ToM7Spec("M7", "Degraded", "Emergency fallback mode throttling", "Framerate capped at 60Hz", "Framerate >= 120Hz"),
        }

        portfolio = ModelPortfolioM0ToM7(
            question_id="Q-VSYNC-01",
            models=models_dict,
        )
        assert len(portfolio.models) == 8
        assert portfolio.vital_max_hp == 6

        # Missing one model (e.g. M7) must raise ContractViolation
        incomplete_dict = {k: v for k, v in models_dict.items() if k != "M7"}
        with pytest.raises(ContractViolation):
            ModelPortfolioM0ToM7(
                question_id="Q-INCOMPLETE",
                models=incomplete_dict,
            )

    def test_04_anti_averaging_conflict_resolution(self):
        """
        Anti-Averaging Invariant:
        Conflicting evidence must NOT be synthesized into a compromised average.
        Must create an EpistemicConflictNode and propose a discriminating test.
        """
        # Case 1: Attempting to create an averaged dispute node raises ContractViolation
        with pytest.raises(ContractViolation):
            EpistemicConflictNode(
                conflict_id="CONF-AVG-BAD",
                disputed_claim="Memory bandwidth throughput",
                thesis_evidence=["Benchmark A reports 45 GB/s"],
                antithesis_evidence=["Benchmark B reports 12 GB/s"],
                proposed_discriminating_test="Averaged to 28.5 GB/s",
                is_averaged=True, # Prohibited!
            )

        # Case 2: Proper dispute node preserves thesis and antithesis, proposes CDT
        node = EpistemicConflictNode(
            conflict_id="CONF-VALID-01",
            disputed_claim="IPC Ring buffer throughput under burst load",
            thesis_evidence=["Paper X claims ring saturates at 50,000 msg/s"],
            antithesis_evidence=["Local hardware benchmark shows 180,000 msg/s with zero-copy"],
            proposed_discriminating_test="CDT-PROBE-RING-HARDWARE-TRACE",
            is_averaged=False,
        )
        assert node.is_averaged is False
        assert len(node.thesis_evidence) == 1
        assert len(node.antithesis_evidence) == 1
        assert node.vital_max_hp == 6

    def test_05_unknown_mechanism_node_zero_confabulation(self):
        """
        Prompt / Protocol for Unknown Mechanism:
        Absence of evidence creates an explicit UNKNOWN_MECHANISM node.
        Confabulation is strictly prohibited.
        """
        # Confabulation detected -> raises ContractViolation
        with pytest.raises(ContractViolation):
            UnknownMechanismNode(
                unknown_id="UNK-CONFAB",
                symptom_description="Unexplained 20ms frame spike",
                missing_evidence=["Bus arbiter logs"],
                proposed_next_test="Probe bus arbiter",
                confabulation_detected=True, # Confabulation attempt
            )

        # Valid unknown node with explicit missing evidence fields
        node = UnknownMechanismNode(
            unknown_id="UNK-VALID-01",
            symptom_description="Periodic micro-stutter observed only under multi-monitor configuration",
            missing_evidence=[
                "Secondary display swapchain latency telemetry",
                "DWM compositor presentation queue depth",
            ],
            proposed_next_test="CDT-DUAL-DISPLAY-VSYNC-SWEEPER",
            confabulation_detected=False,
        )
        assert len(node.missing_evidence) == 2
        assert node.vital_max_hp == 6

    def test_06_non_sycophancy_and_anti_self_certification_output(self):
        """
        Enforces output formatting with status and uncertainty.
        Prevents sycophantic overconfident claims and self-certification.
        """
        router = EpistemicPromptRouter()

        # Proposer cannot certify own claim as APPROVED_FACT
        with pytest.raises(ContractViolation):
            router.format_output_with_status_and_uncertainty(
                status="APPROVED_FACT",
                uncertainty=0.01,
                statement="Self-certified inference",
                proposer_id="explore_agent",
                evaluator_id="explore_agent", # Self-certification breach!
            )

        # Independent audit format output
        out = router.format_output_with_status_and_uncertainty(
            status="APPROVED_FACT",
            uncertainty=0.04,
            statement="DP4A reconstruction validated at 41.2 dB PSNR",
            proposer_id="explore_agent_alpha",
            evaluator_id="verify_test_runner",
        )
        assert "[APPROVED_FACT | Uncertainty: 0.04 | Proposer: explore_agent_alpha | Auditor: verify_test_runner]" in out
        assert "DP4A reconstruction validated" in out

    def test_07_micro_prompt_router_surgical_routing(self):
        """
        Router selects and renders surgical task-specific micro-prompts,
        strictly avoiding monolithic prompt bloat.
        """
        router = EpistemicPromptRouter()

        # Route 1: Constitutional Audit Micro-Prompt
        rendered_const = router.route_micro_prompt(
            task_type=TaskType.CONSTITUTIONAL_AUDIT,
            context_args={
                "statement": "Zero-copy IPC buffer prevents buffer bloat",
                "source": "kernel/zero_copy_ring.py",
            },
        )
        assert "MICRO-PROMPT: Constitutional Claim Audit" in rendered_const
        assert "CONSTITUTIONAL_AUDIT" in rendered_const
        assert "VITAL_MAX_HP == 6" in rendered_const

        # Route 2: M0-M7 Generator Micro-Prompt
        rendered_m0m7 = router.route_micro_prompt(
            task_type=TaskType.HYPOTHESIS_M0_M7_GENERATION,
            context_args={
                "question_id": "Q-UMA-01",
                "problem_statement": "Resolve render stutter under concurrent LLM inference",
            },
        )
        assert "MICRO-PROMPT: 8-Model Portfolio Generator" in rendered_m0m7
        assert "M0 až M7" in rendered_m0m7

        # Route 3: Dispute Resolution Micro-Prompt
        rendered_dispute = router.route_micro_prompt(
            task_type=TaskType.DISPUTE_CONFLICT_RESOLUTION,
            context_args={
                "disputed_claim": "UMA Bandwidth saturation",
                "thesis_evidence": "Report 90%",
                "antithesis_evidence": "Measured 22%",
            },
        )
        assert "MICRO-PROMPT: Dispute Isolation & Anti-Averaging" in rendered_dispute
        assert "ZÁKAZ VYHLADZOVANIA PRIEMEROM" in rendered_dispute

        # Missing required context argument raises ContractViolation
        with pytest.raises(ContractViolation):
            router.route_micro_prompt(
                task_type=TaskType.CONSTITUTIONAL_AUDIT,
                context_args={"statement": "incomplete context"}, # Missing 'source'
            )
