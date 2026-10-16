"""
Integration Test Suite for Epistemic Genesis Engine & 14-Prompt Pipeline
=======================================================================
Milestone: V-PROMPT-GENESIS-01
Constitutional Invariant: VITAL_MAX_HP == 6

Verifies:
  1. Inviolable constitutional invariant VITAL_MAX_HP == 6 across all genesis structures.
  2. Prompt 1 & 2: Raw characteristics extraction and strict correlation-vs-causality separation.
  3. Prompt 4 & 5: Source lineage de-duplication and social truth contamination auditing (Prevalence != Truth).
  4. Prompt 11: Deterministic Fact Promotion Gate with IVP >= 2, empirical test, and anti-self-certification.
  5. Prompt 13: Group prior vs individual verdict isolation (population is prior, never individual verdict).
  6. Prompt 14: Formal Unknown State Representation (explicit missing evidence, zero confabulation).
  7. Complete 14-prompt genesis catalog metadata integrity.
  8. End-to-end genesis pipeline lifecycle from raw unvalidated characteristics to verified fact.
"""

import pytest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.epistemic_provenance_logger import EpistemicVerdict
from krystal_kernel.epistemic_genesis_engine import (
    RawCharacteristic,
    PatternHypothesis,
    SourceNode,
    SourceLineageGraph,
    SocialTruthContaminationAuditor,
    GroupPriorIsolationGuard,
    UnknownMechanismState,
    DeterministicPromotionGate,
    PromptGenesisCatalog,
)


class TestVEpistemicGenesisEngine:

    def test_01_vital_max_hp_invariant(self):
        """Constitutional invariant VITAL_MAX_HP == 6 must be strictly preserved in all genesis structures."""
        assert VITAL_MAX_HP == 6

        raw = RawCharacteristic(
            obs_id="OBS-01",
            feature_name="frame_time_variance",
            feature_value=3.4,
            observation_type="TELEMETRY",
            source_uri="kernel/profiler",
            uncertainty_level="LOW",
            vital_max_hp=6,
        )
        assert raw.vital_max_hp == 6

        with pytest.raises(ContractViolation):
            RawCharacteristic(
                obs_id="OBS-BAD",
                feature_name="bad_hp",
                feature_value=1.0,
                observation_type="SENSOR",
                source_uri="sensor",
                uncertainty_level="LOW",
                vital_max_hp=5,
            )

        with pytest.raises(ContractViolation):
            UnknownMechanismState(
                unknown_id="UNK-BAD",
                symptom_description="Bad invariant",
                missing_evidence_fields=["data"],
                required_discriminating_probe="probe",
                vital_max_hp=7,
            )

        with pytest.raises(ContractViolation):
            DeterministicPromotionGate(vital_max_hp=4)

    def test_02_prompt_1_raw_characteristic_extraction(self):
        """Prompt 1: Raw characteristics are stored without causal narratives or explanations."""
        raw = RawCharacteristic(
            obs_id="OBS-MEM-001",
            feature_name="l3_miss_rate",
            feature_value=0.28,
            observation_type="HARDWARE_COUNTER",
            source_uri="cpu/counters/pcm",
            uncertainty_level="LOW",
        )
        assert raw.feature_name == "l3_miss_rate"
        assert raw.feature_value == 0.28
        assert raw.uncertainty_level == "LOW"

    def test_03_prompt_2_correlation_vs_causality_separation(self):
        """Prompt 2: Patterns explicitly hold unproven correlation status, separating frequency from causality."""
        pattern = PatternHypothesis(
            pattern_id="PAT-UMA-01",
            description="High LLM burst corresponds with render frame stutter",
            frequency_prior=0.88,
            independent_sources_count=3,
        )
        assert pattern.causal_status == "UNPROVEN_CORRELATION"
        assert pattern.frequency_prior == 0.88
        assert pattern.vital_max_hp == 6

    def test_04_prompt_4_and_5_source_lineage_and_social_contamination(self):
        """
        Prompt 4 & 5:
        1. 5 derivative blog posts/citations tracing back to 1 PR origin collapse to 1 IVP.
        2. High prevalence with IVP=1 produces high social truth contamination score.
        """
        graph = SourceLineageGraph()
        
        # Register single origin PR release
        graph.register_source(source_id="PR_RELEASE_ROOT", parent_origin_id=None)
        
        # 4 derivative articles citing the single root
        graph.register_source(source_id="ARTICLE_1", parent_origin_id="PR_RELEASE_ROOT")
        graph.register_source(source_id="ARTICLE_2", parent_origin_id="PR_RELEASE_ROOT")
        graph.register_source(source_id="ARTICLE_3", parent_origin_id="PR_RELEASE_ROOT")
        graph.register_source(source_id="BLOG_POST_4", parent_origin_id="ARTICLE_1")

        # Independent research lab hardware benchmark
        graph.register_source(source_id="LAB_INDEPENDENT_BENCHMARK", parent_origin_id=None, is_primary_hardware_measurement=True)

        # Citations list containing 4 derivative sources: Must collapse to 1 IVP!
        ivp_derivative_only = graph.get_independent_verification_paths(["ARTICLE_1", "ARTICLE_2", "ARTICLE_3", "BLOG_POST_4"])
        assert ivp_derivative_only == 1

        # Citations list including the independent benchmark: Collapses to 2 IVPs!
        ivp_with_independent = graph.get_independent_verification_paths(["ARTICLE_1", "BLOG_POST_4", "LAB_INDEPENDENT_BENCHMARK"])
        assert ivp_with_independent == 2

        # Audit Social Truth Contamination on derivative-only claim
        audit_res = SocialTruthContaminationAuditor.audit_contamination(
            corpus_frequency=0.92, # Very popular on web / papers
            independent_verification_paths=ivp_derivative_only, # But only 1 actual origin!
            commercial_incentive_detected=True,
            circular_citation_detected=False,
        )
        assert audit_res["is_contaminated"] is True
        assert audit_res["contamination_score"] >= 0.70
        assert "PREVALENCE_IS_NOT_TRUTH" in audit_res["epistemic_directive"]

    def test_05_prompt_11_deterministic_promotion_gate(self):
        """
        Prompt 11:
        Validates the strict conditions for promoting a hypothesis to APPROVED_FACT:
          - Proposer cannot self-certify (raises ContractViolation).
          - IVP < 2 without hardware instrument yields EVIDENCE_INSUFFICIENT.
          - Social truth contamination >= 0.50 yields SOURCE_CONFLICT.
          - Independent evaluator + passed test + IVP >= 2 yields APPROVED_FACT.
        """
        gate = DeterministicPromotionGate()

        # 1. Anti-Self-Certification breach
        with pytest.raises(ContractViolation):
            gate.evaluate_promotion(
                hypothesis_id="HYP-01",
                proposer_id="agent_alpha",
                evaluator_id="agent_alpha", # Proposer attempting self-approval!
                independent_verification_paths=3,
                empirical_test_passed=True,
                social_contamination_score=0.1,
            )

        # 2. Failed test execution
        verdict2, reason2 = gate.evaluate_promotion(
            hypothesis_id="HYP-02",
            proposer_id="agent_alpha",
            evaluator_id="agent_beta",
            independent_verification_paths=3,
            empirical_test_passed=False, # Test failed
            social_contamination_score=0.1,
        )
        assert verdict2 == EpistemicVerdict.REJECTED_REFUTED

        # 3. Insufficient independent paths (IVP=1 < 2)
        verdict3, reason3 = gate.evaluate_promotion(
            hypothesis_id="HYP-03",
            proposer_id="agent_alpha",
            evaluator_id="agent_beta",
            independent_verification_paths=1,
            empirical_test_passed=True,
            social_contamination_score=0.2,
        )
        assert verdict3 == EpistemicVerdict.EVIDENCE_INSUFFICIENT
        assert "IVP=1 < 2" in reason3

        # 4. High social contamination score (>= 0.50)
        verdict4, reason4 = gate.evaluate_promotion(
            hypothesis_id="HYP-04",
            proposer_id="agent_alpha",
            evaluator_id="agent_beta",
            independent_verification_paths=3,
            empirical_test_passed=True,
            social_contamination_score=0.65,
        )
        assert verdict4 == EpistemicVerdict.SOURCE_CONFLICT

        # 5. Fully validated through deterministic gate -> APPROVED_FACT
        verdict5, reason5 = gate.evaluate_promotion(
            hypothesis_id="HYP-05",
            proposer_id="agent_alpha",
            evaluator_id="agent_beta",
            independent_verification_paths=2,
            empirical_test_passed=True,
            social_contamination_score=0.15,
        )
        assert verdict5 == EpistemicVerdict.APPROVED_FACT

    def test_06_prompt_13_group_prior_vs_individual_verdict_isolation(self):
        """
        Prompt 13:
        Prohibits converting a population/group statistic into a deterministic verdict about an individual
        without direct individual measurement.
        """
        # Case 1: Attempting to assign individual verdict without direct measurement -> ContractViolation
        with pytest.raises(ContractViolation):
            GroupPriorIsolationGuard.evaluate_individual_attribution(
                group_statistical_prior=0.95,
                individual_direct_measurement=None,
                strict_protection=True,
            )

        # Case 2: Non-strict query returns unverified status requiring direct measurement
        res = GroupPriorIsolationGuard.evaluate_individual_attribution(
            group_statistical_prior=0.95,
            individual_direct_measurement=None,
            strict_protection=False,
        )
        assert res["verdict_allowed"] is False
        assert res["epistemic_status"] == "UNVERIFIED_INDIVIDUAL_REQUIRES_DIRECT_MEASUREMENT"

        # Case 3: Direct individual measurement present -> Verdict allowed
        res_valid = GroupPriorIsolationGuard.evaluate_individual_attribution(
            group_statistical_prior=0.95,
            individual_direct_measurement=14.2,
            strict_protection=True,
        )
        assert res_valid["verdict_allowed"] is True
        assert res_valid["individual_measured_value"] == 14.2
        assert res_valid["epistemic_status"] == "INDIVIDUAL_EMPIRICALLY_VERIFIED"

    def test_07_prompt_14_formal_unknown_state(self):
        """
        Prompt 14:
        Absence of evidence creates an explicit UNKNOWN node with documented missing evidence,
        strictly prohibiting confabulation or made-up narratives.
        """
        unknown_node = UnknownMechanismState(
            unknown_id="UNK-UMA-JITTER-01",
            symptom_description="Periodic 40ms frame spike on Intel Iris Xe under concurrent LPDDR read",
            missing_evidence_fields=[
                "Ring bus arbiter clock gating telemetry",
                "Display scanout line buffer watermark level",
            ],
            required_discriminating_probe="CDT-HARDWARE-TRACE-RING0",
            vital_max_hp=6,
        )
        assert unknown_node.is_formal_unknown is True
        assert len(unknown_node.missing_evidence_fields) == 2
        assert unknown_node.vital_max_hp == 6

    def test_08_catalog_all_14_prompts_complete(self):
        """Validates all 14 prompts exist, have non-empty objectives, and render correctly."""
        assert len(PromptGenesisCatalog.PROMPTS) == 14

        for prompt_num in range(1, 15):
            meta = PromptGenesisCatalog.get_prompt_metadata(prompt_num)
            assert "name" in meta
            assert "objective" in meta
            assert "directive" in meta
            assert len(meta["name"]) > 0
            assert len(meta["objective"]) > 0

            rendered = PromptGenesisCatalog.render_system_prompt(prompt_num, {"sample_feature": "latency"})
            assert f"PROMPT #{prompt_num}" in rendered
            assert "VITAL_MAX_HP == 6" in rendered

        # Out-of-bounds prompt query raises ContractViolation
        with pytest.raises(ContractViolation):
            PromptGenesisCatalog.get_prompt_metadata(15)

    def test_09_end_to_end_genesis_pipeline(self):
        """
        Executes end-to-end genesis flow:
        Raw Observation -> Pattern Synthesis -> Source Deduplication -> Contamination Audit -> Gate Promotion.
        """
        # Step 1: Raw Observation (Prompt 1)
        raw_obs = RawCharacteristic(
            obs_id="OBS-K-NSS-01",
            feature_name="psnr_reconstruction_db",
            feature_value=39.4,
            observation_type="BENCHMARK_PROBE",
            source_uri="tests/test_v_krystal_nexus_accelerator.py",
            uncertainty_level="LOW",
        )
        assert raw_obs.feature_value == 39.4

        # Step 2: Pattern (Prompt 2)
        pattern = PatternHypothesis(
            pattern_id="PAT-DP4A-01",
            description="DP4A 8-bit dot product maintains >= 38 dB PSNR while reducing compute latency by 40%",
            frequency_prior=0.75,
            independent_sources_count=2,
        )

        # Step 3: Source Lineage & Provenance (Prompt 5)
        graph = SourceLineageGraph()
        graph.register_source("HARDWARE_BENCHMARK_1", parent_origin_id=None, is_primary_hardware_measurement=True)
        graph.register_source("INDEPENDENT_VALIDATOR_2", parent_origin_id=None, is_primary_hardware_measurement=True)
        ivp = graph.get_independent_verification_paths(["HARDWARE_BENCHMARK_1", "INDEPENDENT_VALIDATOR_2"])
        assert ivp == 2

        # Step 4: Social Truth Contamination Audit (Prompt 4)
        audit = SocialTruthContaminationAuditor.audit_contamination(
            corpus_frequency=pattern.frequency_prior,
            independent_verification_paths=ivp,
            commercial_incentive_detected=False,
            circular_citation_detected=False,
        )
        assert audit["is_contaminated"] is False
        assert audit["contamination_score"] < 0.50

        # Step 5: Deterministic Fact Promotion Gate (Prompt 11)
        gate = DeterministicPromotionGate()
        verdict, reason = gate.evaluate_promotion(
            hypothesis_id="HYP-DP4A-VERIFIED",
            proposer_id="explore_agent",
            evaluator_id="verify_plane",
            independent_verification_paths=ivp,
            empirical_test_passed=True,
            social_contamination_score=audit["contamination_score"],
            is_hardware_instrument_verified=True,
        )
        assert verdict == EpistemicVerdict.APPROVED_FACT
        assert "successfully validated" in reason
