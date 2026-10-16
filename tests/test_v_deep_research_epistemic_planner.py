"""
Integration Test Suite for Deep Research Epistemic Planner & Hypothesis Portfolio Protocol
=========================================================================================
Milestone: V-DEEP-RESEARCH-01
Constitutional Invariant: VITAL_MAX_HP == 6

Verifies:
  1. Inviolable constitutional invariant VITAL_MAX_HP == 6 across all deep research structures.
  2. Priority score formula: (GoalRelevance * InfoGain * Falsification) / Cost governs what to research, not what to believe.
  3. Evidence hierarchy: Local evidence strictly precedes external consensus (EVOI gate).
  4. Side findings stashing prevents cognitive tunneling and scope drift.
  5. Premise Refutation Escape Hatch: Counter-evidence invalidating a premise escalates to reframing / validator audit.
  6. Output Triad contract: Deliverables are strictly RESULT, UNKNOWN, or NEW_BRANCH with unalterable evidence trace.
  7. End-to-end deep research lifecycle execution.
"""

import pytest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.contracts import ContractViolation
from krystal_kernel.deep_research_epistemic_planner import (
    EvidenceScope,
    EpistemicOutputKind,
    ResearchHypothesis,
    SideFinding,
    PremiseRefutationReceipt,
    EpistemicOutputTriad,
    DeepResearchPlan,
)


class TestVDeepResearchEpistemicPlanner:

    def test_01_vital_max_hp_invariant(self):
        """Constitutional invariant VITAL_MAX_HP == 6 must hold in all structures."""
        assert VITAL_MAX_HP == 6

        hyp = ResearchHypothesis(
            hyp_id="HYP-01",
            premise="Zero-copy ring buffer reduces IPC overhead",
            mechanism="Shared memory ring slots",
            goal_relevance=0.9,
            expected_info_gain=0.8,
            falsification_value=0.85,
            cost_ms=2.0,
            vital_max_hp=6,
        )
        assert hyp.vital_max_hp == 6

        with pytest.raises(ContractViolation):
            ResearchHypothesis(
                hyp_id="HYP-BAD",
                premise="Bad HP",
                mechanism="none",
                goal_relevance=0.5,
                expected_info_gain=0.5,
                falsification_value=0.5,
                cost_ms=1.0,
                vital_max_hp=5,
            )

        with pytest.raises(ContractViolation):
            SideFinding(
                finding_id="SIDE-BAD",
                description="bad hp",
                source_context="context",
                vital_max_hp=7,
            )

        with pytest.raises(ContractViolation):
            DeepResearchPlan(
                plan_id="PLAN-BAD",
                goal_description="Bad Invariant",
                vital_max_hp=4,
            )

    def test_02_priority_formula_governs_investigation_not_belief(self):
        """
        Priority dictates WHAT TO INVESTIGATE, not what to believe.
        Rank is governed by (Relevance * InfoGain * Falsification) / Cost,
        explicitly rejecting sorting by prior probability or popularity.
        """
        plan = DeepResearchPlan(plan_id="PLAN-RANK-01", goal_description="Optimize UMA Latency")

        # Hyp 1: High popularity / high prior probability on web, but low info gain & low falsification
        hyp_popular = ResearchHypothesis(
            hyp_id="HYP-POPULAR-DOGMA",
            premise="Add more worker threads to threadpool",
            mechanism="Concurrency scaling",
            goal_relevance=0.8,
            expected_info_gain=0.2,   # Low new information
            falsification_value=0.2,  # Hard to conclusively falsify
            cost_ms=5.0,              # Slow test
            prior_probability=0.9,    # Very high prior probability / belief
            popularity_index=0.95,    # Widely recommended on forums
        )

        # Hyp 2: Unpopular/novel hypothesis, moderate prior probability, but high relevance, info gain, and falsification
        hyp_discriminating = ResearchHypothesis(
            hyp_id="HYP-NOVEL-DISCRIMINATING",
            premise="Display scanout memory controller phase collision",
            mechanism="VSync blanking micro-slicing",
            goal_relevance=0.95,
            expected_info_gain=0.9,   # High entropy reduction
            falsification_value=0.95, # Direct yes/no falsification
            cost_ms=1.5,              # Very fast probe
            prior_probability=0.3,    # Low conventional belief
            popularity_index=0.05,    # Not popular in generic articles
        )

        plan.register_hypothesis(hyp_popular)
        plan.register_hypothesis(hyp_discriminating)

        # Priority calculation:
        # hyp_popular: (0.8 * 0.2 * 0.2) / 5.0 = 0.032 / 5.0 = 0.0064
        # hyp_discriminating: (0.95 * 0.9 * 0.95) / 1.5 = 0.81225 / 1.5 = 0.5415
        assert hyp_discriminating.priority_score() > hyp_popular.priority_score()

        ranked = plan.rank_portfolio()
        assert ranked[0][0].hyp_id == "HYP-NOVEL-DISCRIMINATING"
        assert ranked[1][0].hyp_id == "HYP-POPULAR-DOGMA"

    def test_03_local_evidence_first_hierarchy(self):
        """
        Local evidence strictly precedes external lookups.
        External search is admitted only when EVOI > Cost and local evidence was evaluated.
        """
        plan = DeepResearchPlan(plan_id="PLAN-SCOPE-01", goal_description="Validate UMA IPC")

        # 1. Local workspace evidence is unconditionally admissible
        admissible_local, reason_local = plan.check_scope_admissibility(
            scope=EvidenceScope.LOCAL_WORKSPACE,
            local_evidence_available=False,
            expected_evoi=1.0,
            external_investigation_cost=1.0,
        )
        assert admissible_local is True
        assert "primary ground truth" in reason_local

        # 2. External consensus lookup before examining local evidence -> Blocked
        admissible_ext_early, reason_ext_early = plan.check_scope_admissibility(
            scope=EvidenceScope.EXTERNAL_CONSENSUS,
            local_evidence_available=False, # Local evidence unexamined!
            expected_evoi=5.0,
            external_investigation_cost=1.0,
        )
        assert admissible_ext_early is False
        assert "strictly precedes external lookups" in reason_ext_early

        # 3. External lookup with EVOI <= Cost -> Blocked (unjustified overhead)
        admissible_ext_low_evoi, reason_ext_low_evoi = plan.check_scope_admissibility(
            scope=EvidenceScope.EXTERNAL_CONSENSUS,
            local_evidence_available=True,
            expected_evoi=1.5,
            external_investigation_cost=2.0, # Cost > EVOI
        )
        assert admissible_ext_low_evoi is False
        assert "does not exceed Cost" in reason_ext_low_evoi

        # 4. External lookup with EVOI > Cost after examining local evidence -> Admitted
        admissible_ext_valid, reason_ext_valid = plan.check_scope_admissibility(
            scope=EvidenceScope.EXTERNAL_CONSENSUS,
            local_evidence_available=True,
            expected_evoi=8.0,
            external_investigation_cost=1.5,
        )
        assert admissible_ext_valid is True
        assert "EVOI exceeds investigation overhead" in reason_ext_valid

    def test_04_side_findings_stash_prevents_tunneling_and_scope_drift(self):
        """Side findings stash preserves serendipitous discoveries without losing research focus."""
        plan = DeepResearchPlan(plan_id="PLAN-STASH-01", goal_description="Measure IPC Ring Throughput")

        finding = plan.stash_side_finding(
            description="L3 cache miss rate drops by 12% when Godot subprocess runs in separate cgroup",
            source_context="tests/test_v_universal_engine_interposer.py",
        )

        assert finding.finding_id.startswith("SIDE-")
        assert finding.stashed is True
        assert finding.vital_max_hp == 6
        assert len(plan.side_findings) == 1
        assert "Godot subprocess" in plan.side_findings[0].description

    def test_05_premise_refutation_escape_hatch_and_reframing(self):
        """
        When counter-evidence refutes a foundational premise, the escape hatch triggers,
        escalating to Reframing / Validator Audit and creating a NEW_BRANCH.
        """
        plan = DeepResearchPlan(plan_id="PLAN-REFUTE-01", goal_description="Validate UMA Contention")

        hyp = ResearchHypothesis(
            hyp_id="HYP-PREMISE-01",
            premise="UMA memory bus bandwidth is saturated above 90%",
            mechanism="DRAM bus contention between GPU and NPU",
            goal_relevance=0.9,
            expected_info_gain=0.8,
            falsification_value=0.9,
            cost_ms=2.0,
        )
        plan.register_hypothesis(hyp)

        # Telemetry measurement reveals actual bus utilization is only 22% (refuting the premise!)
        receipt = plan.evaluate_premise_escape_hatch(
            hyp_id="HYP-PREMISE-01",
            counter_evidence_metric=0.22, # 22% bus utilization
            refutation_threshold=0.50,    # If < 50%, the 90% contention premise is demolished!
            symptom_description="Physical hardware telemetry shows memory bus at 22% utilization; stall is in queue lock, not bandwidth",
            escalate_to_validator_audit=False,
            refutes_when_less_than=True,
        )

        assert receipt is not None
        assert isinstance(receipt, PremiseRefutationReceipt)
        assert receipt.escalated_action == "REFRAME_PROBLEM"
        assert receipt.new_branch_id == "BRANCH-REFRAME-HYP-PREMISE-01"
        assert receipt.vital_max_hp == 6
        assert plan.portfolio["HYP-PREMISE-01"].status == "PREMISE_REFUTED"
        assert len(plan.refutation_escalations) == 1

    def test_06_output_triad_contract(self):
        """Deliverables must conform to Output Triad (RESULT, UNKNOWN, NEW_BRANCH) with evidence trace."""
        plan = DeepResearchPlan(plan_id="PLAN-OUTPUT-01", goal_description="Validate Engine Latency")

        # 1. RESULT Output
        out_result = plan.construct_output(
            output_kind=EpistemicOutputKind.RESULT,
            title="K-NSS DP4A Reconstruction Delivers 41.2 dB PSNR",
            evidence_trace=["tests/test_v_krystal_nexus_accelerator.py::test_05_k_nss_dp4a"],
        )
        assert out_result.kind == EpistemicOutputKind.RESULT
        assert out_result.vital_max_hp == 6

        # 2. UNKNOWN Output
        out_unknown = plan.construct_output(
            output_kind=EpistemicOutputKind.UNKNOWN,
            title="Unknown Ring Bus Phase Lock on ARM64 M-Series",
            evidence_trace=["hardware_trace_log_001"],
            missing_evidence=["Apple Silicon PMU counter trace", "Display pipeline VSync IRQ jitter log"],
        )
        assert out_unknown.kind == EpistemicOutputKind.UNKNOWN
        assert len(out_unknown.missing_evidence) == 2

        # 3. NEW_BRANCH Output
        receipt = PremiseRefutationReceipt(
            receipt_id="REFUT-TEST",
            refuted_premise="Memory bandwidth bottleneck",
            counter_evidence_summary="Bus utilization only 22%",
            measured_anomaly=0.22,
            escalated_action="REFRAME_PROBLEM",
            new_branch_id="BRANCH-REFRAME-LOCKS",
            vital_max_hp=6,
        )
        out_branch = plan.construct_output(
            output_kind=EpistemicOutputKind.NEW_BRANCH,
            title="Escalated Reframing to Lock Contention Model",
            evidence_trace=["kernel/uma_arbiter_telemetry.log"],
            receipt=receipt,
        )
        assert out_branch.kind == EpistemicOutputKind.NEW_BRANCH
        assert out_branch.escalation_receipt is not None
        assert out_branch.escalation_receipt.escalated_action == "REFRAME_PROBLEM"

        # Missing evidence trace raises ContractViolation
        with pytest.raises(ContractViolation):
            EpistemicOutputTriad(
                output_id="OUT-BAD",
                kind=EpistemicOutputKind.RESULT,
                title="Bad Output",
                evidence_trace=[], # Empty evidence trace!
                vital_max_hp=6,
            )

    def test_07_end_to_end_deep_research_lifecycle(self):
        """
        Executes end-to-end deep research flow:
        Plan Setup -> Portfolio Ranking -> Side Finding Stash -> Premise Refutation Escape Hatch -> NEW_BRANCH deliverable.
        """
        plan = DeepResearchPlan(
            plan_id="PLAN-E2E-01",
            goal_description="Eliminate 120Hz VSync scanout micro-stutter during LLM token burst",
        )

        h1 = ResearchHypothesis(
            hyp_id="H-UMA-SAT",
            premise="DRAM memory bus bandwidth is fully saturated during token generation",
            mechanism="Continuous matrix weights load from unified memory",
            goal_relevance=0.9,
            expected_info_gain=0.8,
            falsification_value=0.9,
            cost_ms=2.0,
        )
        h2 = ResearchHypothesis(
            hyp_id="H-VSYNC-LOCK",
            premise="Display controller scanout phase locks with LLM memory read burst",
            mechanism="Micro-slicing LLM burst to VBlank window eliminates scanout starvation",
            goal_relevance=0.95,
            expected_info_gain=0.95,
            falsification_value=0.9,
            cost_ms=1.5,
        )
        plan.register_hypothesis(h1)
        plan.register_hypothesis(h2)

        # Portfolio ranking selects H-VSYNC-LOCK first
        ranked = plan.rank_portfolio()
        assert ranked[0][0].hyp_id == "H-VSYNC-LOCK"

        # Stash an orthogonal discovery
        plan.stash_side_finding(
            description="Whisper acoustic audio ring buffer maintains < 0.2ms jitter at 16kHz",
            source_context="krystal_kernel/audiovisual_graph_diagnostics.py",
        )
        assert len(plan.side_findings) == 1

        # Check telemetry for H1: Bus bandwidth is actually only 25%! Refutes H1 premise!
        refutation_receipt = plan.evaluate_premise_escape_hatch(
            hyp_id="H-UMA-SAT",
            counter_evidence_metric=0.25,
            refutation_threshold=0.50,
            symptom_description="Physical hardware telemetry shows memory bus at 25% utilization; stall is phase-alignment, not saturation",
            escalate_to_validator_audit=False,
            refutes_when_less_than=True,
        )
        assert refutation_receipt is not None

        # Emit output deliverable strictly conforming to Output Triad (NEW_BRANCH)
        deliverable = plan.construct_output(
            output_kind=EpistemicOutputKind.NEW_BRANCH,
            title="Escalated Focus to VSync Blanking Micro-Slicing Scheduler",
            evidence_trace=[
                "tests/test_v_krystal_nexus_accelerator.py::test_03_vsync_micro_slice",
                "tests/test_v_krystal_nexus_accelerator.py::test_04_coexistence_cycle",
            ],
            receipt=refutation_receipt,
        )
        assert deliverable.kind == EpistemicOutputKind.NEW_BRANCH
        assert len(deliverable.evidence_trace) == 2
        assert deliverable.vital_max_hp == 6
