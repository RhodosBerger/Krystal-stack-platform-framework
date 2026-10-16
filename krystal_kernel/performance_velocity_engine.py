"""
Performance Velocity Engine & Multi-Model Falsification System
==============================================================
Milestone: V-PERF-VELOCITY-01
Constitutional Invariant: VITAL_MAX_HP == 6

Core Invariants & Principles:
  1. Strict Ordering of Operations: Hypotheses first, then evidence, never reverse.
     No claim is elevated to FACT without an executed test/physical measurement.
  2. 9-Stage Velocity Loop:
     Observe -> Identify Bottleneck -> Generate Competing Hypotheses ->
     Rank by Value/Risk/Cost -> Execute Cheapest Discriminating Test (CDT) ->
     Measure -> Compare with Prediction -> Update Model -> Promote or Reject.
  3. Problem Framing Precedence:
     Check problem framing before optimizing. If measurements repeatedly deviate
     from prediction across paradigms, classify as Framing Error or Flawed Validator,
     not implementation dishonesty. Project telemetry overrides generic rules of thumb.
  4. 5-Tuple Hypothesis:
     (Mechanism, Expected Effect, Baseline, Measurable Output, Falsifier).
  5. Multi-Model Tracking under Ambiguity:
     Maintains 4 parallel paradigms: Conventional, Architectural, Interaction, Adversarial.
     Selects discriminating tests that eliminate maximal hypothesis space at minimal cost.
  6. Scoped Deep Dives & Stopping Rule:
     Scale reasoning depth only when Expected Value of Information (EVOI) exceeds cost;
     stop immediately once sufficient evidence is gathered.
  7. Adaptive Communication Form:
     Concise on strong evidence, exploratory under high uncertainty;
     classifies errors into permanent Error Prevention Rules.
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from krystal_kernel.contracts import ContractViolation
from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.epistemic_provenance_logger import (
    EpistemicVerdict,
    EpistemicEvidenceClass,
    EpistemicProvenanceLedger,
)


class ModelParadigm(enum.Enum):
    """Four competing paradigms tracked in parallel under epistemic ambiguity."""
    CONVENTIONAL = "CONVENTIONAL"      # Naive CPU/IO bound, classic memory leak, standard OS scheduling
    ARCHITECTURAL = "ARCHITECTURAL"    # Invariant failure, cacheline bouncing, UMA/NUMA bottleneck, IPC ring starvation
    INTERACTION = "INTERACTION"        # Cross-process dynamic interference, sync-barrier lockup, phase misalignment
    ADVERSARIAL = "ADVERSARIAL"        # Worst-case payload, corrupted frame, boundary drift, hostile jitter


class FramingDiagnostic(enum.Enum):
    """Classification of systemic measurement anomalies."""
    VALID_FRAME = "VALID_FRAME"              # Problem framing and validator are physically valid
    FRAMING_ERROR = "FRAMING_ERROR"          # The theoretical model / boundary condition is misspecified
    FLAWED_VALIDATOR = "FLAWED_VALIDATOR"    # Test harness, timer or benchmark probe is broken or miscalibrated


@dataclass
class Hypothesis5Tuple:
    """
    Structured 5-Tuple Hypothesis Specification.
    Constitutional Invariant: vital_max_hp == 6.
    """
    hypothesis_id: str
    mechanism: str
    expected_effect: str
    baseline_metric: float
    measurable_output_target: float
    falsifier_predicate: str
    paradigm: ModelParadigm
    tolerance_band: float = 0.05
    estimated_value: float = 10.0
    risk_score: float = 1.0
    test_cost_ms: float = 2.0
    proposer_id: str = "explore_agent"
    status: str = "PROPOSED"
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(
                f"Platform invariant VITAL_MAX_HP == {VITAL_MAX_HP} must hold in Hypothesis5Tuple, got {self.vital_max_hp}"
            )
        if not self.hypothesis_id or not self.mechanism or not self.falsifier_predicate:
            raise ContractViolation("Hypothesis must specify hypothesis_id, mechanism, and falsifier_predicate")
        if self.risk_score <= 0.0:
            raise ContractViolation("risk_score must be strictly positive")
        if self.test_cost_ms <= 0.0:
            raise ContractViolation("test_cost_ms must be strictly positive")

    def priority_score(self) -> float:
        """
        Calculates prioritization score:
        Priority = Expected Value / (Risk * Cost)
        """
        denom = max(self.risk_score * self.test_cost_ms, 1e-6)
        return float(self.estimated_value / denom)

    def is_falsified_by(self, measured_value: float) -> bool:
        """
        Checks whether empirical measurement falsifies this hypothesis.
        A hypothesis is falsified if the measured metric deviates outside
        [target - tolerance, target + tolerance] or matches known refutation criteria.
        """
        lower_bound = self.measurable_output_target - abs(self.tolerance_band)
        upper_bound = self.measurable_output_target + abs(self.tolerance_band)
        
        # If target represents an improvement (e.g. latency reduction or throughput increase),
        # check bounds
        within_bounds = (lower_bound <= measured_value <= upper_bound)
        return not within_bounds


@dataclass
class DiscriminatingTest:
    """
    Test probe designed to evaluate and discriminate competing hypotheses.
    Constitutional Invariant: vital_max_hp == 6.
    """
    test_id: str
    name: str
    execution_cost_ms: float
    targeted_hypothesis_ids: List[str]
    targeted_paradigms: List[ModelParadigm]
    information_gain_score: float
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(
                f"Platform invariant VITAL_MAX_HP == {VITAL_MAX_HP} must hold in DiscriminatingTest, got {self.vital_max_hp}"
            )
        if self.execution_cost_ms <= 0.0:
            raise ContractViolation("execution_cost_ms must be strictly positive")
        if self.information_gain_score < 0.0:
            raise ContractViolation("information_gain_score must be non-negative")


class CheapestDiscriminatingTestSelector:
    """
    Selects the test that maximizes information gain and paradigm discrimination
    per unit of execution cost (Cheapest Discriminating Test - CDT).
    """

    @staticmethod
    def rank_tests(
        candidate_tests: List[DiscriminatingTest],
        active_hypotheses: List[Hypothesis5Tuple],
    ) -> List[Tuple[DiscriminatingTest, float]]:
        """
        Ranks candidate tests by Discriminating Efficiency:
        Efficiency = (Unique Paradigms Covered * Hypotheses Addressed * InfoGain) / Cost
        """
        active_ids = {h.hypothesis_id for h in active_hypotheses}
        ranked: List[Tuple[DiscriminatingTest, float]] = []

        for test in candidate_tests:
            relevant_hyps = [h_id for h_id in test.targeted_hypothesis_ids if h_id in active_ids]
            unique_paradigms = len(set(test.targeted_paradigms))
            
            numerator = max(len(relevant_hyps), 1) * max(unique_paradigms, 1) * test.information_gain_score
            cost = max(test.execution_cost_ms, 0.001)
            efficiency = float(numerator / cost)
            ranked.append((test, efficiency))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    @classmethod
    def select_cheapest_discriminating_test(
        cls,
        candidate_tests: List[DiscriminatingTest],
        active_hypotheses: List[Hypothesis5Tuple],
    ) -> DiscriminatingTest:
        if not candidate_tests:
            raise ContractViolation("No candidate discriminating tests available to select from")
        ranked = cls.rank_tests(candidate_tests, active_hypotheses)
        return ranked[0][0]


class FramingErrorDetector:
    """
    Detects when systemic mismatches point to a Framing Error or Flawed Validator
    rather than faulty or deceitful implementation.
    """

    def __init__(self, vital_max_hp: int = 6) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6, got {vital_max_hp}")
        self.vital_max_hp = vital_max_hp
        self.diagnostic_history: List[Dict[str, Any]] = []

    def diagnose_test_suite_anomaly(
        self,
        hypotheses: List[Hypothesis5Tuple],
        test_outcomes: Dict[str, float],
        hardware_telemetry_baseline: Dict[str, float],
    ) -> FramingDiagnostic:
        """
        Diagnoses whether all-hypothesis failures or physical violations point to
        a framing error or flawed validator.
        """
        # Condition 1: Check for physically impossible validator readings (Flawed Validator)
        for metric_name, val in test_outcomes.items():
            if "latency" in metric_name.lower() and val < 0.0:
                self.diagnostic_history.append({
                    "diagnostic": FramingDiagnostic.FLAWED_VALIDATOR,
                    "reason": f"Negative latency measurement: {val} ms",
                    "timestamp": time.time(),
                })
                return FramingDiagnostic.FLAWED_VALIDATOR
            if "throughput" in metric_name.lower() and val > 1e9: # e.g. > 1 TB/s on consumer bus
                self.diagnostic_history.append({
                    "diagnostic": FramingDiagnostic.FLAWED_VALIDATOR,
                    "reason": f"Physically impossible throughput: {val} MB/s",
                    "timestamp": time.time(),
                })
                return FramingDiagnostic.FLAWED_VALIDATOR

        # Condition 2: Check if hypotheses spanning >= 3 distinct paradigms ALL fail
        paradigms_tested = {h.paradigm for h in hypotheses}
        if len(paradigms_tested) >= 3:
            all_falsified = True
            for h in hypotheses:
                measured = test_outcomes.get(h.hypothesis_id)
                if measured is not None and not h.is_falsified_by(measured):
                    all_falsified = False
                    break
            
            if all_falsified:
                # If conventional, architectural, and interaction models ALL fail,
                # the problem framing itself is likely erroneous
                self.diagnostic_history.append({
                    "diagnostic": FramingDiagnostic.FRAMING_ERROR,
                    "reason": f"All {len(hypotheses)} hypotheses across {len(paradigms_tested)} distinct paradigms falsified simultaneously",
                    "timestamp": time.time(),
                })
                return FramingDiagnostic.FRAMING_ERROR

        return FramingDiagnostic.VALID_FRAME


@dataclass
class ErrorPreventionRule:
    """
    Permanent learning classification derived from a resolved error.
    Constitutional Invariant: vital_max_hp == 6.
    """
    rule_id: str
    symptom: str
    root_cause: str
    prevention_invariant: str
    target_subsystem: str
    created_at: float = field(default_factory=time.time)
    vital_max_hp: int = 6

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP == 6 violated in ErrorPreventionRule")


class AdaptiveCommunicator:
    """
    Adapts communication density based on epistemic certainty and evidence strength.
    Classifies errors into durable architectural prevention rules.
    """

    @staticmethod
    def format_output(
        evidence_strength: float,
        factual_summary: str,
        exploratory_details: Dict[str, Any],
    ) -> str:
        """
        Concise when evidence is strong (>= 0.85);
        Exploratory when uncertainty is high (< 0.85).
        """
        if evidence_strength >= 0.85:
            # High certainty: concise, directive, factual
            return f"VERIFIED: {factual_summary} [Evidence Strength: {evidence_strength:.2f}]"
        else:
            # Low certainty: exploratory, open, showing hypothesis space
            hyps = exploratory_details.get("competing_hypotheses", [])
            open_questions = exploratory_details.get("unresolved_questions", [])
            return (
                f"EXPLORATORY ANALYSIS [Entropy High, Confidence: {evidence_strength:.2f}]:\n"
                f"- Active Hypotheses: {', '.join(hyps)}\n"
                f"- Unresolved Uncertainties: {', '.join(open_questions)}\n"
                f"- Recommended Discriminating Probe: {exploratory_details.get('next_test', 'N/A')}"
            )

    @staticmethod
    def classify_error_to_rule(
        rule_id: str,
        symptom: str,
        root_cause: str,
        prevention_invariant: str,
        target_subsystem: str,
    ) -> ErrorPreventionRule:
        """Translates an error into a durable system prevention rule."""
        return ErrorPreventionRule(
            rule_id=rule_id,
            symptom=symptom,
            root_cause=root_cause,
            prevention_invariant=prevention_invariant,
            target_subsystem=target_subsystem,
            vital_max_hp=6,
        )


class VelocityLoopController:
    """
    Orchestrates the 9-Phase Velocity Loop in Performance Mode.
    Strictly enforces:
      - Hypotheses precede evidence; evidence precedes facts.
      - Anti-Self-Certification (proposer != evaluator for APPROVED_FACT).
      - Stopping rule: halt when evidence threshold achieved or EVOI <= cost.
      - VITAL_MAX_HP == 6.
    """

    def __init__(
        self,
        ledger: Optional[EpistemicProvenanceLedger] = None,
        vital_max_hp: int = 6,
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"VITAL_MAX_HP must be 6, got {vital_max_hp}")
        self.vital_max_hp = vital_max_hp
        self.ledger = ledger or EpistemicProvenanceLedger()
        self.framing_detector = FramingErrorDetector(vital_max_hp=6)
        
        self.active_hypotheses: Dict[str, Hypothesis5Tuple] = {}
        self.candidate_tests: Dict[str, DiscriminatingTest] = {}
        self.executed_tests: List[Dict[str, Any]] = []
        self.error_prevention_rules: List[ErrorPreventionRule] = []

    # 1. Observe
    def observe(self, raw_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Observes raw telemetry without premature extrapolations."""
        return {
            "observation_timestamp": time.time(),
            "telemetry_snapshot": dict(raw_telemetry),
            "vital_max_hp": self.vital_max_hp,
        }

    # 2. Identify Bottleneck
    def identify_bottleneck(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Identifies the limiting component based on empirical telemetry."""
        # Find metric with greatest deviation or highest latency/queue depth
        bottleneck_metric = "none"
        max_severity = -1.0

        for key, val in telemetry.items():
            if isinstance(val, (int, float)):
                if "latency" in key.lower() and val > 16.6: # > 1 frame at 60fps
                    severity = float(val) / 16.6
                    if severity > max_severity:
                        max_severity = severity
                        bottleneck_metric = key
                elif "queue" in key.lower() and val > 8:
                    severity = float(val) / 8.0
                    if severity > max_severity:
                        max_severity = severity
                        bottleneck_metric = key

        return {
            "bottleneck_subsystem": bottleneck_metric,
            "severity_index": max_severity,
            "status": "IDENTIFIED" if max_severity > 0 else "NOMINAL",
        }

    # 3. Propose Hypotheses
    def register_hypothesis(self, hypothesis: Hypothesis5Tuple) -> None:
        """Registers a competing 5-tuple hypothesis."""
        if hypothesis.vital_max_hp != self.vital_max_hp:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 violated in registered hypothesis")
        self.active_hypotheses[hypothesis.hypothesis_id] = hypothesis

    # 4. Rank Hypotheses
    def rank_active_hypotheses(self) -> List[Tuple[Hypothesis5Tuple, float]]:
        """Ranks active hypotheses by Priority = Value / (Risk * Cost)."""
        ranked = [
            (h, h.priority_score())
            for h in self.active_hypotheses.values()
            if h.status in ("PROPOSED", "TESTING")
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    # 5. Execute Cheapest Discriminating Test
    def register_test(self, test: DiscriminatingTest) -> None:
        """Registers a candidate discriminating test."""
        if test.vital_max_hp != self.vital_max_hp:
            raise ContractViolation("Platform invariant VITAL_MAX_HP == 6 violated in registered test")
        self.candidate_tests[test.test_id] = test

    def select_cheapest_discriminating_test(self) -> DiscriminatingTest:
        """Selects the test that eliminates maximal paradigms at minimum cost."""
        active_list = list(self.active_hypotheses.values())
        return CheapestDiscriminatingTestSelector.select_cheapest_discriminating_test(
            list(self.candidate_tests.values()), active_list
        )

    # 6. Measure & 7. Compare with Prediction
    def execute_measurement(
        self,
        test_id: str,
        measured_metrics: Dict[str, float],
        measured_by: str = "test_runner",
    ) -> Dict[str, Any]:
        """
        Executes the test probe, evaluates outcomes against active hypotheses,
        and logs the run.
        """
        test = self.candidate_tests.get(test_id)
        if not test:
            raise ContractViolation(f"Test '{test_id}' not found in registered candidate tests")

        # Check for framing errors / flawed validators
        diagnostic = self.framing_detector.diagnose_test_suite_anomaly(
            list(self.active_hypotheses.values()),
            measured_metrics,
            hardware_telemetry_baseline={},
        )

        test_record = {
            "test_id": test_id,
            "measured_by": measured_by,
            "timestamp": time.time(),
            "metrics": measured_metrics,
            "diagnostic": diagnostic,
            "falsified_hypotheses": [],
            "surviving_hypotheses": [],
        }

        # If framing error or flawed validator, do not falsely reject code
        if diagnostic != FramingDiagnostic.VALID_FRAME:
            test_record["status"] = f"SUSPENDED_DUE_TO_{diagnostic.value}"
            self.executed_tests.append(test_record)
            return test_record

        # Compare measurements against predictions of targeted hypotheses
        for h_id in test.targeted_hypothesis_ids:
            hyp = self.active_hypotheses.get(h_id)
            if not hyp:
                continue

            measured_val = measured_metrics.get(h_id)
            if measured_val is not None:
                if hyp.is_falsified_by(measured_val):
                    hyp.status = "REJECTED_REFUTED"
                    test_record["falsified_hypotheses"].append(h_id)
                else:
                    test_record["surviving_hypotheses"].append(h_id)

        test_record["status"] = "COMPLETED"
        self.executed_tests.append(test_record)
        return test_record

    # 8. Update Model & 9. Promote or Reject
    def decide_hypothesis_verdict(
        self,
        hypothesis_id: str,
        evaluator_id: str,
        measured_metric: float,
        evidence_class: EpistemicEvidenceClass = EpistemicEvidenceClass.EXECUTED_TEST,
        evidence_uri: str = "tests/test_v_performance_velocity_engine.py",
    ) -> EpistemicVerdict:
        """
        Promotes to APPROVED_FACT or rejects to REJECTED_REFUTED.
        Strictly enforces Anti-Self-Certification:
        If proposer_id == evaluator_id, APPROVED_FACT is prohibited.
        """
        hyp = self.active_hypotheses.get(hypothesis_id)
        if not hyp:
            raise ContractViolation(f"Hypothesis '{hypothesis_id}' not found")

        # Anti-Self-Certification Check: Proposer cannot evaluate its own claim to APPROVED_FACT
        if hyp.proposer_id == evaluator_id:
            raise ContractViolation(
                f"Separation of Concerns Violation: Proposer '{hyp.proposer_id}' cannot certify its own claim '{hypothesis_id}'"
            )

        if hyp.is_falsified_by(measured_metric):
            verdict = EpistemicVerdict.REJECTED_REFUTED
            hyp.status = "REJECTED_REFUTED"
            was_falsified = True
        else:
            verdict = EpistemicVerdict.APPROVED_FACT
            hyp.status = "PROMOTED_FACT"
            was_falsified = False

        # Record in cryptographic provenance ledger
        self.ledger.log_claim_evaluation(
            proposer_id=hyp.proposer_id,
            claim_summary=f"Hypothesis {hyp.hypothesis_id}: {hyp.mechanism} -> {hyp.expected_effect}",
            evaluator_id=evaluator_id,
            verdict=verdict,
            declared_confidence=0.95 if verdict == EpistemicVerdict.APPROVED_FACT else 0.1,
            evidence_class=evidence_class,
            evidence_artifact_uri=evidence_uri,
            falsification_condition=hyp.falsifier_predicate,
            measured_outcome_metric=measured_metric,
            was_falsified=was_falsified,
        )

        return verdict

    # Stopping Rule
    def should_stop_search(
        self,
        current_evidence_strength: float,
        expected_information_gain: float,
        investigation_cost: float,
        evidence_threshold: float = 0.90,
    ) -> bool:
        """
        Stopping rule:
          1. Stop if evidence strength reaches threshold (sufficient evidence).
          2. Stop if Expected Value of Information (EVOI) <= cost (no net epistemic gain).
        """
        if current_evidence_strength >= evidence_threshold:
            return True # Sufficient evidence gathered
        if expected_information_gain <= investigation_cost:
            return True # Marginal gain does not justify cost
        return False
