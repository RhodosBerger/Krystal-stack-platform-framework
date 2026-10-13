#!/usr/bin/env python3
"""
==============================================================================
KARAF: ALGORITHM I — HYPOTHESIS MINER & SCIENTIFIC LEDGER
==============================================================================
Module: krystal_kernel/karaf/hypothesis_miner.py
Description:
    Mines empirical runtime hypotheses from discrepancy reports and prediction
    errors in the closed-loop execution plane:
      - Generates RuntimeHypothesis objects with explicit predictions & falsification criteria.
      - Tracks state lifecycle: PROPOSED -> TESTING -> SUPPORTED / FALSIFIED / CONTEXT_BOUND.
      - Context-bounds findings so local hardware phenomena are never falsely
        promoted to universal invariants.
==============================================================================
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.types import (
    RuntimeHypothesis,
    HypothesisStatus,
    ObservationFabricSnapshot
)


class HypothesisMiner:
    """
    Autonomous scientific hypothesis generator and ledger manager.
    Discovers, verifies, and falsifies runtime execution hypotheses.
    """

    def __init__(self):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP
        self._ledger: Dict[str, RuntimeHypothesis] = {}

    @property
    def ledger(self) -> Dict[str, RuntimeHypothesis]:
        return self._ledger

    def propose_worker_saturation_hypothesis(
        self,
        current_workers: int,
        attempted_workers: int,
        throughput_delta_pct: float,
        context_switch_delta_pct: float,
        hardware_context: Dict[str, Any]
    ) -> RuntimeHypothesis:
        """
        Synthesizes a worker saturation hypothesis when increasing workers
        yields negligible throughput gain but high contention.
        """
        h_id = f"hyp_worker_sat_{attempted_workers}w_{int(time.time())}"
        hyp = (
            f"Worker saturation region identified between {current_workers} and {attempted_workers} workers: "
            f"Throughput delta is {throughput_delta_pct}% despite +{context_switch_delta_pct}% context switches."
        )
        pred = (
            f"Configuring worker count to {current_workers + 1} will achieve >=98% of peak throughput "
            f"with significantly lower p99 latency jitter."
        )
        falsif = (
            f"Randomized multi-trial benchmark comparing {current_workers + 1} vs {attempted_workers} workers "
            f"over identical matrix convolution workloads."
        )

        rh = RuntimeHypothesis(
            hypothesis_id=h_id,
            hypothesis=hyp,
            prediction=pred,
            falsification_test=falsif,
            observations=["OBS_CPU_UTILIZATION", "OBS_CONTEXT_SWITCH_RATE"],
            derived_metrics=["DER_WORKER_CONTENTION"],
            hardware_context=hardware_context,
            workload_context={"type": "MATRIX_CONVOLUTION", "concurrency": attempted_workers},
            runtime_context={"current_workers": current_workers, "candidate_workers": attempted_workers},
            sample_count=1,
            effect_size=round(abs(throughput_delta_pct) / 100.0, 3),
            confidence=0.75,
            status=HypothesisStatus.HYP_PROPOSED,
            vital_max_hp=VITAL_MAX_HP
        )

        self._ledger[h_id] = rh
        return rh

    def record_hypothesis(self, hypothesis: RuntimeHypothesis):
        self._ledger[hypothesis.hypothesis_id] = hypothesis

    def get_hypothesis(self, hypothesis_id: str) -> Optional[RuntimeHypothesis]:
        return self._ledger.get(hypothesis_id)

    def list_hypotheses(self) -> List[Dict[str, Any]]:
        return [h.to_dict() for h in self._ledger.values()]

    def evaluate_test_outcome(
        self,
        hypothesis_id: str,
        observed_gain: float,
        expected_gain_threshold: float = 0.05
    ) -> RuntimeHypothesis:
        """
        Evaluates a hypothesis against experimental evidence.
        Updates status to HYP_SUPPORTED, HYP_FALSIFIED, or HYP_CONTEXT_BOUND.
        """
        rh = self._ledger.get(hypothesis_id)
        if not rh:
            raise KeyError(f"Hypothesis '{hypothesis_id}' not found in ledger")

        rh.sample_count += 1

        if observed_gain >= expected_gain_threshold:
            rh.status = HypothesisStatus.HYP_SUPPORTED
            rh.confidence = min(1.0, rh.confidence + 0.15)
        elif observed_gain <= -expected_gain_threshold:
            rh.status = HypothesisStatus.HYP_FALSIFIED
            rh.confidence = max(0.0, rh.confidence - 0.35)
        else:
            # Result inconclusive or strictly bound to specific hardware
            rh.status = HypothesisStatus.HYP_CONTEXT_BOUND
            rh.confidence = 0.50

        return rh
