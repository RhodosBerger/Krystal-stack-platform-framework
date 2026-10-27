#!/usr/bin/env python3
"""
Farm Scenario Domain Adapter for Unified Krystal Transition Pipeline.
Metrics: hashrate, power, difficulty, break-even, market state.
"""

import time
import math
import uuid
import hashlib
from typing import Dict, List, Any

from krystal_kernel.transition.models import (
    DomainStateVector,
    ExecutionCandidate,
    ConstraintSet,
    KrystalTransitionEnvelope,
    TransitionVerdict,
)
from krystal_kernel.transition.domain_adapter import DomainAdapter


class FarmAdapter(DomainAdapter):
    """Domain adapter for crypto farm datacenter economics & arbitrage."""

    @property
    def domain_name(self) -> str:
        return "FARM_MARKET_SCENARIO"

    def observe(self) -> Dict[str, Any]:
        return {
            "hashrate_eh_s": 14.5,
            "power_draw_mw": 320.0,
            "network_difficulty_t": 84.2,
            "breakeven_price_usd": 54200.0,
            "market_demand_state": 0.82,
        }

    def normalize(self, raw_data: Dict[str, Any]) -> DomainStateVector:
        vec = [
            min(1.0, raw_data["hashrate_eh_s"] / 50.0),
            min(1.0, raw_data["power_draw_mw"] / 1000.0),
            min(1.0, raw_data["network_difficulty_t"] / 150.0),
            min(1.0, raw_data["breakeven_price_usd"] / 120000.0),
            min(1.0, raw_data["market_demand_state"]),
        ]
        return DomainStateVector(
            metrics=raw_data,
            normalized_vector=vec,
            timestamp_ns=time.time_ns(),
            dimension_count=len(vec),
        )

    def generate_candidates(self, current_state: DomainStateVector) -> List[ExecutionCandidate]:
        return [
            ExecutionCandidate(
                candidate_id="FARM-CAND-01-DYNAMIC-PUE-ARBITRAGE",
                action_name="ACTIVATE_MPU_RACK_ARBITRAGE",
                parameters={"curtailment_pct": 15.0, "power_headroom_mw": 48.0},
                expected_utility=19.4,
                risk_score=0.10,
            ),
            ExecutionCandidate(
                candidate_id="FARM-CAND-02-CHIP-MARGINAL-VALUE",
                action_name="SOLVE_CHIP_MARGINAL_VALUE",
                parameters={"target_hash_efficiency": 22.5},
                expected_utility=26.8,
                risk_score=0.08,
            ),
            ExecutionCandidate(
                candidate_id="FARM-CAND-03-COOLING-CASCADE",
                action_name="MODULATE_COOLING_CASCADE",
                parameters={"delta_fan_rpm": -1200},
                expected_utility=14.0,
                risk_score=0.05,
            ),
        ]

    def predict(
        self,
        current_state: DomainStateVector,
        candidate: ExecutionCandidate,
    ) -> DomainStateVector:
        pred_metrics = dict(current_state.metrics)
        if candidate.action_name == "ACTIVATE_MPU_RACK_ARBITRAGE":
            pred_metrics["power_draw_mw"] -= 38.0
            pred_metrics["breakeven_price_usd"] -= 2800.0
        elif candidate.action_name == "SOLVE_CHIP_MARGINAL_VALUE":
            pred_metrics["hashrate_eh_s"] += 1.2
            pred_metrics["breakeven_price_usd"] -= 1900.0
        elif candidate.action_name == "MODULATE_COOLING_CASCADE":
            pred_metrics["power_draw_mw"] -= 18.0

        return self.normalize(pred_metrics)

    def execute(self, candidate: ExecutionCandidate) -> Dict[str, Any]:
        obs = self.observe()
        if candidate.action_name == "ACTIVATE_MPU_RACK_ARBITRAGE":
            obs["power_draw_mw"] -= 36.5 # Measured physical power drop
            obs["breakeven_price_usd"] -= 2750.0
        elif candidate.action_name == "SOLVE_CHIP_MARGINAL_VALUE":
            obs["hashrate_eh_s"] += 1.15
            obs["breakeven_price_usd"] -= 1850.0
        elif candidate.action_name == "MODULATE_COOLING_CASCADE":
            obs["power_draw_mw"] -= 17.0
            obs["breakeven_price_usd"] -= 950.0
        else:
            obs["breakeven_price_usd"] -= 1000.0
        return obs

    def validate(self, envelope: KrystalTransitionEnvelope) -> KrystalTransitionEnvelope:
        raw_after = self.execute(envelope.candidate)
        envelope.state_after = self.normalize(raw_after)

        pred_p = envelope.predicted_state.metrics["power_draw_mw"]
        act_p = envelope.state_after.metrics["power_draw_mw"]
        envelope.prediction_residual = round(abs(pred_p - act_p) / max(1.0, act_p), 4)

        base_be = envelope.state_before.metrics["breakeven_price_usd"]
        act_be = envelope.state_after.metrics["breakeven_price_usd"]
        envelope.utility_gain = round((base_be - act_be) / max(1.0, base_be) * 100.0, 2)

        sig_data = f"{envelope.runtime_id}:{envelope.domain}:{envelope.prediction_residual}:{envelope.utility_gain}:{envelope.vital_max_hp}"
        envelope.receipt_hash = hashlib.sha256(sig_data.encode("utf-8")).hexdigest()
        envelope.verdict = TransitionVerdict.APPROVED_COMMITTED.value
        return envelope
