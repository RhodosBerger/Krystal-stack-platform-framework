#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CRYPTO MARKET & MINING FARM SCENARIO ENGINE
==============================================================================
Module: krystal_kernel/scenario_market_farm_engine.py
Description:
    Authoritative dual-branch scenario model evaluating:
      Branch 1: ETF Market Inflows -> Demand Pressure & Price Projection
      Branch 2: Chip Expansion -> Farm Hashrate, Energy Cost, Yield, Break-Even
      Coupled: Tri-scenario (BEAR / BASE / BULL) and ChipMarginalValue solver.

    Constitutional Invariant: VITAL_MAX_HP == 6.
    Rejection Invariant: not Valid(Action) => Delta S == 0.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

from __future__ import annotations

import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from .constants import VITAL_MAX_HP, ContractViolation


class ScenarioKind(str, Enum):
    BEAR = "BEAR"
    BASE = "BASE"
    BULL = "BULL"


@dataclass(frozen=True)
class MarketState:
    """Branch 1: Financial ETF and Spot Market liquidity state."""
    btc_price_eur: float
    quarterly_etf_inflow_eur: float
    flow_persistence: float = 0.70  # [0.0, 1.0] momentum factor
    reference_liquidity_eur: float = 5_000_000_000.0  # Reference quarterly orderbook depth
    macro_multiplier: float = 1.0


@dataclass(frozen=True)
class FarmState:
    """Branch 2: Physical hardware and operational power parameters."""
    existing_chips: int
    new_chips: int
    hashrate_per_chip_th: float = 120.0  # TH/s per ASIC/GPU chip
    power_per_chip_w: float = 3000.0     # Watts per chip
    overhead_power_w: float = 50000.0    # Facility auxiliary power (cooling, networking)
    uptime_ratio: float = 0.985          # Operational uptime [0.0, 1.0]
    electricity_price_eur_kwh: float = 0.065
    pue: float = 1.15
    pool_fee_ratio: float = 0.015        # 1.5% pool fee
    miner_sell_ratio: float = 0.80       # Mined BTC sold for OpEx


@dataclass(frozen=True)
class NetworkState:
    """Global blockchain network conditions."""
    network_hashrate_eh: float = 650.0   # EH/s (1 EH/s = 1,000,000 TH/s)
    quarterly_total_btc: float = 41062.5 # ~450 BTC/day * 91.25 days per quarter


@dataclass(frozen=True)
class ScenarioOutcome:
    """Projected quarterly outcome for a specific market regime."""
    scenario_kind: ScenarioKind
    projected_btc_price_eur: float
    farm_revenue_eur: float
    farm_net_profit_eur: float
    is_profitable: bool
    confidence_score: float
    vital_max_hp: int = VITAL_MAX_HP


@dataclass(frozen=True)
class FarmExpansionReport:
    """Comprehensive dual-branch scenario evaluation report."""
    total_chips: int
    total_hashrate_th: float
    network_share_pct: float
    expected_quarterly_btc: float
    electricity_cost_eur: float
    breakeven_btc_price_eur: float
    chip_marginal_profit_eur: float
    etf_demand_pressure: float
    miner_sell_pressure_btc: float
    scenarios: Dict[ScenarioKind, ScenarioOutcome]
    vital_max_hp: int = VITAL_MAX_HP


class CryptoMarketFarmScenarioEngine:
    """
    Mathematical solver projecting quarterly crypto market scenarios
    coupled with physical farm capacity additions.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional breach: vital_max_hp must be 6, got {vital_max_hp}")
        self.vital_max_hp = vital_max_hp

    def evaluate_expansion(
        self,
        market: MarketState,
        farm: FarmState,
        network: NetworkState
    ) -> FarmExpansionReport:
        """
        Executes full dual-branch scenario projection.
        """
        # 1. Physical Hashrate & Production
        total_chips = farm.existing_chips + farm.new_chips
        total_hashrate_th = total_chips * farm.hashrate_per_chip_th

        # Network share: 1 EH/s = 1,000,000 TH/s
        total_network_th = network.network_hashrate_eh * 1_000_000.0
        network_share = total_hashrate_th / total_network_th
        network_share_pct = network_share * 100.0

        expected_btc = (
            network_share *
            network.quarterly_total_btc *
            farm.uptime_ratio *
            (1.0 - farm.pool_fee_ratio)
        )

        # 2. Power and Electricity Costs (Quarter = 91.25 days = 2190 hours)
        total_farm_w = (total_chips * farm.power_per_chip_w) + farm.overhead_power_w
        total_facility_kw = (total_farm_w / 1000.0) * farm.pue
        hours_per_quarter = 91.25 * 24.0
        kwh_per_quarter = total_facility_kw * hours_per_quarter
        electricity_cost = kwh_per_quarter * farm.electricity_price_eur_kwh

        # 3. Break-Even BTC Price
        breakeven_price = electricity_cost / expected_btc if expected_btc > 0 else 0.0

        # 4. ETF Demand Pressure (Branch 1)
        etf_pressure = (
            (market.quarterly_etf_inflow_eur * market.flow_persistence) /
            market.reference_liquidity_eur
        )
        miner_sell_pressure = expected_btc * farm.miner_sell_ratio

        # 5. Tri-Scenario Pricing & Profitability
        # Price multipliers calibrated to ETF pressure and macro factors
        scenario_multipliers = {
            ScenarioKind.BEAR: max(0.65, 0.85 + (etf_pressure * 0.10) - 0.15),
            ScenarioKind.BASE: max(0.85, 1.05 + (etf_pressure * 0.25)),
            ScenarioKind.BULL: max(1.10, 1.35 + (etf_pressure * 0.45)),
        }

        confidences = {
            ScenarioKind.BEAR: 0.82,
            ScenarioKind.BASE: 0.90,
            ScenarioKind.BULL: 0.75,
        }

        scenarios: Dict[ScenarioKind, ScenarioOutcome] = {}
        for kind, mult in scenario_multipliers.items():
            proj_price = market.btc_price_eur * mult * market.macro_multiplier
            revenue = expected_btc * proj_price
            profit = revenue - electricity_cost
            scenarios[kind] = ScenarioOutcome(
                scenario_kind=kind,
                projected_btc_price_eur=round(proj_price, 2),
                farm_revenue_eur=round(revenue, 2),
                farm_net_profit_eur=round(profit, 2),
                is_profitable=profit > 0.0,
                confidence_score=confidences[kind],
                vital_max_hp=self.vital_max_hp
            )

        # 6. Marginal Value of New Chips (ChipMarginalValue)
        # Compute baseline profit with existing chips only
        marginal_profit_per_chip = 0.0
        if farm.new_chips > 0:
            base_farm = FarmState(
                existing_chips=farm.existing_chips,
                new_chips=0,
                hashrate_per_chip_th=farm.hashrate_per_chip_th,
                power_per_chip_w=farm.power_per_chip_w,
                overhead_power_w=farm.overhead_power_w,
                uptime_ratio=farm.uptime_ratio,
                electricity_price_eur_kwh=farm.electricity_price_eur_kwh,
                pue=farm.pue,
                pool_fee_ratio=farm.pool_fee_ratio,
                miner_sell_ratio=farm.miner_sell_ratio
            )
            base_report = self.evaluate_expansion(market, base_farm, network)
            base_profit_base_case = base_report.scenarios[ScenarioKind.BASE].farm_net_profit_eur
            new_profit_base_case = scenarios[ScenarioKind.BASE].farm_net_profit_eur
            delta_profit = new_profit_base_case - base_profit_base_case
            marginal_profit_per_chip = delta_profit / farm.new_chips

        return FarmExpansionReport(
            total_chips=total_chips,
            total_hashrate_th=round(total_hashrate_th, 2),
            network_share_pct=round(network_share_pct, 5),
            expected_quarterly_btc=round(expected_btc, 4),
            electricity_cost_eur=round(electricity_cost, 2),
            breakeven_btc_price_eur=round(breakeven_price, 2),
            chip_marginal_profit_eur=round(marginal_profit_per_chip, 2),
            etf_demand_pressure=round(etf_pressure, 4),
            miner_sell_pressure_btc=round(miner_sell_pressure, 4),
            scenarios=scenarios,
            vital_max_hp=self.vital_max_hp
        )


# Global singleton instance
GLOBAL_CRYPTO_MARKET_FARM_ENGINE = CryptoMarketFarmScenarioEngine()
