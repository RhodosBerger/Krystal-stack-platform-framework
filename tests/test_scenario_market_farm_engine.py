#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TESTS FOR CRYPTO MARKET & FARM SCENARIO ENGINE
==============================================================================
Module: tests/test_scenario_market_farm_engine.py
Description:
    Validates dual-branch crypto market & mining farm expansion solver:
      - Hashrate aggregation and network share calculation
      - Quarterly BTC yield with uptime and pool fee deductions
      - Electricity cost and break-even BTC price derivation
      - Tri-scenario price modeling (BEAR / BASE / BULL)
      - ChipMarginalValue marginal profit per chip calculation
      - Constitutional invariant VITAL_MAX_HP == 6 enforcement.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.scenario_market_farm_engine import (
    ScenarioKind,
    MarketState,
    FarmState,
    NetworkState,
    ScenarioOutcome,
    FarmExpansionReport,
    CryptoMarketFarmScenarioEngine,
    GLOBAL_CRYPTO_MARKET_FARM_ENGINE
)


class TestCryptoMarketFarmScenarioEngine(unittest.TestCase):
    """Test suite validating the dual-branch crypto market & farm expansion engine."""

    def setUp(self) -> None:
        self.engine = GLOBAL_CRYPTO_MARKET_FARM_ENGINE
        self.market = MarketState(
            btc_price_eur=65000.0,
            quarterly_etf_inflow_eur=1_200_000_000.0,
            flow_persistence=0.75,
            reference_liquidity_eur=5_000_000_000.0,
            macro_multiplier=1.0
        )
        self.farm = FarmState(
            existing_chips=1000,
            new_chips=500,
            hashrate_per_chip_th=120.0,
            power_per_chip_w=3000.0,
            overhead_power_w=50000.0,
            uptime_ratio=0.985,
            electricity_price_eur_kwh=0.055,
            pue=1.15,
            pool_fee_ratio=0.015,
            miner_sell_ratio=0.80
        )
        self.network = NetworkState(
            network_hashrate_eh=650.0,
            quarterly_total_btc=41062.5
        )

    def test_farm_expansion_and_production(self) -> None:
        """Verifies hashrate scaling and expected quarterly BTC yield."""
        report = self.engine.evaluate_expansion(self.market, self.farm, self.network)

        self.assertEqual(report.total_chips, 1500)
        self.assertEqual(report.total_hashrate_th, 180000.0)  # 1500 * 120
        self.assertAlmostEqual(report.network_share_pct, (180000.0 / 650000000.0) * 100.0, places=4)
        self.assertGreater(report.expected_quarterly_btc, 5.0)
        self.assertEqual(report.vital_max_hp, VITAL_MAX_HP)

    def test_energy_cost_and_breakeven_derivation(self) -> None:
        """Verifies electricity cost calculation and break-even BTC floor."""
        report = self.engine.evaluate_expansion(self.market, self.farm, self.network)

        self.assertGreater(report.electricity_cost_eur, 100000.0)
        self.assertGreater(report.breakeven_btc_price_eur, 10000.0)
        # Break-even should be lower than current spot price of 65k for profitable operation
        self.assertLess(report.breakeven_btc_price_eur, 65000.0)

    def test_tri_scenario_pricing_monotonicity(self) -> None:
        """Verifies monotonic ordering of BEAR <= BASE <= BULL scenario projections."""
        report = self.engine.evaluate_expansion(self.market, self.farm, self.network)

        bear = report.scenarios[ScenarioKind.BEAR]
        base = report.scenarios[ScenarioKind.BASE]
        bull = report.scenarios[ScenarioKind.BULL]

        self.assertLess(bear.projected_btc_price_eur, base.projected_btc_price_eur)
        self.assertLess(base.projected_btc_price_eur, bull.projected_btc_price_eur)

        self.assertLess(bear.farm_revenue_eur, base.farm_revenue_eur)
        self.assertLess(base.farm_revenue_eur, bull.farm_revenue_eur)

        # Profits match revenue - electricity cost
        for outcome in (bear, base, bull):
            self.assertAlmostEqual(
                outcome.farm_net_profit_eur,
                outcome.farm_revenue_eur - report.electricity_cost_eur,
                places=1
            )
            self.assertEqual(outcome.vital_max_hp, VITAL_MAX_HP)

    def test_chip_marginal_profit_calculation(self) -> None:
        """Verifies calculation of marginal profit per newly added chip (ChipMarginalValue)."""
        report = self.engine.evaluate_expansion(self.market, self.farm, self.network)

        # Adding 500 efficient chips at 65k spot should yield positive marginal value per chip
        self.assertGreater(report.chip_marginal_profit_eur, 0.0)

    def test_constitutional_invariant_rejection(self) -> None:
        """Verifies engine rejects initialization with vital_max_hp != 6."""
        with self.assertRaises(ContractViolation):
            CryptoMarketFarmScenarioEngine(vital_max_hp=8)


if __name__ == "__main__":
    unittest.main()
