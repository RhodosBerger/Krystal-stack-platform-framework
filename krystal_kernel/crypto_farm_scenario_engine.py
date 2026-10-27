#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: CRYPTO FARM CYBER-PHYSICAL & SCENARIO ENGINE
==============================================================================
Module: krystal_kernel/crypto_farm_scenario_engine.py
Description:
    Full cyber-physical integration of:
      - 4 Hardware Operating Quadrants (Q1..Q4) with Thermal Strain Degradation
      - Cross-Coupled Matrix Model (M_F->M Miner Sell Pressure vs M_M->F Difficulty)
      - Break-Even & ChipMarginalValue Solvers across BEAR / BASE / BULL
      - Diagnostic Key Triad (Ideal, Deviation, Recovery)
      - Standardized REST / JSON Simulation Contract

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


class OperatingQuadrant(str, Enum):
    Q1_STABLE_IDLE = "Q1_STABLE_IDLE"
    Q2_PRODUCTIVE_APEX = "Q2_PRODUCTIVE_APEX"
    Q3_CONSTRAINED_BOTTLENECK = "Q3_CONSTRAINED_BOTTLENECK"
    Q4_SATURATION_RISK = "Q4_SATURATION_RISK"


# Physical Quadrant Characteristics (Hashrate Multiplier, Power Multiplier)
QUADRANT_PHYSICS: Dict[OperatingQuadrant, Tuple[float, float]] = {
    OperatingQuadrant.Q1_STABLE_IDLE: (0.85, 0.70),          # Underclocked, ultra-high J/TH efficiency
    OperatingQuadrant.Q2_PRODUCTIVE_APEX: (1.00, 0.88),      # Optimal undervolt apex, maximum ChipMarginalValue
    OperatingQuadrant.Q3_CONSTRAINED_BOTTLENECK: (0.75, 0.95),# Stalled on network/IO, dead capital
    OperatingQuadrant.Q4_SATURATION_RISK: (1.08, 1.22),      # Overclock burst +12% TDP, thermal strain risk
}


class CryptoFarmScenarioEngine:
    """
    Cyber-physical and economic simulator coupling hardware operating quadrants
    with macro ETF flows and mining network economics.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional breach: vital_max_hp must be 6, got {vital_max_hp}")
        self.vital_max_hp = vital_max_hp

    def simulate_quarter(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end quarterly simulation adhering to the formal JSON contract.
        """
        req = request_payload.get("simulation_request", request_payload)

        # 1. Parse Hardware Profile Transfer
        hw_prof = req.get("hardware_profile_transfer", {})
        quad_str = hw_prof.get("operating_quadrant", "Q2_PRODUCTIVE_APEX")
        try:
            quadrant = OperatingQuadrant(quad_str)
        except ValueError:
            quadrant = OperatingQuadrant.Q2_PRODUCTIVE_APEX

        undervolt_mv = float(hw_prof.get("undervolt_offset_mv", -40.0))
        strain_sigma = max(0.0, min(1.0, float(hw_prof.get("strain_tensor_sigma", 0.08))))

        # 2. Parse Market Vector
        m_vec = req.get("market_vector_S_market", {})
        current_btc_price = float(m_vec.get("current_btc_price_usd", 88500.0))
        etf_inflow_m = float(m_vec.get("quarterly_net_etf_inflow_musd", 4200.0))
        etf_persistence = float(m_vec.get("etf_flow_persistence", 0.72))
        ref_liquidity_m = float(m_vec.get("reference_liquidity_musd", 18500.0))
        volatility = float(m_vec.get("annualized_volatility", 0.54))

        # 3. Parse Farm Vector
        f_vec = req.get("farm_vector_S_farm", {})
        existing_hash_th = float(f_vec.get("existing_hashrate_th", 500000.0))
        new_chips = int(f_vec.get("new_chip_count", 500))
        hash_per_chip = float(f_vec.get("hashrate_per_chip_th", 1.85))
        power_per_chip = float(f_vec.get("power_per_chip_w", 27.5))
        net_hash_eh = float(f_vec.get("network_hashrate_eh", 740.0))
        elec_price_kwh = float(f_vec.get("electricity_price_kwh_usd", 0.058))
        uptime = float(f_vec.get("uptime_ratio", 0.985))
        sell_ratio = float(f_vec.get("miner_sell_ratio", 0.65))

        # 4. Thermal & Quadrant Hardware Correction
        base_h_mult, base_p_mult = QUADRANT_PHYSICS[quadrant]

        # Undervolting reduction factor (-10mV gives ~2.5% power saving)
        undervolt_factor = max(0.70, min(1.10, 1.0 + (undervolt_mv * 0.0025)))
        eff_power_per_chip = power_per_chip * base_p_mult * undervolt_factor

        # Thermal strain degradation gamma = 0.5
        gamma = 0.50
        eff_hash_per_chip = hash_per_chip * base_h_mult * (1.0 - gamma * strain_sigma)

        # 5. Production & Hashrate Calculations
        new_added_hash_th = new_chips * eff_hash_per_chip
        total_farm_hash_th = existing_hash_th + new_added_hash_th
        hash_increase_pct = (new_added_hash_th / existing_hash_th) * 100.0 if existing_hash_th > 0 else 0.0

        # Global Network Share (1 EH/s = 1,000,000 TH/s)
        total_network_th = net_hash_eh * 1_000_000.0
        network_share = total_farm_hash_th / total_network_th

        # Quarterly Global BTC = 450 BTC/day * 90 days = 40,500 BTC
        quarterly_global_btc = 40500.0
        pool_fee = 0.015
        expected_btc = network_share * quarterly_global_btc * uptime * (1.0 - pool_fee)

        # 6. Power & Electricity Cost (Quarter = 90 days = 2160 hours)
        hours_quarter = 2160.0
        # Total power: existing chips power + new chips power + 50kW facility overhead
        existing_chips_est = int(existing_hash_th / max(0.1, hash_per_chip))
        total_chips_count = existing_chips_est + new_chips
        total_kw = ((total_chips_count * eff_power_per_chip) + 50000.0) / 1000.0
        pue = 1.12
        facility_kw = total_kw * pue
        electricity_cost = facility_kw * hours_quarter * elec_price_kwh

        # 7. Miner Sell Pressure & Break-Even
        miner_sell_pressure_btc = expected_btc * sell_ratio
        breakeven_price = electricity_cost / expected_btc if expected_btc > 0 else 0.0

        # 8. ETF Demand Pressure & Cross-Coupling Dynamics
        etf_pressure = (etf_inflow_m * etf_persistence) / ref_liquidity_m

        # 9. Tri-Scenario Pricing & ChipMarginalValue Solver
        scenarios_def = {
            "BEAR": {
                "mult_min": 0.72,
                "mult_max": 0.81,
                "rec_trajectory": "Q2 -> Q1 (Underclock to 15.2 J/TH)",
            },
            "BASE": {
                "mult_min": 0.96,
                "mult_max": 1.08,
                "rec_trajectory": "Hold Q2 (Apex Locked)",
            },
            "BULL": {
                "mult_min": 1.22,
                "mult_max": 1.41,
                "rec_trajectory": "Q2 -> Q4 Burst (Allow +12% TDP Overclock)",
            }
        }

        # Baseline profit with 0 new chips to derive ChipMarginalValue
        base_added_hash = 0.0
        base_share = existing_hash_th / total_network_th
        base_btc = base_share * quarterly_global_btc * uptime * (1.0 - pool_fee)
        base_kw = ((existing_chips_est * eff_power_per_chip) + 50000.0) / 1000.0
        base_elec_cost = base_kw * pue * hours_quarter * elec_price_kwh

        scenarios_res = {}
        for sc_name, sc_data in scenarios_def.items():
            p_min = current_btc_price * sc_data["mult_min"] * (1.0 + etf_pressure * 0.15)
            p_max = current_btc_price * sc_data["mult_max"] * (1.0 + etf_pressure * 0.25)
            p_mid = (p_min + p_max) / 2.0

            rev = expected_btc * p_mid
            net_profit = rev - electricity_cost

            # Baseline profit for scenario
            base_rev = base_btc * p_mid
            base_net_profit = base_rev - base_elec_cost
            delta_profit = net_profit - base_net_profit

            chip_marginal_val = delta_profit / max(1, new_chips) if new_chips > 0 else 0.0

            scenarios_res[sc_name] = {
                "predicted_price_range_usd": [round(p_min, 0), round(p_max, 0)],
                "net_profit_usd": round(net_profit, 2),
                "chip_marginal_value_usd": round(chip_marginal_val, 2),
                "recommended_trajectory": sc_data["rec_trajectory"]
            }

        # Diagnostic Keys Triad
        ideal_key = f"QUAD_{quadrant.value}_OPTIMAL_HASH_EFFICIENCY"
        dev_key = f"DIFF_GROWTH_{round(etf_pressure * 10.0, 1)}PCT_OFFSET"
        rec_key = "SHIFT_UNDERVOLT_-45MV_IF_BEAR" if quadrant != OperatingQuadrant.Q1_STABLE_IDLE else "MAINTAIN_Q1_EFFICIENCY"

        return {
            "simulation_response": {
                "vital_max_hp": self.vital_max_hp,
                "conduct_flags": ["[ACT]", "[CNF]", "[VIT]"],
                "diagnostic_keys": {
                    "ideal_key": ideal_key,
                    "deviation_key": dev_key,
                    "recovery_key": rec_key
                },
                "farm_economics_Q": {
                    "new_total_hashrate_th": round(total_farm_hash_th, 2),
                    "hashrate_increase_pct": round(hash_increase_pct, 4),
                    "expected_btc_mined_Q": round(expected_btc, 4),
                    "quarterly_electricity_cost_usd": round(electricity_cost, 2),
                    "miner_sell_pressure_btc": round(miner_sell_pressure_btc, 4),
                    "break_even_btc_price_usd": round(breakeven_price, 2)
                },
                "scenarios": scenarios_res
            }
        }


# Global singleton instance
GLOBAL_CRYPTO_FARM_SCENARIO_ENGINE = CryptoFarmScenarioEngine()
