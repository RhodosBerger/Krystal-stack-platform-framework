#!/usr/bin/env python3
"""
OEE Economic Engine (Emerald Logic)
Implements real-time profit optimization and spindle balancing.
Theoretical Grounding: 
- Profit Rate Pr = (Sp - Cu) / Tu
- Spindle Balance Rp = 60 / max(M1, M2)
"""

import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [EMERALD] - %(message)s')
logger = logging.getLogger(__name__)

class EconomicEngine:
    """
    The mathematical brain for the Manager Persona.
    Optimizes for Profit Rate over simple Speed.
    """
    
    @staticmethod
    def calculate_profit_rate(sale_price: float, unit_cost: float, unit_time_mins: float) -> float:
        """
        Pr = (Sp - Cu) / Tu
        Returns Profit per Minute.
        """
        if unit_time_mins <= 0:
            return 0.0
        return (sale_price - unit_cost) / unit_time_mins

    @staticmethod
    def optimize_mode(requirements: Dict[str, Any], current_telemetry: Dict[str, Any]) -> str:
        """
        Decides between RUSH_MODE and ECONOMY_MODE based on Pr curves.
        """
        sp = requirements.get("sale_price", 100.0)
        base_cu = requirements.get("base_unit_cost", 40.0)
        base_tu = requirements.get("base_unit_time", 20.0)
        
        # Rush Mode Projection: 20% faster, but 50% more tool wear/cost
        rush_tu = base_tu * 0.8
        rush_cu = base_cu * 1.5
        rush_pr = EconomicEngine.calculate_profit_rate(sp, rush_cu, rush_tu)
        
        # Economy Mode Projection: 20% slower, but 30% cheaper
        econ_tu = base_tu * 1.2
        econ_cu = base_cu * 0.7
        econ_pr = EconomicEngine.calculate_profit_rate(sp, econ_cu, econ_tu)
        
        if rush_pr > econ_pr:
            logger.info(f"💰 Economic Optimization: RUSH_MODE preferred ($${rush_pr:.2f}/min vs $${econ_pr:.2f}/min)")
            return "RUSH_MODE"
        else:
            logger.info(f"🛡️ Economic Optimization: ECONOMY_MODE preferred ($${econ_pr:.2f}/min vs $${rush_pr:.2f}/min)")
            return "ECONOMY_MODE"

    @staticmethod
    def calculate_spindle_balance(m1_mins: float, m2_mins: float) -> Dict[str, Any]:
        """
        Rp = 60 / max(M1, M2)
        Calculates throughput and unbalance delta.
        """
        bottleneck = max(m1_mins, m2_mins)
        throughput_hr = 60.0 / bottleneck if bottleneck > 0 else 0
        unbalance_delta = abs(m1_mins - m2_mins)
        
        return {
            "throughput_hourly": throughput_hr,
            "unbalance_delta_mins": unbalance_delta,
            "efficiency": min(m1_mins, m2_mins) / bottleneck if bottleneck > 0 else 0
        }

    @staticmethod
    def calculate_machine_gravity(oee_score: float, stability_score: float, current_load: float) -> float:
        """
        Calculates 'Gravity' for job scheduling.
        Critical/Heavy jobs gravitate towards high Gravity nodes.
        """
        # Gravity weights: Stability (40%), OEE (40%), Availability (20%)
        availability = (100.0 - current_load) / 100.0
        gravity = (oee_score * 0.4) + (stability_score * 0.4) + (availability * 100.0 * 0.2)
        return gravity

if __name__ == "__main__":
    # Test PR Optimization
    req = {"sale_price": 500.0, "base_unit_cost": 200.0, "base_unit_time": 60.0}
    mode = EconomicEngine.optimize_mode(req, {})
    
    # Test Spindle Balancing
    balance = EconomicEngine.calculate_spindle_balance(12.5, 8.2)
    print(f"Spindle Balance: {balance}")
