#!/usr/bin/env python3
"""
The Biochemist: Efficiency & Instability Agent
Part of the 'Shadow Council'. 
Monitors 'Dopamine' and Thermal Efficiency.
"""

import logging
import asyncio
from typing import Dict, Any
from backend.cms.message_bus import global_bus, Message

# Configure logging
logger = logging.getLogger("BIOCHEMIST")

class BiochemistAgent:
    """
    Focuses on 'Metabolic' efficiency (Path reduction, Thermal stability).
    Targets the 18.63% path reduction benchmark.
    """
    def __init__(self):
        self.bus = global_bus
        self.path_benchmark = 18.63
        logger.info("Biochemist Agent Initialized. Path Optimization Baselines Loaded.")

    async def start(self):
        """Monitor dopamine and proposed plans."""
        self.bus.subscribe("DRAFT_PLAN", self._evaluate_efficiency)
        logger.info("Biochemist is monitoring efficiency for [DRAFT_PLAN]...")

    async def _evaluate_efficiency(self, msg: Message):
        """
        Calculates efficiency score based on path gains and thermal projections.
        """
        plan = msg.payload
        proposed_gain = plan.get("path_optimization_pct", 5.0)
        thermal_risk = plan.get("thermal_load", 0.3)
        
        # Calculate Vote
        # High Gain + Low Heat = High Vote
        # Deviation from 18.63% benchmark
        gain_score = proposed_gain / self.path_benchmark
        thermal_stability = 1.0 - thermal_risk
        
        composite_score = (gain_score * 0.7) + (thermal_stability * 0.3)
        
        vote = 0.0
        comment = ""
        
        if proposed_gain > self.path_benchmark:
            vote = 1.0
            comment = f"EXCEPTIONAL: Proposed gain {proposed_gain}% exceeds 18.63% Expert baseline."
        elif proposed_gain > 10.0:
            vote = 0.5
            comment = f"ACCEPTABLE: Path efficiency ({proposed_gain}%) is industrially competitive."
        else:
            vote = -0.5
            comment = f"SUBOPTIMAL: Path optimization ({proposed_gain}%) is lower than 18.63% baseline."

        await self.bus.publish(
            channel="VOTE_BIOCHEMIST",
            payload={
                "job_id": plan.get("job_id"),
                "vote": vote,
                "comment": comment,
                "efficiency_score": composite_score,
                "thermal_status": "COOL" if thermal_risk < 0.4 else "STRESS"
            },
            sender_id="BIOCHEMIST"
        )
