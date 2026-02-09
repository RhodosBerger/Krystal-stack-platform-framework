#!/usr/bin/env python3
"""
Thermal Agent 🌡️
A specialized member of the Swarm.
Responsibility: Monitor plans for Heat warnings.
"""
import logging
from typing import Dict, Any
from backend.core.math_engine import math_engine
try:
    from backend.cms.message_bus import global_bus, Message
except ImportError:
    from message_bus import global_bus, Message

logger = logging.getLogger("THERMAL_AGENT")

class ThermalAgent:
    def __init__(self):
        self.bus = global_bus
        logger.info("Thermal Agent Online. Monitoring Flux.")

    async def start(self):
        self.bus.subscribe("DRAFT_PLAN", self._analyze_heat)

    async def _analyze_heat(self, msg: Message):
        plan = msg.payload
        logger.info(f"Thermal Agent Scanning Plan: {plan.get('job_id')}")
        
        # Calculate Flux
        rpm = plan.get("rpm", 0)
        feed = plan.get("feed", 0)
        # Assuming Ti for worst case
        flux = math_engine.calculate_thermal_flux(rpm, feed, 7.0) 
        
        risk = "NORMAL"
        if flux > 3000:
             risk = "HIGH_HEAT"
        
        logger.info(f"Report: Flux={flux}W, Risk={risk}")
        
        # Publish Independent Report
        await self.bus.publish(
             channel="THERMAL_REPORT",
             payload={
                 "job_id": plan.get("job_id"),
                 "flux_watts": flux,
                 "risk_level": risk,
                 "cooling_needed": flux > 2000
             },
             sender_id="THERMAL_BOT_v1"
        )

# Standalone Entry Point
if __name__ == "__main__":
    import asyncio
    async def main():
         agent = ThermalAgent()
         await agent.start()
         # Keep alive
         while True:
             await asyncio.sleep(1)
             
    asyncio.run(main())
