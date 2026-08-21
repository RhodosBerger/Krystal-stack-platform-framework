#!/usr/bin/env python3
"""
The Intervention Agent (The Scalpel)
Performs micro-adjustments to Feed Rate Override (FRO) to maintain stability.
Addresses Gap 3: Hammer vs. Scalpel.
"""

import asyncio
import logging
from typing import Dict, Any, List
from cms.message_bus import global_bus, Message

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SCALPEL] - %(message)s')
logger = logging.getLogger(__name__)

class InterventionAgent:
    """
    Subtle, micro-adjustment agent.
    Goal: Prevent E-Stops by optimizing Feed Rate Override (FRO).
    """
    def __init__(self):
        self.agent_id = "SCALPEL_AGENT"
        self.current_fro = 1.0  # 100%
        self.stability_threshold = 0.8 # Vibration/Load threshold for intervention
        
    async def start(self):
        """Subscribe to the Nervous System."""
        global_bus.subscribe("TELEMETRY_UPDATE", self.on_telemetry)
        logger.info("🔪 Intervention Agent (The Scalpel) Active.")

    async def on_telemetry(self, msg: Message):
        """Analyze telemetry for stability triggers."""
        telemetry = msg.payload
        load = telemetry.get("load", 0)
        vibration = telemetry.get("vibration", 0)
        
        intervention_needed = False
        reason = ""
        
        # Stability Check
        if load > 90:
            intervention_needed = True
            reason = "EXCESSIVE_SPINDLE_LOAD"
        elif vibration > 0.35:
            intervention_needed = True
            reason = "HIGH_HARMONIC_INSTABILITY"
            
        if intervention_needed:
            await self.issue_scalpel_correction(reason, load, vibration)
        elif self.current_fro < 1.0:
            # Recovery logic: slowly ramp back up if stable
            await self.issue_recovery_correction(load, vibration)

    async def issue_scalpel_correction(self, reason: str, load: float, vibration: float):
        """Proposed a subtle FRO reduction."""
        # Calculate reduction (subtle 5% step)
        reduction = 0.05
        new_fro = max(0.5, self.current_fro - reduction)
        
        if new_fro != self.current_fro:
            self.current_fro = new_fro
            logger.info(f"📍 Scalpel Intervention: Reduced FRO to {int(self.current_fro*100)}% | Reason: {reason}")
            
            # Publish correction to the bus
            await global_bus.publish("INTERNAL_CORRECTION", {
                "type": "FRO_ADJUSTMENT",
                "value": self.current_fro,
                "reason": reason,
                "metrics": {"load": load, "vibration": vibration}
            }, sender_id=self.agent_id)

    async def issue_recovery_correction(self, load: float, vibration: float):
        """Slowly restore performance if conditions are safe."""
        if load < 70 and vibration < 0.2:
            new_fro = min(1.0, self.current_fro + 0.02) # Very slow (2%) recovery
            if new_fro != self.current_fro:
                self.current_fro = new_fro
                logger.info(f"📈 Scalpel Recovery: Restored FRO to {int(self.current_fro*100)}%")
                
                await global_bus.publish("INTERNAL_CORRECTION", {
                    "type": "FRO_RECOVERY",
                    "value": self.current_fro,
                    "reason": "STABILITY_RESTORED"
                }, sender_id=self.agent_id)

if __name__ == "__main__":
    # Local Test
    agent = InterventionAgent()
    asyncio.run(agent.start())
