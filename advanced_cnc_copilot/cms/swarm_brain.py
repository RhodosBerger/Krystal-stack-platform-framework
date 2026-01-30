#!/usr/bin/env python3
"""
The Swarm Brain (Swarm Intelligence)
Handles multi-machine job distribution and cross-site migration.
Theory: Gravitational Scheduling (OEE-biased routing).
"""

import logging
import asyncio
from typing import Dict, Any, List
from cms.message_bus import global_bus, Message
from cms.economic_engine import EconomicEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SWARM] - %(message)s')
logger = logging.getLogger(__name__)

class SwarmBrain:
    """
    Fleet-wide coordination engine.
    Balancing workload across 'Gravity Nodes'.
    """
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.agent_id = "SWARM_BRAIN"
        self.migration_threshold = 0.3 # If load difference > 30%, migrate
        
    async def start(self):
        """Monitor the global registry for unbalance."""
        logger.info("🐝 Swarm Brain (Coordination Engine) Active.")
        while True:
            await self.rebalance_fleet()
            await asyncio.sleep(10) # Rebalance every 10s

    async def rebalance_fleet(self):
        """Analyze fleet load and migrate jobs from 'Overloaded' to 'High Gravity' nodes."""
        registry = self.orchestrator.machine_registry
        if len(registry) < 2:
            return

        # 1. Calculate Gravity Scores for all nodes
        node_scores = {}
        for mid, data in registry.items():
            oee = data.get("oee", 85.0)
            stability = data.get("neuro", {}).get("serotonin", 80.0) / 100.0
            load = data.get("load", 0.0)
            
            gravity = EconomicEngine.calculate_machine_gravity(oee, stability, load)
            node_scores[mid] = {"gravity": gravity, "load": load}

        # 2. Find Overloaded and Underloaded targets
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1]["gravity"], reverse=True)
        
        # Candidate for Migration: High Load, Low Gravity
        # Candidate for Receipt: Low Load, High Gravity
        
        provider = None
        receiver = None
        
        for mid, score in node_scores.items():
            if score["load"] > 85: # Overloaded
                provider = mid
            if score["load"] < 40: # Underutilized
                receiver = mid
                
        if provider and receiver and provider != receiver:
            await self.migrate_job(provider, receiver)

    async def migrate_job(self, from_node: str, to_node: str):
        """Issues a migration command عبر the bus."""
        logger.warning(f"🔄 Swarm Rebalance: Migrating job from {from_node} -> {to_node} (Gravity Allocation)")
        
        # Publish migration event
        await global_bus.publish("SWARM_MIGRATION", {
            "source": from_node,
            "target": to_node,
            "reason": "LOAD_UNBALANCE_GRAVITY_MATCH",
            "timestamp": asyncio.get_event_loop().time()
        }, sender_id=self.agent_id)
        
        # Mirror to Cortex
        from backend.core.cortex_transmitter import cortex
        cortex.mirror_log("SwarmBrain", f"Migrated task queue from {from_node} to {to_node}", "SWARM_MOVE")

if __name__ == "__main__":
    # Test would require a mock orchestrator
    pass
