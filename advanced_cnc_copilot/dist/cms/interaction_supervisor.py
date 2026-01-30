#!/usr/bin/env python3
"""
The Supervisor: Orchestrates the "Shadow Council".
Brings together the message bus and all parallel sub-systems.
"""

import asyncio
import logging
from typing import Dict, Any

from message_bus import global_bus, Message
from cms_core import global_kb
from auditor_agent import AuditorAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SUPERVISOR] - %(message)s')
logger = logging.getLogger(__name__)

class InteractionSupervisor:
    """
    The main process that initializes the 'Shadow Council'.
    """
    def __init__(self):
        self.bus = global_bus
        self.active = True
        self.auditor = AuditorAgent()
        logger.info("Interaction Supervisor Initialized.")

    async def start(self):
        """Bind listeners and start the event loop."""
        
        # 0. Start Sub-Agents
        await self.auditor.start()
        
        # 1. Listen for User Input
        self.bus.subscribe("USER_INTENT", self._handle_user_intent)
        
        # 2. Listen for "Visualizer" Data (TF Bridge)
        self.bus.subscribe("PART_FEATURES", self._handle_part_features)
        
        # 3. Listen for Validation Results (Auditor)
        self.bus.subscribe("VALIDATION_RESULT", self._handle_validation)

        logger.info("Supervisor is listening on the Nervous System...")
        
        # Keep alive
        while self.active:
            await asyncio.sleep(1)

    async def _handle_user_intent(self, msg: Message):
        """
        When user speaks, trigger the 'Creator' (and parallel 'Auditor').
        """
        intent = msg.payload.get("text", "")
        logger.info(f"Processing User Intent: '{intent}'")
        
        # In a real app, this would trigger the LLM generation.
        # For now, we simulate the 'Draft Plan' being created.
        
        # SIMULATION:
        draft_plan = {
            "action": "machining_strategy", 
            "rpm": 5000, 
            "material": "Titanium" # Assuming this context exists
        }
        
        await self.bus.publish("DRAFT_PLAN", draft_plan, sender_id="CREATOR_MOCK")

    async def _handle_part_features(self, msg: Message):
        """
        When TensorFlow Bridge sees a part.
        """
        features = msg.payload
        logger.info(f"Visualizer reports: {features}")
        # Logic to update context could go here.

    async def _handle_validation(self, msg: Message):
        """
        When Auditor allows/blocks a plan.
        """
        result = msg.payload
        if result.get("status") == "PASS":
            logger.info(">>> PLAN APPROVED. EXECUTING...")
        else:
            logger.warning(f">>> PLAN BLOCKED: {result.get('reason')}")

# Usage
if __name__ == "__main__":
    supervisor = InteractionSupervisor()
    try:
        asyncio.run(supervisor.start())
    except KeyboardInterrupt:
        print("Supervisor Stopped.")
