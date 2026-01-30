#!/usr/bin/env python3
"""
The Auditor: Safety & Compliance Agent
Part of the 'Shadow Council'. 
Listens for DRAFT_PLAN events and validates them against safety rules.
"""

import asyncio
import logging
from typing import Dict, Any
from message_bus import global_bus, Message

# Configure logging
logger = logging.getLogger("AUDITOR")

class AuditorAgent:
    """
    The 'Auditor' validates manufacturing intents against physical constraints.
    It acts as the Superego to the Creator's Id.
    """
    def __init__(self):
        self.bus = global_bus
        self.safety_rules = {
            # Max RPM per material (Simplified)
            "Titanium": 4000, 
            "Steel4140": 6000,
            "Aluminum": 12000,
            "Inconel": 2500,
            "Plastic": 15000
        }
        logger.info("Auditor Agent Initialized. Safety Protocols Loaded.")

    async def start(self):
        """Subscribe to relevant channels."""
        self.bus.subscribe("DRAFT_PLAN", self._audit_plan)
        logger.info("Auditor is watching [DRAFT_PLAN]...")

    async def _audit_plan(self, msg: Message):
        """
        Triggered when a Draft Plan is proposed.
        """
        plan = msg.payload
        logger.info(f"Auditing Plan: {plan}")
        
        validation = self._validate_parameters(plan)
        
        # Publish the verdict
        await self.bus.publish(
            channel="VALIDATION_RESULT",
            payload=validation,
            sender_id="AUDITOR_CORE"
        )

    def _validate_parameters(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks physics/safety constraints.
        """
        material = plan.get("material", "Unknown")
        rpm = plan.get("rpm", 0)
        feed = plan.get("feed", 0)
        
        errors = []
        warnings = []
        
        # 1. Material RPM Check
        max_rpm = self.safety_rules.get(material)
        if max_rpm:
            if rpm > max_rpm:
                errors.append(f"RPM {rpm} exceeds safety limit for {material} (Max: {max_rpm}). Risk of fire/tool failure.")
            elif rpm > (max_rpm * 0.9):
                warnings.append(f"RPM {rpm} is near the limit for {material}.")
        else:
            warnings.append(f"Unknown material '{material}'. Using default safety baseline.")
            if rpm > 5000:
                errors.append("RPM > 5000 not allowed for unknown materials.")

        # 2. Feed Rate Sanity Check
        if rpm > 0 and feed > 0:
            # Chip load calculation approximation
            if feed > rpm * 0.5:
                errors.append(f"Feed rate {feed} is disproportionately high for RPM {rpm}. Tool breakage imminent.")
        
        status = "PASS"
        if errors:
            status = "FAIL"
        elif warnings:
            status = "WARNING"
            
        return {
            "status": status,
            "original_plan_id": plan.get("job_id"),
            "errors": errors,
            "warnings": warnings,
            "timestamp": asyncio.get_event_loop().time()
        }
