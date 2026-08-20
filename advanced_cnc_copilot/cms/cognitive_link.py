#!/usr/bin/env python3
"""
The Cognitive Link ("Regedit" Logic)
Bridges Perception (Vision/Telemetry) -> Cognition (LLM) -> Action (Registry Update).
CRITICAL: All generated patches MUST pass the Auditor's safety check.
"""

import logging
import json
import asyncio
from typing import Dict, Any

from backend.core.llm_brain import llm_router
from message_bus import global_bus

logger = logging.getLogger("COGNITIVE_LINK")

class CognitiveLink:
    def __init__(self):
        self.bus = global_bus
        logger.info("Cognitive Link (Regedit) Initialized.")

    async def resolve_symptom(self, symptom: str, context: Dict[str, Any]):
        """
        1. Analyze Symptom (e.g. 'Surface Chatter').
        2. Consult LLM for a 'Registry Patch'.
        3. Request Auditor Verification.
        4. Apply Patch (if approved).
        """
        logger.info(f"Resolving Symptom: {symptom}")
        
        # Step 1: Cognition (Think)
        patch_proposal = self._generate_registry_patch(symptom, context)
        
        if not patch_proposal:
            logger.warning("LLM could not generate a valid patch.")
            return

        # Step 2: Governance (Ask for Permission)
        logger.info(f"Proposing Patch: {patch_proposal}")
        
        # We misuse the 'DRAFT_PLAN' channel slightly to get Auditor eyes on this
        # In a full system, we'd use a dedicated 'DRAFT_CONFIG_CHANGE' channel
        audit_request = {
            "job_id": f"PATCH-{symptom[:4].upper()}",
            "action": "CONFIG_CHANGE",
            "material": context.get("material", "Unknown"),
            "proposed_changes": patch_proposal
        }
        
        # Publish for visibility (The Auditor Agent listens to this)
        await self.bus.publish("DRAFT_PLAN", audit_request, sender_id="COGNITIVE_LINK")
        
        # In a real sync flow, we'd wait here. For this prototype, we simulate the "Auto-Approve" 
        # path if the params are within a "safe tuning range".
        
        if self._is_safe_tuning(patch_proposal):
            logger.info("✅ Patch within Safe Tuning Limits. Auto-Applying...")
            self._apply_patch(patch_proposal)
        else:
            logger.warning("⚠️ Patch exceeds safe tuning/auto-limits. Escalating to Human.")

    def _generate_registry_patch(self, symptom: str, context: Dict) -> Dict:
        """
        Asks LLM to tweak system parameters based on the symptom.
        """
        prompt = f"""
        System Symptom: {symptom}
        Context: {json.dumps(context)}
        
        You are the 'Regedit' AI. You have access to these system registers:
        - SYSTEM.SERVO.GAIN (0.5 - 2.0)
        - SYSTEM.SPINDLE.RPM_OFFSET (-500 to +500)
        - SYSTEM.FEED.ADAPTIVE_FACTOR (0.8 - 1.2)
        
        Output a JSON object representing the Registry Patch to fix this symptom.
        Example: {{"SYSTEM.SERVO.GAIN": 0.8}}
        """
        
        try:
            response = llm_router.query(
                system_prompt="You are a CNC Control Systems Engineer.",
                user_prompt=prompt,
                json_mode=True
            )
            return json.loads(response)
        except Exception as e:
            logger.error(f"Cognitive Failure: {e}")
            return {}

    def _is_safe_tuning(self, patch: Dict) -> bool:
        """
        Simple heuristic safety check (The 'Mini-Auditor').
        """
        # Example Rule: Don't increase gain above 1.5 without human sign-off
        if patch.get("SYSTEM.SERVO.GAIN", 1.0) > 1.5:
            return False
            
        # Example Rule: Don't slow down too much (risk of stalling in cut)
        if patch.get("SYSTEM.FEED.ADAPTIVE_FACTOR", 1.0) < 0.5:
            return False
            
        return True

    def _apply_patch(self, patch: Dict):
        """
        Writes the changes to the 'Registry' (dopamine_weights or similar).
        """
        logger.info(f"Writing Registry Keys: {patch}")
        # In a real app, this would update redis or a config file.
        # For prototype, we verify it works by broadcasting the update.
        asyncio.create_task(self.bus.publish("REGISTRY_UPDATE", patch, sender_id="REGEDIT"))

# Global Instance
cognitive_link = CognitiveLink()
