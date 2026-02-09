#!/usr/bin/env python3
"""
The Supervisor: Orchestrates the "Shadow Council".
Brings together the message bus and all parallel sub-systems.
"""

import asyncio
import logging
from typing import Dict, Any

from backend.cms.message_bus import global_bus, Message
from backend.cms.cms_core import global_kb
from backend.cms.auditor_agent import AuditorAgent
from backend.cms.biochemist_agent import BiochemistAgent
from backend.cms.observer_agent import ObserverAgent
from backend.cms.dopamine_engine import dopamine_engine
from backend.cms.vision_cortex import vision_cortex

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
        
        # Initialize The Council members
        self.auditor = AuditorAgent()
        self.biochemist = BiochemistAgent()
        self.observer = ObserverAgent()
        
        # Track pending votes for consensus
        self.pending_votes = {} # job_id -> {votes: [], deadline: int}
        
        logger.info("Interaction Supervisor Initialized. Shadow Council members assembled.")

    async def start(self):
        """Bind listeners and start the event loop."""
        
        # 0. Start Sub-Agents (Activating Listeners)
        await self.auditor.start()
        await self.biochemist.start()
        await self.observer.start()
        
        # 1. Listen for User Input
        self.bus.subscribe("USER_INTENT", self._handle_user_intent)
        
        # 2. Listen for Validation Results and Votes
        self.bus.subscribe("VALIDATION_RESULT", self._handle_auditor_vote)
        self.bus.subscribe("VOTE_BIOCHEMIST", self._handle_biochemist_vote)
        self.bus.subscribe("VOTE_OBSERVER", self._handle_observer_vote)
        self.bus.subscribe("THERMAL_REPORT", self._handle_thermal_report)
        self.bus.subscribe("THERMAL_REPORT", self._handle_thermal_report) # Thermal Bot Vote

        logger.info("Supervisor is listening for Council consensus...")
        
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

    async def _handle_auditor_vote(self, msg: Message):
        await self._add_vote(msg.payload.get("original_plan_id"), "AUDITOR", msg.payload)

    async def _handle_biochemist_vote(self, msg: Message):
        await self._add_vote(msg.payload.get("job_id"), "BIOCHEMIST", msg.payload)

    async def _handle_observer_vote(self, msg: Message):
        await self._add_vote(msg.payload.get("job_id"), "OBSERVER", msg.payload)

    async def _add_vote(self, job_id: str, member: str, payload: Dict):
        """Aggregates votes until consensus is reached."""
        if not job_id: return
        
        if job_id not in self.pending_votes:
            self.pending_votes[job_id] = {"votes": {}, "timestamp": asyncio.get_event_loop().time()}
        
        self.pending_votes[job_id]["votes"][member] = payload
        
        # Check if all 3 members have voted
        votes = self.pending_votes[job_id]["votes"]
        if len(votes) >= 3: # Now waiting for 3 votes (Auditor, Biochemist, Thermal)
            await self._reach_consensus(job_id)

    async def _handle_thermal_report(self, msg: Message):
        """
        Handle independent report from Thermal Agent.
        Treats it as a Council Vote.
        """
        report = msg.payload
        job_id = report.get("job_id")
        risk = report.get("risk_level")
        
        # Thermal Bot Logic: HIGH_HEAT = Veto (-1.0), NORMAL = Approve (1.0)
        vote = -1.0 if risk == "HIGH_HEAT" else 1.0
        comment = f"THERMAL_BOT: Flux {report.get('flux_watts')}W -> Risk: {risk}"
        
        await self._add_vote(job_id, "THERMAL_BOT", {
            "vote": vote,
            "comment": comment
        })

    async def _reach_consensus(self, job_id: str):
        """
        Phase 2: Shadow Council Consensus Engine.
        Approves if total vote sum > 0 and no Critical Errors (Auditor).
        """
        data = self.pending_votes.pop(job_id)
        votes = data["votes"]
        
        total_score = sum(v.get("vote", 0) for v in votes.values())
        if votes["AUDITOR"].get("status") == "PASS":
             total_score += 1.0 # Bonus for safety pass
             
        # Decision logic
        if total_score >= 1.0:
            logger.info(f"🏆 SHADOW COUNCIL CONSENSUS: JOB {job_id} APPROVED (Score: {total_score:.2f})")
            # Sanitize payload (remove numpy types)
            import json
            def safe_serialize(obj):
                if hasattr(obj, 'item'): return obj.item()
                if hasattr(obj, 'tolist'): return obj.tolist()
                return str(obj)
            
            safe_votes = json.loads(json.dumps(votes, default=safe_serialize))

            # Publish Approval
            await self.bus.publish("PLAN_APPROVED", {
                "job_id": job_id,
                "score": total_score,
                "votes": safe_votes
            }, sender_id="SUPERVISOR")
        else:
            logger.warning(f"❌ SHADOW COUNCIL VETO: JOB {job_id} REJECTED (Score: {total_score:.2f})")
            veto_reasons = []
            for m, v in votes.items():
                if v.get("vote", 0) <= 0:
                    reason = v.get('comment') or v.get('errors')
                    logger.warning(f"   - {m} VETO REASON: {reason}")
                    veto_reasons.append(f"{m}: {reason}")
            
            # Publish Rejection
            await self.bus.publish("PLAN_REJECTED", {
                "job_id": job_id,
                "score": total_score,
                "reasons": veto_reasons
            }, sender_id="SUPERVISOR")

# Usage
if __name__ == "__main__":
    supervisor = InteractionSupervisor()
    try:
        asyncio.run(supervisor.start())
    except KeyboardInterrupt:
        print("Supervisor Stopped.")
