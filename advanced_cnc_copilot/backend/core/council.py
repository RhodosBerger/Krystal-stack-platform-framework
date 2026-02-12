"""
The Council Chamber
Where the Agents meet to decide the fate of the machine.
"""

import logging
from typing import Dict, Tuple

from backend.core.agents import AuditorAgent, BiochemistAgent, ObserverAgent

logger = logging.getLogger("SHADOW_COUNCIL")

class ShadowCouncil:
    def __init__(self):
        self.auditor = AuditorAgent()
        self.biochemist = BiochemistAgent()
        self.observer = ObserverAgent()
        
        logger.info("The Shadow Council is in session.")

    def consult_council(self, telemetry: Dict, proposed_action: str) -> Tuple[str, float]:
        """
        The Voting Process.
        Returns: (Decision, ConfidenceScore)
        Decision: APPROVE / REJECT
        """
        
        # 1. 🕵️ The Auditor VETO Check
        if not self.auditor.vote(telemetry, proposed_action):
            logger.info("❌ AUDITOR VETOED the action.")
            return ("REJECT", 1.0)

        # 2. Gather Float Votes
        bio_vote = self.biochemist.vote(telemetry, proposed_action) # 0.0 to 1.0 (Excitement)
        obs_vote = self.observer.vote(telemetry, proposed_action)   # -1.0 to 1.0 (Direction)
        
        # Normalize Observer Vote for Weighted Score
        # We handle Observer differently depending on action
        # If action is SPEED_UP, Observer's negative vote (Throttle) counts heavily against it.
        
        # 3. Calculate Weighted Score
        # Formula: (Biochemist * 0.6) + (Observer_Impact * 0.4)
        
        # Interpretation of Observer Vote for "INCREASE_SPEED"
        # If Obs = -1.0 (Throttle), it opposes speed.
        # If Obs = 1.0 (Speed Up), it supports speed.
        
        obs_impact = 0.0
        if proposed_action == "INCREASE_SPEED":
            obs_impact = obs_vote # Direct mapping
        elif proposed_action == "THROTTLE":
            obs_impact = -obs_vote # Inverse mapping (If Obs wants Throttle (-1), it approves Throttle action)

        weighted_score = (bio_vote * 0.6) + (obs_impact * 0.4)
        
        logger.info(f"Council Votes | Bio: {bio_vote:.2f} | Obs: {obs_vote:.2f} | Weighted: {weighted_score:.2f}")

        # 4. Final Verdict
        if weighted_score > 0.5:
             return ("APPROVE", weighted_score)
        else:
             return ("REJECT", weighted_score)


    async def start_session(self):
        """
        Starts the active session for all council members.
        """
        logger.info("Starting Council Session...")
        await self.auditor.start()
        # Other agents might have async start methods later
        logger.info("Council Session Active.")

# Global Instance
council = ShadowCouncil()

