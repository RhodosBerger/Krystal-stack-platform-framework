#!/usr/bin/env python3
"""
The Observer: Vision & Latency Agent
Part of the 'Shadow Council'. 
Monitors Optical Grounding and AI Inference speeds.
"""

import logging
import asyncio
from typing import Dict, Any
from backend.cms.message_bus import global_bus, Message

# Configure logging
logger = logging.getLogger("OBSERVER")

class ObserverAgent:
    """
    Focuses on 'Perception' (Vision accuracy, Inference latency).
    Targets the 90% latency reduction baseline (Neuro-C).
    """
    def __init__(self):
        self.bus = global_bus
        self.latency_target = 90.0
        logger.info("Observer Agent Initialized. Perception Baselines Loaded.")

    async def start(self):
        """Monitor visual events and proposed plans."""
        self.bus.subscribe("DRAFT_PLAN", self._evaluate_perception)
        logger.info("Observer is monitoring perception for [DRAFT_PLAN]...")

    async def _evaluate_perception(self, msg: Message):
        """
        Evaluates the ROI of the proposed vision/inference strategy.
        """
        plan = msg.payload
        ai_benefit = plan.get("inference_latency_reduction_pct", 50.0)
        optical_risk = plan.get("optical_blind_spot_risk", 0.1)
        
        vote = 0.0
        comment = ""
        
        if ai_benefit >= self.latency_target:
            vote = 1.0
            comment = f"ELITE: AI latency reduction {ai_benefit}% meets Neuro-C baseline."
        elif ai_benefit > 60.0:
            vote = 0.5
            comment = f"VALID: Significant inference gains detected ({ai_benefit}%)."
        else:
            vote = 0.0
            comment = f"MEDIOCRE: Perception gain ({ai_benefit}%) is below the 90% threshold."

        if optical_risk > 0.3:
            vote -= 0.5
            comment += " WARNING: High optical blind spot risk."

        await self.bus.publish(
            channel="VOTE_OBSERVER",
            payload={
                "job_id": plan.get("job_id"),
                "vote": vote,
                "comment": comment,
                "perception_confidence": 1.0 - optical_risk
            },
            sender_id="OBSERVER"
        )
