#!/usr/bin/env python3
"""
OPERATION QUEUE
The Dispatch Buffer for CNC Commands.

Purpose:
To hold parsed actions in a safe state until the Semaphore turns Green.
"""

from typing import List, Optional
from llm_action_parser import PendingAction
from signaling_system import TrafficController
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OP_QUEUE")

class OperationQueue:
    def __init__(self):
        self.queue: List[PendingAction] = []
        self.semaphore = TrafficController() # The Safety Guard

    def add_actions(self, actions: List[PendingAction]):
        """
        Ingests parsed actions into the pending queue.
        """
        for count, action in enumerate(actions):
            self.queue.append(action)
            logger.info(f"[QUEUE] Enqueued: {action.type} (Conf: {action.confidence})")
            
        return len(self.queue)

    def process_next(self, current_metrics: dict) -> Optional[PendingAction]:
        """
        Attempts to execute the next item in the queue.
        Checks Semaphore status first.
        """
        if not self.queue:
            return None
            
        # 1. Safety Check (The Gatekeeper)
        signal = self.semaphore.evaluate(current_metrics)
        
        if signal == "RED":
            logger.error(f"[QUEUE] BLOCKED by RED Signal. Holding {len(self.queue)} ops.")
            return None
            
        if signal == "AMBER":
            # Amber Logic: Only allow "Corrective" actions, not "Aggressive" ones?
            # For MVP, we proceed with warning
            logger.warning("[QUEUE] Proceeding with CAUTION (Amber Signal).")

        # 2. Dequeue
        next_op = self.queue.pop(0)
        logger.info(f"[QUEUE] Dispatching: {next_op.type}")
        return next_op

    def clear(self):
        self.queue = []

# Usage
if __name__ == "__main__":
    from llm_action_parser import PendingAction
    
    q = OperationQueue()
    
    # Simulate adding ops
    op1 = PendingAction("SET_RPM", {"arg_0": 5000}, 0.9, "Speed up")
    q.add_actions([op1])
    
    # Process (with good metrics)
    metrics = {"load": 50, "vibration": 0.05}
    q.process_next(metrics)
    
    # Process (with BAD metrics)
    metrics_bad = {"load": 160, "vibration": 0.9} # RED condition
    q.add_actions([op1])
    q.process_next(metrics_bad) # Should fail
