"""
Cortex Transmitter & Intent Mirroring Module 👁️
Responsibility:
1. "Mirror" system logs from all components (Orchestrator, Workers) to a central Cortex channel.
2. Create a "Database of Intent" by capturing *why* actions are taken.
3. Transmit "Project Specs" to inform the LLM of the current context.
"""
import logging
import json
import redis
import os
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("CortexTransmitter")

class CortexTransmitter:
    def __init__(self):
        self.redis_url = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
        try:
            self.redis = redis.Redis.from_url(self.redis_url, decode_responses=True)
            logger.info("✅ Cortex Transmitter Connected to Shared Memory")
        except Exception as e:
            logger.error(f"❌ Cortex Transmitter Connection Failed: {e}")
            self.redis = None

    def mirror_log(self, component: str, message: str, level: str = "INFO"):
        """
        Mirror a standard log message to the Cortex Stream.
        Useful for "Programmer Base Information".
        """
        if not self.redis: return
        
        try:
            # ABSOLUTE: Immutable Log Seal
            import hashlib
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "LOG_MIRROR",
                "component": component,
                "level": level,
                "message": message
            }
            
            # Create Hash Seal
            entry_str = json.dumps(entry, sort_keys=True)
            entry["hash"] = hashlib.sha256(entry_str.encode()).hexdigest()
            
            # Push to a capped list (Circular Buffer) for real-time monitoring
            self.redis.lpush("cortex:logs", json.dumps(entry))
            self.redis.ltrim("cortex:logs", 0, 999) # Keep last 1000 logs
        except Exception as e:
            logger.warning(f"⚠️ Cortex Link Lost ({e}). Disabling Transmission.")
            self.redis = None

    def transmit_intent(self, actor: str, action: str, reasoning: str, context: Dict = None):
        """
        Log the INTENT behind an action. This builds the "Database of Intent".
        """
        if not self.redis: return

        try:
            intent_packet = {
                "timestamp": datetime.now().isoformat(),
                "type": "INTENT",
                "actor": actor,    # e.g., "MasterOrchestrator", "DataCreator"
                "action": action,  # e.g., "Dispatch G-Code Job"
                "reasoning": reasoning, # e.g., "User requested rapid prototyping"
                "context": context or {}
            }
            
            # Store in the permanent "Database of Intent" (Redis Stream or List)
            self.redis.lpush("cortex:intent_database", json.dumps(intent_packet))
            
            # Also broadcast as a "Project Spec" update if relevant
            if action == "UPDATE_SPECS":
                self.redis.set("cortex:current_project_specs", json.dumps(context))
        except Exception as e:
            logger.warning(f"⚠️ Cortex Link Lost ({e}). Disabling Transmission.")
            self.redis = None

    def log_outcome(self, job_id: str, success_score: float, details: Dict = None):
        """
        Log the FINAL OUTCOME of a job. Labels success/failure in the intent database.
        """
        if not self.redis: return
        
        outcome_packet = {
            "timestamp": datetime.now().isoformat(),
            "type": "OUTCOME",
            "job_id": job_id,
            "success_score": success_score,
            "details": details or {}
        }
        
        # Store in the permanent "Database of Intent" as a result label
        self.redis.lpush("cortex:intent_database", json.dumps(outcome_packet))
        
        # Also maintain a sorted set of successful presets
        if success_score > 0.7:
            # We store the job_id or the specific config string in a 'proven_presets' set
            # For now, we'll store the job_id as a reference to a successful run
            self.redis.zadd("cortex:proven_presets", {job_id: success_score})

    def get_database_of_intent(self, limit: int = 10) -> list:
        """Retrieve recent intents for LLM context injection"""
        if not self.redis: return []
        items = self.redis.lrange("cortex:intent_database", 0, limit - 1)
        return [json.loads(i) for i in items]

# Global Instance
cortex = CortexTransmitter()
