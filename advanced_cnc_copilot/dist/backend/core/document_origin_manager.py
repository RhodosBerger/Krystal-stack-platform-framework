"""
Document Provenance Manager 📜🔍
Responsibility:
1. Generate unique Origin UIDs for every manufacturing intent.
2. Link G-code blocks to their birth intent and evolution state.
3. Provide an audit trail for forensic part analysis.
"""
import hashlib
import json
import time
import redis
import os
from datetime import datetime
from typing import Dict, Any, Optional

class DocumentProvenanceManager:
    def __init__(self):
        self.redis_url = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
        try:
            self.redis = redis.Redis.from_url(self.redis_url, decode_responses=True)
        except:
            self.redis = None

    def create_origin_tag(self, intent_payload: Dict[str, Any], evolution_id: str = "GEN_0") -> str:
        """
        Creates a unique SHA-256 tag for a specific intent/evolution state.
        """
        payload_str = json.dumps(intent_payload, sort_keys=True)
        timestamp = datetime.now().isoformat()
        raw_string = f"{payload_str}|{evolution_id}|{timestamp}"
        
        origin_uid = hashlib.sha256(raw_string.encode()).hexdigest()
        
        # Store metadata in Redis for retrieval
        if self.redis:
            self.redis.hset(f"provenance:{origin_uid}", mapping={
                "intent": payload_str,
                "evolution_id": evolution_id,
                "timestamp": timestamp,
                "status": "SEALED"
            })
            # Link origin to a specific job/program
            if "job_id" in intent_payload:
                self.redis.set(f"program_origin:{intent_payload['job_id']}", origin_uid)
                
        return origin_uid

    def get_origin_details(self, origin_uid: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the full birth details of a program via its UID.
        """
        if not self.redis: return None
        data = self.redis.hgetall(f"provenance:{origin_uid}")
        if data:
            data['intent'] = json.loads(data['intent'])
            return data
        return None

    def fetch_by_job_id(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves provenance data using a job/program ID.
        """
        if not self.redis: return None
        uid = self.redis.get(f"program_origin:{job_id}")
        if uid:
            return self.get_origin_details(uid)
        return None

# Global Instance
provenance_manager = DocumentProvenanceManager()
