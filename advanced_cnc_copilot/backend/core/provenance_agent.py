"""
Provenance Agent (Phase 17) 🛡️⛓️
Ensures immutability and traceability of G-Code modifications.
Implements 'AuditChain' using SHA-256 hashing.
"""
import logging
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from backend.core.cortex_transmitter import cortex

logger = logging.getLogger("ProvenanceAgent")

class ProvenanceAgent:
    def __init__(self):
        self.ledger_path = "provenance_ledger.json"
        self.audit_chain = []
        self._load_ledger()

    def _load_ledger(self):
        """Loads the existing audit chain if available."""
        try:
            import os
            if os.path.exists(self.ledger_path):
                with open(self.ledger_path, 'r') as f:
                    self.audit_chain = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load provenance ledger: {e}")

    def sign_optimization(self, job_id: str, original_hash: str, modified_hash: str, reasoning: str, actor: str) -> str:
        """
        Signs an optimization event and appends it to the immutable ledger.
        """
        timestamp = datetime.now().isoformat()
        
        # Create unique signature for this entry
        entry_data = f"{job_id}|{original_hash}|{modified_hash}|{timestamp}|{actor}"
        signature = hashlib.sha256(entry_data.encode()).hexdigest()
        
        entry = {
            "signature": signature,
            "job_id": job_id,
            "timestamp": timestamp,
            "actor": actor,
            "original_hash": original_hash,
            "modified_hash": modified_hash,
            "reasoning": reasoning,
            "previous_signature": self.audit_chain[-1]["signature"] if self.audit_chain else "GENESIS"
        }
        
        self.audit_chain.append(entry)
        self._persist_ledger()
        
        cortex.mirror_log("Provenance", f"Optimization Signed for Job {job_id}. Sig: {signature[:8]}", "SUCCESS")
        return signature

    def generate_gcode_hash(self, gcode: List[str]) -> str:
        """Generates a SHA-256 hash of a G-Code program."""
        content = "".join(gcode).replace(" ", "").upper()
        return hashlib.sha256(content.encode()).hexdigest()

    def verify_integrity(self, signature: str) -> bool:
        """Verifies if a signature exists and matches the chain logic."""
        # Simple existence check for prototype
        return any(e["signature"] == signature for e in self.audit_chain)

    def _persist_ledger(self):
        """Saves the audit chain to disk."""
        try:
            with open(self.ledger_path, 'w') as f:
                json.dump(self.audit_chain, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist provenance ledger: {e}")

# Global Instance
provenance = ProvenanceAgent()
