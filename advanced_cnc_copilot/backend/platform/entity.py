"""
Platform Entity Base Class 🧱
Responsibility:
1. Define coreAttributes shared by all platform-managed entities.
2. Lifecycle management (CREATED, ACTIVE, IN_PIPELINE, COMPLETED, ARCHIVED).
"""
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timezone

class EntityStatus:
    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    IN_PIPELINE = "IN_PIPELINE"
    COMPLETED = "COMPLETED"
    ARCHIVED = "ARCHIVED"
    ERROR = "ERROR"

class PlatformEntity:
    def __init__(self, entity_type: str, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.id = f"PE-{uuid.uuid4().hex[:8].upper()}"
        self.entity_type = entity_type
        self.name = name
        self.status = EntityStatus.CREATED
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at
        self.metadata = metadata or {}
        self.lifecycle_log = [{"status": self.status, "timestamp": self.created_at}]

    def transition(self, new_status: str):
        """
        Moves the entity to a new lifecycle stage.
        """
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.lifecycle_log.append({"status": new_status, "timestamp": self.updated_at})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "lifecycle_log": self.lifecycle_log
        }
