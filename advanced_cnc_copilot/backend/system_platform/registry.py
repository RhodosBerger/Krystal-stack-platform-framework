"""
Platform Registry 📦
Responsibility:
1. Central registry for all PlatformEntities.
2. Provides CRUD operations for platform management.
"""
from typing import Dict, List, Optional
from backend.system_platform.entity import PlatformEntity

class PlatformRegistry:
    def __init__(self):
        self._entities: Dict[str, PlatformEntity] = {}

    def register(self, entity: PlatformEntity) -> str:
        self._entities[entity.id] = entity
        return entity.id

    def get(self, entity_id: str) -> Optional[PlatformEntity]:
        return self._entities.get(entity_id)

    def list_all(self) -> List[PlatformEntity]:
        return list(self._entities.values())

    def remove(self, entity_id: str) -> bool:
        if entity_id in self._entities:
            del self._entities[entity_id]
            return True
        return False

# Global registry instance
platform_registry = PlatformRegistry()
