"""
Marketplace Logic & Ecosystem Management
Phase 7: Community & Sharing.
"""
import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from backend.database.models import MarketplaceComponent, User

logger = logging.getLogger("MARKETPLACE_SERVICE")

class MarketplaceService:
    def list_components(self, db: Session, category: str = None) -> List[MarketplaceComponent]:
        query = db.query(MarketplaceComponent)
        if category:
            query = query.filter(MarketplaceComponent.category == category)
        return query.order_by(MarketplaceComponent.downloads.desc()).all()

    def share_component(self, db: Session, user_id: int, data: Dict[str, Any]) -> MarketplaceComponent:
        component = MarketplaceComponent(
            name=data["name"],
            category=data["category"],
            description=data.get("description", ""),
            payload=data["payload"],
            author_id=user_id,
            version=data.get("version", "1.0.0")
        )
        db.add(component)
        db.commit()
        db.refresh(component)
        logger.info(f"✅ User {user_id} shared new component: {component.name}")
        return component

    def download_component(self, db: Session, component_id: int) -> MarketplaceComponent:
        component = db.query(MarketplaceComponent).filter(MarketplaceComponent.id == component_id).first()
        if component:
            component.downloads += 1
            db.commit()
            db.refresh(component)
        return component

    def rate_component(self, db: Session, component_id: int, score: float):
        component = db.query(MarketplaceComponent).filter(MarketplaceComponent.id == component_id).first()
        if component:
            component.rating_sum += score
            component.rating_count += 1
            db.commit()
        return component

# Global Instance
marketplace_service = MarketplaceService()
