from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import random
import uuid
import logging

# Configure Logging
logger = logging.getLogger("MARKETPLACE")

router = APIRouter()

# Data Models
class CommunityComponent(BaseModel):
    id: str
    name: str
    category: str
    description: str
    author: str
    rating: float
    downloads: int
    version: str
    success_score: float
    stress_tested: bool = False

class DownloadRequest(BaseModel):
    target_node: str = "local_library"

# Mock Database (The Hive)
HIVE_DB = [
    {
        "id": "comp_titan_v1",
        "name": "Titanium Helix Path",
        "category": "GCODE",
        "description": "Optimized trochoidal milling for Grade 5 Ti.",
        "author": "MachinistX",
        "rating": 4.8,
        "downloads": 1240,
        "version": "1.2.0",
        "success_score": 98.5,
        "stress_tested": True
    },
    {
        "id": "comp_cool_logic",
        "name": "Adaptive Cooling Eco",
        "category": "CONFIG",
        "description": "Reduces coolant usage by 40% using thermal analysis.",
        "author": "EcoHacker",
        "rating": 4.5,
        "downloads": 850,
        "version": "2.0.1",
        "success_score": 92.0,
        "stress_tested": True
    },
    {
        "id": "comp_alu_speed",
        "name": "Alu-Speed Demon",
        "category": "GCODE",
        "description": "High-feed aluminum roughing strategy.",
        "author": "SpeedKing",
        "rating": 4.9,
        "downloads": 3100,
        "version": "3.5",
        "success_score": 95.0,
        "stress_tested": False
    },
    {
        "id": "mat_inc_718",
        "name": "Inconel 718 Profile",
        "category": "MATERIAL",
        "description": "stress-strain curves and thermal limits for Inconel.",
        "author": "AeroDyne",
        "rating": 5.0,
        "downloads": 420,
        "version": "1.0",
        "success_score": 99.9,
        "stress_tested": True
    }
]

@router.get("/components", response_model=Dict[str, List[CommunityComponent]])
async def list_components(category: Optional[str] = None):
    """
    List components from the Hive. 
    Supports filtering by category (GCODE, MATERIAL, CONFIG).
    """
    if category and category != "ALL":
        filtered = [c for c in HIVE_DB if c["category"] == category]
        return {"components": filtered}
    return {"components": HIVE_DB}

@router.post("/download/{component_id}")
async def download_component(component_id: str, request: DownloadRequest, background_tasks: BackgroundTasks):
    """
    Simulate downloading a component to the local system.
    """
    component = next((c for c in HIVE_DB if c["id"] == component_id), None)
    if not component:
        raise HTTPException(status_code=404, detail="Component not found in Hive")
    
    # Simulate download process
    background_tasks.add_task(install_component, component)
    
    return {
        "status": "INITIATED",
        "message": f"Downloading {component['name']} to {request.target_node}...",
        "eta_seconds": 2
    }

async def install_component(component: Dict):
    """Background task to 'install' the component."""
    import asyncio
    await asyncio.sleep(2) # Fake unexpected network/install latency
    logger.info(f"✅ SUCCESSFULLY INSTALLED: {component['name']} (v{component['version']})")
    # In a real app, this would write files to disk or update a DB

class MarketplaceService:
    """
    Service layer for internal backend consumption (Orchestrator).
    """
    def list_components(self, category: Optional[str] = None) -> List[Dict]:
        if category and category != "ALL":
             return [c for c in HIVE_DB if c["category"] == category]
        return HIVE_DB

    def get_component(self, component_id: str) -> Optional[Dict]:
        return next((c for c in HIVE_DB if c["id"] == component_id), None)

marketplace_service = MarketplaceService()

