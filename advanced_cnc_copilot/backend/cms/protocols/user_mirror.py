"""
User Mirror Protocol 🪞
Standardizes the capture of "User Will" (Intent) and "Geometry" (Specs).
This protocol ensures that every project has a rigorous Definition of Done.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import json

class UserWill(BaseModel):
    """The Abstract Intent (The 'Why' & 'What')"""
    intent: str            # e.g., "Create a lightweight bracket"
    constraints: List[str] # e.g., ["Must hold 50kg", "No sharp edges"]
    priority: str          # "Speed", "Quality", "Cost"
    timestamp: str = datetime.now().isoformat()

class GeometryDoc(BaseModel):
    """The Concrete Geometry (The 'How')"""
    shape_type: str        # "Prismatic", "Rotational", "Sheet"
    dimensions: Dict[str, float] # e.g., {"length": 100.0, "width": 50.0}
    features: List[str]    # e.g., ["M6 Holes", "Pocket"]
    material: str          # "Aluminum6061"
    tolerance: str         # "+/- 0.1mm"

class MirrorObject(BaseModel):
    """The Unified Reflection"""
    project_id: str
    user_will: UserWill
    geometry: GeometryDoc
    status: str = "DRAFT"

class UserMirrorProtocol:
    def __init__(self):
        pass

    def create_mirror(self, project_id: str, intent_data: Dict, geometry_data: Dict) -> MirrorObject:
        """
        Factory method to instantiate a full User Mirror.
        """
        will = UserWill(**intent_data)
        geo = GeometryDoc(**geometry_data)
        
        mirror = MirrorObject(
            project_id=project_id,
            user_will=will,
            geometry=geo
        )
        return mirror

    def document_geometry(self, mirror: MirrorObject) -> str:
        """
        Generates a formal text manifest of the geometry.
        """
        geo = mirror.geometry
        return f"GEOMETRY MANIFEST // {mirror.project_id}\n" \
               f"Type: {geo.shape_type}\n" \
               f"Material: {geo.material}\n" \
               f"Dims: {geo.dimensions}\n" \
               f"Features: {', '.join(geo.features)}\n" \
               f"Tolerance: {geo.tolerance}"

    def reflect_will(self, mirror: MirrorObject) -> str:
        """
        Generates a reflection of the user's intent.
        """
        will = mirror.user_will
        return f"INTENT REFLECTION // {mirror.project_id}\n" \
               f"Goal: {will.intent}\n" \
               f"Constraints: {will.constraints}\n" \
               f"Priority: {will.priority}"

# Global Instance
mirror_protocol = UserMirrorProtocol()
