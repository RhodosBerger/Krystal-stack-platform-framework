"""
Knowledge Engine & Presets Module
Responsibility:
1. Load "Seed Data" (Presets) from JSON storage.
2. Provide "Propedeutics" (Educational Content) for topics.
3. Bridge the gap between Raw Data and User Understanding.
"""
import json
import os
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger("KnowledgeEngine")

class KnowledgeEngine:
    def __init__(self):
        self.presets_path = os.path.join(os.path.dirname(__file__), 'presets')
        self.materials_db = self._load_preset('materials.json')
        # Future: self.components_db = self._load_preset('components.json')
        logger.info(f"Loaded {len(self.materials_db)} material presets.")

    def _load_preset(self, filename: str) -> List[Dict]:
        """Load JSON preset file safely"""
        try:
            path = os.path.join(self.presets_path, filename)
            if not os.path.exists(path):
                logger.warning(f"Preset file not found: {path}")
                return []
            
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load preset {filename}: {e}")
            return []

    def get_material_preset(self, material_name: str) -> Optional[Dict]:
        """Find a specific material preset by name"""
        # Fuzzy match or exact match
        for mat in self.materials_db:
            if material_name.lower() in mat['name'].lower():
                return mat
        return None

    def add_verified_preset(self, new_preset: Dict) -> bool:
        """
        2FA Mechanism: Commit verified data to the Persistent Store.
        This is the result of the 'Factor 2' approval.
        """
        try:
            # Check duplicates
            for mat in self.materials_db:
                if mat['name'] == new_preset['name']:
                    logger.warning(f"Material {new_preset['name']} already exists. Updating...")
                    self.materials_db.remove(mat)
            
            self.materials_db.append(new_preset)
            
            # Persist to disk
            path = os.path.join(self.presets_path, 'materials.json')
            with open(path, 'w') as f:
                json.dump(self.materials_db, f, indent=2)
                
            logger.info(f"✅ Verified Preset '{new_preset['name']}' saved to Knowledge Base.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save verified preset: {e}")
            return False

    def generate_propedeutics(self, topic: str) -> str:
        """
        Generate educational content (Propedeutics) for a given topic.
        Uses cached knowledge + LLM synthesis.
        """
        # 1. Check if we have structured data for this topic
        structured_info = ""
        material_data = self.get_material_preset(topic)
        
        if material_data:
            structured_info = f"technical_properties: {json.dumps(material_data, indent=2)}"
        
        # 2. Synthesize Propedeutics (Real LLM)
        from backend.core.llm_brain import llm_router
        
        prompt = f"""
        Generate a 'Propedeutics' (Foundational Guide) for: {topic}
        Context: {structured_info}
        
        Structure:
        1. Concept Definition (What is it?)
        2. Relevance (Why use it in CNC?)
        3. Golden Rules (Best practices)
        
        Format: Markdown.
        """
        
        try:
             return llm_router.query(
                system_prompt="You are a Manufacturing Educator.",
                user_prompt=prompt
             )
        except Exception as e:
             logger.error(f"LLM Generation Failed: {e}")
             return f"Error analyzing {topic}. Please consult the manual."

# Global Instance
knowledge_engine = KnowledgeEngine()
