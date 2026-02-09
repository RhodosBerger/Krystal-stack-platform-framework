"""
LLM Brain Module for FANUC RISE
Refactored to use Augmented LLM Generator System Core.
"""
import os
import logging
from typing import Optional, Dict, List, Any
from enum import Enum

# Import Augmented Core
from backend.core.augmented.llm_processor import LLMProcessor, Provider
from backend.core.augmented.openvino_engine import OpenVINOEngine, CognitiveState
from backend.core.augmented.llama_engine import LlamaCppEngine

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMBrain")

class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama" # Legacy enum, maps to LLAMA_CPP or MOCK
    AUGMENTED = "augmented"

class LLMRouter:
    def __init__(self):
        logger.info("🧠 Initializing Augmented LLM Brain...")
        
        # Initialize Augmented Engines
        # Default to CPU and no specific model path (Mock/Optimization Mode)
        # In a real deployment, these paths would come from config/env
        self.engine = OpenVINOEngine(device=os.getenv("OPENVINO_DEVICE", "CPU"))
        self.processor = LLMProcessor(
            openvino_engine=self.engine, 
            llama_model_path=os.getenv("LLAMA_MODEL_PATH")
        )
        
        logger.info("✅ Augmented Brain Online.")

    def query(self, 
              system_prompt: str, 
              user_prompt: str, 
              provider: Optional[LLMProvider] = None, 
              model: Optional[str] = None,
              temperature: float = 0.7,
              json_mode: bool = False) -> str:
        """
        Proxies the legacy query interface to the new LLMProcessor.
        """
        # Map Provider
        target_provider = None
        if provider == LLMProvider.OPENAI:
            target_provider = Provider.OPENAI
        elif provider == LLMProvider.OLLAMA:
            # Map legacy Ollama request to LlamaCPP or OpenVINO based on state
            target_provider = Provider.LLAMA_CPP 
        
        # Append JSON instruction if needed (since new processor doesn't have strict json_mode param yet)
        final_system = system_prompt
        if json_mode:
             final_system += " Output must be valid JSON."

        try:
            return self.processor.generate(
                prompt=user_prompt,
                system_prompt=final_system,
                provider=target_provider,
                temperature=temperature
            )
        except Exception as e:
            logger.error(f"Brain Error: {e}")
            import json
            return json.dumps({"error": str(e)}) if json_mode else f"Error: {e}"

# Singleton Instance
llm_router = LLMRouter()

class RealTimeLLMAccessor:
    """
    Compatibility wrapper for legacy modules expecting RealTimeLLMAccessor.
    Proxies requests to the central LLMRouter.
    """
    def __init__(self):
        self.router = llm_router

    def generate_completion(self, prompt: str, system_context: str = "You are a CNC Expert.") -> str:
        return self.router.query(
            system_prompt=system_context,
            user_prompt=prompt
        )

    async def generate_json(self, prompt: str, system_context: str) -> Dict:
        response_text = self.router.query(
            system_prompt=system_context,
            user_prompt=prompt,
            json_mode=True
        )
        try:
             import json
             return json.loads(response_text)
        except:
             return {"error": "Failed to parse JSON", "raw": response_text}

