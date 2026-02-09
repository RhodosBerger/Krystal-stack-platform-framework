"""
LLM Processor Module
Routes requests between Cloud APIs (OpenAI) and Local OpenVINO Acceleration.
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any
from enum import Enum

# Import the OpenVINO Engine
from .openvino_engine import OpenVINOEngine, CognitiveState
from .llama_engine import LlamaCppEngine

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM_Processor")

class Provider(Enum):
    OPENAI = "openai"
    OPENVINO = "openvino"
    LLAMA_CPP = "llama_cpp"
    MOCK = "mock"

class LLMProcessor:
    def __init__(self, openvino_engine: OpenVINOEngine, llama_model_path: Optional[str] = None):
        self.engine = openvino_engine
        self.llama = LlamaCppEngine(model_path=llama_model_path)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = None
        
        # Initialize OpenAI if key exists
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI Client Initialized.")
            except ImportError:
                logger.warning("OpenAI library not installed. Cloud fallback disabled.")
        
    def generate(self, 
                 prompt: str, 
                 system_prompt: str = "You are a helpful assistant.", 
                 provider: Optional[Provider] = None,
                 temperature: float = 0.7) -> str:
        """
        Smart generation routing based on cognitive state and availability.
        """
        
        # 1. Determine Provider Strategy
        target_provider = provider
        
        if not target_provider:
            # Strategy: Use OpenVINO for GAMMA (Speed/Local) and OpenAI for ALPHA/BETA (Quality)
            current_state = self.engine.current_state
            
            if current_state == CognitiveState.GAMMA:
                # Prefer LlamaCPP if available/loaded, else OpenVINO/Mock
                if self.llama.model: 
                    target_provider = Provider.LLAMA_CPP
                else:
                    target_provider = Provider.OPENVINO
                logger.info(f"State is GAMMA. Preferring {target_provider.value} for speed.")
            elif self.openai_client:
                target_provider = Provider.OPENAI
                logger.info(f"State is {current_state.value}. Preferring Cloud OpenAI for quality.")
            else:
                 # Fallback chain
                if self.llama.model:
                     target_provider = Provider.LLAMA_CPP
                else:
                     target_provider = Provider.OPENVINO

        # 2. Execute Generation
        try:
            if target_provider == Provider.OPENAI and self.openai_client:
                return self._generate_openai(prompt, system_prompt, temperature)
            elif target_provider == Provider.LLAMA_CPP:
                return self._generate_llama(prompt, system_prompt, temperature)
            elif target_provider == Provider.OPENVINO:
                return self._generate_openvino(prompt, system_prompt, temperature)
            else:
                return self._generate_mock(prompt)
                
        except Exception as e:
            logger.error(f"Generation failed with {target_provider}: {e}")
            logger.info("Falling back to Mock/Simulation.")
            return self._generate_mock(prompt)

    def _generate_openai(self, prompt, system_prompt, temperature):
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content

    def _generate_llama(self, prompt, system_prompt, temperature):
        # LlamaCPP usually expects one prompt string. We format it.
        full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"
        return self.llama.generate(full_prompt, temperature=temperature)

    def _generate_openvino(self, prompt, system_prompt, temperature):
        """
        Uses the OpenVINO Engine (which might be improved with openvino-genai in the future).
        For now, it calls the engine's inference method which might return a mock if models aren't loaded.
        """
        # Contextualize the prompt
        full_prompt = f"{system_prompt}\nUser: {prompt}\nAssistant:"
        
        # In a real implementation, this would stream tokens from the OpenVINO GenAI pipeline
        result = self.engine.infer("genai-llama3-int4", full_prompt)
        
        if isinstance(result, dict) and "tokens" in result:
             return "".join(result["tokens"])
        return "OpenVINO Response: [Data Processing]"

    def _generate_mock(self, prompt):
        time.sleep(1)
        return f"Simulated Response to: {prompt[:30]}..."

