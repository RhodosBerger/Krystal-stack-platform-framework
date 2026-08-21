"""
LLM Brain Module for FANUC RISE
Hybrid Architecture: OpenAI (Primary) + Ollama (Fallback/Local)
"""
import os
import time
import requests
import json
from typing import Optional, Dict, List, Any
from enum import Enum
from openai import OpenAI
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMBrain")

class LLMProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

class LLMRouter:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Debug Logging
        logger.info(f"LLM Brain Initialized")
        logger.info(f"OLLAMA_BASE_URL: {self.ollama_base_url}")
        logger.info(f"OpenAI Key Present: {bool(self.openai_api_key)}")

        # Handle Template Placeholder
        if self.openai_api_key and "your_openai_key" in self.openai_api_key:
            logger.warning("Found placeholder OpenAI Key. Disabling OpenAI.")
            self.openai_api_key = None
        
        # Initialize OpenAI Client
        if self.openai_api_key:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            logger.warning("OpenAI API Key not found. Running in Local Mode (Ollama) only.")

        self.latency_threshold_ms = 2000 # Switch to local if OpenAI is slow (not implemented in v1, placeholder)
        self.cost_optimization_mode = False

    def query(self, 
              system_prompt: str, 
              user_prompt: str, 
              provider: Optional[LLMProvider] = None, 
              model: Optional[str] = None,
              temperature: float = 0.7,
              json_mode: bool = False) -> str:
        """
        Smart routing query method.
        If provider is not specified, it defaults to OpenAI, falling back to Ollama on error.
        """
        
        # Determine Provider
        if provider:
            target_provider = provider
        elif self.openai_client:
            target_provider = LLMProvider.OPENAI
        else:
            target_provider = LLMProvider.OLLAMA

        try:
            if target_provider == LLMProvider.OPENAI:
                return self._query_openai(system_prompt, user_prompt, model, temperature, json_mode)
            else:
                return self._query_ollama(system_prompt, user_prompt, model, temperature, json_mode)

        except Exception as e:
            logger.error(f"Error querying {target_provider.value}: {e}")
            
            # Automatic Fallback Chain
            if target_provider == LLMProvider.OPENAI:
                logger.info("⚠️ OpenAI failed. Falling back to Ollama...")
                try:
                    return self._query_ollama(system_prompt, user_prompt, None, temperature, json_mode)
                except Exception as ollama_e:
                    logger.error(f"❌ Ollama also failed: {ollama_e}")
                    return self._query_mock(system_prompt, user_prompt, json_mode)
            else:
                logger.error(f"❌ Ollama failed. Falling back to Mock.")
                return self._query_mock(system_prompt, user_prompt, json_mode)

    def _query_mock(self, system_prompt, user_prompt, json_mode):
        """
        Simulation Mode: Returns realistic static data when AI is offline.
        """
        logger.warning(f"🤖 ENGAGING SIMULATION MODE (Mock LLM) for: {user_prompt[:50]}...")
        
        # Simulate thinking time
        import time
        time.sleep(1.5)

        if "Generate G-Code" in user_prompt or "G-Code" in system_prompt:
            if json_mode:
                 return json.dumps({
                    "intent": "GENERATE_GCODE",
                    "parameters": {"speed": 1000, "feed": 500},
                    "explanation": "Simulated AI: Generated optimal path for Aluminum 6061."
                })
            return "(SIMULATED G-CODE)\n%O1000\nN10 G90 G21\nN20 G00 X0 Y0\nN30 M03 S3000\nN40 G01 Z-5 F500\nN50 M30\n%"
        
        if json_mode:
            return json.dumps({"response": "Simulation Mode: AI subsystem is offline, but I received your request."})
        
        return "Simulation Mode: I am a fallback response. Connect OpenAI or Ollama for real intelligence."

    def _query_openai(self, system_prompt, user_prompt, model, temperature, json_mode):
        if not self.openai_client:
             raise Exception("OpenAI Client not initialized (Invalid Key)")
        
        if not model:
            model = "gpt-4-turbo-preview"
        
        response_format = {"type": "json_object"} if json_mode else None
        
        # Retry Logic (3 Attempts)
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    response_format=response_format
                )
                duration = (time.time() - start_time) * 1000
                logger.info(f"OpenAI Query Time: {duration:.2f}ms")
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"OpenAI Attempt {attempt+1}/{max_retries} failed: {e}")
                last_error = e
                time.sleep(2) # Simple backoff
        
        raise last_error

    def _query_ollama(self, system_prompt, user_prompt, model, temperature, json_mode):
        if not model:
            model = "llama3" # Or "mistral"
            
        url = f"{self.ollama_base_url}/api/chat"
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        if json_mode:
            payload["format"] = "json"

        # Retry Logic (3 Attempts)
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = requests.post(url, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                duration = (time.time() - start_time) * 1000
                logger.info(f"Ollama Query Time: {duration:.2f}ms")
                return result['message']['content']
            except (requests.exceptions.RequestException, KeyError) as e:
                logger.warning(f"Ollama Attempt {attempt+1}/{max_retries} failed: {e}")
                last_error = e
                time.sleep(2)
                
        # If all retries fail, raise error to trigger Fallback in main query() method
        logger.error(f"Ollama Connection Error after {max_retries} attempts.")
        raise Exception("Both OpenAI and Ollama failed.")

# Singleton Instance
llm_router = LLMRouter()
