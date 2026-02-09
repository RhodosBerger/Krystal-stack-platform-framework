"""
OpenLLaMA / Llama.cpp Engine Module
Provides local inference using quantized GGUF models.
"""

import os
import logging
import time
from typing import Optional, Dict, Any, List, Union

# Try importing llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

logger = logging.getLogger("LlamaCppEngine")

class LlamaCppEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        
        if LLAMA_CPP_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                logger.info(f"Loading Llama model from {model_path}...")
                # Initialize Llama model (context window 2048 default, can be adjusted)
                self.model = Llama(model_path=model_path, n_ctx=2048, verbose=False)
                logger.info("Llama model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Llama model: {e}")
                self.model = None
        else:
            if not LLAMA_CPP_AVAILABLE:
                logger.warning("llama-cpp-python not installed. Running in Mock Mode.")
            elif not model_path:
                logger.info("No model path provided. Running in Mock Mode.")
            elif not os.path.exists(model_path):
                logger.warning(f"Model path {model_path} does not exist. Running in Mock Mode.")

    def generate(self, 
                 prompt: str, 
                 max_tokens: int = 128, 
                 temperature: float = 0.7, 
                 top_k: int = 40,
                 stop: Optional[List[str]] = None) -> str:
        """
        Generates text using the local model or mock fallback.
        """
        if self.model:
            try:
                output = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    stop=stop or ["User:", "\n\n"]
                )
                return output['choices'][0]['text']
            except Exception as e:
                logger.error(f"Llama inference failed: {e}")
                return self._mock_generate(prompt)
        else:
            return self._mock_generate(prompt)

    def _mock_generate(self, prompt: str) -> str:
        """
        Simulates generation for testing without weights.
        """
        # Simple heuristic to make mock output look relevant
        if "def " in prompt:
             return "    pass  # implementation pending"
        elif "class " in prompt:
             return "    def __init__(self):\n        pass"
        return f"[Llama-Mock] Processed: {prompt[:20]}..."
