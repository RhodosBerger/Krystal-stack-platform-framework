"""
Cognitive Agent Core
Defines the base CognitiveAgent and the specific CodingAgent.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

# Updated Imports to new location
from backend.core.augmented.openvino_engine import OpenVINOEngine, CognitiveState
from backend.core.augmented.llm_processor import LLMProcessor, Provider
from backend.agent.tools import FileTool  # Assuming tools.py is in same dir or package

logger = logging.getLogger("CognitiveAgent")

class CognitiveAgent(ABC):
    """
    Base class for state-adaptive agents.
    """
    def __init__(self, engine: OpenVINOEngine, processor: LLMProcessor):
        self.engine = engine
        self.processor = processor
        self.context = [] 

    def run_task(self, task_description: str) -> str:
        """
        Executes a task by determining the cognitive state and applying the appropriate strategy.
        """
        # 1. Determine State
        current_state = self._infer_state(task_description)
        logger.info(f"[{self.__class__.__name__}] Cognitive State: {current_state.value.upper()}")

        # 2. Add to Context
        self.context.append({"role": "user", "content": task_description})

        # 3. Strategy Execution
        if current_state == CognitiveState.ALPHA:
            return self._execute_alpha(task_description)
        elif current_state == CognitiveState.BETA:
            return self._execute_beta(task_description)
        else: # GAMMA
            return self._execute_gamma(task_description)

    @abstractmethod
    def _infer_state(self, task: str) -> CognitiveState:
        pass

    @abstractmethod
    def _execute_alpha(self, task: str) -> str:
        pass

    @abstractmethod
    def _execute_beta(self, task: str) -> str:
        pass

    @abstractmethod
    def _execute_gamma(self, task: str) -> str:
        pass

class CodingAgent(CognitiveAgent):
    """
    Specialized agent for Software Development tasks.
    """
    def _infer_state(self, task: str) -> CognitiveState:
        task_lower = task.lower()
        if any(w in task_lower for w in ["design", "plan", "complex", "refactor", "analyze"]):
             return CognitiveState.ALPHA
        elif any(w in task_lower for w in ["fix", "quick", "one-line", "complete", "typo"]):
             return CognitiveState.GAMMA
        return CognitiveState.BETA

    def _execute_alpha(self, task: str) -> str:
        logger.info("ACTING AS: Architect (Alpha)")
        plan_prompt = f"Create a detailed technical design for: {task}"
        plan = self.processor.generate(plan_prompt, system_prompt="You are a Software Architect. Output strictly markdown plans.", provider=Provider.OPENAI)
        return f"# Architectural Plan\n\n{plan}"

    def _execute_beta(self, task: str) -> str:
        logger.info("ACTING AS: Developer (Beta)")
        prompt = f"Write Python code for: {task}"
        code = self.processor.generate(prompt, system_prompt="You are a Python Developer. Output only code.", provider=Provider.OPENAI)
        return code

    def _execute_gamma(self, task: str) -> str:
        logger.info("ACTING AS: Autocompleter (Gamma)")
        code = self.processor.generate(task, system_prompt="Complete this code:", provider=Provider.LLAMA_CPP)
        return code
