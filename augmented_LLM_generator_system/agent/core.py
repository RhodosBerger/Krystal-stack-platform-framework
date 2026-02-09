"""
Coding Agent Core
Implements the Cognitive Coding Agent that adapts strategies based on system state.
"""

import logging
from typing import List, Dict, Optional

from augmented_LLM_generator_system.core.openvino_engine import OpenVINOEngine, CognitiveState
from augmented_LLM_generator_system.core.llm_processor import LLMProcessor, Provider
from augmented_LLM_generator_system.agent.tools import FileTool, LinterTool

logger = logging.getLogger("CodingAgent")

class CodingAgent:
    def __init__(self, engine: OpenVINOEngine, processor: LLMProcessor):
        self.engine = engine
        self.processor = processor
        self.context = [] # Short-term memory (list of dicts)

    def run_task(self, task_description: str) -> str:
        """
        Executes a coding task by determining the cognitive state and applying the appropriate strategy.
        """
        # 1. Determine State (Mock telemetry for the task context)
        # In a real app, this would come from live system monitoring.
        # Here, we infer state intent from the task description keywords.
        current_state = self._infer_state_from_task(task_description)
        logger.info(f"Agent determined Cognitive State: {current_state.value.upper()}")

        # 2. Add to Context
        self.context.append({"role": "user", "content": task_description})

        # 3. Strategy Execution
        if current_state == CognitiveState.ALPHA:
            return self._execute_architect_flow(task_description)
        elif current_state == CognitiveState.BETA:
            return self._execute_developer_flow(task_description)
        else: # GAMMA
            return self._execute_autocomplete_flow(task_description)

    def _infer_state_from_task(self, task: str) -> CognitiveState:
        task_lower = task.lower()
        if any(w in task_lower for w in ["design", "plan", "complex", "refactor", "analyze"]):
             return CognitiveState.ALPHA
        elif any(w in task_lower for w in ["fix", "quick", "one-line", "complete", "typo"]):
             return CognitiveState.GAMMA
        return CognitiveState.BETA # Default

    def _execute_architect_flow(self, task: str) -> str:
        """
        ALPHA STATE: detailed reasoning, plan before code.
        """
        logger.info("ACTING AS: Architect (Alpha)")
        
        # 1. Generate Plan
        plan_prompt = f"Create a detailed technical design for: {task}"
        plan = self.processor.generate(plan_prompt, system_prompt="You are a Software Architect. Output strictly markdown plans.", provider=Provider.OPENAI)
        
        # 2. Reflect (Simulated)
        logger.info("Architect is reviewing the plan...")
        
        return f"# Architectural Plan\n\n{plan}"

    def _execute_developer_flow(self, task: str) -> str:
        """
        BETA STATE: Standard coding loop (Write -> Lint -> Return).
        """
        logger.info("ACTING AS: Developer (Beta)")
        
        prompt = f"Write Python code for: {task}"
        code = self.processor.generate(prompt, system_prompt="You are a Python Developer. Output only code.", provider=Provider.OPENAI)
        
        # Self-Correction Loop
        lint_result = LinterTool.check_syntax(code)
        if not lint_result.success:
            logger.warning(f"Syntax Error detected: {lint_result.content}. Fixing...")
            fix_prompt = f"Fix this syntax error in the code: {lint_result.content}\nCode:\n{code}"
            code = self.processor.generate(fix_prompt, system_prompt="Fix python syntax errors.", provider=Provider.OPENAI)
            
        return code

    def _execute_autocomplete_flow(self, task: str) -> str:
        """
        GAMMA STATE: Fast, raw completion.
        """
        logger.info("ACTING AS: Autocompleter (Gamma)")
        
        # Use LlamaCPP/OpenVINO for speed
        code = self.processor.generate(task, system_prompt="Complete this code:", provider=Provider.LLAMA_CPP)
        return code
