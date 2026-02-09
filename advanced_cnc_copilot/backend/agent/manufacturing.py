"""
Manufacturing Agent
Specialized Cognitive Agent for CNC and Factory Operations.
"""

import logging
import json
from backend.core.augmented.openvino_engine import CognitiveState 
from backend.core.augmented.llm_processor import Provider
from backend.agent.core import CognitiveAgent
from backend.agent.tools.manufacturing_tools import RoboticsTool, SimulationTool, QATool

logger = logging.getLogger("ManufacturingAgent")

class ManufacturingAgent(CognitiveAgent):
    """
    Adapts manufacturing process based on cognitive state.
    """
    
    def _infer_state(self, task: str) -> CognitiveState:
        task_lower = task.lower()
        if any(w in task_lower for w in ["plan", "process", "strategy", "workflow", "optimize global"]):
             return CognitiveState.ALPHA
        elif any(w in task_lower for w in ["stop", "halt", "monitor", "dashboard", "status"]):
             return CognitiveState.GAMMA # Gamma is High-Alert/Monitoring
        return CognitiveState.BETA # Default: Generation/Calculation (G-Code)

    def _execute_alpha(self, task: str) -> str:
        """
        ALPHA: Process Planner / Architect
        """
        logger.info("ACTING AS: Manufacturing Planner (Alpha)")
        system_prompt = "You are a Manufacturing Engineer. Create a high-level process plan."
        
        plan = self.processor.generate(task, system_prompt=system_prompt, provider=Provider.OPENAI)
        return f"## Process Plan\n{plan}"

    def _execute_beta(self, task: str) -> str:
        """
        BETA: G-Code Generator / Calculator
        """
        logger.info("ACTING AS: G-Code Generator (Beta)")
        
        # 1. Generate G-Code
        system_prompt = "You are a CNC Programmer. Output G-Code only."
        gcode = self.processor.generate(task, system_prompt=system_prompt, provider=Provider.OPENAI)
        
        # 2. Tool: QA Check
        logger.info("Running QA Check on generated code...")
        qa_result = QATool.inspect_gcode(gcode)
        
        return f"```gcode\n{gcode}\n```\n**QA Status**: {qa_result.content}"

    def _execute_gamma(self, task: str) -> str:
        """
        GAMMA: Real-time Monitoring / Quick Reaction
        Uses Local LLM or OpenVINO for low latency.
        """
        logger.info("ACTING AS: Shop Floor Monitor (Gamma)")
        
        # For GAMMA, we might just return a quick status check or simple command
        # Using LlamaCPP for speed/privacy
        response = self.processor.generate(task, system_prompt="You are a minimalist operator interface.", provider=Provider.LLAMA_CPP)
        return response
