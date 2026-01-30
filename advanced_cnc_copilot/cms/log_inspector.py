#!/usr/bin/env python3
"""
Log Inspector Agent
Uses the LLM to analyze 'log_transformer' outputs and generate shift reports.
"""

import json
import logging
from typing import Dict, List
from backend.core.llm_brain import llm_router
from cms.log_transformer import LogTransformer

logger = logging.getLogger("LOG_INSPECTOR")

class LogInspector:
    def __init__(self):
        self.transformer = LogTransformer()
        logger.info("Log Inspector Initialized.")

    def analyze_recent_errors(self) -> str:
        """
        Reads recent technical logs and asks LLM for a root cause analysis.
        """
        # 1. Get stats and raw log sample
        stats = self.transformer.aggregate_stats()
        
        # Read last 20 lines of technical log
        try:
            with open("logs/technical.jsonl", "r") as f:
                lines = f.readlines()[-20:]
        except FileNotFoundError:
            return "No logs found to analyze."

        log_snippet = "".join(lines)
        
        prompt = f"""
        You are a CNC Maintenance Expert. Analyze these machine logs and statistics.
        
        Statistics:
        {json.dumps(stats, indent=2)}
        
        Recent Logs:
        {log_snippet}
        
        Task:
        1. Identify the primary error pattern.
        2. Correlate RPM/Load with the errors.
        3. Recommend a specific maintenance action (e.g., "Check Spindle Bearings", "Reduce Feed Rate").
        
        Output Format: JSON with keys 'analysis' and 'recommendation'.
        """
        
        try:
            response = llm_router.query(
                system_prompt="You are a simplified CNC Diagnostic AI.",
                user_prompt=prompt,
                json_mode=True
            )
            return response
        except Exception as e:
            logger.error(f"LLM Analysis Failed: {e}")
            return json.dumps({"error": str(e)})

    def generate_shift_report(self) -> str:
        """
        Generates a human-readable summary for the Shift Manager.
        """
        stats = self.transformer.aggregate_stats()
        
        prompt = f"""
        Generate a Shift Report based on these metrics:
        {json.dumps(stats, indent=2)}
        
        Tone: Professional, Concise.
        Include:
        - Overall Health Score (0-100) estimated from warnings/errors.
        - Key efficiency metrics.
        - One "Win" (what went well) and one "Risk".
        """
        
        return llm_router.query(
            system_prompt="You are a Manufacturing Plant Manager AI.",
            user_prompt=prompt
        )

# Global Instance
log_inspector = LogInspector()
