#!/usr/bin/env python3
"""
LLM ACTION PARSER
Converts AI Text into Executable CNC Atoms.

Purpose:
To extract structured commands from unstructured LLM responses.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLM_PARSER")

@dataclass
class PendingAction:
    type: str # "SET_RPM", "MOVE_AXIS", "WAIT"
    params: Dict[str, Any]
    confidence: float # 0.0 - 1.0 (Derived from language certainty)
    reasoning: str #Extracted from <thought> tags

class ActionParser:
    def __init__(self):
        # Allowlist of valid commands to prevent hallucinations
        self.valid_commands = {
            "SET_RPM": ["rpm"],
            "FEED_RATE": ["val"],
            "COOLANT": ["state"],
            "MOVE": ["axis", "target"],
            "INJECT_MACRO": ["id", "val"]
        }

    def parse_response(self, llm_text: str) -> List[PendingAction]:
        """
        Extracts actions from a text block.
        Expected format:
        <thought>Reasoning here</thought>
        ACTION: SET_RPM(5000)
        ACTION: COOLANT(ON)
        """
        actions = []
        
        # 1. Extract Reasoning (Context)
        reasoning = "No reasoning provided."
        thought_match = re.search(r"<thought>(.*?)</thought>", llm_text, re.DOTALL)
        if thought_match:
            reasoning = thought_match.group(1).strip()
            
        # 2. Extract Actions (Regex)
        # Matches: ACTION: COMMAND(PARAMS)
        # e.g. ACTION: SET_RPM(5000)
        pattern = r"ACTION:\s*([A-Z_]+)\((.*?)\)"
        matches = re.findall(pattern, llm_text)
        
        for cmd_name, param_str in matches:
            if cmd_name in self.valid_commands:
                parsed_params = self._parse_params(param_str)
                
                # Confidence Estimate (Simple heuristic)
                confidence = 0.9 if "must" in reasoning.lower() else 0.7
                
                action = PendingAction(
                    type=cmd_name,
                    params=parsed_params,
                    confidence=confidence,
                    reasoning=reasoning
                )
                actions.append(action)
                logger.info(f"Parses Action: {cmd_name} {parsed_params}")
            else:
                logger.warning(f"Hallucinated Command Ignored: {cmd_name}")
                
        return actions

    def _parse_params(self, param_str: str) -> Dict[str, Any]:
        """
        Converts '5000' or 'axis=X, target=100' into dict.
        """
        params = {}
        # Simple comma split
        parts = param_str.split(',')
        for p in parts:
            if '=' in p:
                k, v = p.split('=')
                params[k.strip()] = self._cast(v.strip())
            else:
                # Positional arg (fallback)
                params["arg_0"] = self._cast(p.strip())
        return params

    def _cast(self, val: str):
        try:
            if '.' in val: return float(val)
            return int(val)
        except:
            return val

# Usage
if __name__ == "__main__":
    parser = ActionParser()
    text = """
    <thought>
    The spindle load is too high (95%). We must reduce RPM immediately to prevent stall.
    </thought>
    ACTION: SET_RPM(4000)
    ACTION: FEED_RATE(val=1500)
    """
    
    result = parser.parse_response(text)
    print(result)
