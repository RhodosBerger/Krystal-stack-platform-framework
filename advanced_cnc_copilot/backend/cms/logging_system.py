#!/usr/bin/env python3
"""
MULTI-LEVEL LOGGING SYSTEM
Organized logs for every layer of user intent.

Layers:
1. DEVELOPER: Full debug traces, stack dumps
2. TECHNICAL: System events, performance metrics
3. OPERATOR: Human-readable warnings/errors only
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any
from enum import Enum

class UserIntent(Enum):
    DEVELOPER = "developer"
    TECHNICAL = "technical"
    OPERATOR = "operator"

class MultiLevelLogger:
    def __init__(self, base_dir="logs"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
        # Create separate loggers for each intent level
        self.loggers = {}
        self._setup_loggers()

    def _setup_loggers(self):
        """Configure handlers for each user intent level."""
        
        # 1. DEVELOPER LOGGER (Everything, including DEBUG)
        dev_logger = logging.getLogger("fanuc_rise.developer")
        dev_logger.setLevel(logging.DEBUG)
        dev_handler = logging.FileHandler(f"{self.base_dir}/developer.log")
        dev_handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
        ))
        dev_logger.addHandler(dev_handler)
        self.loggers[UserIntent.DEVELOPER] = dev_logger

        # 2. TECHNICAL LOGGER (INFO and above, JSON format for parsing)
        tech_logger = logging.getLogger("fanuc_rise.technical")
        tech_logger.setLevel(logging.INFO)
        tech_handler = logging.FileHandler(f"{self.base_dir}/technical.jsonl")
        tech_handler.setFormatter(logging.Formatter('%(message)s'))
        tech_logger.addHandler(tech_handler)
        self.loggers[UserIntent.TECHNICAL] = tech_logger

        # 3. OPERATOR LOGGER (WARNING/ERROR only, ultra-readable)
        op_logger = logging.getLogger("fanuc_rise.operator")
        op_logger.setLevel(logging.WARNING)
        op_handler = logging.FileHandler(f"{self.base_dir}/operator.txt")
        op_handler.setFormatter(logging.Formatter(
            '⚠️  %(asctime)s - %(message)s', datefmt='%I:%M %p'
        ))
        op_logger.addHandler(op_handler)
        self.loggers[UserIntent.OPERATOR] = op_logger

    def log(self, level: str, message: str, context: Dict[str, Any] = None):
        """
        Universal log method that dispatches to all appropriate levels.
        """
        context = context or {}
        
        # Developer: Gets everything
        dev_logger = self.loggers[UserIntent.DEVELOPER]
        getattr(dev_logger, level.lower())(f"{message} | Context: {context}")

        # Technical: Gets structured JSON
        if level.upper() in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
            tech_logger = self.loggers[UserIntent.TECHNICAL]
            tech_payload = {
                "timestamp": datetime.now().isoformat(),
                "level": level.upper(),
                "message": message,
                "context": context
            }
            tech_logger.info(json.dumps(tech_payload))

        # Operator: Gets human-friendly warnings/errors only
        if level.upper() in ["WARNING", "ERROR", "CRITICAL"]:
            op_logger = self.loggers[UserIntent.OPERATOR]
            human_msg = self._humanize(message, context)
            getattr(op_logger, level.lower())(human_msg)

    def _humanize(self, message: str, context: Dict[str, Any]) -> str:
        """
        Convert technical message to operator-friendly language.
        """
        # Simple translation mapping
        translations = {
            "load > 90": "⚠️ Machine is working too hard! Slow down feed.",
            "vibration > 0.1": "⚠️ Tool is chattering! Check tool sharpness.",
            "tool_life < 0.1": "⚠️ Tool almost worn out! Replace soon.",
        }
        
        for key, friendly in translations.items():
            if key in message.lower():
                return friendly
                
        # Fallback: return original with emojis
        if "error" in message.lower():
            return f"❌ {message}"
        return f"⚠️ {message}"

# Global instance
logger = MultiLevelLogger()

# Convenience functions
def log_debug(msg, **ctx):
    logger.log("debug", msg, ctx)

def log_info(msg, **ctx):
    logger.log("info", msg, ctx)

def log_warning(msg, **ctx):
    logger.log("warning", msg, ctx)

def log_error(msg, **ctx):
    logger.log("error", msg, ctx)

# Usage Example
if __name__ == "__main__":
    log_debug("Starting sensory cortex", module="hal_fanuc")
    log_info("Machine telemetry received", rpm=12000, load=75)
    log_warning("High load detected", load=95, threshold=90)
    log_error("Tool breakage detected", tool_id="T03", reason="Excessive vibration")
    
    print("\n✅ Logs written to:")
    print("  - logs/developer.log (Full debug)")
    print("  - logs/technical.jsonl (Structured JSON)")
    print("  - logs/operator.txt (Human-readable)")
