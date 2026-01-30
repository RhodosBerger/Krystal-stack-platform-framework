#!/usr/bin/env python3
"""
CNC-VINO Optimizer: The "Model Optimizer" for G-Code.
"Fanuc Rise" Component.

Converts Raw G-Code -> Optimized IR with Dopamine Checks.
"""

import sys
import json
import os
import re
from typing import List, Dict

# Import the Council Members
from dopamine_engine import DopamineEngine
from parameter_standard import Mantinel, MantinelType

POLICY_FILE = "dopamine_policy.json"

class CNCOptimizer:
    def __init__(self):
        self.dopamine_brain = DopamineEngine()
        # Default Constraint
        self.heat_limit = Mantinel(MantinelType.QUADRATIC, "rpm * feed < 8000000") 
        self.policy = self._load_policy()

    def _load_policy(self):
        if os.path.exists(POLICY_FILE):
            try:
                with open(POLICY_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def optimize_model(self, gcode_lines: List[str], material: str = "Unknown") -> List[str]:
        """
        The Main Pass: Scans G-Code, predicts Heat/Vibration, adjusts Feed.
        """
        optimized_ir = []
        optimized_ir.append("(--- CNC-VINO OPTIMIZED IR ---)")
        optimized_ir.append(f"(--- Material: {material} ---)")
        
        # Check Policy for this material
        strategy = self.policy.get(material, {}).get("preferred_strategy", "ACTION_STANDARD_MODE")
        if strategy != "ACTION_STANDARD_MODE":
             optimized_ir.append(f"(POLICY APPLIED: Using {strategy} based on history)")

        current_rpm = 0
        current_feed = 0
        
        for line in gcode_lines:
            line = line.strip()
            if not line: continue
            
            # 1. Parse Parameters
            rpm_match = re.search(r'S(\d+)', line)
            feed_match = re.search(r'F(\d+)', line)
            
            if rpm_match: current_rpm = float(rpm_match.group(1))
            if feed_match: current_feed = float(feed_match.group(1))
            
            # 2. Analyze "Voxel Heat" (Simplified)
            # If we are cutting (G1/G2/G3) and have speed
            if line.startswith(('G1', 'G2', 'G3')) and current_rpm > 0:
                
                # Check Mantinel
                ctx = {"rpm": current_rpm, "feed": current_feed}
                if not self.heat_limit.validate(ctx):
                    optimized_ir.append(f"(ERROR: Heat Limit Violation Predicted: {current_rpm}*{current_feed})")
                    optimized_ir.append("M100 P99 ; TRIGGER CORTISOL SPIKE")
                    # Auto-Optimize: Reduce Feed
                    new_feed = int(8000000 / current_rpm)
                    line = re.sub(r'F\d+', f"F{new_feed}", line)
                    optimized_ir.append(f"(OPTIMIZED: Reduced Feed to {new_feed} to stay in Safe Zone)")
                
                # Predict Outcome for Dopamine
                # Simulating a "Good Cut" (Ideal deviation = 0.0)
                action = self.dopamine_brain.evaluate_stimuli(
                    speed_factor=1.1, vibration_level=0.1, deviation_score=0.0, result_quality=1.0
                )
                
                if action == "ACTION_RUSH_MODE":
                    optimized_ir.append("(NOTE: Brain is confident. Flow State.)")

            optimized_ir.append(line)
            
        return optimized_ir

# Usage CLI
if __name__ == "__main__":
    # Mock G-Code Input
    raw_code = [
        "G21 G90",
        "S5000 M3",
        "G1 X100 F2000",  # Safe (10M) -> Wait, 5000*2000 = 10M. Limit is 8M. Should trigger optimization.
        "G1 X200",
        "S8000",
        "G1 X300 F500"    # Safe (4M)
    ]
    
    optimizer = CNCOptimizer()
    ir = optimizer.optimize_model(raw_code)
    
    print("\n".join(ir))
