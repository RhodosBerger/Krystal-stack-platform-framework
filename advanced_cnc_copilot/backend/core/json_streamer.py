import logging
import time
from typing import Dict, List, Generator, Any
from datetime import datetime

# Fallback import if DopamineEngine not in path
try:
    from backend.cms.services.dopamine_engine import DopamineEngine
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from backend.cms.services.dopamine_engine import DopamineEngine

from backend.core.resource_governor import ResourceGovernor
from backend.integrations.grid_interface import GridInterface

class JSONCodeStreamer:
    """
    Executes 'Advanced Profiles' (JSON) by orchestrating the Dopamine Engine
    to ensure safety and optimization for every segment.
    """
    def __init__(self, dopamine_engine: DopamineEngine):
        self.engine = dopamine_engine
        self.governor = ResourceGovernor() # Initialize the Governor
        self.grid = GridInterface() # Initialize Grid
        self.logger = logging.getLogger("JSONCodeStreamer")
        self.safety_buffer: List[Dict] = [] 

    def execute_profile(self, profile: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        Executes a High-Level Profile.
        Yields status updates or commands to the 'driver' (mock CNC).
        """
        profile_id = profile.get("profile_id", "unknown")
        self.logger.info(f"Starting execution of Profile: {profile_id}")

        for segment in profile.get("segments", []):
            # 0. Govern Logic (Look-Ahead) (Simplified for brevity in diff)
            intensity = self._calculate_intensity(segment)
            self.governor.predict_and_adjust(intensity)

            # 1. Prepare (Optimization)
            optimized_segment = self._optimize_segment(segment)
            
            # 2. Safety Check (Predictive + Grid Simulation)
            risk = self._check_risk(optimized_segment)
            
            if risk == "CRITICAL":
                self.logger.warning(f"Critical Risk in Segment {segment.get('id')}. Aborting.")
                yield {"status": "ABORTED", "reason": "Phantom Trauma Risk"}
                break
            
            # 3. Add to Safety Buffer
            self.safety_buffer.append(optimized_segment)
            
            # 4. Execute (Yield Command)
            start_time = time.time()
            yield {
                "status": "EXECUTING",
                "segment_id": segment.get("id"),
                "command": self._convert_to_command(optimized_segment),
                "neuro_state": self.engine.calculate_current_state("machine_1"),
                "governor_mode": self.governor.current_mode
            }
            
            # Simulate real execution time
            time.sleep(0.01)
            
            # 5. Feedback Loop (Introspection)
            actual_latency = (time.time() - start_time) * 1000 # ms
            success = actual_latency < 50 
            self.governor.feedback(self.governor.current_mode, success, actual_latency)

    def _check_risk(self, segment: Dict) -> str:
        """
        Predictive check using Dopamine Engine AND Grid Simulation.
        """
        # 1. Neural Risk (Dopamine)
        if segment.get("simulated_risk") == "high":
            return "CRITICAL"
            
        # 2. Physical Risk (Grid Simulation)
        grid_status = self.grid.simulate_segment(segment)
        if grid_status in ["COLLISION", "RISK"]:
            self.logger.warning(f"Grid Interface reported: {grid_status}")
            return "CRITICAL"
            
        return "SAFE"

    def _convert_to_command(self, segment: Dict) -> str:
        """
        Converts the JSON segment to a machine command (G-Code or internal format).
        """
        target = segment.get("target", {})
        feed = segment.get("optimized_feed", 1000)
        return f"G01 X{target.get('x',0)} Y{target.get('y',0)} F{feed}"
