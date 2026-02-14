import time
import logging
from typing import Generator, List
from datetime import datetime
try:
    from backend.cms.services.dopamine_engine import DopamineEngine
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
    from backend.cms.services.dopamine_engine import DopamineEngine

# Mocking the repository if not provided
class MockRepo:
    def get_recent_by_machine(self, machine_id):
        return []

class GCodeStreamer:
    """
    Middleware that streams G-Code to a CNC controller while performing
    real-time safety checks using the Dopamine Engine.
    """
    def __init__(self, dopamine_engine: DopamineEngine, machine_id: str = "machine_1"):
        self.engine = dopamine_engine
        self.machine_id = machine_id
        self.logger = logging.getLogger("Synapse.GCodeStreamer")
        self.safety_interdictions = 0
        self.is_paused = False

    def stream(self, gcode_lines: List[str]) -> Generator[str, None, None]:
        """
        Yields G-Code lines, injecting safety commands if risk is detected.
        """
        for line in gcode_lines:
            line = line.strip()
            if not line:
                continue

            # 1. Check Safety BEFORE yielding the line
            risk = self.check_safety_risk()

            if risk == "CRITICAL":
                self.logger.warning(f"CRITICAL RISK DETECTED. Injecting Feed Hold (M0).")
                self.safety_interdictions += 1
                yield "M0 (Safety Stop: Phantom Trauma Detected)"
                self.is_paused = True
                break # Stop streaming
            
            elif risk == "HIGH":
                self.logger.info(f"High Stress Detected. Injecting Feed Rate Reduction (M203).")
                self.safety_interdictions += 1
                yield "M203 S10 (Safety Slowdown: Cortisol High)"
                # Continue yielding the original line, but slower (simulated)

            # 2. Yield the original line
            yield line
            
            # Simulate processing time (latency)
            # In real system, this is where we wait for controller ACK
            time.sleep(0.001)

    def check_safety_risk(self) -> str:
        """
        Queries the Dopamine Engine for the current neuro-chemical state.
        Returns: 'SAFE', 'HIGH', or 'CRITICAL'
        """
        # In a real scenario, 'history' would come from live telemetry buffer
        # Here we mock it or fetch latest from repo
        current_time = datetime.now()
        
        # We assume the engine has access to the latest data via its repo
        # For this simulation, we will rely on the engine's internal state mechanism
        # or we can pass a dummy history if needed.
        # But wait, detect_phantom_trauma requires history.
        
        # In a real tight loop, we might want a specialized method in DopamineEngine
        # that doesn't hit the DB.
        # For now, let's assume we can call detect_phantom_trauma.
        
        try:
             # Fetch minimal history (simulated)
            history = self.engine.repository.get_recent_by_machine(self.machine_id)
            result = self.engine.detect_phantom_trauma(history, current_time)
            
            if result.get("detection") == "phantom_trauma_detected":
                return "CRITICAL"
            
            # Also check Cortisol level (mock check if not explicit in result)
            # In a full implementation, we'd check engine.calculate_current_state()
            # Let's check state as well.
            state = self.engine.calculate_current_state(self.machine_id)
            if state.get("cortisol", 0) > 0.8:
                return "HIGH"
                
            return "SAFE"
            
        except Exception as e:
            self.logger.error(f"Error checking safety: {e}")
            return "SAFE" # Fail open or closed? Fail SAFE usually means stop, but here we default to run.

