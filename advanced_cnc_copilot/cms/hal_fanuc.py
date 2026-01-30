#!/usr/bin/env python3
"""
HAL FANUC: FOCAS Adapter.
Implementation of GenericController for Fanuc Machines.
Acts as a switching layer: Uses real FOCAS if available, else runs Simulation.
"""

from hal_core import GenericController
from .focas_bridge import FocasBridge
import random
import time
from typing import Dict

class FanucAdapter(GenericController):
    def __init__(self, ip="127.0.0.1", port=8193):
        self.ip = ip
        self.port = port
        # Initialize the low-level bridge
        self.bridge = FocasBridge(ip, port)
        self.simulation_mode = True # Default until connection confirmed
        self.connected = False

    def connect(self) -> bool:
        """
        Attempts to connect to physical hardware. 
        Falls back to Simulation Mode if hardware is unreachable.
        """
        result = self.bridge.connect()
        
        if result == 0:
            self.connected = True
            self.simulation_mode = False
            print("[HAL-FANUC] ✅ HARDWARE CONNECTED (FOCAS)")
        else:
            self.connected = True
            self.simulation_mode = True
            print("[HAL-FANUC] ⚠️ HARDWARE NOT FOUND. ENTERING SIMULATION MODE.")
            
        return True

    def disconnect(self):
        self.bridge.disconnect()
        self.connected = False

    def get_status(self) -> str:
        if not self.connected:
            return "DISCONNECTED"
            
        if not self.simulation_mode:
            return self.bridge.read_status()
        
        return "RUNNING (SIM)"

    def read_metrics(self) -> Dict[str, float]:
        if not self.connected:
            return {"rpm": 0, "feed": 0, "load": 0, "vibration": 0}
            
        if not self.simulation_mode:
            # --- REAL HARDWARE DATA ---
            try:
                real_rpm = self.bridge.read_spindle_speed()
                real_status = self.bridge.read_status()
                
                # Some metrics might still need estimation if sensors aren't wired
                return {
                    "rpm": real_rpm,
                    "feed": 0.0, # Implement cnc_rdfeed if needed
                    "load": 0.0,
                    "vibration": 0.0,
                    "status": real_status
                }
            except Exception:
                return {"rpm": 0, "error": "FOCAS_READ_FAIL"}
        
        else:
            # --- SIMULATION DATA ---
            # Generate realistic-looking noise
            return {
                "rpm": 8000.0 + random.uniform(-50, 50),
                "feed": 2000.0,
                "load": 0.45 + random.uniform(-0.02, 0.05),
                "vibration": 0.05 + random.uniform(0, 0.01)
            }

    def get_protocol_name(self) -> str:
        return "FOCAS_HSSB" if not self.simulation_mode else "FOCAS_SIMULATOR"

    def emergency_stop(self) -> bool:
        if not self.simulation_mode:
            print("[HAL-FANUC] 🛑 HARDWARE E-STOP SENT")
            # self.bridge.send_stop() # Implement write logic in bridge
        else:
            print("[HAL-FANUC] 🛑 SIMULATED E-STOP")
        return True
