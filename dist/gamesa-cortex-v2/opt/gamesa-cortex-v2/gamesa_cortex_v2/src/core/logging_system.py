import json
import time
import os
import logging
from typing import Dict, Any, List
from .config import GamesaConfig

class IntraspectralLogger:
    """
    Gamesa Cortex V2: Intraspectral Logging System.
    Aggregates logs from various system components into a unified JSON format
    compatible with OpenVINO telemetry or analysis tools.
    """
    def __init__(self, log_dir="logs"):
        self.logger = logging.getLogger("IntraspectralLogger")
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.log_buffer: List[Dict[str, Any]] = []
        self.spectra = {
            "PLANNING": "blue",
            "SAFETY": "red",
            "ECONOMIC": "green",
            "INFERENCE": "purple",
            "SYSTEM": "white"
        }

    def log_event(self, spectrum: str, component: str, message: str, metrics: Dict[str, Any] = None):
        """
        Log an event in a specific spectrum.
        """
        if spectrum not in self.spectra:
            spectrum = "SYSTEM"
            
        event = {
            "timestamp": time.time_ns(),
            "spectrum": spectrum,
            "component": component,
            "message": message,
            "metrics": metrics or {}
        }
        
        self.log_buffer.append(event)
        
        # In a real system, we might stream this or batch write
        # For simplicity, we just print to console for now (simulating stream)
        # print(f"[[{spectrum}]] {component}: {message} {metrics}")

    def export_logs(self, filename="intraspectral_latest.json"):
        """
        Export buffered logs to a JSON file compatible with OpenVINO analysis tools.
        """
        filepath = os.path.join(self.log_dir, filename)
        try:
            with open(filepath, 'w') as f:
                json.dump(self.log_buffer, f, indent=2)
            self.logger.info(f"Logs exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export logs: {e}")
            
    def clear_buffer(self):
        self.log_buffer = []
