#!/usr/bin/env python3
"""
HAL SIEMENS: OPC-UA Adapter.
Implementation of GenericController for Sinumerik Machines.
"""

from hal_core import GenericController
import random
from typing import Dict

class SiemensAdapter(GenericController):
    def __init__(self, endpoint_url="opc.tcp://192.168.1.10:4840"):
        self.endpoint = endpoint_url
        self.connected = False

    def connect(self) -> bool:
        # Mocking asyncua client connect
        self.connected = True
        return True

    def disconnect(self):
        self.connected = False

    def get_status(self) -> str:
        return "RUNNING" if self.connected else "DISCONNECTED"

    def read_metrics(self) -> Dict[str, float]:
        if not self.connected:
            return {"rpm": 0, "feed": 0, "load": 0, "vibration": 0}
            
        # Simulate accessing Nodes "ns=2;s=Machine.Spindle.Speed"
        # OPC-UA is slower, so we simulate higher latency effects if meaningful
        return {
            "rpm": 12000.0,
            "feed": 5000.0,
            "load": 0.60,       # 60% Load
            "vibration": 0.02   # Very smooth (German Engineering mock)
        }

    def get_protocol_name(self) -> str:
        return "OPC_UA"

    def emergency_stop(self) -> bool:
        print("[HAL-SIEMENS] >>> E-STOP TRIGGERED via OPC-UA Method Call!")
        return True
