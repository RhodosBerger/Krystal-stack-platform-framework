#!/usr/bin/env python3
"""
HAL CORE: Generic Controller Interface.
The standardized contract for all CNC machines in the "Fanuc Rise" network.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class GenericController(ABC):
    """
    Abstract Base Class for CNC Hardware Adapters.
    Enforces a unified API for the 'Sensory Cortex'.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the machine."""
        pass
        
    @abstractmethod
    def disconnect(self):
        """Close connection."""
        pass

    @abstractmethod
    def get_status(self) -> str:
        """Returns: 'IDLE', 'RUNNING', 'ALARM', 'DISCONNECTED'"""
        pass

    @abstractmethod
    def read_metrics(self) -> Dict[str, float]:
        """
        Returns normalized metrics:
        {
            'rpm': float,        # Actual Spindle Speed
            'feed': float,       # Actual Feed Rate
            'load': float,       # 0.0 - 1.0 (Max Spindle Load)
            'vibration': float   # 0.0 - 1.0 (Normalized Vibration)
        }
        """
        pass

    @abstractmethod
    def get_protocol_name(self) -> str:
        """Returns protocol name (e.g. 'FOCAS', 'OPC-UA')"""
        pass
        
    @abstractmethod
    def emergency_stop(self) -> bool:
        """Trigger immediate E-STOP."""
        pass
