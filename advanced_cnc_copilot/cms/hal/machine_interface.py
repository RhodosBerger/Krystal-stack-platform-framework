from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class MachineStatus(Enum):
    """Enumeration of possible machine states"""
    IDLE = "idle"
    RUNNING = "running"
    ALARM = "alarm"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class TelemetryData:
    """Data class for telemetry readings"""
    spindle_load: float
    vibration_x: float
    vibration_y: float
    vibration_z: float
    spindle_rpm: float
    feed_rate: float
    temperature: float
    axis_position_x: float
    axis_position_y: float
    axis_position_z: float
    coolant_flow: float
    tool_offset_x: float
    tool_offset_y: float
    tool_offset_z: float
    machine_status: MachineStatus


class MachineInterface(ABC):
    """
    Abstract interface for CNC machine communication
    Implements the Universal HAL pattern as required by directive
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the machine"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close connection to the machine"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if machine is currently connected"""
        pass
    
    @abstractmethod
    def read_telemetry(self) -> TelemetryData:
        """Read current telemetry data from the machine"""
        pass
    
    @abstractmethod
    def read_spindle_load(self) -> float:
        """Read current spindle load percentage"""
        pass
    
    @abstractmethod
    def read_vibration_levels(self) -> Dict[str, float]:
        """Read vibration levels on all axes"""
        pass
    
    @abstractmethod
    def read_temperature(self) -> float:
        """Read current temperature"""
        pass
    
    @abstractmethod
    def read_axis_positions(self) -> Dict[str, float]:
        """Read current axis positions"""
        pass
    
    @abstractmethod
    def send_command(self, command: str) -> bool:
        """Send a command to the machine"""
        pass
    
    @abstractmethod
    def emergency_stop(self) -> bool:
        """Trigger emergency stop"""
        pass
    
    @abstractmethod
    def get_machine_status(self) -> MachineStatus:
        """Get current machine status"""
        pass