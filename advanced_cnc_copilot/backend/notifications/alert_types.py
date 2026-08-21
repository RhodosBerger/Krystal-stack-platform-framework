"""
Alert Types & Priority Definitions 🚨
Responsibility:
1. Define notification severity levels and categories.
2. Provide structured alert payloads.
"""
from enum import Enum
from typing import Dict, Any

class AlertPriority(Enum):
    CRITICAL = "CRITICAL"  # System failures, immediate action required
    WARNING = "WARNING"    # Potential issues, attention needed
    INFO = "INFO"          # General updates, no action required
    SUCCESS = "SUCCESS"    # Successful operations

class AlertCategory(Enum):
    SYSTEM = "SYSTEM"           # System health and status
    GENERATION = "GENERATION"   # G-Code/Payload generation events
    WORKFLOW = "WORKFLOW"       # Workflow execution events
    EXPORT = "EXPORT"           # Data export events
    IMPORT = "IMPORT"           # Data import events
    SECURITY = "SECURITY"       # Authentication and security events

def get_priority_color(priority: AlertPriority) -> str:
    """Returns UI color for priority level."""
    colors = {
        AlertPriority.CRITICAL: "#ff4444",
        AlertPriority.WARNING: "#ffaa00",
        AlertPriority.INFO: "#00d4ff",
        AlertPriority.SUCCESS: "#00ff88"
    }
    return colors.get(priority, "#888888")

def get_priority_icon(priority: AlertPriority) -> str:
    """Returns icon name for priority level."""
    icons = {
        AlertPriority.CRITICAL: "AlertTriangle",
        AlertPriority.WARNING: "AlertCircle",
        AlertPriority.INFO: "Info",
        AlertPriority.SUCCESS: "CheckCircle"
    }
    return icons.get(priority, "Bell")
