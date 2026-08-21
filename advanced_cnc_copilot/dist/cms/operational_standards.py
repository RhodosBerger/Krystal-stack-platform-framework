#!/usr/bin/env python3
"""
OPERATIONAL STANDARDS (Tabuľkové Štandardy)
The "Pillars of Decision" for Fanuc Rise.

Purpose: 
To define rigid constants and norms used by the Signaling System.
Separates 'Rules' from 'Logic'.
"""

from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class MachineNorms:
    """
    Standard Operating Limits (The 'Table')
    """
    # Spindle Load (%)
    LOAD_CONTINUOUS: float = 80.0   # S1 Rating
    LOAD_PEAK_SHORT: float = 120.0  # S6-40% Rating
    LOAD_CRITICAL: float   = 150.0  # Instant Damage
    
    # Vibration (g)
    VIB_OPTIMAL: float     = 0.05
    VIB_WARNING: float     = 0.20
    VIB_CRITICAL: float    = 0.80   # Chatter
    
    # Thermal (deg C) - Simulated
    TEMP_OPTIMAL: float    = 40.0
    TEMP_WARNING: float    = 60.0
    TEMP_CRITICAL: float   = 85.0

# --- PREDEFINED MACHINE CLASSES ---

# Heavy Duty Steel Cutting (e.g., Doosan Mynx / Fanuc Robodrill)
CLASS_A_HEAVY = MachineNorms(
    LOAD_CONTINUOUS=90.0, 
    LOAD_PEAK_SHORT=130.0,
    VIB_WARNING=0.30
)

# Precision Mold & Die (e.g., Hermle / Yasda)
CLASS_B_PRECISION = MachineNorms(
    LOAD_CONTINUOUS=60.0,  # Run cooler for precision
    VIB_WARNING=0.05,      # Zero tolerance for chatter
    TEMP_CRITICAL=50.0     # Thermal stability paramount
)

# Standard General Purpose
CLASS_C_STANDARD = MachineNorms()

# --- SEMAPHORE THRESHOLDS ---
# The mapping logic for converting Value -> Color
def get_load_status(val: float, norms: MachineNorms) -> str:
    if val > norms.LOAD_CRITICAL: return "RED"
    if val > norms.LOAD_CONTINUOUS: return "AMBER"
    return "GREEN"

def get_vib_status(val: float, norms: MachineNorms) -> str:
    if val > norms.VIB_CRITICAL: return "RED"
    if val > norms.VIB_WARNING: return "AMBER"
    return "GREEN"
