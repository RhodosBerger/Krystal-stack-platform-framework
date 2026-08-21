#!/usr/bin/env python3
"""
Elements Database.
The "Seed Knowledge" for Fanuc Rise.
Contains definitions for Materials, Tools, and Strategies.
"""

from typing import Dict, Any

# --- MATERIAL DATABASE ---
# Defines physical properties and "Neuro-Mantinels" for materials.
MATERIALS_DB: Dict[str, Any] = {
    "Aluminum6061": {
        "hardness": "Soft",
        "ideal_sfm": 1200,      # Surface Feet per Minute
        "cortisol_threshold": 0.8, # Tolerates vibration well
        "preferred_strategy": "ACTION_RUSH_MODE",
        "mantinel_limit": 15000000 # High Power Limit
    },
    "Steel4140": {
        "hardness": "Medium",
        "ideal_sfm": 400,
        "cortisol_threshold": 0.5, # Moderate sensitivity
        "preferred_strategy": "ACTION_STANDARD_MODE",
        "mantinel_limit": 8000000
    },
    "Titanium6Al4V": {
        "hardness": "Hard",
        "ideal_sfm": 250,
        "cortisol_threshold": 0.2, # Very sensitive to vibration!
        "preferred_strategy": "ACTION_CAUTIOUS_MODE",
        "mantinel_limit": 4000000
    },
    "Inconel718": {
        "hardness": "Superalloy",
        "ideal_sfm": 120,
        "cortisol_threshold": 0.1, # EXTREMELY sensitive
        "preferred_strategy": "ACTION_SURVIVAL_MODE",
        "mantinel_limit": 2000000
    }
}

# --- STRATEGY DATABASE ---
# Defines how the Optimizer adjusts parameters.
STRATEGY_DB: Dict[str, Any] = {
    "ACTION_RUSH_MODE": {
        "feed_multiplier": 1.5,
        "rpm_multiplier": 1.2,
        "dopamine_weight": 2.0, # High Reward for speed
        "cortisol_penalty": 0.5  # Ignore minor stress
    },
    "ACTION_STANDARD_MODE": {
        "feed_multiplier": 1.0,
        "rpm_multiplier": 1.0,
        "dopamine_weight": 1.0,
        "cortisol_penalty": 1.0
    },
    "ACTION_CAUTIOUS_MODE": {
        "feed_multiplier": 0.8,
        "rpm_multiplier": 0.9,
        "dopamine_weight": 0.5, # Low Reward incentive
        "cortisol_penalty": 2.0  # High Penalty for stress
    },
    "ACTION_SURVIVAL_MODE": {
        "feed_multiplier": 0.5,
        "rpm_multiplier": 0.7,
        "dopamine_weight": 0.0,
        "cortisol_penalty": 5.0  # PARANOID
    }
}

# --- TOOL DATABASE ---
# Common Endmills
TOOLS_DB: Dict[str, Any] = {
    "EM_12mm_Carbide_3Flute": {
        "max_rpm": 12000,
        "max_chip_load": 0.1,
        "ideal_material": "Aluminum6061"
    },
    "EM_10mm_Coated_5Flute": {
        "max_rpm": 8000,
        "max_chip_load": 0.05,
        "ideal_material": "Steel4140"
    }
}

def get_material_profile(name: str) -> Dict[str, Any]:
    return MATERIALS_DB.get(name, MATERIALS_DB["Steel4140"])

def get_strategy_profile(name: str) -> Dict[str, Any]:
    return STRATEGY_DB.get(name, STRATEGY_DB["ACTION_STANDARD_MODE"])
