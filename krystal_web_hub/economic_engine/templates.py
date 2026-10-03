# ==============================================================================
# KRYSTAL-STACK: ECONOMIC ENGINE TEMPLATES & SCHEMAS
# ==============================================================================
# Declarative templates and presets for custom cards, buildings, scenarios,
# and Godot match scenes.
# ==============================================================================

import json
from typing import Dict, Any, List

# ------------------------------------------------------------------------------
# 1. SCENARIO TEMPLATES
# ------------------------------------------------------------------------------
SCENARIO_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "skirmish_severni_stity": {
        "id": "skirmish_severni_stity",
        "title": "Potýčka na Severných Štítoch",
        "description": "Boj v ľadovom priesmyku bohatom na aéterové žily. Kryštálový Kmeň vs Jedovatý Kmeň.",
        "player_tribe": "crystal",
        "enemy_tribe": "toxic",
        "initial_player_resources": {"mana": 10, "aether_crystal": 4, "toxic_slime": 0, "amber_rune": 0},
        "initial_enemy_resources": {"mana": 8, "aether_crystal": 0, "toxic_slime": 4, "amber_rune": 0},
        "environmental_hazards": ["blizzard_slow", "mana_surge"]
    },
    "siege_toxic_wasteland": {
        "id": "siege_toxic_wasteland",
        "title": "Obliehanie Toxickej Pustiny",
        "description": "Korozívne prostredie plné kyslých gejzírov. Jedovatý Kmeň vs Druidi.",
        "player_tribe": "toxic",
        "enemy_tribe": "druid",
        "initial_player_resources": {"mana": 8, "aether_crystal": 0, "toxic_slime": 5, "amber_rune": 0},
        "initial_enemy_resources": {"mana": 10, "aether_crystal": 0, "toxic_slime": 0, "amber_rune": 4},
        "environmental_hazards": ["acid_rain", "spore_fog"]
    },
    "grove_of_deep_forest": {
        "id": "grove_of_deep_forest",
        "title": "Svätyňa Hlbokého Lesa",
        "description": "Prastaré stromy absorbujúce útoky. Druidi vs Kryštálový Kmeň.",
        "player_tribe": "druid",
        "enemy_tribe": "crystal",
        "initial_player_resources": {"mana": 9, "aether_crystal": 0, "toxic_slime": 0, "amber_rune": 5},
        "initial_enemy_resources": {"mana": 10, "aether_crystal": 3, "toxic_slime": 0, "amber_rune": 0},
        "environmental_hazards": ["living_vines", "solar_flare"]
    }
}

# ------------------------------------------------------------------------------
# 2. SCHEMA TEMPLATES
# ------------------------------------------------------------------------------
CARD_TEMPLATE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "KrystalCardTemplate",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "tribe": {"type": "string", "enum": ["crystal", "toxic", "druid", "neutral"]},
        "ability_type": {"type": "string", "enum": ["offensive", "defensive", "economic", "combo", "ultimate"]},
        "cost": {
            "type": "object",
            "properties": {
                "mana": {"type": "integer", "minimum": 0},
                "aether_crystal": {"type": "integer", "minimum": 0},
                "toxic_slime": {"type": "integer", "minimum": 0},
                "amber_rune": {"type": "integer", "minimum": 0}
            },
            "required": ["mana"]
        },
        "damage": {"type": "integer", "minimum": 0},
        "healing": {"type": "integer", "minimum": 0},
        "shield": {"type": "integer", "minimum": 0},
        "status_applied": {"type": "string"},
        "duration_rounds": {"type": "integer", "minimum": 1},
        "description": {"type": "string"},
        "mesh_asset": {"type": "string"},
        "color": {"type": "string"},
        "tier_required": {"type": "integer", "minimum": 1, "maximum": 3}
    },
    "required": ["id", "name", "tribe", "ability_type", "cost"]
}

BUILDING_TEMPLATE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "KrystalBuildingTemplate",
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "name": {"type": "string"},
        "tribe": {"type": "string", "enum": ["crystal", "toxic", "druid", "neutral"]},
        "cost": {
            "type": "object",
            "properties": {
                "mana": {"type": "integer", "minimum": 0},
                "aether_crystal": {"type": "integer", "minimum": 0},
                "toxic_slime": {"type": "integer", "minimum": 0},
                "amber_rune": {"type": "integer", "minimum": 0}
            },
            "required": ["mana"]
        },
        "max_hp": {"type": "integer", "minimum": 1},
        "mana_generation": {"type": "integer", "minimum": 0},
        "resource_generation": {"type": "object"},
        "passive_perk": {"type": "string"},
        "mesh_asset": {"type": "string"},
        "color": {"type": "string"},
        "tier": {"type": "integer", "minimum": 1, "maximum": 3}
    },
    "required": ["id", "name", "tribe", "cost", "max_hp"]
}

def get_all_templates() -> Dict[str, Any]:
    return {
        "scenarios": SCENARIO_TEMPLATES,
        "card_schema": CARD_TEMPLATE_SCHEMA,
        "building_schema": BUILDING_TEMPLATE_SCHEMA
    }
