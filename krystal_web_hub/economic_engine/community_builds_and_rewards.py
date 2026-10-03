# ==============================================================================
# KRYSTAL-STACK: COMMUNITY GAME BUILDS & THREAT-LEVEL EVENT REWARD SYSTEM
# ==============================================================================
# Implements:
#   1. Community build archetypes (Mortar Siegebreaker, Shadowblade, etc.).
#   2. Event hazard and threat classification (Threat 1 to Threat 5).
#   3. Mathematical reward derivation from event hazard, player risk, and combos.
#   4. Double-entry ledger reward payout auditing.
# ==============================================================================

import math
from enum import Enum
from typing import Dict, List, Any, Optional

class EventThreatLevel(int, Enum):
    THREAT_1_PERIMETER_SKIRMISH = 1
    THREAT_2_CANYON_AMBUSH = 2
    THREAT_3_SECTOR_OUTPOST_RAID = 3
    THREAT_4_CITADEL_BOSS_SIEGE = 4
    THREAT_5_CATACLYSMIC_INCURSION = 5

class CommunityBuildRegistry:
    """
    Catalog of community-vetted builds with card decks, synergies, and tactical profiles.
    """

    COMMUNITY_BUILDS = {
        "mortar_siegebreaker": {
            "build_id": "mortar_siegebreaker",
            "name": "Oblivion Mortar Siegebreaker",
            "archetype": "Heavy Artillery & Area Denial",
            "primary_weapon": "Twin-Barrel Heavy Mortar",
            "cold_sidearm": "Trench Trenching Pickaxe",
            "key_cards": ["MORTAR_BARRAGE_A1", "COVER_SHRED_B2", "PLUNGING_SALVO_C3"],
            "optimal_cadence_window_sec": 0.40,
            "playstyle_summary": "Identifies uncovered enemies from elevated positions and wipes clusters using high plunging arcs.",
            "stat_biases": {
                "blast_radius_mult": 1.45,
                "indirect_fire_accuracy": 0.90,
                "melee_parry_bonus": 0.15
            }
        },
        "shadowblade_infiltrator": {
            "build_id": "shadowblade_infiltrator",
            "name": "Shadowblade Cadence Infiltrator",
            "archetype": "High-Tempo Cold Melee Assassin",
            "primary_weapon": "Dual Aetheric Daggers",
            "cold_sidearm": "Spring-Loaded Wrist Kunai",
            "key_cards": ["SHADOW_STEP_S1", "RAPID_FLURRY_S2", "CADENCE_OVERDRIVE_S3"],
            "optimal_cadence_window_sec": 0.25,
            "playstyle_summary": "Triggers rapid dopamine cadence overdrives with consecutive sub-0.5s card plays and bullet time parries.",
            "stat_biases": {
                "blast_radius_mult": 0.20,
                "indirect_fire_accuracy": 0.30,
                "melee_parry_bonus": 0.85
            }
        },
        "aetheric_cadence_duelist": {
            "build_id": "aetheric_cadence_duelist",
            "name": "Aetheric Spellblade Sovereign",
            "archetype": "Set-Theoretic Hybrid Duelist",
            "primary_weapon": "Crystal Greatsword",
            "cold_sidearm": "Aetheric Focus Orb",
            "key_cards": ["AETHERIC_FRACTURE_A1", "WARD_CASCADE_W2", "SET_INTERSECT_BURST_I3"],
            "optimal_cadence_window_sec": 0.50,
            "playstyle_summary": "Manipulates enemy set partitions, strips wards using corrosive rupture, and generates massive allied ward shields.",
            "stat_biases": {
                "blast_radius_mult": 0.70,
                "indirect_fire_accuracy": 0.75,
                "melee_parry_bonus": 0.60
            }
        },
        "verdant_bulwark": {
            "build_id": "verdant_bulwark",
            "name": "Verdant Bulwark Warden",
            "archetype": "Druidic Crowd-Control Tank",
            "primary_weapon": "Living Wood Greatshield",
            "cold_sidearm": "Thorny Bramble Flail",
            "key_cards": ["ROOT_AVALANCHE_R1", "STONE_BARRIER_B2", "EARTHEN_MORTAR_E3"],
            "optimal_cadence_window_sec": 0.70,
            "playstyle_summary": "Immobilizes enemy sets into CC, preventing evasion while calling down slow crushing earthen artillery.",
            "stat_biases": {
                "blast_radius_mult": 1.10,
                "indirect_fire_accuracy": 0.65,
                "melee_parry_bonus": 0.75
            }
        }
    }

    @classmethod
    def get_build(cls, build_id: str) -> Optional[Dict[str, Any]]:
        return cls.COMMUNITY_BUILDS.get(build_id)

    @classmethod
    def list_builds(cls) -> List[Dict[str, Any]]:
        return list(cls.COMMUNITY_BUILDS.values())


class EventRewardDerivationEngine:
    """
    Computes game reward payouts derived from event threat level, player risk factor,
    combat combos, and bullet time performance.
    """

    THREAT_SPECS = {
        EventThreatLevel.THREAT_1_PERIMETER_SKIRMISH: {
            "hazard_name": "Perimeter Skirmish",
            "hazard_mult": 1.0,
            "base_gold": 120,
            "base_krystal_gems": 5,
            "base_xp": 250,
            "rare_drop_chance": 0.05
        },
        EventThreatLevel.THREAT_2_CANYON_AMBUSH: {
            "hazard_name": "Canyon Ambush",
            "hazard_mult": 1.5,
            "base_gold": 220,
            "base_krystal_gems": 12,
            "base_xp": 500,
            "rare_drop_chance": 0.15
        },
        EventThreatLevel.THREAT_3_SECTOR_OUTPOST_RAID: {
            "hazard_name": "Sector Outpost Raid",
            "hazard_mult": 2.25,
            "base_gold": 450,
            "base_krystal_gems": 25,
            "base_xp": 1100,
            "rare_drop_chance": 0.35
        },
        EventThreatLevel.THREAT_4_CITADEL_BOSS_SIEGE: {
            "hazard_name": "Citadel Boss Siege",
            "hazard_mult": 3.75,
            "base_gold": 950,
            "base_krystal_gems": 60,
            "base_xp": 2800,
            "rare_drop_chance": 0.65
        },
        EventThreatLevel.THREAT_5_CATACLYSMIC_INCURSION: {
            "hazard_name": "Cataclysmic Incursion",
            "hazard_mult": 6.0,
            "base_gold": 2400,
            "base_krystal_gems": 180,
            "base_xp": 7500,
            "rare_drop_chance": 0.95
        }
    }

    @staticmethod
    def derive_event_rewards(
        threat_level: EventThreatLevel,
        remaining_hp: int,                # Max 6 HP vital invariant
        dopamine_cadence_score: float,    # 1.0 to 2.5 multiplier from rhythm
        bullet_time_count: int = 0,       # Intersections achieved in bullet time
        uncovered_targets_eliminated: int = 0,
        player_account_id: str = "player_hero_1"
    ) -> Dict[str, Any]:
        """
        Derives rewards based on:
          1. Hazard multiplier H_lvl = (1 + threat_level^1.4)
          2. Risk factor R_factor: Survival at lower remaining HP yields higher bravery bonus
          3. Performance factor P_factor: Cadence combo + Bullet Time + Mortar Kills
        """
        spec = EventRewardDerivationEngine.THREAT_SPECS.get(
            threat_level, EventRewardDerivationEngine.THREAT_SPECS[EventThreatLevel.THREAT_1_PERIMETER_SKIRMISH]
        )

        hazard_mult = spec["hazard_mult"]
        threat_exp = round(1.0 + math.pow(threat_level.value, 1.4) * 0.25, 2)

        # Risk factor based on remaining HP out of 6 (clutch low-HP victory = higher risk multiplier)
        clamped_hp = max(1, min(6, remaining_hp))
        risk_bonus = round((6 - clamped_hp) * 0.08, 2) # up to +40% at 1 HP remaining

        # Performance multiplier
        bt_bonus = bullet_time_count * 0.15
        uncovered_bonus = uncovered_targets_eliminated * 0.10
        total_perf_mult = round(dopamine_cadence_score + bt_bonus + uncovered_bonus, 2)

        # Derived totals
        composite_multiplier = round(hazard_mult * threat_exp * (1.0 + risk_bonus) * total_perf_mult, 3)

        awarded_gold = int(math.floor(spec["base_gold"] * composite_multiplier))
        awarded_gems = int(math.floor(spec["base_krystal_gems"] * composite_multiplier))
        awarded_xp = int(math.floor(spec["base_xp"] * composite_multiplier))

        # Check rare crafting blueprint drops
        blueprints = []
        if spec["rare_drop_chance"] * (composite_multiplier / hazard_mult) >= 0.50:
            blueprints.append("RECIPE_AETHERIC_MORTAR_SHELL_T3")
        if threat_level.value >= 4:
            blueprints.append("BLUEPRINT_OBSIDIAN_BLADE_OF_OVERDRIVE")

        # Double-entry ledger transaction spec
        ledger_transaction_spec = {
            "credit_account": player_account_id,
            "debit_account": "treasury_event_rewards",
            "gold_amount": awarded_gold,
            "gem_amount": awarded_gems,
            "description": f"EVENT_REWARD_THREAT_{threat_level.value}_{spec['hazard_name'].upper()}"
        }

        return {
            "threat_level": threat_level.value,
            "threat_name": spec["hazard_name"],
            "remaining_hp": clamped_hp,
            "max_hp_invariant": 6,
            "risk_bonus_percent": int(risk_bonus * 100),
            "performance_multiplier": total_perf_mult,
            "composite_multiplier": composite_multiplier,
            "rewards": {
                "gold": awarded_gold,
                "krystal_gems": awarded_gems,
                "experience_xp": awarded_xp,
                "blueprints": blueprints
            },
            "ledger_audit": ledger_transaction_spec
        }
