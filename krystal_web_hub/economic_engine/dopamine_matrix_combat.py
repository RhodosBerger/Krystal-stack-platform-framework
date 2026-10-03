# ==============================================================================
# KRYSTAL-STACK: DOPAMINE-STIMULATING CARD COMBINATORICS & SET-MATRIX COMBAT
# ==============================================================================
# Implements:
#   1. Visceral player feedback triggers ("Dopamine Cadence Overdrive").
#   2. Compact Set-Theoretic matrix algebra (Union, Intersection, Difference)
#      on duel combatant states.
#   3. Chain Reaction Timings (perfect parry windows, sequential combo bursts).
# ==============================================================================

import time
import math
from typing import Dict, List, Set, Any, Optional, Tuple

class DopamineCombatState:
    """Represents a set-theoretic matrix partition of all units in a duel."""
    def __init__(self, combatant_ids: List[str]):
        self.all_units: Set[str] = set(combatant_ids)
        self.allied_units: Set[str] = set()
        self.enemy_units: Set[str] = set()
        self.uncovered_units: Set[str] = set()
        self.crowd_controlled_units: Set[str] = set() # Rooted, Frozen, Stunned
        self.ward_shielded_units: Set[str] = set()
        self.supernatural_units: Set[str] = set()
        self.critical_vulnerable_units: Set[str] = set()

    def update_partitions(
        self,
        allies: List[str],
        enemies: List[str],
        uncovered: List[str],
        crowd_controlled: List[str],
        warded: List[str],
        supernatural: List[str]
    ) -> None:
        self.allied_units = set(allies)
        self.enemy_units = set(enemies)
        self.uncovered_units = set(uncovered)
        self.crowd_controlled_units = set(crowd_controlled)
        self.ward_shielded_units = set(warded)
        self.supernatural_units = set(supernatural)
        # Critical vulnerable set: Intersection of uncovered and crowd-controlled enemies
        self.critical_vulnerable_units = (self.enemy_units & self.uncovered_units) | (self.enemy_units & self.crowd_controlled_units)

    def get_set_intersection(self, set_a: Set[str], set_b: Set[str]) -> Set[str]:
        return set_a & set_b

    def get_set_difference(self, set_a: Set[str], set_b: Set[str]) -> Set[str]:
        return set_a - set_b

    def get_symmetric_difference(self, set_a: Set[str], set_b: Set[str]) -> Set[str]:
        return set_a ^ set_b


class DopamineCadenceEngine:
    """
    Evaluates micro-timing intervals between card executions to trigger
    dopamine-stimulating sensory feedback, critical cascades, and mana refunds.
    """

    CRITICAL_CADENCE_WINDOW_SEC = 0.85 # Micro-timing window for perfect rhythm

    @staticmethod
    def evaluate_timing_chain(
        card_timestamps: List[float],
        played_card_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Calculates rhythm cadence and determines if a Dopamine Overdrive is achieved.
        """
        if len(card_timestamps) < 2:
            return {
                "dopamine_overdrive": False,
                "cadence_rating": "single_action",
                "perfect_chain_count": 0,
                "mana_refund": 0,
                "damage_cascade_multiplier": 1.0,
                "sensory_feedback": "normal"
            }

        intervals = [card_timestamps[i] - card_timestamps[i-1] for i in range(1, len(card_timestamps))]
        perfect_chains = sum(1 for dt in intervals if dt <= DopamineCadenceEngine.CRITICAL_CADENCE_WINDOW_SEC)

        is_overdrive = perfect_chains >= 2 and len(played_card_ids) >= 3

        if is_overdrive:
            mana_refund = min(4, perfect_chains + 1)
            dmg_multiplier = round(1.0 + (0.35 * perfect_chains), 2)
            feedback = "DOPAMINE_SURGE_CASCADE: Golden Shimmer, Kinetic Screen Shake, Critical Chime"
            rating = "PERFECT_CADENCE_OVERDRIVE"
        elif perfect_chains >= 1:
            mana_refund = 1
            dmg_multiplier = 1.20
            feedback = "RHYTHMIC_FLOW: Silver Pulse, Sound Resonance"
            rating = "RHYTHMIC_STRIKE"
        else:
            mana_refund = 0
            dmg_multiplier = 1.0
            feedback = "STANDARD"
            rating = "TEMPO_NORMAL"

        return {
            "dopamine_overdrive": is_overdrive,
            "cadence_rating": rating,
            "perfect_chain_count": perfect_chains,
            "average_interval": round(sum(intervals) / len(intervals), 3) if intervals else 0.0,
            "mana_refund": mana_refund,
            "damage_cascade_multiplier": dmg_multiplier,
            "sensory_feedback": feedback
        }

    @staticmethod
    def execute_matrix_shatter_combo(
        state: DopamineCombatState,
        combo_name: str,
        cadence_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        r"""
        Executes a compact set-theoretic matrix combo affecting target sets.
        Combos:
          1. 'aetheric_cadence_shatter': Targets (Enemies ∩ Uncovered). Generates Ward for Allies.
          2. 'corrosive_rupture_cascade': Strips Ward from all targets in (Enemies ∩ Warded).
          3. 'verdant_avalanche_cataclysm': Traps (Enemies \ CC) into CC set, triggers instant mortar.
        """
        mult = cadence_info.get("damage_cascade_multiplier", 1.0)
        is_overdrive = cadence_info.get("dopamine_overdrive", False)

        if combo_name == "aetheric_cadence_shatter":
            affected_targets = state.enemy_units & state.uncovered_units
            base_dmg = 4
            effective_dmg = int(math.floor(base_dmg * mult))
            allied_ward_burst = 3 if is_overdrive else 1
            action_desc = f"AÉTEROVÉ FRAKTÚROVÉ ROZBITIE: Zasahuje nekrytých nepriateľov {list(affected_targets)} za {effective_dmg} DMG. Spojenci získavajú +{allied_ward_burst} Ward."
            return {
                "success": True,
                "combo": combo_name,
                "affected_set": list(affected_targets),
                "damage_per_target": effective_dmg,
                "allied_ward_granted": allied_ward_burst,
                "matrix_operation": "Enemies ∩ Uncovered",
                "action_desc": action_desc
            }

        elif combo_name == "corrosive_rupture_cascade":
            warded_enemies = state.enemy_units & state.ward_shielded_units
            # Strips ward entirely and inflicts DoT
            action_desc = f"KOROZÍVNA RUPTÚRA: Úplne rozbíja Ward štíty cieľov {list(warded_enemies)} a nanáša kyselinu."
            return {
                "success": True,
                "combo": combo_name,
                "affected_set": list(warded_enemies),
                "ward_stripped": True,
                "acid_dot_applied": 2,
                "matrix_operation": "Enemies ∩ Warded",
                "action_desc": action_desc
            }

        elif combo_name == "verdant_avalanche_cataclysm":
            unrooted = state.enemy_units - state.crowd_controlled_units
            action_desc = f"VERDANTNÁ LAVÍNA: Zväzuje všetky voľné nepriateľské ciele {list(unrooted)} koreňmi a pripravuje delostreleckú salvu."
            return {
                "success": True,
                "combo": combo_name,
                "affected_set": list(unrooted),
                "immobilized_set": list(unrooted),
                "ready_for_mortar_barrage": True,
                "matrix_operation": "Enemies \\ CrowdControlled",
                "action_desc": action_desc
            }

        return {"success": False, "error": f"Neznáme kombo '{combo_name}'."}
