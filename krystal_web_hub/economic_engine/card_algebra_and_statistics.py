# ==============================================================================
# KRYSTAL-STACK: COMBINATORIAL STATISTICAL MODEL & CARD ACTION ALGEBRA
# ==============================================================================
# Implements hypergeometric probability distributions for 43-card tribal decks,
# mana curve expectancy, predefined Poslední Kmen card action taxonomy,
# and closed-form combat damage algebra with the 6 Max HP vital invariant.
# ==============================================================================

import math
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional
from .tactical_npc_trigonometrics import CombatSector
from .environment_matrix import EnvironmentalCell, EnvironmentalHazardType

class CardActionType(str, Enum):
    MELEE_STRIKE = "melee_strike"
    RANGED_PROJECTILE = "ranged_projectile"
    AOE_BOMBARDMENT = "aoe_bombardment"
    DEFENSIVE_WARD = "defensive_ward"
    HAZARD_SURFACE = "hazard_surface"
    MANA_SURGE = "mana_surge"
    HEAL_RESTORE = "heal_restore"


class HypergeometricCardStatistics:
    """Combinatorial counting model for draw probability and deck distribution."""

    @staticmethod
    def binomial_coefficient(n: int, k: int) -> int:
        """Computes n choose k: C(n, k) = n! / (k! * (n-k)!)."""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        c = 1
        for i in range(k):
            c = c * (n - i) // (i + 1)
        return c

    @staticmethod
    def probability_draw_exact_k(N: int, K: int, n: int, k: int) -> float:
        """
        Hypergeometric Probability Mass Function (PMF):
        P(X = k) = [C(K, k) * C(N-K, n-k)] / C(N, n)
        - N: Total deck size (e.g. 43 cards)
        - K: Total target copies in deck (e.g. 5 Tier-3 legendaries)
        - n: Sample size / cards drawn (e.g. 5 card opening hand)
        - k: Exact number of target cards desired
        """
        if N <= 0 or n <= 0 or K < 0:
            return 0.0
        denom = HypergeometricCardStatistics.binomial_coefficient(N, n)
        if denom == 0:
            return 0.0

        numer = (
            HypergeometricCardStatistics.binomial_coefficient(K, k) *
            HypergeometricCardStatistics.binomial_coefficient(N - K, n - k)
        )
        return round(float(numer) / float(denom), 6)

    @staticmethod
    def probability_draw_at_least_k(N: int, K: int, n: int, k: int) -> float:
        """Cumulative Hypergeometric Probability P(X >= k)."""
        prob_sum = 0.0
        max_possible = min(n, K)
        for i in range(k, max_possible + 1):
            prob_sum += HypergeometricCardStatistics.probability_draw_exact_k(N, K, n, i)
        return round(min(1.0, prob_sum), 6)

    @staticmethod
    def expected_turn_to_draw(
        deck_size: int = 43,
        copies_in_deck: int = 1,
        opening_hand: int = 5,
        cards_per_turn: int = 1
    ) -> float:
        """
        Computes expected turn E[T] to draw at least one copy of a key card.
        Mean cards drawn before first target: E[Draws] = (deck_size + 1) / (copies_in_deck + 1).
        """
        if copies_in_deck <= 0:
            return float('inf')

        expected_cards = (deck_size + 1.0) / (copies_in_deck + 1.0)
        if expected_cards <= opening_hand:
            return 1.0 # Expected in opening hand

        remaining_draws = expected_cards - opening_hand
        expected_turn = 1.0 + (remaining_draws / float(max(1, cards_per_turn)))
        return round(expected_turn, 2)

    @staticmethod
    def analyze_mana_curve(deck: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates mana cost distribution, mean mana cost, and curve variance."""
        if not deck:
            return {"mean_cost": 0.0, "variance": 0.0, "cost_breakdown": {}}

        cost_breakdown: Dict[int, int] = {}
        costs: List[int] = []

        for card in deck:
            c = card.get("cost", 1)
            if isinstance(c, dict):
                c = c.get("mana", 1)
            cost_val = int(c)
            costs.append(cost_val)
            cost_breakdown[cost_val] = cost_breakdown.get(cost_val, 0) + 1

        n = len(costs)
        mean_cost = sum(costs) / float(n)
        variance = sum((x - mean_cost) ** 2 for x in costs) / float(n)

        return {
            "deck_size": n,
            "mean_cost": round(mean_cost, 2),
            "variance": round(variance, 2),
            "std_dev": round(math.sqrt(variance), 2),
            "cost_breakdown": {k: cost_breakdown[k] for k in sorted(cost_breakdown.keys())}
        }


class CardCombatAlgebra:
    """
    Closed-form algebraic resolution of Poslední Kmen card actions.
    Integrates combat sectors, elevation angles, cover density, and the 6 Max HP rule.
    """

    @staticmethod
    def classify_card_action(card: Dict[str, Any]) -> CardActionType:
        """Maps card dictionary attributes into canonical CardActionType."""
        atype = card.get("attack_type", "ranged").lower()
        status = card.get("status_applied", "")
        hp_delta = card.get("hp_delta", 0)
        armor_delta = card.get("armor_delta", 0)

        if hp_delta > 0:
            return CardActionType.HEAL_RESTORE
        if status in ("mana_surge", "mana_charge"):
            return CardActionType.MANA_SURGE
        if status in ("permanent_miasma", "acid_pool", "rooted", "toxic_pool"):
            return CardActionType.HAZARD_SURFACE
        if armor_delta > 0 and hp_delta == 0:
            return CardActionType.DEFENSIVE_WARD
        if atype == "melee":
            return CardActionType.MELEE_STRIKE
        if atype == "aoe":
            return CardActionType.AOE_BOMBARDMENT
        return CardActionType.RANGED_PROJECTILE

    @staticmethod
    def resolve_card_action(
        card: Dict[str, Any],
        attacker_stats: Dict[str, Any],
        defender_stats: Dict[str, Any],
        sector: CombatSector = CombatSector.FRONT,
        elevation_pitch_rad: float = 0.0,
        env_cell: Optional[EnvironmentalCell] = None,
        d6_hit_roll: Optional[int] = None,
        d6_save_roll: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes exact combat algebra:
        1. Evaluates hit threshold from Ballistic/Weapon Skill, Cover, Elevation, and Flank.
        2. Computes armor mitigation, sector armor penetration, and critical bonuses.
        3. Strictly clamps Defender HP in [0, 6] and Attacker HP (on heal) in [0, 6].
        """
        action_type = CardCombatAlgebra.classify_card_action(card)
        base_power = abs(card.get("hp_delta", card.get("damage", 0)))
        card_ap = abs(card.get("armor_delta", 0)) if card.get("armor_delta", 0) < 0 else 0
        heal_val = card.get("hp_delta", 0) if card.get("hp_delta", 0) > 0 else 0

        current_defender_hp = defender_stats.get("hp", 6)
        current_defender_armor = defender_stats.get("armor", 0)
        current_attacker_hp = attacker_stats.get("hp", 6)

        # Handle Healing actions
        if action_type == CardActionType.HEAL_RESTORE:
            new_attacker_hp = min(6, current_attacker_hp + heal_val)
            return {
                "action_type": action_type.value,
                "hit_success": True,
                "damage_dealt": 0,
                "healed_amount": new_attacker_hp - current_attacker_hp,
                "attacker_new_hp": new_attacker_hp,
                "defender_new_hp": current_defender_hp,
                "description": f"Liečenie obnovilo +{heal_val} HP (Aktuálne: {new_attacker_hp}/6 HP)."
            }

        # Handle Defensive Ward actions
        if action_type == CardActionType.DEFENSIVE_WARD:
            shield_gain = card.get("armor_delta", 1)
            new_attacker_armor = attacker_stats.get("armor", 0) + shield_gain
            return {
                "action_type": action_type.value,
                "hit_success": True,
                "damage_dealt": 0,
                "armor_gained": shield_gain,
                "attacker_new_armor": new_attacker_armor,
                "defender_new_hp": current_defender_hp,
                "description": f"Aktivovaný obranný štít: +{shield_gain} brnenia."
            }

        # 1. Determine Hit Threshold
        base_skill = attacker_stats.get("skill", 3) # e.g. 3+ base hit
        hit_modifier = 0

        # Sector Hit Modifier
        if sector in (CombatSector.FLANK_LEFT, CombatSector.FLANK_RIGHT):
            hit_modifier += 1 # +1 to hit from flank
        elif sector == CombatSector.REAR:
            hit_modifier += 2 # +2 to hit from rear (backstab)

        # High Ground Elevation Modifier
        if elevation_pitch_rad >= 0.12: # High ground angle
            hit_modifier += 1

        # Environmental Cover Modifier for Ranged
        cover_val = env_cell.cover_density if env_cell else 0.0
        if cover_val >= 0.35 and action_type in (CardActionType.RANGED_PROJECTILE, CardActionType.AOE_BOMBARDMENT):
            hit_modifier -= 1 # Cover impairs attacker aim

        effective_hit_threshold = max(2, min(6, base_skill - hit_modifier))

        # Roll or evaluate hit probability
        if d6_hit_roll is None:
            # Deterministic threshold met by default simulation
            hit_roll = effective_hit_threshold
        else:
            hit_roll = d6_hit_roll

        hit_success = (hit_roll >= effective_hit_threshold and hit_roll != 1) or (hit_roll == 6)

        if not hit_success:
            return {
                "action_type": action_type.value,
                "hit_success": False,
                "hit_roll": hit_roll,
                "required_roll": effective_hit_threshold,
                "damage_dealt": 0,
                "defender_new_hp": current_defender_hp,
                "description": f"Útok minul cieľ (Hod: {hit_roll}, Požadované: {effective_hit_threshold}+)."
            }

        # 2. Damage & Armor Mitigation Algebra
        effective_ap = card_ap
        crit_bonus = 0

        if sector in (CombatSector.FLANK_LEFT, CombatSector.FLANK_RIGHT):
            effective_ap += 1
        elif sector == CombatSector.REAR:
            # Backstab ignores armor completely and adds +1 crit
            effective_ap += 99
            crit_bonus += 1

        # High ground damage bonus
        if elevation_pitch_rad >= 0.20:
            crit_bonus += 1

        gross_damage = base_power + crit_bonus
        mitigated_armor = max(0, current_defender_armor - effective_ap)

        net_damage = max(0, gross_damage - mitigated_armor)
        new_defender_armor = max(0, current_defender_armor - gross_damage)

        # 3. ENFORCE 6 MAX HP INVARIANT
        new_defender_hp = max(0, min(6, current_defender_hp - net_damage))

        return {
            "action_type": action_type.value,
            "hit_success": True,
            "hit_roll": hit_roll,
            "required_roll": effective_hit_threshold,
            "gross_damage": gross_damage,
            "armor_absorbed": min(gross_damage, current_defender_armor),
            "net_damage_dealt": net_damage,
            "defender_previous_hp": current_defender_hp,
            "defender_new_hp": new_defender_hp,
            "defender_new_armor": new_defender_armor,
            "crit_applied": crit_bonus > 0,
            "sector": sector.value,
            "elevation_pitch_rad": round(elevation_pitch_rad, 3),
            "description": f"Útok zasiahol zo sektora {sector.value}! Udelené {net_damage} poškodenia (Cieľ: {new_defender_hp}/6 HP)."
        }
