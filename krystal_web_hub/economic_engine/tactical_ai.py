# ==============================================================================
# KRYSTAL-STACK: TACTICAL AI ENGINE (Poslední Kmen)
# ==============================================================================
# Autonomous heuristic decision-making engine for AI combatants.
# Evaluates utility, enforces axial hex distance range bounds, and balances
# offensive strikes, defensive mitigation, and mana economy.
# ==============================================================================

import math
from typing import Dict, List, Any, Optional, Tuple
from .models import Tribe, AttackType, MatchState, CombatantState
from .ability_framework import calculate_hex_distance, ABILITY_REGISTRY

class TacticalAIEngine:
    """Heuristic utility engine for Poslední Kmen tactical combat."""

    @staticmethod
    def evaluate_ai_turn(
        match: MatchState,
        ai_side: str = "enemy",
        available_cards: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Determines the optimal tactical action for the AI combatant.
        Evaluates distance, mana reserves, player threat level, and lethal opportunities.
        """
        ai_combatant: CombatantState = match.enemy if ai_side == "enemy" else match.player
        target_combatant: CombatantState = match.player if ai_side == "enemy" else match.enemy

        ai_hex = [0, 2] # Default enemy home hex
        target_hex = [0, -2] # Default player home hex

        # In standard lane combat across the 19-hex board, base-to-base artillery engagement distance is 3 (via center nexus)
        base_dist = calculate_hex_distance(ai_hex, target_hex)
        distance = min(3, base_dist)
        available_mana = ai_combatant.mana
        ai_hp = ai_combatant.hp
        target_hp = target_combatant.hp

        # If available_cards not passed, select tribal abilities matching AI tribe
        if not available_cards:
            available_cards = TacticalAIEngine._get_default_ai_abilities(ai_combatant.tribe)

        # 1. Filter playable actions based on mana and range
        playable_actions = []
        for card in available_cards:
            cost = card.get("cost", 1)
            if isinstance(cost, dict):
                mana_cost = cost.get("mana", 1)
            else:
                mana_cost = int(cost)

            if mana_cost > available_mana:
                continue

            atype = card.get("attack_type", "ranged")
            min_r = card.get("min_range", 1)
            max_r = card.get("max_range", 3)

            is_in_range = False
            if atype == "self":
                is_in_range = True
            elif atype == "global":
                is_in_range = True
            elif atype == "melee":
                is_in_range = (distance == 1)
            else: # ranged / aoe
                is_in_range = (min_r <= distance <= max_r)

            if is_in_range:
                playable_actions.append(card)

        # 2. Heuristic Scoring
        best_action = None
        best_score = -999.0
        reasoning = "Čakanie na lepšiu príležitosť."

        for action in playable_actions:
            score = 0.0
            dmg = abs(action.get("damage", action.get("hp_delta", 0)))
            shield = action.get("shield", action.get("armor_delta", 0))
            cost = action.get("cost", 1)
            if isinstance(cost, dict):
                cost = cost.get("mana", 1)

            # Heuristic 1: Lethal Strike Priority (Weight = 100)
            if dmg >= target_hp:
                score += 100.0 + dmg

            # Heuristic 2: Emergency Defense (Weight = 50 if HP <= 2)
            if ai_hp <= 2 and shield > 0:
                score += 50.0 + (shield * 10.0)

            # Heuristic 3: Damage Efficiency (Damage per Mana)
            if cost > 0:
                score += (dmg / float(cost)) * 15.0
            else:
                score += dmg * 15.0

            # Heuristic 4: Status application
            if action.get("status_applied") in ["rooted", "stunned", "poisoned"]:
                score += 12.0

            if score > best_score:
                best_score = score
                best_action = action
                if dmg >= target_hp:
                    reasoning = f"Smrteľný úder! Zničenie cieľa pomocou {action.get('name')}."
                elif ai_hp <= 2 and shield > 0:
                    reasoning = f"Kritický stav HP ({ai_hp}/6). Aktivácia obrannej bariéry {action.get('name')}."
                else:
                    reasoning = f"Optimálny taktický úder {action.get('name')} (Poškodenie: {dmg}, Cena: {cost})."

        # 3. If no action can be cast (e.g. out of range or insufficient mana)
        if not best_action:
            return {
                "executed": False,
                "action_type": "skip",
                "card_id": None,
                "card_name": "Pass",
                "target_hex": target_hex,
                "mana_spent": 0,
                "damage_dealt": 0,
                "armor_gained": 0,
                "distance": distance,
                "reasoning": f"Žiadna akcia v dosahu (Vzdialenosť: {distance} hexov, Mana: {available_mana}). AI končí ťah."
            }

        # 4. Resolve Best Action
        dmg = abs(best_action.get("damage", best_action.get("hp_delta", 0)))
        shield = best_action.get("shield", best_action.get("armor_delta", 0))
        cost = best_action.get("cost", 1)
        if isinstance(cost, dict): cost = cost.get("mana", 1)

        # Deduct mana
        ai_combatant.mana = max(0, ai_combatant.mana - cost)

        # Apply damage to target
        if dmg > 0:
            if target_combatant.armor >= dmg:
                target_combatant.armor -= dmg
            else:
                remaining_dmg = dmg - target_combatant.armor
                target_combatant.armor = 0
                target_combatant.hp = max(0, target_combatant.hp - remaining_dmg)

        # Apply shield to AI
        if shield > 0:
            ai_combatant.armor += shield

        return {
            "executed": True,
            "action_type": best_action.get("attack_type", "ranged"),
            "card_id": best_action.get("id"),
            "card_name": best_action.get("name"),
            "source_hex": ai_hex,
            "target_hex": target_hex if best_action.get("attack_type") != "self" else ai_hex,
            "mana_spent": cost,
            "damage_dealt": dmg,
            "armor_gained": shield,
            "target_hp_left": target_combatant.hp,
            "target_armor_left": target_combatant.armor,
            "distance": distance,
            "reasoning": reasoning
        }

    @staticmethod
    def _get_default_ai_abilities(tribe: Tribe) -> List[Dict[str, Any]]:
        """Provides default action profiles based on tribe alignment."""
        abilities = []
        for ab_id, ab in ABILITY_REGISTRY.items():
            if ab.tribe == tribe or ab.tribe == Tribe.NEUTRAL:
                abilities.append(ab.to_dict())
        return abilities
