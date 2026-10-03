# ==============================================================================
# KRYSTAL-STACK: WARHAMMER-INSPIRED STATISTICAL COMBAT MODELS
# ==============================================================================
# Implements exact probabilistic distributions, D6 / 2D6 / D20 dice mechanics,
# Hit-Wound-Save matrices, expected damage formulas, variance, critical rolls,
# and Monte Carlo combat resolution for tactical hex wargaming.
# ==============================================================================

from typing import Dict, List, Tuple, Any, Optional
import math
import random

# ------------------------------------------------------------------------------
# 1. WARHAMMER HIT, WOUND, AND SAVE RESOLUTION MATRICES
# ------------------------------------------------------------------------------

def get_to_hit_threshold(ballistic_or_weapon_skill: int, modifier: int = 0) -> int:
    """
    Computes required D6 roll to hit based on Ballistic Skill (BS) or Weapon Skill (WS).
    e.g. BS 3+ with -1 modifier (moving and heavy) becomes 4+.
    Natural 1 always fails; natural 6 always hits.
    Capped between 2 and 6.
    """
    # Base target: e.g. skill 3 means 3+
    effective = ballistic_or_weapon_skill - modifier
    return max(2, min(6, effective))

def get_to_wound_threshold(strength: int, toughness: int, modifier: int = 0) -> int:
    """
    Standard Warhammer Strength vs Toughness comparison matrix:
      - Strength >= 2 * Toughness -> 2+
      - Strength > Toughness      -> 3+
      - Strength == Toughness     -> 4+
      - Strength < Toughness      -> 5+
      - Strength <= Toughness / 2 -> 6+
    Modifier shifts required threshold (+1 modifier makes 4+ into 3+).
    Natural 1 always fails; natural 6 always wounds.
    """
    if strength >= 2 * toughness:
        base = 2
    elif strength > toughness:
        base = 3
    elif strength == toughness:
        base = 4
    elif strength * 2 <= toughness:
        base = 6
    else: # strength < toughness
        base = 5

    effective = base - modifier
    return max(2, min(6, effective))

def get_armor_save_threshold(armor_save: int, armor_penetration: int, invulnerable_save: Optional[int] = None) -> Tuple[int, bool]:
    """
    Computes required D6 armor save roll against Armor Penetration (AP).
    AP is positive integer reducing save (e.g. Save 3+, AP 2 -> modified save 5+).
    If modified save > 6, unit can only save if it has an Invulnerable Save.
    Returns (required_roll, is_invulnerable).
    """
    modified_save = armor_save + armor_penetration
    if invulnerable_save is not None and invulnerable_save < modified_save:
        return (max(2, min(6, invulnerable_save)), True)
    return (modified_save, False)


# ------------------------------------------------------------------------------
# 2. PROBABILITY DENSITY AND EXPECTED VALUE MATH
# ------------------------------------------------------------------------------

def d6_probability_ge(threshold: int) -> float:
    """Returns probability of rolling >= threshold on a fair D6."""
    if threshold <= 1:
        return 5.0 / 6.0 # Natural 1 always fails
    if threshold > 6:
        return 0.0
    return (7 - threshold) / 6.0

def calculate_expected_damage(
    attacks: int,
    skill: int,
    strength: int,
    toughness: int,
    damage_per_wound: float,
    armor_save: int,
    armor_penetration: int,
    invulnerable_save: Optional[int] = None,
    hit_modifier: int = 0,
    wound_modifier: int = 0,
    crit_on_six: bool = True,
    lethal_hits: bool = False
) -> Dict[str, float]:
    """
    Calculates exact expected damage and stage-by-stage success probabilities:
      E[Damage] = Attacks * P(Hit) * P(Wound) * P(Failed Save) * DamagePerWound
    """
    hit_thresh = get_to_hit_threshold(skill, hit_modifier)
    p_hit = d6_probability_ge(hit_thresh)

    wound_thresh = get_to_wound_threshold(strength, toughness, wound_modifier)
    p_wound = d6_probability_ge(wound_thresh)

    # Lethal hits: natural 6 to-hit auto-wounds
    p_six = 1.0 / 6.0
    if lethal_hits:
        p_wound_effective = p_six + (p_hit - p_six) * p_wound
    else:
        p_wound_effective = p_hit * p_wound

    save_thresh, is_invuln = get_armor_save_threshold(armor_save, armor_penetration, invulnerable_save)
    if save_thresh > 6:
        p_save = 0.0
    else:
        p_save = d6_probability_ge(save_thresh)
    p_fail_save = 1.0 - p_save

    total_conversion_prob = p_wound_effective * p_fail_save
    expected_unsaved_wounds = attacks * total_conversion_prob
    expected_damage = expected_unsaved_wounds * damage_per_wound

    # Variance for Binomial conversion: Var(X) = n * p * (1-p) * D^2
    variance = attacks * total_conversion_prob * (1.0 - total_conversion_prob) * (damage_per_wound ** 2)
    std_dev = math.sqrt(variance)

    return {
        "p_hit": round(p_hit, 4),
        "p_wound": round(p_wound, 4),
        "p_fail_save": round(p_fail_save, 4),
        "single_attack_conversion": round(total_conversion_prob, 4),
        "expected_wounds": round(expected_unsaved_wounds, 4),
        "expected_damage": round(expected_damage, 4),
        "variance": round(variance, 4),
        "std_dev": round(std_dev, 4),
        "hit_threshold": hit_thresh,
        "wound_threshold": wound_thresh,
        "save_threshold": save_thresh,
        "used_invulnerable": is_invuln
    }


# ------------------------------------------------------------------------------
# 3. 2D6 CHARGE AND HAZARD PROBABILITIES
# ------------------------------------------------------------------------------

def probability_2d6_ge(target_distance: int) -> float:
    """
    Returns exact probability of rolling >= target_distance on 2D6 (Warhammer Charge roll).
    Possible outcomes: 36.
    """
    if target_distance <= 2:
        return 1.0
    if target_distance > 12:
        return 0.0

    count = 0
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            if d1 + d2 >= target_distance:
                count += 1
    return count / 36.0


# ------------------------------------------------------------------------------
# 4. MONTE CARLO COMBAT SIMULATOR
# ------------------------------------------------------------------------------

class CombatSimulationModel:
    """Simulates multi-round combat volleys with exact D6 rolls, crits, and saves."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def roll_d6(self) -> int:
        return self.rng.randint(1, 6)

    def roll_2d6(self) -> int:
        return self.roll_d6() + self.roll_d6()

    def simulate_attack_sequence(
        self,
        attacks: int,
        skill: int,
        strength: int,
        toughness: int,
        damage_per_wound: int,
        armor_save: int,
        armor_penetration: int,
        invulnerable_save: Optional[int] = None,
        hit_modifier: int = 0,
        wound_modifier: int = 0,
        lethal_hits: bool = False,
        sustained_hits: int = 0
    ) -> Dict[str, Any]:
        """Runs an individual discrete wargame attack sequence."""
        hit_thresh = get_to_hit_threshold(skill, hit_modifier)
        wound_thresh = get_to_wound_threshold(strength, toughness, wound_modifier)
        save_thresh, is_invuln = get_armor_save_threshold(armor_save, armor_penetration, invulnerable_save)

        hits = 0
        auto_wounds = 0
        bonus_attacks = 0

        for _ in range(attacks):
            roll = self.roll_d6()
            if roll == 1:
                continue # Natural 1 fails
            if roll == 6:
                if lethal_hits:
                    auto_wounds += 1
                else:
                    hits += 1
                if sustained_hits > 0:
                    bonus_attacks += sustained_hits
            elif roll >= hit_thresh:
                hits += 1

        # Resolve sustained hit bonus attacks
        for _ in range(bonus_attacks):
            roll = self.roll_d6()
            if roll != 1 and roll >= hit_thresh:
                hits += 1

        # Wound rolls
        wounds = auto_wounds
        for _ in range(hits):
            roll = self.roll_d6()
            if roll != 1 and (roll == 6 or roll >= wound_thresh):
                wounds += 1

        # Save rolls
        unsaved = 0
        for _ in range(wounds):
            if save_thresh > 6:
                unsaved += 1
            else:
                roll = self.roll_d6()
                if roll == 1 or roll < save_thresh:
                    unsaved += 1

        total_damage = unsaved * damage_per_wound
        return {
            "initial_attacks": attacks,
            "bonus_attacks": bonus_attacks,
            "total_hits": hits + auto_wounds,
            "total_wounds": wounds,
            "unsaved_wounds": unsaved,
            "total_damage": total_damage
        }

    def run_monte_carlo(
        self,
        iterations: int,
        attacks: int,
        skill: int,
        strength: int,
        toughness: int,
        damage_per_wound: int,
        armor_save: int,
        armor_penetration: int,
        invulnerable_save: Optional[int] = None,
        target_hp: int = 6
    ) -> Dict[str, Any]:
        """Executes N iterations and computes empirical kill probability and damage stats."""
        damages = []
        kills = 0

        for _ in range(iterations):
            res = self.simulate_attack_sequence(
                attacks=attacks,
                skill=skill,
                strength=strength,
                toughness=toughness,
                damage_per_wound=damage_per_wound,
                armor_save=armor_save,
                armor_penetration=armor_penetration,
                invulnerable_save=invulnerable_save
            )
            d = res["total_damage"]
            damages.append(d)
            if d >= target_hp:
                kills += 1

        mean_damage = sum(damages) / iterations
        variance = sum((x - mean_damage) ** 2 for x in damages) / iterations
        kill_rate = kills / iterations

        return {
            "iterations": iterations,
            "mean_damage": round(mean_damage, 4),
            "std_dev": round(math.sqrt(variance), 4),
            "kill_rate": round(kill_rate, 4),
            "min_damage": min(damages),
            "max_damage": max(damages)
        }
