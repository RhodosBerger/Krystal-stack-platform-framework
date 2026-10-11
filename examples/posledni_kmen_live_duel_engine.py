#!/usr/bin/env python3
"""
KRYSTAL-STACK // POSLEDNÍ KMEN LIVE DUEL ENGINE
=============================================================================
Live, executable combat and duel simulation engine designed to showcase
and operate within the Poslední Kmen game ecosystem (Godot 4.x, WebGL, CLI).

Integrates:
  1. Multi-axial Aspect Radar Profiles (OBT, KON, POŠ, DOS) from
     krystal_kernel.multi_disciplinary_measurement_summarizer.
  2. Official 43-card decks and templates from
     krystal_web_hub.economic_engine.posledni_kmen_cards.
  3. Action vs. Helper card composition dynamics.
  4. Strict Vital Invariant: VITAL_MAX_HP == 6 (6 discrete pips / hearts).
  5. Status effect algebra: Frozen, Stunned, Poisoned, Rooted, Thorns, Invulnerable.

Can be run directly via CLI or integrated into Godot / WebGL as a headless battle resolver.
=============================================================================
"""

import os
import sys
import random
import time
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.multi_disciplinary_measurement_summarizer import (
    GLOBAL_MEASUREMENT_SUMMARIZER,
    MeasurementDiscipline,
    MeasurementSubjectProfile
)
from krystal_web_hub.economic_engine.posledni_kmen_cards import (
    CRYSTAL_TRIBE_CARD_TEMPLATES,
    TOXIC_TRIBE_CARD_TEMPLATES,
    DRUID_TRIBE_CARD_TEMPLATES
)


@dataclass
class DuelFighter:
    name: str
    tribe_code: str             # "CRYSTAL", "TOXIC", "DRUID"
    title: str
    profile: MeasurementSubjectProfile
    hp: int = VITAL_MAX_HP      # Strictly clamped [0, 6]
    armor: int = 0
    mana: int = 3
    max_mana: int = 6
    stamina: int = 3
    status_effects: Dict[str, int] = field(default_factory=dict) # effect -> turns remaining
    deck: List[Dict[str, Any]] = field(default_factory=list)
    hand: List[Dict[str, Any]] = field(default_factory=list)
    discard_pile: List[Dict[str, Any]] = field(default_factory=list)

    def is_alive(self) -> bool:
        return self.hp > 0

    def apply_damage(self, raw_damage: int, damage_type: str = "physical") -> int:
        """Applies damage considering armor and status effects (e.g. invulnerable)."""
        if self.status_effects.get("invulnerable_bubble", 0) > 0:
            return 0

        # Armor absorbs first
        effective_damage = raw_damage
        if self.armor > 0:
            mitigated = min(self.armor, effective_damage)
            self.armor -= mitigated
            effective_damage -= mitigated

        self.hp = max(0, min(VITAL_MAX_HP, self.hp - effective_damage))
        return effective_damage

    def heal(self, amount: int) -> int:
        """Heals up to the strict invariant limit of 6 HP."""
        old_hp = self.hp
        self.hp = min(VITAL_MAX_HP, self.hp + amount)
        return self.hp - old_hp

    def draw_cards(self, count: int = 2) -> List[Dict[str, Any]]:
        drawn = []
        for _ in range(count):
            if not self.deck:
                if self.discard_pile:
                    self.deck = list(self.discard_pile)
                    random.shuffle(self.deck)
                    self.discard_pile.clear()
                else:
                    break
            if self.deck:
                card = self.deck.pop(0)
                self.hand.append(card)
                drawn.append(card)
        return drawn


@dataclass
class TurnActionRecord:
    turn_number: int
    actor_name: str
    action_type: str            # "CARD_PLAY", "STATUS_TICK", "RADAR_BURST"
    card_name: Optional[str]
    description: str
    damage_dealt: int = 0
    healing_done: int = 0
    armor_change: int = 0
    vital_pips_remaining: int = 6


class PosledniKmenLiveDuelEngine:
    """Headless match simulation engine incorporating multi-axial radar statistics."""

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.summarizer = GLOBAL_MEASUREMENT_SUMMARIZER
        self.combat_log: List[TurnActionRecord] = []

    def create_fighter(self, tribe_code: str, custom_name: Optional[str] = None) -> DuelFighter:
        tribe_map = {
            "CRYSTAL": ("PK-CRYSTAL", CRYSTAL_TRIBE_CARD_TEMPLATES, "Mrazivý Veštec"),
            "TOXIC": ("PK-TOXIC", TOXIC_TRIBE_CARD_TEMPLATES, "Hnijúci Šaman"),
            "DRUID": ("PK-DRUID", DRUID_TRIBE_CARD_TEMPLATES, "Strážca Hvozdu")
        }

        subj_id, templates, default_title = tribe_map.get(tribe_code.upper(), tribe_map["CRYSTAL"])
        profile = self.summarizer.get_profile(subj_id)
        if not profile:
            raise ValueError(f"Profile {subj_id} not registered in summarizer.")

        # Build 25-card deck based on reference composition (Actions vs. Helpers)
        deck = []
        for _ in range(25):
            deck.append(dict(random.choice(templates)))
        random.shuffle(deck)

        fighter = DuelFighter(
            name=custom_name or f"{tribe_code.capitalize()} Hrdina",
            tribe_code=tribe_code.upper(),
            title=default_title,
            profile=profile,
            hp=VITAL_MAX_HP,
            deck=deck
        )
        fighter.draw_cards(4) # Initial hand
        return fighter

    def simulate_match(self, fighter_a: DuelFighter, fighter_b: DuelFighter, max_turns: int = 20) -> Dict[str, Any]:
        """Runs turn-by-turn combat until victory or turn limit."""
        self.combat_log.clear()

        turn = 1
        active, passive = fighter_a, fighter_b

        # Determine first strike based on OBT + DOS aspect scores
        aspects_a = {a.code: a.normalized_score for a in fighter_a.profile.aspects}
        aspects_b = {a.code: a.normalized_score for a in fighter_b.profile.aspects}

        initiative_a = aspects_a.get("DOS", 4.0) * 1.2 + aspects_a.get("OBT", 3.0)
        initiative_b = aspects_b.get("DOS", 4.0) * 1.2 + aspects_b.get("OBT", 3.0)

        if initiative_b > initiative_a:
            active, passive = fighter_b, fighter_a

        while active.is_alive() and passive.is_alive() and turn <= max_turns:
            # 1. Start of Turn Phase: Mana refresh & draw
            active.mana = min(active.max_mana, active.mana + 2)
            active.draw_cards(1)

            # 2. Status Effects Tick (e.g. Poison DoT)
            if active.status_effects.get("poisoned", 0) > 0:
                dmg = active.apply_damage(1, "poison")
                self.combat_log.append(TurnActionRecord(
                    turn_number=turn,
                    actor_name=active.name,
                    action_type="STATUS_TICK",
                    card_name="Zranenie jedom",
                    description=f"{active.name} utrpel 1 bod poškodenia jedom.",
                    damage_dealt=dmg,
                    vital_pips_remaining=active.hp
                ))
                active.status_effects["poisoned"] -= 1

            if not active.is_alive():
                break

            # 3. Action Phase: Play highest affordable card
            played_card = None
            for card in list(active.hand):
                if card["cost"] <= active.mana:
                    played_card = card
                    active.hand.remove(card)
                    active.discard_pile.append(card)
                    active.mana -= card["cost"]
                    break

            if played_card:
                hp_delta = played_card.get("hp_delta", 0)
                armor_delta = played_card.get("armor_delta", 0)
                status = played_card.get("status_applied")

                damage_dealt = 0
                healing_done = 0

                # Offensive card (negative hp_delta)
                if hp_delta < 0:
                    raw_dmg = abs(hp_delta)
                    # Scale with POŠ aspect score
                    active_pos = {a.code: a.normalized_score for a in active.profile.aspects}.get("POŠ", 3.0)
                    if active_pos >= 4.5 and random.random() < 0.25:
                        raw_dmg += 1 # Critical burst

                    damage_dealt = passive.apply_damage(raw_dmg)
                    if status:
                        passive.status_effects[status] = 2

                # Defensive / Healing card
                elif hp_delta > 0:
                    healing_done = active.heal(hp_delta)

                if armor_delta > 0:
                    active.armor += armor_delta
                elif armor_delta < 0:
                    passive.armor = max(0, passive.armor + armor_delta)

                desc = f"{active.name} vyložil [{played_card['name']}]."
                if damage_dealt > 0:
                    desc += f" Spôsobil {damage_dealt} zranenia ({passive.name} má {passive.hp}/6 HP)."
                if healing_done > 0:
                    desc += f" Vyliečil {healing_done} HP ({active.name} má {active.hp}/6 HP)."
                if armor_delta > 0:
                    desc += f" Získal +{armor_delta} brnenia."

                self.combat_log.append(TurnActionRecord(
                    turn_number=turn,
                    actor_name=active.name,
                    action_type="CARD_PLAY",
                    card_name=played_card["name"],
                    description=desc,
                    damage_dealt=damage_dealt,
                    healing_done=healing_done,
                    vital_pips_remaining=passive.hp if damage_dealt > 0 else active.hp
                ))
            else:
                # Basic attack using DOS aspect
                damage_dealt = passive.apply_damage(1)
                self.combat_log.append(TurnActionRecord(
                    turn_number=turn,
                    actor_name=active.name,
                    action_type="BASIC_ATTACK",
                    card_name="Základný Výpad",
                    description=f"{active.name} nemá dostatok many, použil základný výpad za 1 poškodenie.",
                    damage_dealt=damage_dealt,
                    vital_pips_remaining=passive.hp
                ))

            # Swap turns
            active, passive = passive, active
            turn += 1

        winner = None
        if fighter_a.hp > 0 and fighter_b.hp == 0:
            winner = fighter_a.name
        elif fighter_b.hp > 0 and fighter_a.hp == 0:
            winner = fighter_b.name
        else:
            winner = fighter_a.name if fighter_a.hp >= fighter_b.hp else fighter_b.name

        return {
            "status": "COMPLETED",
            "winner": winner,
            "turns_count": turn - 1,
            "fighter_a": {
                "name": fighter_a.name,
                "tribe": fighter_a.tribe_code,
                "final_hp": fighter_a.hp,
                "radar_u": fighter_a.profile.weighted_composite_index
            },
            "fighter_b": {
                "name": fighter_b.name,
                "tribe": fighter_b.tribe_code,
                "final_hp": fighter_b.hp,
                "radar_u": fighter_b.profile.weighted_composite_index
            },
            "combat_log": [
                {
                    "turn": r.turn_number,
                    "actor": r.actor_name,
                    "type": r.action_type,
                    "card": r.card_name,
                    "description": r.description,
                    "target_hp": r.vital_pips_remaining
                }
                for r in self.combat_log
            ],
            "vital_max_hp_preserved": max(fighter_a.hp, fighter_b.hp) <= VITAL_MAX_HP
        }


def print_cli_showcase():
    """Runs an interactive CLI demonstration of a tribal duel."""
    engine = PosledniKmenLiveDuelEngine()
    f1 = engine.create_fighter("CRYSTAL", "Kryštálová Kráľovná")
    f2 = engine.create_fighter("TOXIC", "Močiarny Pán")

    print("\n" + "=" * 76)
    print("  POSLEDNÍ KMEN // LIVE GAME ENGINE COMBAT SHOWCASE")
    print("=" * 76)
    print(f"  FIGHTER 1: {f1.name} ({f1.tribe_code}) | Index U: {f1.profile.weighted_composite_index} | HP: {f1.hp}/{VITAL_MAX_HP}")
    print(f"  FIGHTER 2: {f2.name} ({f2.tribe_code}) | Index U: {f2.profile.weighted_composite_index} | HP: {f2.hp}/{VITAL_MAX_HP}")
    print("-" * 76)

    res = engine.simulate_match(f1, f2)

    for record in res["combat_log"]:
        print(f"  [Turn {record['turn']:02d}] {record['actor']}: {record['description']}")

    print("-" * 76)
    print(f"  VICTORY DECLARED: {res['winner']} v {res['turns_count']} kolách!")
    print(f"  STRICT INVARIANT: VITAL_MAX_HP == {VITAL_MAX_HP} (PRESERVED: {res['vital_max_hp_preserved']})")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    print_cli_showcase()
