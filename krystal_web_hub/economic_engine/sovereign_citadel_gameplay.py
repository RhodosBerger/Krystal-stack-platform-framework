"""
Sovereign Citadel: Gameplay Mechanics & Tactical Tower Defense Engine
Based on the Sovereign Cyber Fortress Graphic (wordpress_subdomain_security_shield.jpg)
Strictly adheres to the platform-wide invariant: VITAL_MAX_HP = 6.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import math
import random
import time

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 1.0 / GOLDEN_RATIO


@dataclass
class TowerPylon:
    id: str
    name: str
    pylon_type: str  # "slavic_mortar" | "hellenic_beam"
    x: float
    y: float
    range_px: float
    damage: float
    attack_cooldown_s: float
    last_attack_time: float = 0.0
    kills_count: int = 0
    total_damage_dealt: float = 0.0


@dataclass
class CitadelCore:
    hp: int = 6
    max_hp: int = 6
    shield: float = 4.0
    max_shield: float = 12.0
    mana: int = 10
    max_mana: int = 10
    vital_max_hp_rule: int = 6

    def repair_hp(self, amount: int = 1) -> int:
        self.hp = min(self.max_hp, self.hp + amount)
        return self.hp

    def add_shield(self, amount: float) -> float:
        self.shield = min(self.max_shield, self.shield + amount)
        return self.shield

    def take_damage(self, amount: float) -> Dict[str, Any]:
        absorbed = min(self.shield, amount)
        self.shield -= absorbed
        remaining_dmg = amount - absorbed
        hp_lost = int(math.ceil(remaining_dmg / 2.0)) if remaining_dmg > 0 else 0
        if hp_lost > 0:
            self.hp = max(0, self.hp - hp_lost)
        return {
            "absorbed_by_shield": absorbed,
            "hp_lost": hp_lost,
            "current_hp": self.hp,
            "current_shield": self.shield,
            "destroyed": self.hp <= 0
        }


@dataclass
class InvaderPacket:
    id: str
    unit_type: str  # "sqli_ram" | "xss_swarm" | "rce_stalker" | "ddos_colossus"
    name: str
    hp: float
    max_hp: float
    speed_px_per_s: float
    damage: float
    x: float
    y: float
    target_x: float
    target_y: float
    is_alive: bool = True
    is_frozen: bool = False
    freeze_duration_s: float = 0.0
    gold_reward: int = 1


@dataclass
class TacticalCard:
    id: str
    name: str
    mana_cost: int
    archetype: str
    description: str
    effect_type: str  # "shield_boost" | "mortar_strike" | "freeze_wave" | "firewall_purge" | "mana_surge" | "repair_vital"
    power: float


@dataclass
class GameProjectile:
    id: str
    source_pylon_id: str
    start_x: float
    start_y: float
    target_x: float
    target_y: float
    damage: float
    projectile_type: str  # "mortar_arc" | "tactical_laser"
    progress: float = 0.0
    speed: float = 0.05
    is_active: bool = True


class SovereignCitadelEngine:
    """
    Simulates the Sovereign Shield Citadel tactical defense encounter.
    Features:
      - 2 Defending Pylons (Perun's Eye & Athena's Owl)
      - Dynamic Sieging Packet waves
      - 6 Tactical Cards directly referencing cryptographic defenses
      - Strict preservation of VITAL_MAX_HP = 6
    """

    def __init__(self):
        self.vital_max_hp: int = VITAL_MAX_HP
        self.citadel = CitadelCore(hp=6, max_hp=6, shield=4.0, max_shield=12.0, mana=10, max_mana=10)
        self.wave_number: int = 1
        self.score: int = 0
        self.is_game_over: bool = False
        self.is_victory: bool = False

        # Initialize Default Pylons
        self.pylons: Dict[str, TowerPylon] = {
            "pylon_left": TowerPylon(
                id="pylon_left",
                name="Perun's Slavic Mortar Pylon",
                pylon_type="slavic_mortar",
                x=180.0,
                y=380.0,
                range_px=420.0,
                damage=3.0,
                attack_cooldown_s=0.8
            ),
            "pylon_right": TowerPylon(
                id="pylon_right",
                name="Athena's Tactical Beam Pylon",
                pylon_type="hellenic_beam",
                x=720.0,
                y=380.0,
                range_px=480.0,
                damage=1.8,
                attack_cooldown_s=0.4
            )
        }

        # Cards Catalog
        self.card_catalog: Dict[str, TacticalCard] = {
            "card_aegis_wall": TacticalCard(
                id="card_aegis_wall",
                name="Sovereign Aegis Wall",
                mana_cost=2,
                archetype="Cryptographic Barrier",
                description="Generates +4 holographic shield points around the Core Gateway.",
                effect_type="shield_boost",
                power=4.0
            ),
            "card_perun_mortar": TacticalCard(
                id="card_perun_mortar",
                name="Perun's Lightning Barrage",
                mana_cost=3,
                archetype="Slavic Artillery",
                description="Strikes target area with plunging thunderbolts dealing 4 AOE damage.",
                effect_type="mortar_strike",
                power=4.0
            ),
            "card_athena_freeze": TacticalCard(
                id="card_athena_freeze",
                name="Athena's Tactical Freeze",
                mana_cost=2,
                archetype="Hellenic Axiom",
                description="Slows and freezes all hostile packets for 3.5 seconds.",
                effect_type="freeze_wave",
                power=3.5
            ),
            "card_bbq_purge": TacticalCard(
                id="card_bbq_purge",
                name="BBQ Firewall Purge",
                mana_cost=4,
                archetype="WAF Core",
                description="Blasts all invading packets with 3 damage across the entire field.",
                effect_type="firewall_purge",
                power=3.0
            ),
            "card_ledger_surge": TacticalCard(
                id="card_ledger_surge",
                name="Ledger Mana Surge",
                mana_cost=1,
                archetype="Double-Entry Accounting",
                description="Reclaims unused compute cycles, restoring +3 Mana immediately.",
                effect_type="mana_surge",
                power=3.0
            ),
            "card_vital_restore": TacticalCard(
                id="card_vital_restore",
                name="Cryptographic Repair",
                mana_cost=3,
                archetype="Spinal Reflex",
                description="Restores 1 Core HP, strictly preserving VITAL_MAX_HP = 6.",
                effect_type="repair_vital",
                power=1.0
            )
        }

        self.hand: List[str] = ["card_aegis_wall", "card_perun_mortar", "card_athena_freeze", "card_bbq_purge"]
        self.active_invaders: List[InvaderPacket] = []
        self.active_projectiles: List[GameProjectile] = []
        self.last_update_ts: float = time.time()

        # Spawn initial wave 1
        self.spawn_wave(1)

    def spawn_wave(self, wave_num: int) -> List[Dict[str, Any]]:
        self.wave_number = wave_num
        self.active_invaders.clear()
        self.active_projectiles.clear()

        # Target Core Gateway is at center bottom: (450, 480)
        target_x, target_y = 450.0, 480.0

        num_units = 4 + wave_num * 2
        rng = random.Random(wave_num * 101)

        types_pool = ["sqli_ram", "xss_swarm", "rce_stalker"]
        if wave_num % 3 == 0:
            types_pool.append("ddos_colossus")

        for i in range(num_units):
            u_type = rng.choice(types_pool)
            start_x = rng.uniform(80.0, 820.0)
            start_y = rng.uniform(-150.0, -20.0) - (i * 35.0)

            if u_type == "sqli_ram":
                hp = 6.0 + wave_num * 1.5
                speed = 22.0
                dmg = 2.0
                name = "SQLi Battering Ram"
                reward = 2
            elif u_type == "xss_swarm":
                hp = 2.5 + wave_num * 0.8
                speed = 55.0
                dmg = 1.0
                name = "XSS Viper Packet"
                reward = 1
            elif u_type == "rce_stalker":
                hp = 4.0 + wave_num * 1.0
                speed = 38.0
                dmg = 2.5
                name = "RCE Phantom Infiltrator"
                reward = 3
            else:  # ddos_colossus
                hp = 18.0 + wave_num * 4.0
                speed = 15.0
                dmg = 4.0
                name = "DDoS Flooding Colossus"
                reward = 5

            self.active_invaders.append(
                InvaderPacket(
                    id=f"invader_{wave_num}_{i}",
                    unit_type=u_type,
                    name=name,
                    hp=hp,
                    max_hp=hp,
                    speed_px_per_s=speed,
                    damage=dmg,
                    x=start_x,
                    y=start_y,
                    target_x=target_x,
                    target_y=target_y,
                    gold_reward=reward
                )
            )

        # Replenish mana on wave start
        self.citadel.mana = min(self.citadel.max_mana, self.citadel.mana + 4)
        return [self._serialize_invader(inv) for inv in self.active_invaders]

    def play_card(self, card_id: str, target_coords: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        if card_id not in self.card_catalog:
            return {"success": False, "error": f"Unknown card: {card_id}"}

        card = self.card_catalog[card_id]
        if self.citadel.mana < card.mana_cost:
            return {"success": False, "error": "Nedostatok Mana energie pre zahranie karty"}

        self.citadel.mana -= card.mana_cost
        result_details: Dict[str, Any] = {"card_id": card_id, "name": card.name, "effect": card.effect_type}

        if card.effect_type == "shield_boost":
            new_shield = self.citadel.add_shield(card.power)
            result_details["new_shield"] = new_shield
            result_details["message"] = f"Aegis Shield zosilnený na {new_shield} bodov."

        elif card.effect_type == "mortar_strike":
            tx = target_coords.get("x", 450.0) if target_coords else 450.0
            ty = target_coords.get("y", 200.0) if target_coords else 200.0
            radius = 120.0
            hits = 0
            for inv in self.active_invaders:
                if not inv.is_alive:
                    continue
                dist = math.hypot(inv.x - tx, inv.y - ty)
                if dist <= radius:
                    inv.hp -= card.power
                    hits += 1
                    if inv.hp <= 0:
                        inv.is_alive = False
                        self.score += inv.gold_reward * 10
            result_details["targets_hit"] = hits
            result_details["message"] = f"Perunov bleskový úder zasiahol {hits} nepriateľských paketov."

        elif card.effect_type == "freeze_wave":
            for inv in self.active_invaders:
                if inv.is_alive:
                    inv.is_frozen = True
                    inv.freeze_duration_s = card.power
            result_details["message"] = f"Všetky pakety zmrazené na {card.power} sekúnd."

        elif card.effect_type == "firewall_purge":
            killed = 0
            for inv in self.active_invaders:
                if inv.is_alive:
                    inv.hp -= card.power
                    if inv.hp <= 0:
                        inv.is_alive = False
                        killed += 1
                        self.score += inv.gold_reward * 10
            result_details["killed_units"] = killed
            result_details["message"] = f"BBQ Firewall prečistil pole a zničil {killed} paketov."

        elif card.effect_type == "mana_surge":
            self.citadel.mana = min(self.citadel.max_mana, self.citadel.mana + int(card.power))
            result_details["new_mana"] = self.citadel.mana
            result_details["message"] = f"Mana navýšená o +{card.power} (Aktuálna: {self.citadel.mana})."

        elif card.effect_type == "repair_vital":
            new_hp = self.citadel.repair_hp(int(card.power))
            result_details["new_hp"] = new_hp
            result_details["vital_max_hp_rule"] = VITAL_MAX_HP
            result_details["message"] = f"Jadro brány opravené na {new_hp}/6 HP."

        # Rotate card out of hand and draw replacement
        if card_id in self.hand:
            self.hand.remove(card_id)
            available = [c for c in self.card_catalog.keys() if c not in self.hand]
            if available:
                self.hand.append(random.choice(available))

        return {
            "success": True,
            "details": result_details,
            "citadel_state": self.get_state()["citadel"]
        }

    def update_simulation_tick(self, delta_time_s: float = 0.05) -> Dict[str, Any]:
        """
        Advances the tactical defense simulation by delta_time_s.
        """
        if self.is_game_over or self.is_victory:
            return self.get_state()

        now = time.time()

        # 1. Update Invaders Movement
        for inv in self.active_invaders:
            if not inv.is_alive:
                continue

            if inv.is_frozen:
                inv.freeze_duration_s -= delta_time_s
                if inv.freeze_duration_s <= 0:
                    inv.is_frozen = False
                continue

            # Move towards target
            dx = inv.target_x - inv.x
            dy = inv.target_y - inv.y
            dist = math.hypot(dx, dy)

            if dist <= 30.0:  # Reached the core gateway
                # Deal damage to core
                dmg_res = self.citadel.take_damage(inv.damage)
                inv.is_alive = False
                if dmg_res["destroyed"]:
                    self.is_game_over = True
            else:
                vx = (dx / dist) * inv.speed_px_per_s * delta_time_s
                vy = (dy / dist) * inv.speed_px_per_s * delta_time_s
                inv.x += vx
                inv.y += vy

        # 2. Update Defensive Pylons Auto-Targeting
        for pylon in self.pylons.values():
            if now - pylon.last_attack_time >= pylon.attack_cooldown_s:
                # Find closest live target in range
                candidates = [
                    inv for inv in self.active_invaders
                    if inv.is_alive and math.hypot(inv.x - pylon.x, inv.y - pylon.y) <= pylon.range_px
                ]
                if candidates:
                    target = min(candidates, key=lambda u: math.hypot(u.x - pylon.x, u.y - pylon.y))
                    target.hp -= pylon.damage
                    pylon.last_attack_time = now
                    pylon.total_damage_dealt += pylon.damage

                    # Spawn projectile
                    self.active_projectiles.append(
                        GameProjectile(
                            id=f"proj_{now}_{pylon.id}",
                            source_pylon_id=pylon.id,
                            start_x=pylon.x,
                            start_y=pylon.y,
                            target_x=target.x,
                            target_y=target.y,
                            damage=pylon.damage,
                            projectile_type="mortar_arc" if pylon.pylon_type == "slavic_mortar" else "tactical_laser"
                        )
                    )

                    if target.hp <= 0:
                        target.is_alive = False
                        pylon.kills_count += 1
                        self.score += target.gold_reward * 10

        # 3. Clean up dead projectiles
        self.active_projectiles = [p for p in self.active_projectiles if p.is_active]

        # 4. Check wave completion
        alive_count = sum(1 for inv in self.active_invaders if inv.is_alive)
        if alive_count == 0 and not self.is_game_over:
            if self.wave_number >= 10:
                self.is_victory = True
            else:
                self.spawn_wave(self.wave_number + 1)

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        return {
            "success": True,
            "wave_number": self.wave_number,
            "score": self.score,
            "is_game_over": self.is_game_over,
            "is_victory": self.is_victory,
            "citadel": {
                "hp": self.citadel.hp,
                "max_hp": self.citadel.max_hp,
                "shield": round(self.citadel.shield, 1),
                "max_shield": self.citadel.max_shield,
                "mana": self.citadel.mana,
                "max_mana": self.citadel.max_mana,
                "vital_max_hp_rule": VITAL_MAX_HP
            },
            "pylons": [
                {
                    "id": p.id,
                    "name": p.name,
                    "type": p.pylon_type,
                    "x": p.x,
                    "y": p.y,
                    "range": p.range_px,
                    "kills": p.kills_count,
                    "total_damage": round(p.total_damage_dealt, 1)
                }
                for p in self.pylons.values()
            ],
            "hand_cards": [
                {
                    "id": self.card_catalog[cid].id,
                    "name": self.card_catalog[cid].name,
                    "mana_cost": self.card_catalog[cid].mana_cost,
                    "archetype": self.card_catalog[cid].archetype,
                    "description": self.card_catalog[cid].description,
                    "effect_type": self.card_catalog[cid].effect_type,
                    "power": self.card_catalog[cid].power
                }
                for cid in self.hand if cid in self.card_catalog
            ],
            "invaders": [self._serialize_invader(inv) for inv in self.active_invaders if inv.is_alive],
            "projectiles": [
                {
                    "id": pr.id,
                    "start_x": pr.start_x,
                    "start_y": pr.start_y,
                    "target_x": pr.target_x,
                    "target_y": pr.target_y,
                    "type": pr.projectile_type
                }
                for pr in self.active_projectiles
            ]
        }

    def reset_game(self) -> Dict[str, Any]:
        self.citadel = CitadelCore(hp=6, max_hp=6, shield=4.0, max_shield=12.0, mana=10, max_mana=10)
        self.wave_number = 1
        self.score = 0
        self.is_game_over = False
        self.is_victory = False
        self.hand = ["card_aegis_wall", "card_perun_mortar", "card_athena_freeze", "card_bbq_purge"]
        for p in self.pylons.values():
            p.kills_count = 0
            p.total_damage_dealt = 0.0
            p.last_attack_time = 0.0
        self.spawn_wave(1)
        return self.get_state()

    def _serialize_invader(self, inv: InvaderPacket) -> Dict[str, Any]:
        return {
            "id": inv.id,
            "type": inv.unit_type,
            "name": inv.name,
            "hp": round(inv.hp, 1),
            "max_hp": inv.max_hp,
            "x": round(inv.x, 1),
            "y": round(inv.y, 1),
            "speed": inv.speed_px_per_s,
            "is_frozen": inv.is_frozen,
            "reward": inv.gold_reward
        }


# Global Singleton Instance for Engine Hub
GLOBAL_SOVEREIGN_CITADEL_ENGINE = SovereignCitadelEngine()
