# ==============================================================================
# KRYSTAL-STACK: TOWER LOCATIONAL ALGEBRA & WARD DEFENSE ENGINE
# ==============================================================================
# Implements 3D locational algebra on hex coordinates (q, r, h), high-ground bonuses
# (zhora bonus), low-ground assault penalties (zospodu na vežu útok), and
# protective Ward energy bubbles (Warda) with shield absorption and status cleansing.
# ==============================================================================

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import math

from .models import Tribe
from .ability_framework import calculate_hex_distance


# ------------------------------------------------------------------------------
# 1. 3D HEX COORDINATE AND LOCATIONAL ALGEBRA
# ------------------------------------------------------------------------------

@dataclass(frozen=True)
class HexCoord3D:
    q: int
    r: int
    h: float = 0.0 # Elevation height tier (0 = ground, 1 = low ridge, 2 = tower, 3 = high spire)

    @property
    def s(self) -> int:
        return -self.q - self.r

    def distance_2d(self, other: "HexCoord3D") -> int:
        """Axial Riemannian 2D distance on hex grid."""
        return (abs(self.q - other.q) + abs(self.q + self.r - other.q - other.r) + abs(self.r - other.r)) // 2

    def distance_3d(self, other: "HexCoord3D", height_scale: float = 1.2) -> float:
        """
        3D Euclidean metric combining axial hex planar distance and vertical elevation:
          D_3D = sqrt( D_2D^2 + (height_scale * Delta_h)^2 )
        """
        d2d = float(self.distance_2d(other))
        dh = (self.h - other.h) * height_scale
        return math.sqrt(d2d * d2d + dh * dh)

    def firing_angle_radians(self, other: "HexCoord3D") -> float:
        """Calculates trajectory angle relative to horizontal plane in radians."""
        d2d = max(0.5, float(self.distance_2d(other)))
        dh = other.h - self.h
        return math.atan2(dh, d2d)


# ------------------------------------------------------------------------------
# 2. HIGH-GROUND & LOW-GROUND COMBAT MODIFIERS (ZHORA / ZOSPODU)
# ------------------------------------------------------------------------------

@dataclass
class ElevationCombatModifiers:
    height_delta: float # h_attacker - h_defender
    range_modifier: int
    hit_modifier: int
    ap_modifier: int
    damage_multiplier: float
    cover_save_bonus: int
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "height_delta": round(self.height_delta, 2),
            "range_modifier": self.range_modifier,
            "hit_modifier": self.hit_modifier,
            "ap_modifier": self.ap_modifier,
            "damage_multiplier": round(self.damage_multiplier, 3),
            "cover_save_bonus": self.cover_save_bonus,
            "description": self.description
        }


def calculate_elevation_advantage(attacker_pos: HexCoord3D, defender_pos: HexCoord3D) -> ElevationCombatModifiers:
    """
    Computes high-ground / low-ground algebraic bonuses:
      - Zhora bonus (Attacker on high ground, Delta_h > 0):
          + Range extension (+1 hex per elevation tier)
          + Plunging fire (+1 AP, +1 to-hit)
          + Gravitational kinetic bonus (+20% damage per tier)
      - Zospodu na vežu útok (Attacker on low ground, Delta_h < 0):
          - Range penalty (-1 hex)
          - Battlement cover (+1 or +2 to defender armor save)
          - Aim penalty (-1 to-hit)
          - Parapet damage reduction (-25% damage)
    """
    dh = attacker_pos.h - defender_pos.h

    if dh >= 0.8:
        # High Ground Bonus (Zhora)
        tier = int(math.floor(dh))
        rng_mod = max(1, tier)
        hit_mod = 1
        ap_mod = 1
        dmg_mult = 1.0 + 0.20 * dh
        save_bonus = 0
        desc = f"ZHORA BONUS (Elevated +{dh:.1f}m): +{rng_mod} Range, +1 Hit, +1 AP, +{int(dh*20)}% DMG"
    elif dh <= -0.8:
        # Low Ground Penalty (Zospodu na vežu)
        tier = int(math.floor(abs(dh)))
        rng_mod = -min(2, max(1, tier))
        hit_mod = -1
        ap_mod = 0
        dmg_mult = 0.75 # 25% damage absorbed by tower ramparts
        save_bonus = 1 if abs(dh) < 2.0 else 2
        desc = f"ZOSPODU NA VEŽU (Below {dh:.1f}m): {rng_mod} Range, -1 Hit, Tower Cover +{save_bonus} Save, -25% DMG"
    else:
        # Level terrain
        rng_mod = 0
        hit_mod = 0
        ap_mod = 0
        dmg_mult = 1.0
        save_bonus = 0
        desc = "LEVEL TERRAIN: No elevation modifiers."

    return ElevationCombatModifiers(
        height_delta=dh,
        range_modifier=rng_mod,
        hit_modifier=hit_mod,
        ap_modifier=ap_mod,
        damage_multiplier=dmg_mult,
        cover_save_bonus=save_bonus,
        description=desc
    )


# ------------------------------------------------------------------------------
# 3. TOWER WARD AURA & DEFENSIVE BUBBLE (WARDA)
# ------------------------------------------------------------------------------

@dataclass
class TowerWard:
    tower_id: str
    name: str
    owner_tribe: Tribe
    position: HexCoord3D
    ward_radius: int = 1 # Hex radius of protective sphere
    ward_max_pool: int = 5
    ward_current_pool: int = 5
    regen_per_turn: int = 1
    active: bool = True

    def is_in_ward_range(self, target_pos: HexCoord3D) -> bool:
        """Checks if coordinate falls inside the spherical protective Ward envelope."""
        d2d = self.position.distance_2d(target_pos)
        return d2d <= self.ward_radius

    def absorb_damage(self, incoming_damage: int) -> Tuple[int, int]:
        """
        Absorbs incoming damage through the ward pool first.
        Returns (absorbed_damage, leftover_damage).
        """
        if not self.active or self.ward_current_pool <= 0:
            return (0, incoming_damage)

        absorbed = min(incoming_damage, self.ward_current_pool)
        self.ward_current_pool -= absorbed
        leftover = incoming_damage - absorbed
        return (absorbed, leftover)

    def regenerate_turn(self) -> int:
        """Regenerates ward pool at the start of a turn."""
        if not self.active:
            return 0
        before = self.ward_current_pool
        self.ward_current_pool = min(self.ward_max_pool, self.ward_current_pool + self.regen_per_turn)
        return self.ward_current_pool - before

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tower_id": self.tower_id,
            "name": self.name,
            "owner_tribe": self.owner_tribe.value,
            "position": {"q": self.position.q, "r": self.position.r, "h": self.position.h},
            "ward_radius": self.ward_radius,
            "ward_max_pool": self.ward_max_pool,
            "ward_current_pool": self.ward_current_pool,
            "regen_per_turn": self.regen_per_turn,
            "active": self.active
        }


# ------------------------------------------------------------------------------
# 4. TOWER COMBAT RESOLVER WITH ALGEBRAIC MODIFIERS
# ------------------------------------------------------------------------------

class TowerLocationalCombatResolver:
    """Orchestrates tower attacks, elevated shots, and ward shielding."""

    @staticmethod
    def resolve_elevated_attack(
        attacker_pos: HexCoord3D,
        defender_pos: HexCoord3D,
        base_range: int,
        base_damage: int,
        base_skill: int = 3,
        base_ap: int = 1,
        active_ward: Optional[TowerWard] = None
    ) -> Dict[str, Any]:
        """Calculates a complete projectile volley taking elevation algebra and ward shields into account."""
        elev = calculate_elevation_advantage(attacker_pos, defender_pos)
        d2d = attacker_pos.distance_2d(defender_pos)
        d3d = attacker_pos.distance_3d(defender_pos)
        angle_rad = attacker_pos.firing_angle_radians(defender_pos)
        angle_deg = math.degrees(angle_rad)

        effective_max_range = max(1, base_range + elev.range_modifier)
        in_range = d2d <= effective_max_range

        # Calculate final modified damage
        raw_damage = int(round(base_damage * elev.damage_multiplier))
        effective_ap = max(0, base_ap + elev.ap_modifier)
        effective_skill = max(2, min(6, base_skill - elev.hit_modifier)) # lower skill = easier to hit

        absorbed_by_ward = 0
        final_damage_to_target = raw_damage

        # Check if defender is sheltered by an active Ward
        if active_ward and active_ward.is_in_ward_range(defender_pos):
            absorbed_by_ward, final_damage_to_target = active_ward.absorb_damage(raw_damage)

        return {
            "in_range": in_range,
            "distance_2d": d2d,
            "distance_3d": round(d3d, 2),
            "effective_max_range": effective_max_range,
            "firing_angle_deg": round(angle_deg, 1),
            "elevation_modifiers": elev.to_dict(),
            "raw_damage": raw_damage,
            "effective_ap": effective_ap,
            "effective_skill": effective_skill,
            "absorbed_by_ward": absorbed_by_ward,
            "final_damage_to_target": final_damage_to_target,
            "ward_current_pool": active_ward.ward_current_pool if active_ward else 0
        }
