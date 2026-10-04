"""
Godot Canvas 3D Arena Engine & PBR Tactical Subsystem
=============================================================================
Manages real-time tactical state, 19-hex board geometry matching Godot 4.x
PBR shaders, turn-based action economy (3 AP), AI counterturns, and strict
6-HP Vital Invariant enforcement across all combat participants.
=============================================================================
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 0.61803398875


@dataclass
class HexTile:
    q: int
    r: int
    x: float
    z: float
    material: str
    biome: str
    display_name: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "q": self.q,
            "r": self.r,
            "x": self.x,
            "z": self.z,
            "material": self.material,
            "biome": self.biome,
            "display_name": self.display_name
        }


# 19-Hex Arena Grid Specification matching posledni_kmen_arena.tscn
HEX_GRID_TILES: List[HexTile] = [
    HexTile(q=-2, r=0, x=-3.46, z=0.0, material="StandardMaterial3D_slime", biome="toxic", display_name="Toxic Slime"),
    HexTile(q=-2, r=1, x=-2.60, z=1.5, material="StandardMaterial3D_slime", biome="toxic", display_name="Toxic Slime"),
    HexTile(q=-2, r=2, x=-1.73, z=3.0, material="StandardMaterial3D_slime", biome="toxic", display_name="Toxic Slime"),
    HexTile(q=-1, r=-1, x=-2.60, z=-1.5, material="StandardMaterial3D_slime", biome="toxic", display_name="Toxic Slime"),
    HexTile(q=-1, r=0, x=-1.73, z=0.0, material="StandardMaterial3D_slime", biome="toxic", display_name="Toxic Slime"),
    HexTile(q=-1, r=1, x=-0.87, z=1.5, material="StandardMaterial3D_slime", biome="toxic", display_name="Toxic Slime"),
    HexTile(q=-1, r=2, x=0.00, z=3.0, material="StandardMaterial3D_slime", biome="toxic", display_name="Toxic Slime"),
    HexTile(q=0, r=-2, x=-1.73, z=-3.0, material="StandardMaterial3D_bark", biome="druid", display_name="Druid Amber"),
    HexTile(q=0, r=-1, x=-0.87, z=-1.5, material="StandardMaterial3D_bark", biome="druid", display_name="Druid Amber"),
    HexTile(q=0, r=0, x=0.00, z=0.0, material="StandardMaterial3D_basalt", biome="volcanic", display_name="Central Basalt"),
    HexTile(q=0, r=1, x=0.87, z=1.5, material="StandardMaterial3D_basalt", biome="volcanic", display_name="Central Basalt"),
    HexTile(q=0, r=2, x=1.73, z=3.0, material="StandardMaterial3D_basalt", biome="volcanic", display_name="Central Basalt"),
    HexTile(q=1, r=-2, x=0.00, z=-3.0, material="StandardMaterial3D_bark", biome="druid", display_name="Druid Amber"),
    HexTile(q=1, r=-1, x=0.87, z=-1.5, material="StandardMaterial3D_bark", biome="druid", display_name="Druid Amber"),
    HexTile(q=1, r=0, x=1.73, z=0.0, material="StandardMaterial3D_cryst", biome="crystal", display_name="Cyan Crystal"),
    HexTile(q=1, r=1, x=2.60, z=1.5, material="StandardMaterial3D_cryst", biome="crystal", display_name="Cyan Crystal"),
    HexTile(q=2, r=-2, x=1.73, z=-3.0, material="StandardMaterial3D_bark", biome="druid", display_name="Druid Amber"),
    HexTile(q=2, r=-1, x=2.60, z=-1.5, material="StandardMaterial3D_bark", biome="druid", display_name="Druid Amber"),
    HexTile(q=2, r=0, x=3.46, z=0.0, material="StandardMaterial3D_cryst", biome="crystal", display_name="Cyan Crystal")
]

HEX_GRID_19: List[Dict[str, Any]] = [h.to_dict() for h in HEX_GRID_TILES]

PBR_MATERIALS_SPEC = {
    "StandardMaterial3D_cryst": {
        "albedo": [0.4, 0.98, 0.94, 0.85],
        "metallic": 0.85,
        "roughness": 0.15,
        "emission": [0.0, 1.0, 1.0, 1.0],
        "emission_energy": 2.4,
        "name": "Krystal Cyan Resonance"
    },
    "StandardMaterial3D_slime": {
        "albedo": [0.22, 1.0, 0.08, 0.9],
        "metallic": 0.1,
        "roughness": 0.05,
        "emission": [0.22, 1.0, 0.08, 1.0],
        "emission_energy": 1.6,
        "name": "Toxic Acid Slime"
    },
    "StandardMaterial3D_bark": {
        "albedo": [0.55, 0.27, 0.07, 1.0],
        "metallic": 0.05,
        "roughness": 0.85,
        "emission": [1.0, 0.84, 0.0, 1.0],
        "emission_energy": 0.8,
        "name": "Druidic Amber Bark"
    },
    "StandardMaterial3D_basalt": {
        "albedo": [0.12, 0.14, 0.18, 1.0],
        "metallic": 0.45,
        "roughness": 0.65,
        "emission": [0.4, 0.8, 1.0, 1.0],
        "emission_energy": 1.2,
        "name": "Obsidian Basalt Monolith"
    }
}


@dataclass
class ArenaHero:
    hero_id: str
    name: str
    faction: str
    hp: int = VITAL_MAX_HP
    max_hp: int = VITAL_MAX_HP
    action_points: int = 3
    max_action_points: int = 3
    q: int = 0
    r: int = -2
    vital_max_hp_rule: int = VITAL_MAX_HP

    @property
    def ap(self) -> int:
        return self.action_points

    @ap.setter
    def ap(self, val: int):
        self.action_points = val

    def take_damage(self, amount: int) -> int:
        actual_dmg = max(0, min(self.hp, amount))
        self.hp = max(0, self.hp - actual_dmg)
        return actual_dmg

    def heal(self, amount: int) -> int:
        before = self.hp
        self.hp = min(VITAL_MAX_HP, self.hp + amount)
        return self.hp - before


class GodotCanvasArenaEngine:
    """Manages the real-time tactical state of the Godot 3D Canvas Arena."""

    def __init__(self, tscn_file_path: Optional[str] = None):
        if not tscn_file_path:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            tscn_file_path = os.path.join(base_dir, "godot_project", "scenes", "posledni_kmen_arena.tscn")
        self.tscn_file_path = tscn_file_path
        self.raw_tscn_content = ""
        self._load_tscn()

        self.player_hero = ArenaHero("player_hero", "Krystal Champion", "Crystal Vanguard", q=0, r=-2)
        self.enemy_hero = ArenaHero("enemy_hero", "Toxic Corruptor", "Acid Swarm", q=0, r=2)
        self.turn_number: int = 1
        self.combat_log: List[str] = ["[GODOT] Poslední Kmen 3D Canvas Inicializovaný."]

    @property
    def hexes(self) -> List[HexTile]:
        return HEX_GRID_TILES

    def _load_tscn(self):
        if os.path.exists(self.tscn_file_path):
            with open(self.tscn_file_path, "r", encoding="utf-8") as f:
                self.raw_tscn_content = f.read()
        else:
            self.raw_tscn_content = self._generate_fallback_tscn()

    def _generate_fallback_tscn(self) -> str:
        lines = [
            '[gd_scene format=3 uid="uid://krystal_posledni_kmen_arena"]',
            '',
            '[sub_resource type="StandardMaterial3D" id="StandardMaterial3D_cryst"]',
            'albedo_color = Color(0.4, 0.98, 0.94, 0.85)',
            'metallic = 0.85',
            'roughness = 0.15',
            'emission_enabled = true',
            'emission = Color(0.0, 1.0, 1.0, 1.0)',
            'emission_energy_multiplier = 2.4',
            '',
            '[node name="PosledniKmenArena" type="Node3D"]',
            '',
            '[node name="TacticalSun" type="DirectionalLight3D"]',
            'transform = Transform3D(0.866, -0.25, 0.433, 0.0, 0.866, 0.5, -0.5, -0.433, 0.75, 12.0, 24.0, 12.0)',
            'light_color = Color(1.0, 0.98, 0.92, 1.0)',
            'light_energy = 1.25',
            'shadow_enabled = true'
        ]
        return "\n".join(lines)

    def export_godot_tscn(self) -> str:
        if self.raw_tscn_content:
            return self.raw_tscn_content
        return self._generate_fallback_tscn()

    export_godot_scene_tscn = export_godot_tscn

    def get_arena_scene_data(self) -> Dict[str, Any]:
        """Returns the full parsed 3D scene data and PBR specification for the Canvas."""
        return {
            "success": True,
            "scene_name": "PosledniKmenArena",
            "vital_max_hp_rule": VITAL_MAX_HP,
            "hex_count": len(HEX_GRID_19),
            "hexes": HEX_GRID_19,
            "materials": PBR_MATERIALS_SPEC,
            "player_hero": {
                "id": self.player_hero.hero_id,
                "name": self.player_hero.name,
                "faction": self.player_hero.faction,
                "hp": self.player_hero.hp,
                "max_hp": self.player_hero.max_hp,
                "ap": self.player_hero.action_points,
                "q": self.player_hero.q,
                "r": self.player_hero.r
            },
            "enemy_hero": {
                "id": self.enemy_hero.hero_id,
                "name": self.enemy_hero.name,
                "faction": self.enemy_hero.faction,
                "hp": self.enemy_hero.hp,
                "max_hp": self.enemy_hero.max_hp,
                "ap": self.enemy_hero.action_points,
                "q": self.enemy_hero.q,
                "r": self.enemy_hero.r
            },
            "turn_number": self.turn_number,
            "raw_tscn_preview": self.raw_tscn_content[:500] + "... (truncated)"
        }

    def execute_tactical_action(self, action_type: str, target_q: Optional[int] = None, target_r: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes a tactical action (move, attack, cast) server-side with strict 6-HP rule.
        """
        if self.player_hero.action_points <= 0:
            return {"success": False, "error": "Nedostatok akčných bodov (AP). Ukončite ťah."}

        result_message = ""
        dmg_dealt = 0

        if action_type == "move":
            if target_q is None or target_r is None:
                return {"success": False, "error": "Pre pohyb je nutné zadať cieľový hex [q, r]."}
            target_hex = next((h for h in HEX_GRID_TILES if h.q == target_q and h.r == target_r), None)
            if not target_hex:
                return {"success": False, "error": f"Neplatné hexové súradnice: [{target_q}, {target_r}]"}

            self.player_hero.q = target_q
            self.player_hero.r = target_r
            self.player_hero.action_points -= 1
            result_message = f"Hrdina sa presunul na hex [{target_q}, {target_r}] ({target_hex.display_name})."

        elif action_type in ("strike", "mortar_shot", "crystal_beam"):
            # Default target to enemy hero coordinates if not explicitly passed
            tq = self.enemy_hero.q if target_q is None else target_q
            tr = self.enemy_hero.r if target_r is None else target_r

            dist = max(abs(self.player_hero.q - tq),
                       abs(self.player_hero.r - tr),
                       abs((-self.player_hero.q - self.player_hero.r) - (-tq - tr)))

            base_dmg = 2 if action_type == "strike" else (3 if action_type == "mortar_shot" else 2)
            dmg_dealt = self.enemy_hero.take_damage(base_dmg)
            self.player_hero.action_points -= 1
            result_message = f"Útok '{action_type}' zasiahol nepriateľa na vzdialenosť {dist} hexov za {dmg_dealt} DMG! (Zostáva {self.enemy_hero.hp}/6 HP)"

        elif action_type == "heal":
            healed = self.player_hero.heal(2)
            self.player_hero.action_points -= 1
            result_message = f"Hrdina obnovil {healed} HP. (Aktuálne {self.player_hero.hp}/6 HP)"

        else:
            return {"success": False, "error": f"Neznáma akcia '{action_type}'."}

        self.combat_log.append(f"[TURN {self.turn_number}] {result_message}")

        return {
            "success": True,
            "action_type": action_type,
            "message": result_message,
            "damage_dealt": dmg_dealt,
            "player_hp": self.player_hero.hp,
            "enemy_hp": self.enemy_hero.hp,
            "remaining_ap": self.player_hero.action_points,
            "vital_max_hp_rule": VITAL_MAX_HP
        }

    def execute_ai_counter_turn(self) -> Dict[str, Any]:
        """Simulates an AI turn: resets AP and attacks or maneuvers."""
        self.enemy_hero.action_points = self.enemy_hero.max_action_points
        self.player_hero.action_points = self.player_hero.max_action_points
        self.turn_number += 1

        # AI decisions: if player in range, attack; else maneuver
        dmg = 0
        ai_action = ""
        dist = max(abs(self.enemy_hero.q - self.player_hero.q),
                   abs(self.enemy_hero.r - self.player_hero.r),
                   abs((-self.enemy_hero.q - self.enemy_hero.r) - (-self.player_hero.q - self.player_hero.r)))

        if dist <= 2 and self.enemy_hero.hp > 0:
            dmg = self.player_hero.take_damage(2)
            ai_action = f"Nepriateľský Toxic Corruptor zaútočil kyselinovým pľuvancom za {dmg} DMG!"
        else:
            # Shift towards center
            if self.enemy_hero.r > 0:
                self.enemy_hero.r -= 1
            ai_action = f"Nepriateľ manévruje bližšie ku stredu bojiska na hex [{self.enemy_hero.q}, {self.enemy_hero.r}]."

        self.combat_log.append(f"[AI TURN] {ai_action}")

        return {
            "success": True,
            "ai_action": ai_action,
            "damage_dealt_to_player": dmg,
            "player_hp": self.player_hero.hp,
            "enemy_hp": self.enemy_hero.hp,
            "turn_number": self.turn_number,
            "vital_max_hp_rule": VITAL_MAX_HP
        }

    def reset_arena(self) -> Dict[str, Any]:
        """Resets the arena to initial 6-HP state."""
        self.player_hero.hp = VITAL_MAX_HP
        self.player_hero.action_points = 3
        self.player_hero.q = 0
        self.player_hero.r = -2

        self.enemy_hero.hp = VITAL_MAX_HP
        self.enemy_hero.action_points = 3
        self.enemy_hero.q = 0
        self.enemy_hero.r = 2

        self.turn_number = 1
        self.combat_log = ["[GODOT] Aréna reštartovaná. Obaja hrdinovia majú plných 6/6 HP."]
        return self.get_arena_scene_data()


GLOBAL_GODOT_CANVAS_ARENA = GodotCanvasArenaEngine()
