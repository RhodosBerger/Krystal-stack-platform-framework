# ==============================================================================
# KRYSTAL-STACK: GODOT .TSCN SCENE COMPILER
# ==============================================================================
# Translates tactical battlefields, hex grids, placed buildings, and combatants
# into pure Godot 4.x text scene format (.tscn).
# ==============================================================================

from typing import List, Dict, Any
from .models import MatchState, BuildingSpec

def generate_godot_tscn(match: MatchState) -> str:
    lines = [
        '[gd_scene format=3 uid="uid://krystal_posledni_kmen_economic_match"]',
        '',
        '# Krystal-Stack: Procedural Economic Match Scene',
        f'# Round: {match.round_number} | Phase: {match.phase.value} | Escalation: {match.escalation.value}',
        '',
        '[node name="EconomicMatchRoot" type="Node3D"]',
        '',
        '[node name="TacticalSun" type="DirectionalLight3D" parent="."]',
        'transform = Transform3D(0.866, -0.25, 0.433, 0, 0.866, 0.5, -0.5, -0.433, 0.75, 10, 20, 10)',
        'light_color = Color(1, 0.98, 0.9)',
        'light_energy = 1.2',
        'shadow_enabled = true',
        '',
        '[node name="TacticalCamera" type="Camera3D" parent="."]',
        'transform = Transform3D(1, 0, 0, 0, 0.707, 0.707, 0, -0.707, 0.707, 0, 15, 15)',
        'fov = 60.0',
        '',
        '[node name="HexBattleGrid" type="Node3D" parent="."]',
        ''
    ]

    # Generate Hex Grid Nodes (19 Hex Tiles)
    hex_w = 1.732
    hex_h = 1.5
    radius = 2
    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            x = round(hex_w * (q + r * 0.5), 2)
            z = round(hex_h * r, 2)
            lines.extend([
                f'[node name="HexTile_{q}_{r}" type="MeshInstance3D" parent="HexBattleGrid"]',
                f'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {x}, 0, {z})',
                '# Mesh reference: res://godot_assets/hex_tile.obj',
                ''
            ])

    # Player Buildings
    lines.append('[node name="PlayerInfrastructure" type="Node3D" parent="."]')
    lines.append('')
    for idx, b in enumerate(match.player.buildings):
        lines.extend([
            f'[node name="{b.name.replace(" ", "_")}_{idx}" type="Node3D" parent="PlayerInfrastructure"]',
            f'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {b.position[0]}, {b.position[1]}, {b.position[2]})',
            f'[node name="Mesh" type="MeshInstance3D" parent="PlayerInfrastructure/{b.name.replace(" ", "_")}_{idx}"]',
            f'# Mesh: res://godot_assets/{b.mesh_asset}',
            f'modulate = Color("{b.color}")',
            ''
        ])

    # Enemy Buildings
    lines.append('[node name="EnemyInfrastructure" type="Node3D" parent="."]')
    lines.append('')
    for idx, b in enumerate(match.enemy.buildings):
        lines.extend([
            f'[node name="{b.name.replace(" ", "_")}_{idx}" type="Node3D" parent="EnemyInfrastructure"]',
            f'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {b.position[0]}, {b.position[1]}, {b.position[2]})',
            f'[node name="Mesh" type="MeshInstance3D" parent="EnemyInfrastructure/{b.name.replace(" ", "_")}_{idx}"]',
            f'# Mesh: res://godot_assets/{b.mesh_asset}',
            f'modulate = Color("{b.color}")',
            ''
        ])

    return "\n".join(lines)
