# ==============================================================================
# KRYSTAL-STACK: GNOME DUEL WINDOW & CLOSE-UP ASPECT-RATIO COMPOSITOR
# ==============================================================================
# Implements:
#   1. GNOME shell windowing principles (clean CSD, aspect-ratio scaling).
#   2. Adaptive aspect ratios: 16:9 (Cinematic), 21:9 (Ultra-wide), 4:3, 1:1 (Split).
#   3. Close-up duel zoom displaying cold melee weapons and ranged projectiles.
# ==============================================================================

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

class AspectRatio(str, Enum):
    CINEMATIC_16_9 = "16:9"
    ULTRAWIDE_21_9 = "21:9"
    CLASSIC_4_3 = "4:3"
    SPLIT_DUEL_1_1 = "1:1"

class DuelWindowMode(str, Enum):
    CINEMATIC_ZOOM = "cinematic_zoom"
    SPLIT_OPPONENT = "split_opponent"
    PICTURE_IN_PICTURE = "picture_in_picture"

class GnomeDuelWindowCompositor:
    """
    Composes duel viewports adhering to GNOME desktop windowing conventions
    and renders close-up combat containing both melee weapons and projectiles.
    """

    ASPECT_RATIO_DIMENSIONS = {
        AspectRatio.CINEMATIC_16_9: {"width": 1280, "height": 720, "ratio": 16.0 / 9.0},
        AspectRatio.ULTRAWIDE_21_9: {"width": 1680, "height": 720, "ratio": 21.0 / 9.0},
        AspectRatio.CLASSIC_4_3: {"width": 960, "height": 720, "ratio": 4.0 / 3.0},
        AspectRatio.SPLIT_DUEL_1_1: {"width": 720, "height": 720, "ratio": 1.0}
    }

    @staticmethod
    def compose_duel_viewport(
        character_left: Dict[str, Any],
        character_right: Dict[str, Any],
        aspect_ratio: AspectRatio = AspectRatio.CINEMATIC_16_9,
        window_mode: DuelWindowMode = DuelWindowMode.CINEMATIC_ZOOM,
        incoming_projectiles: Optional[List[Dict[str, Any]]] = None,
        zoom_level: float = 1.85 # Close-up combat zoom
    ) -> Dict[str, Any]:
        """
        Creates a structured viewport frame containing characters, cold melee
        weapons (blades, flails), and incoming projectiles with GNOME layout geometry.
        """
        dims = GnomeDuelWindowCompositor.ASPECT_RATIO_DIMENSIONS.get(
            aspect_ratio, GnomeDuelWindowCompositor.ASPECT_RATIO_DIMENSIONS[AspectRatio.CINEMATIC_16_9]
        )

        w = dims["width"]
        h = dims["height"]

        # Viewport layout calculation
        if window_mode == DuelWindowMode.SPLIT_OPPONENT:
            left_pane = {"x": 0, "y": 0, "width": w // 2, "height": h}
            right_pane = {"x": w // 2, "y": 0, "width": w // 2, "height": h}
        else:
            left_pane = {"x": 0, "y": 0, "width": w, "height": h}
            right_pane = {"x": 0, "y": 0, "width": w, "height": h}

        # Cold weapon representations
        left_weapon = {
            "type": character_left.get("weapon_type", "crystal_blade"),
            "slot": "main_hand",
            "position": [round(w * 0.28), round(h * 0.55)],
            "angle_deg": 35.0,
            "visual_fx": "aetheric_blade_sheen"
        }
        right_weapon = {
            "type": character_right.get("weapon_type", "toxic_censer_flail"),
            "slot": "main_hand",
            "position": [round(w * 0.72), round(h * 0.55)],
            "angle_deg": -35.0,
            "visual_fx": "corrosive_mist_trail"
        }

        # In-flight projectiles
        projectiles_render = []
        if incoming_projectiles:
            for proj in incoming_projectiles:
                projectiles_render.append({
                    "shell_id": proj.get("shell_index", 1),
                    "pos_normalized": [0.50, 0.20], # Falling from top center
                    "flight_vector": [0.0, 1.0],
                    "apex_fx": "mortar_plunging_streak"
                })

        return {
            "window_title": f"Duel: {character_left.get('name', 'Hero A')} vs {character_right.get('name', 'Hero B')}",
            "aspect_ratio": aspect_ratio.value,
            "resolution": [w, h],
            "zoom_level": zoom_level,
            "gnome_csd_header": {
                "show_close_button": True,
                "theme": "Adwaita-Dark-Aether",
                "accent_color": "#66fcf1"
            },
            "viewports": {
                "left_combatant": {
                    "unit_id": character_left.get("unit_id"),
                    "name": character_left.get("name"),
                    "current_hp": character_left.get("current_wounds", 6),
                    "ward": character_left.get("ward", 0),
                    "pane": left_pane,
                    "weapon": left_weapon
                },
                "right_combatant": {
                    "unit_id": character_right.get("unit_id"),
                    "name": character_right.get("name"),
                    "current_hp": character_right.get("current_wounds", 6),
                    "ward": character_right.get("ward", 0),
                    "pane": right_pane,
                    "weapon": right_weapon
                }
            },
            "active_projectiles": projectiles_render
        }
