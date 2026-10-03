# ==============================================================================
# KRYSTAL-STACK: WORDPRESS MCP BRIDGE & DUEL OPPONENT STREAMING ENGINE
# ==============================================================================
# Implements:
#   1. MCP (Model Context Protocol) Bridge exposing tactical combat & visual tools.
#   2. Frame Streaming to Opponent Duel Windows (GNOME aspect ratio compliance).
#   3. Uncovered target detection & Mortar cadence pipeline.
#   4. Ubisoft Bullet Time Choreography dispatch.
#   5. Integration with WordPress REST API and Oxygen Builder viewports.
# ==============================================================================

import time
import json
from typing import Dict, List, Any, Optional, Tuple

from krystal_web_hub.economic_engine.dopamine_matrix_combat import (
    DopamineCombatState,
    DopamineCadenceEngine
)
from krystal_web_hub.economic_engine.mortar_dispersion_engine import (
    UncoveredTargetCalculator,
    MortarDispersionEngine
)
from krystal_web_hub.economic_engine.gnome_duel_window import (
    GnomeDuelWindowCompositor,
    AspectRatio,
    DuelWindowMode
)
from krystal_web_hub.economic_engine.ubisoft_bullet_time_compositor import (
    UbisoftBulletTimeCompositor,
    ParallelActionTrack,
    ActionType
)
from krystal_web_hub.economic_engine.community_builds_and_rewards import (
    CommunityBuildRegistry,
    EventRewardDerivationEngine,
    EventThreatLevel
)


class WordPressMcpBridge:
    """
    MCP Server Bridge enabling AI agents, WordPress instances, and game clients
    to execute combat calculations and stream synchronized duel frames into opponent windows.
    """

    def __init__(self):
        # Ring buffers for streamed frames: duel_id -> list of frame payloads
        self.duel_frame_streams: Dict[str, List[Dict[str, Any]]] = {}
        # Max frames retained per stream
        self.max_stream_history = 60

    # --------------------------------------------------------------------------
    # MCP TOOL: CALCULATE UNCOVERED TARGETS & MORTAR SALVO CADENCE
    # --------------------------------------------------------------------------
    def calculate_uncovered_and_mortar_cadence(
        self,
        combatants: List[Dict[str, Any]],
        cover_map: Dict[str, float],
        elevation_map: Dict[str, float],
        attacker_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        salvo_count: int = 3,
        elevation_angle_deg: float = 65.0,
        cadence_interval_sec: float = 0.40
    ) -> Dict[str, Any]:
        """
        Evaluates cover status of all combatants and computes mortar dispersion footprint
        and projectile firing cadence.
        """
        attacker_hex = [int(attacker_pos[0]), int(attacker_pos[2])]
        target_eval = UncoveredTargetCalculator.identify_uncovered_targets(
            combatants=combatants,
            cover_map=cover_map,
            elevation_map=elevation_map,
            attacker_hex=attacker_hex,
            is_plunging_fire=True
        )

        salvo_data = MortarDispersionEngine.calculate_mortar_salvo(
            attacker_pos=attacker_pos,
            target_pos=target_pos,
            salvo_count=salvo_count,
            elevation_angle_deg=elevation_angle_deg,
            muzzle_velocity=45.0,
            cadence_interval_sec=cadence_interval_sec
        )

        uncovered_units = [u["unit_id"] for u in target_eval if u["status"] == "UNCOVERED"]

        return {
            "success": True,
            "target_exposure_evaluation": target_eval,
            "uncovered_target_ids": uncovered_units,
            "mortar_salvo": salvo_data,
            "tactical_recommendation": (
                f"Zistené {len(uncovered_units)} nekrytých cieľov! Plunging mortar má 100% efektivitu pri uhle {elevation_angle_deg}°."
                if uncovered_units else "Všetky ciele sú v plnom krytí. Použite korozívne zbrane na rozbitie krytia."
            )
        }

    # --------------------------------------------------------------------------
    # MCP TOOL: COMPOSE GNOME DUEL WINDOW FRAME
    # --------------------------------------------------------------------------
    def compose_gnome_duel_frame(
        self,
        character_left: Dict[str, Any],
        character_right: Dict[str, Any],
        aspect_ratio: str = "16:9",
        window_mode: str = "cinematic_zoom",
        incoming_projectiles: Optional[List[Dict[str, Any]]] = None,
        zoom_level: float = 1.85
    ) -> Dict[str, Any]:
        """
        Renders a GNOME-compliant duel frame adhering to the requested aspect ratio
        with close-up weapon and projectile visual layering.
        """
        try:
            ar_enum = AspectRatio(aspect_ratio)
        except ValueError:
            ar_enum = AspectRatio.CINEMATIC_16_9

        try:
            mode_enum = DuelWindowMode(window_mode)
        except ValueError:
            mode_enum = DuelWindowMode.CINEMATIC_ZOOM

        frame_data = GnomeDuelWindowCompositor.compose_duel_viewport(
            character_left=character_left,
            character_right=character_right,
            aspect_ratio=ar_enum,
            window_mode=mode_enum,
            incoming_projectiles=incoming_projectiles,
            zoom_level=zoom_level
        )

        # Generate ASCII Canvas representation for immediate terminal/streaming preview
        w_chars = 48
        ascii_lines = [
            f"┌{'─' * (w_chars - 2)}┐",
            f"│ GNOME CSD: [{frame_data['aspect_ratio']}] {frame_data['window_title'][:28]:<28} [✕] │",
            f"├{'─' * (w_chars - 2)}┤",
            f"│ LEFT: {character_left.get('name', 'P1')[:12]:<12} HP:{character_left.get('current_wounds', 6)}/6  WRD:{character_left.get('ward', 0):<2} │",
            f"│ WEAPON: {character_left.get('weapon_type', 'blade')[:14]:<14}            ⚔️       │",
            f"│                   [BULLET TIME ZONE]          │",
            f"│ RIGHT: {character_right.get('name', 'P2')[:12]:<12} HP:{character_right.get('current_wounds', 6)}/6  WRD:{character_right.get('ward', 0):<2} │",
            f"│ WEAPON: {character_right.get('weapon_type', 'flail')[:14]:<14}            🛡️       │",
            f"│ IN-FLIGHT PROJECTILES: {len(incoming_projectiles or []):<2} SHELLS IN AIR      │",
            f"└{'─' * (w_chars - 2)}┘"
        ]

        frame_data["ascii_canvas_preview"] = "\n".join(ascii_lines)
        return frame_data

    # --------------------------------------------------------------------------
    # MCP TOOL: EXECUTE UBISOFT BULLET TIME CHOREOGRAPHY
    # --------------------------------------------------------------------------
    def execute_ubisoft_bullet_time(
        self,
        melee_actor_id: str,
        melee_start: Tuple[float, float, float],
        melee_end: Tuple[float, float, float],
        projectile_start: Tuple[float, float, float],
        projectile_end: Tuple[float, float, float],
        aspect_ratio: str = "16:9"
    ) -> Dict[str, Any]:
        """
        Constructs parallel action tracks for cold weapon attack and incoming mortar projectile,
        checks space-time intersection, and compiles bullet time camera orbit keyframes.
        """
        track_melee = ParallelActionTrack(
            action_id="melee_strike_action_01",
            actor_id=melee_actor_id,
            action_type=ActionType.COLD_MELEE_STRIKE,
            start_pos=melee_start,
            end_pos=melee_end,
            start_time=0.0,
            duration=1.20,
            bounding_radius=0.75
        )

        track_mortar = ParallelActionTrack(
            action_id="mortar_shell_action_01",
            actor_id="mortar_battery",
            action_type=ActionType.MORTAR_PROJECTILE,
            start_pos=projectile_start,
            end_pos=projectile_end,
            start_time=0.20,
            duration=1.10,
            bounding_radius=0.85
        )

        return UbisoftBulletTimeCompositor.compose_bullet_time_sequence(
            track_melee=track_melee,
            track_projectile=track_mortar,
            aspect_ratio=aspect_ratio
        )

    # --------------------------------------------------------------------------
    # MCP TOOL: STREAM FRAME TO OPPONENT WINDOW
    # --------------------------------------------------------------------------
    def stream_frame_to_opponent(
        self,
        duel_id: str,
        sender_id: str,
        recipient_id: str,
        frame_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Pushes a generated frame to the opponent's active window stream buffer.
        """
        now = time.time()
        if duel_id not in self.duel_frame_streams:
            self.duel_frame_streams[duel_id] = []

        packet = {
            "stream_id": f"{duel_id}_{int(now * 1000)}",
            "duel_id": duel_id,
            "sender_id": sender_id,
            "recipient_id": recipient_id,
            "timestamp": now,
            "frame": frame_payload
        }

        self.duel_frame_streams[duel_id].append(packet)
        if len(self.duel_frame_streams[duel_id]) > self.max_stream_history:
            self.duel_frame_streams[duel_id].pop(0)

        return {
            "success": True,
            "duel_id": duel_id,
            "packet_id": packet["stream_id"],
            "recipient_id": recipient_id,
            "buffer_depth": len(self.duel_frame_streams[duel_id]),
            "status": "STREAMED_TO_OPPONENT_VIEWPORT"
        }

    def poll_opponent_frames(
        self,
        duel_id: str,
        recipient_id: str,
        since_timestamp: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieves newly arrived frames for the recipient's duel window.
        """
        stream = self.duel_frame_streams.get(duel_id, [])
        return [
            pkt for pkt in stream
            if pkt["recipient_id"] == recipient_id and pkt["timestamp"] > since_timestamp
        ]

    # --------------------------------------------------------------------------
    # MCP TOOL: EVALUATE EVENT REWARDS & COMMUNITY BUILDS
    # --------------------------------------------------------------------------
    def evaluate_threat_rewards(
        self,
        threat_level: int,
        remaining_hp: int,
        cadence_score: float,
        bullet_time_count: int = 1,
        uncovered_kills: int = 2,
        player_id: str = "player_hero_1"
    ) -> Dict[str, Any]:
        try:
            t_enum = EventThreatLevel(threat_level)
        except ValueError:
            t_enum = EventThreatLevel.THREAT_1_PERIMETER_SKIRMISH

        return EventRewardDerivationEngine.derive_event_rewards(
            threat_level=t_enum,
            remaining_hp=remaining_hp,
            dopamine_cadence_score=cadence_score,
            bullet_time_count=bullet_time_count,
            uncovered_targets_eliminated=uncovered_kills,
            player_account_id=player_id
        )

    def get_community_builds(self) -> List[Dict[str, Any]]:
        return CommunityBuildRegistry.list_builds()

    # --------------------------------------------------------------------------
    # WORDPRESS INTEGRATION PAYLOAD
    # --------------------------------------------------------------------------
    def generate_wordpress_rest_payload(
        self,
        duel_id: str,
        frame_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Encapsulates duel frames for consumption by WordPress REST API endpoints
        and Oxygen Builder web components.
        """
        return {
            "wp_channel": f"krystal_duel_{duel_id}",
            "oxygen_component_id": "oxy-krystal-gnome-duel-window",
            "css_theme_vars": {
                "--gnome-aspect-ratio": frame_data.get("aspect_ratio", "16:9"),
                "--gnome-accent": "#66fcf1",
                "--bullet-time-active": "1" if frame_data.get("bullet_time_triggered") else "0"
            },
            "frame_content": frame_data
        }
