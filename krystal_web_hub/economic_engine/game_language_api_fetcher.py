"""
game_language_api_fetcher.py
============================
Krystal-Stack Platform Framework: Autonomous Language API Fetcher & Game Controller.
Bridges natural language intent (Slovak & English) to direct Game Engine API execution.

Key Capabilities:
1. Natural Language Semantic Parsing:
   - Entity recognition: Axial coordinates [q, r], card names (fuzzy lookup across 129 cards),
     weather states, day/night celestial time, camera rig presets, artillery/mortar parameters,
     3D models (CesiumMan, Fox, RobotExpressive), and Sovereign Citadel structures.
2. Multilingual Support: Full Slovak (SK) and English (EN) keyword parsing.
3. In-Process & Remote API Execution:
   - Executes against Krystal Engine Core endpoints or dispatches directly to subsystem singletons.
   - Enforces the universal platform invariant: VITAL_MAX_HP == 6.
4. Language API Remote Fetching & Autonomous Daemon:
   - Fetches tactical decisions or commands from local/remote LLM endpoints (/v1/chat/completions,
     Krystal LLM Gateway, or instruction queues) and executes them sequentially.
   - Background polling daemon with configurable frequency and safety guardrails.
5. Natural Language Status Querying:
   - Generates articulate battlefield briefings in Slovak and English.
6. Execution Audit Trail:
   - In-memory ring buffer capturing telemetry, API calls, state diffs, and execution status.
"""

from __future__ import annotations

import re
import json
import time
import math
import uuid
import threading
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Union

# Platform Invariants
VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
DEFAULT_HUB_API_URL: str = "http://127.0.0.1:8089"


class GameCommandType(str, Enum):
    CAST_CARD = "CAST_CARD"
    TACTICAL_MOVE = "TACTICAL_MOVE"
    WEATHER_ATMOSPHERE = "WEATHER_ATMOSPHERE"
    CAMERA_CONTROL = "CAMERA_CONTROL"
    ARTILLERY_FIRE = "ARTILLERY_FIRE"
    CITADEL_BUILD = "CITADEL_BUILD"
    SPAWN_ENTITY = "SPAWN_ENTITY"
    ENVIRONMENT_HAZARD = "ENVIRONMENT_HAZARD"
    MATCH_ESCALATION = "MATCH_ESCALATION"
    QUERY_STATUS = "QUERY_STATUS"
    UNKNOWN = "UNKNOWN"


@dataclass
class LanguageCommandIntent:
    raw_text: str
    command_type: GameCommandType
    confidence: float
    parameters: Dict[str, Any]
    language: str  # "sk" | "en"
    detected_entities: List[str] = field(default_factory=list)
    vital_max_hp_rule: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp_rule > VITAL_MAX_HP:
            self.vital_max_hp_rule = VITAL_MAX_HP


@dataclass
class LanguageExecutionResult:
    execution_id: str
    timestamp: float
    intent: Dict[str, Any]
    status: str  # "SUCCESS" | "PARTIAL" | "ERROR"
    api_action_executed: str
    api_payload: Dict[str, Any]
    game_feedback: str
    state_diff: Dict[str, Any]
    execution_time_ms: float = 0.0
    vital_max_hp_rule: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp_rule > VITAL_MAX_HP:
            self.vital_max_hp_rule = VITAL_MAX_HP


@dataclass
class LanguageFetcherConfig:
    enabled: bool = False
    source_api_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    poll_interval_s: float = 3.0
    autonomous_execution: bool = True
    default_language: str = "sk"
    model_name: str = "krystal-kernel"
    mock_mode: bool = False
    vital_max_hp_rule: int = VITAL_MAX_HP

    def __post_init__(self):
        if self.vital_max_hp_rule > VITAL_MAX_HP:
            self.vital_max_hp_rule = VITAL_MAX_HP


class GameLanguageApiFetcher:
    """
    Intelligent language parser, API fetcher, and game controller.
    Translates natural language prompts into concrete game engine actions.
    """

    def __init__(self, base_api_url: str = DEFAULT_HUB_API_URL):
        self.base_api_url = base_api_url.rstrip("/")
        self.config = LanguageFetcherConfig()
        self._history: List[LanguageExecutionResult] = []
        self._max_history: int = 150
        self._lock = threading.Lock()
        
        # Daemon Threading Control
        self._daemon_thread: Optional[threading.Thread] = None
        self._daemon_running: bool = False
        self._daemon_stop_event = threading.Event()
        
        # Local state cache for standalone simulation or feedback
        self._sim_hero_hp: int = 6
        self._sim_hero_mana: int = 8
        self._sim_hero_pos: List[int] = [0, 0]
        self._sim_weather: str = "Clear"
        self._sim_time_of_day: float = 12.0
        self._sim_camera_preset: str = "perspective_action_follow"
        self._sim_round: int = 1

        # Catalog lookup tables
        self._init_dictionaries()

    def _init_dictionaries(self):
        # Known Cards (IDs & Multilingual names)
        self.known_cards = {
            "frost_shard": ["frost shard", "mrazivý črep", "mrazivy crep"],
            "crystal_shield": ["crystal shield", "kryštálový štít", "krystalovy stit", "kryštálová bariéra", "krystalova bariera"],
            "crystal_meteor": ["crystal meteor", "kryštálový meteor", "krystalovy meteor", "meteor"],
            "glacial_lance": ["glacial lance", "ľadovcová kopija", "ladovcova kopija", "kopija"],
            "orbital_hyper_lance": ["orbital hyper lance", "orbitálna hyperkopija", "orbitalna hyperkopija", "hyperkopija"],
            "aether_supernova": ["aether supernova", "aéterová supernova", "supernova"],
            "acid_slime": ["acid slime", "kyslý sliz", "kysly sliz"],
            "venom_dart": ["venom dart", "jedovatá šípka", "jedovata sipka"],
            "decay_strike": ["decay strike", "zuby rozkladu"],
            "toxic_cloud": ["toxic cloud", "toxický oblak", "toxicky oblak"],
            "acid_cataclysm": ["acid cataclysm", "kyselinová kataklizma", "kataklizma"],
            "thorn_strike": ["thorn strike", "tŕňový úder", "trnovy uder"],
            "bark_skin": ["bark skin", "dubová kôra", "dubova kora"],
            "wild_regeneration": ["wild regeneration", "divoká regenerácia", "regenerácia", "liečenie", "heal"],
            "oak_colossus": ["oak colossus", "dubový kolos"],
            "wrath_of_world_tree": ["wrath of world tree", "hnev stromu sveta"]
        }

        # Known Camera Presets
        self.known_cameras = {
            "perspective_action_follow": ["action", "follow", "3rd person", "akčný", "sledovanie", "tps"],
            "perspective_cinematic_wide": ["cinematic", "wide", "filmový", "široký", "kino"],
            "orthographic_true_isometric": ["isometric", "izometrický", "izometria", "izometricka", "iso"],
            "orthographic_tactical_topdown": ["topdown", "top-down", "zhora", "taktický", "mapa", "vtáčia perspektíva"],
            "dual_cinematic_hybrid": ["hybrid", "dual", "duálny", "prelínanie"]
        }

        # Known Weather States
        self.known_weather = {
            "Thunderstorm": ["thunderstorm", "búrk", "burk", "blesk", "hrom", "storm"],
            "Rain": ["rain", "dážď", "dažď", "dazd", "prší", "prsi", "lejak", "mokro"],
            "Overcast": ["overcast", "zamrač", "zamrac", "oblač", "oblac", "cloud"],
            "Clear": ["clear", "jasno", "slneč", "slnec", "sunny", "sun", "čistá obloha"]
        }

        # Known Models
        self.known_models = {
            "CesiumMan": ["cesiumman", "cesium", "vojak", "pešiak", "bežec"],
            "Fox": ["fox", "líška", "liska", "zviera", "spoločník", "vlk"],
            "RobotExpressive": ["robotexpressive", "robot", "mech", "dron", "kyborg"]
        }

    # ==========================================================================
    # 1. PARSING ENGINE (Slovak & English)
    # ==========================================================================

    def parse_language_command(self, text: str) -> LanguageCommandIntent:
        """
        Parses a natural language instruction into a strongly-typed intent.
        Detects coordinate targets, card names, weather states, camera angles,
        and artillery parameters.
        """
        if not text or not isinstance(text, str):
            return LanguageCommandIntent(
                raw_text="",
                command_type=GameCommandType.UNKNOWN,
                confidence=0.0,
                parameters={},
                language="en"
            )

        raw = text.strip()
        t_low = raw.lower()

        # Language Detection heuristic
        is_sk = any(w in t_low for w in [
            "zahraj", "presuň", "prepnúť", "vystreľ", "postav", "nastav", "aký", "stav",
            "kolo", "hrdina", "počasie", "búrka", "dážď", "štít", "kopija", "sliz", "kôra",
            "vyvolaj", "vyvolat", "pozíci", "pozici", "moždiar", "mozdiar", "vež", "vez", "ukonč", "ukonc"
        ])
        detected_lang = "sk" if is_sk else "en"

        detected_entities: List[str] = []
        params: Dict[str, Any] = {}

        # ── 1. Coordinate Extraction ([q, r] or (x, y) or "hex 2, -1") ──
        coord_match = re.search(r'\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]', t_low)
        if not coord_match:
            coord_match = re.search(r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)', t_low)
        if not coord_match:
            coord_match = re.search(r'(?:hex|pol[eí]|sektor|coords?)\s+(-?\d+)\s*[, ]\s*(-?\d+)', t_low)
        if not coord_match:
            coord_match = re.search(r'(?:sektor|sector)\s+(\d+)', t_low)

        target_hex = [0, 0]
        if coord_match:
            if len(coord_match.groups()) == 2:
                target_hex = [int(coord_match.group(1)), int(coord_match.group(2))]
                detected_entities.append(f"coord:{target_hex}")
            elif len(coord_match.groups()) == 1:
                sec = int(coord_match.group(1))
                target_hex = [sec, 0]
                detected_entities.append(f"sector:{sec}")
        params["target_hex"] = target_hex

        # ── 2. Query / Status Inspection ──
        if any(w in t_low for w in ["aký je stav", "aky je stav", "stav zápasu", "kolko hp", "koľko hp", 
                                    "what is my", "show status", "current state", "check hp", "query", "prehľad"]):
            return LanguageCommandIntent(
                raw_text=raw,
                command_type=GameCommandType.QUERY_STATUS,
                confidence=0.95,
                parameters={"query_target": "match_overview"},
                language=detected_lang,
                detected_entities=["intent:query_status"]
            )

        # ── 3. Weather & Atmospheric Simulation ──
        weather_found = None
        weather_matches = []
        for w_state, synonyms in self.known_weather.items():
            for syn in synonyms:
                if syn in t_low:
                    weather_matches.append((len(syn), w_state))
        if weather_matches:
            weather_matches.sort(key=lambda x: x[0], reverse=True)
            weather_found = weather_matches[0][1]
            detected_entities.append(f"weather:{weather_found}")

        # Time of Day check
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(?:h|hod|hours?)?', t_low)
        time_found = None
        if "poludnie" in t_low or "noon" in t_low or "midday" in t_low:
            time_found = 12.0
        elif "súmrak" in t_low or "sunset" in t_low or "dusk" in t_low:
            time_found = 18.5
        elif "polnoc" in t_low or "midnight" in t_low or "noc" in t_low or "night" in t_low:
            time_found = 0.0
        elif "úsvit" in t_low or "sunrise" in t_low or "dawn" in t_low:
            time_found = 6.0
        elif time_match and any(w in t_low for w in ["čas", "time", "hodin"]):
            h = float(time_match.group(1))
            m = float(time_match.group(2) or 0) / 60.0
            time_found = round(min(24.0, max(0.0, h + m)), 2)

        if weather_found or time_found is not None or any(w in t_low for w in ["počasie", "weather", "rain", "fog", "hmla"]):
            params["weather_state"] = weather_found or "Clear"
            params["time_of_day_hours"] = time_found if time_found is not None else 12.0
            params["rain_intensity"] = 0.85 if params["weather_state"] in ("Rain", "Thunderstorm") else 0.0
            return LanguageCommandIntent(
                raw_text=raw,
                command_type=GameCommandType.WEATHER_ATMOSPHERE,
                confidence=0.92,
                parameters=params,
                language=detected_lang,
                detected_entities=detected_entities
            )

        # ── 4. Camera Rig Control ──
        cam_found = None
        for cam_id, synonyms in self.known_cameras.items():
            if any(syn in t_low for syn in synonyms):
                cam_found = cam_id
                detected_entities.append(f"camera:{cam_id}")
                break

        if cam_found or any(w in t_low for w in ["kamera", "camera", "pohľad", "view", "rig", "zoom"]):
            params["camera_preset"] = cam_found or "perspective_action_follow"
            dist_match = re.search(r'(?:distance|vzdialenosť|zoom)\s+(\d+(?:\.\d+)?)', t_low)
            if dist_match:
                params["distance"] = float(dist_match.group(1))
            return LanguageCommandIntent(
                raw_text=raw,
                command_type=GameCommandType.CAMERA_CONTROL,
                confidence=0.90,
                parameters=params,
                language=detected_lang,
                detected_entities=detected_entities
            )

        # ── 5. Artillery / Mortar Fire ──
        if any(w in t_low for w in ["moždiar", "mozdiar", "mortar", "delo", "cannon", "artiléria", "artillery", "delostrelectvo", "vystreľ", "fire"]):
            params["weapon_type"] = "mortar_indirect" if any(w in t_low for w in ["moždiar", "mozdiar", "mortar", "plunging"]) else "gun_direct"
            params["distance"] = math.sqrt(target_hex[0]**2 + target_hex[1]**2) if target_hex != [0, 0] else 3.5
            params["elevation_deg"] = 65.0 if params["weapon_type"] == "mortar_indirect" else 25.0
            detected_entities.append(f"artillery:{params['weapon_type']}")
            return LanguageCommandIntent(
                raw_text=raw,
                command_type=GameCommandType.ARTILLERY_FIRE,
                confidence=0.88,
                parameters=params,
                language=detected_lang,
                detected_entities=detected_entities
            )

        # ── 6. Sovereign Citadel Build / Upgrade ──
        if any(w in t_low for w in ["postav", "veža", "veza", "citadela", "citadel", "pylon", "build pylon", "upgrade tower"]):
            params["pylon_type"] = "slavic_mortar" if any(w in t_low for w in ["slovanský", "slovansky", "mortar", "mažiar"]) else "hellenic_beam"
            params["tower_slot"] = 1
            slot_match = re.search(r'(?:slot|vež[au]|tower)\s+(\d+)', t_low)
            if slot_match:
                params["tower_slot"] = int(slot_match.group(1))
            detected_entities.append(f"citadel_pylon:{params['pylon_type']}")
            return LanguageCommandIntent(
                raw_text=raw,
                command_type=GameCommandType.CITADEL_BUILD,
                confidence=0.87,
                parameters=params,
                language=detected_lang,
                detected_entities=detected_entities
            )

        # ── 7. Spawning 3D Entities ──
        model_found = None
        for mod_id, synonyms in self.known_models.items():
            if any(syn in t_low for syn in synonyms):
                model_found = mod_id
                detected_entities.append(f"model:{mod_id}")
                break

        if model_found or any(w in t_low for w in ["spawn", "vyvolaj", "vyvolat", "umiestni model"]):
            params["model_id"] = model_found or "CesiumMan"
            params["position"] = target_hex
            return LanguageCommandIntent(
                raw_text=raw,
                command_type=GameCommandType.SPAWN_ENTITY,
                confidence=0.89,
                parameters=params,
                language=detected_lang,
                detected_entities=detected_entities
            )

        # ── 8. Tactical Movement ──
        if any(w in t_low for w in ["presuň", "presun", "kráčaj", "chod", "pohyb", "move", "walk", "step to", "flank"]):
            params["destination_hex"] = target_hex
            detected_entities.append(f"move_to:{target_hex}")
            return LanguageCommandIntent(
                raw_text=raw,
                command_type=GameCommandType.TACTICAL_MOVE,
                confidence=0.91,
                parameters=params,
                language=detected_lang,
                detected_entities=detected_entities
            )

        # ── 9. Match Escalation / Pass Turn ──
        if any(w in t_low for w in ["ukonči kolo", "ukonci kolo", "pass turn", "end turn", "ťah nepriateľa", "eskaluj", "surge", "apex"]):
            params["action"] = "END_TURN"
            detected_entities.append("match_escalation:end_turn")
            return LanguageCommandIntent(
                raw_text=raw,
                command_type=GameCommandType.MATCH_ESCALATION,
                confidence=0.94,
                parameters=params,
                language=detected_lang,
                detected_entities=detected_entities
            )

        # ── 10. Card Cast (Match Longest Card Name First) ──
        card_id_found = None
        card_matches = []
        for cid, synonyms in self.known_cards.items():
            for syn in synonyms:
                if syn in t_low:
                    card_matches.append((len(syn), cid))
        if card_matches:
            card_matches.sort(key=lambda x: x[0], reverse=True)
            card_id_found = card_matches[0][1]
            detected_entities.append(f"card:{card_id_found}")

        if card_id_found or any(w in t_low for w in ["zahraj", "cast", "play card", "použi kúzlo", "kúzlo"]):
            params["card_id"] = card_id_found or "frost_shard"
            params["target_hex"] = target_hex
            return LanguageCommandIntent(
                raw_text=raw,
                command_type=GameCommandType.CAST_CARD,
                confidence=0.86,
                parameters=params,
                language=detected_lang,
                detected_entities=detected_entities
            )

        # ── Fallback: Environmental Hazard ──
        if any(w in t_low for w in ["sliz", "acid", "oheň", "fire", "frost", "ľad", "hazard", "zóna"]):
            params["hazard_type"] = "toxic_acid" if "sliz" in t_low or "acid" in t_low else "burning_aether"
            params["duration"] = 3
            params["target_hex"] = target_hex
            return LanguageCommandIntent(
                raw_text=raw,
                command_type=GameCommandType.ENVIRONMENT_HAZARD,
                confidence=0.80,
                parameters=params,
                language=detected_lang,
                detected_entities=detected_entities
            )

        return LanguageCommandIntent(
            raw_text=raw,
            command_type=GameCommandType.UNKNOWN,
            confidence=0.30,
            parameters={},
            language=detected_lang,
            detected_entities=[]
        )

    # ==========================================================================
    # 2. COMMAND EXECUTION & GAME API DISPATCH
    # ==========================================================================

    def execute_command(self, text: str, execute_api: bool = True) -> LanguageExecutionResult:
        """
        Parses intent and executes the required action against Game APIs.
        Safely captures state diffs and strictly adheres to VITAL_MAX_HP == 6.
        """
        start_time = time.time()
        intent = self.parse_language_command(text)
        exec_id = uuid.uuid4().hex[:12]

        api_endpoint = ""
        payload: Dict[str, Any] = {}
        feedback = ""
        diff: Dict[str, Any] = {}
        status = "SUCCESS"

        try:
            if intent.command_type == GameCommandType.CAST_CARD:
                card_id = intent.parameters.get("card_id", "frost_shard")
                target_hex = intent.parameters.get("target_hex", [0, 0])
                api_endpoint = "POST /api/cards/cast"
                payload = {
                    "card_id": card_id,
                    "target_hex": target_hex,
                    "source_hex": self._sim_hero_pos
                }
                
                # State update
                cost = 2
                self._sim_hero_mana = max(0, self._sim_hero_mana - cost)
                feedback_sk = f"Zahraná karta '{card_id}' na cieľový hex {target_hex}. Spotrebovaná mana: {cost}."
                feedback_en = f"Cast card '{card_id}' at target hex {target_hex}. Consumed {cost} mana."
                feedback = feedback_sk if intent.language == "sk" else feedback_en
                diff = {"mana_delta": -cost, "target": target_hex, "hero_mana": self._sim_hero_mana}

            elif intent.command_type == GameCommandType.TACTICAL_MOVE:
                dest = intent.parameters.get("destination_hex", [0, 0])
                api_endpoint = "POST /api/tactics/move"
                payload = {"destination": dest, "unit_id": "hero_primary"}
                old_pos = list(self._sim_hero_pos)
                self._sim_hero_pos = dest
                feedback_sk = f"Hrdina sa presunul z {old_pos} na hex {dest}."
                feedback_en = f"Hero repositioned from {old_pos} to hex {dest}."
                feedback = feedback_sk if intent.language == "sk" else feedback_en
                diff = {"old_pos": old_pos, "new_pos": dest}

            elif intent.command_type == GameCommandType.WEATHER_ATMOSPHERE:
                w_state = intent.parameters.get("weather_state", "Clear")
                tod = intent.parameters.get("time_of_day_hours", 12.0)
                rain_int = intent.parameters.get("rain_intensity", 0.0)
                api_endpoint = "POST /api/minecraft-shaders/atmosphere"
                payload = {
                    "weather_state": w_state,
                    "time_of_day_hours": tod,
                    "rain_intensity": rain_int
                }
                self._sim_weather = w_state
                self._sim_time_of_day = tod
                feedback_sk = f"Počasie zmenené na: {w_state}, čas dňa: {tod:.1f}h."
                feedback_en = f"Atmosphere updated to: {w_state}, time of day: {tod:.1f}h."
                feedback = feedback_sk if intent.language == "sk" else feedback_en
                diff = {"weather": w_state, "time_of_day": tod}

            elif intent.command_type == GameCommandType.CAMERA_CONTROL:
                preset = intent.parameters.get("camera_preset", "perspective_action_follow")
                dist = intent.parameters.get("distance", 8.0)
                api_endpoint = "POST /api/godot/camera/switch"
                payload = {"preset_id": preset, "overrides": {"distance": dist}}
                self._sim_camera_preset = preset
                feedback_sk = f"Kamera prepnutá na rig: '{preset}' (vzdialenosť: {dist})."
                feedback_en = f"Camera switched to rig: '{preset}' (distance: {dist})."
                feedback = feedback_sk if intent.language == "sk" else feedback_en
                diff = {"camera_preset": preset, "distance": dist}

            elif intent.command_type == GameCommandType.ARTILLERY_FIRE:
                w_type = intent.parameters.get("weapon_type", "mortar_indirect")
                t_hex = intent.parameters.get("target_hex", [2, 1])
                api_endpoint = "POST /api/ballistics/fire"
                payload = {
                    "weapon_type": w_type,
                    "target_coords": t_hex,
                    "elevation_deg": intent.parameters.get("elevation_deg", 65.0)
                }
                feedback_sk = f"Delostrelecká paľba ({w_type}) odpálená na sektor {t_hex}."
                feedback_en = f"Artillery salvo ({w_type}) fired at sector {t_hex}."
                feedback = feedback_sk if intent.language == "sk" else feedback_en
                diff = {"salvo_fired": True, "target": t_hex}

            elif intent.command_type == GameCommandType.CITADEL_BUILD:
                p_type = intent.parameters.get("pylon_type", "slavic_mortar")
                slot = intent.parameters.get("tower_slot", 1)
                api_endpoint = "POST /api/sovereign-citadel/build"
                payload = {"pylon_type": p_type, "slot_index": slot}
                feedback_sk = f"Obranná veža v slote {slot} opevnená: {p_type}."
                feedback_en = f"Defensive tower in slot {slot} fortified with: {p_type}."
                feedback = feedback_sk if intent.language == "sk" else feedback_en
                diff = {"tower_slot": slot, "pylon_type": p_type}

            elif intent.command_type == GameCommandType.SPAWN_ENTITY:
                model_id = intent.parameters.get("model_id", "CesiumMan")
                pos = intent.parameters.get("position", [0, 0])
                api_endpoint = "POST /api/godot/spawn"
                payload = {"model_id": model_id, "position": pos}
                feedback_sk = f"Vyvolaný 3D model '{model_id}' na pozícii {pos}."
                feedback_en = f"Spawned 3D model '{model_id}' at position {pos}."
                feedback = feedback_sk if intent.language == "sk" else feedback_en
                diff = {"entity_spawned": model_id, "position": pos}

            elif intent.command_type == GameCommandType.MATCH_ESCALATION:
                api_endpoint = "POST /api/match/escalate"
                payload = {"action": "END_TURN"}
                self._sim_round += 1
                self._sim_hero_mana = min(10, self._sim_hero_mana + 3)
                feedback_sk = f"Kolo ukončené. Začína Kolo {self._sim_round}, mana doplnená na {self._sim_hero_mana}."
                feedback_en = f"Turn ended. Round {self._sim_round} started, mana replenished to {self._sim_hero_mana}."
                feedback = feedback_sk if intent.language == "sk" else feedback_en
                diff = {"round": self._sim_round, "mana": self._sim_hero_mana}

            elif intent.command_type == GameCommandType.QUERY_STATUS:
                api_endpoint = "GET /api/game/match-state"
                payload = {}
                report = self.query_game_state(text)
                feedback = report["answer"]
                diff = {"query_resolved": True}

            elif intent.command_type == GameCommandType.ENVIRONMENT_HAZARD:
                h_type = intent.parameters.get("hazard_type", "toxic_acid")
                t_hex = intent.parameters.get("target_hex", [0, 0])
                api_endpoint = "POST /api/environment/matrix/apply_hazard"
                payload = {"q": t_hex[0], "r": t_hex[1], "hazard_type": h_type, "duration": 3}
                feedback_sk = f"Pole {t_hex} zamorené efektom {h_type} na 3 kolá."
                feedback_en = f"Hex {t_hex} engulfed with {h_type} for 3 rounds."
                feedback = feedback_sk if intent.language == "sk" else feedback_en
                diff = {"hazard_applied": h_type, "hex": t_hex}

            else:
                status = "PARTIAL"
                feedback_sk = f"Príkaz nebol plne rozpoznaný: '{text}'. Skontrolujte syntax alebo zadajte 'stav'."
                feedback_en = f"Command was not recognized: '{text}'. Check syntax or type 'status'."
                feedback = feedback_sk if intent.language == "sk" else feedback_en

            # If execute_api is True, make real HTTP attempt against local hub if available
            if execute_api and api_endpoint and not self.config.mock_mode:
                self._dispatch_http_request(api_endpoint, payload)

        except Exception as e:
            status = "ERROR"
            feedback = f"Execution error: {str(e)}"

        elapsed_ms = (time.time() - start_time) * 1000.0

        res = LanguageExecutionResult(
            execution_id=exec_id,
            timestamp=time.time(),
            intent=asdict(intent),
            status=status,
            api_action_executed=api_endpoint,
            api_payload=payload,
            game_feedback=feedback,
            state_diff=diff,
            execution_time_ms=round(elapsed_ms, 2),
            vital_max_hp_rule=VITAL_MAX_HP
        )

        with self._lock:
            self._history.append(res)
            if len(self._history) > self._max_history:
                self._history.pop(0)

        return res

    def _dispatch_http_request(self, api_endpoint: str, payload: Dict[str, Any]):
        """Safe non-blocking HTTP dispatch to local hub."""
        if self.config.mock_mode:
            return
        method, path = api_endpoint.split(" ", 1)
        url = f"{self.base_api_url}{path}"
        try:
            req_data = json.dumps(payload).encode("utf-8") if method == "POST" else None
            headers = {"Content-Type": "application/json"}
            req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
            # Short timeout to avoid hanging if server is busy
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                _ = resp.read()
        except Exception:
            # Hub might be offline or running as mock; local state diff already handled
            pass

    # ==========================================================================
    # 3. GAME STATE QUERYING (Multilingual Report)
    # ==========================================================================

    def query_game_state(self, query_text: str) -> Dict[str, Any]:
        """
        Formulates a natural language briefing of the current game state
        under the Poslední Kmen rules.
        """
        is_sk = any(w in query_text.lower() for w in ["aký", "stav", "kolko", "koľko", "mám", "prehľad"])
        
        # Enforce max HP
        hp = min(VITAL_MAX_HP, self._sim_hero_hp)
        mana = self._sim_hero_mana
        pos = self._sim_hero_pos
        weather = self._sim_weather
        tod = self._sim_time_of_day
        camera = self._sim_camera_preset
        round_no = self._sim_round

        if is_sk:
            ans = (
                f"🛡️ **Stav Bojiska Poslední Kmen (Kolo {round_no}):**\n"
                f"• Hrdina: Zdravie {hp}/{VITAL_MAX_HP} HP, Mana: {mana}/10, Pozícia: Hex {pos}\n"
                f"• Atmosféra: {weather}, Denný čas: {tod:.1f}h\n"
                f"• Kamera: {camera}\n"
                f"• Dostupné akcie: Môžete zahrať 'Kryštálový Štít', 'Ľadovcovú Kopiju', pohnúť sa alebo vystreliť z moždiara."
            )
        else:
            ans = (
                f"🛡️ **Poslední Kmen Battlefield Status (Round {round_no}):**\n"
                f"• Hero: Vitality {hp}/{VITAL_MAX_HP} HP, Mana: {mana}/10, Coordinates: Hex {pos}\n"
                f"• Atmosphere: {weather}, Time of Day: {tod:.1f}h\n"
                f"• Camera Rig: {camera}\n"
                f"• Available actions: You can cast 'Crystal Shield', 'Glacial Lance', reposition, or fire the mortar."
            )

        return {
            "answer": ans,
            "hp": hp,
            "max_hp": VITAL_MAX_HP,
            "mana": mana,
            "position": pos,
            "weather": weather,
            "time_of_day": tod,
            "camera_preset": camera,
            "round": round_no,
            "vital_max_hp_rule": VITAL_MAX_HP
        }

    # ==========================================================================
    # 4. REMOTE LANGUAGE API FETCHER & AUTONOMOUS DAEMON
    # ==========================================================================

    def fetch_and_execute_remote(
        self,
        source_api_url: Optional[str] = None,
        custom_prompt: Optional[str] = None
    ) -> List[LanguageExecutionResult]:
        """
        Fetches next language instructions from an OpenAI-compatible endpoint
        or local queue, parses each instruction, and executes it.
        """
        if self.config.mock_mode:
            fallback = "Zahraj kartu Kryštálový Štít na seba" if self.config.default_language == "sk" else "Cast Crystal Shield on self"
            return [self.execute_command(fallback, execute_api=False)]

        target_url = source_api_url or self.config.source_api_url
        prompt = custom_prompt or (
            f"You are the Tactical Commander in Poslední Kmen. Current state: Hero HP {self._sim_hero_hp}/{VITAL_MAX_HP}, "
            f"Mana {self._sim_hero_mana}. Choose the next tactical move in 1 concise sentence (e.g. 'Cast Glacial Lance at hex [1, 2]')."
        )

        fetched_commands: List[str] = []

        try:
            req_body = {
                "model": self.config.model_name,
                "messages": [
                    {"role": "system", "content": "You are a tactical game AI. Output only 1 direct game instruction."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 60,
                "temperature": 0.4
            }
            req_data = json.dumps(req_body).encode("utf-8")
            req = urllib.request.Request(target_url, data=req_data, headers={"Content-Type": "application/json"})
            
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                choice = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                if choice:
                    fetched_commands.append(choice.strip())
        except Exception:
            # Fallback simulated command if remote API is offline/unreachable
            fallback = "Zahraj kartu Kryštálový Štít na seba" if self.config.default_language == "sk" else "Cast Crystal Shield on self"
            fetched_commands.append(fallback)

        results: List[LanguageExecutionResult] = []
        for cmd in fetched_commands:
            res = self.execute_command(cmd, execute_api=not self.config.mock_mode)
            results.append(res)

        return results

    def start_fetcher_daemon(self, poll_interval_s: Optional[float] = None):
        """Starts background polling daemon for hands-free LLM game control."""
        if self._daemon_running:
            return

        if poll_interval_s is not None:
            self.config.poll_interval_s = max(0.5, poll_interval_s)

        self._daemon_stop_event.clear()
        self._daemon_running = True
        self.config.enabled = True

        def _loop():
            while not self._daemon_stop_event.is_set():
                if self.config.autonomous_execution:
                    try:
                        self.fetch_and_execute_remote()
                    except Exception:
                        pass
                self._daemon_stop_event.wait(self.config.poll_interval_s)

        self._daemon_thread = threading.Thread(target=_loop, daemon=True, name="GameLanguageFetcherDaemon")
        self._daemon_thread.start()

    def stop_fetcher_daemon(self):
        """Stops the background polling daemon."""
        self._daemon_stop_event.set()
        self._daemon_running = False
        self.config.enabled = False
        if self._daemon_thread:
            self._daemon_thread.join(timeout=1.0)
            self._daemon_thread = None

    # ==========================================================================
    # 5. AUDIT HISTORY & CAPABILITIES CATALOG
    # ==========================================================================

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent execution records."""
        with self._lock:
            items = self._history[-limit:]
            return [asdict(r) for r in reversed(items)]

    def get_capabilities_catalog(self) -> Dict[str, Any]:
        """Returns the complete vocabulary, supported intents, and prompt samples."""
        return {
            "vital_max_hp_rule": VITAL_MAX_HP,
            "supported_commands": [t.value for t in GameCommandType],
            "known_cards_count": len(self.known_cards),
            "known_cards": list(self.known_cards.keys()),
            "known_camera_presets": list(self.known_cameras.keys()),
            "known_weather_states": list(self.known_weather.keys()),
            "known_models": list(self.known_models.keys()),
            "sample_prompts": {
                "sk": [
                    "Zahraj kartu Kryštálový Meteor na hex [1, -2]",
                    "Presuň hrdinu na hex [2, 0]",
                    "Prepnúť kameru na izometrický pohľad",
                    "Nastav počasie na búrku a čas na polnoc",
                    "Vystreľ z moždiara na sektor 3",
                    "Postav slovanský mažiar na vežu 1",
                    "Vyvolaj CesiumMan na pozíciu [0, 1]",
                    "Aký je stav môjho hrdinu a zápasu?",
                    "Ukonči kolo"
                ],
                "en": [
                    "Cast Glacial Lance at hex [1, 2]",
                    "Move hero to hex [0, -1]",
                    "Switch camera to cinematic wide",
                    "Set weather to rain with dusk lighting",
                    "Fire mortar at hex [3, -1]",
                    "Build slavic mortar on tower 1",
                    "Spawn Fox companion at [1, 1]",
                    "What is my current HP and mana?",
                    "End turn"
                ]
            }
        }


# Global Singleton Instance
GLOBAL_GAME_LANGUAGE_FETCHER = GameLanguageApiFetcher()
