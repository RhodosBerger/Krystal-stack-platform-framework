# ==============================================================================
# KRYSTAL-STACK: JANET TRANSMITTERS, NOCTURNAL SKY, WEATHER ENTROPY & COSMOLOGY
# ==============================================================================
# Implements:
#   1. Janet-Compatible Bot Transmitters & Premium Step-by-Step Travel Mode.
#   2. Debug Functions: Step Forward, Rewind History, Teleport, Stimulus Injection.
#   3. Nocturnal Atmospheric Physics: Aurora Borealis, Spirit Rays, Astral Apparitions.
#   4. Thermodynamic Weather Entropy & Natural Catastrophes (Storms, Floods, Rifts).
#   5. Intertwined Subterranean Tree Network (Wood Wide Web & Mycorrhizal Mana Grid).
#   6. Lake Mirrors, Spatial Portals & Subterranean Caverns.
#   7. Five Cosmological Planes (Peklo/Tartarus, Jaskyne, Aréna, Aéter, Nebo).
#   8. The Twelve Apostles & Angelic Guardians Divine Patronage Profiles.
# ==============================================================================

import math
import time
import secrets
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

# ------------------------------------------------------------------------------
# 1. COSMOLOGICAL PLANES (PÄŤ POSCHODÍ EXISTENCIE)
# ------------------------------------------------------------------------------
class CosmologicalPlane(str, Enum):
    ABYSSAL_INFERNO = "abyssal_inferno"           # Tier -2: Peklo / Tartarus
    SUBTERRANEAN_CAVERNS = "subterranean_caverns" # Tier -1: Jaskyne, Zrkadlá, Podsvetie
    MORTAL_TERRESTRIAL = "mortal_terrestrial"     # Tier  0: Aréna Poslední Kmen, Povrch
    AETHERIC_SKY = "aetheric_sky"                 # Tier  1: Oblačná Sféra, Plávajúce Ostrovy
    EMPYREAN_HEAVEN = "empyrean_heaven"           # Tier  2: Nebo, Archónsky Trón, Božská Harmónia

COSMOLOGICAL_PLANE_DATA: Dict[CosmologicalPlane, Dict[str, Any]] = {
    CosmologicalPlane.ABYSSAL_INFERNO: {
        "tier": -2,
        "name": "Peklo (Tartarus / Abyssal Inferno)",
        "entropy_base": 0.95,
        "ambient_color": "#ff1a1a",
        "gravitational_pull": 1.45,
        "hazard": "Sulfuric Magma & Hellfire",
        "description": "Najnižšie poschodie existencie. Žeravá láva, extrémna entropia a neustály nepokoj."
    },
    CosmologicalPlane.SUBTERRANEAN_CAVERNS: {
        "tier": -1,
        "name": "Podzemné Jaskyne a Jazerové Zrkadlá",
        "entropy_base": 0.55,
        "ambient_color": "#1c2b36",
        "gravitational_pull": 1.15,
        "hazard": "Stalactite Crushes & Subterranean Dampening",
        "description": "Priestor hlbokých dutín, prepletených koreňových sietí a zrkadliacich vodných portálov."
    },
    CosmologicalPlane.MORTAL_TERRESTRIAL: {
        "tier": 0,
        "name": "Smrteľný Pozemský Svet (Aréna Poslední Kmen)",
        "entropy_base": 0.30,
        "ambient_color": "#66fcf1",
        "gravitational_pull": 1.00,
        "hazard": "Toxic Slime Puddles & Warfront Cataclysms",
        "description": "Hlavné bojisko troch kmeňov (Kryštálový, Jedovatý, Druidi) s vyváženou fyzikou."
    },
    CosmologicalPlane.AETHERIC_SKY: {
        "tier": 1,
        "name": "Oblačná Sféra (Aéterová Obloha)",
        "entropy_base": 0.15,
        "ambient_color": "#a8e6cf",
        "gravitational_pull": 0.65,
        "hazard": "Volumetric Spirit Vortexes",
        "description": "Plávajúce ostrovy, svetelné lúče duchov a polárna žiara prelínajúca sa nočnou oblohou."
    },
    CosmologicalPlane.EMPYREAN_HEAVEN: {
        "tier": 2,
        "name": "Nebeský Trón (Empyrean Heaven)",
        "entropy_base": 0.00,
        "ambient_color": "#ffd700",
        "gravitational_pull": 0.40,
        "hazard": "Divine Transcendence Resonance",
        "description": "Najvyššie poschodie nebeskej slávy, sídlo dvanástich apoštolov a chóru serafínov."
    }
}


# ------------------------------------------------------------------------------
# 2. JANET BOT TRANSMITTER & PREMIUM TRAVEL STEP CONTROLLER
# ------------------------------------------------------------------------------
class BotTransmitter:
    """
    Simulates a high-frequency transmitter channel communicating with bots,
    managing step-by-step travel in Premium Mode, and handling time-travel rewinds.
    """
    def __init__(
        self,
        bot_id: str,
        channel_freq_mhz: float = 433.92,
        plane: CosmologicalPlane = CosmologicalPlane.MORTAL_TERRESTRIAL
    ):
        self.bot_id = bot_id
        self.channel_freq_mhz = channel_freq_mhz
        self.plane = plane
        self.position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.step_index: int = 0
        self.premium_mode: bool = True
        self.god_mode: bool = False
        self.active_ward: int = 6
        self.active_mana: int = 10
        self.history_stack: List[Dict[str, Any]] = []
        self.waypoints: List[Tuple[float, float, float]] = []
        self.active_stimuli: List[str] = []

    def step_forward(self, delta_vector: Tuple[float, float, float]) -> Dict[str, Any]:
        """
        Executes a single discrete step in Premium Travel Mode.
        Saves current state snapshot into history for instant rewind.
        """
        snapshot = {
            "step": self.step_index,
            "position": self.position,
            "ward": self.active_ward,
            "mana": self.active_mana,
            "plane": self.plane.value,
            "timestamp": time.time()
        }
        self.history_stack.append(snapshot)

        new_x = round(self.position[0] + delta_vector[0], 2)
        new_y = round(self.position[1] + delta_vector[1], 2)
        new_z = round(self.position[2] + delta_vector[2], 2)
        self.position = (new_x, new_y, new_z)
        self.step_index += 1

        return {
            "success": True,
            "bot_id": self.bot_id,
            "step_index": self.step_index,
            "position": self.position,
            "plane": self.plane.value,
            "history_depth": len(self.history_stack)
        }

    def rewind_step(self) -> Dict[str, Any]:
        """
        Time-travel rewind: pops the previous step state from history.
        """
        if not self.history_stack:
            return {"success": False, "error": "HISTORY_STACK_EMPTY"}

        previous_state = self.history_stack.pop()
        self.position = previous_state["position"]
        self.step_index = previous_state["step"]
        self.active_ward = previous_state["ward"]
        self.active_mana = previous_state["mana"]
        self.plane = CosmologicalPlane(previous_state["plane"])

        return {
            "success": True,
            "rewound_to_step": self.step_index,
            "position": self.position,
            "plane": self.plane.value,
            "history_remaining": len(self.history_stack)
        }

    def teleport_bot(self, target_pos: Tuple[float, float, float], target_plane: Optional[CosmologicalPlane] = None) -> Dict[str, Any]:
        """
        Debug teleport function: instantaneously places bot at coordinate and plane.
        """
        self.position = target_pos
        if target_plane:
            self.plane = target_plane
        return {
            "success": True,
            "teleported_to": self.position,
            "plane": self.plane.value,
            "bot_id": self.bot_id
        }

    def inject_stimulus(self, stimulus_name: str) -> Dict[str, Any]:
        """
        Debug function: Injects environmental trigger (e.g. 'AETHER_SURGE', 'DIVINE_GRACE').
        """
        self.active_stimuli.append(stimulus_name)
        return {
            "success": True,
            "stimulus_injected": stimulus_name,
            "active_stimuli": self.active_stimuli
        }

    def toggle_god_mode(self) -> bool:
        self.god_mode = not self.god_mode
        if self.god_mode:
            self.active_ward = 999
            self.active_mana = 999
        else:
            self.active_ward = 6
            self.active_mana = 10
        return self.god_mode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "channel_freq_mhz": self.channel_freq_mhz,
            "plane": self.plane.value,
            "position": self.position,
            "step_index": self.step_index,
            "premium_mode": self.premium_mode,
            "god_mode": self.god_mode,
            "active_ward": self.active_ward,
            "active_mana": self.active_mana,
            "history_depth": len(self.history_stack),
            "active_stimuli": self.active_stimuli
        }


# ------------------------------------------------------------------------------
# 3. NOCTURNAL SKY ATMOSPHERIC PHYSICS, SPIRIT RAYS & SPECTRAL ENTITIES
# ------------------------------------------------------------------------------
class NocturnalAtmosphereEngine:
    """
    Simulates nocturnal celestial optics: Aurora Borealis ribbons,
    Rayleigh/Mie scattering from the moon, volumetric spirit light beams,
    and wandering astral spirit entities.
    """

    @staticmethod
    def calculate_nocturnal_sky_optics(
        celestial_time_sec: float,
        lunar_phase: float = 0.50
    ) -> Dict[str, Any]:
        """
        Computes celestial light scattering and volumetric crepuscular rays.
        """
        t = celestial_time_sec
        lunar_intensity = max(0.1, math.sin(lunar_phase * math.pi))

        # Rayleigh & Mie scattering coefficient
        rayleigh_coeff = round(0.05 + 0.08 * math.sin(t * 0.1), 4)
        mie_scattering = round(0.12 + 0.05 * math.cos(t * 0.15), 4)

        # Volumetric Spirit Light Beams (Crepuscular Rays)
        ray_count = int(math.floor(4.0 + 3.0 * math.sin(t * 0.2)))
        ray_angles_deg = [round((i * (360.0 / max(1, ray_count)) + (t * 5.0)) % 360.0, 1) for i in range(ray_count)]

        # Aurora Borealis color oscillation
        r = int(60 + 50 * math.sin(t * 0.3))
        g = int(220 + 35 * math.cos(t * 0.2))
        b = int(180 + 75 * math.sin(t * 0.25))
        aurora_hex = f"#{r:02x}{g:02x}{b:02x}"

        # Spectral Spirit Entities wandering the nocturnal sky
        spirit_entities = [
            {
                "id": "spirit_will_o_wisp_01",
                "name": "Bludné Svetielko (Will-o'-the-Wisp)",
                "type": "spectral_entity",
                "luminescence_lumens": 1200,
                "position_y": 8.5 + 2.0 * math.sin(t * 0.5),
                "spirit_spell": "Ghostfire Flash",
                "color": "#66fcf1"
            },
            {
                "id": "spirit_astral_phantom_02",
                "name": "Astrálny Prízrak Oblohy",
                "type": "spectral_entity",
                "luminescence_lumens": 2500,
                "position_y": 24.0 + 4.0 * math.cos(t * 0.3),
                "spirit_spell": "Ethereal Phase Shift",
                "color": "#a8e6cf"
            },
            {
                "id": "spirit_aether_apparition_03",
                "name": "Zjav Aéterovej Polárnej Žiary",
                "type": "spectral_entity",
                "luminescence_lumens": 4000,
                "position_y": 45.0 + 3.0 * math.sin(t * 0.15),
                "spirit_spell": "Astral Surge & Starlight Beam",
                "color": "#ffd700"
            }
        ]

        return {
            "lunar_intensity": round(lunar_intensity, 2),
            "rayleigh_scattering": rayleigh_coeff,
            "mie_scattering": mie_scattering,
            "aurora_borealis_ribbon_color": aurora_hex,
            "volumetric_spirit_rays": {
                "count": ray_count,
                "angles_deg": ray_angles_deg,
                "beam_intensity": round(lunar_intensity * 2.2, 2)
            },
            "spectral_spirit_entities": spirit_entities
        }


# ------------------------------------------------------------------------------
# 4. ENTROPY-DRIVEN WEATHER & NATURAL CATASTROPHES
# ------------------------------------------------------------------------------
class EntropyWeatherEngine:
    """
    Calculates atmospheric entropy S in [0.0, 1.0] and drives weather states
    and natural catastrophes (storms, flash floods, acid rain, volcanic rifts).
    """

    @staticmethod
    def calculate_weather_entropy(
        barometric_pressure_delta: float,
        relative_humidity_pct: float,
        wind_shear_mps: float
    ) -> Dict[str, Any]:
        """
        Thermodynamic entropy formula normalized to [0.0, 1.0].
        """
        raw_s = (abs(barometric_pressure_delta) * 0.40 + relative_humidity_pct * 0.30 + wind_shear_mps * 0.30) / 100.0
        entropy = round(max(0.0, min(1.0, raw_s)), 3)

        if entropy < 0.20:
            state = "CLEAR_STARRY_NIGHT"
            catastrophe = "None (Dokonalý pokoj)"
            hazard_dmg = 0
        elif entropy < 0.45:
            state = "DRUIDIC_VERDANT_RAIN"
            catastrophe = "None (Úrodný dážď, regenerácia +1 HP)"
            hazard_dmg = 0
        elif entropy < 0.65:
            state = "ACID_TOXIC_FOG"
            catastrophe = "Toxická Korózia (Znižuje pancier o 1 bod)"
            hazard_dmg = 1
        elif entropy < 0.85:
            state = "SUPERCELL_LIGHTNING_STORM"
            catastrophe = "Búrková Kanonáda a Zásahy Bleskov (2 Bleskové DMG)"
            hazard_dmg = 2
        else:
            state = "AETHERIC_CATACLYSM_RIFT"
            catastrophe = "Priestorová Katastrofa: Trhlina Prázdnoty a Prívalová Povodeň (3 Drvivé DMG)"
            hazard_dmg = 3

        return {
            "entropy_value": entropy,
            "weather_state": state,
            "catastrophe_event": catastrophe,
            "is_catastrophe_active": entropy >= 0.65,
            "hazard_damage_per_turn": hazard_dmg,
            "storm_surge_power": round(entropy * 10.0, 1)
        }


# ------------------------------------------------------------------------------
# 5. ARBOR & MYCORRHIZAL TREE NETWORK (WOOD WIDE WEB)
# ------------------------------------------------------------------------------
class ArborMycorrhizalNetwork:
    """
    Subterranean root and fungal network connecting sacred grove trees,
    transmitting mana and collective protective barriers across the landscape.
    """

    DEFAULT_SACRED_TREES = [
        {"id": "tree_mother_oak", "name": "Materský Dub Života", "x": 0.0, "z": 0.0, "mana_output": 50, "ward_share": 3},
        {"id": "tree_crystal_pine", "name": "Kryštálová Prapôvodná Borovica", "x": 25.0, "z": 15.0, "mana_output": 40, "ward_share": 2},
        {"id": "tree_toxic_willow", "name": "Kyslá Vŕba Miazmy", "x": -20.0, "z": 30.0, "mana_output": 35, "ward_share": 2},
        {"id": "tree_elder_yew", "name": "Tis Prastarých Druidov", "x": -15.0, "z": -25.0, "mana_output": 45, "ward_share": 3}
    ]

    @classmethod
    def evaluate_root_network(cls, connected_nodes_count: int = 4) -> Dict[str, Any]:
        trees = cls.DEFAULT_SACRED_TREES[:connected_nodes_count]
        total_mana_pool = sum(t["mana_output"] for t in trees)
        collective_barrier = sum(t["ward_share"] for t in trees)

        # Subterranean ley-line connections (graph edges)
        channels = []
        for i in range(len(trees)):
            t1 = trees[i]
            t2 = trees[(i + 1) % len(trees)]
            channels.append({
                "from_tree": t1["id"],
                "to_tree": t2["id"],
                "channel_throughput_bps": 1000,
                "status": "HYPHA_FLOW_ACTIVE"
            })

        return {
            "network_status": "INTERTWINED_HEALTHY",
            "active_tree_nodes_count": len(trees),
            "total_subterranean_mana_pool": total_mana_pool,
            "collective_ward_barrier": collective_barrier,
            "mycorrhizal_channels": channels
        }


# ------------------------------------------------------------------------------
# 6. LAKE MIRRORS, SPATIAL PORTALS & SUBTERRANEAN CAVERNS
# ------------------------------------------------------------------------------
class DimensionalPortalsAndMirrors:
    """
    Manages reflective lake mirrors, spatial portals between planes,
    and cavern acoustic/environmental properties.
    """

    PORTAL_GATEWAYS = [
        {
            "portal_id": "lake_mirror_portal_01",
            "name": "Jazerové Zrkadlo Odrazu (Lake Mirror Gateway)",
            "origin_plane": CosmologicalPlane.MORTAL_TERRESTRIAL.value,
            "destination_plane": CosmologicalPlane.SUBTERRANEAN_CAVERNS.value,
            "surface_reflection_matrix": "Planar 4x4 Inverted Z",
            "status": "OPEN_SURFACE"
        },
        {
            "portal_id": "aether_sky_rift_02",
            "name": "Nebeská Hviezdna Brána (Aether Stargate)",
            "origin_plane": CosmologicalPlane.MORTAL_TERRESTRIAL.value,
            "destination_plane": CosmologicalPlane.AETHERIC_SKY.value,
            "surface_reflection_matrix": "Aurora Vortex Lens",
            "status": "OPEN_CELESTIAL"
        },
        {
            "portal_id": "tartarus_rift_03",
            "name": "Trhlina Pekelnej Priepasti (Abyssal Gate)",
            "origin_plane": CosmologicalPlane.SUBTERRANEAN_CAVERNS.value,
            "destination_plane": CosmologicalPlane.ABYSSAL_INFERNO.value,
            "surface_reflection_matrix": "Molten Obsidian Reflection",
            "status": "SEALED_REQUIRES_KEY_OF_PETER"
        },
        {
            "portal_id": "empyrean_ascent_04",
            "name": "Zlatá Brána Nebeského Trónu",
            "origin_plane": CosmologicalPlane.AETHERIC_SKY.value,
            "destination_plane": CosmologicalPlane.EMPYREAN_HEAVEN.value,
            "surface_reflection_matrix": "Prismatic Divine Mirror",
            "status": "ANGELIC_CHOIR_OPEN"
        }
    ]

    @classmethod
    def get_portals(cls) -> List[Dict[str, Any]]:
        return cls.PORTAL_GATEWAYS

    @staticmethod
    def inspect_cavern_depth(cavern_depth_meters: float) -> Dict[str, Any]:
        """
        Calculates echo dampening, darkness index, and stalactite risk.
        """
        darkness = min(1.0, cavern_depth_meters / 100.0)
        stalactite_risk = min(0.85, (cavern_depth_meters * 0.008))
        echo_delay_ms = round(cavern_depth_meters * 5.8, 1)

        return {
            "cavern_depth_m": cavern_depth_meters,
            "darkness_factor": round(darkness, 2),
            "stalactite_fall_risk": round(stalactite_risk, 2),
            "acoustic_echo_delay_ms": echo_delay_ms,
            "subterranean_ley_resonance": "RESONANT_ROOT_CHAMBER"
        }


# ------------------------------------------------------------------------------
# 7. THE TWELVE APOSTLES & ANGELIC GUARDIANS (DIVINE PATRONAGE)
# ------------------------------------------------------------------------------
def _make_apostle(
    apostle_id: str,
    name: str,
    title: str,
    symbol: str,
    warhammer_stats: Dict[str, Any],
    duel_stats: Dict[str, int],
    signature_divine_spell: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "id": apostle_id,
        "name": name,
        "title": title,
        "symbol": symbol,
        "warhammer_stats": warhammer_stats,
        "duel_stats": duel_stats,
        "signature_divine_spell": signature_divine_spell,
        "patronage_plane": CosmologicalPlane.EMPYREAN_HEAVEN.value
    }

TWELVE_APOSTLES: List[Dict[str, Any]] = [
    _make_apostle("apostle_peter_01", "Šimon Peter", "Skala Cirkvi (Cephas)", "Kľúče Kráľovstva & Prevrátený Kríž",
                  {"WS": 2, "BS": 2, "S": 5, "T": 6, "W": 6, "A": 4, "Ld": 10, "Sv": "2+"},
                  {"toughness": 38, "reflexes": 28, "aim": 30, "dodge": 28, "appearance": 40, "tactics": 45, "mobility": 18},
                  {"name": "Pevná Skala Viery", "desc": "Nerozbitný štít absorbujúci 10 poškodenia a uzamykajúci brány pekla.", "cost_mana": 4}),

    _make_apostle("apostle_andrew_02", "Ondrej", "Prvopovolaný Rybár", "Ondrejský Kríž v tvare X",
                  {"WS": 3, "BS": 2, "S": 4, "T": 5, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
                  {"toughness": 28, "reflexes": 34, "aim": 32, "dodge": 32, "appearance": 34, "tactics": 36, "mobility": 24},
                  {"name": "Sieť Rybára Ľudí", "desc": "Zviaže všetkých protivníkov v sektore do posvätnej vodnej siete.", "cost_mana": 3}),

    _make_apostle("apostle_james_elder_03", "Jakub Starší", "Syn Hromu (Boanerges)", "Pútnická Palica a Mušľa",
                  {"WS": 2, "BS": 2, "S": 5, "T": 5, "W": 6, "A": 4, "Ld": 10, "Sv": "2+"},
                  {"toughness": 32, "reflexes": 30, "aim": 34, "dodge": 26, "appearance": 38, "tactics": 38, "mobility": 20},
                  {"name": "Búrková Kanonáda Boží Hrom", "desc": "Zosiela bleskový úder z nebies za 5 plošných DMG.", "cost_mana": 4}),

    _make_apostle("apostle_john_04", "Ján Evanjelista", "Milovaný Učeník & Syn Svetla", "Orol a Kalich so Zmijov",
                  {"WS": 3, "BS": 2, "S": 4, "T": 5, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
                  {"toughness": 26, "reflexes": 36, "aim": 36, "dodge": 34, "appearance": 48, "tactics": 42, "mobility": 26},
                  {"name": "Zjavenie Večného Svetla", "desc": "Ožiari celú nočnú oblohu a odhalí všetkých skrytých nepriateľov.", "cost_mana": 3}),

    _make_apostle("apostle_philip_05", "Filip", "Zvestovateľ Chleba", "Chlieb a Dva Kríže",
                  {"WS": 3, "BS": 3, "S": 4, "T": 5, "W": 6, "A": 3, "Ld": 9, "Sv": "3+"},
                  {"toughness": 28, "reflexes": 26, "aim": 28, "dodge": 26, "appearance": 36, "tactics": 35, "mobility": 22},
                  {"name": "Rozmnoženie Životodarnej Manny", "desc": "Doplní 5 bodov many a ošetrí zranených spojencov.", "cost_mana": 2}),

    _make_apostle("apostle_bartholomew_06", "Bartolomej (Natanael)", "Muž Čistého Srdca", "Nôž a Stiahnutá Koža",
                  {"WS": 2, "BS": 3, "S": 5, "T": 6, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
                  {"toughness": 36, "reflexes": 24, "aim": 30, "dodge": 24, "appearance": 35, "tactics": 38, "mobility": 18},
                  {"name": "Nezlomné Mučeníctvo", "desc": "Ignoruje akékoľvek zníženie životov pod 1 HP po dobu 1 kola.", "cost_mana": 4}),

    _make_apostle("apostle_thomas_07", "Tomáš (Didymos)", "Pátrač po Pravde", "Staviteľský Uhlomer a Oštep",
                  {"WS": 3, "BS": 2, "S": 4, "T": 5, "W": 6, "A": 3, "Ld": 9, "Sv": "3+"},
                  {"toughness": 28, "reflexes": 32, "aim": 38, "dodge": 30, "appearance": 32, "tactics": 44, "mobility": 22},
                  {"name": "Prierazný Lúč Pravdy", "desc": "Chirurgický zásah ignorujúci akékoľvek krytie a klamlivé ilúzie.", "cost_mana": 3}),

    _make_apostle("apostle_matthew_08", "Matúš (Lévi)", "Strážca Nebeskej Knihy", "Kniha Účtov a Peniaze",
                  {"WS": 3, "BS": 3, "S": 4, "T": 5, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
                  {"toughness": 26, "reflexes": 28, "aim": 30, "dodge": 28, "appearance": 38, "tactics": 46, "mobility": 20},
                  {"name": "Spravodlivé Vyrovnanie Účtov", "desc": "Ukradnuté Nugety a zdroje nepriateľa sa obrátia proti nemu.", "cost_mana": 3}),

    _make_apostle("apostle_james_less_09", "Jakub Mladší", "Spravodlivý Pilier Jeruzalema", "Pltnícka Valcha a Kniha",
                  {"WS": 2, "BS": 3, "S": 5, "T": 6, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
                  {"toughness": 34, "reflexes": 25, "aim": 28, "dodge": 24, "appearance": 37, "tactics": 42, "mobility": 16},
                  {"name": "Pilier Neotrasiteľnosti", "desc": "Zablokuje akékoľvek posunutie, odhodenie a pád jednotiek.", "cost_mana": 3}),

    _make_apostle("apostle_jude_thaddaeus_10", "Júda Tadeáš", "Patrón Zúfalých a Neriešiteľných Situácií", "Obraz Mandylion a Palica",
                  {"WS": 3, "BS": 3, "S": 4, "T": 5, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
                  {"toughness": 30, "reflexes": 30, "aim": 30, "dodge": 30, "appearance": 45, "tactics": 40, "mobility": 22},
                  {"name": "Zázrak v Poslednej Sekunde", "desc": "Oživí padlého hrdinu zo stavu smrti späť na plných 6 HP.", "cost_mana": 5}),

    _make_apostle("apostle_simon_zealot_11", "Šimon Horlivec", "Svätý Očisťovateľ", "Píla a Horiaci Fakľový Plameň",
                  {"WS": 2, "BS": 2, "S": 5, "T": 5, "W": 6, "A": 4, "Ld": 9, "Sv": "3+"},
                  {"toughness": 32, "reflexes": 32, "aim": 32, "dodge": 28, "appearance": 42, "tactics": 36, "mobility": 24},
                  {"name": "Svätý Očisťujúci Oheň", "desc": "Spáli všetky toxické polia a kliatby na ploche troch hexov.", "cost_mana": 4}),

    _make_apostle("apostle_matthias_12", "Matej", "Dvanásty Vyvolený", "Sekera a Zvitok Voľby",
                  {"WS": 3, "BS": 3, "S": 4, "T": 5, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
                  {"toughness": 30, "reflexes": 28, "aim": 30, "dodge": 30, "appearance": 40, "tactics": 44, "mobility": 20},
                  {"name": "Harmonická Pečať Dvanástky", "desc": "Uzatvára kruh 12 apoštolov a zaručuje absolútne kvórum svetla.", "cost_mana": 4})
]

ANGELIC_GUARDIANS: List[Dict[str, Any]] = [
    {
        "id": "archangel_michael",
        "name": "Archanjel Michael",
        "role": "Vodca Nebeských Vojsk (Prince of the Seraphim)",
        "weapon": "Plamenný Meč Spravodlivosti",
        "shield": "Štít Pravdy s nápisom Quis ut Deus",
        "attributes": {"WS": 1, "BS": 1, "S": 7, "T": 7, "W": 6, "A": 5, "Ld": 10, "Sv": "2+"},
        "divine_power": "Pád Lucifera do Tartaru - zničí akúkoľvek démonickú entitu jedným úderom."
    },
    {
        "id": "archangel_gabriel",
        "name": "Archanjel Gabriel",
        "role": "Boží Posol a Zvestovateľ",
        "weapon": "Strieborná Trúba Vzkriesenia a Biela Ľalia",
        "shield": "Aura Pokory",
        "attributes": {"WS": 2, "BS": 1, "S": 6, "T": 6, "W": 6, "A": 4, "Ld": 10, "Sv": "2+"},
        "divine_power": "Trúba Zmŕtvychvstania - obnoví všetkým spojencom plné zdravie a Ward."
    },
    {
        "id": "archangel_raphael",
        "name": "Archanjel Raphael",
        "role": "Božský Lekár a Pútnik",
        "weapon": "Liečivý Smaragdový Prút a Nádoba s Balzamom",
        "shield": "Aura Ochrany na Cestách",
        "attributes": {"WS": 2, "BS": 2, "S": 5, "T": 6, "W": 6, "A": 4, "Ld": 10, "Sv": "2+"},
        "divine_power": "Smaragdový Lúč Uzdravenia - neutralizuje všetky jedy a obnovuje integritu."
    },
    {
        "id": "archangel_uriel",
        "name": "Archanjel Uriel",
        "role": "Plameň Božej Múdrosti",
        "weapon": "Svitok Poznania a Planúca Guľa v Dlani",
        "shield": "Slnečný Pilier Pravdy",
        "attributes": {"WS": 2, "BS": 1, "S": 6, "T": 6, "W": 6, "A": 4, "Ld": 10, "Sv": "2+"},
        "divine_power": "Ohnivý Pilier Múdrosti - spáli temnotu a nočnú oblohu premení na poludňajší jas."
    },
    {
        "id": "seraphim_choir",
        "name": "Chór Serafínov a Cherubínov",
        "role": "Šesťkrídli Strážcovia Božského Trónu",
        "weapon": "Žeravé Uhlie z Nebeského Oltára",
        "shield": "Nekonečný Spev Sanctus",
        "attributes": {"WS": 2, "BS": 2, "S": 5, "T": 6, "W": 6, "A": 6, "Ld": 10, "Sv": "2+"},
        "divine_power": "Nepreniknuteľná Harmónia - ruší akúkoľvek negatívnu entropiu v okruhu 5 hexov."
    }
]

class ApostlesAndAngelsRegistry:
    """
    Registry for the 12 Apostles and Angelic Guardians.
    """
    @classmethod
    def get_apostles(cls) -> List[Dict[str, Any]]:
        return TWELVE_APOSTLES

    @classmethod
    def get_apostle_by_id(cls, apostle_id: str) -> Optional[Dict[str, Any]]:
        for a in TWELVE_APOSTLES:
            if a["id"] == apostle_id:
                return a
        return None

    @classmethod
    def get_angels(cls) -> List[Dict[str, Any]]:
        return ANGELIC_GUARDIANS
