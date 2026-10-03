# ==============================================================================
# KRYSTAL-STACK: PYTHON NATIVE 3D ENGINE CORE (CMS & AST BACKEND)
# ==============================================================================
# High-performance local engine replacing WordPress and Oxygen Builder.
# Bridges Janet 3D CMS AST logic to Godot Engine and WebGL viewports.
# Features:
#   1. Dual-Engine Interop: Native Janet CLI execution with automated
#      high-fidelity embedded Janet-AST fallback if Janet CLI is absent.
#   2. Procedural Godot SceneTree (AST) generation for "Poslední Kmen".
#   3. Direct Wavefront .OBJ asset serving with CORS for WebGL & Godot.
#   4. Poslední Kmen Ledger & Card Combat Engine (6 HP max rule).
#   5. Godot .tscn (Text Scene Format) generation and export.
# ==============================================================================

import os
import sys
import time
import json
import math
import shutil
import subprocess
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import urllib.request
import urllib.error
import mimetypes

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = (sys.platform != "win32")

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, 'godot_assets')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Economic Game Framework Imports
from krystal_web_hub.economic_engine import (
    Tribe, ResourceType, BuildingType, AbilityType, AttackType, RoundPhase,
    EscalationStage, ResourceCost, BuildingSpec, AbilitySpec,
    LedgerEntry, CombatantState, MatchState,
    SectorContestStatus, GarrisonUnit, Sector,
    BUILDING_REGISTRY, EconomicLedger,
    ABILITY_REGISTRY, AbilityEngine,
    RoundController, SCENARIO_TEMPLATES, get_all_templates,
    generate_godot_tscn, generate_default_sectors, SectorConquestEngine,
    hex_riemannian_distance, hex_to_world_cartesian,
    calculate_ballistic_apex_height, sample_ballistic_bezier_hermite_curve,
    hex_prism_sdf, crystal_spire_sdf, polynomial_smooth_min,
    update_hex_biome_transition, fuse_cards, compact_godot_ast,
    TacticalAIEngine, generate_tribal_deck, PlayerDeckManager,
    ItemRarity, ItemSlot, ItemAffix, ItemTemplate, CraftedItem,
    BASE_TEMPLATES, AFFIX_REGISTRY, RARITY_AFFIX_LIMITS, CraftingEngine,
    CraftingError, AffixCapExceededError, CraftingInstabilityError, PowerBudgetExceededError,
    HexCoord3D, ElevationCombatModifiers, calculate_elevation_advantage,
    TowerWard, TowerLocationalCombatResolver,
    CheckpointFlag, FlagControlState, CheckpointManager,
    MovementType, MobilityClass, TerrainType, UnitMobilityProfile, WarhammerMobilityEngine,
    get_to_hit_threshold, get_to_wound_threshold, get_armor_save_threshold,
    calculate_expected_damage, probability_2d6_ge, CombatSimulationModel,
    get_full_tribal_card_catalog, get_all_unit_archetypes,
    CANONICAL_HERO_PROFILES, CANONICAL_MINION_PROFILES,
    get_race_and_specializations_catalog, get_map_legend_data,
    EnvironmentalHazardType, EnvironmentalCell, EnvironmentalMatrix,
    CombatSector, TacticalTrigonometry, MatrixTacticalAI,
    CardActionType, HypergeometricCardStatistics, CardCombatAlgebra,
    WeaponType, EnchantmentType, BallisticParameters,
    TrigonometricTargetingSystem, ActionCombinationEngine,
    EnchantmentBonusCalculator, ArtilleryCombatCalculator,
    DopamineCombatState, DopamineCadenceEngine,
    UncoveredTargetCalculator, MortarDispersionEngine,
    AspectRatio, DuelWindowMode, GnomeDuelWindowCompositor,
    ActionType, ParallelActionTrack, UbisoftBulletTimeCompositor,
    EventThreatLevel, CommunityBuildRegistry, EventRewardDerivationEngine,
    WordPressMcpBridge,
    TunnelStreamCipher, VpnTunnelPenaltyEvaluator,
    PenaltyResolutionType, DeathPenaltyEvaluator, RewardedAdQuotaManager,
    ChestTier, ArtifactRarity, CHEST_CATALOG, ARTIFACT_CATALOG,
    ChestLootResolver, AuctionLotStatus, AuctionHouseEngine,
    TacticalGradientEngine, PinnedOrbitalSpellcraftingEngine,
    DuelTargetZone, DuelDodgeStance, DuelWeaponCategory, TheWestDuelAlgebra,
    CRYSTAL_ROSTER, TOXIC_ROSTER, DRUID_ROSTER, ALL_SIXTY_CHARACTERS, SixtyCharacterRosterEngine,
    AdFormat, AdCampaignStatus, AdCampaign, SecureAdExchangeProtocol,
    ClusterNodeRole, NodeHealthStatus, FirewallThreatCategory,
    ClusterNode, MonitoringClusterEngine, AdaptiveApplicationFirewall,
    CosmologicalPlane, COSMOLOGICAL_PLANE_DATA, BotTransmitter,
    NocturnalAtmosphereEngine, EntropyWeatherEngine, ArborMycorrhizalNetwork,
    DimensionalPortalsAndMirrors, TWELVE_APOSTLES, ANGELIC_GUARDIANS, ApostlesAndAngelsRegistry,
    RACES_CATALOG, MALE_ARCHETYPES_20, FEMALE_ARCHETYPES_20, ALL_40_REPRESENTATIVES,
    HELPERS_120_CATALOG, HELPERS_BY_ID, HELPERS_BY_INDEX, ArchetypeAndHelperEngine,
    DamageAilmentType, ImmunityStatus, ZODIAC_CONSTELLATIONS, GRID_PRESET_SPECS,
    SacredNumerologyEngine, ImmunitySystemEngine, ZodiacSkyEngine,
    EquipmentZoomOpticsEngine, PlusInventoryEngine, PrerequisitesValidator,
    PaintingAuctionHouseEngine, AetherOrdinalsProtocolEngine,
    ContentReplayEngine, HeroMatrixEngine, FrameRateEncodingProtocol,
    EvolutionaryPhysicsEngine, EventStreamCongestionController,
    GLOBAL_PAINTING_AUCTIONS, GLOBAL_ORDINALS_PROTOCOL,
    MetaverseAssetType, OrderType, MetaverseMarketplaceEngine,
    GraniteAndEdgeLLMEngine, GLOBAL_METAVERSE_MARKET, GLOBAL_GRANITE_LLM,
    VisualPhenomenonType, SpellProjectionType, SectorAnomalyType,
    TotemStatus, SectorTotem, VisualPhenomenaEngine,
    SpellProjectionEngine, AnomalyDetectorSensorArray, SectorTotemManager,
    GLOBAL_VISUAL_PHENOMENA, GLOBAL_SPELL_PROJECTION,
    GLOBAL_ANOMALY_DETECTOR, GLOBAL_TOTEM_MANAGER,
    CANONICAL_MULLIGAN_CARDS, MulliganPhaseManager, CombinatorialTacticalMoveEngine,
    GLOBAL_MULLIGAN_MANAGER, GLOBAL_COMBINATORIAL_ENGINE
)
from krystal_web_hub.economic_engine.ability_framework import (
    calculate_hex_distance, validate_target_range
)
GLOBAL_MCP_BRIDGE = WordPressMcpBridge()
GLOBAL_AD_QUOTA_MGR = RewardedAdQuotaManager()
GLOBAL_AUCTION_HOUSE = AuctionHouseEngine()
ACTIVE_ENV_MATRIX = EnvironmentalMatrix(radius=2)
GLOBAL_AD_EXCHANGE = SecureAdExchangeProtocol()
GLOBAL_MONITORING_CLUSTER = MonitoringClusterEngine()
GLOBAL_FIREWALL = AdaptiveApplicationFirewall(requests_per_minute=300, ban_duration_sec=600.0)
GLOBAL_BOT_TRANSMITTER = BotTransmitter("bot_prime_explorer", channel_freq_mhz=433.92)
GLOBAL_PLUS_INVENTORY = PlusInventoryEngine(preset="6x5", plus_tokens=2)
GLOBAL_EVO_PHYSICS = EvolutionaryPhysicsEngine(population_size=16)

def world_pos_to_hex(coords):
    if not coords or not isinstance(coords, (list, tuple)):
        return [0, 0]
    if len(coords) == 2:
        return [int(coords[0]), int(coords[1])]
    x = float(coords[0])
    z = float(coords[2]) if len(coords) >= 3 else float(coords[1])
    q = round(x / 1.732)
    r = round(z / 1.5)
    return [q, r]

# Active Economic Match State
ACTIVE_ECONOMIC_MATCH = RoundController.initialize_match("match_alpha_01", Tribe.CRYSTAL, Tribe.TOXIC)
PLAYER_DECK_MGR = PlayerDeckManager(Tribe.CRYSTAL)
ENEMY_DECK_MGR = PlayerDeckManager(Tribe.TOXIC)
PLAYER_DECK_MGR.draw_to_full()
ENEMY_DECK_MGR.draw_to_full()

# Strategic Checkpoint / Flag System
ACTIVE_CHECKPOINT_MGR = CheckpointManager(home_base_player=[0, -2], home_base_enemy=[0, 2])
ACTIVE_CHECKPOINT_MGR.register_flag(CheckpointFlag(
    id="flag_nexus",
    name="Aéterový Nexus (Stred)",
    hex_coords=[0, 0],
    controlling_side="neutral",
    control_percentage=50,
    victory_points_per_turn=2,
    color="#ffd700"
))
ACTIVE_CHECKPOINT_MGR.register_flag(CheckpointFlag(
    id="flag_north",
    name="Severná Svätyňa",
    hex_coords=[0, -1],
    controlling_side="player",
    control_percentage=100,
    victory_points_per_turn=1,
    color="#00ffff"
))
ACTIVE_CHECKPOINT_MGR.register_flag(CheckpointFlag(
    id="flag_south",
    name="Toxická Bašta",
    hex_coords=[0, 1],
    controlling_side="enemy",
    control_percentage=100,
    victory_points_per_turn=1,
    color="#39ff14"
))

# Defensive Tower with Locational Algebra & Ward Aura (Warda)
ACTIVE_TOWER_WARD = TowerWard(
    tower_id="crystal_sanctum_tower",
    name="Obranná Kryštálová Veža (Warda)",
    owner_tribe=Tribe.CRYSTAL,
    position=HexCoord3D(q=0, r=-2, h=2.5),
    ward_radius=1,
    ward_max_pool=6,
    ward_current_pool=6,
    regen_per_turn=1
)

# Warhammer Mobility Engine & Active Units
ACTIVE_MOBILITY_ENGINE = WarhammerMobilityEngine(terrain_map={
    (0, 1): TerrainType.DIFFICULT,
    (-1, 0): TerrainType.DIFFICULT,
    (1, 0): TerrainType.OPEN
})
PLAYER_MOBILITY_UNIT = UnitMobilityProfile(
    unit_id="player_archon",
    name="Kryštálový Archón",
    tribe=Tribe.CRYSTAL,
    move_stat=2,
    current_hex=[0, -2]
)
ENEMY_MOBILITY_UNIT = UnitMobilityProfile(
    unit_id="enemy_abomination",
    name="Toxický Abomination",
    tribe=Tribe.TOXIC,
    move_stat=2,
    current_hex=[0, 2]
)

# ------------------------------------------------------------------------------
# 1. HARDCODED CARDS & MESH SPECIFICATIONS (Poslední Kmen)
# ------------------------------------------------------------------------------
KMEN_CARDS = {
    "crystal_meteor": {
        "id": "crystal_meteor",
        "name": "Kryštálový Meteor",
        "tribe": "crystal",
        "tribe_name": "Kryštálový Kmeň",
        "cost": 3,
        "hp_delta": -2,
        "armor_delta": 0,
        "status_effect": "none",
        "attack_type": "ranged",
        "min_range": 1,
        "max_range": 4,
        "trajectory_type": "arc",
        "animation_fx": "animated_arrow",
        "description": "Zasiahne cieľ kryštálovým meteorom a spôsobí 2 body priameho zranenia (Dosah: 1-4 hexov).",
        "mesh_asset": "crystal_shard.obj",
        "color": "#00ffff",
        "mesh_node": {
            "type": "Spatial",
            "name": "Spell_CrystalMeteor",
            "properties": {"position": [0, 0, 0]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "CoreShard",
                    "properties": {
                        "mesh": "res://godot_assets/crystal_shard.obj",
                        "scale": [1.6, 1.6, 1.6],
                        "material": "CyanGlowingShader",
                        "color": "#00ffff"
                    },
                    "children": [
                        {"type": "AnimationPlayer", "name": "DropAnim", "properties": {"anim": "meteor_strike", "duration": 1.2}}
                    ]
                },
                {"type": "Particles", "name": "ManaDust", "properties": {"amount": 180, "color": "#66fcf1", "velocity": [0, 8, 0]}},
                {"type": "OmniLight", "name": "ImpactGlow", "properties": {"color": "#00ffff", "energy": 2.8, "range": 8.0}}
            ]
        }
    },
    "crystal_shield": {
        "id": "crystal_shield",
        "name": "Kryštálový Štít",
        "tribe": "crystal",
        "tribe_name": "Kryštálový Kmeň",
        "cost": 2,
        "hp_delta": 0,
        "armor_delta": 2,
        "status_effect": "shielded",
        "attack_type": "self",
        "min_range": 0,
        "max_range": 0,
        "trajectory_type": "instant",
        "animation_fx": "shield_pulse",
        "description": "Vytvorí kryštálovú bariéru pridávajúcu +2 brnenia.",
        "mesh_asset": "crystal_shield.obj",
        "color": "#66fcf1",
        "mesh_node": {
            "type": "Spatial",
            "name": "Spell_CrystalShield",
            "properties": {"position": [0, 0, 0]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "HexBarrier",
                    "properties": {
                        "mesh": "res://godot_assets/crystal_shield.obj",
                        "scale": [1.2, 1.2, 1.2],
                        "material": "GlassCrystalline",
                        "color": "#66fcf1",
                        "opacity": 0.85
                    }
                },
                {"type": "OmniLight", "name": "ShieldAura", "properties": {"color": "#00ffff", "energy": 1.6, "range": 5.0}}
            ]
        }
    },
    "crystal_pylon": {
        "id": "crystal_pylon",
        "name": "Rezonančný Pylón",
        "tribe": "crystal",
        "tribe_name": "Kryštálový Kmeň",
        "cost": 4,
        "hp_delta": 0,
        "armor_delta": 0,
        "status_effect": "mana_regen",
        "attack_type": "self",
        "min_range": 0,
        "max_range": 2,
        "trajectory_type": "ground",
        "animation_fx": "pylon_spawn",
        "description": "Umiestni rezonančný pylón na pole, ktorý generuje +1 manu za ťah.",
        "mesh_asset": "crystal_shard.obj",
        "color": "#e0ffff",
        "mesh_node": {
            "type": "Spatial",
            "name": "Building_CrystalPylon",
            "properties": {"position": [0, 0, 0]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "PylonSpire",
                    "properties": {
                        "mesh": "res://godot_assets/crystal_shard.obj",
                        "scale": [2.2, 2.8, 2.2],
                        "material": "PureAether",
                        "color": "#e0ffff"
                    }
                },
                {"type": "Particles", "name": "ResonanceRings", "properties": {"amount": 80, "color": "#00ffff", "spread": 3.0}}
            ]
        }
    },
    "acid_slime": {
        "id": "acid_slime",
        "name": "Kyslý Sliz",
        "tribe": "toxic",
        "tribe_name": "Jedovatý Kmeň",
        "cost": 1,
        "hp_delta": 0,
        "armor_delta": 0,
        "status_effect": "rooted",
        "attack_type": "ranged",
        "min_range": 1,
        "max_range": 3,
        "trajectory_type": "arc",
        "animation_fx": "slime_arrow",
        "description": "Znehybní cieľovú jednotku na 1 ťah v bublajúcom kyslom slize (Dosah: 1-3 hexov).",
        "mesh_asset": "acid_slime.obj",
        "color": "#39ff14",
        "mesh_node": {
            "type": "Spatial",
            "name": "Spell_AcidSlime",
            "properties": {"position": [0, 0, 0]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "SlimePool",
                    "properties": {
                        "mesh": "res://godot_assets/acid_slime.obj",
                        "scale": [1.8, 1.0, 1.8],
                        "material": "ViscousAcidGreen",
                        "color": "#39ff14"
                    },
                    "children": [
                        {"type": "AnimationPlayer", "name": "BubbleAnim", "properties": {"anim": "bubbling", "speed": 1.5}}
                    ]
                },
                {"type": "Particles", "name": "AcidVapor", "properties": {"amount": 120, "color": "#39ff14", "spread": 2.5}}
            ]
        }
    },
    "toxic_cloud": {
        "id": "toxic_cloud",
        "name": "Toxický Oblak",
        "tribe": "toxic",
        "tribe_name": "Jedovatý Kmeň",
        "cost": 4,
        "hp_delta": -1,
        "armor_delta": 0,
        "status_effect": "poison_aoe",
        "attack_type": "aoe",
        "min_range": 1,
        "max_range": 3,
        "trajectory_type": "arc",
        "animation_fx": "cloud_spread",
        "description": "Spore totem uvoľní jedovatý mrak; 1 poškodenie každé kolo pre všetkých v dosahu.",
        "mesh_asset": "toxic_totem.obj",
        "color": "#8a2be2",
        "mesh_node": {
            "type": "Spatial",
            "name": "Building_ToxicTotem",
            "properties": {"position": [0, 0, 0]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "SporeColumn",
                    "properties": {
                        "mesh": "res://godot_assets/toxic_totem.obj",
                        "scale": [1.4, 1.8, 1.4],
                        "material": "OrganicChitin",
                        "color": "#7fff00"
                    }
                },
                {"type": "Particles", "name": "PoisonMist", "properties": {"amount": 240, "color": "#8a2be2", "radius": 4.5}},
                {"type": "Area", "name": "DamageZone", "properties": {"radius": 4.0}}
            ]
        }
    },
    "earth_roots": {
        "id": "earth_roots",
        "name": "Korene Zeme",
        "tribe": "druid",
        "tribe_name": "Druidi",
        "cost": 2,
        "hp_delta": -1,
        "armor_delta": 0,
        "status_effect": "stunned",
        "attack_type": "ranged",
        "min_range": 1,
        "max_range": 3,
        "trajectory_type": "ground",
        "animation_fx": "roots_crawl",
        "description": "Z koreňov stromov vyrastú výhonky, ktoré omráčia nepriateľa na 1 kolo a udelia 1 poškodenie.",
        "mesh_asset": "earth_roots.obj",
        "color": "#8b4513",
        "mesh_node": {
            "type": "Spatial",
            "name": "Spell_EarthRoots",
            "properties": {"position": [0, 0, 0]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "GnarledRoots",
                    "properties": {
                        "mesh": "res://godot_assets/earth_roots.obj",
                        "scale": [1.5, 1.5, 1.5],
                        "material": "AncientBark",
                        "color": "#8b4513"
                    }
                },
                {"type": "Particles", "name": "FloatingLeaves", "properties": {"amount": 60, "color": "#2e8b57"}}
            ]
        }
    },
    "nature_bless": {
        "id": "nature_bless",
        "name": "Požehnanie Prírody",
        "tribe": "druid",
        "tribe_name": "Druidi",
        "cost": 3,
        "hp_delta": 2,
        "armor_delta": 0,
        "status_effect": "healed",
        "attack_type": "self",
        "min_range": 0,
        "max_range": 0,
        "trajectory_type": "instant",
        "animation_fx": "nature_shield",
        "description": "Posvätný monolit obnoví 2 životy (až do maxima 6 HP).",
        "mesh_asset": "druid_monolith.obj",
        "color": "#ffd700",
        "mesh_node": {
            "type": "Spatial",
            "name": "Building_DruidMonolith",
            "properties": {"position": [0, 0, 0]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "RuneStone",
                    "properties": {
                        "mesh": "res://godot_assets/druid_monolith.obj",
                        "scale": [1.3, 1.6, 1.3],
                        "material": "MossyGranite",
                        "color": "#556b2f"
                    }
                },
                {"type": "OmniLight", "name": "SanctuaryLight", "properties": {"color": "#ffd700", "energy": 2.2, "range": 6.0}},
                {"type": "Particles", "name": "LifeSparks", "properties": {"amount": 100, "color": "#ffd700"}}
            ]
        }
    },
    "druid_strike": {
        "id": "druid_strike",
        "name": "Úder Druidskej Palice",
        "tribe": "druid",
        "tribe_name": "Druidi",
        "cost": 1,
        "hp_delta": -1,
        "armor_delta": 0,
        "status_effect": "none",
        "attack_type": "melee",
        "min_range": 1,
        "max_range": 1,
        "trajectory_type": "linear",
        "animation_fx": "melee_slash",
        "description": "Útok na blízko (dosah 1 hex): Rýchly úder okovanou palicou spôsobí 1 poškodenie susednej jednotke.",
        "mesh_asset": "earth_roots.obj",
        "color": "#ffd700",
        "mesh_node": {
            "type": "Spatial",
            "name": "Spell_DruidStrike",
            "properties": {"position": [0, 0, 0]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "StrikeSlash",
                    "properties": {
                        "mesh": "res://godot_assets/earth_roots.obj",
                        "scale": [1.2, 1.2, 1.2],
                        "material": "AmberRuneEnergy",
                        "color": "#ffd700"
                    }
                },
                {"type": "OmniLight", "name": "SlashGlow", "properties": {"color": "#ffd700", "energy": 2.5, "range": 4.0}}
            ]
        }
    },
    "decay_strike": {
        "id": "decay_strike",
        "name": "Zuby Rozkladu",
        "tribe": "toxic",
        "tribe_name": "Jedovatý Kmeň",
        "cost": 2,
        "hp_delta": -2,
        "armor_delta": 0,
        "status_effect": "poisoned",
        "attack_type": "melee",
        "min_range": 1,
        "max_range": 1,
        "trajectory_type": "linear",
        "animation_fx": "decay_claw",
        "description": "Útok na blízko (dosah 1 hex): Leptavý útok na susednú jednotku spôsobí 2 poškodenia s nákazou.",
        "mesh_asset": "acid_slime.obj",
        "color": "#7fff00",
        "mesh_node": {
            "type": "Spatial",
            "name": "Spell_DecayStrike",
            "properties": {"position": [0, 0, 0]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "DecayBite",
                    "properties": {
                        "mesh": "res://godot_assets/acid_slime.obj",
                        "scale": [1.4, 1.4, 1.4],
                        "material": "AcidicVenom",
                        "color": "#7fff00"
                    }
                },
                {"type": "Particles", "name": "VenomSplatter", "properties": {"amount": 80, "color": "#7fff00"}}
            ]
        }
    }
}

# ------------------------------------------------------------------------------
# 2. GAME STATE & LEDGER (Poslední Kmen Rules: 6 Max HP, Mana Ledger)
# ------------------------------------------------------------------------------
GAME_STATE = {
    "player_hp": 6,
    "player_max_hp": 6,
    "player_armor": 0,
    "player_mana": 10,
    "enemy_hp": 6,
    "enemy_max_hp": 6,
    "enemy_armor": 0,
    "turn": 1,
    "ledger": [
        {"turn": 0, "event": "INIT", "desc": "Bojová aréna Poslední Kmen inicializovaná. Každý bojovník má 6 HP."}
    ]
}

# ------------------------------------------------------------------------------
# 3. PROCEDURAL SCENE GENERATION (EMBEDDED JANET AST ENGINE)
# ------------------------------------------------------------------------------
def generate_tactical_hex_grid(radius=2):
    """Generates a 19-hex tactical board AST with sector and tribal affiliations."""
    tiles = []
    hex_w = 1.732
    hex_h = 1.5

    # Strategic sector partitioning for (q, r)
    sector_coord_map = {
        (0, -2): ("sector_north_crystal", "Severné Ľadovce", "#00ffff", "crystal"),
        (1, -2): ("sector_north_crystal", "Severné Ľadovce", "#00ffff", "crystal"),
        (0, -1): ("sector_north_crystal", "Severné Ľadovce", "#00ffff", "crystal"),
        (1, -1): ("sector_north_crystal", "Severné Ľadovce", "#00ffff", "crystal"),

        (-1, 1): ("sector_south_toxic", "Toxická Kotlina", "#39ff14", "toxic"),
        (0, 1): ("sector_south_toxic", "Toxická Kotlina", "#39ff14", "toxic"),
        (-1, 2): ("sector_south_toxic", "Toxická Kotlina", "#39ff14", "toxic"),
        (0, 2): ("sector_south_toxic", "Toxická Kotlina", "#39ff14", "toxic"),

        (1, 0): ("sector_east_druid", "Svätyňa Dubov", "#2e8b57", "druid"),
        (2, -1): ("sector_east_druid", "Svätyňa Dubov", "#2e8b57", "druid"),
        (2, 0): ("sector_east_druid", "Svätyňa Dubov", "#2e8b57", "druid"),

        (-2, 0): ("sector_west_sulfur", "Sírny Priesmyk", "#9d4edd", "toxic"),
        (-2, 1): ("sector_west_sulfur", "Sírny Priesmyk", "#9d4edd", "toxic"),
        (-1, 0): ("sector_west_sulfur", "Sírny Priesmyk", "#9d4edd", "toxic"),

        (0, 0): ("sector_center_citadel", "Aéterová Citadela", "#ffd700", "neutral"),
    }

    for q in range(-radius, radius + 1):
        r1 = max(-radius, -q - radius)
        r2 = min(radius, -q + radius)
        for r in range(r1, r2 + 1):
            x = round(hex_w * (q + r * 0.5), 2)
            z = round(hex_h * r, 2)
            sec_info = sector_coord_map.get((q, r))
            if sec_info:
                sec_id, sec_name, col, t_type = sec_info
                mat = f"TileSector_{sec_id}"
            else:
                sec_id, sec_name, col, t_type = "sector_neutral_periphery", "Neutrálny Okraj", "#555555", "neutral"
                mat = "TileNeutralGrey"
                
            tile_node = {
                "type": "MeshInstance",
                "name": f"GridTile_{q}_{r}",
                "properties": {
                    "position": [x, 0.0, z],
                    "hex_coords": [q, r],
                    "sector_id": sec_id,
                    "sector_name": sec_name,
                    "mesh": "res://godot_assets/hex_tile.obj",
                    "tile_type": t_type,
                    "material": mat,
                    "color": col
                },
                "children": []
            }
            tiles.append(tile_node)
            
    return {
        "type": "Spatial",
        "name": "TacticalHexGrid",
        "properties": {"tile_count": len(tiles)},
        "children": tiles
    }

def generate_scene_ast_internal(prompt: str) -> dict:
    """Procedurally compiles a complete Godot 3D SceneTree AST from semantic prompt."""
    p_lower = prompt.lower()
    
    root_node = {
        "type": "Node",
        "name": "WorldRoot",
        "properties": {},
        "children": [
            {
                "type": "DirectionalLight",
                "name": "GlobalSun",
                "properties": {"color": "#ffffff", "energy": 0.9, "rotation": [-45, 30, 0]},
                "children": []
            },
            {
                "type": "Camera",
                "name": "TacticalCamera",
                "properties": {"position": [0, 14, 16], "rotation": [-40, 0, 0], "fov": 65},
                "children": []
            },
            generate_tactical_hex_grid(radius=2)
        ]
    }
    
    # 1. Crystal Sanctum
    if any(k in p_lower for k in ["kryst", "crystal", "sever", "ľad", "štít", "meteor"]):
        root_node["children"].append({
            "type": "Spatial",
            "name": "CrystalSanctum",
            "properties": {"position": [0, 0, -5]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "CoreCrystal",
                    "properties": {
                        "mesh": "res://godot_assets/crystal_shard.obj",
                        "scale": [2.5, 3.5, 2.5],
                        "material": "GlowingCyanShader",
                        "color": "#00ffff"
                    },
                    "children": []
                },
                {
                    "type": "MeshInstance",
                    "name": "OrbitalShield",
                    "properties": {
                        "mesh": "res://godot_assets/crystal_shield.obj",
                        "position": [1.5, 1.2, 0],
                        "scale": [1.0, 1.0, 1.0],
                        "color": "#66fcf1"
                    },
                    "children": []
                },
                {"type": "OmniLight", "name": "SanctumBeacon", "properties": {"color": "#66fcf1", "energy": 3.0, "range": 10.0}, "children": []},
                {"type": "Particles", "name": "AetherSwarm", "properties": {"amount": 120, "color": "#00ffff"}, "children": []},
                {"type": "Area", "name": "SanctumTerritory", "properties": {"radius": 6.0}, "children": []}
            ]
        })
        
    # 2. Toxic Wasteland
    if any(k in p_lower for k in ["tox", "jed", "pustin", "sliz", "totem", "oblak"]):
        root_node["children"].append({
            "type": "Spatial",
            "name": "ToxicWasteland",
            "properties": {"position": [-6, 0, 2]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "SlimeMother",
                    "properties": {
                        "mesh": "res://godot_assets/acid_slime.obj",
                        "scale": [3.0, 1.2, 3.0],
                        "material": "ViscousGreenSlime",
                        "color": "#39ff14"
                    },
                    "children": []
                },
                {
                    "type": "MeshInstance",
                    "name": "TotemSpire",
                    "properties": {
                        "mesh": "res://godot_assets/toxic_totem.obj",
                        "position": [0, 0.5, 0],
                        "scale": [1.6, 2.2, 1.6],
                        "color": "#7fff00"
                    },
                    "children": []
                },
                {"type": "OmniLight", "name": "SporeGaze", "properties": {"color": "#39ff14", "energy": 2.2, "range": 8.0}, "children": []},
                {"type": "Particles", "name": "PoisonCloud", "properties": {"amount": 250, "color": "#8a2be2"}, "children": []}
            ]
        })
        
    # 3. Druid Grove
    if any(k in p_lower for k in ["druid", "les", "koren", "nature", "strom", "bless"]):
        root_node["children"].append({
            "type": "Spatial",
            "name": "DruidGrove",
            "properties": {"position": [6, 0, 2]},
            "children": [
                {
                    "type": "MeshInstance",
                    "name": "WorldRoots",
                    "properties": {
                        "mesh": "res://godot_assets/earth_roots.obj",
                        "scale": [2.0, 2.2, 2.0],
                        "material": "LivingBark",
                        "color": "#8b4513"
                    },
                    "children": []
                },
                {
                    "type": "MeshInstance",
                    "name": "SunAltar",
                    "properties": {
                        "mesh": "res://godot_assets/druid_monolith.obj",
                        "position": [2.0, 0, 1.5],
                        "scale": [1.4, 1.8, 1.4],
                        "color": "#556b2f"
                    },
                    "children": []
                },
                {"type": "OmniLight", "name": "SolarBlessing", "properties": {"color": "#ffd700", "energy": 2.5, "range": 9.0}, "children": []},
                {"type": "Particles", "name": "GoldenSpores", "properties": {"amount": 80, "color": "#ffd700"}, "children": []}
            ]
        })
        
    # Default: if no specific tribe mentioned, assemble full arena with all 3 tribal sanctums
    if len(root_node["children"]) == 3:
        # Add all three
        root_node["children"].append({
            "type": "Spatial",
            "name": "CrystalSanctum",
            "properties": {"position": [0, 0, -5]},
            "children": [
                {"type": "MeshInstance", "name": "CoreCrystal", "properties": {"mesh": "res://godot_assets/crystal_shard.obj", "scale": [2.5, 3.5, 2.5], "color": "#00ffff"}, "children": []},
                {"type": "OmniLight", "name": "SanctumBeacon", "properties": {"color": "#66fcf1", "energy": 3.0, "range": 10.0}, "children": []}
            ]
        })
        root_node["children"].append({
            "type": "Spatial",
            "name": "ToxicWasteland",
            "properties": {"position": [-6, 0, 2]},
            "children": [
                {"type": "MeshInstance", "name": "SlimeMother", "properties": {"mesh": "res://godot_assets/acid_slime.obj", "scale": [2.5, 1.0, 2.5], "color": "#39ff14"}, "children": []},
                {"type": "MeshInstance", "name": "TotemSpire", "properties": {"mesh": "res://godot_assets/toxic_totem.obj", "position": [0, 0.4, 0], "scale": [1.4, 2.0, 1.4], "color": "#7fff00"}, "children": []}
            ]
        })
        root_node["children"].append({
            "type": "Spatial",
            "name": "DruidGrove",
            "properties": {"position": [6, 0, 2]},
            "children": [
                {"type": "MeshInstance", "name": "WorldRoots", "properties": {"mesh": "res://godot_assets/earth_roots.obj", "scale": [1.8, 2.0, 1.8], "color": "#8b4513"}, "children": []},
                {"type": "MeshInstance", "name": "SunAltar", "properties": {"mesh": "res://godot_assets/druid_monolith.obj", "position": [1.8, 0, 1.2], "scale": [1.2, 1.6, 1.2], "color": "#556b2f"}, "children": []}
            ]
        })

    return {
        "id": "generated_arena",
        "title": f"Poslední Kmen: {prompt.strip() or 'Taktická Aréna'}",
        "tree": root_node
    }

def convert_ast_to_tscn(node: dict, parent_path: str = ".") -> list:
    """Converts a Godot JSON AST node into Godot 4 .tscn text format lines."""
    lines = []
    node_name = node.get("name", "Node")
    node_type = node.get("type", "Node")
    
    if parent_path == ".":
        lines.append(f'[node name="{node_name}" type="{node_type}"]')
        current_path = node_name
    else:
        lines.append(f'[node name="{node_name}" type="{node_type}" parent="{parent_path}"]')
        current_path = f"{parent_path}/{node_name}"
        
    props = node.get("properties", {})
    if "position" in props:
        pos = props["position"]
        lines.append(f'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, {pos[0]}, {pos[1]}, {pos[2]})')
    if "scale" in props:
        sc = props["scale"]
        lines.append(f'scale = Vector3({sc[0]}, {sc[1]}, {sc[2]})')
    if "color" in props:
        lines.append(f'modulate = Color("{props["color"]}")')
        
    lines.append("")
    for child in node.get("children", []):
        lines.extend(convert_ast_to_tscn(child, current_path))
        
    return lines

def export_scene_to_tscn(scene_data: dict) -> str:
    header = [
        '[gd_scene format=3 uid="uid://krystal_posledni_kmen"]',
        '',
        '# Generated by Krystal-Stack Janet 3D CMS // Godot Engine Bridge',
        ''
    ]
    body = convert_ast_to_tscn(scene_data.get("tree", {}), ".")
    return "\n".join(header + body)

# ------------------------------------------------------------------------------
# 4. JANET INTEROP (CLI + DUAL FALLBACK)
# ------------------------------------------------------------------------------
def execute_janet_cms(prompt: str) -> dict:
    """Tries native Janet CLI if available; seamlessly falls back to embedded AST compiler."""
    janet_bin = shutil.which("janet")
    if janet_bin:
        try:
            escaped_prompt = prompt.replace('"', '\\"')
            janet_script = f"""
            (dofile "krystal_web_hub/krystal_3d_cms.janet")
            (bot-generate-scene "{escaped_prompt}")
            (print (export-scene-to-json :generated_arena))
            """
            temp_path = os.path.join(BASE_DIR, "_temp_run.janet")
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(janet_script)
                
            res = subprocess.run([janet_bin, temp_path], capture_output=True, text=True, check=True)
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            lines = res.stdout.strip().split('\n')
            last_line = lines[-1].strip()
            data = json.loads(last_line)
            data["tscn"] = export_scene_to_tscn(data)
            return data
        except Exception as e:
            print(f"[Engine Interop] Native Janet failed ({e}), switching to Embedded Janet-AST Engine...")

    # Autonomous Embedded Janet-AST Engine
    data = generate_scene_ast_internal(prompt)
    data["tscn"] = export_scene_to_tscn(data)
    data["engine_mode"] = "Embedded Janet-AST Core (Zero-dependency)"
    return data

# ------------------------------------------------------------------------------
# 5. HTTP API SERVER (PORT 8089)
# ------------------------------------------------------------------------------
class KrystalEngineHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text, content_type='text/plain', status=200):
        body = text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, filepath, content_type=None):
        if not os.path.exists(filepath) or not os.path.isfile(filepath):
            self._send_json({"error": f"File {os.path.basename(filepath)} not found"}, status=404)
            return
        if not content_type:
            ext = os.path.splitext(filepath)[1].lower()
            mime_map = {
                ".html": "text/html",
                ".htm": "text/html",
                ".css": "text/css",
                ".js": "application/javascript",
                ".mjs": "application/javascript",
                ".json": "application/json",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".svg": "image/svg+xml",
                ".ico": "image/x-icon",
                ".webp": "image/webp",
                ".obj": "text/plain",
                ".tscn": "text/plain",
                ".wasm": "application/wasm",
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".ogg": "audio/ogg",
                ".ttf": "font/ttf",
                ".woff": "font/woff",
                ".woff2": "font/woff2",
            }
            content_type = mime_map.get(ext, mimetypes.guess_type(str(filepath))[0] or "application/octet-stream")
        with open(filepath, "rb") as f:
            content = f.read()
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content)

    def _proxy_to_hub(self, method="GET", body=None) -> bool:
        target_url = f"http://127.0.0.1:8080{self.path}"
        req = urllib.request.Request(target_url, data=body, method=method)
        for h, v in self.headers.items():
            if h.lower() not in ("host", "content-length"):
                req.add_header(h, v)
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for h, v in resp.headers.items():
                    if h.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(h, v)
                self.send_header("Content-Length", str(len(resp_body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(resp_body)
                return True
        except urllib.error.HTTPError as e:
            err_body = e.read()
            self.send_response(e.code)
            for h, v in e.headers.items():
                if h.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(h, v)
            self.send_header("Content-Length", str(len(err_body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(err_body)
            return True
        except Exception:
            return False

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        # ── Firewall Layer 7 Deep Packet Inspection ─────────────────
        client_ip = self.client_address[0] if hasattr(self, 'client_address') and self.client_address else "127.0.0.1"
        headers_dict = {k: v for k, v in self.headers.items()}
        allowed, threat = GLOBAL_FIREWALL.inspect_request(client_ip, self.path, headers_dict, body_text="")
        if not allowed:
            self._send_json({"error": "FORBIDDEN_BY_FIREWALL", "incident": threat}, status=403)
            return

        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # ── Static / UI Routing on Port 8089 ───────────────────────
        if path in ('/', '/index.html'):
            self._serve_file(os.path.join(STATIC_DIR, "index.html"), "text/html")
            return
        if path in ('/game', '/arena', '/posledni-kmen', '/godot_builder_extension.html', '/builder', '/studio-3d'):
            self._serve_file(os.path.join(STATIC_DIR, "godot_builder_extension.html"), "text/html")
            return
        if path in ('/manual', '/design-manual', '/game_elements_design_manual.html'):
            self._serve_file(os.path.join(STATIC_DIR, "game_elements_design_manual.html"), "text/html")
            return
        if path.startswith('/static/'):
            rel_path = path[8:].lstrip('/\\')
            target = os.path.normpath(os.path.join(STATIC_DIR, rel_path))
            if not target.startswith(os.path.normpath(STATIC_DIR)):
                self._send_json({"error": "Access denied"}, status=403)
                return
            self._serve_file(target)
            return

        # Health / Status
        if path == '/api/status':
            self._send_json({
                "status": "ONLINE",
                "engine": "Krystal-Stack 3D CMS (Janet + Python)",
                "port": 8089,
                "native_janet": shutil.which("janet") is not None,
                "cards_count": len(KMEN_CARDS),
                "assets": os.listdir(ASSET_DIR) if os.path.exists(ASSET_DIR) else []
            })
            return

        # Cards Library
        if path == '/api/cards':
            self._send_json({
                "game": "Poslední Kmen",
                "rules": {
                    "max_hp": 6,
                    "mana_accounting": "Ledger-backed",
                    "tribes": ["Kryštálový Kmeň", "Jedovatý Kmeň", "Druidi"]
                },
                "cards": list(KMEN_CARDS.values())
            })
            return

        # Game State & Ledger
        if path == '/api/state':
            self._send_json(GAME_STATE)
            return

        # Player Hand & Tribal Deck Management
        if path == '/api/player/hand':
            self._send_json({
                "success": True,
                "tribe": PLAYER_DECK_MGR.tribe.value,
                "hand": PLAYER_DECK_MGR.get_hand_summary(),
                "draw_pile_count": len(PLAYER_DECK_MGR.draw_pile),
                "discard_pile_count": len(PLAYER_DECK_MGR.discard_pile)
            })
            return

        if path == '/api/match/status':
            self._send_json({
                "turn": GAME_STATE.get("turn", 1),
                "player": {
                    "hp": GAME_STATE.get("player_hp", 6),
                    "max_hp": GAME_STATE.get("player_max_hp", 6),
                    "armor": GAME_STATE.get("player_armor", 0),
                    "mana": GAME_STATE.get("player_mana", 10),
                    "hand_count": len(PLAYER_DECK_MGR.hand)
                },
                "enemy": {
                    "hp": GAME_STATE.get("enemy_hp", 6),
                    "max_hp": GAME_STATE.get("enemy_max_hp", 6),
                    "armor": GAME_STATE.get("enemy_armor", 0),
                    "mana": ACTIVE_ECONOMIC_MATCH.enemy.mana,
                    "hand_count": len(ENEMY_DECK_MGR.hand)
                },
                "winner": "player" if GAME_STATE.get("enemy_hp", 6) <= 0 else ("enemy" if GAME_STATE.get("player_hp", 6) <= 0 else None)
            })
            return

        # ── ECONOMIC GAME FRAMEWORK API (GET) ─────────────────────────────────
        if path == '/api/economy/state':
            self._send_json(ACTIVE_ECONOMIC_MATCH.to_dict())
            return

        if path == '/api/economy/buildings':
            buildings_list = [b.to_dict() for b in BUILDING_REGISTRY.values()]
            self._send_json({"buildings": buildings_list})
            return

        if path == '/api/economy/abilities':
            abilities_list = [a.to_dict() for a in ABILITY_REGISTRY.values()]
            self._send_json({"abilities": abilities_list})
            return

        if path == '/api/economy/templates':
            self._send_json(get_all_templates())
            return

        if path == '/api/economy/tscn':
            tscn_content = generate_godot_tscn(ACTIVE_ECONOMIC_MATCH)
            self._send_text(tscn_content, content_type='text/plain')
            return

        # Contested Sectors & Territorial State
        if path in ['/api/sectors', '/api/sectors/state']:
            self._send_json({
                "sectors": [s.to_dict() for s in ACTIVE_ECONOMIC_MATCH.sectors],
                "round_number": ACTIVE_ECONOMIC_MATCH.round_number,
                "escalation": ACTIVE_ECONOMIC_MATCH.escalation.value
            })
            return

        # ── WARHAMMER & CRAFTING GET ENDPOINTS ────────────────────────────────
        if path == '/api/crafting/templates':
            self._send_json({
                "templates": [t.to_dict() for t in BASE_TEMPLATES.values()],
                "affixes": [a.to_dict() for a in AFFIX_REGISTRY.values()],
                "rarity_limits": {r.value: lim for r, lim in RARITY_AFFIX_LIMITS.items()}
            })
            return

        if path == '/api/checkpoints/state':
            self._send_json({
                "flags": [f.to_dict() for f in ACTIVE_CHECKPOINT_MGR.flags.values()],
                "total_victory_points": ACTIVE_CHECKPOINT_MGR.total_vp
            })
            return

        if path == '/api/locational/ward-status':
            self._send_json(ACTIVE_TOWER_WARD.to_dict())
            return

        if path == '/api/mobility/units':
            self._send_json({
                "player": PLAYER_MOBILITY_UNIT.to_dict(),
                "enemy": ENEMY_MOBILITY_UNIT.to_dict()
            })
            return

        if path == '/api/environment/matrix/state':
            self._send_json(ACTIVE_ENV_MATRIX.to_matrix_payload())
            return

        if path == '/api/calculator/presets':
            self._send_json({
                "weapons": [
                    {"id": "mortar_indirect", "name": "Obliehací Moždiar (Plunging Fire)", "type": "mortar", "min_range": 2, "max_range": 5, "base_damage": 3, "base_ap": 1, "desc": "Strmá parabolická paľba (φ >= 45°), ignoruje bežné prekážky."},
                    {"id": "gun_direct", "name": "Prierazné Delo (Direct Fire)", "type": "gun", "min_range": 1, "max_range": 4, "base_damage": 3, "base_ap": 2, "desc": "Priama paľba s vysokou úsťovou rýchlosťou vyžadujúca priamu viditeľnosť."},
                    {"id": "magic_artillery", "name": "Aéterové Orbitálne Delo", "type": "artillery", "min_range": 2, "max_range": 6, "base_damage": 4, "base_ap": 2, "desc": "Kozmický energetický lúč s minimálnym rozptylom."},
                    {"id": "sniper_railgun", "name": "Kryštálový Railgun", "type": "gun", "min_range": 2, "max_range": 5, "base_damage": 3, "base_ap": 3, "desc": "Extrémna kinetická prieraznosť po priamke."}
                ],
                "enchantments": [
                    {"id": "none", "name": "Bez očarovania", "element": "none"},
                    {"id": "crystal_resonance", "name": "Kryštálová Rezonancia (+2 AP, +1 Rozsah)", "element": "crystal"},
                    {"id": "toxic_corrosion", "name": "Toxická Korózia (+2 Kyslé DMG, Pole Kyseliny)", "element": "toxic"},
                    {"id": "druidic_verdance", "name": "Druidská Sila (+1 DMG, Zviazanie Koreňmi)", "element": "druid"},
                    {"id": "aether_inferno", "name": "Aéterové Peklo (+2 Oheň, Horľavé Pole)", "element": "aether"}
                ],
                "combos": [
                    {"id": "spotter_beacon", "name": "Zameriavací Maják (Spotter Beacon)", "pairs_with": "mortar_barrage"},
                    {"id": "armor_pierce_gun", "name": "Prierazná Kanonáda (Armor Pierce)", "pairs_with": "melee_charge"},
                    {"id": "toxic_spore_mist", "name": "Toxické Spóry (Toxic Mist)", "pairs_with": "aether_mortar"},
                    {"id": "druidic_roots", "name": "Zväzujúce Korene (Roots)", "pairs_with": "mortar_barrage"}
                ]
            })
            return

        # ── ASSET MANIFEST & CONTENT CATALOG HYDRATION ────────────────────────
        if path == '/api/assets/manifest':
            manifest_data = {
                "version": "1.0.0",
                "game": "Poslední Kmen",
                "vendor_scripts": [
                    {"name": "Three.js", "local": "/static/vendor/three.min.js", "cdn": "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"},
                    {"name": "OrbitControls", "local": "/static/vendor/OrbitControls.js", "cdn": "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"}
                ],
                "procedural_meshes": [
                    "crystal_archon_hero", "toxic_defiler_hero", "druid_elder_hero",
                    "hex_tile_crystal", "hex_tile_toxic", "hex_tile_druid",
                    "defense_tower_pylon", "ward_energy_bubble", "checkpoint_banner"
                ],
                "audio_cues": ["card_cast", "projectile_flight", "damage_impact", "ward_absorb", "victory_triumph"],
                "status": "HEALTHY_OFFLINE_READY"
            }
            self._send_json(manifest_data)
            return

        if path == '/api/game/content-catalog':
            catalog = {
                "tribal_cards": get_full_tribal_card_catalog(),
                "unit_archetypes": get_all_unit_archetypes(),
                "crafting_templates": [t.to_dict() for t in BASE_TEMPLATES.values()],
                "affixes": [a.to_dict() for a in AFFIX_REGISTRY.values()],
                "rarity_limits": {r.value: lim for r, lim in RARITY_AFFIX_LIMITS.items()},
                "checkpoints": [f.to_dict() for f in ACTIVE_CHECKPOINT_MGR.flags.values()],
                "sectors": [s.to_dict() for s in (ACTIVE_ECONOMIC_MATCH.sectors.values() if isinstance(ACTIVE_ECONOMIC_MATCH.sectors, dict) else ACTIVE_ECONOMIC_MATCH.sectors)]
            }
            self._send_json(catalog)
            return

        if path == '/api/game/match-state':
            state_snapshot = {
                "turn": GAME_STATE.get("turn", 1),
                "game_state": GAME_STATE,
                "player_hand": PLAYER_DECK_MGR.get_hand_summary() if PLAYER_DECK_MGR else [],
                "draw_pile_count": len(PLAYER_DECK_MGR.draw_pile) if PLAYER_DECK_MGR else 0,
                "discard_pile_count": len(PLAYER_DECK_MGR.discard_pile) if PLAYER_DECK_MGR else 0,
                "units": {
                    "player": PLAYER_MOBILITY_UNIT.to_dict(),
                    "enemy": ENEMY_MOBILITY_UNIT.to_dict()
                },
                "ward": ACTIVE_TOWER_WARD.to_dict(),
                "flags": [f.to_dict() for f in ACTIVE_CHECKPOINT_MGR.flags.values()],
                "total_victory_points": ACTIVE_CHECKPOINT_MGR.total_vp,
                "ledger": GAME_STATE.get("ledger", [])
            }
            self._send_json(state_snapshot)
            return

        if path == '/api/game/races':
            self._send_json(get_race_and_specializations_catalog())
            return

        if path == '/api/game/legend-map':
            self._send_json(get_map_legend_data())
            return

        # Serve baked 3D .OBJ Assets directly with CORS
        if path.startswith('/api/assets/'):
            asset_name = os.path.basename(path)
            asset_path = os.path.join(ASSET_DIR, asset_name)
            if os.path.exists(asset_path) and os.path.isfile(asset_path):
                with open(asset_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._send_text(content, content_type='text/plain')
            else:
                self._send_json({"error": f"Asset {asset_name} not found"}, status=404)
            return

        if path == '/api/tactical/community_builds':
            self._send_json({"success": True, "builds": GLOBAL_MCP_BRIDGE.get_community_builds()})
            return

        if path == '/api/mcp/poll_frames':
            qs = urllib.parse.parse_qs(parsed.query)
            duel_id = qs.get("duel_id", [""])[0]
            recipient_id = qs.get("recipient_id", [""])[0]
            since_ts = float(qs.get("since_ts", [0.0])[0])
            frames = GLOBAL_MCP_BRIDGE.poll_opponent_frames(duel_id, recipient_id, since_ts)
            self._send_json({"success": True, "duel_id": duel_id, "frames": frames})
            return

        # ── ADS & QUOTAS (GET) ────────────────────────────────────────────────
        if path == '/api/ads/quota_status':
            qs = urllib.parse.parse_qs(parsed.query)
            player_id = qs.get("player_id", ["default_player"])[0]
            status_data = GLOBAL_AD_QUOTA_MGR.get_player_status(player_id)
            next_reward = GLOBAL_AD_QUOTA_MGR.calculate_ad_reward(player_id)
            self._send_json({"success": True, "quota_status": status_data, "next_reward": next_reward})
            return

        # ── MARKET, CHESTS & AUCTIONS (GET) ──────────────────────────────────
        if path == '/api/market/chests':
            self._send_json({"success": True, "chests": list(CHEST_CATALOG.values())})
            return

        if path == '/api/market/artifacts':
            self._send_json({"success": True, "artifacts": list(ARTIFACT_CATALOG.values())})
            return

        if path == '/api/market/auctions':
            self._send_json({"success": True, "auctions": GLOBAL_AUCTION_HOUSE.list_active_auctions()})
            return

        # ── 60-CHARACTER ROSTER & HUD SKILL PANELS (GET) ─────────────────────
        if path == '/api/roster/characters':
            qs = urllib.parse.parse_qs(parsed.query)
            race = qs.get("race", [None])[0]
            tier_str = qs.get("tier", [None])[0]
            tier = int(tier_str) if tier_str is not None else None

            chars = list(ALL_SIXTY_CHARACTERS.values())
            if race:
                chars = [c for c in chars if c["race"].lower() == race.lower()]
            if tier is not None:
                chars = [c for c in chars if c["tier"] == tier]

            self._send_json({
                "success": True,
                "total_available": len(ALL_SIXTY_CHARACTERS),
                "matched_count": len(chars),
                "characters": chars
            })
            return

        if path == '/api/roster/character':
            qs = urllib.parse.parse_qs(parsed.query)
            char_id = qs.get("id", [""])[0]
            char = SixtyCharacterRosterEngine.get_character(char_id)
            if char:
                self._send_json({"success": True, "character": char})
            else:
                self._send_json({"success": False, "error": f"Character {char_id} not found"}, status=404)
            return

        if path == '/api/duel/the_west_builds':
            self._send_json({"success": True, "builds": TheWestDuelAlgebra.DUEL_BUILDS})
            return

        # ── AD EXCHANGE PROTOCOL (GET) ───────────────────────────────────────
        if path == '/api/ad_protocol/campaigns':
            self._send_json({
                "success": True,
                "campaigns": GLOBAL_AD_EXCHANGE.list_campaigns()
            })
            return

        if path == '/api/ad_protocol/ledger':
            self._send_json({
                "success": True,
                "ledger": GLOBAL_AD_EXCHANGE.get_ledger_summary()
            })
            return

        # ── CLUSTER MONITORING (GET) ─────────────────────────────────────────
        if path == '/api/cluster/monitoring/health':
            self._send_json({
                "success": True,
                "cluster_health": GLOBAL_MONITORING_CLUSTER.evaluate_cluster_health()
            })
            return

        # ── FIREWALL STATUS (GET) ────────────────────────────────────────────
        if path == '/api/security/firewall/status':
            self._send_json({
                "success": True,
                "firewall": GLOBAL_FIREWALL.get_firewall_status()
            })
            return

        # ── COSMOLOGY, APOSTLES & NATURE SYSTEMS (GET) ───────────────────────
        if path == '/api/cosmology/planes':
            self._send_json({
                "success": True,
                "planes": [p.value for p in CosmologicalPlane],
                "plane_details": {p.value: data for p, data in COSMOLOGICAL_PLANE_DATA.items()}
            })
            return

        if path == '/api/cosmology/apostles':
            self._send_json({
                "success": True,
                "count": len(TWELVE_APOSTLES),
                "apostles": ApostlesAndAngelsRegistry.get_apostles()
            })
            return

        if path == '/api/cosmology/angels':
            self._send_json({
                "success": True,
                "count": len(ANGELIC_GUARDIANS),
                "angels": ApostlesAndAngelsRegistry.get_angels()
            })
            return

        if path == '/api/cosmology/portals':
            self._send_json({
                "success": True,
                "portals": DimensionalPortalsAndMirrors.get_portals()
            })
            return

        if path == '/api/cosmology/tree_network':
            self._send_json({
                "success": True,
                "network": ArborMycorrhizalNetwork.evaluate_root_network()
            })
            return

        if path == '/api/transmitter/status':
            self._send_json({
                "success": True,
                "transmitter": GLOBAL_BOT_TRANSMITTER.to_dict()
            })
            return

        # ── 20 MALE / 20 FEMALE ARCHETYPES, RACES & 120 HELPERS (GET) ────────
        if path == '/api/archetypes/representatives':
            qs = urllib.parse.parse_qs(parsed.query)
            gender_filter = qs.get("gender", ["all"])[0].lower()
            class_filter = qs.get("class", [""])[0]

            if gender_filter == "male":
                reps = ArchetypeAndHelperEngine.list_characters(gender="male", archetype_class=class_filter)
            elif gender_filter == "female":
                reps = ArchetypeAndHelperEngine.list_characters(gender="female", archetype_class=class_filter)
            else:
                reps = ArchetypeAndHelperEngine.list_characters(gender=None, archetype_class=class_filter)

            self._send_json({
                "success": True,
                "total_male": len(MALE_ARCHETYPES_20),
                "total_female": len(FEMALE_ARCHETYPES_20),
                "returned_count": len(reps),
                "representatives": reps
            })
            return

        if path == '/api/archetypes/character':
            qs = urllib.parse.parse_qs(parsed.query)
            char_id = qs.get("id", [""])[0]
            char = ArchetypeAndHelperEngine.get_character(char_id)
            if char:
                self._send_json({"success": True, "character": char})
            else:
                self._send_json({"success": False, "error": f"Character {char_id} not found"}, status=404)
            return

        if path == '/api/archetypes/races':
            self._send_json({
                "success": True,
                "count": len(RACES_CATALOG),
                "races": list(RACES_CATALOG.values())
            })
            return

        if path == '/api/archetypes/helpers':
            qs = urllib.parse.parse_qs(parsed.query)
            race_q = qs.get("race", [""])[0]
            tier_q = qs.get("tier", [""])[0]
            tier_int = int(tier_q) if tier_q.isdigit() else None
            helpers = ArchetypeAndHelperEngine.list_helpers(race=race_q if race_q else None, tier=tier_int)
            self._send_json({
                "success": True,
                "total_count": len(HELPERS_120_CATALOG),
                "filtered_count": len(helpers),
                "helpers": helpers
            })
            return

        if path == '/api/archetypes/helper':
            qs = urllib.parse.parse_qs(parsed.query)
            idx_q = qs.get("index", [""])[0]
            id_q = qs.get("id", [""])[0]
            helper = None
            if idx_q:
                helper = ArchetypeAndHelperEngine.get_helper(int(idx_q) if idx_q.isdigit() else idx_q)
            elif id_q:
                helper = ArchetypeAndHelperEngine.get_helper(id_q)
            if helper:
                self._send_json({"success": True, "helper": helper})
            else:
                self._send_json({"success": False, "error": "Helper not found"}, status=404)
            return

        # ── ZODIAC, GRIDS & PLUS INVENTORY (GET) ─────────────────────────────
        if path == '/api/cosmic/zodiac_sky':
            qs = urllib.parse.parse_qs(parsed.query)
            t_str = qs.get("time_sec", [""])[0]
            t_sec = float(t_str) if t_str else time.time()
            zodiac_res = ZodiacSkyEngine.get_celestial_zodiac(t_sec)
            self._send_json({
                "success": True,
                "zodiac_sky": zodiac_res,
                "all_constellations": list(ZODIAC_CONSTELLATIONS.values())
            })
            return

        if path == '/api/system/grid_matrices':
            grids_list = []
            for name, spec in GRID_PRESET_SPECS.items():
                res = SacredNumerologyEngine.evaluate_grid_resonance(name)
                grids_list.append({**spec, "resonance": res})
            self._send_json({
                "success": True,
                "count": len(GRID_PRESET_SPECS),
                "grid_matrices": grids_list
            })
            return

        if path == '/api/inventory/plus_status':
            self._send_json({
                "success": True,
                "inventory": GLOBAL_PLUS_INVENTORY.to_dict()
            })
            return

        # ── ART AUCTIONS & ORDINALS (GET) ────────────────────────────────────
        if path == '/api/art/auctions':
            qs = urllib.parse.parse_qs(parsed.query)
            st = qs.get("status", ["active"])[0]
            lots = GLOBAL_PAINTING_AUCTIONS.list_lots(status=st)
            self._send_json({
                "success": True,
                "count": len(lots),
                "status_filter": st,
                "lots": lots
            })
            return

        if path == '/api/ordinals/list':
            inscriptions = GLOBAL_ORDINALS_PROTOCOL.list_inscriptions()
            self._send_json({
                "success": True,
                "count": len(inscriptions),
                "inscriptions": inscriptions
            })
            return

        if path == '/api/ordinals/export':
            qs = urllib.parse.parse_qs(parsed.query)
            insc_id = qs.get("inscription_id", [""])[0]
            if not insc_id:
                # Return all exported envelopes
                exported = [GLOBAL_ORDINALS_PROTOCOL.export_ordinal(i["id"]) for i in GLOBAL_ORDINALS_PROTOCOL.list_inscriptions()]
                self._send_json({"success": True, "count": len(exported), "exported_ordinals": exported})
            else:
                exp = GLOBAL_ORDINALS_PROTOCOL.export_ordinal(insc_id)
                if exp:
                    self._send_json({"success": True, "exported_ordinal": exp})
                else:
                    self._send_json({"success": False, "error": f"Inscription {insc_id} not found"}, status=404)
            return

        # ── CONTENT REPLAY & LEVEL SNAPSHOTS (GET) ───────────────────────────
        if path == '/api/replay/records':
            matches = GLOBAL_CONTENT_REPLAY.list_matches()
            self._send_json({
                "success": True,
                "count": len(matches),
                "matches": matches
            })
            return

        if path == '/api/replay/level':
            qs = urllib.parse.parse_qs(parsed.query)
            m_id = qs.get("match_id", [""])[0]
            lvl_idx = int(qs.get("level_idx", [0])[0])
            segment = GLOBAL_CONTENT_REPLAY.get_level_segment(m_id, lvl_idx)
            if segment:
                self._send_json({"success": True, "level_segment": segment})
            else:
                self._send_json({"success": False, "error": f"Level segment not found for {m_id}:{lvl_idx}"}, status=404)
            return

        # ── ML CONGESTION CONTROL (GET) ──────────────────────────────────────
        if path == '/api/ml/congestion_control':
            self._send_json({
                "success": True,
                "congestion_status": GLOBAL_CONGESTION_CONTROLLER.get_status()
            })
            return

        # ── METAVERSE MARKETPLACE & GRANITE LLM (GET) ────────────────────────
        if path == '/api/metaverse/market/catalog':
            cat = GLOBAL_METAVERSE_MARKET.get_catalog()
            self._send_json({
                "success": True,
                "count": len(cat),
                "catalog": cat
            })
            return

        if path == '/api/metaverse/market/orderbook':
            qs = urllib.parse.parse_qs(parsed.query)
            pair = qs.get("pair", ["LAND_042/AET"])[0]
            ob = GLOBAL_METAVERSE_MARKET.get_order_book(pair)
            self._send_json({"success": True, "order_book": ob})
            return

        if path == '/api/metaverse/market/amm_pools':
            pools = GLOBAL_METAVERSE_MARKET.get_amm_pools()
            self._send_json({
                "success": True,
                "count": len(pools),
                "pools": pools
            })
            return

        if path == '/api/metaverse/llm/models':
            models = GLOBAL_GRANITE_LLM.get_registered_models()
            self._send_json({
                "success": True,
                "count": len(models),
                "models": models
            })
            return

        if path == '/api/metaverse/llm/budget':
            qs = urllib.parse.parse_qs(parsed.query)
            m_key = qs.get("model_key", ["ibm-granite-3.1-4b-instruct-2026"])[0]
            budget = GLOBAL_GRANITE_LLM.verify_hardware_budget(m_key)
            self._send_json({"success": True, "budget": budget})
            return

        if path == '/api/totems/sector_status':
            qs = urllib.parse.parse_qs(parsed.query)
            sector_id = qs.get("sector_id", [None])[0]
            totems = GLOBAL_TOTEM_MANAGER.get_totems(sector_id)
            self._send_json({
                "success": True,
                "count": len(totems),
                "vital_hp_invariant": 6,
                "totems": totems
            })
            return

        if path == '/api/totems/anomalies':
            anomalies = GLOBAL_ANOMALY_DETECTOR.get_active_anomalies()
            self._send_json({
                "success": True,
                "count": len(anomalies),
                "anomalies": anomalies
            })
            return

        if path == '/api/totems/phenomena_catalog':
            catalog = GLOBAL_VISUAL_PHENOMENA.get_all_phenomena_catalog()
            self._send_json({
                "success": True,
                "count": len(catalog),
                "phenomena": catalog
            })
            return

        if path == '/api/totems/spell_catalog':
            spells = SpellProjectionEngine.SPELL_CATALOG
            self._send_json({
                "success": True,
                "count": len(spells),
                "spells": spells
            })
            return

        if path == '/api/cards/mulligan/hand':
            self._send_json({
                "success": True,
                "count": len(GLOBAL_MULLIGAN_MANAGER.opening_hand),
                "selected_count": len(GLOBAL_MULLIGAN_MANAGER.selected_indices),
                "selected_indices": list(GLOBAL_MULLIGAN_MANAGER.selected_indices),
                "hand": GLOBAL_MULLIGAN_MANAGER.opening_hand,
                "is_completed": GLOBAL_MULLIGAN_MANAGER.is_completed
            })
            return

        if path == '/api/cards/mulligan/catalog':
            self._send_json({
                "success": True,
                "count": len(CANONICAL_MULLIGAN_CARDS),
                "cards": CANONICAL_MULLIGAN_CARDS
            })
            return

        if path == '/api/tactical/combinatorial_moves/specs':
            self._send_json({
                "success": True,
                "total_combinations": CombinatorialTacticalMoveEngine.TOTAL_COMBINATORIAL_STATES,
                "formula": "2^18 = 512 macro-branches x 512 spatial paths = 262,144 tactical combinations",
                "vital_max_hp_rule": 6,
                "evaluated_mechanics": [
                    "hero_move_1_and_escort_unit_1",
                    "connecting_crystal_conduit_extension",
                    "slow_to_root_projectile_combo",
                    "sector_freeze_aoe_control"
                ]
            })
            return

        if path.startswith('/api/') and self._proxy_to_hub("GET"):
            return
        self._send_json({"error": f"Endpoint {path} not found"}, status=404)

    def do_POST(self):
        global ACTIVE_ECONOMIC_MATCH
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8', errors='replace') if content_length > 0 else "{}"
        
        # ── Firewall Layer 7 Deep Packet Inspection ─────────────────
        client_ip = self.client_address[0] if hasattr(self, 'client_address') and self.client_address else "127.0.0.1"
        headers_dict = {k: v for k, v in self.headers.items()}
        allowed, threat = GLOBAL_FIREWALL.inspect_request(client_ip, self.path, headers_dict, body_text=post_data)
        if not allowed:
            self._send_json({"error": "FORBIDDEN_BY_FIREWALL", "incident": threat}, status=403)
            return

        try:
            req_data = json.loads(post_data) if post_data else {}
        except json.JSONDecodeError:
            req_data = {}

        # 1. Procedural Scene Generation
        if path == '/api/bot/generate-scene':
            prompt = req_data.get("prompt", "Taktická Aréna Poslední Kmen")
            print(f"[Python Engine Core] Compiling Scene for: '{prompt}'")
            scene_data = execute_janet_cms(prompt)
            self._send_json(scene_data)
            return

        # 2. Card Cast & Ledger Accounting
        if path == '/api/cards/cast':
            card_id = req_data.get("card_id")
            target_pos = req_data.get("target_hex", [0, 0, 0])
            source_raw = req_data.get("source_hex", [0, -2])
            
            card = KMEN_CARDS.get(card_id)
            if not card and PLAYER_DECK_MGR:
                # Search player's hand and deck
                d_card = next((c for c in PLAYER_DECK_MGR.hand if (c.get("id") == card_id or c.get("id_base") == card_id)), None)
                if not d_card:
                    d_card = next((c for c in PLAYER_DECK_MGR.draw_pile if (c.get("id") == card_id or c.get("id_base") == card_id)), None)
                if not d_card:
                    d_card = next((c for c in PLAYER_DECK_MGR.discard_pile if (c.get("id") == card_id or c.get("id_base") == card_id)), None)
                if d_card:
                    atk_type = d_card.get("attack_type", "ranged")
                    is_atk = atk_type in ("melee", "ranged", "aoe")
                    power = d_card.get("power", 2)
                    tribe_val = d_card.get("tribe", "crystal")
                    if hasattr(tribe_val, "value"):
                        tribe_val = tribe_val.value
                    c_name = d_card.get("name", card_id)
                    card = {
                        "id": d_card.get("id", card_id),
                        "name": c_name,
                        "tribe": str(tribe_val),
                        "cost": d_card.get("cost", 2),
                        "hp_delta": -power if is_atk else (power if "heal" in c_name.lower() or "regen" in c_name.lower() else 0),
                        "armor_delta": power if "shield" in c_name.lower() or "overclock" in c_name.lower() or atk_type == "self" else 0,
                        "mana_delta": 0,
                        "attack_type": atk_type,
                        "min_range": d_card.get("min_range", 1),
                        "max_range": d_card.get("max_range", 3),
                        "trajectory_type": d_card.get("trajectory_type", "arc"),
                        "spawn_prop": "crystal_shield" if "shield" in card_id else "crystal_shard",
                        "color": d_card.get("color", "#00ffff"),
                        "description": d_card.get("description", "")
                    }

            if not card:
                self._send_json({"error": f"Card '{card_id}' not found in Poslední Kmen library"}, status=400)
                return

            source_hex = world_pos_to_hex(source_raw)
            target_hex = world_pos_to_hex(target_pos)

            # Validate range
            dist = calculate_hex_distance(source_hex, target_hex)
            card_type = card.get("attack_type", "ranged")
            min_r = card.get("min_range", 1)
            max_r = card.get("max_range", 3)

            if card_type != "self" and card_type != "global":
                if dist < min_r or dist > max_r:
                    self._send_json({
                        "error": f"Cieľ je mimo dosahu pre kartu '{card['name']}'! (Vzdialenosť: {dist}, povolené: {min_r}-{max_r})",
                        "distance": dist,
                        "min_range": min_r,
                        "max_range": max_r
                    }, status=400)
                    return

            cost = card["cost"]
            if GAME_STATE["player_mana"] < cost:
                self._send_json({"error": "Nedostatok Many pre vyloženie karty!", "required": cost, "current": GAME_STATE["player_mana"]}, status=400)
                return

            GAME_STATE["player_mana"] -= cost
            dmg_dealt = 0
            if card["hp_delta"] < 0:
                dmg = abs(card["hp_delta"])
                dmg_dealt = dmg
                effective_dmg = max(0, dmg - GAME_STATE["enemy_armor"])
                GAME_STATE["enemy_armor"] = max(0, GAME_STATE["enemy_armor"] - dmg)
                GAME_STATE["enemy_hp"] = max(0, GAME_STATE["enemy_hp"] - effective_dmg)
            elif card["hp_delta"] > 0:
                GAME_STATE["player_hp"] = min(GAME_STATE["player_max_hp"], GAME_STATE["player_hp"] + card["hp_delta"])

            if card.get("armor_delta", 0) > 0:
                GAME_STATE["player_armor"] += card["armor_delta"]

            # If played from player's hand, play & discard
            if PLAYER_DECK_MGR and any(c.get("id") == card_id for c in PLAYER_DECK_MGR.hand):
                PLAYER_DECK_MGR.play_card(card_id)

            # Sync active economic match
            if ACTIVE_ECONOMIC_MATCH:
                ACTIVE_ECONOMIC_MATCH.player.hp = GAME_STATE["player_hp"]
                ACTIVE_ECONOMIC_MATCH.enemy.hp = GAME_STATE["enemy_hp"]
                ACTIVE_ECONOMIC_MATCH.player.mana = GAME_STATE["player_mana"]

            animation_payload = {
                "type": card.get("animation_fx", "animated_arrow"),
                "trajectory": card.get("trajectory_type", "arc"),
                "attack_type": card_type,
                "source_hex": source_hex,
                "target_hex": target_hex,
                "distance": dist,
                "min_range": min_r,
                "max_range": max_r,
                "color": card.get("color", "#00ffff"),
                "duration": round(0.45 + dist * 0.12, 2)
            }

            ledger_entry = {
                "turn": GAME_STATE["turn"],
                "card": card["name"],
                "tribe": card["tribe"],
                "cost": cost,
                "mana_remaining": GAME_STATE["player_mana"],
                "player_hp": GAME_STATE["player_hp"],
                "enemy_hp": GAME_STATE["enemy_hp"],
                "desc": f"Zahraná karta {card['name']} (Cena: {cost} Mana). Cieľ: {target_hex} (Vzdialenosť: {dist})"
            }
            GAME_STATE["ledger"].append(ledger_entry)

            base_mesh = card.get("mesh_node")
            if not base_mesh:
                base_mesh = {
                    "type": "MeshInstance",
                    "name": f"Spawned_{card_id}",
                    "properties": {
                        "mesh": f"res://godot_assets/{card.get('spawn_prop', 'crystal_shard')}.obj",
                        "scale": [1.0, 1.0, 1.0],
                        "color": card.get("color", "#00ffff")
                    },
                    "children": []
                }
            mesh_spawn = json.loads(json.dumps(base_mesh))
            mesh_spawn["properties"]["position"] = target_pos

            self._send_json({
                "status": "SUCCESS",
                "success": True,
                "card_played": card["name"],
                "damage": dmg_dealt,
                "cost": cost,
                "ledger_entry": ledger_entry,
                "game_state": GAME_STATE,
                "player_hp": GAME_STATE["player_hp"],
                "enemy_hp": GAME_STATE["enemy_hp"],
                "remaining_hand": PLAYER_DECK_MGR.get_hand_summary() if PLAYER_DECK_MGR else [],
                "spawn_node": mesh_spawn,
                "animation": animation_payload
            })
            return

        # ── ECONOMIC GAME FRAMEWORK API (POST) ────────────────────────────────
        # A. Construct Building on Hex Grid
        if path == '/api/economy/build':
            b_type_str = req_data.get("building_type")
            hex_coords = req_data.get("hex_coords", [0, 0])
            position = req_data.get("position", [0.0, 0.0, 0.0])
            try:
                b_type = BuildingType(b_type_str)
            except ValueError:
                self._send_json({"error": f"Neznámy typ budovy '{b_type_str}'"}, status=400)
                return

            new_building, deltas, status_msg = EconomicLedger.construct_building(
                ACTIVE_ECONOMIC_MATCH.player, b_type, hex_coords, position, ACTIVE_ECONOMIC_MATCH.escalation
            )
            if not new_building:
                self._send_json({"error": status_msg}, status=400)
                return

            # Append to ledger
            ledger_entry = LedgerEntry(
                timestamp_turn=int(time.time()),
                round_number=ACTIVE_ECONOMIC_MATCH.round_number,
                phase=ACTIVE_ECONOMIC_MATCH.phase.value,
                event_type="BUILDING_CONSTRUCTED",
                description=f"Postavená budova {new_building.name} na poli {hex_coords}. Náklady: {new_building.cost.to_dict()}.",
                resource_delta=deltas or {},
                balance_after=EconomicLedger.get_current_balances(ACTIVE_ECONOMIC_MATCH.player)
            )
            ACTIVE_ECONOMIC_MATCH.ledger.append(ledger_entry)

            # Node AST for 3D Viewport & Godot
            building_node = {
                "type": "Spatial",
                "name": f"Building_{new_building.id}",
                "properties": {
                    "position": position,
                    "hex_coords": hex_coords,
                    "tier": new_building.tier,
                    "color": new_building.color
                },
                "children": [
                    {
                        "type": "MeshInstance",
                        "name": "StructureMesh",
                        "properties": {
                            "mesh": f"res://godot_assets/{new_building.mesh_asset}",
                            "color": new_building.color,
                            "scale": [1.2, 1.2, 1.2]
                        },
                        "children": []
                    },
                    {
                        "type": "OmniLight",
                        "name": "Beacon",
                        "properties": {"color": new_building.color, "energy": 2.2, "range": 6.0},
                        "children": []
                    }
                ]
            }

            self._send_json({
                "status": "SUCCESS",
                "building": new_building.to_dict(),
                "building_node": building_node,
                "match_state": ACTIVE_ECONOMIC_MATCH.to_dict()
            })
            return

        # B. Step Round / Phase
        if path == '/api/economy/round/step':
            step_result = RoundController.step_phase(ACTIVE_ECONOMIC_MATCH)
            self._send_json(step_result)
            return

        # C. Cast Ability with Economic Resource Accounting & Combos
        if path == '/api/economy/ability/cast':
            ability_id = req_data.get("ability_id")
            target_pos = req_data.get("target_hex", [0, 0, 0])
            source_raw = req_data.get("source_hex")
            caster_who = req_data.get("caster", "player")

            caster = ACTIVE_ECONOMIC_MATCH.player if caster_who == "player" else ACTIVE_ECONOMIC_MATCH.enemy
            target = ACTIVE_ECONOMIC_MATCH.enemy if caster_who == "player" else ACTIVE_ECONOMIC_MATCH.player

            source_hex = world_pos_to_hex(source_raw) if source_raw else ([0, -2] if caster_who == "player" else [0, 2])
            target_hex = world_pos_to_hex(target_pos)

            success, details, err_msg = AbilityEngine.cast_ability(
                caster, target, ability_id, ACTIVE_ECONOMIC_MATCH.escalation,
                source_hex=source_hex, target_hex=target_hex
            )
            if not success:
                self._send_json({"error": err_msg}, status=400)
                return

            # Append to Ledger
            ledger_entry = LedgerEntry(
                timestamp_turn=int(ACTIVE_ECONOMIC_MATCH.round_number),
                round_number=ACTIVE_ECONOMIC_MATCH.round_number,
                phase=ACTIVE_ECONOMIC_MATCH.phase.value,
                event_type="ABILITY_CAST",
                description=f"Použitá schopnosť {details['ability_name']} ({'COMBO! ' if details['is_combo'] else ''}Dmg: {details['damage_dealt']}, Heal: {details['healing_done']}). Cieľ: {target_hex}.",
                resource_delta=details["cost_deltas"],
                balance_after=EconomicLedger.get_current_balances(caster)
            )
            ACTIVE_ECONOMIC_MATCH.ledger.append(ledger_entry)

            # Build 3D Spell Node for Viewport
            spell_node = {
                "type": "Spatial",
                "name": f"FX_{ability_id}",
                "properties": {"position": target_pos},
                "children": [
                    {
                        "type": "MeshInstance",
                        "name": "SpellCore",
                        "properties": {
                            "mesh": f"res://godot_assets/{details['mesh_asset']}",
                            "color": details["color"],
                            "scale": [1.5, 1.5, 1.5]
                        },
                        "children": []
                    },
                    {
                        "type": "OmniLight",
                        "name": "CastLight",
                        "properties": {"color": details["color"], "energy": 3.0, "range": 8.0},
                        "children": []
                    },
                    {
                        "type": "Particles",
                        "name": "SpellParticles",
                        "properties": {"amount": 100, "color": details["color"]},
                        "children": []
                    }
                ]
            }

            self._send_json({
                "status": "SUCCESS",
                "details": details,
                "spell_node": spell_node,
                "animation": details.get("animation"),
                "match_state": ACTIVE_ECONOMIC_MATCH.to_dict()
            })
            return

        # D. Reset Match
        if path == '/api/economy/reset':
            p_tribe = req_data.get("player_tribe", "crystal")
            e_tribe = req_data.get("enemy_tribe", "toxic")
            ACTIVE_ECONOMIC_MATCH = RoundController.initialize_match(
                "match_reloaded", Tribe(p_tribe), Tribe(e_tribe)
            )
            GAME_STATE["player_hp"] = 6
            GAME_STATE["player_max_hp"] = 6
            GAME_STATE["player_armor"] = 0
            GAME_STATE["player_mana"] = 10
            GAME_STATE["enemy_hp"] = 6
            GAME_STATE["enemy_max_hp"] = 6
            GAME_STATE["enemy_armor"] = 0
            GAME_STATE["turn"] = 1
            GAME_STATE["ledger"] = [
                {"turn": 0, "event": "INIT", "desc": "Bojová aréna Poslední Kmen inicializovaná. Každý bojovník má 6 HP."}
            ]
            self._send_json({
                "status": "RESET_COMPLETE",
                "match_state": ACTIVE_ECONOMIC_MATCH.to_dict(),
                "game_state": GAME_STATE
            })
            return

        # E. Sector Assault & Territorial Fight
        if path == '/api/sectors/assault':
            sector_id = req_data.get("sector_id")
            attacker_side = req_data.get("attacker", "player")
            attack_power = int(req_data.get("attack_power", 4))
            mana_committed = int(req_data.get("mana_committed", 2))

            success, assault_details, err_msg = SectorConquestEngine.resolve_sector_assault(
                ACTIVE_ECONOMIC_MATCH,
                sector_id=sector_id,
                attacker_side=attacker_side,
                attack_power=attack_power,
                mana_committed=mana_committed
            )
            if not success:
                self._send_json({"error": err_msg}, status=400)
                return

            self._send_json({
                "status": "SUCCESS",
                "assault": assault_details,
                "match_state": ACTIVE_ECONOMIC_MATCH.to_dict()
            })
            return

        # F. Sector Fortify & Wall Upgrade
        if path == '/api/sectors/fortify':
            sector_id = req_data.get("sector_id")
            side = req_data.get("side", "player")

            success, fortify_details, err_msg = SectorConquestEngine.fortify_sector(
                ACTIVE_ECONOMIC_MATCH,
                sector_id=sector_id,
                side=side
            )
            if not success:
                self._send_json({"error": err_msg}, status=400)
                return

            self._send_json({
                "status": "SUCCESS",
                "fortify": fortify_details,
                "match_state": ACTIVE_ECONOMIC_MATCH.to_dict()
            })
            return

        # G. Range Rule Validation & Targeting Inspection
        if path == '/api/targeting/validate':
            source_raw = req_data.get("source_hex", [0, -2])
            target_raw = req_data.get("target_hex", [0, 0])
            source_hex = world_pos_to_hex(source_raw)
            target_hex = world_pos_to_hex(target_raw)

            ability_id = req_data.get("ability_id")
            card_id = req_data.get("card_id")

            min_r, max_r = 1, 3
            atype = "ranged"
            traj = "arc"
            color = "#00ffff"
            name = "Akcia"

            if ability_id and ability_id in ABILITY_REGISTRY:
                ab = ABILITY_REGISTRY[ability_id]
                min_r = ab.min_range
                max_r = ab.max_range
                atype = ab.attack_type.value if hasattr(ab.attack_type, "value") else str(ab.attack_type)
                traj = ab.trajectory_type
                color = ab.color
                name = ab.name
            elif card_id and card_id in KMEN_CARDS:
                cd = KMEN_CARDS[card_id]
                min_r = cd.get("min_range", 1)
                max_r = cd.get("max_range", 3)
                atype = cd.get("attack_type", "ranged")
                traj = cd.get("trajectory_type", "arc")
                color = cd.get("color", "#00ffff")
                name = cd.get("name", "Karta")

            dist = calculate_hex_distance(source_hex, target_hex)
            if atype == "self":
                is_valid = (dist == 0)
            elif atype == "global":
                is_valid = True
            else:
                is_valid = (min_r <= dist <= max_r)

            self._send_json({
                "valid": is_valid,
                "distance": dist,
                "min_range": min_r,
                "max_range": max_r,
                "attack_type": atype,
                "trajectory": traj,
                "color": color,
                "name": name,
                "source_hex": source_hex,
                "target_hex": target_hex
            })
            return

        # 3. Export to Godot .TSCN
        if path == '/api/export/godot-tscn':
            scene_ast = req_data.get("tree")
            if scene_ast:
                tscn_text = export_scene_to_tscn({"tree": scene_ast})
                self._send_text(tscn_text, content_type='text/plain')
            else:
                self._send_json({"error": "Missing tree in request body"}, status=400)
            return

        # 4. Card Fusion Endpoint (Formula 5)
        if path == '/api/cards/fuse':
            card_a_id = req_data.get("card_a")
            card_b_id = req_data.get("card_b")
            card_a = KMEN_CARDS.get(card_a_id) or ABILITY_REGISTRY.get(card_a_id)
            card_b = KMEN_CARDS.get(card_b_id) or ABILITY_REGISTRY.get(card_b_id)
            if hasattr(card_a, "to_dict"): card_a = card_a.to_dict()
            if hasattr(card_b, "to_dict"): card_b = card_b.to_dict()
            if not card_a or not card_b:
                self._send_json({"error": "Invalid card IDs for fusion"}, status=400)
                return
            fused = fuse_cards(card_a, card_b)
            self._send_json({"success": True, "fused_card": fused})
            return

        # 5. Hex Dynamic Terraforming Endpoint (Formula 4)
        if path == '/api/hex/terraforming':
            current_vec = tuple(req_data.get("current_vector", [0.333, 0.333, 0.334]))
            spell_element = req_data.get("element", "crystal")
            dist = req_data.get("distance", 0)
            res = update_hex_biome_transition(current_vec, spell_element, dist)
            self._send_json({"success": True, "result": res})
            return

        # 6. Ballistic Bezier-Hermite Trajectory Endpoint (Formula 2)
        if path == '/api/ballistics/calculate':
            p0 = req_data.get("p0", [0.0, 0.35, 0.0])
            p3 = req_data.get("p3", [0.0, 0.20, -3.0])
            dist = req_data.get("distance", 2)
            atype = req_data.get("attack_type", "ranged")
            tension = req_data.get("tension", 0.55)
            drag = req_data.get("drag", 0.04)
            wind = req_data.get("wind", [0.0, 0.0, 0.0])
            curve_data = sample_ballistic_bezier_hermite_curve(p0, p3, dist, atype, tension, drag, wind)
            self._send_json({"success": True, "curve": curve_data})
            return

        # 7. Godot AST Compaction Endpoint (Formula 6)
        if path == '/api/godot/ast/compact':
            tree = req_data.get("tree")
            if not tree:
                self._send_json({"error": "Missing tree in request body"}, status=400)
                return
            compaction = compact_godot_ast(tree)
            self._send_json({"success": True, "compaction": compaction})
            return

        # 8. AI Autonomous Tactical Turn
        if path == '/api/ai/turn':
            ai_cards = ENEMY_DECK_MGR.get_hand_summary()
            ai_result = TacticalAIEngine.evaluate_ai_turn(ACTIVE_ECONOMIC_MATCH, ai_side="enemy", available_cards=ai_cards)

            if ai_result.get("executed"):
                dmg = ai_result.get("damage_dealt", 0)
                if dmg > 0:
                    eff_dmg = max(0, dmg - GAME_STATE["player_armor"])
                    GAME_STATE["player_armor"] = max(0, GAME_STATE["player_armor"] - dmg)
                    GAME_STATE["player_hp"] = max(0, GAME_STATE["player_hp"] - eff_dmg)

                if ai_result.get("armor_gained", 0) > 0:
                    GAME_STATE["enemy_armor"] += ai_result["armor_gained"]

                if ai_result.get("card_id"):
                    ENEMY_DECK_MGR.play_card(ai_result["card_id"])
                ENEMY_DECK_MGR.draw_to_full()

                GAME_STATE["ledger"].append({
                    "turn": GAME_STATE.get("turn", 1),
                    "event": "AI_ATTACK",
                    "desc": f"AI zahral {ai_result.get('card_name')} (-{dmg} HP hráčovi)."
                })

            winner = "player" if GAME_STATE["enemy_hp"] <= 0 else ("enemy" if GAME_STATE["player_hp"] <= 0 else None)
            self._send_json({
                "success": True,
                "ai_action": ai_result,
                "winner": winner,
                "game_state": GAME_STATE
            })
            return

        # 9. Match End Turn (Round Progression & Allowance)
        if path == '/api/match/end-turn':
            GAME_STATE["turn"] = GAME_STATE.get("turn", 1) + 1
            GAME_STATE["player_mana"] = min(10, GAME_STATE.get("player_mana", 0) + 3)
            newly_drawn = PLAYER_DECK_MGR.draw_to_full()

            ai_cards = ENEMY_DECK_MGR.get_hand_summary()
            ai_result = TacticalAIEngine.evaluate_ai_turn(ACTIVE_ECONOMIC_MATCH, ai_side="enemy", available_cards=ai_cards)

            if ai_result.get("executed"):
                dmg = ai_result.get("damage_dealt", 0)
                if dmg > 0:
                    eff_dmg = max(0, dmg - GAME_STATE["player_armor"])
                    GAME_STATE["player_armor"] = max(0, GAME_STATE["player_armor"] - dmg)
                    GAME_STATE["player_hp"] = max(0, GAME_STATE["player_hp"] - eff_dmg)
                if ai_result.get("armor_gained", 0) > 0:
                    GAME_STATE["enemy_armor"] += ai_result["armor_gained"]
                if ai_result.get("card_id"):
                    ENEMY_DECK_MGR.play_card(ai_result["card_id"])
                ENEMY_DECK_MGR.draw_to_full()

            winner = "player" if GAME_STATE["enemy_hp"] <= 0 else ("enemy" if GAME_STATE["player_hp"] <= 0 else None)

            self._send_json({
                "success": True,
                "turn": GAME_STATE["turn"],
                "round_number": GAME_STATE["turn"],
                "player_mana": GAME_STATE["player_mana"],
                "player_hp": GAME_STATE["player_hp"],
                "enemy_hp": GAME_STATE["enemy_hp"],
                "player_hand": PLAYER_DECK_MGR.get_hand_summary(),
                "new_hand": PLAYER_DECK_MGR.get_hand_summary(),
                "newly_drawn_count": len(newly_drawn),
                "ai_action": ai_result,
                "winner": winner,
                "game_state": GAME_STATE
            })
            return

        # ── WARHAMMER & CRAFTING POST ENDPOINTS ───────────────────────────────
        # 1. Craft Item with Combinatorial Limits & Ledger Validation
        if path == '/api/crafting/craft':
            tpl_id = req_data.get("template_id", "crystal_blade")
            rarity_str = req_data.get("rarity", "rare").lower()
            try:
                rarity = ItemRarity(rarity_str)
            except ValueError:
                rarity = ItemRarity.RARE

            pref_ids = req_data.get("prefixes", [])
            suff_ids = req_data.get("suffixes", [])
            cat_dict = req_data.get("catalyst_resources", {})
            catalyst = ResourceCost(
                mana=cat_dict.get("mana", 0),
                aether_crystal=cat_dict.get("aether_crystal", 0),
                toxic_slime=cat_dict.get("toxic_slime", 0),
                amber_rune=cat_dict.get("amber_rune", 0)
            ) if cat_dict else None

            try:
                crafted = CraftingEngine.craft_item(
                    template_id=tpl_id,
                    rarity=rarity,
                    prefix_ids=pref_ids,
                    suffix_ids=suff_ids,
                    combatant=ACTIVE_ECONOMIC_MATCH.player,
                    combatant_tribe=ACTIVE_ECONOMIC_MATCH.player.tribe,
                    catalyst_resources=catalyst
                )
                GAME_STATE["ledger"].append({
                    "turn": GAME_STATE["turn"],
                    "event": "CRAFT_ITEM",
                    "desc": f"Vykovaný predmet: {crafted.name} (Rarita: {crafted.rarity.value.upper()})"
                })
                self._send_json({
                    "success": True,
                    "item": crafted.to_dict(),
                    "player_resources": EconomicLedger.get_current_balances(ACTIVE_ECONOMIC_MATCH.player)
                })
            except CraftingError as ce:
                self._send_json({"success": False, "error": str(ce)}, status=400)
            return

        # 2. Checkpoints Contest Resolution (ZoC & Victory Points)
        if path == '/api/checkpoints/capture':
            p_pos = [PLAYER_MOBILITY_UNIT.current_hex]
            e_pos = [ENEMY_MOBILITY_UNIT.current_hex]
            report = ACTIVE_CHECKPOINT_MGR.resolve_turn_contests(p_pos, e_pos)
            self._send_json({
                "success": True,
                "report": report,
                "flags": [f.to_dict() for f in ACTIVE_CHECKPOINT_MGR.flags.values()]
            })
            return

        # 3. Locational Tower Attack (High ground bonus / Low ground penalty & Ward defense)
        if path == '/api/locational/tower-shot':
            atk_pos_raw = req_data.get("attacker_pos", [0, -2, 2.5])
            def_pos_raw = req_data.get("defender_pos", [0, 0, 0.0])
            atk_pos = HexCoord3D(q=int(atk_pos_raw[0]), r=int(atk_pos_raw[1]), h=float(atk_pos_raw[2]) if len(atk_pos_raw) > 2 else 0.0)
            def_pos = HexCoord3D(q=int(def_pos_raw[0]), r=int(def_pos_raw[1]), h=float(def_pos_raw[2]) if len(def_pos_raw) > 2 else 0.0)
            base_dmg = int(req_data.get("base_damage", 3))
            base_rng = int(req_data.get("base_range", 3))
            base_ap = int(req_data.get("base_ap", 1))
            base_skl = int(req_data.get("base_skill", 3))

            shot_res = TowerLocationalCombatResolver.resolve_elevated_attack(
                attacker_pos=atk_pos,
                defender_pos=def_pos,
                base_range=base_rng,
                base_damage=base_dmg,
                base_skill=base_skl,
                base_ap=base_ap,
                active_ward=ACTIVE_TOWER_WARD
            )
            self._send_json({"success": True, "result": shot_res})
            return

        # 4. Warhammer Mobility: Normal Move & Advance Sprint
        if path == '/api/mobility/move':
            target_unit_name = req_data.get("unit", "player")
            unit = PLAYER_MOBILITY_UNIT if target_unit_name == "player" else ENEMY_MOBILITY_UNIT
            opp = [ENEMY_MOBILITY_UNIT if target_unit_name == "player" else PLAYER_MOBILITY_UNIT]
            dest = req_data.get("destination", unit.current_hex)
            m_type = req_data.get("move_type", "normal")

            if m_type == "advance":
                adv_roll = req_data.get("advance_roll")
                move_res = ACTIVE_MOBILITY_ENGINE.execute_advance_move(unit, dest, opp, advance_roll=adv_roll)
            elif m_type == "fall_back":
                move_res = ACTIVE_MOBILITY_ENGINE.execute_fall_back(unit, dest, opp)
            else:
                move_res = ACTIVE_MOBILITY_ENGINE.execute_normal_move(unit, dest, opp)

            self._send_json({"success": move_res.get("success", False), "result": move_res, "unit": unit.to_dict()})
            return

        # 5. Warhammer Mobility: 2D6 Charge
        if path == '/api/mobility/charge':
            charger_name = req_data.get("charger", "player")
            charger = PLAYER_MOBILITY_UNIT if charger_name == "player" else ENEMY_MOBILITY_UNIT
            target = ENEMY_MOBILITY_UNIT if charger_name == "player" else PLAYER_MOBILITY_UNIT
            c_roll = req_data.get("charge_roll")
            charge_res = ACTIVE_MOBILITY_ENGINE.execute_charge_move(charger, target, charge_roll_2d6=c_roll)
            self._send_json({"success": charge_res.get("success", False), "result": charge_res, "charger": charger.to_dict()})
            return

        # 6. Statistical Combat Probability Simulation & Monte Carlo
        if path == '/api/statistics/simulate-combat':
            atks = int(req_data.get("attacks", 4))
            skl = int(req_data.get("skill", 3))
            str_val = int(req_data.get("strength", 4))
            tgh = int(req_data.get("toughness", 4))
            dmg = float(req_data.get("damage_per_wound", 2))
            sv = int(req_data.get("armor_save", 4))
            ap_val = int(req_data.get("armor_penetration", 1))
            inv = req_data.get("invulnerable_save")
            iters = int(req_data.get("iterations", 200))

            expected = calculate_expected_damage(
                attacks=atks, skill=skl, strength=str_val, toughness=tgh,
                damage_per_wound=dmg, armor_save=sv, armor_penetration=ap_val,
                invulnerable_save=int(inv) if inv else None
            )
            sim = CombatSimulationModel()
            mc = sim.run_monte_carlo(
                iterations=iters, attacks=atks, skill=skl, strength=str_val, toughness=tgh,
                damage_per_wound=int(dmg), armor_save=sv, armor_penetration=ap_val,
                invulnerable_save=int(inv) if inv else None
            )
            self._send_json({"success": True, "analytical_expected": expected, "monte_carlo": mc})
            return

        # ── ENVIRONMENTAL MATRICES & HAZARD TIMERS ─────────────────────────
        if path == '/api/environment/matrix/apply_hazard':
            cq = int(req_data.get("center_q", 0))
            cr = int(req_data.get("center_r", 0))
            fp = req_data.get("footprint_type", "point")
            haz_str = req_data.get("hazard", "toxic_acid")
            dur = int(req_data.get("duration", 2))
            pot = int(req_data.get("potency", 1))
            cov = float(req_data.get("cover_delta", 0.0))
            res_delta = int(req_data.get("resonance_delta", 0))
            try:
                haz_enum = EnvironmentalHazardType(haz_str)
            except ValueError:
                haz_enum = EnvironmentalHazardType.TOXIC_ACID

            res = ACTIVE_ENV_MATRIX.apply_action_footprint(
                center_q=cq, center_r=cr, footprint_type=fp,
                hazard=haz_enum, duration=dur, potency=pot,
                cover_delta=cov, resonance_delta=res_delta
            )
            self._send_json({"success": True, "result": res, "matrix_state": ACTIVE_ENV_MATRIX.to_matrix_payload()})
            return

        if path == '/api/environment/matrix/tick':
            units = req_data.get("unit_positions", {
                "player": PLAYER_MOBILITY_UNIT.current_hex,
                "enemy": ENEMY_MOBILITY_UNIT.current_hex
            })
            decay_res = ACTIVE_ENV_MATRIX.process_round_decay_and_ticks(units)
            self._send_json({"success": True, "decay_summary": decay_res, "matrix_state": ACTIVE_ENV_MATRIX.to_matrix_payload()})
            return

        # ── TRIGONOMETRIC NPC KINEMATICS & SECTOR AI ───────────────────────
        if path == '/api/tactics/trigonometrics/sector':
            atk_coord = tuple(req_data.get("attacker_coord", [0, -2]))
            tgt_coord = tuple(req_data.get("target_coord", [0, 0]))
            facing = float(req_data.get("target_facing_angle", 0.0))
            sector, delta_deg = TacticalTrigonometry.determine_engagement_sector(atk_coord, tgt_coord, facing)
            facing_toward_tgt = TacticalTrigonometry.calculate_facing_angle(atk_coord, tgt_coord)
            los = TacticalTrigonometry.check_line_of_sight(atk_coord, tgt_coord, ACTIVE_ENV_MATRIX)
            self._send_json({
                "success": True,
                "sector": sector.value,
                "delta_angle_degrees": delta_deg,
                "attacker_facing_rad": round(facing_toward_tgt, 4),
                "line_of_sight": los
            })
            return

        if path == '/api/tactics/trigonometrics/evaluate_turn':
            ai_tribe_str = req_data.get("tribe", "toxic").lower()
            try:
                ai_tribe = Tribe(ai_tribe_str)
            except ValueError:
                ai_tribe = Tribe.TOXIC

            npc_c = tuple(req_data.get("npc_coord", ENEMY_MOBILITY_UNIT.current_hex))
            tgt_c = tuple(req_data.get("target_coord", PLAYER_MOBILITY_UNIT.current_hex))
            npc_hp = int(req_data.get("npc_hp", GAME_STATE["enemy_hp"]))
            npc_mana = int(req_data.get("npc_mana", ACTIVE_ECONOMIC_MATCH.enemy.mana))
            tgt_hp = int(req_data.get("target_hp", GAME_STATE["player_hp"]))
            tgt_facing = float(req_data.get("target_facing_angle", 0.0))
            cards = req_data.get("cards", ENEMY_DECK_MGR.hand if ENEMY_DECK_MGR else [])

            ai_agent = MatrixTacticalAI(tribe=ai_tribe, env_matrix=ACTIVE_ENV_MATRIX)
            decision = ai_agent.evaluate_tactical_position_and_action(
                npc_coord=npc_c, npc_hp=npc_hp, npc_mana=npc_mana,
                target_coord=tgt_c, target_hp=tgt_hp,
                target_facing_angle=tgt_facing,
                available_cards=cards
            )
            self._send_json({"success": True, "decision": decision})
            return

        # ── COMBINATORIAL STATISTICAL MODEL & CARD ALGEBRA ─────────────────
        if path == '/api/cards/statistics/hypergeometric':
            N = int(req_data.get("deck_size", 43))
            K = int(req_data.get("target_cards_in_deck", 5))
            n = int(req_data.get("cards_drawn", 5))
            k = int(req_data.get("desired_hits", 1))

            p_exact = HypergeometricCardStatistics.probability_draw_exact_k(N, K, n, k)
            p_at_least = HypergeometricCardStatistics.probability_draw_at_least_k(N, K, n, k)
            e_turn = HypergeometricCardStatistics.expected_turn_to_draw(N, K, opening_hand=n)
            self._send_json({
                "success": True,
                "deck_size_N": N,
                "target_copies_K": K,
                "sample_drawn_n": n,
                "desired_hits_k": k,
                "probability_exact_k": p_exact,
                "probability_at_least_k": p_at_least,
                "expected_turn_to_draw": e_turn
            })
            return

        if path == '/api/cards/combat/resolve_algebra':
            card_obj = req_data.get("card", {"name": "Test Strike", "hp_delta": -2, "attack_type": "melee"})
            atk_stats = req_data.get("attacker_stats", {"hp": 6, "armor": 0, "skill": 3})
            def_stats = req_data.get("defender_stats", {"hp": 6, "armor": 1, "toughness": 4})
            sec_str = req_data.get("sector", "front")
            elev_rad = float(req_data.get("elevation_pitch_rad", 0.0))
            hit_roll = req_data.get("d6_hit_roll")
            save_roll = req_data.get("d6_save_roll")

            try:
                sec_enum = CombatSector(sec_str)
            except ValueError:
                sec_enum = CombatSector.FRONT

            res = CardCombatAlgebra.resolve_card_action(
                card=card_obj,
                attacker_stats=atk_stats,
                defender_stats=def_stats,
                sector=sec_enum,
                elevation_pitch_rad=elev_rad,
                d6_hit_roll=int(hit_roll) if hit_roll is not None else None,
                d6_save_roll=int(save_roll) if save_roll is not None else None
            )
            self._send_json({"success": True, "resolution": res})
            return

        # ── ARTILLERY, MORTAR & GUN BALLISTICS CALCULATOR ──────────────────
        if path == '/api/calculator/ballistics':
            w_type_str = req_data.get("weapon_type", "mortar_indirect")
            try:
                w_type = WeaponType(w_type_str)
            except ValueError:
                w_type = WeaponType.MORTAR_INDIRECT

            atk_c = tuple(req_data.get("attacker_coord", [0, -2]))
            atk_h = float(req_data.get("attacker_height", 2.0))
            tgt_c = tuple(req_data.get("target_coord", [0, 1]))
            tgt_h = float(req_data.get("target_height", 0.0))
            tgt_hp = int(req_data.get("target_hp", 6))
            tgt_arm = int(req_data.get("target_armor", 2))
            base_dmg = int(req_data.get("base_damage", 3))
            base_ap = int(req_data.get("base_ap", 1))

            ench_str = req_data.get("enchantment", "none")
            try:
                ench_type = EnchantmentType(ench_str)
            except ValueError:
                ench_type = EnchantmentType.NONE

            has_catalyst = bool(req_data.get("has_amber_catalyst", False))
            combo_action = req_data.get("combo_secondary_action")
            hit_roll = req_data.get("d6_hit_roll")

            calc_res = ArtilleryCombatCalculator.calculate_attack_resolution(
                weapon_type=w_type,
                attacker_coord=atk_c,
                attacker_height=atk_h,
                target_coord=tgt_c,
                target_height=tgt_h,
                target_hp=tgt_hp,
                target_armor=tgt_arm,
                base_damage=base_dmg,
                base_ap=base_ap,
                enchantment=ench_type,
                has_amber_catalyst=has_catalyst,
                combo_secondary_action=combo_action,
                d6_hit_roll=int(hit_roll) if hit_roll is not None else None,
                env_matrix=ACTIVE_ENV_MATRIX
            )
            self._send_json({"success": True, "result": calc_res})
            return

        if path == '/api/calculator/targeting_trig':
            atk_c = tuple(req_data.get("attacker_coord", [0, -2]))
            atk_h = float(req_data.get("attacker_height", 2.0))
            tgt_c = tuple(req_data.get("target_coord", [0, 1]))
            tgt_h = float(req_data.get("target_height", 0.0))
            w_type_str = req_data.get("weapon_type", "mortar_indirect")
            try:
                w_type = WeaponType(w_type_str)
            except ValueError:
                w_type = WeaponType.MORTAR_INDIRECT

            trig_profile = TrigonometricTargetingSystem.calculate_targeting_trigonometry(
                origin_coord=atk_c,
                origin_height=atk_h,
                target_coord=tgt_c,
                target_height=tgt_h,
                weapon_type=w_type
            )
            self._send_json({"success": True, "trigonometry": trig_profile})
            return

        if path == '/api/calculator/action_combo':
            act_a = req_data.get("action_a", "spotter_beacon")
            act_b = req_data.get("action_b", "mortar_barrage")
            combo_eval = ActionCombinationEngine.evaluate_combo(act_a, act_b)
            self._send_json({"success": True, "combo": combo_eval})
            return

        # ── TACTICAL DUELS & MCP BRIDGE (POST) ────────────────────────────────
        if path == '/api/tactical/uncovered_and_mortar':
            combatants = req_data.get("combatants", [])
            cover_map = req_data.get("cover_map", {})
            elevation_map = req_data.get("elevation_map", {})
            attacker_pos = tuple(req_data.get("attacker_pos", [0.0, 0.0, 0.0]))
            target_pos = tuple(req_data.get("target_pos", [40.0, 0.0, 40.0]))
            salvo_count = int(req_data.get("salvo_count", 3))
            elevation_angle_deg = float(req_data.get("elevation_angle_deg", 65.0))
            cadence_interval = float(req_data.get("cadence_interval_sec", 0.40))

            res = GLOBAL_MCP_BRIDGE.calculate_uncovered_and_mortar_cadence(
                combatants=combatants,
                cover_map=cover_map,
                elevation_map=elevation_map,
                attacker_pos=attacker_pos,
                target_pos=target_pos,
                salvo_count=salvo_count,
                elevation_angle_deg=elevation_angle_deg,
                cadence_interval_sec=cadence_interval
            )
            self._send_json(res)
            return

        if path == '/api/tactical/gnome_duel_frame':
            char_left = req_data.get("character_left", {})
            char_right = req_data.get("character_right", {})
            aspect_ratio = req_data.get("aspect_ratio", "16:9")
            window_mode = req_data.get("window_mode", "cinematic_zoom")
            projectiles = req_data.get("incoming_projectiles", [])
            zoom = float(req_data.get("zoom_level", 1.85))

            frame = GLOBAL_MCP_BRIDGE.compose_gnome_duel_frame(
                character_left=char_left,
                character_right=char_right,
                aspect_ratio=aspect_ratio,
                window_mode=window_mode,
                incoming_projectiles=projectiles,
                zoom_level=zoom
            )
            self._send_json({"success": True, "frame": frame})
            return

        if path == '/api/tactical/ubisoft_bullet_time':
            melee_actor = req_data.get("melee_actor_id", "duelist_a")
            m_start = tuple(req_data.get("melee_start", [0.0, 0.0, 0.0]))
            m_end = tuple(req_data.get("melee_end", [3.0, 0.0, 0.0]))
            p_start = tuple(req_data.get("projectile_start", [2.5, 4.5, 0.0]))
            p_end = tuple(req_data.get("projectile_end", [2.5, 0.0, 0.0]))
            ar = req_data.get("aspect_ratio", "16:9")

            bt_res = GLOBAL_MCP_BRIDGE.execute_ubisoft_bullet_time(
                melee_actor_id=melee_actor,
                melee_start=m_start,
                melee_end=m_end,
                projectile_start=p_start,
                projectile_end=p_end,
                aspect_ratio=ar
            )
            self._send_json({"success": True, "bullet_time": bt_res})
            return

        if path == '/api/tactical/dopamine_cadence':
            timestamps = req_data.get("timestamps", [])
            card_ids = req_data.get("card_ids", [])
            combo_name = req_data.get("combo_name")

            cadence_eval = DopamineCadenceEngine.evaluate_timing_chain(timestamps, card_ids)
            combo_result = None
            if combo_name:
                units = req_data.get("units", [])
                st = DopamineCombatState(units)
                st.update_partitions(
                    allies=req_data.get("allies", []),
                    enemies=req_data.get("enemies", []),
                    uncovered=req_data.get("uncovered", []),
                    crowd_controlled=req_data.get("crowd_controlled", []),
                    warded=req_data.get("warded", []),
                    supernatural=req_data.get("supernatural", [])
                )
                combo_result = DopamineCadenceEngine.execute_matrix_shatter_combo(st, combo_name, cadence_eval)

            self._send_json({
                "success": True,
                "cadence": cadence_eval,
                "combo_result": combo_result
            })
            return

        if path == '/api/mcp/stream_frame':
            duel_id = req_data.get("duel_id", "default_duel")
            sender_id = req_data.get("sender_id", "p1")
            recipient_id = req_data.get("recipient_id", "p2")
            frame_payload = req_data.get("frame_payload", {})

            stream_res = GLOBAL_MCP_BRIDGE.stream_frame_to_opponent(
                duel_id=duel_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                frame_payload=frame_payload
            )
            self._send_json(stream_res)
            return

        if path == '/api/tactical/event_rewards':
            t_lvl = int(req_data.get("threat_level", 1))
            rem_hp = int(req_data.get("remaining_hp", 6))
            cad_score = float(req_data.get("cadence_score", 1.0))
            bt_count = int(req_data.get("bullet_time_count", 0))
            unc_kills = int(req_data.get("uncovered_kills", 0))
            p_id = req_data.get("player_id", "player_hero_1")

            rewards = GLOBAL_MCP_BRIDGE.evaluate_threat_rewards(
                threat_level=t_lvl,
                remaining_hp=rem_hp,
                cadence_score=cad_score,
                bullet_time_count=bt_count,
                uncovered_kills=unc_kills,
                player_id=p_id
            )
            self._send_json({"success": True, "rewards": rewards})
            return

        # ── TUNNEL STREAM CRYPTO & VPN EVALUATION (POST) ───────────────────────
        if path == '/api/tunnel/encrypt_frame':
            frame = req_data.get("frame", {})
            key = req_data.get("tunnel_key_hex") or TunnelStreamCipher.generate_tunnel_key()
            sender = req_data.get("sender_id", "local_node")
            tun_id = req_data.get("tunnel_id", "vpn_tunnel_default")

            envelope = TunnelStreamCipher.encrypt_frame_packet(
                raw_frame=frame,
                tunnel_key_hex=key,
                sender_id=sender,
                tunnel_id=tun_id
            )
            self._send_json({"success": True, "envelope": envelope, "tunnel_key_hex": key})
            return

        if path == '/api/tunnel/decrypt_frame':
            env = req_data.get("envelope", {})
            key = req_data.get("tunnel_key_hex", "")
            try:
                decrypted = TunnelStreamCipher.decrypt_frame_packet(env, key)
                self._send_json({"success": True, "frame": decrypted})
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=400)
            return

        if path == '/api/tunnel/evaluate_vpn':
            ping = float(req_data.get("ping_rtt_ms", 35.0))
            jit = float(req_data.get("jitter_ms", 10.0))
            loss = float(req_data.get("packet_loss_percent", 0.0))
            vpn = bool(req_data.get("is_vpn_detected", False))

            metrics = VpnTunnelPenaltyEvaluator.evaluate_tunnel_metrics(
                ping_rtt_ms=ping,
                jitter_ms=jit,
                packet_loss_percent=loss,
                is_vpn_detected=vpn
            )
            self._send_json({"success": True, "metrics": metrics})
            return

        # ── DEATH AD PENALTY & REWARDED ADS (POST) ───────────────────────────
        if path == '/api/ads/evaluate_death_penalty':
            p_id = req_data.get("player_id", "default_hero")
            wins = int(req_data.get("rounds_won", 0))
            plus = bool(req_data.get("has_plus_membership", False))
            lifespan = float(req_data.get("lifespan_sec", 120.0))
            last_ad = float(req_data.get("last_ad_prompt_timestamp", 0.0))
            deaths = int(req_data.get("consecutive_deaths", 1))

            penalty_eval = DeathPenaltyEvaluator.evaluate_death_penalty(
                player_id=p_id,
                rounds_won=wins,
                has_plus_membership=plus,
                lifespan_sec=lifespan,
                last_ad_prompt_timestamp=last_ad,
                consecutive_deaths=deaths
            )
            self._send_json({"success": True, "evaluation": penalty_eval})
            return

        if path == '/api/ads/record_watched':
            p_id = req_data.get("player_id", "default_hero")
            ad_res = GLOBAL_AD_QUOTA_MGR.record_watched_ad(p_id)
            self._send_json(ad_res)
            return

        # ── CHESTS & AUCTION ACTIONS (POST) ──────────────────────────────────
        if path == '/api/market/open_chest':
            chest_tier = req_data.get("chest_tier", ChestTier.BRONZE_SCAVENGER.value)
            try:
                loot = ChestLootResolver.open_chest(chest_tier)
                self._send_json({"success": True, "loot_result": loot})
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=400)
            return

        if path == '/api/market/auctions/create':
            seller = req_data.get("seller_id", "default_seller")
            item = req_data.get("item_id", "artifact_chrono_shard_of_ubisoft")
            i_type = req_data.get("item_type", "artifact")
            start_bid = int(req_data.get("starting_bid_nuggets", 100))
            buyout = int(req_data.get("buyout_nuggets", 300))
            duration = int(req_data.get("duration_sec", 3600))

            try:
                lot = GLOBAL_AUCTION_HOUSE.create_auction_lot(
                    seller_id=seller,
                    item_id=item,
                    item_type=i_type,
                    starting_bid_nuggets=start_bid,
                    buyout_nuggets=buyout,
                    duration_sec=duration
                )
                self._send_json({"success": True, "lot": lot})
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=400)
            return

        if path == '/api/market/auctions/bid':
            auc_id = req_data.get("auction_id", "")
            bidder = req_data.get("bidder_id", "default_bidder")
            amount = int(req_data.get("bid_amount", 100))
            avail = int(req_data.get("bidder_nuggets_available", 1000))

            res = GLOBAL_AUCTION_HOUSE.place_bid(
                auction_id=auc_id,
                bidder_id=bidder,
                bid_amount=amount,
                bidder_nuggets_available=avail
            )
            self._send_json(res)
            return

        if path == '/api/market/auctions/buyout':
            auc_id = req_data.get("auction_id", "")
            buyer = req_data.get("buyer_id", "default_buyer")
            avail = int(req_data.get("buyer_nuggets_available", 1000))

            res = GLOBAL_AUCTION_HOUSE.buyout_auction(
                auction_id=auc_id,
                buyer_id=buyer,
                buyer_nuggets_available=avail
            )
            self._send_json(res)
            return

        # ── TACTICAL GRADIENTS, VECTOR ARROWS & DYNAMIC WEAPONS (POST) ───────
        if path == '/api/tactical/gradients':
            h_map = req_data.get("height_map", {"0_0": 0.0, "1_0": 1.5, "-1_0": -0.5, "0_1": 0.8, "0_-1": -0.2})
            center_hex = tuple(req_data.get("center_hex", [0, 0]))
            hex_spacing = float(req_data.get("hex_spacing", 1.732))
            grad_info = TacticalGradientEngine.calculate_terrain_gradient(h_map, center_hex, hex_spacing)

            # Dynamic Weapon Scale if requested
            weapon_scale = None
            if "base_range" in req_data:
                b_rng = float(req_data.get("base_range", 3.0))
                b_dmg = int(req_data.get("base_damage", 3))
                atk_elev = float(req_data.get("attacker_elevation", 2.0))
                tgt_elev = float(req_data.get("target_elevation", 0.0))
                dot_dir = float(req_data.get("gradient_dot_fire_dir", 0.0))
                is_mort = bool(req_data.get("is_mortar", False))
                weapon_scale = TacticalGradientEngine.calculate_dynamic_weapon_scale(
                    base_range=b_rng,
                    base_damage=b_dmg,
                    attacker_elevation=atk_elev,
                    target_elevation=tgt_elev,
                    gradient_dot_fire_dir=dot_dir,
                    is_mortar=is_mort
                )

            # Movement Vector Arrow if requested
            vector_arrow = None
            if "start_pos" in req_data and "end_pos" in req_data:
                s_pos = tuple(req_data.get("start_pos", [0.0, 0.0, 0.0]))
                e_pos = tuple(req_data.get("end_pos", [3.0, 1.2, 0.0]))
                vector_arrow = TacticalGradientEngine.generate_movement_vector_arrow(s_pos, e_pos, grad_info)

            self._send_json({
                "success": True,
                "gradient": grad_info,
                "weapon_scale": weapon_scale,
                "movement_arrow": vector_arrow
            })
            return

        # ── PINNED ORBITAL SPELLCRAFTING & AURA FIELDS (POST) ────────────────
        if path == '/api/tactical/orbital_spell':
            caster_pos = tuple(req_data.get("caster_pos", [0.0, 0.0, 0.0]))
            t = float(req_data.get("time_sec", 1.0))
            r_maj = float(req_data.get("orbit_radius_major", 2.4))
            r_min = float(req_data.get("orbit_radius_minor", 1.8))
            inc_deg = float(req_data.get("inclination_deg", 22.5))
            harmony_key = req_data.get("harmony_key", "fibonacci_triad")
            c_tgh = int(req_data.get("caster_toughness", 15))
            c_ref = int(req_data.get("caster_reflexes", 20))

            bodies = PinnedOrbitalSpellcraftingEngine.calculate_celestial_orbit_positions(
                caster_pos=caster_pos,
                t=t,
                orbit_radius_major=r_maj,
                orbit_radius_minor=r_min,
                inclination_deg=inc_deg,
                harmony_key=harmony_key
            )
            aura = PinnedOrbitalSpellcraftingEngine.evaluate_aura_resonance_field(
                orbiting_bodies=bodies,
                caster_toughness=c_tgh,
                caster_reflexes=c_ref
            )
            self._send_json({
                "success": True,
                "caster_pos": caster_pos,
                "harmony_key": harmony_key,
                "orbiting_bodies": bodies,
                "aura_resonance_field": aura
            })
            return

        # ── THE WEST DUEL RESOLUTION (POST) ──────────────────────────────────
        if path == '/api/duel/the_west_round':
            atk_stats = req_data.get("attacker_stats", {
                "toughness": 12, "reflexes": 15, "aim": 28, "dodge": 18,
                "appearance": 32, "tactics": 10, "mobility": 16
            })
            def_stats = req_data.get("defender_stats", {
                "toughness": 28, "reflexes": 26, "aim": 14, "dodge": 10,
                "appearance": 8, "tactics": 20, "mobility": 12
            })
            atk_zone = req_data.get("attack_zone", DuelTargetZone.HEAD.value)
            def_stance = req_data.get("defense_stance", DuelDodgeStance.STAND_FIRM.value)
            w_cat_str = req_data.get("weapon_type", DuelWeaponCategory.RANGED_PROJECTILE.value)
            w_cat = DuelWeaponCategory(w_cat_str) if w_cat_str in [e.value for e in DuelWeaponCategory] else DuelWeaponCategory.RANGED_PROJECTILE
            b_dmg = int(req_data.get("base_weapon_damage", 12))
            def_hp = int(req_data.get("defender_current_hp", 6))

            round_res = TheWestDuelAlgebra.resolve_duel_round(
                attacker_stats=atk_stats,
                defender_stats=def_stats,
                attack_zone=atk_zone,
                defense_stance=def_stance,
                weapon_type=w_cat,
                base_weapon_damage=b_dmg,
                defender_current_hp=def_hp
            )
            self._send_json({"success": True, "duel_round": round_res})
            return

        # ── AD EXCHANGE PROTOCOL (POST) ──────────────────────────────────────
        if path == '/api/ad_protocol/auction_bid':
            p_id = req_data.get("player_id", "player_hero_1")
            p_tribe = req_data.get("player_tribe", "crystal")
            fmt_str = req_data.get("ad_format", AdFormat.REWARDED_VIDEO.value)
            fmt = AdFormat(fmt_str) if fmt_str in [f.value for f in AdFormat] else AdFormat.REWARDED_VIDEO
            auction_res = GLOBAL_AD_EXCHANGE.run_ad_auction(p_id, p_tribe, fmt)
            self._send_json(auction_res, status=200 if auction_res.get("success") else 400)
            return

        if path == '/api/ad_protocol/verify_impression':
            ticket = req_data.get("ticket", {})
            pov = req_data.get("proof_of_viewing", {})
            verify_res = GLOBAL_AD_EXCHANGE.verify_proof_of_viewing_and_settle(ticket, pov)
            self._send_json(verify_res, status=200 if verify_res.get("success") else 400)
            return

        # ── CLUSTER MONITORING (POST) ────────────────────────────────────────
        if path == '/api/cluster/monitoring/heartbeat':
            node_id = req_data.get("node_id", "edge_gw_alpha_01")
            cpu = float(req_data.get("cpu_percent", 15.0))
            ram = float(req_data.get("ram_percent", 22.0))
            qps = float(req_data.get("qps", 100.0))
            lat = float(req_data.get("latency_ms", 4.0))
            updated = GLOBAL_MONITORING_CLUSTER.record_node_heartbeat(node_id, cpu, ram, qps, lat)
            self._send_json({"success": updated, "node_id": node_id})
            return

        if path == '/api/cluster/monitoring/drain':
            node_id = req_data.get("node_id", "")
            drained = GLOBAL_MONITORING_CLUSTER.drain_node(node_id)
            self._send_json({"success": drained, "node_id": node_id})
            return

        # ── FIREWALL IP MANAGEMENT (POST) ────────────────────────────────────
        if path == '/api/security/firewall/manage_ip':
            target_ip = req_data.get("ip", "")
            action = req_data.get("action", "blacklist")
            if action == "whitelist":
                GLOBAL_FIREWALL.add_to_whitelist(target_ip)
            elif action == "unban":
                GLOBAL_FIREWALL.remove_from_blacklist(target_ip)
            else:
                GLOBAL_FIREWALL.add_to_blacklist(target_ip)
            self._send_json({
                "success": True,
                "ip": target_ip,
                "action": action,
                "firewall": GLOBAL_FIREWALL.get_firewall_status()
            })
            return

        # ── JANET BOT TRANSMITTER & TRAVEL (POST) ────────────────────────────
        if path == '/api/transmitter/step':
            delta = tuple(req_data.get("delta_vector", [1.0, 0.0, 0.0]))
            step_res = GLOBAL_BOT_TRANSMITTER.step_forward(delta)
            self._send_json(step_res)
            return

        if path == '/api/transmitter/rewind':
            rewind_res = GLOBAL_BOT_TRANSMITTER.rewind_step()
            self._send_json(rewind_res, status=200 if rewind_res.get("success") else 400)
            return

        if path == '/api/transmitter/teleport':
            target_pos = tuple(req_data.get("target_position", [0.0, 0.0, 0.0]))
            target_plane_str = req_data.get("plane")
            plane_enum = CosmologicalPlane(target_plane_str) if target_plane_str in [p.value for p in CosmologicalPlane] else None
            tp_res = GLOBAL_BOT_TRANSMITTER.teleport_bot(target_pos, plane_enum)
            self._send_json(tp_res)
            return

        if path == '/api/transmitter/stimulus':
            stim = req_data.get("stimulus_name", "AETHER_SURGE")
            toggle_gm = bool(req_data.get("toggle_god_mode", False))
            if toggle_gm:
                is_gm = GLOBAL_BOT_TRANSMITTER.toggle_god_mode()
                self._send_json({"success": True, "god_mode": is_gm, "transmitter": GLOBAL_BOT_TRANSMITTER.to_dict()})
            else:
                stim_res = GLOBAL_BOT_TRANSMITTER.inject_stimulus(stim)
                self._send_json(stim_res)
            return

        # ── NOCTURNAL SKY & ATMOSPHERIC SPIRITS (POST) ───────────────────────
        if path == '/api/nocturnal/atmosphere':
            t_sec = float(req_data.get("celestial_time_sec", time.time()))
            lunar_ph = float(req_data.get("lunar_phase", 0.50))
            sky_res = NocturnalAtmosphereEngine.calculate_nocturnal_sky_optics(t_sec, lunar_ph)
            self._send_json({"success": True, "nocturnal_atmosphere": sky_res})
            return

        # ── ENTROPY WEATHER & CATASTROPHES (POST) ────────────────────────────
        if path == '/api/environment/weather_entropy':
            dp = float(req_data.get("pressure_delta", 10.0))
            hum = float(req_data.get("humidity_pct", 50.0))
            wind = float(req_data.get("wind_shear_mps", 15.0))
            weather_res = EntropyWeatherEngine.calculate_weather_entropy(dp, hum, wind)
            self._send_json({"success": True, "weather": weather_res})
            return

        # ── 20 MALE / 20 FEMALE ARCHETYPES & 120 HELPERS (POST) ──────────────
        if path == '/api/archetypes/composite_build':
            hero_id = req_data.get("hero_id", "m_cold_15")
            helper_idx = int(req_data.get("helper_index", 1))
            race_id = req_data.get("race_id", "crystal")
            slider_adj = req_data.get("slider_adjustments", {})
            syn_scale = float(req_data.get("synergy_scale", 1.0))

            composite = ArchetypeAndHelperEngine.calculate_composite_build(
                hero_id=hero_id,
                helper_index=helper_idx,
                race_id=race_id,
                slider_adjustments=slider_adj,
                synergy_scale=syn_scale
            )
            self._send_json({"success": True, "composite_build": composite})
            return

        if path == '/api/archetypes/duel_round':
            hero_id = req_data.get("hero_id", "m_cold_15")
            helper_idx = int(req_data.get("helper_index", 1))
            race_id = req_data.get("race_id", "crystal")
            slider_adj = req_data.get("slider_adjustments", {})
            syn_scale = float(req_data.get("synergy_scale", 1.0))

            target_zone = req_data.get("attack_zone", "head")
            defense_stance = req_data.get("defense_stance", "stand_firm")
            w_cat_str = req_data.get("weapon_type", "ranged")
            w_cat = DuelWeaponCategory.COLD_MELEE if w_cat_str == "cold_melee" else DuelWeaponCategory.RANGED_PROJECTILE
            base_dmg = int(req_data.get("base_damage", 4))
            def_hp = int(req_data.get("defender_current_hp", 6))
            def_stats = req_data.get("defender_stats", None)

            composite = ArchetypeAndHelperEngine.calculate_composite_build(
                hero_id=hero_id,
                helper_index=helper_idx,
                race_id=race_id,
                slider_adjustments=slider_adj,
                synergy_scale=syn_scale
            )

            duel_res = ArchetypeAndHelperEngine.resolve_interactive_duel_round(
                composite_build=composite,
                defender_stats=def_stats,
                attack_zone=target_zone,
                defense_stance=defense_stance,
                weapon_type=w_cat,
                base_damage=base_dmg,
                defender_current_hp=def_hp
            )

            self._send_json({
                "success": True,
                "composite_build": composite,
                "duel_result": duel_res
            })
            return

        # ── COMBAT IMMUNITY & RESISTANCE (POST) ──────────────────────────────
        if path == '/api/combat/immunity_check':
            race_id = req_data.get("race_id", "crystal")
            ailment_str = req_data.get("ailment_type", "poison_acid")
            raw_pot = int(req_data.get("raw_potency", 3))
            gear_res = req_data.get("gear_resists", {})
            ward_hp = int(req_data.get("active_ward", 6))

            try:
                ailment_enum = DamageAilmentType(ailment_str)
            except ValueError:
                ailment_enum = DamageAilmentType.POISON_ACID

            profile = ImmunitySystemEngine.build_hero_immunity_profile(race_id, gear_res, ward_hp)
            mitigation_res = ImmunitySystemEngine.resolve_ailment_attack(ailment_enum, raw_pot, profile)

            self._send_json({
                "success": True,
                "immunity_profile": profile,
                "ailment_mitigation": mitigation_res
            })
            return

        # ── EQUIPMENT OPTICS & ZOOM (POST) ──────────────────────────────────
        if path == '/api/equipment/optics_zoom':
            tier = int(req_data.get("gear_tier", 1))
            dist = int(req_data.get("target_distance_hex", 3))
            optics_res = EquipmentZoomOpticsEngine.calculate_zoom_optics(tier, dist)
            self._send_json({"success": True, "optics": optics_res})
            return

        # ── PLUS INVENTORY EXPANSION (POST) ─────────────────────────────────
        if path == '/api/inventory/plus_slots':
            action = req_data.get("action", "view")
            if action == "add_token":
                cnt = int(req_data.get("count", 1))
                GLOBAL_PLUS_INVENTORY.add_plus_token(cnt)
            elif action == "store":
                s_idx = int(req_data.get("slot_index", 0))
                i_data = req_data.get("item_data", {"name": "Aéterový Kryštál"})
                GLOBAL_PLUS_INVENTORY.store_item(s_idx, i_data)

            self._send_json({
                "success": True,
                "inventory": GLOBAL_PLUS_INVENTORY.to_dict()
            })
            return

        # ── PREREQUISITES VALIDATION (POST) ──────────────────────────────────
        if path == '/api/system/prerequisites_check':
            hero_prof = req_data.get("hero_profile", {})
            hero_id = req_data.get("hero_id")
            if hero_id and not hero_prof:
                hero_prof = ArchetypeAndHelperEngine.get_character(hero_id) or {}
            reqs = req_data.get("requirements", {})
            val_res = PrerequisitesValidator.validate_prerequisites(hero_prof, reqs)
            self._send_json({"success": True, "prerequisites": val_res})
            return

        # ── ART AUCTIONS (POST) ──────────────────────────────────────────────
        if path == '/api/art/auctions/create':
            title = req_data.get("title", "Nová Aéterická Maľba")
            artist = req_data.get("artist", "Krystal Generative Artisan")
            style = req_data.get("style", "Aetheric High-Renaissance")
            rarity = req_data.get("rarity", "Legendary")
            starting_bid = int(req_data.get("starting_bid", 100))
            buyout_price = int(req_data.get("buyout_price", 400))
            resolution = req_data.get("resolution", "1920x1080")
            img_url = req_data.get("image_url", "/api/assets/paintings/default.png")

            lot = GLOBAL_PAINTING_AUCTIONS.create_lot(
                title=title,
                artist=artist,
                style=style,
                rarity=rarity,
                starting_bid=starting_bid,
                buyout_price=buyout_price,
                resolution=resolution,
                image_url=img_url
            )
            self._send_json({"success": True, "lot": lot})
            return

        if path == '/api/art/auctions/bid':
            lot_id = req_data.get("lot_id", "")
            bidder = req_data.get("bidder", "Anonymous Collector")
            amount = int(req_data.get("amount", 0))

            res = GLOBAL_PAINTING_AUCTIONS.place_bid(lot_id=lot_id, bidder=bidder, amount=amount)
            if res.get("success"):
                self._send_json(res)
            else:
                self._send_json(res, status=400)
            return

        # ── AETHER ORDINALS PROTOCOL (POST) ──────────────────────────────────
        if path == '/api/ordinals/inscribe':
            art_lot_id = req_data.get("art_lot_id")
            content_payload = req_data.get("content_payload", "Krystal Ordinal Genesis Inscription")
            content_type = req_data.get("content_type", "text/plain;charset=utf-8")
            owner_address = req_data.get("owner_address", "bc1p_krystal_stack_aether_inscription")

            insc = GLOBAL_ORDINALS_PROTOCOL.inscribe(
                art_lot_id=art_lot_id,
                content_payload=content_payload,
                content_type=content_type,
                owner_address=owner_address
            )
            exported = GLOBAL_ORDINALS_PROTOCOL.export_ordinal(insc["id"])
            self._send_json({
                "success": True,
                "inscription": insc,
                "exported_ordinal": exported
            })
            return

        # ── CONTENT REPLAY & LEVEL FRAGMENTATION (POST) ──────────────────────
        if path == '/api/replay/record_level':
            m_id = req_data.get("match_id", "match_alpha")
            lvl_name = req_data.get("level_name", "Aréna Zmrazených Špicov")
            lvl_idx = int(req_data.get("level_idx", 1))
            seed = req_data.get("seed", 42)
            hero_cnt = int(req_data.get("hero_count", 4))
            events = req_data.get("events", [{"type": "level_init", "tick": 0}])
            v_layers = req_data.get("visual_layers", {"lighting": "dusk", "fog_density": 0.4})

            segment = GLOBAL_CONTENT_REPLAY.record_level_segment(
                match_id=m_id,
                level_name=lvl_name,
                level_idx=lvl_idx,
                level_seed=seed,
                hero_count=hero_cnt,
                action_events=events,
                visual_layers=v_layers
            )
            self._send_json({"success": True, "level_segment": segment})
            return

        # ── HERO MATRICES & VARIABLE SAMPLING FREQUENCY (POST) ───────────────
        if path == '/api/ml/hero_matrices':
            hero_cnt = int(req_data.get("hero_count", 1))
            target_dim = req_data.get("target_dims", "all")
            intensity = float(req_data.get("battle_intensity", 0.5))

            dimensions = ["5x2", "4x5", "30x20", "90x120"] if target_dim == "all" else [target_dim]
            matrices_out = {}
            for dim in dimensions:
                if dim in ["5x2", "4x5", "30x20", "90x120"]:
                    matrices_out[dim] = HeroMatrixEngine.generate_matrix_for_hero_count(hero_count=hero_cnt, dim_key=dim)

            # Sampling frequency schedule
            freq_sched = FrameRateEncodingProtocol.compute_sampling_frequency_schedule(hero_count=hero_cnt, battle_intensity=intensity)
            header = FrameRateEncodingProtocol.pack_stream_header(total_frames=120, base_hz=freq_sched["effective_hz"], channel_count=hero_cnt)

            self._send_json({
                "success": True,
                "hero_count": hero_cnt,
                "sampling_frequency_schedule": freq_sched,
                "frame_stream_header": header,
                "matrices": matrices_out
            })
            return

        # ── EVOLUTIONARY PHYSICS & KINEMATICS (POST) ──────────────────────────
        if path == '/api/ml/evolutionary_physics':
            gens = int(req_data.get("generations", 5))
            friction = float(req_data.get("friction", 0.05))
            gravity = float(req_data.get("gravity", 9.81))
            collision_target = req_data.get("collision_target", [10.0, 0.0, 5.0])

            GLOBAL_EVO_PHYSICS.friction = friction
            GLOBAL_EVO_PHYSICS.gravity = gravity
            GLOBAL_EVO_PHYSICS.collision_target = tuple(collision_target)

            history = []
            for g in range(gens):
                res = GLOBAL_EVO_PHYSICS.evolve_generation()
                history.append(res)
                # Enqueue generation event to Congestion Controller
                GLOBAL_CONGESTION_CONTROLLER.push_event({
                    "type": "evolution_generation_completed",
                    "generation": res["generation"],
                    "best_fitness": res["best_fitness"],
                    "avg_fitness": res["avg_fitness"]
                })

            best_summary = GLOBAL_EVO_PHYSICS.get_best_individual()
            self._send_json({
                "success": True,
                "generations_run": gens,
                "final_generation": GLOBAL_EVO_PHYSICS.generation,
                "best_individual": best_summary,
                "history": history
            })
            return

        # ── CONGESTION CONTROL DISPATCH (POST) ───────────────────────────────
        if path == '/api/ml/congestion_dispatch':
            dispatch_res = GLOBAL_CONGESTION_CONTROLLER.dispatch_batch()
            self._send_json({
                "success": True,
                "dispatch": dispatch_res
            })
        # ── METAVERSE MARKETPLACE & GRANITE LLM (POST) ───────────────────────
        if path == '/api/metaverse/market/order':
            pair = req_data.get("pair", "LAND_042/AET")
            o_type = req_data.get("order_type", OrderType.LIMIT_BUY.value)
            price = float(req_data.get("price", 100.0))
            amount = float(req_data.get("amount", 1.0))
            trader = req_data.get("trader", "metaverse_player")

            res = GLOBAL_METAVERSE_MARKET.place_order(pair, o_type, price, amount, trader)
            self._send_json(res)
            return

        if path == '/api/metaverse/market/amm_swap':
            pool_id = req_data.get("pool_id", "pool_aet_gold")
            token_in = req_data.get("token_in", "AET")
            amount_in = float(req_data.get("amount_in", 10.0))
            slippage = float(req_data.get("slippage_tolerance", 0.05))

            res = GLOBAL_METAVERSE_MARKET.execute_amm_swap(pool_id, token_in, amount_in, slippage)
            if res.get("success"):
                self._send_json(res)
            else:
                self._send_json(res, status=400)
            return

        if path == '/api/metaverse/llm/market_eval':
            pair = req_data.get("pair", "LAND_042/AET")
            model_key = req_data.get("model_key", "ibm-granite-3.1-4b-instruct-2026")
            ob = GLOBAL_METAVERSE_MARKET.get_order_book(pair)
            res = GLOBAL_GRANITE_LLM.evaluate_market_sentiment(pair, ob, model_key)
            self._send_json({"success": True, "evaluation": res})
            return

        if path == '/api/metaverse/llm/merchant_barter':
            offered_item = req_data.get("offered_item", "Kryštálové Brnenie")
            offered_val = float(req_data.get("offered_nominal_val", 120.0))
            requested_item = req_data.get("requested_item", "Aéterová Batéria")
            requested_val = float(req_data.get("requested_nominal_val", 100.0))
            merchant_name = req_data.get("merchant_archetype", "Aéterový Obchodník z Citadely")
            greed = float(req_data.get("greed_factor", 0.15))
            model_key = req_data.get("model_key", "ibm-granite-3.0-2b-instruct")

            res = GLOBAL_GRANITE_LLM.simulate_merchant_barter(
                offered_item=offered_item,
                offered_nominal_val=offered_val,
                requested_item=requested_item,
                requested_nominal_val=requested_val,
                merchant_archetype=merchant_name,
                greed_factor=greed,
                model_key=model_key
            )
            self._send_json({"success": True, "barter": res})
            return

        if path == '/api/metaverse/llm/manipulation_audit':
            events = req_data.get("order_events", [])
            model_key = req_data.get("model_key", "ibm-granite-3.1-4b-instruct-2026")
            res = GLOBAL_GRANITE_LLM.detect_market_manipulation(events, model_key)
            self._send_json({"success": True, "audit": res})
            return

        if path == '/api/totems/spell_projection':
            spell_key = req_data.get("spell_key", "frost_crystal_nova")
            caster_origin = req_data.get("caster_origin", [0.0, 0.0])
            heading_deg = float(req_data.get("heading_deg", 0.0))
            sector_id = req_data.get("sector_id", None)

            totems = GLOBAL_TOTEM_MANAGER.get_totems(sector_id)
            projection_res = GLOBAL_SPELL_PROJECTION.project_spell(
                spell_key=spell_key,
                caster_origin=tuple(caster_origin[:2]),
                target_heading_deg=heading_deg,
                totems_in_sector=totems
            )
            impact_results = []
            for hit in projection_res.get("affected_totems", []):
                t_impact = GLOBAL_TOTEM_MANAGER.apply_spell_impact_to_totem(
                    totem_id=hit["totem_id"],
                    spell_element=hit["element"],
                    potency=projection_res["damage_potency"],
                    hit_factor=hit["hit_effectiveness"]
                )
                impact_results.append(t_impact)

            self._send_json({
                "success": True,
                "projection": projection_res,
                "totem_impacts": impact_results
            })
            return

        if path == '/api/totems/trigger_anomaly':
            sector_id = req_data.get("sector_id", "sector_north_crystal")
            anomaly_type = req_data.get("anomaly_type", SectorAnomalyType.DIMENSIONAL_RIFT.value)
            epicenter = req_data.get("epicenter", [0.0, -8.66])
            magnitude = float(req_data.get("magnitude", 0.75))
            duration_sec = float(req_data.get("duration_sec", 60.0))

            anomaly = GLOBAL_ANOMALY_DETECTOR.trigger_anomaly(
                sector_id=sector_id,
                anomaly_type=anomaly_type,
                epicenter_coords=tuple(epicenter[:2]),
                magnitude=magnitude,
                duration_sec=duration_sec
            )

            affected_totems = []
            for totem in GLOBAL_TOTEM_MANAGER.get_totems(sector_id):
                res = GLOBAL_TOTEM_MANAGER.apply_anomaly_flux_to_totem(
                    totem_id=totem["id"],
                    anomaly_type=anomaly_type,
                    magnitude=magnitude
                )
                affected_totems.append(res)

            self._send_json({
                "success": True,
                "anomaly": anomaly,
                "totem_responses": affected_totems
            })
            return

        if path == '/api/totems/detect_anomalies':
            sector_id = req_data.get("sector_id", "sector_north_crystal")
            sector_totems = GLOBAL_TOTEM_MANAGER.get_totems(sector_id)
            scan_results = GLOBAL_ANOMALY_DETECTOR.scan_sector(sector_id, sector_totems)
            self._send_json({
                "success": True,
                "scan": scan_results
            })
            return

        if path == '/api/totems/attune':
            totem_id = req_data.get("totem_id", "totem_north_crystal")
            caster_tribe = req_data.get("caster_tribe", "crystal")
            channel_energy = float(req_data.get("channel_energy", 25.0))
            attune_res = GLOBAL_TOTEM_MANAGER.attune_totem(
                totem_id=totem_id,
                caster_tribe=caster_tribe,
                channel_energy=channel_energy
            )
            self._send_json(attune_res)
            return

        if path == '/api/totems/phenomenon':
            phenom_type = req_data.get("phenomenon_type", VisualPhenomenonType.CHROMATIC_ABERRATION_BURST.value)
            coords = req_data.get("coordinates", [0.0, 0.0, 0.0])
            intensity = float(req_data.get("intensity", 0.8))
            freq = float(req_data.get("resonance_frequency_hz", 432.0))
            p_data = GLOBAL_VISUAL_PHENOMENA.generate_phenomenon(
                phenomenon_type=phenom_type,
                coordinates=tuple(coords[:3]),
                intensity=intensity,
                resonance_hz=freq
            )
            self._send_json({
                "success": True,
                "phenomenon": p_data
            })
            return

        if path == '/api/cards/mulligan/toggle':
            card_idx = int(req_data.get("card_index", 0))
            res = GLOBAL_MULLIGAN_MANAGER.toggle_card_selection(card_idx)
            self._send_json({"success": True, **res})
            return

        if path == '/api/cards/mulligan/exchange':
            indices = req_data.get("indices", None)
            res = GLOBAL_MULLIGAN_MANAGER.execute_mulligan(indices)
            self._send_json(res)
            return

        if path == '/api/cards/mulligan/reset':
            GLOBAL_MULLIGAN_MANAGER.deal_initial_hand()
            self._send_json({
                "success": True,
                "message": "Mulligan hand reset to canonical initial draw",
                "hand": GLOBAL_MULLIGAN_MANAGER.opening_hand
            })
            return

        if path == '/api/tactical/combinatorial_moves/simulate':
            hero_pos = req_data.get("hero_pos", [0, -2])
            enemy_pos = req_data.get("enemy_pos", [0, 2])
            crystals = req_data.get("crystals", None)
            sim_res = GLOBAL_COMBINATORIAL_ENGINE.simulate_256k_combinations(
                hero_pos=tuple(hero_pos[:2]),
                enemy_pos=tuple(enemy_pos[:2]),
                active_crystals=[tuple(c[:2]) for c in crystals] if crystals else None,
                available_hand=GLOBAL_MULLIGAN_MANAGER.opening_hand
            )
            self._send_json({"success": True, **sim_res})
            return

        if path.startswith('/api/') and self._proxy_to_hub("POST", body=post_data.encode('utf-8') if isinstance(post_data, str) else post_data):
            return
        self._send_json({"error": f"Endpoint {path} not found"}, status=404)

def run_engine_server(port=8089):
    server_address = ('127.0.0.1', port)
    httpd = ThreadedHTTPServer(server_address, KrystalEngineHandler)
    print("=================================================================")
    print(f" KRYSTAL-STACK 3D ENGINE CORE // API ONLINE (PORT {port})")
    print(" Mode: Local Competitive Headless CMS (Python + Janet)")
    print(f" Native Janet CLI: {'DETECTED' if shutil.which('janet') else 'EMBEDDED AST ENGINE ACTIVE'}")
    print(f" Cards Registered: {len(KMEN_CARDS)} | Max HP Rule: 6")
    print(f" Baked .OBJ Assets: {len(os.listdir(ASSET_DIR)) if os.path.exists(ASSET_DIR) else 0}")
    print("=================================================================")
    httpd.serve_forever()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8089
    run_engine_server(port)
