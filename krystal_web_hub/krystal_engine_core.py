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
from dataclasses import asdict

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
    GLOBAL_MULLIGAN_MANAGER, GLOBAL_COMBINATORIAL_ENGINE,
    VorpXVRBridgeEngine, JustCauseKineticPhysicsEngine,
    BorderlandsCelShadingEngine, NconProductMarketingEngine,
    GLOBAL_VORPX_VR_BRIDGE, GLOBAL_JUSTCAUSE_PHYSICS,
    GLOBAL_CEL_SHADING, GLOBAL_NCON_MARKETING,
    AerostatProfile, FloatingIslandNode,
    AerialBalloonIslandEngine, PlungingMortarArtilleryEngine, RogaloAndParachuteFlightEngine,
    GLOBAL_BALLOON_ISLAND_ENGINE, GLOBAL_PLUNGING_MORTAR_ENGINE, GLOBAL_ROGALO_PARACHUTE_ENGINE,
    CampaignChoice, CampaignChapter, EpicCampaignEngine, GLOBAL_EPIC_CAMPAIGN_ENGINE,
    GLOBAL_MRP_HARMONIC_STREET_ENGINE, PINK_PANTHER_PALETTE, VITAL_MAX_HP, GOLDEN_RATIO,
    GLOBAL_WORDPRESS_SECURITY, GLOBAL_PROJECTOR_ANALOG_BRIDGE, GLOBAL_PROCEDURAL_ISLAND_ENGINE, GLOBAL_JAVA_TRANSPILER,
    GLOBAL_WORDPRESS_SUBDOMAIN_GATE,
    GLOBAL_2FA_AUTHENTICATOR, TIME_STEP_SECONDS, TOKEN_DIGITS,
    GLOBAL_VULKAN_IRIS_XE_ENGINE, GLOBAL_GREEK_BOHEMIA_ENGINE,
    GLOBAL_QUADRATIC_TRANSFORMER, GLOBAL_SURFACE_NODE_ENGINE,
    GLOBAL_CHINESE_ZODIAC_SECTOR_ENGINE,
    GLOBAL_HIGH_FIDELITY_3D_ENGINE,
    GLOBAL_WSL_EMULATION_SUBSYSTEM,
    PosledniKmenBiome,
    GLOBAL_TERRAIN_SYNTHESIS_ENGINE,
    GLOBAL_EVOLVED_SVG_ENGINE,
    GLOBAL_EXECUTION_ARCHITECTURE_ENGINE
)
from krystal_web_hub.economic_engine.sovereign_citadel_gameplay import (
    GLOBAL_SOVEREIGN_CITADEL_ENGINE
)
from krystal_web_hub.economic_engine.minecraft_glsl_shader_architecture import (
    GLOBAL_MINECRAFT_SHADER_ENGINE
)
from krystal_web_hub.economic_engine.godot_canvas_arena_engine import (
    GLOBAL_GODOT_CANVAS_ARENA
)
from krystal_web_hub.economic_engine.godot_interior_surface_node_engine import (
    GLOBAL_GODOT_INTERIOR_NODE_ENGINE
)
from krystal_web_hub.economic_engine.cnc_machining_and_drawing_engine import (
    GLOBAL_CNC_ENGINE,
    CANONICAL_TOOLS,
    CANONICAL_MATERIALS
)
from krystal_web_hub.economic_engine.godot_asset_and_camera_pipeline import (
    GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE
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
        if path in ('/desktop', '/webos', '/krystal-os', '/os'):
            self._serve_file(os.path.join(STATIC_DIR, "krystal_webos_desktop.html"), "text/html")
            return
        if path in ('/quadratic', '/quadratic-studio', '/quadratic-transformer'):
            self._serve_file(os.path.join(STATIC_DIR, "quadratic_transformer_studio.html"), "text/html")
            return
        if path in ('/surface-nodes', '/surface-editor', '/surface-studio', '/node-editor'):
            self._serve_file(os.path.join(STATIC_DIR, "surface_node_editor_studio.html"), "text/html")
            return
        if path in ('/chinese-zodiac-sectors', '/zodiac-sectors', '/terrestrial-phenomena', '/chinese-zodiac'):
            self._serve_file(os.path.join(STATIC_DIR, "chinese_zodiac_sectors_studio.html"), "text/html")
            return
        if path in ('/high-fidelity-3d-studio', '/3d-studio', '/3d-display', '/3d-models'):
            self._serve_file(os.path.join(STATIC_DIR, "high_fidelity_3d_studio.html"), "text/html")
            return
        if path in ('/wsl-emulator', '/wsl-studio', '/virtual-linux', '/wsl-terminal'):
            self._serve_file(os.path.join(STATIC_DIR, "wsl_emulator_studio.html"), "text/html")
            return
        if path in ('/terrain-synthesis', '/continuous-world', '/npu-sdf-terrain', '/posledni-kmen-synthesis', '/sdf-terrain'):
            self._serve_file(os.path.join(STATIC_DIR, "npu_sdf_terrain_studio.html"), "text/html")
            return
        if path in ('/evolved-svg-studio', '/svg-blueprint', '/vector-studio', '/posledni-kmen-svg'):
            self._serve_file(os.path.join(STATIC_DIR, "evolved_svg_studio.html"), "text/html")
            return
        if path in ('/hybrid-portal-arena', '/hybrid-mode', '/krystal-ui-system', '/portal-arena'):
            self._serve_file(os.path.join(STATIC_DIR, "hybrid_portal_arena_studio.html"), "text/html")
            return
        if path in ('/wordpress-subdomain-security', '/wp-subdomain-security', '/subdomain-security', '/wp-security-gate'):
            self._serve_file(os.path.join(STATIC_DIR, "wordpress_subdomain_security_studio.html"), "text/html")
            return
        if path in ('/sovereign-citadel', '/citadel-game', '/sovereign-citadel-game', '/citadel-defense'):
            self._serve_file(os.path.join(STATIC_DIR, "sovereign_citadel_game_studio.html"), "text/html")
            return
        if path in ('/godot-canvas-3d', '/godot-game', '/godot-canvas', '/posledni-kmen-3d', '/arena-3d', '/godot_canvas_3d_game.html'):
            self._serve_file(os.path.join(STATIC_DIR, "godot_canvas_3d_game.html"), "text/html")
            return
        if path in ('/interior-surface-node-editor', '/interior-nodes', '/interior-world-generator', '/godot-interior-nodes', '/interior-editor'):
            self._serve_file(os.path.join(STATIC_DIR, "godot_interior_surface_node_studio.html"), "text/html")
            return
        if path in ('/cnc-simulator', '/cnc-drawing', '/cnc', '/cnc_drawing_simulator.html'):
            self._serve_file(os.path.join(STATIC_DIR, "cnc_drawing_simulator.html"), "text/html")
            return
        if path in ('/godot-camera-studio', '/godot-assets', '/godot-camera', '/godot_camera_and_asset_studio.html'):
            self._serve_file(os.path.join(STATIC_DIR, "godot_camera_and_asset_studio.html"), "text/html")
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
        if path in ('/api/status', '/api/health'):
            self._send_json({
                "status": "ONLINE",
                "healthy": True,
                "rules": {
                    "max_hp": VITAL_MAX_HP
                },
                "vital_max_hp_rule": VITAL_MAX_HP,
                "engine": "Krystal-Stack 3D CMS (Janet + Python)",
                "port": 8089,
                "native_janet": shutil.which("janet") is not None,
                "cards_count": len(KMEN_CARDS),
                "assets": os.listdir(ASSET_DIR) if os.path.exists(ASSET_DIR) else []
            })
            return

        # Vulkan Iris Xe Hardware & Memory Grid Telemetry
        if path == '/api/vulkan-iris-xe/telemetry':
            self._send_json(GLOBAL_VULKAN_IRIS_XE_ENGINE.get_full_telemetry())
            return

        # Greek Pantheon & Bohemian Coalitions
        if path == '/api/greek-bohemia/pantheon':
            self._send_json(GLOBAL_GREEK_BOHEMIA_ENGINE.get_pantheon_roster())
            return

        # Philosophical Memory Leveling Strategies
        if path == '/api/greek-bohemia/memory-strategies':
            self._send_json(GLOBAL_GREEK_BOHEMIA_ENGINE.get_philosophical_memory_axioms())
            return

        # Minecraft GLSL Shaders & Godot 4 Pipeline Catalog
        if path == '/api/minecraft-shaders/catalog':
            self._send_json(GLOBAL_MINECRAFT_SHADER_ENGINE.get_full_catalog())
            return

        if path.startswith('/api/minecraft-shaders/simulate-pipeline'):
            self._send_json(GLOBAL_MINECRAFT_SHADER_ENGINE.simulate_pipeline_run())
            return

        # Godot 3D Animated Models & Camera Presets
        if path in ('/api/godot/models', '/api/godot/models/catalog'):
            self._send_json(GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE.get_models_catalog())
            return

        if path in ('/api/godot/camera/templates', '/api/godot/camera-templates'):
            self._send_json(GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE.get_camera_templates())
            return

        if path.startswith('/api/godot/projection-matrices'):
            fov = 75.0
            ortho_size = 14.0
            if '?' in path:
                query_str = path.split('?', 1)[1]
                for part in query_str.split('&'):
                    if '=' in part:
                        k, v = part.split('=', 1)
                        if k == 'fov':
                            try: fov = float(v)
                            except: pass
                        elif k == 'ortho_size':
                            try: ortho_size = float(v)
                            except: pass
            self._send_json(GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE.compute_projection_matrices(fov, ortho_size))
            return

        if path.startswith('/api/godot/package-manager/query'):
            q = "camera"
            if '?' in path:
                query_str = path.split('?', 1)[1]
                for part in query_str.split('&'):
                    if '=' in part:
                        k, v = part.split('=', 1)
                        if k in ('q', 'query'):
                            q = urllib.parse.unquote(v)
            self._send_json(GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE.query_godot_asset_lib(q))
            return

        # Quadratic Variable Transformer & Latent x Bridge
        if path == '/api/quadratic/domains':
            self._send_json(GLOBAL_QUADRATIC_TRANSFORMER.get_domains())
            return

        if path == '/api/quadratic/golden-state':
            self._send_json(GLOBAL_QUADRATIC_TRANSFORMER.evaluate_golden_mean())
            return

        # Surface Node Shader Composer & Reverse Memory Slot Scavenger
        if path == '/api/surface-nodes/catalog':
            self._send_json(GLOBAL_SURFACE_NODE_ENGINE.get_node_catalog())
            return

        if path == '/api/surface-nodes/default-graph':
            self._send_json(GLOBAL_SURFACE_NODE_ENGINE.get_default_graph())
            return

        # Godot Interior Surface & Procedural World Node Engine
        if path == '/api/interior-nodes/catalog':
            self._send_json(GLOBAL_GODOT_INTERIOR_NODE_ENGINE.get_node_catalog())
            return

        if path == '/api/interior-nodes/default-graph':
            self._send_json(GLOBAL_GODOT_INTERIOR_NODE_ENGINE.get_current_graph())
            return

        if path == '/api/interior-nodes/godot-export':
            self._send_json({
                "vital_max_hp_rule": VITAL_MAX_HP,
                "godot_tscn": GLOBAL_GODOT_INTERIOR_NODE_ENGINE.generate_godot_scene_tscn(),
                "godot_gdscript": GLOBAL_GODOT_INTERIOR_NODE_ENGINE.generate_godot_gdscript()
            })
            return

        # ── CNC DRAWING & MACHINING SIMULATOR ─────────────────────────────────
        if path == '/api/cnc/catalog':
            from dataclasses import asdict
            self._send_json({
                "vital_max_hp_rule": VITAL_MAX_HP,
                "tools": {k: asdict(v) for k, v in CANONICAL_TOOLS.items()},
                "materials": {k: asdict(v) for k, v in CANONICAL_MATERIALS.items()},
                "presets": GLOBAL_CNC_ENGINE.get_preset_drawings()
            })
            return

        if path == '/api/cnc/presets':
            self._send_json({
                "vital_max_hp_rule": VITAL_MAX_HP,
                "presets": GLOBAL_CNC_ENGINE.get_preset_drawings()
            })
            return



        # ── CHINESE ZODIAC TERRESTRIAL PHENOMENA & SECTORS ──────────────────
        if path == '/api/zodiac-sectors/all':
            self._send_json({
                "sectors": GLOBAL_CHINESE_ZODIAC_SECTOR_ENGINE.get_all_sectors(),
                "vital_max_hp_rule": VITAL_MAX_HP
            })
            return

        if path == '/api/zodiac-sectors/cycle':
            turn_val = 1
            if parsed.query:
                try:
                    params = urllib.parse.parse_qs(parsed.query)
                    turn_val = int(params.get("turn", [1])[0])
                except Exception:
                    turn_val = 1
            self._send_json(GLOBAL_CHINESE_ZODIAC_SECTOR_ENGINE.evaluate_global_terrestrial_cycle(turn_val))
            return

        # ── HIGH-FIDELITY 3D DISPLAY & WSL MODEL PIPELINE ──────────────────
        if path == '/api/3d-models/catalog':
            self._send_json(GLOBAL_HIGH_FIDELITY_3D_ENGINE.get_catalog())
            return

        if path == '/api/3d-models/wsl-diagnostic':
            self._send_json(GLOBAL_HIGH_FIDELITY_3D_ENGINE.run_wsl_dnf_diagnostic())
            return

        # ── WSL EMULATION SUBSYSTEM ENDPOINTS ──────────────────────────────
        if path == '/api/wsl/subsystem-status':
            self._send_json(GLOBAL_WSL_EMULATION_SUBSYSTEM.get_status())
            return

        if path == '/api/wsl/virtual-fs':
            fs_summary = {
                "vital_max_hp_rule": VITAL_MAX_HP,
                "node_count": len(GLOBAL_WSL_EMULATION_SUBSYSTEM.vfs),
                "nodes": [{"path": k, "is_dir": v.is_dir, "mode": v.mode} for k, v in GLOBAL_WSL_EMULATION_SUBSYSTEM.vfs.items()]
            }
            self._send_json(fs_summary)
            return

        # ── CONTINUOUS TERRAIN SYNTHESIS & NPU OPTIMIZATION ───────────────
        if path == '/api/terrain-synthesis/telemetry':
            self._send_json(GLOBAL_TERRAIN_SYNTHESIS_ENGINE.get_system_telemetry())
            return

        # ── EVOLVED HIGH-DETAIL SVG VECTOR ASSETS ──────────────────────────
        if path == '/api/svg/catalog':
            self._send_json({
                "vital_max_hp_rule": VITAL_MAX_HP,
                "golden_ratio": GOLDEN_RATIO,
                "presets": [
                    {"id": "master-blueprint", "title": "Poslední Kmen Celestial Island Master Blueprint", "format": "SVG", "aspect": "16:9", "url": "/api/svg/master-blueprint.svg"},
                    {"id": "crystal-crest", "title": "Crystal Tribe Frost Spire Emblem", "format": "SVG", "aspect": "1:1", "url": "/api/svg/crystal-crest.svg"},
                    {"id": "toxic-crest", "title": "Toxic Tribe Voronoi Serpent Emblem", "format": "SVG", "aspect": "1:1", "url": "/api/svg/toxic-crest.svg"},
                    {"id": "druid-crest", "title": "Druid Tribe World Tree Emblem", "format": "SVG", "aspect": "1:1", "url": "/api/svg/druid-crest.svg"},
                    {"id": "studna-dusi-crest", "title": "Studna Duší Celestial Vortex Singularity", "format": "SVG", "aspect": "1:1", "url": "/api/svg/studna-dusi-crest.svg"}
                ]
            })
            return

        if path == '/api/svg/master-blueprint.svg':
            svg_content = GLOBAL_EVOLVED_SVG_ENGINE.generate_master_blueprint_svg()
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Content-Length", str(len(svg_content.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(svg_content.encode('utf-8'))
            return

        if path == '/api/svg/crystal-crest.svg':
            svg_content = GLOBAL_EVOLVED_SVG_ENGINE.generate_crystal_tribe_svg()
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Content-Length", str(len(svg_content.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(svg_content.encode('utf-8'))
            return

        if path == '/api/svg/toxic-crest.svg':
            svg_content = GLOBAL_EVOLVED_SVG_ENGINE.generate_toxic_tribe_svg()
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Content-Length", str(len(svg_content.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(svg_content.encode('utf-8'))
            return

        if path == '/api/svg/druid-crest.svg':
            svg_content = GLOBAL_EVOLVED_SVG_ENGINE.generate_druid_tribe_svg()
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Content-Length", str(len(svg_content.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(svg_content.encode('utf-8'))
            return

        if path == '/api/svg/studna-dusi-crest.svg':
            svg_content = GLOBAL_EVOLVED_SVG_ENGINE.generate_studna_dusi_svg()
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Content-Length", str(len(svg_content.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(svg_content.encode('utf-8'))
            return

        # ── EXECUTION ARCHITECTURE & METRIC STRATIFICATION ─────────────────
        if path == '/api/execution-architecture/metrics':
            self._send_json(GLOBAL_EXECUTION_ARCHITECTURE_ENGINE.get_metrics_catalog())
            return

        # ── GODOT 4 CANVAS & 3D ARENA ENDPOINTS (GET) ──────────────────────
        if path == '/api/godot/arena-scene':
            self._send_json(GLOBAL_GODOT_CANVAS_ARENA.get_arena_scene_data())
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

        if path == '/api/vr/headsets/profiles':
            profiles = VorpXVRBridgeEngine.get_headset_profiles()
            self._send_json({
                "success": True,
                "count": len(profiles),
                "profiles": profiles
            })
            return

        if path == '/api/marketing/ncon_product/specs':
            pkg = NconProductMarketingEngine.get_marketing_package()
            self._send_json({
                "success": True,
                "marketing": pkg
            })
            return

        if path == '/api/graphics/cel_shading/spec':
            spec = BorderlandsCelShadingEngine.get_cel_shading_uniforms()
            self._send_json({
                "success": True,
                "spec": spec
            })
            return

        if path == '/api/aerial/islands/network':
            network = GLOBAL_BALLOON_ISLAND_ENGINE.get_island_network()
            self._send_json({
                "success": True,
                "network": network
            })
            return

        if path == '/api/aerial/balloons/profiles':
            profiles = GLOBAL_BALLOON_ISLAND_ENGINE.get_aerostat_profiles()
            self._send_json({
                "success": True,
                "count": len(profiles),
                "profiles": profiles
            })
            return

        if path == '/api/campaign/chapters':
            chapters = GLOBAL_EPIC_CAMPAIGN_ENGINE.get_all_chapters()
            self._send_json({
                "success": True,
                "count": len(chapters),
                "chapters": chapters
            })
            return

        if path == '/api/campaign/codex':
            codex = GLOBAL_EPIC_CAMPAIGN_ENGINE.get_codex()
            self._send_json({
                "success": True,
                "codex": codex
            })
        if path == '/api/mrp/hierarchy/levels':
            levels = GLOBAL_MRP_HARMONIC_STREET_ENGINE.get_hierarchy_levels()
            self._send_json({
                "success": True,
                "count": len(levels),
                "vital_max_hp_rule": VITAL_MAX_HP,
                "golden_ratio": GOLDEN_RATIO,
                "levels": levels
            })
            return

        if path == '/api/mrp/regions':
            regions = GLOBAL_MRP_HARMONIC_STREET_ENGINE.get_canonical_regions()
            self._send_json({
                "success": True,
                "count": len(regions),
                "regions": regions
            })
            return

        if path == '/api/mrp/palette':
            palette = GLOBAL_MRP_HARMONIC_STREET_ENGINE.get_pink_panther_palette()
            self._send_json({
                "success": True,
                "palette": palette
            })
            return

        if path == '/api/mrp/vehicles':
            query = urllib.parse.parse_qs(parsed.query)
            region_filter = query.get("region", [None])[0]
            vehicles = GLOBAL_MRP_HARMONIC_STREET_ENGINE.get_active_vehicles(region_id=region_filter)
            self._send_json({
                "success": True,
                "count": len(vehicles),
                "vital_max_hp_rule": VITAL_MAX_HP,
                "vehicles": vehicles
            })
            return

        if path == '/api/mrp/manifesto':
            manifesto = GLOBAL_MRP_HARMONIC_STREET_ENGINE.get_engine_manifesto()
            self._send_json({
                "success": True,
                "manifesto": manifesto
            })
            return

        if path == '/api/wordpress/security/status':
            self._send_json({
                "success": True,
                "bbq_firewall": "ACTIVE_REGEX_INSPECTION",
                "antispam_bee": {
                    "honeypot_field": GLOBAL_WORDPRESS_SECURITY.honeypot_field_name,
                    "min_submission_duration_sec": GLOBAL_WORDPRESS_SECURITY.min_submission_duration_sec,
                    "zero_database_bloat": True
                },
                "wordfence_bruteforce": {
                    "max_failed_attempts": GLOBAL_WORDPRESS_SECURITY.max_failed_attempts,
                    "lockout_duration_sec": GLOBAL_WORDPRESS_SECURITY.lockout_duration_sec,
                    "active_lockouts": len(GLOBAL_WORDPRESS_SECURITY._lockouts)
                },
                "cryptography": {
                    "primary": "Argon2id (m=65536, t=3, p=4)",
                    "fallback": "PBKDF2-HMAC-SHA512 (100k rounds)"
                }
            })
            return

        if path in ('/api/wordpress/subdomain/security-status', '/api/wp/subdomain/security-status'):
            telemetry = GLOBAL_WORDPRESS_SUBDOMAIN_GATE.get_security_telemetry()
            telemetry["success"] = True
            self._send_json(telemetry)
            return

        if path in ('/api/citadel/state', '/api/citadel/status'):
            self._send_json(GLOBAL_SOVEREIGN_CITADEL_ENGINE.get_state())
            return

        if path == '/api/islands/realms':
            realms = GLOBAL_PROCEDURAL_ISLAND_ENGINE.get_all_realms()
            self._send_json({
                "success": True,
                "count": len(realms),
                "vital_max_hp_rule": VITAL_MAX_HP,
                "realms": realms
            })
            return

        if path == '/api/projector/analog_stream/status':
            frame = GLOBAL_PROJECTOR_ANALOG_BRIDGE.convert_analog_frame_to_npu_tensor(frame_index=1)
            self._send_json({
                "success": True,
                "bridge_status": "ONLINE",
                "analog_stream": frame
            })
            return

        if path == '/api/transpiler/java/realms':
            realms_dict = GLOBAL_PROCEDURAL_ISLAND_ENGINE._realms
            java_code = GLOBAL_JAVA_TRANSPILER.generate_java_island_source(realms_dict)
            self._send_json({
                "success": True,
                "language": "Java 21",
                "features": ["records", "sealed_interfaces", "CompletableFuture", "ForkJoinPool"],
                "vital_max_hp_rule": VITAL_MAX_HP,
                "java_source": java_code
            })
        if path == '/api/auth/2fa/setup':
            query = urllib.parse.parse_qs(parsed.query)
            user = query.get("username", ["admin"])[0]
            setup_data = GLOBAL_2FA_AUTHENTICATOR.setup_user_2fa(username=user)
            self._send_json({
                "success": True,
                "two_factor_setup": setup_data
            })
            return

        if path == '/api/auth/2fa/status':
            query = urllib.parse.parse_qs(parsed.query)
            user = query.get("username", ["admin"])[0]
            status_data = GLOBAL_2FA_AUTHENTICATOR.get_user_status(username=user)
            self._send_json({
                "success": True,
                "two_factor_status": status_data
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
        if path != '/api/wordpress/security/inspect_bbq':
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

        if path == '/api/vr/stereo_projection':
            headset_id = req_data.get("headset_id", "ncon_by_korrado")
            custom_ipd = req_data.get("custom_ipd_mm", None)
            cam_pos = req_data.get("camera_pos", [0.0, 1.7, 0.0])
            res = VorpXVRBridgeEngine.compute_stereoscopic_projection(
                headset_id=headset_id,
                custom_ipd_mm=float(custom_ipd) if custom_ipd is not None else None,
                world_camera_pos=tuple(cam_pos[:3])
            )
            self._send_json({"success": True, "projection": res})
            return

        if path == '/api/physics/grapple_tether/simulate':
            origin = req_data.get("origin_pos", [0.0, 1.7, 0.0])
            target = req_data.get("target_pos", [15.0, 1.0, 20.0])
            p_mass = float(req_data.get("player_mass_kg", 85.0))
            t_mass = float(req_data.get("target_mass_kg", 50.0))
            reel_n = float(req_data.get("reel_in_force_n", 1200.0))
            tether_res = JustCauseKineticPhysicsEngine.simulate_grapple_tether(
                origin_pos=tuple(origin[:3]),
                target_pos=tuple(target[:3]),
                player_mass_kg=p_mass,
                target_mass_kg=t_mass,
                reel_in_force_n=reel_n
            )
            self._send_json({"success": True, "tether": tether_res})
            return

        if path == '/api/physics/slingshot/simulate':
            v_curr = float(req_data.get("current_velocity_mps", 18.0))
            tension = float(req_data.get("tether_tension_n", 1200.0))
            p_mass = float(req_data.get("player_mass_kg", 85.0))
            angle = float(req_data.get("release_angle_deg", 25.0))
            boost_res = JustCauseKineticPhysicsEngine.simulate_slingshot_momentum(
                current_velocity_mps=v_curr,
                tether_tension_n=tension,
                player_mass_kg=p_mass,
                release_angle_deg=angle
            )
            self._send_json({"success": True, "slingshot": boost_res})
            return

        if path == '/api/physics/wingsuit_glide/simulate':
            alt = float(req_data.get("drop_altitude_m", 150.0))
            speed = float(req_data.get("airspeed_mps", 35.0))
            pitch = float(req_data.get("dive_pitch_deg", -12.0))
            glide_res = JustCauseKineticPhysicsEngine.simulate_wingsuit_glide(
                drop_altitude_m=alt,
                airspeed_mps=speed,
                dive_pitch_deg=pitch
            )
            self._send_json({"success": True, "glide": glide_res})
            return

        if path == '/api/aerial/balloons/buoyancy':
            p_id = req_data.get("profile_id", "ironclad_siege_island")
            payload = float(req_data.get("current_payload_kg", 12000.0))
            try:
                buoyancy_res = GLOBAL_BALLOON_ISLAND_ENGINE.compute_buoyancy(p_id, payload)
                self._send_json({"success": True, "buoyancy": buoyancy_res})
            except Exception as e:
                self._send_json({"success": False, "error": str(e)}, status=400)
            return

        if path == '/api/aerial/mortar/fire_plunge':
            elev = float(req_data.get("island_elevation_m", 280.0))
            v0 = float(req_data.get("muzzle_velocity_mps", 82.0))
            pitch = float(req_data.get("pitch_angle_deg", 65.0))
            yaw = float(req_data.get("yaw_angle_deg", 0.0))
            caliber = float(req_data.get("shell_caliber_mm", 240.0))
            w_speed = float(req_data.get("wind_speed_mps", 4.5))
            w_dir = float(req_data.get("wind_direction_deg", 90.0))
            target_dist = float(req_data.get("target_dist_m", 450.0))
            target_hp = int(req_data.get("target_initial_hp", 6))

            mortar_res = GLOBAL_PLUNGING_MORTAR_ENGINE.fire_plunging_mortar(
                island_elevation_m=elev,
                muzzle_velocity_mps=v0,
                pitch_angle_deg=pitch,
                yaw_angle_deg=yaw,
                shell_caliber_mm=caliber,
                wind_speed_mps=w_speed,
                wind_direction_deg=w_dir,
                target_dist_m=target_dist,
                target_initial_hp=target_hp
            )
            self._send_json({"success": True, "mortar_fire": mortar_res})
            return

        if path == '/api/aerial/flight/simulate':
            v_type = req_data.get("vehicle_type", "rogallo_hang_glider")
            if v_type == "rogallo_hang_glider":
                launch_alt = float(req_data.get("launch_altitude_m", 340.0))
                speed = float(req_data.get("initial_airspeed_mps", 18.0))
                g_ratio = float(req_data.get("glide_ratio", 7.5))
                thermal = float(req_data.get("thermal_updraft_mps", 3.2))
                duration = float(req_data.get("flight_duration_sec", 45.0))
                headwind = float(req_data.get("wind_headwind_mps", 2.0))
                sim_res = GLOBAL_ROGALO_PARACHUTE_ENGINE.simulate_rogallo_glider(
                    launch_altitude_m=launch_alt,
                    initial_airspeed_mps=speed,
                    glide_ratio=g_ratio,
                    thermal_updraft_mps=thermal,
                    flight_duration_sec=duration,
                    wind_headwind_mps=headwind
                )
            else:
                dep_alt = float(req_data.get("deployment_altitude_m", 280.0))
                mass = float(req_data.get("payload_mass_kg", 85.0))
                area = float(req_data.get("canopy_area_m2", 28.0))
                drag = float(req_data.get("drag_coefficient", 1.45))
                steer = float(req_data.get("steer_lateral_mps", 3.5))
                dur = float(req_data.get("descent_duration_sec", 35.0))
                sim_res = GLOBAL_ROGALO_PARACHUTE_ENGINE.simulate_steerable_parachute(
                    deployment_altitude_m=dep_alt,
                    payload_mass_kg=mass,
                    canopy_area_m2=area,
                    drag_coefficient=drag,
                    steer_lateral_mps=steer,
                    descent_duration_sec=dur
                )
            self._send_json({"success": True, "flight_simulation": sim_res})
            return

        if path == '/api/aerial/skybridge/traverse':
            b_id = req_data.get("bridge_id", "bridge_alpha_beta")
            weight = float(req_data.get("traveler_weight_kg", 85.0))
            method = req_data.get("method", "walk")
            trav_res = GLOBAL_BALLOON_ISLAND_ENGINE.traverse_skybridge(b_id, weight, method)
            self._send_json({"success": True, "traversal": trav_res})
            return

        if path == '/api/aerial/grapple/island_board':
            hero_pos = req_data.get("hero_pos", [0.0, 220.0, 50.0])
            island_id = req_data.get("target_island_id", "island_alpha")
            island_pos = req_data.get("target_island_pos", [0.0, 280.0, 0.0])
            max_cable = float(req_data.get("grapple_cable_length_max_m", 80.0))
            board_res = GLOBAL_ROGALO_PARACHUTE_ENGINE.execute_just_cause_grapple_to_balloon_island(
                hero_initial_pos=tuple(hero_pos[:3]),
                target_island_id=island_id,
                target_island_pos=tuple(island_pos[:3]),
                grapple_cable_length_max_m=max_cable
            )
            self._send_json({"success": True, "boarding": board_res})
            return

        if path == '/api/campaign/generate_mission':
            tier = req_data.get("altitude_tier", "skybridge_midways")
            threat = int(req_data.get("threat_level", 3))
            weather = req_data.get("weather_condition", "totem_rift_lightning")
            faction = req_data.get("enemy_faction", "toxic_tribe")

            mission = GLOBAL_EPIC_CAMPAIGN_ENGINE.generate_procedural_mission(
                altitude_tier=tier,
                threat_level=threat,
                weather_condition=weather,
                enemy_faction=faction
            )
            self._send_json({"success": True, "mission": mission})
            return

        if path == '/api/campaign/simulate_operation':
            c_id = req_data.get("chapter_id", "chapter_1_desert_caravan")
            choice_id = req_data.get("chosen_choice_id", "choice_silent_drop")
            p_hp = int(req_data.get("player_squad_hp", 6))
            artillery = bool(req_data.get("artillery_active", True))
            flight = bool(req_data.get("flight_support_active", True))

            sim_res = GLOBAL_EPIC_CAMPAIGN_ENGINE.simulate_combat_operation(
                chapter_id=c_id,
                chosen_choice_id=choice_id,
                player_squad_hp=p_hp,
                artillery_active=artillery,
                flight_support_active=flight
            )
            self._send_json({"success": True, "operation": sim_res})
            return

        if path == '/api/mrp/color_tint/harmonic':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            base_col = tuple(req_data.get("base_color", [236, 72, 153]))
            step = int(req_data.get("step", 1))
            weight = float(req_data.get("weight", 1.0))
            style = req_data.get("style", "pink_panther_chic")
            tint = GLOBAL_MRP_HARMONIC_STREET_ENGINE.compute_harmonic_color_tint(
                base_color=base_col,
                step=step,
                weight=weight,
                style=style
            )
            self._send_json({
                "success": True,
                "golden_ratio": GOLDEN_RATIO,
                "tint": tint.to_dict()
            })
            return

        if path == '/api/mrp/street_map/generate':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            r_id = req_data.get("region_id", "gotham_city")
            gw = float(req_data.get("grid_width", 800.0))
            gh = float(req_data.get("grid_height", 600.0))
            s_seed = req_data.get("seed", None)
            network = GLOBAL_MRP_HARMONIC_STREET_ENGINE.generate_procedural_street_network(
                region_id=r_id,
                grid_width=gw,
                grid_height=gh,
                seed=s_seed
            )
            self._send_json({
                "success": True,
                "street_network": network
            })
            return

        if path == '/api/mrp/vehicles/simulate':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            v_id = req_data.get("vehicle_id", "panther_coupe_01")
            dt = float(req_data.get("dt_seconds", 0.05))
            throttle = float(req_data.get("throttle", 1.0))
            steering = float(req_data.get("steering", 0.0))
            handbrake = bool(req_data.get("handbrake", False))
            drift_boost = bool(req_data.get("drift_boost", False))

            try:
                updated_veh = GLOBAL_MRP_HARMONIC_STREET_ENGINE.simulate_vehicle_tick(
                    vehicle_id=v_id,
                    dt_seconds=dt,
                    throttle=throttle,
                    steering=steering,
                    handbrake=handbrake,
                    drift_boost=drift_boost
                )
                self._send_json({
                    "success": True,
                    "vital_max_hp_rule": VITAL_MAX_HP,
                    "vehicle": updated_veh
                })
            except KeyError as e:
                self._send_json({"error": str(e)}, status=404)
            return

        if path == '/api/mrp/vehicles/damage':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            v_id = req_data.get("vehicle_id", "panther_coupe_01")
            dmg = int(req_data.get("damage_amount", 1))
            try:
                veh = GLOBAL_MRP_HARMONIC_STREET_ENGINE.apply_damage_to_vehicle(v_id, dmg)
                self._send_json({
                    "success": True,
                    "vital_max_hp_rule": VITAL_MAX_HP,
                    "vehicle": veh
                })
            except KeyError as e:
                self._send_json({"error": str(e)}, status=404)
            return

        if path == '/api/mrp/vehicles/repair':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            v_id = req_data.get("vehicle_id", "panther_coupe_01")
            rep = int(req_data.get("repair_amount", 2))
            try:
                veh = GLOBAL_MRP_HARMONIC_STREET_ENGINE.repair_vehicle(v_id, rep)
                self._send_json({
                    "success": True,
                    "vital_max_hp_rule": VITAL_MAX_HP,
                    "vehicle": veh
                })
            except KeyError as e:
                self._send_json({"error": str(e)}, status=404)
            return

        if path == '/api/wordpress/security/inspect_bbq':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            q_str = req_data.get("query_string", "")
            req_uri = req_data.get("request_uri", "")
            ua = req_data.get("user_agent", "")
            res = GLOBAL_WORDPRESS_SECURITY.inspect_bbq_firewall(q_str, request_uri=req_uri, user_agent=ua)
            self._send_json({"success": True, "result": res})
            return

        if path == '/api/wordpress/security/antispam_bee':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            f_data = req_data.get("form_data", {})
            sub_ts = float(req_data.get("submission_timestamp_ms", time.time() * 1000.0 - 5000.0))
            res = GLOBAL_WORDPRESS_SECURITY.evaluate_antispam_bee(f_data, submission_timestamp_ms=sub_ts)
            self._send_json({"success": True, "result": res})
            return

        if path == '/api/wordpress/security/login_attempt':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            ip = req_data.get("ip", "127.0.0.1")
            user = req_data.get("username", "admin")
            is_success = bool(req_data.get("success", False))
            res = GLOBAL_WORDPRESS_SECURITY.record_login_attempt(ip, user, is_success)
            self._send_json({"success": True, "result": res})
            return

        if path in ('/api/wordpress/subdomain/verify-token', '/api/wp/subdomain/verify-token'):
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            token = req_data.get("token", "")
            if not token:
                auth_hdr = self.headers.get("Authorization", "")
                if auth_hdr.startswith("Bearer "):
                    token = auth_hdr[7:].strip()
            ok, reason, session = GLOBAL_WORDPRESS_SUBDOMAIN_GATE.verify_signed_token(token)
            if ok and session:
                self._send_json({
                    "success": True,
                    "status": "VERIFIED",
                    "reason": reason,
                    "session": {
                        "user_id": session.user_id,
                        "username": session.username,
                        "role": session.role,
                        "subdomain": session.subdomain,
                        "issued_at": session.issued_at,
                        "vital_max_hp_rule": session.vital_max_hp
                    }
                })
            else:
                self._send_json({
                    "success": False,
                    "status": "REJECTED",
                    "reason": reason,
                    "vital_max_hp_rule": VITAL_MAX_HP
                }, status=401)
            return

        if path in ('/api/wordpress/subdomain/generate-token', '/api/wp/subdomain/generate-token'):
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            uid = int(req_data.get("user_id", 1))
            usr = str(req_data.get("username", "admin"))
            rol = str(req_data.get("role", "administrator"))
            sub = str(req_data.get("subdomain", "krystal.poslednikmen.cz"))
            token = GLOBAL_WORDPRESS_SUBDOMAIN_GATE.create_signed_token(uid, usr, rol, sub)
            self._send_json({
                "success": True,
                "token": token,
                "subdomain": sub,
                "user_id": uid,
                "vital_max_hp_rule": VITAL_MAX_HP
            })
            return

        if path == '/api/citadel/play-card':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            cid = req_data.get("card_id", "")
            coords = req_data.get("target_coords", None)
            res = GLOBAL_SOVEREIGN_CITADEL_ENGINE.play_card(cid, target_coords=coords)
            self._send_json(res)
            return

        if path == '/api/citadel/spawn-wave':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            wn = int(req_data.get("wave_number", 1))
            res = GLOBAL_SOVEREIGN_CITADEL_ENGINE.spawn_wave(wn)
            self._send_json({"success": True, "wave_number": wn, "invaders": res})
            return

        if path == '/api/citadel/tick':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            dt = float(req_data.get("delta_time", 0.05))
            state = GLOBAL_SOVEREIGN_CITADEL_ENGINE.update_simulation_tick(dt)
            self._send_json(state)
            return

        if path == '/api/citadel/reset':
            state = GLOBAL_SOVEREIGN_CITADEL_ENGINE.reset_game()
            self._send_json(state)
            return

        # ── GODOT 4 CANVAS & 3D ARENA ENDPOINTS (POST) ─────────────────────
        if path == '/api/godot/tactical-action':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            act = req_data.get("action_type", "move")
            t_q = int(req_data.get("target_q", 0))
            t_r = int(req_data.get("target_r", 0))
            res = GLOBAL_GODOT_CANVAS_ARENA.execute_tactical_action(act, t_q, t_r)
            self._send_json(res)
            return

        if path == '/api/godot/ai-counter':
            res = GLOBAL_GODOT_CANVAS_ARENA.execute_ai_counter_turn()
            self._send_json(res)
            return

        if path == '/api/godot/reset':
            res = GLOBAL_GODOT_CANVAS_ARENA.reset_arena()
            self._send_json(res)
            return

        if path == '/api/islands/geometry':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            r_id = req_data.get("realm_id", "denmark_heaven")
            seed = int(req_data.get("seed", 42))
            geo = GLOBAL_PROCEDURAL_ISLAND_ENGINE.generate_island_geometry(r_id, seed=seed)
            self._send_json({"success": True, "vital_max_hp_rule": VITAL_MAX_HP, "geometry": geo})
            return

        if path == '/api/projector/analog_stream/sample':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            f_idx = int(req_data.get("frame_index", 1))
            intensity = float(req_data.get("beam_intensity", 0.65))
            noise = float(req_data.get("analog_noise_factor", 0.02))
            frame = GLOBAL_PROJECTOR_ANALOG_BRIDGE.convert_analog_frame_to_npu_tensor(
                frame_index=f_idx,
                beam_intensity=intensity,
                analog_noise_factor=noise
            )
            self._send_json({"success": True, "analog_frame": frame})
            return

        if path == '/api/auth/2fa/confirm':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            user = req_data.get("username", "admin")
            tok = req_data.get("token", "")
            res = GLOBAL_2FA_AUTHENTICATOR.confirm_user_2fa(username=user, submitted_token=tok)
            self._send_json(res, status=200 if res.get("success") else 400)
            return

        if path == '/api/auth/2fa/verify':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            user = req_data.get("username", "admin")
            tok = req_data.get("token_or_backup", "")
            res = GLOBAL_2FA_AUTHENTICATOR.verify_credentials_2fa(username=user, token_or_backup=tok)
            self._send_json(res, status=200 if res.get("authenticated") else 401)
            return

        if path == '/api/auth/2fa/challenge_number':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            user = req_data.get("username", "admin")
            chal = GLOBAL_2FA_AUTHENTICATOR.create_number_matching_challenge(username=user)
            self._send_json({
                "success": True,
                "challenge": chal
            })
            return

        if path == '/api/auth/2fa/verify_number_match':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            c_id = req_data.get("challenge_id", "")
            num = int(req_data.get("entered_number", 0))
            res = GLOBAL_2FA_AUTHENTICATOR.verify_number_matching_challenge(challenge_id=c_id, entered_number=num)
            self._send_json(res, status=200 if res.get("success") else 400)
            return

        # ── VULKAN IRIS XE CUSTOM ENGINE & CPU WHISPERER ENDPOINTS ────────────
        if path == '/api/vulkan-iris-xe/simulate-dispatch':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            chunks = int(req_data.get("workload_chunks", 16))
            load = float(req_data.get("eu_load_percent", 85.0))
            res = GLOBAL_VULKAN_IRIS_XE_ENGINE.simulate_dispatch(workload_chunks=chunks, eu_load_percent=load)
            self._send_json(res)
            return

        if path == '/api/vulkan-iris-xe/tune-whisperer':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            en_whisp = bool(req_data.get("enable_whisperer", True))
            en_zero = bool(req_data.get("enable_zero_copy", True))
            en_npu = bool(req_data.get("enable_npu", True))
            en_ssd = bool(req_data.get("enable_ssd_prefetch", True))
            res = GLOBAL_VULKAN_IRIS_XE_ENGINE.tune_whisperer(en_whisp, en_zero, en_npu, en_ssd)
            self._send_json(res)
            return

        if path == '/api/vulkan-iris-xe/trigger-swap':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            coords = req_data.get("coordinates", [0, 0, 0])
            coords_tuple = (int(coords[0]), int(coords[1]), int(coords[2])) if len(coords) >= 3 else (0, 0, 0)
            res = GLOBAL_VULKAN_IRIS_XE_ENGINE.trigger_memory_swap(coords_tuple)
            self._send_json(res)
            return

        # ── GREEK BOHEMIA PANTHEON & AUTONOMOUS MEMORY LEVELING POST ENDPOINTS ──
        if path == '/api/greek-bohemia/simulate-leveling':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            axiom_ids = req_data.get("axiom_ids", None)
            res = GLOBAL_GREEK_BOHEMIA_ENGINE.simulate_autonomous_memory_leveling(active_axiom_ids=axiom_ids)
            self._send_json(res)
            return

        if path == '/api/greek-bohemia/form-pact':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            g_id = req_data.get("greek_deity_id", "zeus")
            b_id = req_data.get("bohemian_ally_id", "perun")
            title = req_data.get("pact_title", "Nový Diplomatický Pakt")
            lore = req_data.get("lore", "Spojenie síl antického Grécka a pohanskej Bohemie.")
            res = GLOBAL_GREEK_BOHEMIA_ENGINE.form_new_pact(g_id, b_id, title, lore)
            self._send_json(res)
            return

        # ── GODOT 3D ASSETS, CAMERA & ADDON INSTALLATION POST ENDPOINTS ──
        if path == '/api/godot/camera/configure':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            template_id = req_data.get("template_id", "tps_orbit")
            custom_overrides = req_data.get("overrides", None)
            res = GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE.configure_camera(template_id, custom_overrides)
            self._send_json(res)
            return

        if path == '/api/godot/package-manager/install':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            package_id = req_data.get("package_id", "camera-controller-3d")
            res = GLOBAL_GODOT_ASSET_AND_CAMERA_PIPELINE.install_addon_package(package_id)
            self._send_json(res)
            return

        # ── QUADRATIC VARIABLE TRANSFORMER POST ENDPOINTS ───────────────────
        if path == '/api/quadratic/solve-x':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            domain_id = req_data.get("domain_id", "memory_latency_ns")
            val = float(req_data.get("value", 20.0))
            try:
                res = GLOBAL_QUADRATIC_TRANSFORMER.solve_latent_x(domain_id, val)
                self._send_json(res)
            except Exception as err:
                self._send_json({"error": str(err)}, status=400)
            return

        if path == '/api/quadratic/convert':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            src = req_data.get("source_domain", "memory_latency_ns")
            val = float(req_data.get("value", 20.0))
            tgt = req_data.get("target_domain", "vram_allocation_mb")
            try:
                res = GLOBAL_QUADRATIC_TRANSFORMER.convert(src, val, tgt)
                self._send_json(res)
            except Exception as err:
                self._send_json({"error": str(err)}, status=400)
            return

        if path == '/api/quadratic/convert-all':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            src = req_data.get("source_domain", "memory_latency_ns")
            val = float(req_data.get("value", 20.0))
            try:
                res = GLOBAL_QUADRATIC_TRANSFORMER.convert_all(src, val)
                self._send_json(res)
            except Exception as err:
                self._send_json({"error": str(err)}, status=400)
            return

        # ── SURFACE NODE SHADER & MEMORY RECLAIMER POST ENDPOINTS ──────────
        if path == '/api/surface-nodes/evaluate':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            res = GLOBAL_SURFACE_NODE_ENGINE.evaluate_node_graph(req_data)
            self._send_json(res)
            return

        if path == '/api/surface-nodes/scavenge-memory':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            alloc_p = req_data.get("alloc_pattern", None)
            free_p = req_data.get("free_pattern", None)
            res = GLOBAL_SURFACE_NODE_ENGINE.scavenge_and_generate_scenarios(alloc_pattern=alloc_p, free_pattern=free_p)
            self._send_json(res)
            return

        # Godot Interior Surface & Procedural World Node Engine
        if path in ('/api/interior-nodes/evaluate-graph', '/api/interior-nodes/generate-world'):
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            res = GLOBAL_GODOT_INTERIOR_NODE_ENGINE.evaluate_graph(req_data)
            self._send_json(res)
            return

        # ── CNC DRAWING & MACHINING POST ENDPOINTS ────────────────────────────
        if path == '/api/cnc/generate-gcode':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            entities = req_data.get("entities", [])
            tool_id = req_data.get("tool_id", "t1_endmill_3mm")
            material_id = req_data.get("material_id", "al_6061")
            prog_name = req_data.get("program_name", "KRYSTAL_CNC_JOB")
            target_depth = float(req_data.get("target_depth", 3.0))

            toolpath = GLOBAL_CNC_ENGINE.generate_toolpath(entities, tool_id, material_id, target_depth)
            res = GLOBAL_CNC_ENGINE.generate_gcode(toolpath, tool_id, material_id, prog_name)
            self._send_json(res)
            return

        if path == '/api/cnc/simulate-toolpath':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            entities = req_data.get("entities", [])
            tool_id = req_data.get("tool_id", "t1_endmill_3mm")
            material_id = req_data.get("material_id", "al_6061")
            target_depth = float(req_data.get("target_depth", 3.0))

            res = GLOBAL_CNC_ENGINE.simulate_machining(entities, tool_id, material_id, target_depth)
            self._send_json(res)
            return

        if path == '/api/cnc/speeds-and-feeds':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            tool_id = req_data.get("tool_id", "t1_endmill_3mm")
            material_id = req_data.get("material_id", "al_6061")
            res = GLOBAL_CNC_ENGINE.calculate_speeds_and_feeds(tool_id, material_id)
            self._send_json(res)
            return



        # ── CHINESE ZODIAC SECTORS POST ENDPOINTS ──────────────────────────
        if path == '/api/zodiac-sectors/trigger-phenomenon':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            sec_id = req_data.get("sector_id", "sector_01_rat")
            intensity = float(req_data.get("intensity_delta", 0.35))
            try:
                res = GLOBAL_CHINESE_ZODIAC_SECTOR_ENGINE.trigger_phenomenon(sec_id, intensity)
                self._send_json(res)
            except Exception as err:
                self._send_json({"error": str(err)}, status=400)
            return

        # ── HIGH-FIDELITY 3D DISPLAY & WSL POST ENDPOINTS ──────────────────
        if path == '/api/3d-models/bake-model':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            target = req_data.get("model_target", "dragon")
            try:
                if target == "dragon":
                    fpath = GLOBAL_HIGH_FIDELITY_3D_ENGINE.bake_crystal_dragon_sanctuary()
                elif target == "astrolabe":
                    fpath = GLOBAL_HIGH_FIDELITY_3D_ENGINE.bake_zodiac_celestial_astrolabe()
                elif target == "titan":
                    fpath = GLOBAL_HIGH_FIDELITY_3D_ENGINE.bake_cybernetic_titan_mech()
                elif target == "tree":
                    fpath = GLOBAL_HIGH_FIDELITY_3D_ENGINE.bake_biomorphic_tree_of_life()
                else:
                    fpath = GLOBAL_HIGH_FIDELITY_3D_ENGINE.bake_crystal_dragon_sanctuary()
                self._send_json({
                    "status": "BAKE_SUCCESS",
                    "model_target": target,
                    "filepath": fpath,
                    "vital_max_hp_rule": VITAL_MAX_HP
                })
            except Exception as err:
                self._send_json({"error": str(err)}, status=400)
            return

        if path == '/api/3d-models/wsl-validate':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            m_filename = req_data.get("filename", "crystal_dragon_sanctuary.obj")
            try:
                res = GLOBAL_HIGH_FIDELITY_3D_ENGINE.import_and_validate_model_via_wsl(m_filename)
                self._send_json(res)
            except Exception as err:
                self._send_json({"error": str(err)}, status=400)
            return

        # ── WSL EMULATION SUBSYSTEM POST ENDPOINTS ─────────────────────────
        if path == '/api/wsl/exec':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            cmd = req_data.get("command", "uname -a")
            try:
                res = GLOBAL_WSL_EMULATION_SUBSYSTEM.execute_command(cmd)
                self._send_json(res)
            except Exception as err:
                self._send_json({"error": str(err)}, status=400)
            return

        if path == '/api/wsl/dnf-emulate':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            action = req_data.get("action", "install")
            pkg_name = req_data.get("package", "assimp")
            try:
                if action == "install":
                    res = GLOBAL_WSL_EMULATION_SUBSYSTEM.dnf.install(pkg_name)
                elif action == "remove":
                    res = GLOBAL_WSL_EMULATION_SUBSYSTEM.dnf.remove(pkg_name)
                elif action == "repolist":
                    res = {"repos": GLOBAL_WSL_EMULATION_SUBSYSTEM.dnf.repolist(), "vital_max_hp_rule": VITAL_MAX_HP}
                else:
                    res = {"installed": GLOBAL_WSL_EMULATION_SUBSYSTEM.dnf.list_installed(), "vital_max_hp_rule": VITAL_MAX_HP}
                self._send_json(res)
            except Exception as err:
                self._send_json({"error": str(err)}, status=400)
            return

        # ── CONTINUOUS TERRAIN SYNTHESIS & NPU OPTIMIZATION POST ENDPOINTS ──
        if path == '/api/terrain-synthesis/evaluate-sdf':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            origin = req_data.get("origin", [0.0, 5.0, 5.0])
            direction = req_data.get("direction", [0.0, -0.7071, -0.7071])
            biome_str = req_data.get("biome", "Crystal_Severni_Stity")
            try:
                biome_enum = PosledniKmenBiome(biome_str)
            except Exception:
                biome_enum = PosledniKmenBiome.CRYSTAL
            try:
                res = GLOBAL_TERRAIN_SYNTHESIS_ENGINE.raymarch(
                    ray_origin=(float(origin[0]), float(origin[1]), float(origin[2])),
                    ray_dir=(float(direction[0]), float(direction[1]), float(direction[2])),
                    biome_key=biome_enum
                )
                self._send_json(asdict(res))
            except Exception as err:
                self._send_json({"error": str(err)}, status=400)
            return

        if path == '/api/terrain-synthesis/simulate-erosion':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            x = float(req_data.get("x", 2.0))
            z = float(req_data.get("z", 2.0))
            biome_str = req_data.get("biome", "Druid_Pradavny_Les")
            try:
                biome_enum = PosledniKmenBiome(biome_str)
            except Exception:
                biome_enum = PosledniKmenBiome.DRUID
            try:
                thermal = GLOBAL_TERRAIN_SYNTHESIS_ENGINE.evaluate_thermal_weathering(x, z, biome_key=biome_enum)
                hydraulic = GLOBAL_TERRAIN_SYNTHESIS_ENGINE.evaluate_hydraulic_erosion(x, z, biome_key=biome_enum)
                height = GLOBAL_TERRAIN_SYNTHESIS_ENGINE.evaluate_master_terrain(x, z, biome_key=biome_enum)
                self._send_json({
                    "height": round(height, 4),
                    "thermal_weathering": thermal,
                    "hydraulic_erosion": hydraulic,
                    "vital_max_hp_rule": VITAL_MAX_HP
                })
            except Exception as err:
                self._send_json({"error": str(err)}, status=400)
            return

        if path == '/api/terrain-synthesis/npu-driver-dispatch':
            try:
                req_data = json.loads(post_data) if post_data else {}
            except Exception:
                req_data = {}
            enable_npu = req_data.get("enable_npu", True)
            GLOBAL_TERRAIN_SYNTHESIS_ENGINE.npu_acceleration_enabled = bool(enable_npu)
            telemetry = GLOBAL_TERRAIN_SYNTHESIS_ENGINE.get_system_telemetry()
            self._send_json({
                "status": "DISPATCH_CONFIGURED",
                "npu_acceleration_enabled": GLOBAL_TERRAIN_SYNTHESIS_ENGINE.npu_acceleration_enabled,
                "telemetry": telemetry,
                "vital_max_hp_rule": VITAL_MAX_HP
            })
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
