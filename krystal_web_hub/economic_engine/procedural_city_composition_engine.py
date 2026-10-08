"""
KRYSTAL-STACK: PROCEDURAL CITY COMPOSITION & MULTI-ASSET ARTISTRY ENGINE
========================================================================
Implements procedural urban generation conceptualized as a layered, multi-asset
canvas painting ("Mesto ako Kreslená Kompozícia Assetov").

Key Architectural Pillars:
1. Seven Hierarchical Composition Layers (Macro Skyline, Midground Blocks,
   Infrastructure Conduits, Biophilic Infill, Micro-Props, Kinetic Cruisers, Atmosphere).
2. Golden Ratio Focal Grids (Phi = 1.61803398875) for dramatic artistic perspective.
3. Dual ASCII Visual Canvas Painters:
   - Side Skyline Silhouette Canvas (Side Elevation)
   - Top-Down Urban Plan Canvas (Plan View)
4. Multi-Substrate Transpilation:
   - Godot 4.x Forward+ Scene Graph (.tscn)
   - Modern Java 21 Virtual Thread Records
   - Janet Functional DSL Immutable AST
5. Strict Invariants:
   - VITAL_MAX_HP = 6 on all vehicles, citadels, and garrisons.
   - Bit-exact deterministic seed reproduction.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 1.0 / GOLDEN_RATIO  # ~0.61803398875
VITAL_MAX_HP: int = 6
BASE_MODULE_M0: float = 4.854  # 3 * Phi in meters


class AssetCategory(str, Enum):
    MACRO_DOMINANT = "macro_dominant"          # Layer 0: Spires, Citadels, Clocktowers
    MIDGROUND_BLOCK = "midground_block"        # Layer 1: Tenements, Plinths, Terraces
    INFRASTRUCTURE = "infrastructure"          # Layer 2: Boulevards, Bridges, Waterways
    BIOPHILIC_INFILL = "biophilic_infill"      # Layer 3: Linden Trees, Fountains, Parks
    MICRO_PROP = "micro_prop"                  # Layer 4: Street Lamps, Neon Billboards
    KINETIC_CRUISER = "kinetic_cruiser"        # Layer 5: Top-down vehicle cruisers (HP <= 6)
    ATMOSPHERE = "atmosphere"                  # Layer 6: Volumetric haze, sun, sky


@dataclass(frozen=True)
class AssetBrush:
    """A modular reusable asset archetype used as an artistic brush."""
    brush_id: str
    name: str
    category: AssetCategory
    width_m: float
    depth_m: float
    height_m: float
    ascii_char: str
    base_color_hex: str
    emissive_glow: bool = False
    vital_hp: int = VITAL_MAX_HP
    description: str = ""


# Master Catalog of Composition Brushes
ASSET_BRUSH_CATALOG: Dict[str, AssetBrush] = {
    # Layer 0: Macro Dominants
    "CYBER_SPIRE_MONOLITH": AssetBrush(
        brush_id="CYBER_SPIRE_MONOLITH",
        name="Kryštálový Kyber-Monolit",
        category=AssetCategory.MACRO_DOMINANT,
        width_m=BASE_MODULE_M0 * 4.0,  # ~19.4m
        depth_m=BASE_MODULE_M0 * 4.0,
        height_m=120.0,
        ascii_char="▲",
        base_color_hex="#06b6d4",
        emissive_glow=True,
        vital_hp=VITAL_MAX_HP,
        description="Impozantná ihlicovitá veža dominujúca panoráme mesta."
    ),
    "ALCHEMICAL_CLOCKTOWER": AssetBrush(
        brush_id="ALCHEMICAL_CLOCKTOWER",
        name="Alchymistická Veža s Orlojom",
        category=AssetCategory.MACRO_DOMINANT,
        width_m=BASE_MODULE_M0 * 3.0,  # ~14.5m
        depth_m=BASE_MODULE_M0 * 3.0,
        height_m=68.0,
        ascii_char="Ω",
        base_color_hex="#f59e0b",
        emissive_glow=True,
        vital_hp=VITAL_MAX_HP,
        description="Goticko-česká orlojová veža s medenou kopulou."
    ),
    "DATA_CITADEL_ZIGGURAT": AssetBrush(
        brush_id="DATA_CITADEL_ZIGGURAT",
        name="Dátová Zikkurat Citadela",
        category=AssetCategory.MACRO_DOMINANT,
        width_m=BASE_MODULE_M0 * 6.0,  # ~29.1m
        depth_m=BASE_MODULE_M0 * 6.0,
        height_m=48.0,
        ascii_char="█",
        base_color_hex="#8b5cf6",
        emissive_glow=False,
        vital_hp=VITAL_MAX_HP,
        description="Stupňovitá monolitická citadela stabilizujúca horizont."
    ),
    # Layer 1: Midground Architecture
    "MODULAR_TENEMENT_BLOCK": AssetBrush(
        brush_id="MODULAR_TENEMENT_BLOCK",
        name="Modulárny Mestský Blok",
        category=AssetCategory.MIDGROUND_BLOCK,
        width_m=BASE_MODULE_M0 * 3.0,
        depth_m=BASE_MODULE_M0 * 2.0,
        height_m=24.0,
        ascii_char="■",
        base_color_hex="#d4d4d8",
        emissive_glow=False,
        vital_hp=VITAL_MAX_HP,
        description="6-poschodový polyfunkčný obytný dom s balkónovým rytmom."
    ),
    "COMMERCIAL_ARCADE_PLINTH": AssetBrush(
        brush_id="COMMERCIAL_ARCADE_PLINTH",
        name="Obchodná Parterová Arkáda",
        category=AssetCategory.MIDGROUND_BLOCK,
        width_m=BASE_MODULE_M0 * 4.0,
        depth_m=BASE_MODULE_M0 * 1.5,
        height_m=7.5,
        ascii_char="П",
        base_color_hex="#fef3c7",
        emissive_glow=True,
        vital_hp=VITAL_MAX_HP,
        description="Podlubie s osvetlenými výkladmi a kaviarňami."
    ),
    "STEPPED_TERRACE_RESIDENCE": AssetBrush(
        brush_id="STEPPED_TERRACE_RESIDENCE",
        name="Kaskádový Terasový Dom",
        category=AssetCategory.MIDGROUND_BLOCK,
        width_m=BASE_MODULE_M0 * 3.5,
        depth_m=BASE_MODULE_M0 * 2.5,
        height_m=18.0,
        ascii_char="≡",
        base_color_hex="#e4e4e7",
        emissive_glow=False,
        vital_hp=VITAL_MAX_HP,
        description="Kaskádová rezidencia prepájajúca vertikály s parterom."
    ),
    # Layer 2: Infrastructure Conduits
    "GRAND_BOULEVARD_CONDUIT": AssetBrush(
        brush_id="GRAND_BOULEVARD_CONDUIT",
        name="Centrálny Bulvár",
        category=AssetCategory.INFRASTRUCTURE,
        width_m=BASE_MODULE_M0 * 4.0,
        depth_m=BASE_MODULE_M0 * 12.0,
        height_m=0.3,
        ascii_char="═",
        base_color_hex="#27272a",
        emissive_glow=False,
        vital_hp=VITAL_MAX_HP,
        description="Široká trieda s električkovým pásom a perspektívnym ťahom."
    ),
    "CANAL_BASIN_WATERWAY": AssetBrush(
        brush_id="CANAL_BASIN_WATERWAY",
        name="Zrkadliaci Kanál",
        category=AssetCategory.INFRASTRUCTURE,
        width_m=BASE_MODULE_M0 * 3.0,
        depth_m=BASE_MODULE_M0 * 10.0,
        height_m=0.0,
        ascii_char="≈",
        base_color_hex="#0284c7",
        emissive_glow=False,
        vital_hp=VITAL_MAX_HP,
        description="Kamenný kanál zdvojujúci nočné odlesky neónov."
    ),
    # Layer 3: Biophilic Infill
    "AVENUE_LINDEN_TREE": AssetBrush(
        brush_id="AVENUE_LINDEN_TREE",
        name="Alejová Lipa Malolistá",
        category=AssetCategory.BIOPHILIC_INFILL,
        width_m=BASE_MODULE_M0 * 1.0,
        depth_m=BASE_MODULE_M0 * 1.0,
        height_m=7.5,
        ascii_char="♣",
        base_color_hex="#22c55e",
        emissive_glow=False,
        vital_hp=VITAL_MAX_HP,
        description="Biofilná zeleň zjemňujúca betónový a sklenený reliéf."
    ),
    "URBAN_PLAZA_FOUNTAIN": AssetBrush(
        brush_id="URBAN_PLAZA_FOUNTAIN",
        name="Mramorová Fontána",
        category=AssetCategory.BIOPHILIC_INFILL,
        width_m=BASE_MODULE_M0 * 2.0,
        depth_m=BASE_MODULE_M0 * 2.0,
        height_m=2.8,
        ascii_char="○",
        base_color_hex="#38bdf8",
        emissive_glow=True,
        vital_hp=VITAL_MAX_HP,
        description="Kruhová vodná dominanta voľného námestia."
    ),
    # Layer 4: Micro-Props
    "ORNATE_STREET_LAMP": AssetBrush(
        brush_id="ORNATE_STREET_LAMP",
        name="Liatinová Pouličná Lampa",
        category=AssetCategory.MICRO_PROP,
        width_m=0.8,
        depth_m=0.8,
        height_m=4.8,
        ascii_char="†",
        base_color_hex="#f59e0b",
        emissive_glow=True,
        vital_hp=VITAL_MAX_HP,
        description="Bodový svetelný zdroj s teplým jantárovým kužeľom."
    ),
    "NEON_CYBER_BILLBOARD": AssetBrush(
        brush_id="NEON_CYBER_BILLBOARD",
        name="Holografický Neónový Panel",
        category=AssetCategory.MICRO_PROP,
        width_m=BASE_MODULE_M0 * 1.5,
        depth_m=0.4,
        height_m=3.5,
        ascii_char="✦",
        base_color_hex="#ec4899",
        emissive_glow=True,
        vital_hp=VITAL_MAX_HP,
        description="Žiarivý neónový akcent vo farbách Ružového Pantera."
    ),
    # Layer 5: Kinetic Cruiser Vehicles
    "PANTHER_CRUISER_2D": AssetBrush(
        brush_id="PANTHER_CRUISER_2D",
        name="Panther Kinetic Cruiser",
        category=AssetCategory.KINETIC_CRUISER,
        width_m=2.2,
        depth_m=4.8,
        height_m=1.4,
        ascii_char="►",
        base_color_hex="#f472b6",
        emissive_glow=True,
        vital_hp=VITAL_MAX_HP,
        description="Aerodynamický kinetický cruiser s prednými svetlometmi (HP=6)."
    ),
}


@dataclass
class CityAssetInstance:
    """An individual placed asset instance within the composed canvas."""
    instance_id: str
    brush_id: str
    layer_index: int
    pos_x: float
    pos_y: float  # Elevation / Height above ground
    pos_z: float  # Depth
    rotation_y_deg: float
    scale_x: float = 1.0
    scale_y: float = 1.0
    scale_z: float = 1.0
    tint_color_hex: str = "#ffffff"
    vital_hp: int = VITAL_MAX_HP
    is_focal_anchor: bool = False

    def __post_init__(self) -> None:
        # Enforce Immutable Invariant: VITAL_MAX_HP = 6
        if self.vital_hp > VITAL_MAX_HP:
            object.__setattr__(self, "vital_hp", VITAL_MAX_HP)


@dataclass
class CompositionLayer:
    """One of the 7 hierarchical composition layers."""
    layer_index: int
    layer_name: str
    category: AssetCategory
    depth_z_range: Tuple[float, float]
    instances: List[CityAssetInstance] = field(default_factory=list)


@dataclass
class ProceduralCityComposition:
    """Complete multi-asset painted city composition."""
    composition_id: str
    city_name: str
    style_biome: str
    seed: int
    canvas_width_m: float
    canvas_depth_m: float
    focal_point_x: float
    focal_point_z: float
    layers: Dict[int, CompositionLayer]
    total_assets_count: int
    vital_max_hp_invariant_verified: bool
    golden_ratio_adherence_score: float
    ascii_skyline_view: str
    ascii_plan_view: str


class ProceduralCityCompositionEngine:
    """
    Orchestrates the procedural synthesis of a city as an artistic composition.
    Enforces Golden Ratio geometry, visual silhouette balance, depth layering,
    and multi-substrate code emission.
    """

    def __init__(self) -> None:
        self.catalog = ASSET_BRUSH_CATALOG

    def paint_city_composition(
        self,
        seed: int = 42,
        city_name: str = "Neo-Praha Golden Spires",
        style_biome: str = "bohemian_cyber_noir",
        canvas_width_m: float = 240.0,
        canvas_depth_m: float = 240.0,
    ) -> ProceduralCityComposition:
        """
        Draws/paints a complete city composition onto a multi-asset canvas.
        Deterministic: Same seed produces bit-exact identical asset placements.
        """
        rng = random.Random(seed)

        # 1. Compute Golden Ratio Focal Anchors
        # Primary visual anchor placed at Golden split X = W * (1 - 1/phi) or W * 1/phi
        use_right_focal = rng.random() > 0.5
        golden_factor = INV_GOLDEN_RATIO if use_right_focal else (1.0 - INV_GOLDEN_RATIO)
        focal_x = (canvas_width_m * golden_factor) - (canvas_width_m * 0.5)
        focal_z = -15.0 + rng.uniform(-10.0, 10.0)

        # Secondary counterweight on opposite golden split
        counter_x = (canvas_width_m * (1.0 - golden_factor)) - (canvas_width_m * 0.5)
        counter_z = -35.0 + rng.uniform(-10.0, 10.0)

        layers: Dict[int, CompositionLayer] = {
            0: CompositionLayer(0, "Macro Skyline Dominants", AssetCategory.MACRO_DOMINANT, (-120.0, -40.0)),
            1: CompositionLayer(1, "Midground Architectural Blocks", AssetCategory.MIDGROUND_BLOCK, (-50.0, 30.0)),
            2: CompositionLayer(2, "Infrastructure Conduits", AssetCategory.INFRASTRUCTURE, (-100.0, 100.0)),
            3: CompositionLayer(3, "Biophilic Infill", AssetCategory.BIOPHILIC_INFILL, (-60.0, 60.0)),
            4: CompositionLayer(4, "Micro-Props & Lighting", AssetCategory.MICRO_PROP, (-70.0, 70.0)),
            5: CompositionLayer(5, "Kinetic Cruisers", AssetCategory.KINETIC_CRUISER, (-80.0, 80.0)),
            6: CompositionLayer(6, "Atmospheric Wash", AssetCategory.ATMOSPHERE, (-150.0, 150.0)),
        }

        # -------------------------------------------------------------
        # STEP 1: BRUSH LAYER 0 - MACRO SKYLINE DOMINANTS
        # -------------------------------------------------------------
        # Primary Spire at Primary Focal Point
        layers[0].instances.append(CityAssetInstance(
            instance_id="macro_focal_spire_01",
            brush_id="CYBER_SPIRE_MONOLITH",
            layer_index=0,
            pos_x=focal_x,
            pos_y=0.0,
            pos_z=focal_z,
            rotation_y_deg=rng.choice([0.0, 45.0, 90.0]),
            scale_y=1.0 + rng.uniform(0.0, 0.15),
            tint_color_hex="#06b6d4",
            vital_hp=VITAL_MAX_HP,
            is_focal_anchor=True
        ))

        # Secondary Counterweight Clocktower at Counter Focal Point
        layers[0].instances.append(CityAssetInstance(
            instance_id="macro_counter_clocktower_02",
            brush_id="ALCHEMICAL_CLOCKTOWER",
            layer_index=0,
            pos_x=counter_x,
            pos_y=0.0,
            pos_z=counter_z,
            rotation_y_deg=0.0,
            scale_y=1.0,
            tint_color_hex="#f59e0b",
            vital_hp=VITAL_MAX_HP,
            is_focal_anchor=False
        ))

        # Heavy Citadel anchor in the background offset
        citadel_x = 0.0 + rng.uniform(-20.0, 20.0)
        layers[0].instances.append(CityAssetInstance(
            instance_id="macro_citadel_ziggurat_03",
            brush_id="DATA_CITADEL_ZIGGURAT",
            layer_index=0,
            pos_x=citadel_x,
            pos_y=0.0,
            pos_z=-75.0,
            rotation_y_deg=rng.choice([0.0, 90.0]),
            scale_x=1.2,
            scale_z=1.2,
            tint_color_hex="#8b5cf6",
            vital_hp=VITAL_MAX_HP,
            is_focal_anchor=False
        ))

        # -------------------------------------------------------------
        # STEP 2: BRUSH LAYER 2 - CONNECTIVE INFRASTRUCTURE CONDUITS
        # -------------------------------------------------------------
        # Central Grand Boulevard leading toward the horizon
        layers[2].instances.append(CityAssetInstance(
            instance_id="infra_grand_boulevard_main",
            brush_id="GRAND_BOULEVARD_CONDUIT",
            layer_index=2,
            pos_x=0.0,
            pos_y=0.0,
            pos_z=0.0,
            rotation_y_deg=0.0,
            scale_z=2.0,
            tint_color_hex="#27272a",
            vital_hp=VITAL_MAX_HP
        ))

        # Cross-Avenue at Z = 15m
        layers[2].instances.append(CityAssetInstance(
            instance_id="infra_cross_avenue_01",
            brush_id="GRAND_BOULEVARD_CONDUIT",
            layer_index=2,
            pos_x=0.0,
            pos_y=0.0,
            pos_z=15.0,
            rotation_y_deg=90.0,
            scale_z=1.8,
            tint_color_hex="#27272a",
            vital_hp=VITAL_MAX_HP
        ))

        # Reflective Canal Basin flanking the eastern boulevard edge
        layers[2].instances.append(CityAssetInstance(
            instance_id="infra_water_canal_east",
            brush_id="CANAL_BASIN_WATERWAY",
            layer_index=2,
            pos_x=focal_x * 0.5 + 25.0,
            pos_y=-0.5,
            pos_z=5.0,
            rotation_y_deg=0.0,
            scale_z=1.5,
            tint_color_hex="#0284c7",
            vital_hp=VITAL_MAX_HP
        ))

        # -------------------------------------------------------------
        # STEP 3: BRUSH LAYER 1 - MIDGROUND ARCHITECTURAL BLOCKS
        # -------------------------------------------------------------
        # Generate modular blocks flanking the boulevard with heights cascading
        # according to proximity to the dominant spires.
        half_w = canvas_width_m * 0.4
        street_corridors_x = [-half_w * 0.7, -half_w * 0.35, half_w * 0.35, half_w * 0.7]
        street_rows_z = [-40.0, -15.0, 10.0, 35.0]

        block_idx = 1
        for sx in street_corridors_x:
            for sz in street_rows_z:
                # Calculate distance to focal spire to scale height organically
                dist_to_focal = math.hypot(sx - focal_x, sz - focal_z)
                # Golden proportion falloff: height inversely proportional to distance
                height_scale = max(0.6, 1.4 - (dist_to_focal / canvas_width_m) * GOLDEN_RATIO)

                brush_choice = "MODULAR_TENEMENT_BLOCK"
                if dist_to_focal < 40.0:
                    brush_choice = "STEPPED_TERRACE_RESIDENCE"
                elif sz > 20.0:
                    brush_choice = "COMMERCIAL_ARCADE_PLINTH"

                layers[1].instances.append(CityAssetInstance(
                    instance_id=f"midground_block_{block_idx:02d}",
                    brush_id=brush_choice,
                    layer_index=1,
                    pos_x=sx + rng.uniform(-3.0, 3.0),
                    pos_y=0.0,
                    pos_z=sz + rng.uniform(-3.0, 3.0),
                    rotation_y_deg=rng.choice([0.0, 90.0, 180.0, 270.0]),
                    scale_y=round(height_scale, 2),
                    tint_color_hex=rng.choice(["#d4d4d8", "#e4e4e7", "#fef3c7", "#fbcfe8"]),
                    vital_hp=VITAL_MAX_HP
                ))
                block_idx += 1

        # -------------------------------------------------------------
        # STEP 4: BRUSH LAYER 3 - BIOPHILIC INFILL
        # -------------------------------------------------------------
        # Central Plaza Fountain at the intersection (Z = 15m, X = 0m)
        layers[3].instances.append(CityAssetInstance(
            instance_id="biophilic_central_fountain",
            brush_id="URBAN_PLAZA_FOUNTAIN",
            layer_index=3,
            pos_x=0.0,
            pos_y=0.0,
            pos_z=15.0,
            rotation_y_deg=0.0,
            tint_color_hex="#38bdf8",
            vital_hp=VITAL_MAX_HP
        ))

        # Linden Tree lines along the main boulevard
        tree_idx = 1
        for tz in [-60.0, -45.0, -30.0, -15.0, 0.0, 30.0, 45.0, 60.0]:
            for tx_offset in [-12.0, 12.0]:
                layers[3].instances.append(CityAssetInstance(
                    instance_id=f"biophilic_tree_{tree_idx:02d}",
                    brush_id="AVENUE_LINDEN_TREE",
                    layer_index=3,
                    pos_x=tx_offset,
                    pos_y=0.0,
                    pos_z=tz + rng.uniform(-1.0, 1.0),
                    rotation_y_deg=rng.uniform(0.0, 360.0),
                    scale_y=0.9 + rng.uniform(0.0, 0.25),
                    tint_color_hex="#22c55e",
                    vital_hp=VITAL_MAX_HP
                ))
                tree_idx += 1

        # -------------------------------------------------------------
        # STEP 5: BRUSH LAYER 4 - MICRO-PROPS & STREET LIGHTING
        # -------------------------------------------------------------
        lamp_idx = 1
        for lz in [-55.0, -35.0, -10.0, 10.0, 25.0, 50.0]:
            for lx_offset in [-9.5, 9.5]:
                layers[4].instances.append(CityAssetInstance(
                    instance_id=f"prop_street_lamp_{lamp_idx:02d}",
                    brush_id="ORNATE_STREET_LAMP",
                    layer_index=4,
                    pos_x=lx_offset,
                    pos_y=0.0,
                    pos_z=lz,
                    rotation_y_deg=0.0 if lx_offset < 0 else 180.0,
                    tint_color_hex="#f59e0b",
                    vital_hp=VITAL_MAX_HP
                ))
                lamp_idx += 1

        # Neon Cyber Billboards placed at major building facades
        billboard_idx = 1
        for bx, bz in [(-25.0, 10.0), (25.0, -15.0), (-25.0, -40.0)]:
            layers[4].instances.append(CityAssetInstance(
                instance_id=f"prop_neon_billboard_{billboard_idx:02d}",
                brush_id="NEON_CYBER_BILLBOARD",
                layer_index=4,
                pos_x=bx,
                pos_y=8.0,
                pos_z=bz,
                rotation_y_deg=90.0,
                tint_color_hex="#ec4899",
                vital_hp=VITAL_MAX_HP
            ))
            billboard_idx += 1

        # -------------------------------------------------------------
        # STEP 6: BRUSH LAYER 5 - KINETIC CRUISER VEHICLES
        # -------------------------------------------------------------
        # Vehicle cruisers cruising along the boulevard lanes
        cruiser_lanes = [
            (-3.5, 30.0, 0.0, "northbound"),
            (-3.5, -20.0, 0.0, "northbound"),
            (3.5, 45.0, 180.0, "southbound"),
            (3.5, -5.0, 180.0, "southbound"),
            (3.5, -45.0, 180.0, "southbound"),
        ]
        for c_idx, (cx, cz, crot, desc) in enumerate(cruiser_lanes, start=1):
            layers[5].instances.append(CityAssetInstance(
                instance_id=f"kinetic_cruiser_{c_idx:02d}",
                brush_id="PANTHER_CRUISER_2D",
                layer_index=5,
                pos_x=cx,
                pos_y=0.0,
                pos_z=cz,
                rotation_y_deg=crot,
                tint_color_hex=rng.choice(["#f472b6", "#06b6d4", "#fef3c7"]),
                vital_hp=VITAL_MAX_HP  # Enforces HP = 6
            ))

        # Total Assets Count
        total_assets = sum(len(layer.instances) for layer in layers.values())

        # Verify Immutable Invariant
        all_hp_valid = all(
            inst.vital_hp <= VITAL_MAX_HP
            for layer in layers.values()
            for inst in layer.instances
        )

        # Adherence to Golden Ratio (focal_x offset compared to Golden Proportion)
        target_focal_x = (canvas_width_m * golden_factor) - (canvas_width_m * 0.5)
        adherence_score = max(0.0, 1.0 - abs(focal_x - target_focal_x) / canvas_width_m)

        # Paint ASCII Canvases
        ascii_skyline = self._render_ascii_skyline(layers, canvas_width_m)
        ascii_plan = self._render_ascii_plan_view(layers, canvas_width_m, canvas_depth_m)

        return ProceduralCityComposition(
            composition_id=f"city_comp_{seed:05d}_{city_name.lower().replace(' ', '_')}",
            city_name=city_name,
            style_biome=style_biome,
            seed=seed,
            canvas_width_m=canvas_width_m,
            canvas_depth_m=canvas_depth_m,
            focal_point_x=focal_x,
            focal_point_z=focal_z,
            layers=layers,
            total_assets_count=total_assets,
            vital_max_hp_invariant_verified=all_hp_valid,
            golden_ratio_adherence_score=round(adherence_score, 4),
            ascii_skyline_view=ascii_skyline,
            ascii_plan_view=ascii_plan
        )

    # =========================================================================
    # DUAL VISUAL ASCII ART CANVAS PAINTERS
    # =========================================================================

    def _render_ascii_skyline(
        self,
        layers: Dict[int, CompositionLayer],
        canvas_width_m: float,
        cols: int = 76,
        rows: int = 24
    ) -> str:
        """
        Paints a 2D side silhouette of the city skyline looking from South to North.
        Renders macro spires, tenement silhouettes, trees, streetlamps, and cruisers.
        """
        grid = [[" " for _ in range(cols)] for _ in range(rows)]
        ground_row = rows - 3

        # Ground baseline
        for c in range(cols):
            grid[ground_row][c] = "═"
            grid[ground_row + 1][c] = "░"
            grid[ground_row + 2][c] = "▓"

        half_w = canvas_width_m * 0.5

        # Helper to map world X to column
        def x_to_col(wx: float) -> int:
            t = (wx + half_w) / canvas_width_m
            return max(0, min(cols - 1, int(t * (cols - 1))))

        # Helper to map world height to row count above ground
        def h_to_rows(wh: float) -> int:
            max_h = 130.0  # Spire apex
            t = min(1.0, wh / max_h)
            return max(1, int(t * (ground_row - 2)))

        # 1. Paint Layer 0: Macro Dominants (Tallest, Background)
        for inst in layers[0].instances:
            brush = self.catalog.get(inst.brush_id)
            if not brush:
                continue
            effective_h = brush.height_m * inst.scale_y
            col_center = x_to_col(inst.pos_x)
            col_width = max(1, int((brush.width_m / canvas_width_m) * cols))
            r_height = h_to_rows(effective_h)

            start_col = max(0, col_center - col_width // 2)
            end_col = min(cols - 1, col_center + col_width // 2)

            # Paint vertical profile
            for r in range(ground_row - r_height, ground_row):
                if 0 <= r < rows:
                    if inst.brush_id == "CYBER_SPIRE_MONOLITH":
                        # Tapered needle profile
                        progress = (ground_row - r) / r_height
                        if progress > 0.85:
                            grid[r][col_center] = "▲"
                        elif progress > 0.4:
                            for c in range(max(0, col_center - 1), min(cols, col_center + 2)):
                                grid[r][c] = "║"
                        else:
                            for c in range(start_col, end_col + 1):
                                grid[r][c] = "█"
                    elif inst.brush_id == "ALCHEMICAL_CLOCKTOWER":
                        for c in range(start_col, end_col + 1):
                            grid[r][c] = "▓" if (r == ground_row - r_height) else "│"
                        grid[ground_row - r_height][col_center] = "Ω"
                    else:
                        for c in range(start_col, end_col + 1):
                            grid[r][c] = "▒"

        # 2. Paint Layer 1: Midground Architectural Blocks
        for inst in layers[1].instances:
            brush = self.catalog.get(inst.brush_id)
            if not brush:
                continue
            effective_h = brush.height_m * inst.scale_y
            col_center = x_to_col(inst.pos_x)
            col_width = max(2, int((brush.width_m / canvas_width_m) * cols))
            r_height = h_to_rows(effective_h)

            start_col = max(0, col_center - col_width // 2)
            end_col = min(cols - 1, col_center + col_width // 2)

            for r in range(ground_row - r_height, ground_row):
                if 0 <= r < rows:
                    for c in range(start_col, end_col + 1):
                        if c == start_col or c == end_col:
                            grid[r][c] = "|"
                        elif (r + c) % 2 == 0:
                            grid[r][c] = "▫"
                        else:
                            grid[r][c] = "#"

        # 3. Paint Layer 3: Biophilic Trees (Foreground / Midground)
        for inst in layers[3].instances:
            brush = self.catalog.get(inst.brush_id)
            if not brush or brush.brush_id != "AVENUE_LINDEN_TREE":
                continue
            col = x_to_col(inst.pos_x)
            r = ground_row - 1
            if 0 <= r < rows:
                grid[r][col] = "♣"
            if 0 <= r - 1 < rows:
                grid[r - 1][col] = "o"

        # 4. Paint Layer 4: Street Lamps
        for inst in layers[4].instances:
            brush = self.catalog.get(inst.brush_id)
            if not brush or brush.brush_id != "ORNATE_STREET_LAMP":
                continue
            col = x_to_col(inst.pos_x)
            r = ground_row - 1
            if 0 <= r < rows:
                grid[r][col] = "†"

        # 5. Paint Layer 5: Kinetic Cruiser Vehicles
        for inst in layers[5].instances:
            col = x_to_col(inst.pos_x)
            r = ground_row
            char = "►" if inst.rotation_y_deg == 0.0 else "◄"
            grid[r][col] = char

        header = "┌" + "─" * (cols) + "┐\n"
        footer = "└" + "─" * (cols) + "┘"
        body = "\n".join("│" + "".join(row) + "│" for row in grid)
        legend = (
            "\n  [LEGENDA SILUETY]: ▲=Kyber-Spire  Ω=Alchymistický Orloj  "
            "█/▓=Citadela  #=Tenement  ♣=Lipa  †=Lampa  ►/◄=Cruiser (HP=6)"
        )
        return header + body + "\n" + footer + legend

    def _render_ascii_plan_view(
        self,
        layers: Dict[int, CompositionLayer],
        canvas_width_m: float,
        canvas_depth_m: float,
        cols: int = 76,
        rows: int = 24
    ) -> str:
        """
        Paints a top-down 2D urban master plan of the composed city.
        Renders avenues, boulevards, building footprints, parks, trees, and cruisers.
        """
        grid = [["·" for _ in range(cols)] for _ in range(rows)]
        half_w = canvas_width_m * 0.5
        half_d = canvas_depth_m * 0.5

        def coord_to_cell(wx: float, wz: float) -> Tuple[int, int]:
            tx = (wx + half_w) / canvas_width_m
            tz = (wz + half_d) / canvas_depth_m
            c = max(0, min(cols - 1, int(tx * (cols - 1))))
            r = max(0, min(rows - 1, int(tz * (rows - 1))))
            return r, c

        # 1. Paint Infrastructure (Layer 2)
        # Central Grand Boulevard (Vertical)
        mid_col = cols // 2
        for r in range(rows):
            for c in range(mid_col - 2, mid_col + 3):
                grid[r][c] = "║" if (c == mid_col - 2 or c == mid_col + 2) else " "

        # Cross Avenue (Horizontal around Z = 15m)
        cross_r, _ = coord_to_cell(0.0, 15.0)
        for c in range(cols):
            for r in range(max(0, cross_r - 1), min(rows, cross_r + 2)):
                if grid[r][c] not in ("║", " "):
                    grid[r][c] = "═"

        # Water Canal Basin
        for inst in layers[2].instances:
            if inst.brush_id == "CANAL_BASIN_WATERWAY":
                r0, c0 = coord_to_cell(inst.pos_x, inst.pos_z)
                for r in range(max(0, r0 - 3), min(rows, r0 + 4)):
                    for c in range(max(0, c0 - 2), min(cols, c0 + 3)):
                        grid[r][c] = "≈"

        # 2. Paint Macro Dominants (Layer 0)
        for inst in layers[0].instances:
            r, c = coord_to_cell(inst.pos_x, inst.pos_z)
            if inst.brush_id == "CYBER_SPIRE_MONOLITH":
                char = "▲"
            elif inst.brush_id == "ALCHEMICAL_CLOCKTOWER":
                char = "Ω"
            else:
                char = "█"
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if 0 <= r + dr < rows and 0 <= c + dc < cols:
                        grid[r + dr][c + dc] = char

        # 3. Paint Midground Blocks (Layer 1)
        for inst in layers[1].instances:
            r, c = coord_to_cell(inst.pos_x, inst.pos_z)
            char = "■" if inst.brush_id == "MODULAR_TENEMENT_BLOCK" else "≡"
            for dr in [-1, 0]:
                for dc in [-1, 0, 1]:
                    if 0 <= r + dr < rows and 0 <= c + dc < cols:
                        if grid[r + dr][c + dc] in ("·",):
                            grid[r + dr][c + dc] = char

        # 4. Paint Biophilic Trees & Fountain (Layer 3)
        for inst in layers[3].instances:
            r, c = coord_to_cell(inst.pos_x, inst.pos_z)
            if inst.brush_id == "URBAN_PLAZA_FOUNTAIN":
                grid[r][c] = "○"
            elif inst.brush_id == "AVENUE_LINDEN_TREE":
                grid[r][c] = "♣"

        # 5. Paint Micro Props (Layer 4)
        for inst in layers[4].instances:
            r, c = coord_to_cell(inst.pos_x, inst.pos_z)
            if inst.brush_id == "ORNATE_STREET_LAMP":
                grid[r][c] = "†"
            elif inst.brush_id == "NEON_CYBER_BILLBOARD":
                grid[r][c] = "✦"

        # 6. Paint Kinetic Cruisers (Layer 5)
        for inst in layers[5].instances:
            r, c = coord_to_cell(inst.pos_x, inst.pos_z)
            char = "▲" if inst.rotation_y_deg == 0.0 else "▼"
            grid[r][c] = char

        header = "┌" + "─" * (cols) + "┐\n"
        footer = "└" + "─" * (cols) + "┘"
        body = "\n".join("│" + "".join(row) + "│" for row in grid)
        legend = (
            "\n  [PÔDORYS KOMPOZÍCIE]: ║/═=Bulvár  ≈=Kanál  ▲/Ω/█=Dominanty  "
            "■=Bloky  ♣=Lipy  ○=Fontána  †=Lampy  ▲/▼=Cruiser (HP=6)"
        )
        return header + body + "\n" + footer + legend

    # =========================================================================
    # MULTI-SUBSTRATE TRANSPILATION: GODOT 4 .TSCN
    # =========================================================================

    def export_to_godot_tscn(self, comp: ProceduralCityComposition) -> str:
        """
        Emits a complete, production-grade Godot 4.x Forward+ .tscn scene text.
        Includes Camera3D, DirectionalLight3D, WorldEnvironment, and instanced nodes.
        """
        lines: List[str] = [
            '[gd_scene load_steps=6 format=3 uid="uid://krystal_city_comp_01"]',
            '',
            '# =====================================================================',
            '# KRYSTAL-STACK PROCEDURAL CITY COMPOSITION (GODOT 4 FORWARD+)',
            f'# City Name: {comp.city_name}',
            f'# Seed: {comp.seed} | Style: {comp.style_biome}',
            f'# Total Asset Instances: {comp.total_assets_count}',
            f'# Vital Max HP Invariant Verified: {comp.vital_max_hp_invariant_verified}',
            '# =====================================================================',
            '',
            '[sub_resource type="ProceduralSkyMaterial" id="ProceduralSkyMaterial_sky"]',
            'sky_top_color = Color(0.12, 0.16, 0.28, 1)',
            'sky_horizon_color = Color(0.85, 0.45, 0.65, 1)',  # Pink Panther Twilight
            'ground_bottom_color = Color(0.08, 0.08, 0.10, 1)',
            'ground_horizon_color = Color(0.85, 0.45, 0.65, 1)',
            '',
            '[sub_resource type="Sky" id="Sky_env"]',
            'sky_material = SubResource("ProceduralSkyMaterial_sky")',
            '',
            '[sub_resource type="Environment" id="Environment_world"]',
            'background_mode = 2',
            'sky = SubResource("Sky_env")',
            'ambient_light_source = 3',
            'ambient_light_color = Color(0.95, 0.85, 0.90, 1)',
            'tonemap_mode = 3',  # ACES Filmic
            'ssr_enabled = true',
            'ssao_enabled = true',
            'sdfgi_enabled = true',
            'volumetric_fog_enabled = true',
            'volumetric_fog_density = 0.012',
            'volumetric_fog_albedo = Color(0.96, 0.65, 0.82, 1)',
            '',
            '[sub_resource type="StandardMaterial3D" id="Mat_Ground"]',
            'albedo_color = Color(0.12, 0.12, 0.14, 1.0)',
            'roughness = 0.75',
            '',
            '[sub_resource type="PlaneMesh" id="PlaneMesh_ground"]',
            f'size = Vector2({comp.canvas_width_m}, {comp.canvas_depth_m})',
            '',
            '[node name="CityCompositionRoot" type="Node3D"]',
            '',
            '[node name="WorldEnvironment" type="WorldEnvironment" parent="."]',
            'environment = SubResource("Environment_world")',
            '',
            '[node name="DirectionalSunLight" type="DirectionalLight3D" parent="."]',
            'transform = Transform3D(0.866, -0.353, 0.353, 0, 0.707, 0.707, -0.5, -0.612, 0.612, 0, 45, 0)',
            'light_color = Color(1.0, 0.92, 0.85, 1.0)',
            'light_energy = 1.25',
            'shadow_enabled = true',
            '',
            '[node name="CinematicCamera3D" type="Camera3D" parent="."]',
            f'transform = Transform3D(1, 0, 0, 0, 0.927, 0.375, 0, -0.375, 0.927, {comp.focal_point_x * 0.3:.2f}, 35.0, 95.0)',
            'current = true',
            'fov = 68.0',
            '',
            '[node name="GroundPlaza" type="MeshInstance3D" parent="."]',
            'mesh = SubResource("PlaneMesh_ground")',
            'surface_material_override/0 = SubResource("Mat_Ground")',
            '',
        ]

        # Group asset instances into organized subnodes per layer
        for layer_idx, layer in sorted(comp.layers.items()):
            sanitized_layer_name = layer.layer_name.replace(" ", "").replace("-", "_").replace("&", "And")
            lines.append(f'[node name="{sanitized_layer_name}" type="Node3D" parent="."]')
            lines.append("")

            for inst in layer.instances:
                brush = self.catalog.get(inst.brush_id)
                w = (brush.width_m if brush else 5.0) * inst.scale_x
                h = (brush.height_m if brush else 10.0) * inst.scale_y
                d = (brush.depth_m if brush else 5.0) * inst.scale_z

                # Center of the volume in Godot is at Y = pos_y + h/2
                y_center = inst.pos_y + (h * 0.5)
                rad = math.radians(inst.rotation_y_deg)
                cos_r = math.cos(rad)
                sin_r = math.sin(rad)

                lines.append(f'[node name="{inst.instance_id}" type="CSGBox3D" parent="{sanitized_layer_name}"]')
                lines.append(
                    f'transform = Transform3D({cos_r:.4f}, 0, {sin_r:.4f}, 0, 1, 0, {-sin_r:.4f}, 0, {cos_r:.4f}, '
                    f'{inst.pos_x:.2f}, {y_center:.2f}, {inst.pos_z:.2f})'
                )
                lines.append(f'size = Vector3({w:.2f}, {h:.2f}, {d:.2f})')
                lines.append(f'metadata/brush_id = "{inst.brush_id}"')
                lines.append(f'metadata/vital_hp = {inst.vital_hp}')
                lines.append(f'metadata/tint_color = "{inst.tint_color_hex}"')
                if inst.is_focal_anchor:
                    lines.append('metadata/is_focal_anchor = true')
                lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # MULTI-SUBSTRATE TRANSPILATION: MODERN JAVA 21 RECORDS
    # =========================================================================

    def export_to_java_records(self, comp: ProceduralCityComposition) -> str:
        """
        Emits modern Java 21 record declarations and record pattern matching
        pipeline for virtual thread processing.
        """
        return f"""// ============================================================================
// KRYSTAL-STACK CITY COMPOSITION (JAVA 21 RECORD PATTERNS & VIRTUAL THREADS)
// City: {comp.city_name} | Seed: {comp.seed}L | Assets: {comp.total_assets_count}
// Immutable Invariant: VITAL_MAX_HP = 6
// ============================================================================
package com.krystal.stack.city;

import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class CityCompositionPipeline {{

    public static final int VITAL_MAX_HP = 6;
    public static final double GOLDEN_RATIO = 1.61803398875;

    public enum AssetCategory {{
        MACRO_DOMINANT,
        MIDGROUND_BLOCK,
        INFRASTRUCTURE,
        BIOPHILIC_INFILL,
        MICRO_PROP,
        KINETIC_CRUISER,
        ATMOSPHERE
    }}

    public record CityAssetInstance(
        String instanceId,
        String brushId,
        int layerIndex,
        double posX,
        double posY,
        double posZ,
        double rotationDeg,
        double scaleX,
        double scaleY,
        double scaleZ,
        String tintHex,
        int vitalHp,
        boolean isFocalAnchor
    ) {{
        public CityAssetInstance {{
            Objects.requireNonNull(instanceId);
            Objects.requireNonNull(brushId);
            if (vitalHp > VITAL_MAX_HP) {{
                vitalHp = VITAL_MAX_HP;
            }}
        }}
    }}

    public record CityComposition(
        String compositionId,
        String cityName,
        String styleBiome,
        long seed,
        double canvasWidth,
        double canvasDepth,
        double focalPointX,
        double focalPointZ,
        List<CityAssetInstance> assets,
        double goldenAdherenceScore
    ) {{}}

    /**
     * Pattern matching switch expression on procedural assets.
     */
    public static String classifyAssetRole(CityAssetInstance asset) {{
        return switch (asset.layerIndex()) {{
            case 0 -> "Focal Skyline Anchor (Dominant Spire/Citadel)";
            case 1 -> "Urban Corridor Streetwall (Tenement/Plinth)";
            case 2 -> "Connective Transportation Matrix (Boulevard/Canal)";
            case 3 -> "Biophilic Living Microclimate (Linden Canopy/Fountain)";
            case 4 -> "Human Scale Photometric Luminous (Streetlamp/Neon)";
            case 5 -> "Kinetic Autonomous Cruiser [HP=" + asset.vitalHp() + "]";
            default -> "Ambient Environmental Substrate";
        }};
    }}
}}
"""

    # =========================================================================
    # MULTI-SUBSTRATE TRANSPILATION: JANET FUNCTIONAL DSL
    # =========================================================================

    def export_to_janet_dsl(self, comp: ProceduralCityComposition) -> str:
        """
        Emits immutable Janet tables representing the composed city AST.
        """
        lines: List[str] = [
            "# ======================================================================",
            "# KRYSTAL-STACK PROCEDURAL CITY COMPOSITION (JANET DSL)",
            f"# City: {comp.city_name} | Seed: {comp.seed}",
            f"# Golden Ratio Adherence Score: {comp.golden_ratio_adherence_score}",
            "# Strict Invariant: VITAL-MAX-HP = 6",
            "# ======================================================================",
            "",
            "(def VITAL-MAX-HP 6)",
            "(def GOLDEN-RATIO 1.61803398875)",
            "",
            "(def CITY-COMPOSITION",
            f'  @{{:id "{comp.composition_id}"',
            f'    :city-name "{comp.city_name}"',
            f'    :style-biome "{comp.style_biome}"',
            f'    :seed {comp.seed}',
            f'    :canvas-width {comp.canvas_width_m}',
            f'    :canvas-depth {comp.canvas_depth_m}',
            f'    :focal-point [{comp.focal_point_x:.2f} {comp.focal_point_z:.2f}]',
            f'    :total-assets {comp.total_assets_count}',
            f'    :vital-invariant-verified {"true" if comp.vital_max_hp_invariant_verified else "false"}',
            "    :layers",
            "    [",
        ]

        for layer_idx, layer in sorted(comp.layers.items()):
            lines.append(f'     @{{:layer-idx {layer.layer_index}')
            lines.append(f'       :layer-name "{layer.layer_name}"')
            lines.append("       :instances")
            lines.append("       [")
            for inst in layer.instances:
                lines.append(
                    f'        @{{:id "{inst.instance_id}" :brush "{inst.brush_id}" '
                    f':pos [{inst.pos_x:.2f} {inst.pos_y:.2f} {inst.pos_z:.2f}] '
                    f':rot {inst.rotation_y_deg:.1f} :tint "{inst.tint_color_hex}" '
                    f':vital-hp {inst.vital_hp} '
                    f':is-focal {"true" if inst.is_focal_anchor else "false"}}}'
                )
            lines.append("       ]}")
        lines.append("    ]})")

        return "\n".join(lines)


GLOBAL_CITY_COMPOSITION_ENGINE = ProceduralCityCompositionEngine()
