"""
KRYSTAL-STACK // GODOT INTERIOR-SURFACE PROCEDURAL NODE ENGINE
=============================================================================
Simulates surface materials and patterns derived from Interior Design principles
(biophilic living walls, harmonic spatial proportions, herringbone parquet,
travertine marble, terrazzo, acoustic slat panels, micro-cement) and bridges
them to macro-scale procedural pre-generation of:
  - Trees & Vegetation (MultiMeshInstance3D batching, L-System branching)
  - Buildings & Architecture (modular floor plates, facade rhythm, column grids)
  - Cities & Urban Superblocks (pedestrian concourses, transit veins, zoning)
  - Soil, Substrates & Microclimate (porosity, soil horizons, moisture retention)
  - Godot 4.x Engine Process Integration (_physics_process budgeting, LOD bands)

Strict Invariants:
  - VITAL_MAX_HP = 6
  - GOLDEN_RATIO = 1.61803398875
  - INV_GOLDEN_RATIO = 0.61803398875
=============================================================================
"""

import math
import random
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 0.61803398875


# =============================================================================
# 1. INTERIOR DESIGN MATERIAL PATTERNS SPECIFICATION
# =============================================================================

@dataclass
class InteriorMaterialPattern:
    """Represents an interior design surface material with PBR and spatial properties."""
    material_id: str
    name: str
    category: str  # "wood", "stone", "biophilic", "mineral", "metal", "composite"
    base_color: List[float]  # RGBA [0.0 - 1.0]
    roughness: float
    metallic: float
    normal_depth: float
    subsurface_scatter: float
    pattern_scale: float
    thermal_conductivity: float  # W/(m*K)
    acoustic_absorption: float   # NRC coefficient [0.0 - 1.0]
    biophilic_resonance: float   # Subjective psychological wellbeing score [0.0 - 1.0]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "material_id": self.material_id,
            "name": self.name,
            "category": self.category,
            "base_color": self.base_color,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "normal_depth": self.normal_depth,
            "subsurface_scatter": self.subsurface_scatter,
            "pattern_scale": self.pattern_scale,
            "thermal_conductivity": self.thermal_conductivity,
            "acoustic_absorption": self.acoustic_absorption,
            "biophilic_resonance": self.biophilic_resonance,
            "description": self.description
        }


INTERIOR_MATERIALS_CATALOG: Dict[str, InteriorMaterialPattern] = {
    "oak_herringbone_parquet": InteriorMaterialPattern(
        material_id="oak_herringbone_parquet",
        name="Prémiové Dubové Rybie Parkety (Herringbone)",
        category="wood",
        base_color=[0.68, 0.48, 0.32, 1.0],
        roughness=0.38,
        metallic=0.02,
        normal_depth=0.85,
        subsurface_scatter=0.15,
        pattern_scale=1.618,
        thermal_conductivity=0.17,
        acoustic_absorption=0.45,
        biophilic_resonance=0.88,
        description="Klasický interiérový vzor podlahy s 45-stupňovým skosením lamiel prináša pocit stability a prírodného rytmu."
    ),
    "roman_travertine_stone": InteriorMaterialPattern(
        material_id="roman_travertine_stone",
        name="Rímsky Pórovitý Travertín",
        category="stone",
        base_color=[0.86, 0.82, 0.73, 1.0],
        roughness=0.55,
        metallic=0.04,
        normal_depth=0.92,
        subsurface_scatter=0.22,
        pattern_scale=2.4,
        thermal_conductivity=1.45,
        acoustic_absorption=0.25,
        biophilic_resonance=0.76,
        description="Svetlý vápenatý kameň s prírodnými dutinami a jemným žilkovaním, chladivý na dotyk s vysokou tepelnou zotrvačnosťou."
    ),
    "venetian_terrazzo": InteriorMaterialPattern(
        material_id="venetian_terrazzo",
        name="Benátske Terazzo s Mramorovou Drvou",
        category="composite",
        base_color=[0.92, 0.90, 0.88, 1.0],
        roughness=0.18,
        metallic=0.08,
        normal_depth=0.40,
        subsurface_scatter=0.12,
        pattern_scale=1.2,
        thermal_conductivity=1.30,
        acoustic_absorption=0.15,
        biophilic_resonance=0.65,
        description="Leštená zmes cementového spojiva a pestrofarebných mramorových čiastočiek, vysoká odolnosť a minerálny lesk."
    ),
    "biophilic_vertical_moss": InteriorMaterialPattern(
        material_id="biophilic_vertical_moss",
        name="Biofilná Živá Machová Stena (Moss Wall)",
        category="biophilic",
        base_color=[0.18, 0.65, 0.22, 1.0],
        roughness=0.92,
        metallic=0.0,
        normal_depth=1.45,
        subsurface_scatter=0.68,
        pattern_scale=0.8,
        thermal_conductivity=0.06,
        acoustic_absorption=0.85,
        biophilic_resonance=0.98,
        description="Hustý stabilizovaný lišajník a mach absorbujúci ozvenu, regulujúci vlhkosť a maximalizujúci pokoj mysle."
    ),
    "acoustic_slat_oak_felt": InteriorMaterialPattern(
        material_id="acoustic_slat_oak_felt",
        name="Akustické Dubové Lamely na Čiernej Plsti",
        category="wood",
        base_color=[0.45, 0.32, 0.22, 1.0],
        roughness=0.60,
        metallic=0.02,
        normal_depth=1.80,
        subsurface_scatter=0.10,
        pattern_scale=1.0,
        thermal_conductivity=0.12,
        acoustic_absorption=0.92,
        biophilic_resonance=0.84,
        description="Pravidelný zvislý rytmus drevených líšt na zvukovo nepriepustnom recyklovanom filci pre dokonalé akustické prostredie."
    ),
    "brushed_champagne_brass": InteriorMaterialPattern(
        material_id="brushed_champagne_brass",
        name="Kefovaná Mosadz Šampanské Zlato",
        category="metal",
        base_color=[0.88, 0.78, 0.52, 1.0],
        roughness=0.25,
        metallic=0.94,
        normal_depth=0.35,
        subsurface_scatter=0.02,
        pattern_scale=0.5,
        thermal_conductivity=115.0,
        acoustic_absorption=0.05,
        biophilic_resonance=0.55,
        description="Teplý kovový akcent s jemným lineárnym brúsením odrážajúci okolité svetlo v harmonických zlatých tónoch."
    ),
    "architectural_microcement": InteriorMaterialPattern(
        material_id="architectural_microcement",
        name="Architektonický Bezšpárový Mikrocement",
        category="mineral",
        base_color=[0.58, 0.60, 0.62, 1.0],
        roughness=0.48,
        metallic=0.05,
        normal_depth=0.55,
        subsurface_scatter=0.08,
        pattern_scale=2.0,
        thermal_conductivity=0.95,
        acoustic_absorption=0.20,
        biophilic_resonance=0.60,
        description="Monolitický betónový povrch bez rušivých škár, poskytujúci čisté vizuálne pozadie pre vyniknutie vegetácie a nábytku."
    )
}


# =============================================================================
# 2. PROCEDURAL PRE-GENERATION DATA SCHEMAS
# =============================================================================

@dataclass
class PreGenTree:
    """Instanced tree entity generated from biophilic patterns for Godot MultiMesh."""
    id: int
    x: float
    y: float
    z: float
    scale: float
    rotation_y_deg: float
    trunk_height: float
    canopy_radius: float
    foliage_density: float
    chlorophyll_tint: List[float]
    species: str
    vital_hp: int = VITAL_MAX_HP


@dataclass
class PreGenBuilding:
    """Procedural building footprint and volume derived from interior room grammars."""
    id: int
    footprint_x: float
    footprint_z: float
    width: float
    depth: float
    height: float
    floors: int
    facade_material: str
    interior_primary_material: str
    column_count: int
    window_density: float
    rooftop_garden_area_m2: float
    vital_hp: int = VITAL_MAX_HP


@dataclass
class PreGenCitySuperblock:
    """Urban district combining residential/commercial blocks and biophilic corridors."""
    id: int
    grid_col: int
    grid_row: int
    center_x: float
    center_z: float
    zone_type: str  # "Biophilic Residential", "Commercial Agora", "Green Lung Park", "Artisan Tech"
    density_ratio: float
    building_ids: List[int]
    green_coverage_percent: float
    vital_hp: int = VITAL_MAX_HP


@dataclass
class PreGenSoilLayer:
    """Substrate profile from deep bedrock to living biological topsoil."""
    horizon_code: str  # "R" (Bedrock), "C" (Saprolite), "B" (Subsoil), "A" (Topsoil/Humus)
    name: str
    thickness_m: float
    porosity_percent: float
    organic_matter_percent: float
    water_retention_capacity: float
    dominant_color: List[float]
    permeability_mm_h: float
    suited_vegetation: List[str]


# =============================================================================
# 3. NODE GRAPH SYSTEM FOR INTERIOR-TO-MACRO GENERATION
# =============================================================================

@dataclass
class NodeSocket:
    name: str
    data_type: str  # "material", "spatial_grid", "vegetation_stream", "mesh_data", "climate_data"
    is_output: bool = False


@dataclass
class ProceduralNode:
    node_id: str
    node_type: str
    title: str
    category: str
    pos_x: float
    pos_y: float
    inputs: List[NodeSocket]
    outputs: List[NodeSocket]
    parameters: Dict[str, Any]
    last_evaluated_output: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "title": self.title,
            "category": self.category,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "inputs": [{"name": s.name, "type": s.data_type} for s in self.inputs],
            "outputs": [{"name": s.name, "type": s.data_type} for s in self.outputs],
            "parameters": self.parameters,
            "last_output_summary": {k: str(v)[:60] for k, v in self.last_evaluated_output.items()}
        }


@dataclass
class NodeConnection:
    connection_id: str
    from_node: str
    from_socket: str
    to_node: str
    to_socket: str


# =============================================================================
# 4. GODOT PROCESS & GAME MECHANICS SPECIFICATION
# =============================================================================

@dataclass
class GodotProcessProfile:
    """Specifications for Godot 4.x process, physics ticks, and rendering optimizations."""
    physics_ticks_per_second: int = 60
    max_physics_steps_per_frame: int = 8
    use_multithreaded_rendering: bool = True
    vulkan_compute_queue_priority: str = "high"
    multimesh_instance_budget: int = 15000
    lod_distance_bands: Dict[str, float] = field(default_factory=lambda: {
        "lod_0_high_detail_m": 25.0,
        "lod_1_medium_detail_m": 75.0,
        "lod_2_billboard_imposter_m": 250.0,
        "cull_distance_m": 600.0
    })
    vital_max_hp_rule: int = VITAL_MAX_HP


# =============================================================================
# 5. CORE ENGINE CLASS: GodotInteriorSurfaceNodeEngine
# =============================================================================

class GodotInteriorSurfaceNodeEngine:
    """
    Central coordinator managing:
      1. Node graph of interior-to-world procedural generators.
      2. Deterministic execution pipeline translating interior patterns into macro worlds.
      3. Generation of Godot 4.x `.tscn` scenes with MultiMeshInstance3D and PBR Shaders.
      4. Godot process lifecycle metrics and simulation telemetry.
    """

    def __init__(self, seed: int = 161803):
        self.seed = seed
        self.nodes: Dict[str, ProceduralNode] = {}
        self.connections: List[NodeConnection] = []
        self.godot_profile = GodotProcessProfile()
        self.active_materials = INTERIOR_MATERIALS_CATALOG
        self._init_default_graph()

    def _init_default_graph(self):
        """Initializes a rich default graph demonstrating interior surface -> world pre-generation."""
        self.nodes = {
            "node_mat_herringbone": ProceduralNode(
                node_id="node_mat_herringbone",
                node_type="InteriorMaterialNode",
                title="Dubové Rybie Parkety",
                category="Interior Materials",
                pos_x=50.0,
                pos_y=80.0,
                inputs=[],
                outputs=[NodeSocket("material_out", "material", True)],
                parameters={
                    "material_id": "oak_herringbone_parquet",
                    "scale": 1.618,
                    "roughness_bias": 0.0,
                    "tint_warmth": 1.05
                }
            ),
            "node_mat_moss": ProceduralNode(
                node_id="node_mat_moss",
                node_type="InteriorMaterialNode",
                title="Biofilná Machová Stena",
                category="Interior Materials",
                pos_x=50.0,
                pos_y=280.0,
                inputs=[],
                outputs=[NodeSocket("material_out", "material", True)],
                parameters={
                    "material_id": "biophilic_vertical_moss",
                    "density": 0.95,
                    "chlorophyll_vibrancy": 1.15
                }
            ),
            "node_spatial_grammar": ProceduralNode(
                node_id="node_spatial_grammar",
                node_type="HarmonicSpaceDivisionNode",
                title="Harmonické Delenie Priestoru (Phi)",
                category="Spatial Grammar",
                pos_x=340.0,
                pos_y=60.0,
                inputs=[NodeSocket("floor_material", "material", False)],
                outputs=[
                    NodeSocket("spatial_grid", "spatial_grid", True),
                    NodeSocket("column_layout", "mesh_data", True)
                ],
                parameters={
                    "room_aspect_ratio": GOLDEN_RATIO,
                    "column_module_m": 4.854,  # 3 * 1.618
                    "circulation_efficiency": 0.85
                }
            ),
            "node_biophilic_veg": ProceduralNode(
                node_id="node_biophilic_veg",
                node_type="BiophilicVegetationPreGenNode",
                title="Predgenerátor Vegetácie & Stromov",
                category="Flora Pre-Gen",
                pos_x=340.0,
                pos_y=300.0,
                inputs=[NodeSocket("biophilic_source", "material", False)],
                outputs=[
                    NodeSocket("tree_multimesh", "mesh_data", True),
                    NodeSocket("foliage_density_map", "climate_data", True)
                ],
                parameters={
                    "tree_count": 240,
                    "canopy_variation": 0.35,
                    "l_system_iterations": 4,
                    "fractal_branch_angle_deg": 27.5
                }
            ),
            "node_building_extruder": ProceduralNode(
                node_id="node_building_extruder",
                node_type="ArchitecturalBuildingNode",
                title="Architektonický Extrudér Budov",
                category="Architecture Pre-Gen",
                pos_x=650.0,
                pos_y=80.0,
                inputs=[
                    NodeSocket("spatial_grid", "spatial_grid", False),
                    NodeSocket("facade_material", "material", False)
                ],
                outputs=[
                    NodeSocket("building_footprints", "mesh_data", True),
                    NodeSocket("rooftop_biophilic_zones", "spatial_grid", True)
                ],
                parameters={
                    "building_count": 18,
                    "min_floors": 3,
                    "max_floors": 12,
                    "window_to_wall_ratio": 0.42,
                    "cantilever_terrace_depth_m": 2.618
                }
            ),
            "node_soil_substrate": ProceduralNode(
                node_id="node_soil_substrate",
                node_type="SoilClimateSubstrateNode",
                title="Pôdne Horizonty & Mikroklima",
                category="Substrate & Climate",
                pos_x=650.0,
                pos_y=340.0,
                inputs=[NodeSocket("foliage_density_map", "climate_data", False)],
                outputs=[
                    NodeSocket("soil_profile", "climate_data", True),
                    NodeSocket("moisture_contours", "climate_data", True)
                ],
                parameters={
                    "permeability_target_mm_h": 45.0,
                    "organic_layer_thickness_cm": 35.0,
                    "subsurface_drainage_porosity": 0.618
                }
            ),
            "node_city_superblock": ProceduralNode(
                node_id="node_city_superblock",
                node_type="UrbanCitySuperblockNode",
                title="Mestský Superblok & Zóny",
                category="Urban Synthesis",
                pos_x=960.0,
                pos_y=120.0,
                inputs=[
                    NodeSocket("buildings", "mesh_data", False),
                    NodeSocket("vegetation", "mesh_data", False),
                    NodeSocket("soil_data", "climate_data", False)
                ],
                outputs=[
                    NodeSocket("city_world_graph", "spatial_grid", True)
                ],
                parameters={
                    "superblock_dimension_m": 240.0,
                    "green_corridor_width_m": 32.0,
                    "pedestrian_artery_count": 6
                }
            ),
            "node_godot_exporter": ProceduralNode(
                node_id="node_godot_exporter",
                node_type="GodotMultiMeshExporterNode",
                title="Godot 4 MultiMesh Exporter & Process",
                category="Godot Engine Pipeline",
                pos_x=1280.0,
                pos_y=200.0,
                inputs=[
                    NodeSocket("world_graph", "spatial_grid", False),
                    NodeSocket("soil_profile", "climate_data", False)
                ],
                outputs=[
                    NodeSocket("godot_tscn", "mesh_data", True),
                    NodeSocket("godot_gdscript", "mesh_data", True)
                ],
                parameters={
                    "scene_name": "InteriorToWorldProceduralStage",
                    "use_octree_culling": True,
                    "bake_lightmaps": True
                }
            )
        }

        self.connections = [
            NodeConnection("c1", "node_mat_herringbone", "material_out", "node_spatial_grammar", "floor_material"),
            NodeConnection("c2", "node_mat_moss", "material_out", "node_biophilic_veg", "biophilic_source"),
            NodeConnection("c3", "node_spatial_grammar", "spatial_grid", "node_building_extruder", "spatial_grid"),
            NodeConnection("c4", "node_mat_herringbone", "material_out", "node_building_extruder", "facade_material"),
            NodeConnection("c5", "node_biophilic_veg", "foliage_density_map", "node_soil_substrate", "foliage_density_map"),
            NodeConnection("c6", "node_building_extruder", "building_footprints", "node_city_superblock", "buildings"),
            NodeConnection("c7", "node_biophilic_veg", "tree_multimesh", "node_city_superblock", "vegetation"),
            NodeConnection("c8", "node_soil_substrate", "soil_profile", "node_city_superblock", "soil_data"),
            NodeConnection("c9", "node_city_superblock", "city_world_graph", "node_godot_exporter", "world_graph"),
            NodeConnection("c10", "node_soil_substrate", "soil_profile", "node_godot_exporter", "soil_profile")
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # CATALOG & GRAPH RETRIEVAL
    # ─────────────────────────────────────────────────────────────────────────

    def get_node_catalog(self) -> Dict[str, Any]:
        """Returns metadata for all available node types in the system."""
        return {
            "vital_max_hp_rule": VITAL_MAX_HP,
            "golden_ratio": GOLDEN_RATIO,
            "categories": [
                "Interior Materials",
                "Spatial Grammar",
                "Flora Pre-Gen",
                "Architecture Pre-Gen",
                "Substrate & Climate",
                "Urban Synthesis",
                "Godot Engine Pipeline"
            ],
            "materials_catalog": [m.to_dict() for m in self.active_materials.values()],
            "node_types": [
                {
                    "type": "InteriorMaterialNode",
                    "label": "Interiérový Materiálový Vzor",
                    "category": "Interior Materials",
                    "description": "Simuluje PBR parametre dreva, kameňa, machu, kovu a terazza."
                },
                {
                    "type": "HarmonicSpaceDivisionNode",
                    "label": "Harmonické Delenie Priestoru (Phi)",
                    "category": "Spatial Grammar",
                    "description": "Aplikuje zlatý rez na modularitu izieb, stĺporadia a zónovanie."
                },
                {
                    "type": "BiophilicVegetationPreGenNode",
                    "label": "Predgenerátor Vegetácie & Stromov",
                    "category": "Flora Pre-Gen",
                    "description": "Škáluje biofilné machové a kvetinové vzory na L-System stromy a Godot MultiMesh."
                },
                {
                    "type": "ArchitecturalBuildingNode",
                    "label": "Architektonický Extrudér Budov",
                    "category": "Architecture Pre-Gen",
                    "description": "Extruduje podlažia, fasády a zelené strechy z pôdorysných gramatík."
                },
                {
                    "type": "SoilClimateSubstrateNode",
                    "label": "Pôdne Horizonty & Mikroklima",
                    "category": "Substrate & Climate",
                    "description": "Generuje pôdne podložie (skala, il, ornica), priepustnosť a vlahový profil."
                },
                {
                    "type": "UrbanCitySuperblockNode",
                    "label": "Mestský Superblok & Zóny",
                    "category": "Urban Synthesis",
                    "description": "Združuje budovy a zelené tepny do harmonických mestských štvrtí."
                },
                {
                    "type": "GodotMultiMeshExporterNode",
                    "label": "Godot 4 MultiMesh Exporter & Process",
                    "category": "Godot Engine Pipeline",
                    "description": "Generuje Godot 4 .tscn scénu, GDScript riadenie procesov a LOD buffery."
                }
            ]
        }

    def get_current_graph(self) -> Dict[str, Any]:
        """Returns the full node graph topology and connections."""
        return {
            "vital_max_hp_rule": VITAL_MAX_HP,
            "seed": self.seed,
            "nodes_count": len(self.nodes),
            "connections_count": len(self.connections),
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "connections": [
                {
                    "id": c.connection_id,
                    "from_node": c.from_node,
                    "from_socket": c.from_socket,
                    "to_node": c.to_node,
                    "to_socket": c.to_socket
                } for c in self.connections
            ]
        }

    # ─────────────────────────────────────────────────────────────────────────
    # PROCEDURAL EVALUATION & PRE-GENERATION PIPELINE
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_graph(self, custom_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the procedural pipeline from interior materials to macro-scale world generation.
        Returns generated trees, buildings, city superblocks, soil profiles, and Godot 4 scene stats.
        """
        if custom_params and "seed" in custom_params:
            self.seed = int(custom_params["seed"])

        rng = random.Random(self.seed)

        # 1. Evaluate Interior Materials
        mat_herringbone = self.active_materials["oak_herringbone_parquet"]
        mat_moss = self.active_materials["biophilic_vertical_moss"]
        mat_travertine = self.active_materials["roman_travertine_stone"]

        # 2. Evaluate Spatial Grammar (Golden Ratio modules)
        base_module_m = 4.854  # 3 * 1.618
        grid_cells_x = 8
        grid_cells_z = 8

        # 3. Pre-generate Trees (MultiMesh batching from biophilic patterns)
        tree_node_params = self.nodes.get("node_biophilic_veg", ProceduralNode("", "", "", "", 0, 0, [], [], {})).parameters
        target_tree_count = int(tree_node_params.get("tree_count", 240))
        trees: List[PreGenTree] = []

        species_list = ["Pinus Sylvestris (Borovica)", "Quercus Robur (Dub Letný)", "Acer Palmatum (Javor)", "Betula Pendula (Breza)"]
        for i in range(target_tree_count):
            # Cluster trees along golden spiral / harmonic radii
            angle = (i * 137.508) * (math.pi / 180.0)  # Golden angle
            dist = math.sqrt(i) * 14.5 + rng.uniform(-3.5, 3.5)
            tx = dist * math.cos(angle)
            tz = dist * math.sin(angle)
            # Avoid placing trees on core central plaza
            if math.hypot(tx, tz) < 20.0:
                dist += 25.0
                tx = dist * math.cos(angle)
                tz = dist * math.sin(angle)

            spec = species_list[i % len(species_list)]
            scale = round(rng.uniform(0.75, 1.45) * GOLDEN_RATIO * 0.7, 3)
            trunk_h = round(scale * 3.8, 2)
            canopy_r = round(scale * 2.4, 2)
            chlorophyll_g = round(0.55 + rng.uniform(-0.1, 0.25), 3)

            trees.append(PreGenTree(
                id=i,
                x=round(tx, 2),
                y=0.0,
                z=round(tz, 2),
                scale=scale,
                rotation_y_deg=round(rng.uniform(0.0, 360.0), 1),
                trunk_height=trunk_h,
                canopy_radius=canopy_r,
                foliage_density=round(rng.uniform(0.75, 0.98), 2),
                chlorophyll_tint=[0.12, chlorophyll_g, 0.18, 1.0],
                species=spec,
                vital_hp=VITAL_MAX_HP
            ))

        # 4. Pre-generate Buildings & Architecture
        bldg_node_params = self.nodes.get("node_building_extruder", ProceduralNode("", "", "", "", 0, 0, [], [], {})).parameters
        target_bldg_count = int(bldg_node_params.get("building_count", 18))
        buildings: List[PreGenBuilding] = []

        for b_id in range(target_bldg_count):
            angle = (b_id / float(target_bldg_count)) * 2.0 * math.pi
            ring_radius = 65.0 + (b_id % 3) * 35.0
            bx = ring_radius * math.cos(angle) + rng.uniform(-5.0, 5.0)
            bz = ring_radius * math.sin(angle) + rng.uniform(-5.0, 5.0)

            w = round(base_module_m * rng.choice([2, 3, 4]), 2)
            d = round(base_module_m * rng.choice([2, 3]) * INV_GOLDEN_RATIO, 2)
            floors = rng.randint(3, 9)
            fl_height = 3.2
            h = round(floors * fl_height, 2)

            facade = rng.choice(["roman_travertine_stone", "architectural_microcement", "acoustic_slat_oak_felt"])
            interior_mat = rng.choice(["oak_herringbone_parquet", "venetian_terrazzo"])

            buildings.append(PreGenBuilding(
                id=b_id,
                footprint_x=round(bx, 2),
                footprint_z=round(bz, 2),
                width=w,
                depth=d,
                height=h,
                floors=floors,
                facade_material=facade,
                interior_primary_material=interior_mat,
                column_count=int(w / base_module_m) * 2 + 2,
                window_density=round(rng.uniform(0.35, 0.55), 2),
                rooftop_garden_area_m2=round(w * d * 0.618, 1),
                vital_hp=VITAL_MAX_HP
            ))

        # 5. Pre-generate City Superblocks (Zoning & Corridors)
        superblocks: List[PreGenCitySuperblock] = []
        zones = [
            ("Biophilic Residential", 0.65, 0.72),
            ("Commercial Agora", 0.88, 0.35),
            ("Green Lung Park", 0.15, 0.95),
            ("Artisan Tech Concourse", 0.75, 0.55)
        ]

        bldg_idx = 0
        for r in range(2):
            for c in range(2):
                sb_id = r * 2 + c
                z_name, d_ratio, g_cov = zones[sb_id % len(zones)]
                cx = (c - 0.5) * 160.0
                cz = (r - 0.5) * 160.0
                assigned_bldgs = [b.id for b in buildings[bldg_idx:bldg_idx + 4]]
                bldg_idx += 4

                superblocks.append(PreGenCitySuperblock(
                    id=sb_id,
                    grid_col=c,
                    grid_row=r,
                    center_x=cx,
                    center_z=cz,
                    zone_type=z_name,
                    density_ratio=d_ratio,
                    building_ids=assigned_bldgs,
                    green_coverage_percent=round(g_cov * 100.0, 1),
                    vital_hp=VITAL_MAX_HP
                ))

        # 6. Pre-generate Soil & Substrate Horizons
        soil_layers: List[PreGenSoilLayer] = [
            PreGenSoilLayer(
                horizon_code="A",
                name="Humózna Ornica & Biofilný Substrát (Topsoil)",
                thickness_m=0.35,
                porosity_percent=48.5,
                organic_matter_percent=12.5,
                water_retention_capacity=0.78,
                dominant_color=[0.24, 0.18, 0.12, 1.0],
                permeability_mm_h=45.0,
                suited_vegetation=["Machové koberce", "Paprade", "Kríkové podrasty", "Mladé dreviny"]
            ),
            PreGenSoilLayer(
                horizon_code="B",
                name="Minerálna Pôdna Vrstva (Subsoil / Loam)",
                thickness_m=0.85,
                porosity_percent=38.0,
                organic_matter_percent=3.2,
                water_retention_capacity=0.62,
                dominant_color=[0.48, 0.32, 0.18, 1.0],
                permeability_mm_h=22.0,
                suited_vegetation=["Hlboké koreňové sústavy stromov", "Dospelé duby", "Borovice"]
            ),
            PreGenSoilLayer(
                horizon_code="C",
                name="Zvetralé Kamenisté Podložie (Saprolite)",
                thickness_m=1.60,
                porosity_percent=24.5,
                organic_matter_percent=0.4,
                water_retention_capacity=0.35,
                dominant_color=[0.58, 0.54, 0.48, 1.0],
                permeability_mm_h=12.0,
                suited_vegetation=["Skalné lišajníky", "Kotviace primárne korene"]
            ),
            PreGenSoilLayer(
                horizon_code="R",
                name="Kryštalická Materská Skala (Solid Bedrock)",
                thickness_m=10.0,
                porosity_percent=4.2,
                organic_matter_percent=0.0,
                water_retention_capacity=0.05,
                dominant_color=[0.22, 0.24, 0.28, 1.0],
                permeability_mm_h=0.5,
                suited_vegetation=["Základové pylóny budov", "Geotermálne studne"]
            )
        ]

        # 7. Package Godot 4.x MultiMesh and Process Telemetry
        multimesh_instance_count = len(trees) + sum(b.column_count for b in buildings)
        total_building_floors = sum(b.floors for b in buildings)
        total_green_space_m2 = sum(b.rooftop_garden_area_m2 for b in buildings) + len(trees) * 18.5

        return {
            "status": "SUCCESS_WORLD_PREGENERATED",
            "seed": self.seed,
            "vital_max_hp_rule": VITAL_MAX_HP,
            "metrics": {
                "total_trees_instanced": len(trees),
                "total_buildings": len(buildings),
                "total_building_floors": total_building_floors,
                "total_superblocks": len(superblocks),
                "soil_horizons_count": len(soil_layers),
                "total_green_space_m2": round(total_green_space_m2, 1),
                "multimesh_instance_count": multimesh_instance_count,
                "physics_tick_budget_ms": round(1000.0 / self.godot_profile.physics_ticks_per_second, 2),
                "estimated_gpu_draw_calls": 14  # Godot MultiMesh batches everything into single-digit drawcalls
            },
            "trees_sample": [
                {
                    "id": t.id, "x": t.x, "y": t.y, "z": t.z, "scale": t.scale,
                    "species": t.species, "vital_hp": t.vital_hp
                } for t in trees[:10]
            ],
            "buildings_sample": [
                {
                    "id": b.id, "x": b.footprint_x, "z": b.footprint_z,
                    "width": b.width, "depth": b.depth, "height": b.height,
                    "floors": b.floors, "facade": b.facade_material, "vital_hp": b.vital_hp
                } for b in buildings[:6]
            ],
            "superblocks": [
                {
                    "id": s.id, "col": s.grid_col, "row": s.grid_row,
                    "zone": s.zone_type, "green_percent": s.green_coverage_percent,
                    "building_count": len(s.building_ids), "vital_hp": s.vital_hp
                } for s in superblocks
            ],
            "soil_horizons": [
                {
                    "horizon": s.horizon_code, "name": s.name,
                    "thickness_m": s.thickness_m, "porosity_pct": s.porosity_percent,
                    "permeability_mm_h": s.permeability_mm_h, "color": s.dominant_color
                } for s in soil_layers
            ],
            "godot_process_profile": {
                "physics_ticks_hz": self.godot_profile.physics_ticks_per_second,
                "multithreaded_rendering": self.godot_profile.use_multithreaded_rendering,
                "vulkan_queue": self.godot_profile.vulkan_compute_queue_priority,
                "lod_bands": self.godot_profile.lod_distance_bands
            }
        }

    # ─────────────────────────────────────────────────────────────────────────
    # GODOT 4.X SCENE (.TSCN) & GDSCRIPT EXPORTER
    # ─────────────────────────────────────────────────────────────────────────

    def generate_godot_scene_tscn(self) -> str:
        """
        Generates a valid Godot 4.x scene file with:
          - MultiMeshInstance3D for tree vegetation.
          - StandardMaterial3D shaders with interior PBR values.
          - DirectionalLight3D sun and WorldEnvironment with volumetric fog.
        """
        lines = [
            '[gd_scene format=3 uid="uid://krystal_interior_surface_world_pregen"]',
            '',
            '# =====================================================================',
            '# KRYSTAL-STACK: GODOT 4.X PROCEDURAL INTERIOR-TO-MACRO STAGE',
            '# Generated by GodotInteriorSurfaceNodeEngine',
            '# =====================================================================',
            '',
            '[sub_resource type="StandardMaterial3D" id="Mat_OakHerringbone"]',
            'albedo_color = Color(0.68, 0.48, 0.32, 1.0)',
            'roughness = 0.38',
            'metallic = 0.02',
            'normal_enabled = true',
            'normal_scale = 0.85',
            'uv1_scale = Vector3(1.618, 1.618, 1.618)',
            '',
            '[sub_resource type="StandardMaterial3D" id="Mat_VerticalMoss"]',
            'albedo_color = Color(0.18, 0.65, 0.22, 1.0)',
            'roughness = 0.92',
            'metallic = 0.0',
            'subsurface_scattering_enabled = true',
            'subsurface_scattering_transmittance_depth = 0.68',
            '',
            '[sub_resource type="StandardMaterial3D" id="Mat_TravertineStone"]',
            'albedo_color = Color(0.86, 0.82, 0.73, 1.0)',
            'roughness = 0.55',
            'metallic = 0.04',
            '',
            '[sub_resource type="CylinderMesh" id="TreeTrunkMesh"]',
            'top_radius = 0.25',
            'bottom_radius = 0.4',
            'height = 3.8',
            '',
            '[sub_resource type="SphereMesh" id="TreeCanopyMesh"]',
            'radius = 2.4',
            'height = 4.0',
            'radial_segments = 16',
            'rings = 8',
            '',
            '[sub_resource type="MultiMesh" id="MultiMesh_Trees"]',
            'transform_format = 1',
            'use_colors = true',
            'instance_count = 240',
            'mesh = SubResource("TreeCanopyMesh")',
            '',
            '[node name="InteriorWorldStage" type="Node3D"]',
            '',
            '[node name="WorldEnvironment" type="WorldEnvironment" parent="."]',
            '',
            '[node name="SunLight" type="DirectionalLight3D" parent="."]',
            'transform = Transform3D(0.866, -0.25, 0.433, 0.0, 0.866, 0.5, -0.5, -0.433, 0.75, 50.0, 100.0, 50.0)',
            'light_color = Color(1.0, 0.96, 0.90, 1.0)',
            'light_energy = 1.35',
            'shadow_enabled = true',
            '',
            '[node name="CityDistrict" type="Node3D" parent="."]',
            '',
            '[node name="VegetationMultiMesh" type="MultiMeshInstance3D" parent="."]',
            'multimesh = SubResource("MultiMesh_Trees")',
            'material_override = SubResource("Mat_VerticalMoss")',
            '',
            '[node name="SubstrateFloor" type="CSGBox3D" parent="."]',
            'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, -1.0, 0)',
            'size = Vector3(400.0, 2.0, 400.0)',
            'material = SubResource("Mat_OakHerringbone")'
        ]
        return "\n".join(lines)

    def generate_godot_gdscript(self) -> str:
        """
        Generates the GDScript controller handling real-time runtime streaming,
        LOD distance evaluations, and MultiMesh buffer updates.
        """
        lines = [
            'extends Node3D',
            '',
            '# =====================================================================',
            '# KRYSTAL-STACK // GODOT 4.X PROCEDURAL WORLD RUNTIME CONTROLLER',
            '# Governed by VITAL_MAX_HP = 6 Invariant and MultiMesh Batching',
            '# =====================================================================',
            '',
            'const VITAL_MAX_HP: int = 6',
            'const GOLDEN_RATIO: float = 1.61803398875',
            '',
            '@onready var vegetation_multimesh: MultiMeshInstance3D = $VegetationMultiMesh',
            '@onready var sun_light: DirectionalLight3D = $SunLight',
            '',
            'var time_accumulator: float = 0.0',
            'var player_camera: Camera3D = null',
            '',
            'func _ready() -> void:',
            '    print("[KRYSTAL-GODOT] Interior-to-Macro Procedural Stage Initialized.")',
            '    _populate_tree_transforms()',
            '',
            'func _process(delta: float) -> void:',
            '    time_accumulator += delta',
            '    # Subtle sun arc mimicking daylight harvesting',
            '    sun_light.rotation_degrees.y = fmod(time_accumulator * 1.5, 360.0)',
            '',
            'func _physics_process(delta: float) -> void:',
            '    # Strict 60 Hz physics tick budget for deterministic entity updates',
            '    pass',
            '',
            'func _populate_tree_transforms() -> void:',
            '    if not vegetation_multimesh or not vegetation_multimesh.multimesh:',
            '        return',
            '    var mm: MultiMesh = vegetation_multimesh.multimesh',
            '    var count: int = mm.instance_count',
            '    for i in range(count):',
            '        var angle: float = (i * 137.508) * (PI / 180.0)',
            '        var dist: float = sqrt(float(i)) * 14.5 + 25.0',
            '        var tx: float = dist * cos(angle)',
            '        var tz: float = dist * sin(angle)',
            '        var t: Transform3D = Transform3D().translated(Vector3(tx, 0.0, tz))',
            '        t = t.scaled(Vector3(1.2, 1.2, 1.2))',
            '        mm.set_instance_transform(i, t)',
            '        mm.set_instance_color(i, Color(0.18, 0.72, 0.24, 1.0))',
            ''
        ]
        return "\n".join(lines)


GLOBAL_GODOT_INTERIOR_NODE_ENGINE = GodotInteriorSurfaceNodeEngine()
