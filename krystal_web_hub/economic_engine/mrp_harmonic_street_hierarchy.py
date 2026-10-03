"""
MRP Harmonic Street Hierarchy & Pink Panther Aesthetic Procedural Engine
========================================================================
Transforms traditional manufacturing/material requirements planning (MRP) into a 5-tier
procedural engine hierarchy governed by harmonic color axioms, golden ratio proportions (phi = 1.618),
Pink Panther 1960s suave aesthetic tints, 12 canonical world regions generated via Google Street Map/OSM
vector blueprints, and top-down 2D vehicle cruisers subject to the strict 6 Max HP vital invariant.

Hierarchy Breakdown:
- Level 0: Axiomatic Root & Harmonic Color Tints (Golden Ratio phi, Fibonacci, Pink Panther axioms)
- Level 1: Macro Metropolises (12 canonical world regions: Gotham, Arkham, Sin City, Vegas, Alabama,
           Ohio, Florida, Australia, Canary Islands, Bolivia, Ecuador, Peru)
- Level 2: Meso Street Networks & Google Street Map/OSM Vector Blueprints (nodes, conduits, lanes, intersections)
- Level 3: Micro 2D Vehicles & Cruisers (Top-down physics, steering, drift, friction, and engine power)
- Level 4: Execution & Strict 6 Max HP Vital Invariant (vital health, armor, garrison, and convoy integrity)
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 1.0 / GOLDEN_RATIO  # ~0.61803398875
VITAL_MAX_HP: int = 6  # Strict invariant across all vehicles, garrisons, and convoys

# Canonical Pink Panther Palette & Noir Harmonic Accents
PINK_PANTHER_PALETTE: Dict[str, Dict[str, Any]] = {
    "panther_pink": {
        "hex": "#ec4899",
        "rgb": (236, 72, 153),
        "name": "Panther Classic Pink",
        "role": "Hero accent & bodywork tint"
    },
    "hot_magenta": {
        "hex": "#db2777",
        "rgb": (219, 39, 119),
        "name": "Inspector Magenta",
        "role": "High velocity speedlines & neon conduits"
    },
    "blush_pastel": {
        "hex": "#f472b6",
        "rgb": (244, 114, 182),
        "name": "Suave Blush Pastel",
        "role": "Sidewalk highlights & street ambient glow"
    },
    "champagne_cream": {
        "hex": "#fef3c7",
        "rgb": (254, 243, 199),
        "name": "Riviera Champagne Cream",
        "role": "Headlight cones & building façades"
    },
    "noir_charcoal": {
        "hex": "#18181b",
        "rgb": (24, 24, 27),
        "name": "Diamond Heist Noir Charcoal",
        "role": "Asphalt streets & deep drop shadows"
    },
    "gotham_slate": {
        "hex": "#334155",
        "rgb": (51, 65, 85),
        "name": "Gotham Rain Slate",
        "role": "Industrial roadbed & steel bridges"
    },
    "sin_city_crimson": {
        "hex": "#e11d48",
        "rgb": (225, 29, 72),
        "name": "Sin City Blood Crimson",
        "role": "Emergency brake glow & alert indicators"
    },
    "vegas_gold": {
        "hex": "#f59e0b",
        "rgb": (245, 158, 11),
        "name": "Vegas Strip Amber Gold",
        "role": "Casino boulevard street lamps & wheel hubs"
    }
}


@dataclass
class HarmonicColorTint:
    """Represents a computed harmonic color tint based on phi axioms."""
    step: int
    r: int
    g: int
    b: int
    hex_code: str
    tint_ratio: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "r": self.r,
            "g": self.g,
            "b": self.b,
            "hex_code": self.hex_code,
            "tint_ratio": round(self.tint_ratio, 4),
            "description": self.description
        }


@dataclass
class MRPHierarchyLevel:
    """Represents one of the 5 tiers in the recast MRP Engine Hierarchy."""
    level_id: int
    name: str
    mrp_traditional_analog: str
    engine_procedural_role: str
    axiomatic_invariant: str
    dependencies: List[int]
    lead_time_ticks: int
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level_id": self.level_id,
            "name": self.name,
            "mrp_traditional_analog": self.mrp_traditional_analog,
            "engine_procedural_role": self.engine_procedural_role,
            "axiomatic_invariant": self.axiomatic_invariant,
            "dependencies": self.dependencies,
            "lead_time_ticks": self.lead_time_ticks,
            "description": self.description
        }


@dataclass
class WorldRegionMapSpec:
    """Canonical world metropolis / territory for procedural street map generation."""
    region_id: str
    display_name: str
    country_or_lore: str
    latitude: float
    longitude: float
    street_topology: str  # e.g., "grid", "organic_switchback", "boulevard_strip", "canyon_ribbon"
    primary_tint: str     # Hex key from PINK_PANTHER_PALETTE or custom
    secondary_tint: str
    asphalt_shade: str
    ambient_weather: str
    osm_bounding_box: Tuple[float, float, float, float]  # min_lat, min_lon, max_lat, max_lon
    default_speed_limit_kmh: int
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "display_name": self.display_name,
            "country_or_lore": self.country_or_lore,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "street_topology": self.street_topology,
            "primary_tint": self.primary_tint,
            "secondary_tint": self.secondary_tint,
            "asphalt_shade": self.asphalt_shade,
            "ambient_weather": self.ambient_weather,
            "osm_bounding_box": list(self.osm_bounding_box),
            "default_speed_limit_kmh": self.default_speed_limit_kmh,
            "description": self.description
        }


@dataclass
class VehicleCruiser2D:
    """Top-down 2D cruiser with drift physics subject to strict 6 Max HP vital invariant."""
    vehicle_id: str
    name: str
    cruiser_type: str  # "panther_coupe", "gotham_interceptor", "sin_city_cruiser", "vegas_convertible", "outback_runner", "andean_rally"
    region_id: str
    x: float
    y: float
    heading_deg: float
    velocity_mps: float
    max_velocity_mps: float
    acceleration_mps2: float
    braking_mps2: float
    turn_rate_degps: float
    drift_factor: float      # 0.0 (high grip) to 1.0 (extreme drift)
    hp: int                  # STRICT INVARIANT: 0 <= hp <= 6
    max_hp: int = VITAL_MAX_HP
    armor: int = 2           # 0 to 6
    body_color_hex: str = "#ec4899"
    stripe_color_hex: str = "#fef3c7"
    is_drifting: bool = False
    status: str = "operational"  # "operational", "damaged", "critical", "destroyed"

    def clamp_hp(self) -> None:
        if self.hp > VITAL_MAX_HP:
            self.hp = VITAL_MAX_HP
        elif self.hp < 0:
            self.hp = 0

        if self.hp == 0:
            self.status = "destroyed"
            self.velocity_mps = 0.0
        elif self.hp <= 2:
            self.status = "critical"
        elif self.hp <= 4:
            self.status = "damaged"
        else:
            self.status = "operational"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "name": self.name,
            "cruiser_type": self.cruiser_type,
            "region_id": self.region_id,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "heading_deg": round(self.heading_deg, 2),
            "velocity_mps": round(self.velocity_mps, 2),
            "velocity_kmh": round(self.velocity_mps * 3.6, 1),
            "max_velocity_mps": self.max_velocity_mps,
            "acceleration_mps2": self.acceleration_mps2,
            "braking_mps2": self.braking_mps2,
            "turn_rate_degps": self.turn_rate_degps,
            "drift_factor": round(self.drift_factor, 2),
            "hp": self.hp,
            "max_hp": self.max_hp,
            "armor": self.armor,
            "body_color_hex": self.body_color_hex,
            "stripe_color_hex": self.stripe_color_hex,
            "is_drifting": self.is_drifting,
            "status": self.status
        }


class MRPHarmonicStreetEngine:
    """
    Core engine implementation transforming MRP BOM hierarchies into procedural
    city blueprints, Pink Panther harmonic tints, and 2D vehicle simulations.
    """

    def __init__(self):
        self._hierarchy_levels: List[MRPHierarchyLevel] = self._init_mrp_levels()
        self._world_regions: Dict[str, WorldRegionMapSpec] = self._init_world_regions()
        self._active_vehicles: Dict[str, VehicleCruiser2D] = {}
        self._init_default_cruisers()

    def _init_mrp_levels(self) -> List[MRPHierarchyLevel]:
        return [
            MRPHierarchyLevel(
                level_id=0,
                name="Axiomatic Root & Harmonic Color Tints",
                mrp_traditional_analog="Master Production Schedule (MPS) & Root Product Spec",
                engine_procedural_role="Generates Golden Ratio (phi=1.618) chromatic axioms, Pink Panther palettes, and tint tensors",
                axiomatic_invariant="Strict phi chromatic ratio; max harmonic step = 4; RGB clamp [0, 255]",
                dependencies=[],
                lead_time_ticks=1,
                description="The fundamental mathematical root determining harmonic palettes and visual tones for the entire universe."
            ),
            MRPHierarchyLevel(
                level_id=1,
                name="Macro Metropolises & Global Boundaries",
                mrp_traditional_analog="Major Subassemblies & Regional Warehouses",
                engine_procedural_role="Instantiates the 12 canonical world regions with geographic bounds, topography, and ambient weather",
                axiomatic_invariant="Exactly 12 canonical regions; immutable OSM GPS coordinate bounding boxes",
                dependencies=[0],
                lead_time_ticks=2,
                description="Macro territories ranging from Gotham and Arkham to Florida, Canary Islands, Bolivia, and Peru."
            ),
            MRPHierarchyLevel(
                level_id=2,
                name="Meso Street Networks & Google Street Map Blueprints",
                mrp_traditional_analog="Sub-components & Bill of Materials (BOM) Line Items",
                engine_procedural_role="Synthesizes vector street graphs, conduits, intersections, lane widths, and building parcels",
                axiomatic_invariant="Connected street graph; Euclidean road segment continuity; lane count in [1, 8]",
                dependencies=[0, 1],
                lead_time_ticks=3,
                description="Procedural street conduits following Google Street Map / OSM conventions with Pink Panther noir aesthetics."
            ),
            MRPHierarchyLevel(
                level_id=3,
                name="Micro 2D Vehicles & Cruisers",
                mrp_traditional_analog="Work Centers, Finished Goods & Mobile Units",
                engine_procedural_role="Spawns and steers 2D top-down vehicles with drift kinematics, acceleration curves, and steering",
                axiomatic_invariant="Top-down 2D physics; velocity in [0, max_velocity]; drift angle bounded by drift_factor",
                dependencies=[0, 1, 2],
                lead_time_ticks=4,
                description="Suave 2D cruisers drifting through neon avenues, slick alleys, and mountain passes."
            ),
            MRPHierarchyLevel(
                level_id=4,
                name="Execution & Strict 6 Max HP Vital Invariant",
                mrp_traditional_analog="Quality Assurance, Scrap Tolerances & Final Dispatch",
                engine_procedural_role="Enforces unit life cycles, collision damage, convoy cohesion, and garrison integrity",
                axiomatic_invariant="STRICT INVARIANT: max_hp == 6; 0 <= hp <= 6 for all units at all times",
                dependencies=[0, 1, 2, 3],
                lead_time_ticks=1,
                description="Governs vital unit health, armor soaking, repair intervals, and scrap recovery under the 6 Max HP rule."
            )
        ]

    def _init_world_regions(self) -> Dict[str, WorldRegionMapSpec]:
        specs = [
            WorldRegionMapSpec(
                region_id="gotham_city",
                display_name="Gotham City",
                country_or_lore="DC Universe Noir",
                latitude=40.7128,
                longitude=-74.0060,
                street_topology="gothic_art_deco_grid",
                primary_tint="#334155",
                secondary_tint="#ec4899",
                asphalt_shade="#0f172a",
                ambient_weather="heavy_acid_rain_fog",
                osm_bounding_box=(40.6800, -74.0400, 40.7600, -73.9600),
                default_speed_limit_kmh=60,
                description="Gothic art-deco metropolis with elevated rails, gothic bridges, rain-slicked asphalt, and neon Pink Panther accents."
            ),
            WorldRegionMapSpec(
                region_id="arkham_city",
                display_name="Arkham City",
                country_or_lore="DC Supermax Enclave",
                latitude=40.7300,
                longitude=-74.0150,
                street_topology="asylum_barricade_maze",
                primary_tint="#1e293b",
                secondary_tint="#db2777",
                asphalt_shade="#09090b",
                ambient_weather="frozen_freezing_mist",
                osm_bounding_box=(40.7100, -74.0300, 40.7500, -73.9900),
                default_speed_limit_kmh=45,
                description="Walled maximum-security enclave featuring broken cobblestones, barbed wire alleys, and surveillance watchtowers."
            ),
            WorldRegionMapSpec(
                region_id="sin_city",
                display_name="Sin City (Basin City)",
                country_or_lore="Miller Graphic Novel Noir",
                latitude=36.1716,
                longitude=-115.1391,
                street_topology="monochrome_noir_alleys",
                primary_tint="#18181b",
                secondary_tint="#e11d48",
                asphalt_shade="#000000",
                ambient_weather="monochrome_stark_rain",
                osm_bounding_box=(36.1400, -115.1700, 36.2000, -115.1100),
                default_speed_limit_kmh=55,
                description="High-contrast black-and-white graphic novel city where only blood crimson and Pink Panther magentas reflect on the chrome."
            ),
            WorldRegionMapSpec(
                region_id="las_vegas",
                display_name="Las Vegas Strip",
                country_or_lore="Nevada, USA",
                latitude=36.1147,
                longitude=-115.1728,
                street_topology="grand_boulevard_strip",
                primary_tint="#f59e0b",
                secondary_tint="#ec4899",
                asphalt_shade="#1c1917",
                ambient_weather="desert_neon_dusk",
                osm_bounding_box=(36.0800, -115.1900, 36.1500, -115.1400),
                default_speed_limit_kmh=70,
                description="Glittering multi-lane boulevard with casino fountains, pyramid searchlights, and gold-leaf champagne curbs."
            ),
            WorldRegionMapSpec(
                region_id="alabama",
                display_name="Alabama Crossroads & Talladega",
                country_or_lore="Alabama, USA",
                latitude=33.5207,
                longitude=-86.8025,
                street_topology="rural_highway_corridor",
                primary_tint="#b45309",
                secondary_tint="#f472b6",
                asphalt_shade="#292524",
                ambient_weather="humid_pine_sunset",
                osm_bounding_box=(33.4800, -86.8500, 33.5600, -86.7500),
                default_speed_limit_kmh=90,
                description="Long straight interstate corridors bordered by red clay banks, southern pine forests, and high-speed oval circuits."
            ),
            WorldRegionMapSpec(
                region_id="ohio",
                display_name="Ohio Rust Belt & River Crossing",
                country_or_lore="Ohio, USA",
                latitude=41.4993,
                longitude=-81.6944,
                street_topology="industrial_river_grid",
                primary_tint="#475569",
                secondary_tint="#f472b6",
                asphalt_shade="#1e293b",
                ambient_weather="lake_effect_overcast",
                osm_bounding_box=(41.4600, -81.7400, 41.5300, -81.6500),
                default_speed_limit_kmh=65,
                description="Heavy industrial steel bridges, railyards, and riverbank cloverleafs tinted in steel blue and blush pastel."
            ),
            WorldRegionMapSpec(
                region_id="florida",
                display_name="Florida Ocean Drive & Keys Causeway",
                country_or_lore="Florida, USA",
                latitude=25.7617,
                longitude=-80.1918,
                street_topology="coastal_causeway_boulevard",
                primary_tint="#06b6d4",
                secondary_tint="#ec4899",
                asphalt_shade="#334155",
                ambient_weather="tropical_neon_breeze",
                osm_bounding_box=(25.7200, -80.2300, 25.8000, -80.1500),
                default_speed_limit_kmh=80,
                description="Vibrant palm-lined coastal causeways, turquoise canal bridges, and art deco hotels bathed in Pink Panther neon."
            ),
            WorldRegionMapSpec(
                region_id="australia",
                display_name="Australia Outback & Red Centre Highway",
                country_or_lore="Northern Territory, Australia",
                latitude=-23.6980,
                longitude=133.8807,
                street_topology="desert_road_train_turnpike",
                primary_tint="#ea580c",
                secondary_tint="#f472b6",
                asphalt_shade="#44403c",
                ambient_weather="blazing_red_sand_mirage",
                osm_bounding_box=(-23.7500, 133.8200, -23.6500, 133.9400),
                default_speed_limit_kmh=110,
                description="Infinite ochre bitumen, road-train corridors, spinifex flats, and high-traction Mad Max style cruiser routes."
            ),
            WorldRegionMapSpec(
                region_id="canary_islands",
                display_name="Canary Islands (Islas Canarias)",
                country_or_lore="Canary Islands, Spain",
                latitude=28.2916,
                longitude=-16.6291,
                street_topology="volcanic_cliffside_serpentine",
                primary_tint="#0284c7",
                secondary_tint="#ec4899",
                asphalt_shade="#18181b",
                ambient_weather="atlantic_trade_winds",
                osm_bounding_box=(28.2400, -16.6800, 28.3400, -16.5800),
                default_speed_limit_kmh=50,
                description="Dramatic volcanic switchbacks carving through black basalt cliffs above the Atlantic Ocean with champagne harbor curves."
            ),
            WorldRegionMapSpec(
                region_id="bolivia",
                display_name="Bolivia Yungas & Salar Salt Flats",
                country_or_lore="Bolivia",
                latitude=-16.2902,
                longitude=-67.7600,
                street_topology="death_road_altitude_ribbon",
                primary_tint="#059669",
                secondary_tint="#db2777",
                asphalt_shade="#27272a",
                ambient_weather="altitude_cloud_forest",
                osm_bounding_box=(-16.3400, -67.8200, -16.2400, -67.7000),
                default_speed_limit_kmh=40,
                description="Extreme altitude serpentine mountain ribbons flanked by sheer waterfalls, misty precipices, and vast white salt flat vectors."
            ),
            WorldRegionMapSpec(
                region_id="ecuador",
                display_name="Ecuador Equatorial Andes & Quito Arterials",
                country_or_lore="Ecuador",
                latitude=-0.1807,
                longitude=-78.4678,
                street_topology="equatorial_volcano_contour",
                primary_tint="#10b981",
                secondary_tint="#ec4899",
                asphalt_shade="#1e293b",
                ambient_weather="equatorial_mountain_sunshine",
                osm_bounding_box=(-0.2200, -78.5100, -0.1400, -78.4200),
                default_speed_limit_kmh=50,
                description="Equatorial mountain avenues tracing the slopes of Pichincha volcano, crossing historical plazas and lush Andean canyons."
            ),
            WorldRegionMapSpec(
                region_id="peru",
                display_name="Peru Costa Verde & Sacred Valley Pass",
                country_or_lore="Peru",
                latitude=-12.0464,
                longitude=-77.0428,
                street_topology="pacific_cliff_switchback",
                primary_tint="#d97706",
                secondary_tint="#db2777",
                asphalt_shade="#18181b",
                ambient_weather="pacific_garua_mist",
                osm_bounding_box=(-12.0800, -77.0800, -12.0100, -77.0000),
                default_speed_limit_kmh=60,
                description="Coastal cliff highways overlooking Pacific surf and ascending into ancient stone terraces and Sacred Valley hairpin turns."
            )
        ]
        return {spec.region_id: spec for spec in specs}

    def _init_default_cruisers(self) -> None:
        defaults = [
            VehicleCruiser2D(
                vehicle_id="panther_coupe_01",
                name="Pink Panther Grand Tourer 1964",
                cruiser_type="panther_coupe",
                region_id="gotham_city",
                x=250.0,
                y=300.0,
                heading_deg=0.0,
                velocity_mps=15.0,
                max_velocity_mps=42.0,
                acceleration_mps2=8.5,
                braking_mps2=14.0,
                turn_rate_degps=120.0,
                drift_factor=0.65,
                hp=VITAL_MAX_HP,
                armor=3,
                body_color_hex="#ec4899",
                stripe_color_hex="#fef3c7"
            ),
            VehicleCruiser2D(
                vehicle_id="gotham_interceptor_01",
                name="Gotham Dark Knight Interceptor",
                cruiser_type="gotham_interceptor",
                region_id="gotham_city",
                x=180.0,
                y=420.0,
                heading_deg=45.0,
                velocity_mps=12.0,
                max_velocity_mps=48.0,
                acceleration_mps2=11.0,
                braking_mps2=18.0,
                turn_rate_degps=95.0,
                drift_factor=0.30,
                hp=VITAL_MAX_HP,
                armor=5,
                body_color_hex="#18181b",
                stripe_color_hex="#ec4899"
            ),
            VehicleCruiser2D(
                vehicle_id="sin_city_cruiser_01",
                name="Sin City Noir Cadillac 1959",
                cruiser_type="sin_city_cruiser",
                region_id="sin_city",
                x=320.0,
                y=180.0,
                heading_deg=270.0,
                velocity_mps=10.0,
                max_velocity_mps=35.0,
                acceleration_mps2=6.0,
                braking_mps2=10.0,
                turn_rate_degps=80.0,
                drift_factor=0.55,
                hp=VITAL_MAX_HP,
                armor=4,
                body_color_hex="#09090b",
                stripe_color_hex="#e11d48"
            ),
            VehicleCruiser2D(
                vehicle_id="vegas_convertible_01",
                name="Vegas Golden Mirage Convertible",
                cruiser_type="vegas_convertible",
                region_id="las_vegas",
                x=400.0,
                y=350.0,
                heading_deg=180.0,
                velocity_mps=18.0,
                max_velocity_mps=44.0,
                acceleration_mps2=9.0,
                braking_mps2=13.0,
                turn_rate_degps=110.0,
                drift_factor=0.70,
                hp=VITAL_MAX_HP,
                armor=2,
                body_color_hex="#f59e0b",
                stripe_color_hex="#ec4899"
            ),
            VehicleCruiser2D(
                vehicle_id="outback_runner_01",
                name="Australian Outback Road-Train V8",
                cruiser_type="outback_runner",
                region_id="australia",
                x=150.0,
                y=150.0,
                heading_deg=90.0,
                velocity_mps=22.0,
                max_velocity_mps=50.0,
                acceleration_mps2=7.5,
                braking_mps2=12.0,
                turn_rate_degps=75.0,
                drift_factor=0.45,
                hp=VITAL_MAX_HP,
                armor=5,
                body_color_hex="#ea580c",
                stripe_color_hex="#f472b6"
            ),
            VehicleCruiser2D(
                vehicle_id="andean_rally_01",
                name="Bolivian Yungas Alpine Turbo",
                cruiser_type="andean_rally",
                region_id="bolivia",
                x=220.0,
                y=280.0,
                heading_deg=315.0,
                velocity_mps=14.0,
                max_velocity_mps=38.0,
                acceleration_mps2=10.0,
                braking_mps2=16.0,
                turn_rate_degps=135.0,
                drift_factor=0.80,
                hp=VITAL_MAX_HP,
                armor=3,
                body_color_hex="#059669",
                stripe_color_hex="#db2777"
            )
        ]
        for c in defaults:
            self._active_vehicles[c.vehicle_id] = c

    # --------------------------------------------------------------------------
    # LEVEL 0: Axiomatic Root & Harmonic Color Tints
    # --------------------------------------------------------------------------

    def compute_harmonic_color_tint(
        self,
        base_color: Tuple[int, int, int] = (236, 72, 153),
        step: int = 1,
        weight: float = 1.0,
        style: str = "pink_panther_chic"
    ) -> HarmonicColorTint:
        """
        Computes a harmonic tint based on golden ratio axioms:
        R_tint = clamp(R * (1.0 + step * phi^-1 * 0.35 * weight))
        G_tint = clamp(G * (0.45 + step * phi^-1 * 0.15 * weight))
        B_tint = clamp(B * (0.72 + step * phi^-1 * 0.22 * weight))
        """
        step = max(0, min(step, 4))
        weight = max(0.0, min(weight, 1.0))
        phi_inv = INV_GOLDEN_RATIO

        r_base, g_base, b_base = base_color

        if style == "pink_panther_chic":
            r_scale = 1.0 + (step * phi_inv * 0.35 * weight)
            g_scale = 0.45 + (step * phi_inv * 0.15 * weight)
            b_scale = 0.72 + (step * phi_inv * 0.22 * weight)
        elif style == "noir_monochrome":
            mono = int(0.299 * r_base + 0.587 * g_base + 0.114 * b_base)
            r_scale = 1.0 + (step * phi_inv * 0.10 * weight)
            g_scale = 0.30
            b_scale = 0.40
            r_base, g_base, b_base = mono, mono, mono
        else:  # golden_champagne
            r_scale = 1.10 + (step * phi_inv * 0.20 * weight)
            g_scale = 0.90 + (step * phi_inv * 0.15 * weight)
            b_scale = 0.40 + (step * phi_inv * 0.10 * weight)

        r_out = int(max(0, min(255, round(r_base * r_scale))))
        g_out = int(max(0, min(255, round(g_base * g_scale))))
        b_out = int(max(0, min(255, round(b_base * b_scale))))

        hex_code = f"#{r_out:02x}{g_out:02x}{b_out:02x}"
        ratio = (r_out + g_out + b_out) / (3.0 * 255.0)

        desc = f"Harmonic Tint Step {step} [Style: {style}, Weight: {weight:.2f}, Phi: {GOLDEN_RATIO:.3f}]"

        return HarmonicColorTint(
            step=step,
            r=r_out,
            g=g_out,
            b=b_out,
            hex_code=hex_code,
            tint_ratio=ratio,
            description=desc
        )

    def get_pink_panther_palette(self) -> Dict[str, Dict[str, Any]]:
        """Returns the canonical Pink Panther & Noir palette dictionary."""
        return PINK_PANTHER_PALETTE

    # --------------------------------------------------------------------------
    # LEVEL 1 & 2: Macro Metropolises & Meso Street Networks
    # --------------------------------------------------------------------------

    def get_hierarchy_levels(self) -> List[Dict[str, Any]]:
        """Returns all 5 tiers of the recast MRP Engine Hierarchy."""
        return [lvl.to_dict() for lvl in self._hierarchy_levels]

    def get_canonical_regions(self) -> Dict[str, Dict[str, Any]]:
        """Returns all 12 canonical world regions."""
        return {k: v.to_dict() for k, v in self._world_regions.items()}

    def get_region(self, region_id: str) -> Optional[Dict[str, Any]]:
        spec = self._world_regions.get(region_id)
        return spec.to_dict() if spec else None

    def generate_procedural_street_network(
        self,
        region_id: str,
        grid_width: float = 800.0,
        grid_height: float = 600.0,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates a Google Street Map / OSM style procedural vector road network
        tailored to the region's specific street topology and Pink Panther color scheme.
        """
        region = self._world_regions.get(region_id, self._world_regions["gotham_city"])
        rng = random.Random(seed if seed is not None else 42 + sum(ord(c) for c in region_id))

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        parcels: List[Dict[str, Any]] = []

        topology = region.street_topology

        if "grid" in topology:
            # Art-Deco or Industrial Grid (Gotham, Ohio, etc.)
            cols = 6
            rows = 5
            dx = grid_width / (cols + 1)
            dy = grid_height / (rows + 1)

            node_grid = []
            node_counter = 0
            for r in range(rows):
                row_nodes = []
                for c in range(cols):
                    jitter_x = rng.uniform(-10.0, 10.0)
                    jitter_y = rng.uniform(-10.0, 10.0)
                    nx = (c + 1) * dx + jitter_x
                    ny = (r + 1) * dy + jitter_y
                    node_id = f"node_{node_counter}"
                    node_counter += 1
                    n_dict = {"id": node_id, "x": round(nx, 1), "y": round(ny, 1), "type": "intersection"}
                    nodes.append(n_dict)
                    row_nodes.append(node_id)
                node_grid.append(row_nodes)

            # Horizontal & Vertical edges
            edge_counter = 0
            for r in range(rows):
                for c in range(cols - 1):
                    edges.append({
                        "id": f"edge_{edge_counter}",
                        "source": node_grid[r][c],
                        "target": node_grid[r][c + 1],
                        "lanes": 4 if r % 2 == 0 else 2,
                        "name": f"Avenue {r + 1}",
                        "speed_limit": region.default_speed_limit_kmh,
                        "road_type": "boulevard" if r % 2 == 0 else "street"
                    })
                    edge_counter += 1

            for c in range(cols):
                for r in range(rows - 1):
                    edges.append({
                        "id": f"edge_{edge_counter}",
                        "source": node_grid[r][c],
                        "target": node_grid[r + 1][c],
                        "lanes": 4 if c % 2 == 0 else 2,
                        "name": f"Boulevard {chr(65 + c)}",
                        "speed_limit": region.default_speed_limit_kmh,
                        "road_type": "boulevard" if c % 2 == 0 else "cross_street"
                    })
                    edge_counter += 1

        elif "serpentine" in topology or "ribbon" in topology or "switchback" in topology:
            # Alpine / Volcanic / Coastal Ribbon (Bolivia, Peru, Canary Islands)
            num_points = 12
            node_ids = []
            for i in range(num_points):
                t = i / (num_points - 1)
                curve_x = 80.0 + (grid_width - 160.0) * t
                wave = math.sin(t * math.pi * 3.5) * (grid_height * 0.35)
                curve_y = (grid_height * 0.5) + wave + rng.uniform(-15.0, 15.0)
                n_id = f"switchback_{i}"
                node_ids.append(n_id)
                nodes.append({"id": n_id, "x": round(curve_x, 1), "y": round(curve_y, 1), "type": "cliff_turn"})

            for i in range(len(node_ids) - 1):
                edges.append({
                    "id": f"edge_ribbon_{i}",
                    "source": node_ids[i],
                    "target": node_ids[i + 1],
                    "lanes": 2,
                    "name": f"Pass Segment {i + 1}",
                    "speed_limit": region.default_speed_limit_kmh,
                    "road_type": "hairpin_pass"
                })

        else:
            # Grand Strip or Highway Corridor (Vegas, Florida, Australia, Alabama)
            mid_y = grid_height * 0.5
            main_nodes = []
            for i in range(7):
                mx = 60.0 + i * ((grid_width - 120.0) / 6.0)
                my = mid_y + rng.uniform(-20.0, 20.0)
                m_id = f"strip_node_{i}"
                main_nodes.append(m_id)
                nodes.append({"id": m_id, "x": round(mx, 1), "y": round(my, 1), "type": "boulevard_junction"})

            for i in range(len(main_nodes) - 1):
                edges.append({
                    "id": f"strip_edge_{i}",
                    "source": main_nodes[i],
                    "target": main_nodes[i + 1],
                    "lanes": 6,
                    "name": "The Great Strip Highway",
                    "speed_limit": region.default_speed_limit_kmh + 20,
                    "road_type": "super_highway"
                })

            # Feeder perpendicular roads
            for idx, mn in enumerate(main_nodes):
                fn_top = f"feeder_top_{idx}"
                fn_bot = f"feeder_bot_{idx}"
                nx = nodes[idx]["x"]
                nodes.append({"id": fn_top, "x": nx, "y": 80.0, "type": "feeder"})
                nodes.append({"id": fn_bot, "x": nx, "y": grid_height - 80.0, "type": "feeder"})
                edges.append({
                    "id": f"edge_feed_top_{idx}",
                    "source": fn_top,
                    "target": mn,
                    "lanes": 2,
                    "name": f"Casino Feeder {idx}N",
                    "speed_limit": region.default_speed_limit_kmh,
                    "road_type": "side_avenue"
                })
                edges.append({
                    "id": f"edge_feed_bot_{idx}",
                    "source": mn,
                    "target": fn_bot,
                    "lanes": 2,
                    "name": f"Casino Feeder {idx}S",
                    "speed_limit": region.default_speed_limit_kmh,
                    "road_type": "side_avenue"
                })

        # Generate procedural city parcels / building blocks
        for i in range(12):
            px = rng.uniform(40.0, grid_width - 100.0)
            py = rng.uniform(40.0, grid_height - 100.0)
            pw = rng.uniform(35.0, 75.0)
            ph = rng.uniform(30.0, 65.0)
            parcels.append({
                "id": f"parcel_{i}",
                "x": round(px, 1),
                "y": round(py, 1),
                "width": round(pw, 1),
                "height": round(ph, 1),
                "facade_tint": region.primary_tint if i % 2 == 0 else region.secondary_tint,
                "building_type": "deco_tower" if i % 3 == 0 else ("residential" if i % 3 == 1 else "commercial")
            })

        return {
            "region_id": region_id,
            "region_name": region.display_name,
            "street_topology": region.street_topology,
            "bounds": {"width": grid_width, "height": grid_height},
            "nodes_count": len(nodes),
            "edges_count": len(edges),
            "parcels_count": len(parcels),
            "nodes": nodes,
            "edges": edges,
            "parcels": parcels,
            "osm_metadata": {
                "osm_bounding_box": list(region.osm_bounding_box),
                "asphalt_shade": region.asphalt_shade,
                "primary_tint": region.primary_tint,
                "secondary_tint": region.secondary_tint,
                "ambient_weather": region.ambient_weather
            }
        }

    # --------------------------------------------------------------------------
    # LEVEL 3 & 4: Micro 2D Vehicles & Strict 6 Max HP Vital Invariant
    # --------------------------------------------------------------------------

    def get_active_vehicles(self, region_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns all registered 2D cruiser vehicles, optionally filtered by region."""
        if region_id:
            return [v.to_dict() for v in self._active_vehicles.values() if v.region_id == region_id]
        return [v.to_dict() for v in self._active_vehicles.values()]

    def get_vehicle(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        v = self._active_vehicles.get(vehicle_id)
        return v.to_dict() if v else None

    def register_vehicle(self, cruiser: VehicleCruiser2D) -> Dict[str, Any]:
        """Registers a new 2D cruiser, strictly enforcing 6 Max HP invariant."""
        cruiser.clamp_hp()
        self._active_vehicles[cruiser.vehicle_id] = cruiser
        return cruiser.to_dict()

    def simulate_vehicle_tick(
        self,
        vehicle_id: str,
        dt_seconds: float = 0.05,
        throttle: float = 1.0,     # -1.0 (reverse) to 1.0 (full forward)
        steering: float = 0.0,     # -1.0 (left) to 1.0 (right)
        handbrake: bool = False,
        drift_boost: bool = False
    ) -> Dict[str, Any]:
        """
        Simulates 2D driving kinematics with steering, drifting, friction, and
        vital status integrity under the strict 6 Max HP rule.
        """
        cruiser = self._active_vehicles.get(vehicle_id)
        if not cruiser:
            raise KeyError(f"Vehicle '{vehicle_id}' not found in active roster")

        if cruiser.status == "destroyed":
            return cruiser.to_dict()

        throttle = max(-1.0, min(1.0, throttle))
        steering = max(-1.0, min(1.0, steering))
        dt = max(0.001, min(0.5, dt_seconds))

        # Acceleration or braking
        if handbrake:
            cruiser.is_drifting = True
            cruiser.velocity_mps = max(0.0, cruiser.velocity_mps - cruiser.braking_mps2 * 1.5 * dt)
        elif throttle > 0:
            target_accel = cruiser.acceleration_mps2 * throttle
            cruiser.velocity_mps = min(cruiser.max_velocity_mps, cruiser.velocity_mps + target_accel * dt)
        elif throttle < 0:
            target_brake = cruiser.braking_mps2 * abs(throttle)
            cruiser.velocity_mps = max(-10.0, cruiser.velocity_mps - target_brake * dt)
        else:
            # Natural rolling resistance / air drag
            cruiser.velocity_mps = max(0.0, cruiser.velocity_mps - 1.5 * dt)

        # Steering & Drift Kinematics
        effective_turn_rate = cruiser.turn_rate_degps * steering
        if abs(steering) > 0.4 and cruiser.velocity_mps > 10.0:
            cruiser.is_drifting = True
            if drift_boost:
                cruiser.drift_factor = min(1.0, cruiser.drift_factor + 0.1)
        else:
            cruiser.is_drifting = False

        # Apply heading update
        cruiser.heading_deg = (cruiser.heading_deg + effective_turn_rate * dt) % 360.0

        # Translation update in 2D plane
        rad = math.radians(cruiser.heading_deg)
        drift_angle = rad + (math.radians(25.0 * cruiser.drift_factor) if cruiser.is_drifting else 0.0)

        cruiser.x += cruiser.velocity_mps * math.cos(drift_angle) * dt * 10.0  # Scale for 2D pixels
        cruiser.y += cruiser.velocity_mps * math.sin(drift_angle) * dt * 10.0

        # Enforce bounds wrap-around (0 to 1000)
        cruiser.x = cruiser.x % 1000.0
        cruiser.y = cruiser.y % 800.0

        # Invariant check
        cruiser.clamp_hp()

        return cruiser.to_dict()

    def apply_damage_to_vehicle(self, vehicle_id: str, damage_amount: int) -> Dict[str, Any]:
        """
        Applies damage to a vehicle cruiser, strictly respecting armor and
        the 6 Max HP vital invariant (HP never drops below 0 or exceeds 6).
        """
        cruiser = self._active_vehicles.get(vehicle_id)
        if not cruiser:
            raise KeyError(f"Vehicle '{vehicle_id}' not found")

        net_damage = max(1, damage_amount - max(0, cruiser.armor // 2))
        cruiser.hp = max(0, cruiser.hp - net_damage)
        cruiser.clamp_hp()
        return cruiser.to_dict()

    def repair_vehicle(self, vehicle_id: str, repair_amount: int = 2) -> Dict[str, Any]:
        """
        Repairs vehicle cruiser, strictly capped at VITAL_MAX_HP == 6.
        """
        cruiser = self._active_vehicles.get(vehicle_id)
        if not cruiser:
            raise KeyError(f"Vehicle '{vehicle_id}' not found")

        if cruiser.status == "destroyed":
            # Rebuild chassis
            cruiser.hp = 1
            cruiser.status = "critical"
        else:
            cruiser.hp = min(VITAL_MAX_HP, cruiser.hp + repair_amount)

        cruiser.clamp_hp()
        return cruiser.to_dict()

    def get_engine_manifesto(self) -> Dict[str, Any]:
        """Returns the complete systemic manifesto of the MRP Harmonic Street Engine."""
        return {
            "title": "MRP Harmonic Street Engine & Pink Panther Chic Aesthetic Framework",
            "version": "1.0.0",
            "vital_max_hp_invariant": VITAL_MAX_HP,
            "golden_ratio_phi": GOLDEN_RATIO,
            "total_mrp_hierarchy_levels": len(self._hierarchy_levels),
            "total_canonical_world_regions": len(self._world_regions),
            "total_active_cruisers": len(self._active_vehicles),
            "pink_panther_palette_keys": list(PINK_PANTHER_PALETTE.keys()),
            "canonical_regions": list(self._world_regions.keys()),
            "status": "active_and_invariant_compliant"
        }


# Global singleton instance
GLOBAL_MRP_HARMONIC_STREET_ENGINE = MRPHarmonicStreetEngine()
