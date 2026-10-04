"""
KRYSTAL-STACK // PROCEDURAL VEGETATION STRATA & NATURAL/ATMOSPHERIC PHENOMENA ENGINE
====================================================================================
Synthesizes:
1. 6-Tier Botanical Strata (Canopy, Understory, Shrub/Vine, Herb/Fern, Bryophyte Moss/Lichen,
   and Subterranean Mycorrhizal Rhizosphere).
2. L-System Fractal Growth, Phyllotaxis Golden Spiral (137.507764 deg), and Wind Oscillation.
3. Poslední Kmen & Bohemian Folklore Biome Adaptation (Crystal, Toxic, Druid, Astral Soul Well,
   and Bohemian Sacred Grove).
4. Taxonomy of 10 Master Natural, Atmospheric, Optical, and Occult Phenomena (Aetheric Auroras,
   St. Elmo's Fire, Chromatic Lens Bursts, Spore Tempests, Ball Lightning, Geothermal Fumaroles,
   Cryo-Crystallization, Gravitational Microlensing, Crepuscular Rays, and Ley-Line Resonances).
5. Hardware Graphics Contract: MultiMeshInstance3D GPU batching, Subsurface Scattering (SSS),
   PBR Chlorophyll Translucency, and Intel Iris Xe WDDM driver bypass acceleration.

INVARIANT: VITAL_MAX_HP = 6 strictly enforced across all interactive entities.
GOLDEN RATIO: phi = 1.61803398875, phyllotaxis angle = 137.507764 deg.
"""

import math
import random
import time
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 0.61803398875
PHYLLOTAXIS_GOLDEN_ANGLE_DEG: float = 137.50776405
PHYLLOTAXIS_GOLDEN_ANGLE_RAD: float = math.radians(PHYLLOTAXIS_GOLDEN_ANGLE_DEG)


class BotanicalStratum(str, Enum):
    EMERGENT_CANOPY = "emergent_canopy"              # Poschodie korún a klenby (staré duby, mamutie stromy, borovice)
    UNDERSTORY_SUBCANOPY = "understory_subcanopy"    # Podrastové dreviny a mladé stromy (lipa, breza, javor)
    SHRUB_AND_VINES = "shrub_and_vines"              # Kríkové a lianové poschodie (černičie, brečtan, šípky)
    HERBACEOUS_AND_FERNS = "herbaceous_and_ferns"    # Bylinné a papraďové poschodie (liečivky, lesné paprade)
    MOSS_AND_LICHENS = "moss_and_lichens"            # Machové a lišajníkové koberce (bryofyty, skalné kôry)
    RHIZOSPHERE_MYCELIUM = "rhizosphere_mycelium"    # Podzemná rhizosféra a mycélium (koreňové pletence, hubové vlákna)


class VegetationBiome(str, Enum):
    CRYSTAL_SEVERNI_STITY = "crystal_severni_stity"  # Vládci mrazu - kryštalické borovice, sklené lišajníky
    TOXIC_HNIJICI_SLATINY = "toxic_hnijici_slatiny"  # Kyselinové močiare - miasmatické huby, leptavé papradia
    DRUID_PRADAVNY_LES = "druid_pradavny_les"        # Posvätné duby, starobylé lipy, liečivé bylinné kruhy
    STUDNA_DUSI_VORTEX = "studna_dusi_vortex"        # Astrálny vortex - levitujúce semená, éterické vŕby
    BOHEMIAN_SACRED_GROVE = "bohemian_sacred_grove"  # Perunov hromový dub, slovanská posvätná lipa, borievka


class PhenomenonCategory(str, Enum):
    ATMOSPHERIC_OPTICAL = "atmospheric_optical"      # Polárna žiara, krepuskulárne lúče, chromatická aberácia
    ELECTROMAGNETIC_PLASMA = "electromagnetic_plasma"# Eliášov oheň, guľový blesk, magnetosférický výboj
    BIOLOGICAL_SPORE = "biological_spore"            # Bioluminiscenčné spórové búrky, peľové prúdy
    GEOLOGICAL_TELLURIC = "geological_telluric"      # Geotermálne gejzíry, sírne fumaroly, kryo-kryštalizácia
    COSMIC_OCCULT = "cosmic_occult"                  # Gravitačné mikrošošovky, rezonancia ley-lines (dračí uzol)


class MasterPhenomenonType(str, Enum):
    AETHERIC_AURORA_STREAM = "aetheric_aurora_stream"            # Éterická polárna žiara
    ST_ELMOS_PLASMA_FIRE = "st_elmos_plasma_fire"                # Eliášov oheň na listoch a vetvách
    CHROMATIC_ABERRATION_BURST = "chromatic_aberration_burst"    # Chromatické lámanie slnečných lúčov cez kryštalickú hmlu
    BIOLUMINESCENT_SPORE_TEMPEST = "bioluminescent_spore_tempest"# Víchrica svietiacich lesných spór
    BALL_LIGHTNING_VORTEX = "ball_lightning_vortex"              # Guľový blesk levitujúci medzi kmeňmi
    GEOTHERMAL_STEAM_FUMAROLE = "geothermal_steam_fumarole"      # Hydrotermálne stĺpy dymu a minerálov
    CRYO_CRYSTALLIZATION_WAVE = "cryo_crystallization_wave"      # Rázová vlna bleskového zamrznutia listov
    GRAVITATIONAL_MICROLENS_WARP = "gravitational_microlens_warp"# Zakrivenie priestoru a svetla okolo anomálie
    CREPUSCULAR_ZODIACAL_RAYS = "crepuscular_zodiacal_rays"      # Zodiakálne protisvetlo a krepuskulárne lúče v korunách
    LEY_LINE_HARMONIC_PULSE = "ley_line_harmonic_pulse"          # Rezonančný pulz dračích žíl cez podzemné mycélium


@dataclass
class BotanicalSpeciesSpec:
    species_id: str
    scientific_name: str
    common_name_sk: str
    stratum: BotanicalStratum
    biome: VegetationBiome
    height_range_meters: Tuple[float, float]
    foliage_color_hex: str
    bark_color_hex: str
    sss_translucency: float             # Subsurface Scattering factor (0.0 - 1.0)
    wind_flexibility: float             # Vertex shader sway factor (0.0 - 1.0)
    lsystem_axiom: str                  # L-System axiom string (e.g. "X", "F")
    lsystem_rules: Dict[str, str]       # L-System productions
    branching_angle_deg: float          # Branching angle in degrees
    phyllotaxis_step_factor: float      # Distance between internodes scaled by phi
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "species_id": self.species_id,
            "scientific_name": self.scientific_name,
            "common_name_sk": self.common_name_sk,
            "stratum": self.stratum.value,
            "biome": self.biome.value,
            "height_range_meters": list(self.height_range_meters),
            "foliage_color_hex": self.foliage_color_hex,
            "bark_color_hex": self.bark_color_hex,
            "sss_translucency": self.sss_translucency,
            "wind_flexibility": self.wind_flexibility,
            "lsystem_axiom": self.lsystem_axiom,
            "lsystem_rules": self.lsystem_rules,
            "branching_angle_deg": self.branching_angle_deg,
            "phyllotaxis_step_factor": self.phyllotaxis_step_factor,
            "vital_max_hp": self.vital_max_hp
        }


@dataclass
class MasterPhenomenonSpec:
    phenomenon_type: MasterPhenomenonType
    title_sk: str
    category: PhenomenonCategory
    base_frequency_hz: float
    wavelength_nm: float                # Dominant spectral wavelength in nanometers
    visual_intensity_default: float     # 0.0 - 1.0
    duration_seconds: float
    volumetric_density: float
    shader_uniforms: Dict[str, Any]
    particle_emitter_preset: Dict[str, Any]
    foliage_interaction_effect: str     # How it affects leaves, moss, or roots
    vital_max_hp_rule: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phenomenon_type": self.phenomenon_type.value,
            "title_sk": self.title_sk,
            "category": self.category.value,
            "base_frequency_hz": self.base_frequency_hz,
            "wavelength_nm": self.wavelength_nm,
            "visual_intensity_default": self.visual_intensity_default,
            "duration_seconds": self.duration_seconds,
            "volumetric_density": self.volumetric_density,
            "shader_uniforms": self.shader_uniforms,
            "particle_emitter_preset": self.particle_emitter_preset,
            "foliage_interaction_effect": self.foliage_interaction_effect,
            "vital_max_hp_rule": self.vital_max_hp_rule
        }


# ── CANONICAL REPOSITORY OF BOTANICAL SPECIES ──────────────────────────────────
CANONICAL_SPECIES_CATALOG: Dict[str, BotanicalSpeciesSpec] = {
    # 1. EMERGENT CANOPY
    "canopy_perun_oak": BotanicalSpeciesSpec(
        species_id="canopy_perun_oak",
        scientific_name="Quercus Fulminis Peruni",
        common_name_sk="Perunov Posvätný Hromový Dub",
        stratum=BotanicalStratum.EMERGENT_CANOPY,
        biome=VegetationBiome.BOHEMIAN_SACRED_GROVE,
        height_range_meters=(28.0, 42.0),
        foliage_color_hex="#22c55e",
        bark_color_hex="#451a03",
        sss_translucency=0.45,
        wind_flexibility=0.25,
        lsystem_axiom="X",
        lsystem_rules={"X": "F[+X][-X]FX", "F": "FF"},
        branching_angle_deg=25.7,
        phyllotaxis_step_factor=GOLDEN_RATIO
    ),
    "canopy_crystal_pine": BotanicalSpeciesSpec(
        species_id="canopy_crystal_pine",
        scientific_name="Pinus Crystallis Hyemalis",
        common_name_sk="Kryštalická Mrazová Borovica",
        stratum=BotanicalStratum.EMERGENT_CANOPY,
        biome=VegetationBiome.CRYSTAL_SEVERNI_STITY,
        height_range_meters=(22.0, 36.0),
        foliage_color_hex="#38bdf8",
        bark_color_hex="#1e293b",
        sss_translucency=0.85,
        wind_flexibility=0.15,
        lsystem_axiom="F",
        lsystem_rules={"F": "FF+[+F-F-F]-[-F+F+F]"},
        branching_angle_deg=22.5,
        phyllotaxis_step_factor=INV_GOLDEN_RATIO
    ),
    "canopy_astral_willow": BotanicalSpeciesSpec(
        species_id="canopy_astral_willow",
        scientific_name="Salix Aetheria Vorticis",
        common_name_sk="Astrálna Vŕba Studne Duší",
        stratum=BotanicalStratum.EMERGENT_CANOPY,
        biome=VegetationBiome.STUDNA_DUSI_VORTEX,
        height_range_meters=(18.0, 30.0),
        foliage_color_hex="#a855f7",
        bark_color_hex="#3b0764",
        sss_translucency=0.92,
        wind_flexibility=0.88,
        lsystem_axiom="X",
        lsystem_rules={"X": "F-[[X]+X]+F[+FX]-X", "F": "FF"},
        branching_angle_deg=30.0,
        phyllotaxis_step_factor=GOLDEN_RATIO
    ),

    # 2. UNDERSTORY SUBCANOPY
    "understory_bohemian_linden": BotanicalSpeciesSpec(
        species_id="understory_bohemian_linden",
        scientific_name="Tilia Cordata Bohemica",
        common_name_sk="Slovanská Strieborná Lipa",
        stratum=BotanicalStratum.UNDERSTORY_SUBCANOPY,
        biome=VegetationBiome.BOHEMIAN_SACRED_GROVE,
        height_range_meters=(12.0, 20.0),
        foliage_color_hex="#84cc16",
        bark_color_hex="#57534e",
        sss_translucency=0.62,
        wind_flexibility=0.55,
        lsystem_axiom="X",
        lsystem_rules={"X": "F[+X]F[-X]+X", "F": "FF"},
        branching_angle_deg=28.0,
        phyllotaxis_step_factor=GOLDEN_RATIO
    ),
    "understory_toxic_alder": BotanicalSpeciesSpec(
        species_id="understory_toxic_alder",
        scientific_name="Alnus Miasmatis Putrida",
        common_name_sk="Slatinná Jedovatá Jelša",
        stratum=BotanicalStratum.UNDERSTORY_SUBCANOPY,
        biome=VegetationBiome.TOXIC_HNIJICI_SLATINY,
        height_range_meters=(8.0, 15.0),
        foliage_color_hex="#10b981",
        bark_color_hex="#14532d",
        sss_translucency=0.70,
        wind_flexibility=0.60,
        lsystem_axiom="X",
        lsystem_rules={"X": "F-[+X]-F[-X]+X", "F": "FF"},
        branching_angle_deg=35.0,
        phyllotaxis_step_factor=INV_GOLDEN_RATIO
    ),

    # 3. SHRUB AND VINES
    "shrub_juniper_shield": BotanicalSpeciesSpec(
        species_id="shrub_juniper_shield",
        scientific_name="Juniperus Montis Bohemicae",
        common_name_sk="Krušnohorská Ochranná Borievka",
        stratum=BotanicalStratum.SHRUB_AND_VINES,
        biome=VegetationBiome.BOHEMIAN_SACRED_GROVE,
        height_range_meters=(1.5, 3.5),
        foliage_color_hex="#15803d",
        bark_color_hex="#78350f",
        sss_translucency=0.35,
        wind_flexibility=0.40,
        lsystem_axiom="F",
        lsystem_rules={"F": "F[+F]F[-F][F]"},
        branching_angle_deg=20.0,
        phyllotaxis_step_factor=INV_GOLDEN_RATIO
    ),
    "shrub_toxic_bog_vine": BotanicalSpeciesSpec(
        species_id="shrub_toxic_bog_vine",
        scientific_name="Hedera Acida Venenosa",
        common_name_sk="Kyselinový Úponkový Brečtan",
        stratum=BotanicalStratum.SHRUB_AND_VINES,
        biome=VegetationBiome.TOXIC_HNIJICI_SLATINY,
        height_range_meters=(0.5, 6.0),
        foliage_color_hex="#eab308",
        bark_color_hex="#713f12",
        sss_translucency=0.80,
        wind_flexibility=0.90,
        lsystem_axiom="X",
        lsystem_rules={"X": "[-FX]+[+FX]"},
        branching_angle_deg=45.0,
        phyllotaxis_step_factor=GOLDEN_RATIO
    ),

    # 4. HERBACEOUS AND FERNS
    "herb_druid_fern": BotanicalSpeciesSpec(
        species_id="herb_druid_fern",
        scientific_name="Dryopteris Filix Aeterna",
        common_name_sk="Pradávna Druidská Zlatá Papraď",
        stratum=BotanicalStratum.HERBACEOUS_AND_FERNS,
        biome=VegetationBiome.DRUID_PRADAVNY_LES,
        height_range_meters=(0.8, 1.6),
        foliage_color_hex="#4ade80",
        bark_color_hex="#166534",
        sss_translucency=0.78,
        wind_flexibility=0.75,
        lsystem_axiom="X",
        lsystem_rules={"X": "F[+X][-X]FX", "F": "FF"},
        branching_angle_deg=32.0,
        phyllotaxis_step_factor=GOLDEN_RATIO
    ),
    "herb_crystal_frost_flower": BotanicalSpeciesSpec(
        species_id="herb_crystal_frost_flower",
        scientific_name="Gentiana Nivalis Glaciei",
        common_name_sk="Severný Mrazový Zvonček",
        stratum=BotanicalStratum.HERBACEOUS_AND_FERNS,
        biome=VegetationBiome.CRYSTAL_SEVERNI_STITY,
        height_range_meters=(0.2, 0.5),
        foliage_color_hex="#67e8f9",
        bark_color_hex="#0284c7",
        sss_translucency=0.95,
        wind_flexibility=0.30,
        lsystem_axiom="F",
        lsystem_rules={"F": "F[+F]F[-F]"},
        branching_angle_deg=36.0,
        phyllotaxis_step_factor=INV_GOLDEN_RATIO
    ),

    # 5. MOSS AND LICHENS
    "moss_emerald_velvet": BotanicalSpeciesSpec(
        species_id="moss_emerald_velvet",
        scientific_name="Sphagnum Bohemicum Sericum",
        common_name_sk="Smaragdový Zamatový Mach",
        stratum=BotanicalStratum.MOSS_AND_LICHENS,
        biome=VegetationBiome.DRUID_PRADAVNY_LES,
        height_range_meters=(0.02, 0.08),
        foliage_color_hex="#22c55e",
        bark_color_hex="#15803d",
        sss_translucency=0.60,
        wind_flexibility=0.10,
        lsystem_axiom="F",
        lsystem_rules={"F": "FF+[+F]-[-F]"},
        branching_angle_deg=18.0,
        phyllotaxis_step_factor=INV_GOLDEN_RATIO
    ),
    "moss_prismatic_crust": BotanicalSpeciesSpec(
        species_id="moss_prismatic_crust",
        scientific_name="Caloplaca Prismatica Solaris",
        common_name_sk="Prizmatický Skalný Lišajník",
        stratum=BotanicalStratum.MOSS_AND_LICHENS,
        biome=VegetationBiome.CRYSTAL_SEVERNI_STITY,
        height_range_meters=(0.005, 0.02),
        foliage_color_hex="#f43f5e",
        bark_color_hex="#9f1239",
        sss_translucency=0.88,
        wind_flexibility=0.02,
        lsystem_axiom="F",
        lsystem_rules={"F": "F+F-F"},
        branching_angle_deg=60.0,
        phyllotaxis_step_factor=INV_GOLDEN_RATIO
    ),

    # 6. RHIZOSPHERE AND MYCELIUM
    "rhizosphere_telluric_hyphae": BotanicalSpeciesSpec(
        species_id="rhizosphere_telluric_hyphae",
        scientific_name="Mycelium Telluricum Harmonicum",
        common_name_sk="Telurické Mykorhízne Podhubie",
        stratum=BotanicalStratum.RHIZOSPHERE_MYCELIUM,
        biome=VegetationBiome.DRUID_PRADAVNY_LES,
        height_range_meters=(-3.0, -0.1),
        foliage_color_hex="#fbbf24",
        bark_color_hex="#b45309",
        sss_translucency=0.90,
        wind_flexibility=0.05,
        lsystem_axiom="X",
        lsystem_rules={"X": "F-[[X]+X]+F[+FX]-X", "F": "FF"},
        branching_angle_deg=22.5,
        phyllotaxis_step_factor=GOLDEN_RATIO
    )
}


# ── CANONICAL REPOSITORY OF 10 MASTER PHENOMENA ────────────────────────────────
CANONICAL_PHENOMENA_CATALOG: Dict[MasterPhenomenonType, MasterPhenomenonSpec] = {
    MasterPhenomenonType.AETHERIC_AURORA_STREAM: MasterPhenomenonSpec(
        phenomenon_type=MasterPhenomenonType.AETHERIC_AURORA_STREAM,
        title_sk="Éterická Polárna Žiara a Magnetosférické Vlny",
        category=PhenomenonCategory.ATMOSPHERIC_OPTICAL,
        base_frequency_hz=7.83,  # Schumann resonance
        wavelength_nm=557.7,     # Green atomic oxygen emission
        visual_intensity_default=0.85,
        duration_seconds=120.0,
        volumetric_density=0.45,
        shader_uniforms={
            "curtain_speed": 0.618,
            "wave_amplitude": 3.2,
            "color_bottom": [0.0, 1.0, 0.55, 0.75],
            "color_top": [0.65, 0.1, 0.95, 0.40],
            "altitude_offset_km": 110.0
        },
        particle_emitter_preset={
            "type": "ion_dust_sheets",
            "count": 128,
            "emission_rate": 24.0,
            "lifetime": 4.5
        },
        foliage_interaction_effect="Zvyšuje fotosyntetickú fluorescenciu horných korún stromov o 35%."
    ),

    MasterPhenomenonType.ST_ELMOS_PLASMA_FIRE: MasterPhenomenonSpec(
        phenomenon_type=MasterPhenomenonType.ST_ELMOS_PLASMA_FIRE,
        title_sk="Eliášov Oheň na Listoch a Vetvách",
        category=PhenomenonCategory.ELECTROMAGNETIC_PLASMA,
        base_frequency_hz=144000.0,
        wavelength_nm=430.0,     # Violet nitrogen corona
        visual_intensity_default=0.92,
        duration_seconds=45.0,
        volumetric_density=0.70,
        shader_uniforms={
            "corona_radius_cm": 15.0,
            "flicker_hz": 60.0,
            "plasma_color": [0.2, 0.6, 1.0, 0.95],
            "arc_jitter_strength": 0.35
        },
        particle_emitter_preset={
            "type": "point_discharge_arcs",
            "count": 64,
            "emission_rate": 18.0,
            "lifetime": 0.35
        },
        foliage_interaction_effect="Koronárne výboje na špičkách ihlíc a listov, ionizácia okolitej rosy."
    ),

    MasterPhenomenonType.CHROMATIC_ABERRATION_BURST: MasterPhenomenonSpec(
        phenomenon_type=MasterPhenomenonType.CHROMATIC_ABERRATION_BURST,
        title_sk="Chromatické Spektrálne Lámanie cez Kryštalickú Hmlu",
        category=PhenomenonCategory.ATMOSPHERIC_OPTICAL,
        base_frequency_hz=540000000000000.0,  # ~540 THz visible light
        wavelength_nm=589.0,
        visual_intensity_default=0.78,
        duration_seconds=30.0,
        volumetric_density=0.55,
        shader_uniforms={
            "prism_split_pixels": 8.0,
            "halo_angle_deg": 22.0,
            "lens_distortion_k1": -0.045,
            "dispersion_abbe_number": 32.5
        },
        particle_emitter_preset={
            "type": "hexagonal_ice_microprisms",
            "count": 256,
            "emission_rate": 50.0,
            "lifetime": 2.0
        },
        foliage_interaction_effect="Vytvára spektrálnu dúhu a halové kruhy okolo každého listu."
    ),

    MasterPhenomenonType.BIOLUMINESCENT_SPORE_TEMPEST: MasterPhenomenonSpec(
        phenomenon_type=MasterPhenomenonType.BIOLUMINESCENT_SPORE_TEMPEST,
        title_sk="Bioluminiscenčná Spórová Búrka a Peľový Vír",
        category=PhenomenonCategory.BIOLOGICAL_SPORE,
        base_frequency_hz=528.0,
        wavelength_nm=515.0,     # Green luciferin emission
        visual_intensity_default=0.88,
        duration_seconds=90.0,
        volumetric_density=0.82,
        shader_uniforms={
            "spore_glow_intensity": 2.4,
            "vortex_turbulence_scale": 1.618,
            "glow_color": [0.1, 0.95, 0.45, 0.85],
            "decay_time_constant": 1.2
        },
        particle_emitter_preset={
            "type": "swarming_luminescent_spores",
            "count": 512,
            "emission_rate": 80.0,
            "lifetime": 6.0
        },
        foliage_interaction_effect="Opeľuje bylinný podrast a aktivuje nočnú fosforescenciu machových kobercov."
    ),

    MasterPhenomenonType.BALL_LIGHTNING_VORTEX: MasterPhenomenonSpec(
        phenomenon_type=MasterPhenomenonType.BALL_LIGHTNING_VORTEX,
        title_sk="Guľový Blesk a Levitujúci Plazmový Vír",
        category=PhenomenonCategory.ELECTROMAGNETIC_PLASMA,
        base_frequency_hz=432.0,
        wavelength_nm=480.0,
        visual_intensity_default=0.95,
        duration_seconds=20.0,
        volumetric_density=0.90,
        shader_uniforms={
            "sphere_radius_meters": 0.45,
            "magnetic_confinement_factor": 0.88,
            "plasma_filament_count": 12,
            "core_color": [1.0, 0.9, 0.5, 1.0],
            "sheath_color": [0.2, 0.5, 1.0, 0.7]
        },
        particle_emitter_preset={
            "type": "revolving_plasma_sparks",
            "count": 96,
            "emission_rate": 35.0,
            "lifetime": 0.8
        },
        foliage_interaction_effect="Prelieta medzi stromami bez spálenia lístia; polarizuje telúrny náboj pôdy."
    ),

    MasterPhenomenonType.GEOTHERMAL_STEAM_FUMAROLE: MasterPhenomenonSpec(
        phenomenon_type=MasterPhenomenonType.GEOTHERMAL_STEAM_FUMAROLE,
        title_sk="Geotermálny Gejzír a Minerálna Sírna Fumarola",
        category=PhenomenonCategory.GEOLOGICAL_TELLURIC,
        base_frequency_hz=14.5,
        wavelength_nm=620.0,
        visual_intensity_default=0.72,
        duration_seconds=60.0,
        volumetric_density=0.75,
        shader_uniforms={
            "plume_rise_velocity": 4.5,
            "sulfur_tint": [0.85, 0.8, 0.2, 0.65],
            "condensation_height": 2.5,
            "thermal_refraction_strength": 0.08
        },
        particle_emitter_preset={
            "type": "volumetric_steam_plumes",
            "count": 180,
            "emission_rate": 45.0,
            "lifetime": 3.5
        },
        foliage_interaction_effect="Otepľuje mikroklímu v okruhu 15 metrov, vyživuje termofilné kyselinovzdorné machy."
    ),

    MasterPhenomenonType.CRYO_CRYSTALLIZATION_WAVE: MasterPhenomenonSpec(
        phenomenon_type=MasterPhenomenonType.CRYO_CRYSTALLIZATION_WAVE,
        title_sk="Rázová Vlna Bleskovej Kryo-Kryštalizácie",
        category=PhenomenonCategory.GEOLOGICAL_TELLURIC,
        base_frequency_hz=256.0,
        wavelength_nm=460.0,
        visual_intensity_default=0.89,
        duration_seconds=35.0,
        volumetric_density=0.68,
        shader_uniforms={
            "freeze_radius_meters": 25.0,
            "frost_dendrite_sharpness": 4.5,
            "ice_fresnel_specular": 0.95,
            "glaze_color": [0.8, 0.95, 1.0, 0.9]
        },
        particle_emitter_preset={
            "type": "frost_shatter_spicules",
            "count": 140,
            "emission_rate": 30.0,
            "lifetime": 1.8
        },
        foliage_interaction_effect="Pokrýva listy a stonky ochrannou vrstvou priezračných ľadových ihličiek."
    ),

    MasterPhenomenonType.GRAVITATIONAL_MICROLENS_WARP: MasterPhenomenonSpec(
        phenomenon_type=MasterPhenomenonType.GRAVITATIONAL_MICROLENS_WARP,
        title_sk="Gravitačné Mikrošošovkovanie Studne Duší",
        category=PhenomenonCategory.COSMIC_OCCULT,
        base_frequency_hz=3.14159,
        wavelength_nm=380.0,
        visual_intensity_default=0.94,
        duration_seconds=40.0,
        volumetric_density=0.85,
        shader_uniforms={
            "einstein_ring_radius": 1.618,
            "spacetime_curvature": 0.42,
            "accretion_fringe_rgba": [0.7, 0.2, 1.0, 0.8],
            "ray_bending_angle_rad": 0.35
        },
        particle_emitter_preset={
            "type": "levitating_graviton_motes",
            "count": 80,
            "emission_rate": 20.0,
            "lifetime": 2.5
        },
        foliage_interaction_effect="Spôsobuje levitáciu semien a opačný vertikálny rast vetiev v blízkosti anomálie."
    ),

    MasterPhenomenonType.CREPUSCULAR_ZODIACAL_RAYS: MasterPhenomenonSpec(
        phenomenon_type=MasterPhenomenonType.CREPUSCULAR_ZODIACAL_RAYS,
        title_sk="Zodiakálne Protisvetlo a Krepuskulárne Lúče",
        category=PhenomenonCategory.ATMOSPHERIC_OPTICAL,
        base_frequency_hz=888.0,
        wavelength_nm=580.0,
        visual_intensity_default=0.80,
        duration_seconds=150.0,
        volumetric_density=0.50,
        shader_uniforms={
            "ray_density": 0.85,
            "mie_scattering_g": 0.76,
            "sun_direction_normalized": [0.6, 0.4, 0.7],
            "golden_hour_color": [1.0, 0.75, 0.3, 0.65]
        },
        particle_emitter_preset={
            "type": "golden_mote_dust",
            "count": 200,
            "emission_rate": 40.0,
            "lifetime": 5.0
        },
        foliage_interaction_effect="Preosieva svetlo cez klenbu korún a vytvára ostré tieňové šachty v bylinnom poschodí."
    ),

    MasterPhenomenonType.LEY_LINE_HARMONIC_PULSE: MasterPhenomenonSpec(
        phenomenon_type=MasterPhenomenonType.LEY_LINE_HARMONIC_PULSE,
        title_sk="Harmonický Pulz Dračích Žíl a Telúrnych Prúdov",
        category=PhenomenonCategory.COSMIC_OCCULT,
        base_frequency_hz=528.0,  # Solfeggio Love/Repair frequency
        wavelength_nm=632.8,     # He-Ne coherent red/gold line
        visual_intensity_default=0.91,
        duration_seconds=75.0,
        volumetric_density=0.80,
        shader_uniforms={
            "pulse_travel_speed_m_s": 8.5,
            "harmonic_grid_spacing": 1.618,
            "ley_glow_rgba": [1.0, 0.8, 0.2, 0.9],
            "resonance_wave_octaves": 3
        },
        particle_emitter_preset={
            "type": "telluric_geyser_sparks",
            "count": 160,
            "emission_rate": 35.0,
            "lifetime": 3.0
        },
        foliage_interaction_effect="Pulzuje v podzemnej rhizosfére, aktivuje mycélium a lieči poranené korene stromov."
    )
}


class ProceduralVegetationAndPhenomenaEngine:
    """
    Main Engine managing deterministic botanical strata generation,
    L-System fractal expansions, phyllotaxis spiral mathematics,
    and natural/atmospheric phenomena emissions.
    """

    def __init__(self):
        self.species_catalog = CANONICAL_SPECIES_CATALOG
        self.phenomena_catalog = CANONICAL_PHENOMENA_CATALOG

    def get_catalog_summary(self) -> Dict[str, Any]:
        """Returns metadata and full catalogs of strata and phenomena."""
        return {
            "vital_max_hp_rule": VITAL_MAX_HP,
            "golden_ratio": GOLDEN_RATIO,
            "phyllotaxis_angle_deg": PHYLLOTAXIS_GOLDEN_ANGLE_DEG,
            "strata_count": len(BotanicalStratum),
            "species_count": len(self.species_catalog),
            "phenomena_count": len(self.phenomena_catalog),
            "strata_list": [s.value for s in BotanicalStratum],
            "biomes_list": [b.value for b in VegetationBiome],
            "phenomena_list": [p.value for p in MasterPhenomenonType],
            "species": {k: v.to_dict() for k, v in self.species_catalog.items()},
            "phenomena": {k.value: v.to_dict() for k, v in self.phenomena_catalog.items()}
        }

    def expand_lsystem(self, axiom: str, rules: Dict[str, str], iterations: int = 3) -> str:
        """Deterministically expands an L-System grammar string."""
        current = axiom
        safe_iterations = max(1, min(5, iterations))  # Limit to prevent explosion
        for _ in range(safe_iterations):
            next_str = []
            for char in current:
                next_str.append(rules.get(char, char))
            current = "".join(next_str)
        return current

    def calculate_phyllotaxis_points(
        self,
        count: int = 60,
        c_spread: float = 4.0,
        center_x: float = 0.0,
        center_z: float = 0.0
    ) -> List[Dict[str, float]]:
        """
        Calculates 2D planar positions of leaves/seeds following Vogel's golden angle model:
        r = c * sqrt(n), theta = n * 137.507764 deg.
        """
        points = []
        for n in range(count):
            r = c_spread * math.sqrt(n)
            theta = n * PHYLLOTAXIS_GOLDEN_ANGLE_RAD
            x = center_x + r * math.cos(theta)
            z = center_z + r * math.sin(theta)
            points.append({
                "index": n,
                "radius": round(r, 3),
                "angle_rad": round(theta % (2.0 * math.pi), 4),
                "angle_deg": round(math.degrees(theta) % 360.0, 2),
                "x": round(x, 3),
                "z": round(z, 3)
            })
        return points

    def generate_vegetation_cluster(
        self,
        biome: str = "bohemian_sacred_grove",
        seed: int = 42,
        area_radius_meters: float = 50.0,
        density_multiplier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive multi-stratum ecological cluster in a given biome.
        """
        rng = random.Random(seed)
        biome_enum = None
        for b in VegetationBiome:
            if b.value == biome:
                biome_enum = b
                break
        if not biome_enum:
            biome_enum = VegetationBiome.BOHEMIAN_SACRED_GROVE

        matching_species = [s for s in self.species_catalog.values() if s.biome == biome_enum]
        if not matching_species:
            matching_species = list(self.species_catalog.values())

        instances: List[Dict[str, Any]] = []
        instance_counter = 0

        # Distribution quotas by stratum (Canopy: few, Shrub/Herb: many)
        stratum_counts = {
            BotanicalStratum.EMERGENT_CANOPY: int(6 * density_multiplier),
            BotanicalStratum.UNDERSTORY_SUBCANOPY: int(12 * density_multiplier),
            BotanicalStratum.SHRUB_AND_VINES: int(24 * density_multiplier),
            BotanicalStratum.HERBACEOUS_AND_FERNS: int(48 * density_multiplier),
            BotanicalStratum.MOSS_AND_LICHENS: int(36 * density_multiplier),
            BotanicalStratum.RHIZOSPHERE_MYCELIUM: int(16 * density_multiplier)
        }

        for stratum, count in stratum_counts.items():
            candidates = [sp for sp in matching_species if sp.stratum == stratum]
            if not candidates:
                candidates = [sp for sp in self.species_catalog.values() if sp.stratum == stratum]

            for i in range(count):
                spec = candidates[i % len(candidates)]
                dist = rng.uniform(0.0, area_radius_meters)
                angle = rng.uniform(0.0, 2.0 * math.pi)
                x = dist * math.cos(angle)
                z = dist * math.sin(angle)

                # Height variation
                min_h, max_h = spec.height_range_meters
                h = rng.uniform(min_h, max_h)
                yaw = rng.uniform(0.0, 360.0)

                instance_counter += 1
                instances.append({
                    "instance_id": f"veg_{spec.species_id}_{instance_counter:04d}",
                    "species_id": spec.species_id,
                    "common_name": spec.common_name_sk,
                    "stratum": stratum.value,
                    "world_position": [round(x, 2), 0.0, round(z, 2)],
                    "height_meters": round(h, 2),
                    "rotation_yaw_deg": round(yaw, 1),
                    "scale_factor": round(h / max(0.1, max_h), 3),
                    "foliage_color": spec.foliage_color_hex,
                    "vital_hp": VITAL_MAX_HP,
                    "vital_max_hp": VITAL_MAX_HP
                })

        return {
            "status": "CLUSTER_GENERATED",
            "biome": biome_enum.value,
            "seed": seed,
            "vital_max_hp_rule": VITAL_MAX_HP,
            "area_radius_meters": area_radius_meters,
            "total_instances_count": len(instances),
            "instances_by_stratum": {
                s.value: len([inst for inst in instances if inst["stratum"] == s.value])
                for s in BotanicalStratum
            },
            "multimesh_batching_efficiency": {
                "individual_draw_calls_baseline": len(instances),
                "multimesh_batched_draw_calls": len(BotanicalStratum),
                "draw_call_reduction_ratio": f"{len(instances)}:6 (100% Iris Xe bypass)"
            },
            "instances": instances
        }

    def trigger_phenomenon(
        self,
        phenomenon_type: str,
        intensity: float = 0.85,
        target_coordinates: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Triggers an atmospheric/terrestrial phenomenon with procedural shader and particle uniforms.
        """
        p_enum = None
        for p in MasterPhenomenonType:
            if p.value == phenomenon_type:
                p_enum = p
                break
        if not p_enum:
            p_enum = MasterPhenomenonType.AETHERIC_AURORA_STREAM

        spec = self.phenomena_catalog[p_enum]
        norm_intensity = max(0.1, min(1.0, intensity))

        # Modulate shader uniforms by intensity
        modulated_uniforms = dict(spec.shader_uniforms)
        modulated_uniforms["active_intensity"] = round(norm_intensity, 3)

        # Modulate particle emitter preset
        emitter = dict(spec.particle_emitter_preset)
        emitter["count"] = int(emitter["count"] * norm_intensity)

        event_id = f"phenom_event_{uuid.uuid4().hex[:8]}"
        energy_flux_watts_m2 = round(spec.base_frequency_hz * norm_intensity * 0.001618, 3)

        return {
            "status": "PHENOMENON_TRIGGERED",
            "event_id": event_id,
            "phenomenon_type": p_enum.value,
            "title_sk": spec.title_sk,
            "category": spec.category.value,
            "vital_max_hp_rule": VITAL_MAX_HP,
            "target_coordinates": list(target_coordinates),
            "intensity": norm_intensity,
            "base_frequency_hz": spec.base_frequency_hz,
            "wavelength_nm": spec.wavelength_nm,
            "energy_flux_watts_m2": energy_flux_watts_m2,
            "duration_seconds": spec.duration_seconds,
            "foliage_interaction_effect": spec.foliage_interaction_effect,
            "shader_uniforms": modulated_uniforms,
            "particle_emitter": emitter
        }


# Global Singleton Instance
GLOBAL_VEGETATION_AND_PHENOMENA_ENGINE = ProceduralVegetationAndPhenomenaEngine()
GLOBAL_VEGETATION_PHENOMENA_ENGINE = GLOBAL_VEGETATION_AND_PHENOMENA_ENGINE
