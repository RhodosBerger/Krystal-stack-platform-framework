"""
KRYSTAL-STACK: SKEUOMORPHIC PROCEDURAL SYNTHESIS ENGINE
======================================================
Replaces abstract mathematical fractals and noise with tangible, skeuomorphic
reproductions of real-world items, humanoid characters, and tactile environments.

Key Methodical Axioms:
1. Physical Material Substrates (Aged Walnut, Damascus Steel, Stitched Leather,
   Champagne Brass, Blown Glass, Parchment, Travertine Stone).
2. Golden Ratio Ergonomic & Vitruvian Proportions (Phi = 1.61803398875).
3. Structural Constructibility: Seams, bevels, rivets, joinery, and fasteners.
4. Tripartite Multi-Substrate Parity:
   - High-Fidelity Skeuomorphic SVG/Canvas Vector Generator
   - Tactile ASCII Visual Silhouettes
   - Godot 4.x Forward+ .tscn Scene Graphs
   - Modern Java 21 Virtual Thread Records
   - Janet Functional DSL Immutable AST
5. Strict Invariants:
   - VITAL_MAX_HP = 6 on all characters, weapons, and tools.
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


# =============================================================================
# 1. PHYSICAL MATERIAL SUBSTRATE TAXONOMY
# =============================================================================

@dataclass(frozen=True)
class MaterialSubstrate:
    """A real-world physical material with optical and tactile characteristics."""
    substrate_id: str
    name_sk: str
    category: str
    albedo_hex: str
    secondary_tint_hex: str
    roughness: float
    metallic: float
    specular: float
    tactile_feature: str
    ascii_glyph: str


MATERIAL_SUBSTRATES: Dict[str, MaterialSubstrate] = {
    "AGED_BOHEMIAN_WALNUT": MaterialSubstrate(
        substrate_id="AGED_BOHEMIAN_WALNUT",
        name_sk="Starožitný Český Orech",
        category="wood",
        albedo_hex="#4a2e18",
        secondary_tint_hex="#2e1a0b",
        roughness=0.38,
        metallic=0.02,
        specular=0.45,
        tactile_feature="Satinovo voskovaný povrch s výraznou štruktúrou letokruhov",
        ascii_glyph="≡"
    ),
    "FORGED_DAMASCUS_STEEL": MaterialSubstrate(
        substrate_id="FORGED_DAMASCUS_STEEL",
        name_sk="Kovaná Damascénska Oceľ",
        category="metal",
        albedo_hex="#cbd5e1",
        secondary_tint_hex="#64748b",
        roughness=0.22,
        metallic=0.94,
        specular=0.88,
        tactile_feature="Kyselinou leptaný vlnitý damaškový vzor s mikroskosenými fazetami",
        ascii_glyph="═"
    ),
    "SADDLE_STITCHED_LEATHER": MaterialSubstrate(
        substrate_id="SADDLE_STITCHED_LEATHER",
        name_sk="Ručne Prešívaná Hovädzia Useň",
        category="leather",
        albedo_hex="#78350f",
        secondary_tint_hex="#451a03",
        roughness=0.72,
        metallic=0.05,
        specular=0.35,
        tactile_feature="Prírodná zrnitá useň s obvodovým sedlárskym stehom voskovanou niťou",
        ascii_glyph="░"
    ),
    "TARNISHED_CHAMPAGNE_BRASS": MaterialSubstrate(
        substrate_id="TARNISHED_CHAMPAGNE_BRASS",
        name_sk="Patinovaná Šampanská Mosadz",
        category="metal",
        albedo_hex="#f59e0b",
        secondary_tint_hex="#b45309",
        roughness=0.30,
        metallic=0.88,
        specular=0.82,
        tactile_feature="Jemne kartáčovaný kov s teplou patinou a gravírovanou kalibráciou",
        ascii_glyph="▒"
    ),
    "ALCHEMICAL_BLOWN_GLASS": MaterialSubstrate(
        substrate_id="ALCHEMICAL_BLOWN_GLASS",
        name_sk="Fúkané Alchymistické Sklo",
        category="glass",
        albedo_hex="#e0f2fe",
        secondary_tint_hex="#38bdf8",
        roughness=0.08,
        metallic=0.00,
        specular=0.95,
        tactile_feature="Priehľadné hrubostenné sklo s vysokým lomom svetla (n=1.52) a meniskom",
        ascii_glyph="○"
    ),
    "ILLUMINATED_PARCHMENT": MaterialSubstrate(
        substrate_id="ILLUMINATED_PARCHMENT",
        name_sk="Ručne Kreslený Pergamen",
        category="parchment",
        albedo_hex="#fef3c7",
        secondary_tint_hex="#fde68a",
        roughness=0.82,
        metallic=0.00,
        specular=0.15,
        tactile_feature="Organické teľacie vellum s prirodzene nerovným okrajom a kaligrafickým atramentom",
        ascii_glyph="·"
    ),
    "ROMAN_TRAVERTINE_STONE": MaterialSubstrate(
        substrate_id="ROMAN_TRAVERTINE_STONE",
        name_sk="Rímsky Sekaný Travertín",
        category="stone",
        albedo_hex="#d6d3d1",
        secondary_tint_hex="#a8a29e",
        roughness=0.68,
        metallic=0.03,
        specular=0.25,
        tactile_feature="Pórovitý prírodný kameň s viditeľnými stopami po kamenárskom dláte",
        ascii_glyph="▓"
    ),
}


# =============================================================================
# 2. SKEUOMORPHIC REAL-WORLD ITEMS (DOMAIN A)
# =============================================================================

class SkeuomorphicItemType(str, Enum):
    ALCHEMIST_LEATHER_GRIMOIRE = "alchemist_leather_grimoire"
    FORGED_DAMASCUS_DAGGER = "forged_damascus_dagger"
    BRASS_ASTROLABE_SEXTANT = "brass_astrolabe_sextant"
    POTION_CRYSTAL_FLASK = "potion_crystal_flask"


@dataclass
class ItemComponent:
    name: str
    substrate_id: str
    width_mm: float
    height_mm: float
    depth_mm: float
    fastener_type: str  # Rivets, Stitches, Mortise/Tenon, Wax Seal, Thread
    affordance_description: str


@dataclass
class SkeuomorphicItem:
    item_id: str
    item_type: SkeuomorphicItemType
    name_sk: str
    seed: int
    overall_width_mm: float
    overall_height_mm: float
    overall_depth_mm: float
    golden_ratio_fit_score: float
    primary_substrate: str
    components: List[ItemComponent]
    vital_hp: int = VITAL_MAX_HP
    svg_vector_rendering: str = ""
    ascii_tactile_silhouette: str = ""

    def __post_init__(self) -> None:
        if self.vital_hp > VITAL_MAX_HP:
            self.vital_hp = VITAL_MAX_HP


# =============================================================================
# 3. SKEUOMORPHIC REAL-WORLD CHARACTERS (DOMAIN B)
# =============================================================================

class CharacterArchetype(str, Enum):
    BOHEMIAN_ALCHEMIST_HERO = "bohemian_alchemist_hero"
    CRYSTAL_KNIGHT_GUARDIAN = "crystal_knight_guardian"
    DRUID_WOODLAND_RANGER = "druid_woodland_ranger"


@dataclass
class ClothingLayer:
    layer_name: str
    substrate_id: str
    fit_style: str  # Fitted, Draped, Articulated, Buckled
    color_hex: str
    seam_detail: str


@dataclass
class SkeuomorphicCharacter:
    character_id: str
    archetype: CharacterArchetype
    name_sk: str
    seed: int
    height_cm: float
    head_ratio_vitruvian: float  # Exactly 8.0 heads
    shoulder_width_cm: float
    vital_hp: int = VITAL_MAX_HP
    clothing_layers: List[ClothingLayer] = field(default_factory=list)
    equipped_items: List[str] = field(default_factory=list)
    svg_character_portrait: str = ""
    ascii_character_silhouette: str = ""

    def __post_init__(self) -> None:
        if self.vital_hp > VITAL_MAX_HP:
            self.vital_hp = VITAL_MAX_HP


# =============================================================================
# 4. SKEUOMORPHIC REAL-WORLD ENVIRONMENTS / ROOMS (DOMAIN C)
# =============================================================================

class RoomArchetype(str, Enum):
    ALCHEMIST_WORKSHOP_CHAMBER = "alchemist_workshop_chamber"
    MASTER_FORGE_ARMORY = "master_forge_armory"
    GOTHIC_LIBRARY_STUDIO = "gothic_library_studio"


@dataclass
class ArchitecturalFeature:
    feature_name: str
    substrate_id: str
    pos_x_m: float
    pos_y_m: float
    pos_z_m: float
    width_m: float
    height_m: float
    tactile_joinery: str


@dataclass
class SkeuomorphicRoom:
    room_id: str
    archetype: RoomArchetype
    name_sk: str
    seed: int
    width_m: float
    length_m: float
    ceiling_height_m: float
    floor_substrate: str
    wall_substrate: str
    features: List[ArchitecturalFeature]
    vital_hp: int = VITAL_MAX_HP
    svg_elevation_rendering: str = ""
    ascii_interior_view: str = ""


# =============================================================================
# 5. PROCEDURAL SYNTHESIS ENGINE
# =============================================================================

class SkeuomorphicProceduralEngine:
    """
    Synthesizes tangible, real-world skeuomorphic items, characters, and rooms.
    Strictly departs from abstract fractals while enforcing Golden Ratio geometry,
    material textures, and VITAL_MAX_HP = 6.
    """

    def __init__(self) -> None:
        self.substrates = MATERIAL_SUBSTRATES

    # -------------------------------------------------------------------------
    # SYNTHESIZE SKEUOMORPHIC ITEM
    # -------------------------------------------------------------------------
    def synthesize_item(
        self,
        seed: int = 42,
        item_type: SkeuomorphicItemType = SkeuomorphicItemType.FORGED_DAMASCUS_DAGGER
    ) -> SkeuomorphicItem:
        """Procedurally crafts a tangible real-world item."""
        rng = random.Random(seed)

        if item_type == SkeuomorphicItemType.FORGED_DAMASCUS_DAGGER:
            # Golden ratio proportions: Blade length / Handle length = 1.618
            handle_len_mm = 115.0 + rng.uniform(-5.0, 5.0)
            blade_len_mm = handle_len_mm * GOLDEN_RATIO  # ~186mm
            guard_w_mm = handle_len_mm * 0.65            # ~75mm
            total_len = handle_len_mm + blade_len_mm + 12.0

            components = [
                ItemComponent(
                    name="Damašková Čepeľ s Drážkou",
                    substrate_id="FORGED_DAMASCUS_STEEL",
                    width_mm=28.0,
                    height_mm=blade_len_mm,
                    depth_mm=4.5,
                    fastener_type="Kovaný Tŕň cez Rukoväť",
                    affordance_description="Obojstranne brúsená čepeľ s centrálnou odľahčovacou drážkou (fuller)"
                ),
                ItemComponent(
                    name="Tvarovaná Rukoväť z Orecha",
                    substrate_id="AGED_BOHEMIAN_WALNUT",
                    width_mm=24.0,
                    height_mm=handle_len_mm,
                    depth_mm=22.0,
                    fastener_type="Mosadzné Nity & Ovinutie Drôtom",
                    affordance_description="Ergonomické zárezy pre prsty a jemné ryhované drážky proti kĺzaniu"
                ),
                ItemComponent(
                    name="Ochranná Mosadzná Záštita",
                    substrate_id="TARNISHED_CHAMPAGNE_BRASS",
                    width_mm=guard_w_mm,
                    height_mm=12.0,
                    depth_mm=14.0,
                    fastener_type="Za tepla nalisované uloženie",
                    affordance_description="Zakrivené ramená záštity chrániace ruku bojovníka"
                ),
                ItemComponent(
                    name="Hruška Rukoväte (Pommel)",
                    substrate_id="TARNISHED_CHAMPAGNE_BRASS",
                    width_mm=26.0,
                    height_mm=16.0,
                    depth_mm=24.0,
                    fastener_type="Zanitovaný koniec tŕňa",
                    affordance_description="Vyvažovacie ťažisko stabilizujúce zbraň v ruke"
                )
            ]

            ascii_art = (
                "         /\\         \n"
                "        /  \\        \n"
                "       | || |       \n"
                "       | || |       \n"
                "       | || |       \n"
                "       | || |       \n"
                "      [======]      \n"
                "        (==)        \n"
                "        (==)        \n"
                "        (==)        \n"
                "        (==)        \n"
                "         ()         \n"
                "   [Kovaný Damašek] "
            )

            svg_xml = self._generate_dagger_svg(blade_len_mm, handle_len_mm, guard_w_mm)

            return SkeuomorphicItem(
                item_id=f"item_dagger_{seed:05d}",
                item_type=item_type,
                name_sk="Kovaná Damašková Dýka",
                seed=seed,
                overall_width_mm=guard_w_mm,
                overall_height_mm=total_len,
                overall_depth_mm=24.0,
                golden_ratio_fit_score=0.992,
                primary_substrate="FORGED_DAMASCUS_STEEL",
                components=components,
                vital_hp=VITAL_MAX_HP,
                svg_vector_rendering=svg_xml,
                ascii_tactile_silhouette=ascii_art
            )

        elif item_type == SkeuomorphicItemType.ALCHEMIST_LEATHER_GRIMOIRE:
            # Golden rectangle aspect ratio: Height / Width = 1.618
            w_mm = 160.0
            h_mm = w_mm * GOLDEN_RATIO  # ~258.8 mm
            d_mm = 42.0

            components = [
                ItemComponent(
                    name="Kožená Väzba z Hovädzej Usne",
                    substrate_id="SADDLE_STITCHED_LEATHER",
                    width_mm=w_mm + 8.0,
                    height_mm=h_mm + 12.0,
                    depth_mm=d_mm,
                    fastener_type="Sedlársky voskovaný obvodový steh",
                    affordance_description="Za tepla vtláčaný alchymistický reliéf s ochranou chrbta"
                ),
                ItemComponent(
                    name="Kované Mosadzné Rohové Kovania",
                    substrate_id="TARNISHED_CHAMPAGNE_BRASS",
                    width_mm=32.0,
                    height_mm=32.0,
                    depth_mm=4.0,
                    fastener_type="Obojstranné guľové nity",
                    affordance_description="Štyri rohové výstuhy chrániace knihu pred opotrebovaním"
                ),
                ItemComponent(
                    name="Puklicová Zámková Pracka",
                    substrate_id="TARNISHED_CHAMPAGNE_BRASS",
                    width_mm=48.0,
                    height_mm=24.0,
                    depth_mm=8.0,
                    fastener_type="Kĺbový čap s perom",
                    affordance_description="Mechanická západka držiaca pergamenové listy pevne zovreté"
                ),
                ItemComponent(
                    name="Knižný Blok z Pergamenu (320 strán)",
                    substrate_id="ILLUMINATED_PARCHMENT",
                    width_mm=w_mm,
                    height_mm=h_mm,
                    depth_mm=d_mm - 6.0,
                    fastener_type="Konopné šité väzy",
                    affordance_description="Zlatené oriezy a ručne kaligrafované receptúry"
                )
            ]

            ascii_art = (
                "   .==================.   \n"
                "  /||  [☼]      [☼]  ||\\  \n"
                " //||                ||\\\\ \n"
                "|| ||   ALCHYMISTICKÝ|| ||\n"
                "|| ||     GRIMOÁR    || ||\n"
                "|| ||     [===o]     || ||\n"
                "|| ||   RECEPTÚRY    || ||\n"
                " \\\\||                ||// \n"
                "  \\||  [☼]      [☼]  ||/  \n"
                "   '=================='   \n"
                "    [Koža & Mosadz]       "
            )

            svg_xml = self._generate_grimoire_svg(w_mm, h_mm, d_mm)

            return SkeuomorphicItem(
                item_id=f"item_grimoire_{seed:05d}",
                item_type=item_type,
                name_sk="Kožený Alchymistický Grimoár",
                seed=seed,
                overall_width_mm=w_mm,
                overall_height_mm=h_mm,
                overall_depth_mm=d_mm,
                golden_ratio_fit_score=0.998,
                primary_substrate="SADDLE_STITCHED_LEATHER",
                components=components,
                vital_hp=VITAL_MAX_HP,
                svg_vector_rendering=svg_xml,
                ascii_tactile_silhouette=ascii_art
            )

        elif item_type == SkeuomorphicItemType.POTION_CRYSTAL_FLASK:
            w_mm = 85.0
            h_mm = w_mm * GOLDEN_RATIO  # ~137.5 mm
            d_mm = 85.0

            components = [
                ItemComponent(
                    name="Fúkaná Guľovitá Banka",
                    substrate_id="ALCHEMICAL_BLOWN_GLASS",
                    width_mm=w_mm,
                    height_mm=h_mm * 0.75,
                    depth_mm=d_mm,
                    fastener_type="Tavené bezšvíkové sklo",
                    affordance_description="Hrubostenná sklenená banka s bublinkami a mierkou"
                ),
                ItemComponent(
                    name="Rezaná Korková Zátka",
                    substrate_id="AGED_BOHEMIAN_WALNUT",
                    width_mm=28.0,
                    height_mm=32.0,
                    depth_mm=28.0,
                    fastener_type="Tlakové tesnenie voskom",
                    affordance_description="Korkový uzáver zapečatený červeným včelím voskom s pečaťou"
                ),
                ItemComponent(
                    name="Mosadzné Hrdlové Objímky",
                    substrate_id="TARNISHED_CHAMPAGNE_BRASS",
                    width_mm=34.0,
                    height_mm=14.0,
                    depth_mm=34.0,
                    fastener_type="Kované pásky",
                    affordance_description="Závesné očká pre upevnenie na opasok dobrodruha"
                )
            ]

            ascii_art = (
                "        [===]       \n"
                "         | |        \n"
                "       .-' '-.      \n"
                "      /       \\     \n"
                "     |  ~ ~ ~  |    \n"
                "     | ~ ~ ~ ~ |    \n"
                "      \\_______/     \n"
                "   [Elixír 6 HP]    "
            )

            svg_xml = self._generate_flask_svg(w_mm, h_mm)

            return SkeuomorphicItem(
                item_id=f"item_flask_{seed:05d}",
                item_type=item_type,
                name_sk="Krištáľový Elixírový Flakón",
                seed=seed,
                overall_width_mm=w_mm,
                overall_height_mm=h_mm,
                overall_depth_mm=d_mm,
                golden_ratio_fit_score=0.988,
                primary_substrate="ALCHEMICAL_BLOWN_GLASS",
                components=components,
                vital_hp=VITAL_MAX_HP,
                svg_vector_rendering=svg_xml,
                ascii_tactile_silhouette=ascii_art
            )

        else:  # SkeuomorphicItemType.BRASS_ASTROLABE_SEXTANT
            diam_mm = 145.0
            h_mm = diam_mm * 1.15

            components = [
                ItemComponent(
                    name="Mosadzné Základné Teleso (Mater)",
                    substrate_id="TARNISHED_CHAMPAGNE_BRASS",
                    width_mm=diam_mm,
                    height_mm=diam_mm,
                    depth_mm=8.0,
                    fastener_type="Presné frézované čapy",
                    affordance_description="Kruhové gravírované teleso s 360-stupňovým uhlomerným delením"
                ),
                ItemComponent(
                    name="Otáčavá Pozorovacia Alhidáda",
                    substrate_id="FORGED_DAMASCUS_STEEL",
                    width_mm=18.0,
                    height_mm=diam_mm + 15.0,
                    depth_mm=3.5,
                    fastener_type="Stredový skrutkový nit s maticou",
                    affordance_description="Priehľadové štrbiny pre presné zameranie nebeských telies"
                )
            ]

            ascii_art = (
                "        .-^-.       \n"
                "       /  |  \\      \n"
                "      | --O-- |     \n"
                "       \\  |  /      \n"
                "        '-v-'       \n"
                "    [Astroláb 360°] "
            )

            svg_xml = self._generate_astrolabe_svg(diam_mm)

            return SkeuomorphicItem(
                item_id=f"item_astrolabe_{seed:05d}",
                item_type=item_type,
                name_sk="Mosadzný Hviezdny Astroláb",
                seed=seed,
                overall_width_mm=diam_mm,
                overall_height_mm=h_mm,
                overall_depth_mm=12.0,
                golden_ratio_fit_score=0.995,
                primary_substrate="TARNISHED_CHAMPAGNE_BRASS",
                components=components,
                vital_hp=VITAL_MAX_HP,
                svg_vector_rendering=svg_xml,
                ascii_tactile_silhouette=ascii_art
            )

    # -------------------------------------------------------------------------
    # SYNTHESIZE SKEUOMORPHIC CHARACTER
    # -------------------------------------------------------------------------
    def synthesize_character(
        self,
        seed: int = 101,
        archetype: CharacterArchetype = CharacterArchetype.BOHEMIAN_ALCHEMIST_HERO
    ) -> SkeuomorphicCharacter:
        """Procedurally crafts an anatomically proportioned skeuomorphic character."""
        rng = random.Random(seed)

        # Vitruvian 8-head canon with Golden Ratio waist / limbs
        height_cm = 182.0 + rng.uniform(-4.0, 4.0)
        head_h_cm = height_cm / 8.0  # Exactly 8 heads
        shoulder_w_cm = height_cm * (1.0 / (GOLDEN_RATIO ** 2))  # ~42.8 cm

        if archetype == CharacterArchetype.BOHEMIAN_ALCHEMIST_HERO:
            layers = [
                ClothingLayer(
                    layer_name="Tkaná Ľanová Košeľa",
                    substrate_id="ILLUMINATED_PARCHMENT",
                    fit_style="Fitted",
                    color_hex="#f8fafc",
                    seam_detail="Ručný krížikový steh okolo manžiet a goliera"
                ),
                ClothingLayer(
                    layer_name="Kožený Zástupcovský Dublet",
                    substrate_id="SADDLE_STITCHED_LEATHER",
                    fit_style="Fitted",
                    color_hex="#78350f",
                    seam_detail="Spevnené okraje s mosadznými dierkami na šnurovanie"
                ),
                ClothingLayer(
                    layer_name="Opasok s Laboratórnymi Vreckami",
                    substrate_id="SADDLE_STITCHED_LEATHER",
                    fit_style="Buckled",
                    color_hex="#451a03",
                    seam_detail="Masívna mosadzná pracka s úchytmi na flakóny (HP=6)"
                ),
                ClothingLayer(
                    layer_name="Vlhkosť odpudzujúci Plášť s Kapucňou",
                    substrate_id="AGED_BOHEMIAN_WALNUT",
                    fit_style="Draped",
                    color_hex="#1e293b",
                    seam_detail="Ťažký vlnený drapériový plášť zopnutý sponou"
                )
            ]
            items = ["Krištáľový Elixírový Flakón", "Kožený Alchymistický Grimoár"]

            ascii_char = (
                "       .-.       \n"
                "     /`   `\\     \n"
                "    | (o o) |    \n"
                "     \\  =  /     \n"
                "     .-'\"'-.     \n"
                "    / /| |\\ \\    \n"
                "   | | | | | |   \n"
                "   |_|[===]|_|   \n"
                "    |  | |  |    \n"
                "    |  | |  |    \n"
                "    [__] [__]    \n"
                "  [Alchymista 6HP]"
            )

        elif archetype == CharacterArchetype.CRYSTAL_KNIGHT_GUARDIAN:
            layers = [
                ClothingLayer(
                    layer_name="Prešívaná Vypchávka (Gambeson)",
                    substrate_id="ILLUMINATED_PARCHMENT",
                    fit_style="Fitted",
                    color_hex="#cbd5e1",
                    seam_detail="Husto prešívaná tlmiaca vrstva"
                ),
                ClothingLayer(
                    layer_name="Kovaný Oceľový Prsný Pancier",
                    substrate_id="FORGED_DAMASCUS_STEEL",
                    fit_style="Articulated",
                    color_hex="#94a3b8",
                    seam_detail="Fazetovaný stredový kýl odrážajúci nárazy zbraní"
                ),
                ClothingLayer(
                    layer_name="Segmentové Ramenné Náramenníky",
                    substrate_id="FORGED_DAMASCUS_STEEL",
                    fit_style="Articulated",
                    color_hex="#64748b",
                    seam_detail="Posuvné lamely nitované na kožených remeňoch"
                )
            ]
            items = ["Kovaná Damašková Dýka", "Oceľový Štít Suveréna"]

            ascii_char = (
                "       .-.       \n"
                "     /|   |\\     \n"
                "    | |[_]| |    \n"
                "     \\_v_v_/     \n"
                "     /|===|\\     \n"
                "    / /| |\\ \\    \n"
                "   | ||| ||| |   \n"
                "   |_|[===]|_|   \n"
                "    |  | |  |    \n"
                "    |  | |  |    \n"
                "    [__] [__]    \n"
                "   [Rytier 6 HP] "
            )

        else:  # DRUID_WOODLAND_RANGER
            layers = [
                ClothingLayer(
                    layer_name="Liesková Jemná Košeľa",
                    substrate_id="ILLUMINATED_PARCHMENT",
                    fit_style="Fitted",
                    color_hex="#dcfce7",
                    seam_detail="Zelenkavá prírodná priadza"
                ),
                ClothingLayer(
                    layer_name="Reliéfne Zdobená Lovcova Vesta",
                    substrate_id="SADDLE_STITCHED_LEATHER",
                    fit_style="Fitted",
                    color_hex="#166534",
                    seam_detail="Motív dubových listov vyrezávaný do usne"
                )
            ]
            items = ["Tisový Reflexný Luk", "Poľovnícky Tesák"]

            ascii_char = (
                "       .-.       \n"
                "     /`   `\\     \n"
                "    | (o o) |    \n"
                "     \\  -  /     \n"
                "     .-'\"'-.     \n"
                "    / /|♣|\\ \\    \n"
                "   | | | | | |   \n"
                "   |_|[===]|_|   \n"
                "    |  | |  |    \n"
                "    |  | |  |    \n"
                "    [__] [__]    \n"
                "   [Hraničiar]   "
            )

        return SkeuomorphicCharacter(
            character_id=f"char_{archetype.value}_{seed:05d}",
            archetype=archetype,
            name_sk=archetype.name.replace("_", " ").title(),
            seed=seed,
            height_cm=round(height_cm, 1),
            head_ratio_vitruvian=8.0,
            shoulder_width_cm=round(shoulder_w_cm, 1),
            vital_hp=VITAL_MAX_HP,  # Enforces HP = 6
            clothing_layers=layers,
            equipped_items=items,
            svg_character_portrait=self._generate_character_svg(archetype, height_cm),
            ascii_character_silhouette=ascii_char
        )

    # -------------------------------------------------------------------------
    # SYNTHESIZE SKEUOMORPHIC ENVIRONMENT / ROOM
    # -------------------------------------------------------------------------
    def synthesize_room(
        self,
        seed: int = 202,
        room_type: RoomArchetype = RoomArchetype.ALCHEMIST_WORKSHOP_CHAMBER
    ) -> SkeuomorphicRoom:
        """Procedurally constructs a real-world room with tactile architectural joinery."""
        rng = random.Random(seed)

        w_m = 7.5
        l_m = w_m * GOLDEN_RATIO  # ~12.13 m
        h_m = w_m * INV_GOLDEN_RATIO  # ~4.63 m (Harmonic ceiling height)

        features = [
            ArchitecturalFeature(
                feature_name="Kamenný Krb z Travertínu",
                substrate_id="ROMAN_TRAVERTINE_STONE",
                pos_x_m=0.0,
                pos_y_m=0.0,
                pos_z_m=-l_m * 0.45,
                width_m=2.4,
                height_m=3.2,
                tactile_joinery="Zarovnané kvádre spájané hydraulickým vápnom s liatinovým roštom"
            ),
            ArchitecturalFeature(
                feature_name="Masívny Orechový Pracovný Pult",
                substrate_id="AGED_BOHEMIAN_WALNUT",
                pos_x_m=w_m * 0.25,
                pos_y_m=0.0,
                pos_z_m=0.0,
                width_m=2.8,
                height_m=0.95,
                tactile_joinery="Rybinkové ozuby (dovetail joints) a mosadzné úchytky zásuviek"
            ),
            ArchitecturalFeature(
                feature_name="Vitrážové Olované Okno",
                substrate_id="ALCHEMICAL_BLOWN_GLASS",
                pos_x_m=-w_m * 0.48,
                pos_y_m=1.2,
                pos_z_m=0.0,
                width_m=1.8,
                height_m=2.4,
                tactile_joinery="Kosoštvorcové sklenené terče zalievané do olovených prútov"
            )
        ]

        ascii_room = (
            "+-----------------------------------------+\n"
            "| [Timber Roof Trusses - Masívne Krovy]   |\n"
            "|                                         |\n"
            "|  +======+      [===Krb===]              |\n"
            "|  |Okno  |      |  (o)    |              |\n"
            "|  |Vitráž|      |  / \\    |  +========+  |\n"
            "|  +======+      +---------+  |Orechový|  |\n"
            "|                             |Stôl    |  |\n"
            "|                             +========+  |\n"
            "|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|\n"
            "| [Parkety z Dubu & Rybie Kostry 45°]     |\n"
            "+-----------------------------------------+\n"
            " [Dielňa Alchymistu // Zlatý Rez 1.618]    "
        )

        return SkeuomorphicRoom(
            room_id=f"room_{room_type.value}_{seed:05d}",
            archetype=room_type,
            name_sk="Alchymistická Komora s Krbom",
            seed=seed,
            width_m=round(w_m, 2),
            length_m=round(l_m, 2),
            ceiling_height_m=round(h_m, 2),
            floor_substrate="AGED_BOHEMIAN_WALNUT",
            wall_substrate="ROMAN_TRAVERTINE_STONE",
            features=features,
            vital_hp=VITAL_MAX_HP,
            svg_elevation_rendering=self._generate_room_svg(w_m, l_m, h_m),
            ascii_interior_view=ascii_room
        )

    # =========================================================================
    # HIGH-FIDELITY SKEUOMORPHIC SVG VECTOR RENDERERS
    # =========================================================================

    def _generate_dagger_svg(self, blade_len: float, handle_len: float, guard_w: float) -> str:
        """Emits high-detail SVG for forged damascus dagger with realistic bevels and reflections."""
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 600" width="100%" height="100%">
  <defs>
    <!-- Damascus steel wavy gradient -->
    <linearGradient id="damascusGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#94a3b8"/>
      <stop offset="25%" stop-color="#f8fafc"/>
      <stop offset="50%" stop-color="#64748b"/>
      <stop offset="75%" stop-color="#cbd5e1"/>
      <stop offset="100%" stop-color="#475569"/>
    </linearGradient>
    <!-- Brass metallic gradient -->
    <linearGradient id="brassGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#b45309"/>
      <stop offset="35%" stop-color="#fbbf24"/>
      <stop offset="70%" stop-color="#f59e0b"/>
      <stop offset="100%" stop-color="#78350f"/>
    </linearGradient>
    <!-- Walnut wood grain -->
    <linearGradient id="walnutGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#451a03"/>
      <stop offset="50%" stop-color="#78350f"/>
      <stop offset="100%" stop-color="#2e1002"/>
    </linearGradient>
    <!-- Drop shadow -->
    <filter id="skeuoDropShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="8" dy="16" stdDeviation="12" flood-color="#000000" flood-opacity="0.65"/>
    </filter>
  </defs>

  <!-- Background Matte Plinth -->
  <rect width="400" height="600" fill="#09090b" rx="16"/>
  <rect x="20" y="20" width="360" height="560" fill="#121218" stroke="#ffffff" stroke-opacity="0.08" rx="12"/>

  <g transform="translate(200, 300)" filter="url(#skeuoDropShadow)">
    <!-- BLADE -->
    <path d="M 0,-240 L 26,-70 L 18,0 L -18,0 L -26,-70 Z" fill="url(#damascusGrad)" stroke="#334155" stroke-width="1.5"/>
    <!-- Blade Center Ridge & Fuller Groove -->
    <line x1="0" y1="-235" x2="0" y2="-15" stroke="#1e293b" stroke-width="2.5"/>
    <line x1="1" y1="-230" x2="1" y2="-20" stroke="#ffffff" stroke-opacity="0.7" stroke-width="1"/>

    <!-- CROSSGUARD -->
    <path d="M -75,-6 C -40,-12 40,-12 75,-6 C 82,2 80,10 70,8 C 30,2 -30,2 -70,8 C -80,10 -82,2 -75,-6 Z" fill="url(#brassGrad)" stroke="#78350f" stroke-width="1.5"/>

    <!-- HANDLE GRIP (WALNUT WOOD) -->
    <rect x="-14" y="8" width="28" height="110" rx="8" fill="url(#walnutGrad)" stroke="#2e1002" stroke-width="1.5"/>
    <!-- Spiral Brass Wire Wrap -->
    <line x1="-14" y1="25" x2="14" y2="35" stroke="url(#brassGrad)" stroke-width="2.5"/>
    <line x1="-14" y1="50" x2="14" y2="60" stroke="url(#brassGrad)" stroke-width="2.5"/>
    <line x1="-14" y1="75" x2="14" y2="85" stroke="url(#brassGrad)" stroke-width="2.5"/>
    <line x1="-14" y1="100" x2="14" y2="110" stroke="url(#brassGrad)" stroke-width="2.5"/>

    <!-- POMMEL (COUNTERWEIGHT) -->
    <circle cx="0" cy="132" r="16" fill="url(#brassGrad)" stroke="#78350f" stroke-width="2"/>
    <circle cx="0" cy="132" r="6" fill="#451a03"/>
  </g>

  <!-- Legend & Invariant Badge -->
  <text x="200" y="530" text-anchor="middle" fill="#f472b6" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="bold">KRYSTAL SKEUOMORPHIC: FORGED DAMASCUS DAGGER</text>
  <text x="200" y="550" text-anchor="middle" fill="#a1a1aa" font-family="'JetBrains Mono', monospace" font-size="10">Čepeľ: {blade_len:.1f}mm | Rukoväť: {handle_len:.1f}mm | Zlatý Rez: φ=1.618 | VITAL HP: 6</text>
</svg>"""

    def _generate_grimoire_svg(self, w: float, h: float, d: float) -> str:
        """Emits high-detail SVG for leather-bound alchemical grimoire with brass clasps."""
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 600" width="100%" height="100%">
  <defs>
    <linearGradient id="leatherGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#854d0e"/>
      <stop offset="50%" stop-color="#713f12"/>
      <stop offset="100%" stop-color="#3b1f04"/>
    </linearGradient>
    <linearGradient id="brassClasp" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#d97706"/>
      <stop offset="50%" stop-color="#fde68a"/>
      <stop offset="100%" stop-color="#92400e"/>
    </linearGradient>
    <filter id="bookShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="12" dy="20" stdDeviation="16" flood-color="#000000" flood-opacity="0.75"/>
    </filter>
  </defs>

  <rect width="400" height="600" fill="#09090b" rx="16"/>

  <g transform="translate(80, 80)" filter="url(#bookShadow)">
    <!-- Leather Cover Base -->
    <rect x="0" y="0" width="240" height="388" rx="14" fill="url(#leatherGrad)" stroke="#261202" stroke-width="3"/>

    <!-- Perimeter Saddle Stitching (Dashed Line) -->
    <rect x="12" y="12" width="216" height="364" rx="8" fill="none" stroke="#fde047" stroke-opacity="0.6" stroke-width="1.8" stroke-dasharray="4,4"/>

    <!-- Debossed Alchemical Seal -->
    <circle cx="120" cy="194" r="54" fill="none" stroke="#261202" stroke-width="4"/>
    <circle cx="120" cy="194" r="50" fill="none" stroke="#f59e0b" stroke-opacity="0.8" stroke-width="2"/>
    <polygon points="120,150 160,218 80,218" fill="none" stroke="#f59e0b" stroke-width="2"/>
    <polygon points="120,238 80,170 160,170" fill="none" stroke="#f59e0b" stroke-width="2"/>

    <!-- Brass Corner Protectors -->
    <path d="M 0,0 L 40,0 C 20,20 20,20 0,40 Z" fill="url(#brassClasp)" stroke="#78350f"/>
    <path d="M 240,0 L 200,0 C 220,20 220,20 240,40 Z" fill="url(#brassClasp)" stroke="#78350f"/>
    <path d="M 0,388 L 40,388 C 20,368 20,368 0,348 Z" fill="url(#brassClasp)" stroke="#78350f"/>
    <path d="M 240,388 L 200,388 C 220,368 220,368 240,348 Z" fill="url(#brassClasp)" stroke="#78350f"/>

    <!-- Brass Locking Clasp -->
    <rect x="220" y="176" width="32" height="36" rx="4" fill="url(#brassClasp)" stroke="#78350f" stroke-width="1.5"/>
    <circle cx="236" cy="194" r="5" fill="#451a03"/>
  </g>

  <text x="200" y="530" text-anchor="middle" fill="#f59e0b" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="bold">SKEUOMORPHIC LEATHER GRIMOIRE</text>
  <text x="200" y="550" text-anchor="middle" fill="#a1a1aa" font-family="'JetBrains Mono', monospace" font-size="10">Pomer strán: φ=1.618 | Useň &amp; Mosadz | 320 Pergamenových Listov</text>
</svg>"""

    def _generate_flask_svg(self, w: float, h: float) -> str:
        """Emits high-detail SVG for blown glass alchemist potion flask."""
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500" width="100%" height="100%">
  <defs>
    <radialGradient id="liquidGrad" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#38bdf8"/>
      <stop offset="60%" stop-color="#0284c7"/>
      <stop offset="100%" stop-color="#0369a1"/>
    </radialGradient>
    <linearGradient id="glassSpecular" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#ffffff" stop-opacity="0.8"/>
      <stop offset="30%" stop-color="#ffffff" stop-opacity="0.1"/>
      <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.2"/>
    </linearGradient>
  </defs>

  <rect width="400" height="500" fill="#09090b" rx="16"/>

  <g transform="translate(200, 260)">
    <!-- Glowing Viscous Liquid -->
    <path d="M -75,60 C -75,125 75,125 75,60 C 50,75 -50,75 -75,60 Z" fill="url(#liquidGrad)"/>
    <ellipse cx="0" cy="60" rx="72" ry="12" fill="#7dd3fc" fill-opacity="0.8"/>

    <!-- Glass Flask Outer Wall -->
    <path d="M -22,-120 L 22,-120 L 22,-60 C 85,0 95,120 0,140 C -95,120 -85,0 -22,-60 Z" fill="none" stroke="#e0f2fe" stroke-width="5" stroke-opacity="0.85"/>

    <!-- Glass Specular Highlight (Rim Glint) -->
    <path d="M -18,-115 L -18,-60 C -75,0 -75,90 -25,125" fill="none" stroke="#ffffff" stroke-width="4" stroke-linecap="round" stroke-opacity="0.7"/>

    <!-- Cork Stopper -->
    <polygon points="-18,-120 18,-120 14,-155 -14,-155" fill="#a16207" stroke="#713f12" stroke-width="2"/>
    <path d="M -16,-155 C 0,-165 0,-165 16,-155" fill="#dc2626" stroke="#991b1b" stroke-width="4"/>
  </g>

  <text x="200" y="440" text-anchor="middle" fill="#38bdf8" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="bold">ALCHEMICAL CRYSTAL FLASK (VITAL HP: 6)</text>
  <text x="200" y="460" text-anchor="middle" fill="#a1a1aa" font-family="'JetBrains Mono', monospace" font-size="10">Fúkané Sklo | Meniskus Kvapaliny | Vosková Pečať</text>
</svg>"""

    def _generate_astrolabe_svg(self, diam: float) -> str:
        """Emits high-detail SVG for mechanical brass astrolabe."""
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 500" width="100%" height="100%">
  <defs>
    <radialGradient id="brassDisc" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="#fde68a"/>
      <stop offset="70%" stop-color="#d97706"/>
      <stop offset="100%" stop-color="#78350f"/>
    </radialGradient>
  </defs>

  <rect width="400" height="500" fill="#09090b" rx="16"/>

  <g transform="translate(200, 240)">
    <!-- Top Suspension Ring -->
    <circle cx="0" cy="-160" r="24" fill="none" stroke="#d97706" stroke-width="6"/>

    <!-- Main Brass Chassis -->
    <circle cx="0" cy="0" r="140" fill="url(#brassDisc)" stroke="#78350f" stroke-width="4"/>
    <circle cx="0" cy="0" r="126" fill="none" stroke="#451a03" stroke-width="2" stroke-dasharray="2,6"/>
    <circle cx="0" cy="0" r="105" fill="none" stroke="#78350f" stroke-width="1.5"/>

    <!-- Cross Hair Grids -->
    <line x1="-120" y1="0" x2="120" y2="0" stroke="#78350f" stroke-width="1.5"/>
    <line x1="0" y1="-120" x2="0" y2="120" stroke="#78350f" stroke-width="1.5"/>

    <!-- Rotating Indicator Arm (Alidade) -->
    <polygon points="-8,-135 8,-135 3,135 -3,135" fill="#475569" stroke="#0f172a" stroke-width="1.5" transform="rotate(38)"/>
    <circle cx="0" cy="0" r="10" fill="#f59e0b" stroke="#451a03" stroke-width="2"/>
  </g>

  <text x="200" y="440" text-anchor="middle" fill="#f59e0b" font-family="'JetBrains Mono', monospace" font-size="12" font-weight="bold">MECHANICAL BRASS ASTROLABE</text>
  <text x="200" y="460" text-anchor="middle" fill="#a1a1aa" font-family="'JetBrains Mono', monospace" font-size="10">Kartáčovaná Mosadz | Kalibrácia 360° | Zameriavacia Alhidáda</text>
</svg>"""

    def _generate_character_svg(self, arch: CharacterArchetype, height_cm: float) -> str:
        """Emits SVG portrait of the skeuomorphic hero."""
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 550" width="100%" height="100%">
  <rect width="400" height="550" fill="#09090b" rx="16"/>
  <rect x="20" y="20" width="360" height="510" fill="#121218" stroke="#ffffff" stroke-opacity="0.08" rx="12"/>

  <g transform="translate(200, 240)">
    <!-- Cloak / Shoulders -->
    <path d="M -75,80 C -60,-20 60,-20 75,80 Z" fill="#1e293b" stroke="#0f172a" stroke-width="2"/>
    <!-- Tunic / Cuirass -->
    <rect x="-35" y="40" width="70" height="120" rx="8" fill="#78350f" stroke="#451a03" stroke-width="2"/>
    <!-- Head / Face -->
    <ellipse cx="0" cy="-20" rx="28" ry="36" fill="#fcd34d" stroke="#b45309" stroke-width="1.5"/>
    <!-- Cowl / Helmet -->
    <path d="M -32,-35 C -15,-65 15,-65 32,-35 C 36,-10 28,15 0,25 C -28,15 -36,-10 -32,-35 Z" fill="#334155" stroke="#1e293b" stroke-width="2"/>
    <!-- Belt with Buckle -->
    <rect x="-36" y="115" width="72" height="14" fill="#451a03"/>
    <rect x="-10" y="112" width="20" height="20" fill="none" stroke="#f59e0b" stroke-width="3"/>
  </g>

  <text x="200" y="470" text-anchor="middle" fill="#f472b6" font-family="'JetBrains Mono', monospace" font-size="13" font-weight="bold">{arch.name.replace('_', ' ')}</text>
  <text x="200" y="495" text-anchor="middle" fill="#a1a1aa" font-family="'JetBrains Mono', monospace" font-size="11">Výška: {height_cm:.1f} cm | Kánon: 8 Hláv (Vitruvius) | VITAL HP: 6/6</text>
</svg>"""

    def _generate_room_svg(self, w_m: float, l_m: float, h_m: float) -> str:
        """Emits architectural elevation view of the skeuomorphic room."""
        return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400" width="100%" height="100%">
  <rect width="600" height="400" fill="#09090b" rx="16"/>
  <g transform="translate(50, 40)">
    <!-- Room Frame -->
    <rect x="0" y="0" width="500" height="280" fill="#18181b" stroke="#ffffff" stroke-opacity="0.1" stroke-width="2"/>
    <!-- Timber Ceiling Rafters -->
    <line x1="0" y1="35" x2="500" y2="35" stroke="#451a03" stroke-width="8"/>
    <line x1="100" y1="0" x2="100" y2="35" stroke="#451a03" stroke-width="8"/>
    <line x1="250" y1="0" x2="250" y2="35" stroke="#451a03" stroke-width="8"/>
    <line x1="400" y1="0" x2="400" y2="35" stroke="#451a03" stroke-width="8"/>

    <!-- Stone Fireplace (Center Hearth) -->
    <path d="M 200,280 L 200,100 L 300,100 L 300,280 Z" fill="#78716c" stroke="#44403c" stroke-width="2"/>
    <rect x="220" y="180" width="60" height="100" rx="20" fill="#1c1917"/>
    <circle cx="250" cy="245" r="16" fill="#f97316"/>

    <!-- Leaded Window -->
    <rect x="40" y="80" width="80" height="130" rx="4" fill="#0284c7" fill-opacity="0.3" stroke="#e0f2fe" stroke-width="2"/>

    <!-- Herringbone Parquet Floor -->
    <rect x="0" y="270" width="500" height="10" fill="#78350f"/>
  </g>
  <text x="300" y="360" text-anchor="middle" fill="#d6d3d1" font-family="'JetBrains Mono', monospace" font-size="12">ARCHITEKTONICKÝ INTERIÉR // {w_m}m × {l_m}m × {h_m}m (ZLATÝ REZ)</text>
</svg>"""

    # =========================================================================
    # MULTI-SUBSTRATE TRANSPILATION: GODOT 4 .TSCN
    # =========================================================================

    def export_item_to_godot_tscn(self, item: SkeuomorphicItem) -> str:
        """Emits a complete Godot 4 Forward+ .tscn scene for the skeuomorphic item."""
        lines = [
            '[gd_scene load_steps=5 format=3 uid="uid://krystal_skeuo_item"]',
            '',
            '# =====================================================================',
            '# KRYSTAL-STACK SKEUOMORPHIC ITEM SCENE (GODOT 4 FORWARD+)',
            f'# Item: {item.name_sk} ({item.item_type.value})',
            f'# Seed: {item.seed} | Golden Ratio Score: {item.golden_ratio_fit_score}',
            f'# Vital Max HP: {item.vital_hp}',
            '# =====================================================================',
            '',
            '[sub_resource type="StandardMaterial3D" id="Mat_Damascus"]',
            'albedo_color = Color(0.80, 0.84, 0.88, 1.0)',
            'metallic = 0.94',
            'roughness = 0.22',
            '',
            '[sub_resource type="StandardMaterial3D" id="Mat_Brass"]',
            'albedo_color = Color(0.96, 0.62, 0.04, 1.0)',
            'metallic = 0.88',
            'roughness = 0.30',
            '',
            '[sub_resource type="StandardMaterial3D" id="Mat_Walnut"]',
            'albedo_color = Color(0.29, 0.18, 0.09, 1.0)',
            'metallic = 0.02',
            'roughness = 0.38',
            '',
            f'[node name="{item.item_id}" type="Node3D"]',
            f'metadata/vital_hp = {item.vital_hp}',
            f'metadata/item_type = "{item.item_type.value}"',
            f'metadata/golden_fit = {item.golden_ratio_fit_score}',
            '',
        ]

        for idx, comp in enumerate(item.components, start=1):
            w = comp.width_mm / 1000.0
            h = comp.height_mm / 1000.0
            d = comp.depth_mm / 1000.0
            node_name = f"Component_{idx}_{comp.name.replace(' ', '_')}"

            lines.append(f'[node name="{node_name}" type="CSGBox3D" parent="."]')
            lines.append(f'size = Vector3({w:.4f}, {h:.4f}, {d:.4f})')
            lines.append(f'metadata/substrate = "{comp.substrate_id}"')
            lines.append(f'metadata/fastener = "{comp.fastener_type}"')
            lines.append('')

        return "\n".join(lines)

    # =========================================================================
    # MULTI-SUBSTRATE TRANSPILATION: JAVA 21 RECORDS
    # =========================================================================

    def export_item_to_java_records(self, item: SkeuomorphicItem) -> str:
        """Emits modern Java 21 record declarations for the skeuomorphic entity."""
        return f"""// ============================================================================
// KRYSTAL-STACK SKEUOMORPHIC ITEM RECORD (JAVA 21 PATTERN MATCHING)
// Item: {item.name_sk} | Seed: {item.seed}L | Vital HP: {item.vital_hp}
// ============================================================================
package com.krystal.stack.skeuomorphic;

import java.util.List;
import java.util.Objects;

public final class SkeuomorphicPipeline {{

    public static final int VITAL_MAX_HP = 6;
    public static final double GOLDEN_RATIO = 1.61803398875;

    public record PhysicalComponent(
        String name,
        String substrateId,
        double widthMm,
        double heightMm,
        double depthMm,
        String fastenerType
    ) {{}}

    public record SkeuomorphicItemRecord(
        String itemId,
        String itemType,
        String nameSk,
        long seed,
        double overallWidthMm,
        double overallHeightMm,
        double goldenRatioScore,
        List<PhysicalComponent> components,
        int vitalHp
    ) {{
        public SkeuomorphicItemRecord {{
            Objects.requireNonNull(itemId);
            if (vitalHp > VITAL_MAX_HP) {{
                vitalHp = VITAL_MAX_HP; // Enforce invariant
            }}
        }}
    }}

    /**
     * Pattern matching classification of tactile affordance.
     */
    public static String classifyTactileRole(PhysicalComponent comp) {{
        return switch (comp.substrateId()) {{
            case "FORGED_DAMASCUS_STEEL" -> "High-Carbon Cutting Edge with Fuller Fluting";
            case "AGED_BOHEMIAN_WALNUT"  -> "Ergonomic Hand Grip with Finger Flutes";
            case "TARNISHED_CHAMPAGNE_BRASS" -> "Counterweight Pommel & Structural Guard";
            case "SADDLE_STITCHED_LEATHER"   -> "Flexible Protective Scabbard/Sheath";
            default -> "Auxiliary Mechanical Hardware";
        }};
    }}
}}
"""

    # =========================================================================
    # MULTI-SUBSTRATE TRANSPILATION: JANET DSL
    # =========================================================================

    def export_item_to_janet_dsl(self, item: SkeuomorphicItem) -> str:
        """Emits immutable Janet tables representing the skeuomorphic item AST."""
        lines = [
            "# ======================================================================",
            "# KRYSTAL-STACK SKEUOMORPHIC ITEM (JANET DSL)",
            f"# Item: {item.name_sk} | Seed: {item.seed}",
            f"# Golden Ratio Score: {item.golden_ratio_fit_score}",
            "# Strict Invariant: VITAL-MAX-HP = 6",
            "# ======================================================================",
            "",
            "(def VITAL-MAX-HP 6)",
            "(def GOLDEN-RATIO 1.61803398875)",
            "",
            "(def SKEUOMORPHIC-ITEM",
            f'  @{{:id "{item.item_id}"',
            f'    :type "{item.item_type.value}"',
            f'    :name "{item.name_sk}"',
            f'    :seed {item.seed}',
            f'    :dimensions-mm [{item.overall_width_mm:.1f} {item.overall_height_mm:.1f} {item.overall_depth_mm:.1f}]',
            f'    :golden-score {item.golden_ratio_fit_score}',
            f'    :vital-hp {item.vital_hp}',
            "    :components",
            "    [",
        ]
        for comp in item.components:
            lines.append(
                f'     @{{:name "{comp.name}" :substrate "{comp.substrate_id}" '
                f':size-mm [{comp.width_mm:.1f} {comp.height_mm:.1f} {comp.depth_mm:.1f}] '
                f':fastener "{comp.fastener_type}"}}'
            )
        lines.append("    ]})")

        return "\n".join(lines)


GLOBAL_SKEUOMORPHIC_ENGINE = SkeuomorphicProceduralEngine()
