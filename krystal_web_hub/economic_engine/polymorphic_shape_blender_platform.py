#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: POLYMORPHIC SHAPE BLENDER PLATFORM & ECONOMIC PLANNING ENGINE
==============================================================================
Module: krystal_web_hub/economic_engine/polymorphic_shape_blender_platform.py
Description:
    Unifies parametric 3D shape libraries, a Template Factory, a Command Shape
    Platform, a Multilingual Lookup Registry (SK, EN, CZ, DE), and an Economic
    Planning Observer with Cartesian & Blender coordinate system definitions:
      1. Multilingual Lookup Mode Management:
         - Dynamic runtime language mode switching (SK, EN, CZ, DE).
         - Distributed dictionary covering archetypes, modifiers, economic
           indicators, Cartesian coordinates, octants, and UI descriptors.
      2. Descriptive Mesh & Blender Cartesian Coordinate System:
         - Standard Blender 4.x Cartesian Right-Handed convention (Z-Up, Y-Forward, X-Right).
         - Precise geometric centroid, AABB bounding box, and extents.
         - 8-Octant vertex spatial distribution analysis with percentages.
         - Bidirectional coordinate transformation matrices (Blender Z-Up,
           OpenGL/WebGL Y-Up, Unreal Engine Z-Up LH, Godot Y-Up).
         - Centroid Spherical (r, θ, φ) and Cylindrical (ρ, φ, z) mappings.
      3. Polymorphic Shape Library & Template Factory:
         - 6 Canonical Parametric Archetypes (Tactical Obelisk, Citadel Spire,
           Biophilic Organic, Kinetic Mechanism, Ethereal Crystal, Tactical Drone).
         - Procedural modifier stack (Bevel, Solidify, Array, Mirror, Displace,
           Subsurf, SimpleDeform).
      4. Blender 4.x Python Automation Engine:
         - Generates standalone, executable Blender scripts (`bpy`) with metric units,
           reference Cartesian coordinate gizmos, and localized commentary.
         - Wavefront .OBJ 3D exporter with normals and face topology.
      5. Economic Planning & Shape Observation Engine:
         - Real-time costing (Crystal, Mana, Gold, Build Time, Upkeep).
         - Strict structural integrity bounding: VITAL_MAX_HP == 6.

Core Invariants:
    - VITAL_MAX_HP == 6
    - GOLDEN_RATIO == 1.61803398875

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import math
import time
import uuid
import json
import threading
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875


# ==============================================================================
# ENUMS & CONSTANTS
# ==============================================================================

class LanguageMode(str, Enum):
    SK = "SK"  # Slovenčina (Native Default)
    EN = "EN"  # English
    CZ = "CZ"  # Čeština
    DE = "DE"  # Deutsch


class CoordinateSystemType(str, Enum):
    BLENDER_Z_UP       = "BLENDER_Z_UP"        # Blender 4.x: Right-Handed (+X Right, +Y Forward, +Z Up)
    OPENGL_WEBGL_Y_UP  = "OPENGL_WEBGL_Y_UP"   # WebGL/OpenGL: Right-Handed (+X Right, +Y Up, -Z Forward)
    UNREAL_Z_UP_LH     = "UNREAL_Z_UP_LH"      # Unreal Engine: Left-Handed (+X Forward, +Y Right, +Z Up)
    GODOT_Y_UP         = "GODOT_Y_UP"          # Godot 3D: Right-Handed (+X Right, +Y Up, +Z Back)


class ShapeArchetype(str, Enum):
    TACTICAL_OBELISK   = "TACTICAL_OBELISK"     # Monolithic resonant obelisk
    CITADEL_SPIRE      = "CITADEL_SPIRE"        # Cyber-gothic defensive spire with crenellations
    BIOPHILIC_ORGANIC  = "BIOPHILIC_ORGANIC"    # Golden spiral phyllotaxis sanctuary
    KINETIC_MECHANISM  = "KINETIC_MECHANISM"    # Modular kinetic gear with precise teeth
    ETHEREAL_CRYSTAL   = "ETHEREAL_CRYSTAL"     # Floating aetheric octahedron cluster
    TACTICAL_DRONE     = "TACTICAL_DRONE"       # Aerodynamic quad-rotor drone chassis


class ModifierType(str, Enum):
    BEVEL         = "BEVEL"
    SOLIDIFY      = "SOLIDIFY"
    ARRAY         = "ARRAY"
    MIRROR        = "MIRROR"
    DISPLACE      = "DISPLACE"
    SIMPLE_DEFORM = "SIMPLE_DEFORM"
    SUBSURF       = "SUBSURF"


class ShapeCommandType(str, Enum):
    CREATE_SHAPE           = "CREATE_SHAPE"
    MORPH_PARAMETERS       = "MORPH_PARAMETERS"
    APPLY_MODIFIER         = "APPLY_MODIFIER"
    CALCULATE_PLAN         = "CALCULATE_PLAN"
    EXPORT_BLENDER_SCRIPT  = "EXPORT_BLENDER_SCRIPT"
    EXPORT_OBJ_GEOMETRY    = "EXPORT_OBJ_GEOMETRY"
    SET_LANGUAGE_MODE      = "SET_LANGUAGE_MODE"
    CONVERT_COORDINATES    = "CONVERT_COORDINATES"


# ==============================================================================
# MULTILINGUAL LOOKUP REGISTRY
# ==============================================================================

class MultilingualLookupRegistry:
    """
    Central registry providing localized terms, archetype narratives,
    economic indicators, and coordinate system descriptions across SK, EN, CZ, DE.
    """

    _DICTIONARY: Dict[str, Dict[LanguageMode, str]] = {
        # Platform & Branding
        "platform_title": {
            LanguageMode.SK: "Krystal-Stack Polymorfná Platforma Tvarov & Blender Knižnica",
            LanguageMode.EN: "Krystal-Stack Polymorphic Shape Platform & Blender Library",
            LanguageMode.CZ: "Krystal-Stack Polymorfní Platforma Tvarů & Blender Knihovna",
            LanguageMode.DE: "Krystal-Stack Polymorphe Formen-Plattform & Blender-Bibliothek"
        },
        "vital_hp_rule": {
            LanguageMode.SK: "VITAL Pravidlo: Maximálne 6 Štrukturálnych HP",
            LanguageMode.EN: "VITAL Rule: Maximum 6 Structural HP",
            LanguageMode.CZ: "VITAL Pravidlo: Maximálně 6 Strukturálních HP",
            LanguageMode.DE: "VITAL-Regel: Maximal 6 Strukturelle Trefferpunkte"
        },
        # Archetypes Names
        "archetype_TACTICAL_OBELISK_name": {
            LanguageMode.SK: "Kryštálový Taktický Obelisk",
            LanguageMode.EN: "Tactical Crystal Obelisk",
            LanguageMode.CZ: "Krystalový Taktický Obelisk",
            LanguageMode.DE: "Taktischer Kristall-Obelisk"
        },
        "archetype_TACTICAL_OBELISK_desc": {
            LanguageMode.SK: "Monolitický rezonátor s fasetovanými hranami a skoseným bevelom.",
            LanguageMode.EN: "Monolithic resonator with faceted edges and bevel modifier.",
            LanguageMode.CZ: "Monolitický rezonátor s fasetovanými hranami a zkoseným bevelem.",
            LanguageMode.DE: "Monolithischer Resonator mit facettierten Kanten und Bevel-Modifikator."
        },
        "archetype_CITADEL_SPIRE_name": {
            LanguageMode.SK: "Kybernetická Citadela - Veža",
            LanguageMode.EN: "Cybernetic Citadel Spire",
            LanguageMode.CZ: "Kybernetická Citadela - Věž",
            LanguageMode.DE: "Kybernetische Zitadelle - Turm"
        },
        "archetype_CITADEL_SPIRE_desc": {
            LanguageMode.SK: "Viacposchodová pevnostná veža s cimburím a zrkadlenou symetriou.",
            LanguageMode.EN: "Multi-tiered stronghold tower with crenellations and mirror symmetry.",
            LanguageMode.CZ: "Vícepatrová pevnostní věž s cimbuřím a zrcadlenou symetrií.",
            LanguageMode.DE: "Mehrstöckiger Festungsturm mit Zinnen und Spiegelsymmetrie."
        },
        "archetype_BIOPHILIC_ORGANIC_name": {
            LanguageMode.SK: "Druidská Špirálová Svätyňa",
            LanguageMode.EN: "Biophilic Spiral Sanctuary",
            LanguageMode.CZ: "Druidská Spirálová Svatyně",
            LanguageMode.DE: "Biophiles Spiralförmiges Heiligtum"
        },
        "archetype_BIOPHILIC_ORGANIC_desc": {
            LanguageMode.SK: "Organická špirála inšpirovaná fylotaxiou a zlatým rezom.",
            LanguageMode.EN: "Organic logarithmic spiral inspired by phyllotaxis and golden ratio.",
            LanguageMode.CZ: "Organická spirála inspirovaná fylotaxií a zlatým řezem.",
            LanguageMode.DE: "Organische Spirale inspiriert von Phyllotaxis und Goldenem Schnitt."
        },
        "archetype_KINETIC_MECHANISM_name": {
            LanguageMode.SK: "Modulárne Ozubené Koleso",
            LanguageMode.EN: "Modular Kinetic Gear",
            LanguageMode.CZ: "Modulární Ozubené Kolo",
            LanguageMode.DE: "Modulares Kinetisches Zahnrad"
        },
        "archetype_KINETIC_MECHANISM_desc": {
            LanguageMode.SK: "Kinetický mechanický prvok s presnými zubami pre prevodové sústavy.",
            LanguageMode.EN: "Kinetic mechanical element with precision teeth for drive systems.",
            LanguageMode.CZ: "Kinetický mechanický prvek s přesnými zuby pro převodové soustavy.",
            LanguageMode.DE: "Kinetisches mechanisches Element mit Präzisionszähnen für Antriebssysteme."
        },
        "archetype_ETHEREAL_CRYSTAL_name": {
            LanguageMode.SK: "Oktaedrický Klastr Aetheru",
            LanguageMode.EN: "Aetheric Octahedron Cluster",
            LanguageMode.CZ: "Oktaedrický Klastr Aetheru",
            LanguageMode.DE: "Ätherischer Oktaeder-Cluster"
        },
        "archetype_ETHEREAL_CRYSTAL_desc": {
            LanguageMode.SK: "Vznášajúci sa kryštálový útvar s disperziou svetla a satelitnými nodmi.",
            LanguageMode.EN: "Levitating crystalline structure with light dispersion and satellite nodes.",
            LanguageMode.CZ: "Vznášející se krystalový útvar s disperzí světla a satelitními uzly.",
            LanguageMode.DE: "Schwebendes Kristallgebilde mit Lichtstreuung und Satellitenknoten."
        },
        "archetype_TACTICAL_DRONE_name": {
            LanguageMode.SK: "Prieskumný Taktický Dron",
            LanguageMode.EN: "Tactical Recon Drone",
            LanguageMode.CZ: "Průzkumný Taktický Dron",
            LanguageMode.DE: "Taktische Aufklärungsdrohne"
        },
        "archetype_TACTICAL_DRONE_desc": {
            LanguageMode.SK: "Kvadrokoptérový rám s integrovaným senzorovým krytom a symetrickými ramenami.",
            LanguageMode.EN: "Quadcopter chassis with integrated sensor hull and symmetrical arms.",
            LanguageMode.CZ: "Kvadrokoptérový rám s integrovaným senzorovým krytem a symetrickými rameny.",
            LanguageMode.DE: "Quadrocopter-Rahmen mit integrierter Sensorkuppel und symmetrischen Auslegern."
        },

        # Economic Metrics
        "crystal_cost": {
            LanguageMode.SK: "Náklad Kryštálov",
            LanguageMode.EN: "Crystal Cost",
            LanguageMode.CZ: "Náklad Krystalů",
            LanguageMode.DE: "Kristallkosten"
        },
        "mana_cost": {
            LanguageMode.SK: "Mana Náklad",
            LanguageMode.EN: "Mana Cost",
            LanguageMode.CZ: "Mana Náklad",
            LanguageMode.DE: "Manakosten"
        },
        "gold_cost": {
            LanguageMode.SK: "Zlato",
            LanguageMode.EN: "Gold Cost",
            LanguageMode.CZ: "Zlato",
            LanguageMode.DE: "Goldkosten"
        },
        "build_time": {
            LanguageMode.SK: "Výrobný Čas",
            LanguageMode.EN: "Build Time",
            LanguageMode.CZ: "Výrobní Čas",
            LanguageMode.DE: "Bauzeit"
        },
        "structural_hp": {
            LanguageMode.SK: "Štrukturálna Odolnosť (HP)",
            LanguageMode.EN: "Structural HP",
            LanguageMode.CZ: "Strukturální Odolnost (HP)",
            LanguageMode.DE: "Strukturelle HP"
        },
        "complexity_score": {
            LanguageMode.SK: "Skóre Komplexity",
            LanguageMode.EN: "Complexity Score",
            LanguageMode.CZ: "Skóre Komplexity",
            LanguageMode.DE: "Komplexitätswert"
        },
        "upkeep_mana": {
            LanguageMode.SK: "Údržbová Mana / Cyklus",
            LanguageMode.EN: "Mana Upkeep / Cycle",
            LanguageMode.CZ: "Údržbová Mana / Cyklus",
            LanguageMode.DE: "Mana-Unterhalt / Zyklus"
        },

        # Coordinate System & Cartesian Terms
        "coord_system_name": {
            LanguageMode.SK: "Súradnicový Systém",
            LanguageMode.EN: "Coordinate System",
            LanguageMode.CZ: "Souřadnicový Systém",
            LanguageMode.DE: "Koordinatensystem"
        },
        "blender_cartesian_desc": {
            LanguageMode.SK: "Pravotočivá Karteziánska Sústava Blender (Z-Nahor, Y-Dopredu, X-Doprava, Metrické)",
            LanguageMode.EN: "Blender Right-Handed Cartesian System (Z-Up, Y-Forward, X-Right, Metric)",
            LanguageMode.CZ: "Pravotočivá Kartézská Soustava Blender (Z-Nahoru, Y-Dopředu, X-Doprava, Metrické)",
            LanguageMode.DE: "Blender Rechtshändiges Kartesisches System (Z-Oben, Y-Vorwärts, X-Rechts, Metrisch)"
        },
        "origin": {
            LanguageMode.SK: "Počiatok Sústavy (0,0,0)",
            LanguageMode.EN: "Coordinate Origin (0,0,0)",
            LanguageMode.CZ: "Počátek Soustavy (0,0,0)",
            LanguageMode.DE: "Koordinatenursprung (0,0,0)"
        },
        "centroid": {
            LanguageMode.SK: "Geometrické Ťažisko Siete",
            LanguageMode.EN: "Geometric Mesh Centroid",
            LanguageMode.CZ: "Geometrické Těžiště Sítě",
            LanguageMode.DE: "Geometrischer Netzschwerpunkt"
        },
        "bounding_box": {
            LanguageMode.SK: "Ohraničujúci Kváder (AABB)",
            LanguageMode.EN: "Axis-Aligned Bounding Box (AABB)",
            LanguageMode.CZ: "Ohraničující Kvádr (AABB)",
            LanguageMode.DE: "Achsenausgerichtete Bounding-Box (AABB)"
        },
        "dimensions": {
            LanguageMode.SK: "Rozmery (Šírka × Hĺbka × Výška)",
            LanguageMode.EN: "Dimensions (Width × Depth × Height)",
            LanguageMode.CZ: "Rozměry (Šířka × Hloubka × Výška)",
            LanguageMode.DE: "Abmessungen (Breite × Tiefe × Höhe)"
        },
        "axis_x": {
            LanguageMode.SK: "Os X (Abscisa / Doprava / Červená)",
            LanguageMode.EN: "X-Axis (Abscissa / Right / Red)",
            LanguageMode.CZ: "Osa X (Abscisa / Doprava / Červená)",
            LanguageMode.DE: "X-Achse (Abszisse / Rechts / Rot)"
        },
        "axis_y": {
            LanguageMode.SK: "Os Y (Ordináta / Dopredu / Zelená)",
            LanguageMode.EN: "Y-Axis (Ordinate / Forward / Green)",
            LanguageMode.CZ: "Osa Y (Ordináta / Dopředu / Zelená)",
            LanguageMode.DE: "Y-Achse (Ordinate / Vorwärts / Grün)"
        },
        "axis_z": {
            LanguageMode.SK: "Os Z (Aplikáta / Nahor / Modrá)",
            LanguageMode.EN: "Z-Axis (Applicate / Up / Blue)",
            LanguageMode.CZ: "Osa Z (Aplikáta / Nahoru / Modrá)",
            LanguageMode.DE: "Z-Achse (Applikate / Oben / Blau)"
        },
        "octants_distribution": {
            LanguageMode.SK: "Distribúcia Vrcholov v 8 Oktantoch",
            LanguageMode.EN: "Vertex Distribution Across 8 Octants",
            LanguageMode.CZ: "Distribuce Vrcholů v 8 Oktantech",
            LanguageMode.DE: "Knotenverteilung über 8 Oktanten"
        },
        "spherical_coords": {
            LanguageMode.SK: "Sférické Súradnice Ťažiska (r, θ, φ)",
            LanguageMode.EN: "Spherical Coordinates of Centroid (r, θ, φ)",
            LanguageMode.CZ: "Sférické Souřadnice Těžiště (r, θ, φ)",
            LanguageMode.DE: "Kugelkoordinaten des Schwerpunkts (r, θ, φ)"
        },
        "cylindrical_coords": {
            LanguageMode.SK: "Cylindrické Súradnice Ťažiska (ρ, φ, z)",
            LanguageMode.EN: "Cylindrical Coordinates of Centroid (ρ, φ, z)",
            LanguageMode.CZ: "Cylindrické Souřadnice Těžiště (ρ, φ, z)",
            LanguageMode.DE: "Zylinderkoordinaten des Schwerpunkts (ρ, φ, z)"
        },
        "transform_matrix": {
            LanguageMode.SK: "Transformačná Matica 4×4 z Blenderu",
            LanguageMode.EN: "4×4 Transformation Matrix from Blender",
            LanguageMode.CZ: "Transformační Matice 4×4 z Blenderu",
            LanguageMode.DE: "4×4 Transformationsmatrix aus Blender"
        },

        # Octant Names
        "octant_1": {
            LanguageMode.SK: "O1 (+,+,+) Horný Predný Pravý",
            LanguageMode.EN: "O1 (+,+,+) Upper Front Right",
            LanguageMode.CZ: "O1 (+,+,+) Horní Přední Pravý",
            LanguageMode.DE: "O1 (+,+,+) Oberer Vorderer Rechter"
        },
        "octant_2": {
            LanguageMode.SK: "O2 (-,+,+) Horný Predný Ľavý",
            LanguageMode.EN: "O2 (-,+,+) Upper Front Left",
            LanguageMode.CZ: "O2 (-,+,+) Horní Přední Levý",
            LanguageMode.DE: "O2 (-,+,+) Oberer Vorderer Linker"
        },
        "octant_3": {
            LanguageMode.SK: "O3 (-,-,+) Horný Zadný Ľavý",
            LanguageMode.EN: "O3 (-,-,+) Upper Rear Left",
            LanguageMode.CZ: "O3 (-,-,+) Horní Zadní Levý",
            LanguageMode.DE: "O3 (-,-,+) Oberer Hinterer Linker"
        },
        "octant_4": {
            LanguageMode.SK: "O4 (+,-,+) Horný Zadný Pravý",
            LanguageMode.EN: "O4 (+,-,+) Upper Rear Right",
            LanguageMode.CZ: "O4 (+,-,+) Horní Zadní Pravý",
            LanguageMode.DE: "O4 (+,-,+) Oberer Hinterer Rechter"
        },
        "octant_5": {
            LanguageMode.SK: "O5 (+,+,-) Dolný Predný Pravý",
            LanguageMode.EN: "O5 (+,+,-) Lower Front Right",
            LanguageMode.CZ: "O5 (+,+,-) Dolní Přední Pravý",
            LanguageMode.DE: "O5 (+,+,-) Unterer Vorderer Rechter"
        },
        "octant_6": {
            LanguageMode.SK: "O6 (-,+,-) Dolný Predný Ľavý",
            LanguageMode.EN: "O6 (-,+,-) Lower Front Left",
            LanguageMode.CZ: "O6 (-,+,-) Dolní Přední Levý",
            LanguageMode.DE: "O6 (-,+,-) Unterer Vorderer Linker"
        },
        "octant_7": {
            LanguageMode.SK: "O7 (-,-,-) Dolný Zadný Ľavý",
            LanguageMode.EN: "O7 (-,-,-) Lower Rear Left",
            LanguageMode.CZ: "O7 (-,-,-) Dolní Zadní Levý",
            LanguageMode.DE: "O7 (-,-,-) Unterer Hinterer Linker"
        },
        "octant_8": {
            LanguageMode.SK: "O8 (+,-,-) Dolný Zadný Pravý",
            LanguageMode.EN: "O8 (+,-,-) Lower Rear Right",
            LanguageMode.CZ: "O8 (+,-,-) Dolní Zadní Pravý",
            LanguageMode.DE: "O8 (+,-,-) Unterer Hinterer Rechter"
        },

        # UI & Buttons
        "btn_create_shape": {
            LanguageMode.SK: "Vytvoriť Tvar",
            LanguageMode.EN: "Create Shape",
            LanguageMode.CZ: "Vytvořit Tvar",
            LanguageMode.DE: "Form Erstellen"
        },
        "btn_copy_script": {
            LanguageMode.SK: "Kopírovať Blender Skript",
            LanguageMode.EN: "Copy Blender Script",
            LanguageMode.CZ: "Kopírovat Blender Skript",
            LanguageMode.DE: "Blender-Skript Kopieren"
        },
        "btn_download_obj": {
            LanguageMode.SK: "Stiahnuť .OBJ Sieť",
            LanguageMode.EN: "Download .OBJ Mesh",
            LanguageMode.CZ: "Stáhnout .OBJ Síť",
            LanguageMode.DE: ".OBJ-Netz Herunterladen"
        },
        "btn_convert_coord": {
            LanguageMode.SK: "Prepočítať Sústavu",
            LanguageMode.EN: "Convert Coordinates",
            LanguageMode.CZ: "Přepočítat Soustavu",
            LanguageMode.DE: "Koordinaten Umrechnen"
        }
    }

    @classmethod
    def get_term(cls, key: str, lang: LanguageMode = LanguageMode.SK) -> str:
        entry = cls._DICTIONARY.get(key)
        if not entry:
            return key
        return entry.get(lang, entry.get(LanguageMode.EN, key))

    @classmethod
    def get_archetype_info(cls, archetype: ShapeArchetype, lang: LanguageMode = LanguageMode.SK) -> Dict[str, str]:
        name_key = f"archetype_{archetype.value}_name"
        desc_key = f"archetype_{archetype.value}_desc"
        return {
            "name": cls.get_term(name_key, lang),
            "description": cls.get_term(desc_key, lang)
        }

    @classmethod
    def get_all_translations(cls, lang: LanguageMode = LanguageMode.SK) -> Dict[str, str]:
        return {k: cls.get_term(k, lang) for k in cls._DICTIONARY.keys()}


# ==============================================================================
# CARTESIAN COORDINATE SYSTEM & DESCRIPTIVE MESH SPECIFICATIONS
# ==============================================================================

@dataclass
class CartesianCoordinateDescriptor:
    """
    Complete mathematical description of a 3D mesh in a defined coordinate system,
    originating from Blender 4.x Right-Handed Cartesian convention.
    """
    coordinate_system: CoordinateSystemType
    origin: Tuple[float, float, float]
    bounding_box: Dict[str, float]  # min_x, max_x, min_y, max_y, min_z, max_z
    dimensions: Dict[str, float]    # width, depth, height
    centroid: Tuple[float, float, float]
    octant_distribution: Dict[str, Dict[str, Any]]
    spherical_centroid: Dict[str, float]    # r, theta_deg, phi_deg
    cylindrical_centroid: Dict[str, float]  # rho, phi_deg, z
    transform_matrix_4x4: List[List[float]]
    handedness: str
    axes_definitions: Dict[str, Dict[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinate_system": self.coordinate_system.value,
            "origin": list(self.origin),
            "bounding_box": self.bounding_box,
            "dimensions": self.dimensions,
            "centroid": list(self.centroid),
            "octant_distribution": self.octant_distribution,
            "spherical_centroid": self.spherical_centroid,
            "cylindrical_centroid": self.cylindrical_centroid,
            "transform_matrix_4x4": self.transform_matrix_4x4,
            "handedness": self.handedness,
            "axes_definitions": self.axes_definitions
        }


class CoordinateSystemConverter:
    """
    Transforms vertices, bounding boxes, and centroids between Blender 4.x
    Cartesian system and other standard 3D systems (OpenGL/WebGL, Unreal, Godot).
    """

    # 4x4 Homogeneous transformation matrices from Blender Z-Up
    TRANSFORM_MATRICES: Dict[CoordinateSystemType, List[List[float]]] = {
        CoordinateSystemType.BLENDER_Z_UP: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        CoordinateSystemType.OPENGL_WEBGL_Y_UP: [
            [1.0, 0.0,  0.0, 0.0],
            [0.0, 0.0,  1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0,  0.0, 1.0]
        ],
        CoordinateSystemType.UNREAL_Z_UP_LH: [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        CoordinateSystemType.GODOT_Y_UP: [
            [1.0, 0.0,  0.0, 0.0],
            [0.0, 0.0,  1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0,  0.0, 1.0]
        ]
    }

    @classmethod
    def transform_vertex(
        cls,
        v: Tuple[float, float, float],
        target_system: CoordinateSystemType
    ) -> Tuple[float, float, float]:
        x, y, z = v
        if target_system == CoordinateSystemType.BLENDER_Z_UP:
            return (x, y, z)
        elif target_system in (CoordinateSystemType.OPENGL_WEBGL_Y_UP, CoordinateSystemType.GODOT_Y_UP):
            return (x, z, -y)
        elif target_system == CoordinateSystemType.UNREAL_Z_UP_LH:
            return (y, x, z)
        return (x, y, z)

    @classmethod
    def get_matrix_for_system(cls, system: CoordinateSystemType) -> List[List[float]]:
        return cls.TRANSFORM_MATRICES.get(system, cls.TRANSFORM_MATRICES[CoordinateSystemType.BLENDER_Z_UP])


class MeshGeometryDescriptor:
    """
    Computes exact vertex lists, topology, AABB bounds, centroids,
    and 8-octant spatial distributions for polymorphic archetypes.
    """

    @staticmethod
    def generate_canonical_vertices(
        archetype: ShapeArchetype,
        params: Dict[str, float]
    ) -> List[Tuple[float, float, float]]:
        """Generates representative 3D vertices in Blender Cartesian (Z-Up) coordinates."""
        verts: List[Tuple[float, float, float]] = []

        if archetype == ShapeArchetype.TACTICAL_OBELISK:
            h = params.get("height", 4.5)
            r = params.get("base_radius", 1.2)
            top_r = r * params.get("taper_ratio", 0.78) * 0.4
            segs = 8
            for i in range(segs):
                angle = (2 * math.pi / segs) * i
                verts.append((r * math.cos(angle), r * math.sin(angle), 0.0))
            for i in range(segs):
                angle = (2 * math.pi / segs) * i
                verts.append((top_r * math.cos(angle), top_r * math.sin(angle), h))
            verts.append((0.0, 0.0, h + r * 0.35))

        elif archetype == ShapeArchetype.CITADEL_SPIRE:
            h = params.get("height", 6.2)
            r = params.get("base_radius", 2.0)
            levels = int(params.get("levels", 4))
            for lv in range(levels + 1):
                cur_z = (h / levels) * lv
                cur_r = r * (1.0 - (lv / (levels + 1)) * 0.5)
                for i in range(8):
                    ang = (2 * math.pi / 8) * i
                    verts.append((cur_r * math.cos(ang), cur_r * math.sin(ang), cur_z))
            # Crenellations at top
            cren_cnt = int(params.get("crenellation_count", 8))
            for i in range(cren_cnt):
                ang = (2 * math.pi / cren_cnt) * i
                verts.append(((r * 0.55) * math.cos(ang), (r * 0.55) * math.sin(ang), h + 0.4))

        elif archetype == ShapeArchetype.BIOPHILIC_ORGANIC:
            rad = params.get("radius", 1.8)
            turns = params.get("spiral_turns", 2.5)
            steps = 28
            for i in range(steps):
                t = i / steps
                theta = t * turns * 2 * math.pi
                growth = (GOLDEN_RATIO ** t) * 0.6
                cur_r = rad * growth
                cur_z = t * rad * 2.2
                verts.append((cur_r * math.cos(theta), cur_r * math.sin(theta), cur_z))
                verts.append((cur_r * 0.7 * math.cos(theta), cur_r * 0.7 * math.sin(theta), cur_z + 0.1))

        elif archetype == ShapeArchetype.KINETIC_MECHANISM:
            r = params.get("radius", 2.2)
            th = params.get("thickness", 0.45)
            teeth = int(params.get("teeth_count", 16))
            inner_r = params.get("inner_hole_radius", 0.6)
            for z_val in (0.0, th):
                for i in range(teeth * 2):
                    ang = (math.pi / teeth) * i
                    is_tip = (i % 2 == 0)
                    cur_r = r if is_tip else (r * 0.82)
                    verts.append((cur_r * math.cos(ang), cur_r * math.sin(ang), z_val))
                # Inner bore hole ring
                for i in range(8):
                    ang = (2 * math.pi / 8) * i
                    verts.append((inner_r * math.cos(ang), inner_r * math.sin(ang), z_val))

        elif archetype == ShapeArchetype.ETHEREAL_CRYSTAL:
            sc = params.get("scale", 1.5)
            # Octahedron core
            verts.extend([
                (sc, 0.0, sc * 1.2),
                (-sc, 0.0, sc * 1.2),
                (0.0, sc, sc * 1.2),
                (0.0, -sc, sc * 1.2),
                (0.0, 0.0, sc * 1.2 + sc * GOLDEN_RATIO),
                (0.0, 0.0, sc * 1.2 - sc * GOLDEN_RATIO)
            ])
            # Satellites
            sat_cnt = int(params.get("satellite_count", 4))
            for i in range(sat_cnt):
                ang = (2 * math.pi / sat_cnt) * i
                dist = sc * 1.8
                verts.append((dist * math.cos(ang), dist * math.sin(ang), sc * 1.2 + 0.3 * math.sin(i)))

        else:
            # Tactical drone
            span = params.get("arm_span", 2.8)
            body_r = params.get("body_radius", 0.65)
            # Central chassis box
            verts.extend([
                (-body_r, -body_r, 0.2), (body_r, -body_r, 0.2),
                (body_r, body_r, 0.2), (-body_r, body_r, 0.2),
                (-body_r, -body_r, 0.7), (body_r, -body_r, 0.7),
                (body_r, body_r, 0.7), (-body_r, body_r, 0.7)
            ])
            # 4 rotor arm endpoints
            half_s = span * 0.5
            verts.extend([
                (half_s, half_s, 0.5),
                (-half_s, half_s, 0.5),
                (-half_s, -half_s, 0.5),
                (half_s, -half_s, 0.5)
            ])

        return verts

    @classmethod
    def build_coordinate_descriptor(
        cls,
        archetype: ShapeArchetype,
        params: Dict[str, float],
        system: CoordinateSystemType = CoordinateSystemType.BLENDER_Z_UP,
        lang: LanguageMode = LanguageMode.SK
    ) -> CartesianCoordinateDescriptor:
        """Constructs an exhaustive Cartesian coordinate descriptor."""
        base_verts = cls.generate_canonical_vertices(archetype, params)
        if not base_verts:
            base_verts = [(0.0, 0.0, 0.0)]

        # Transform vertices if another coordinate system was targeted
        verts = [CoordinateSystemConverter.transform_vertex(v, system) for v in base_verts]

        # 1. Bounding Box & Dimensions
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        zs = [v[2] for v in verts]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        dx = round(max_x - min_x, 4)
        dy = round(max_y - min_y, 4)
        dz = round(max_z - min_z, 4)

        bbox = {
            "min_x": round(min_x, 4), "max_x": round(max_x, 4),
            "min_y": round(min_y, 4), "max_y": round(max_y, 4),
            "min_z": round(min_z, 4), "max_z": round(max_z, 4)
        }
        dims = {"width_x": dx, "depth_y": dy, "height_z": dz}

        # 2. Geometric Centroid
        n = len(verts)
        cx = round(sum(xs) / n, 4)
        cy = round(sum(ys) / n, 4)
        cz = round(sum(zs) / n, 4)
        centroid = (cx, cy, cz)

        # 3. 8-Octant Spatial Distribution
        octant_counts = {f"octant_{i}": 0 for i in range(1, 9)}
        for x, y, z in verts:
            if x >= 0 and y >= 0 and z >= 0:
                octant_counts["octant_1"] += 1
            elif x < 0 and y >= 0 and z >= 0:
                octant_counts["octant_2"] += 1
            elif x < 0 and y < 0 and z >= 0:
                octant_counts["octant_3"] += 1
            elif x >= 0 and y < 0 and z >= 0:
                octant_counts["octant_4"] += 1
            elif x >= 0 and y >= 0 and z < 0:
                octant_counts["octant_5"] += 1
            elif x < 0 and y >= 0 and z < 0:
                octant_counts["octant_6"] += 1
            elif x < 0 and y < 0 and z < 0:
                octant_counts["octant_7"] += 1
            else:
                octant_counts["octant_8"] += 1

        octant_dist: Dict[str, Dict[str, Any]] = {}
        for i in range(1, 9):
            key = f"octant_{i}"
            cnt = octant_counts[key]
            pct = round((cnt / n) * 100.0, 1) if n > 0 else 0.0
            octant_dist[key] = {
                "octant_index": i,
                "label": MultilingualLookupRegistry.get_term(key, lang),
                "vertex_count": cnt,
                "percentage": pct
            }

        # 4. Spherical & Cylindrical Mapping of Centroid
        r_spherical = round(math.sqrt(cx**2 + cy**2 + cz**2), 4)
        theta_rad = math.acos(cz / r_spherical) if r_spherical > 1e-6 else 0.0
        phi_rad = math.atan2(cy, cx)

        spherical = {
            "radius_r": r_spherical,
            "theta_deg": round(math.degrees(theta_rad), 2),
            "phi_deg": round(math.degrees(phi_rad), 2)
        }

        rho_cyl = round(math.sqrt(cx**2 + cy**2), 4)
        cylindrical = {
            "radius_rho": rho_cyl,
            "phi_deg": round(math.degrees(phi_rad), 2),
            "height_z": cz
        }

        # 5. Handedness & Axis Definitions
        handedness = "LEFT_HANDED" if system == CoordinateSystemType.UNREAL_Z_UP_LH else "RIGHT_HANDED"
        axes_defs = {
            "X": {
                "vector": [1.0, 0.0, 0.0],
                "direction": "Right (+X)" if system != CoordinateSystemType.UNREAL_Z_UP_LH else "Forward (+X)",
                "color_hex": "#ef4444",
                "label": MultilingualLookupRegistry.get_term("axis_x", lang)
            },
            "Y": {
                "vector": [0.0, 1.0, 0.0],
                "direction": "Forward (+Y)" if system == CoordinateSystemType.BLENDER_Z_UP else "Up (+Y)",
                "color_hex": "#22c55e",
                "label": MultilingualLookupRegistry.get_term("axis_y", lang)
            },
            "Z": {
                "vector": [0.0, 0.0, 1.0],
                "direction": "Up (+Z)" if system in (CoordinateSystemType.BLENDER_Z_UP, CoordinateSystemType.UNREAL_Z_UP_LH) else "Back (+Z)",
                "color_hex": "#3b82f6",
                "label": MultilingualLookupRegistry.get_term("axis_z", lang)
            }
        }

        return CartesianCoordinateDescriptor(
            coordinate_system=system,
            origin=(0.0, 0.0, 0.0),
            bounding_box=bbox,
            dimensions=dims,
            centroid=centroid,
            octant_distribution=octant_dist,
            spherical_centroid=spherical,
            cylindrical_centroid=cylindrical,
            transform_matrix_4x4=CoordinateSystemConverter.get_matrix_for_system(system),
            handedness=handedness,
            axes_definitions=axes_defs
        )


# ==============================================================================
# DATA MODELS
# ==============================================================================

@dataclass
class ShapeModifierSpec:
    modifier_type: ModifierType
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modifier_type": self.modifier_type.value,
            "parameters": self.parameters
        }


@dataclass
class ShapeEconomicPlan:
    shape_id: str
    archetype: ShapeArchetype
    crystal_cost: float
    mana_cost: float
    gold_cost: float
    build_time_seconds: float
    structural_hp: int
    complexity_score: float
    upkeep_mana_per_cycle: float
    vital_max_hp_rule: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["archetype"] = self.archetype.value
        return d


@dataclass
class PolymorphicShapeInstance:
    shape_id: str
    name: str
    archetype: ShapeArchetype
    parameters: Dict[str, float]
    modifiers: List[ShapeModifierSpec]
    economic_plan: ShapeEconomicPlan
    coordinate_descriptor: CartesianCoordinateDescriptor
    vertex_count: int
    face_count: int
    bounding_box: Dict[str, float]
    vital_max_hp_rule: int = VITAL_MAX_HP
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape_id": self.shape_id,
            "name": self.name,
            "archetype": self.archetype.value,
            "parameters": self.parameters,
            "modifiers": [m.to_dict() for m in self.modifiers],
            "economic_plan": self.economic_plan.to_dict(),
            "coordinate_descriptor": self.coordinate_descriptor.to_dict(),
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "bounding_box": self.bounding_box,
            "vital_max_hp_rule": self.vital_max_hp_rule,
            "created_at": self.created_at
        }

    def get_localized(self, lang: LanguageMode = LanguageMode.SK) -> Dict[str, Any]:
        d = self.to_dict()
        arch_info = MultilingualLookupRegistry.get_archetype_info(self.archetype, lang)
        d["localized_name"] = arch_info["name"]
        d["localized_description"] = arch_info["description"]
        d["localized_language"] = lang.value
        # Localize coordinate descriptor octant labels
        coord_dict = d["coordinate_descriptor"]
        for oct_key, oct_val in coord_dict.get("octant_distribution", {}).items():
            oct_val["label"] = MultilingualLookupRegistry.get_term(oct_key, lang)
        return d


# ==============================================================================
# TEMPLATE FACTORY & ECONOMIC OBSERVER
# ==============================================================================

class ShapeTemplateFactory:
    """
    Factory generating canonical parametric shape templates with predefined
    Blender modifier stacks and parametric ranges.
    """

    @staticmethod
    def get_canonical_templates(lang: LanguageMode = LanguageMode.SK) -> Dict[ShapeArchetype, Dict[str, Any]]:
        return {
            ShapeArchetype.TACTICAL_OBELISK: {
                "name": MultilingualLookupRegistry.get_archetype_info(ShapeArchetype.TACTICAL_OBELISK, lang)["name"],
                "description": MultilingualLookupRegistry.get_archetype_info(ShapeArchetype.TACTICAL_OBELISK, lang)["description"],
                "default_params": {
                    "height": 4.5,
                    "base_radius": 1.2,
                    "top_radius": 0.35,
                    "taper_ratio": 0.78,
                    "twist_deg": 15.0
                },
                "modifiers": [
                    ShapeModifierSpec(ModifierType.BEVEL, {"width": 0.08, "segments": 3}),
                    ShapeModifierSpec(ModifierType.ARRAY, {"count": 3, "offset_z": 0.0, "angle_deg": 120.0})
                ]
            },
            ShapeArchetype.CITADEL_SPIRE: {
                "name": MultilingualLookupRegistry.get_archetype_info(ShapeArchetype.CITADEL_SPIRE, lang)["name"],
                "description": MultilingualLookupRegistry.get_archetype_info(ShapeArchetype.CITADEL_SPIRE, lang)["description"],
                "default_params": {
                    "height": 6.2,
                    "base_radius": 2.0,
                    "levels": 4.0,
                    "crenellation_count": 8.0
                },
                "modifiers": [
                    ShapeModifierSpec(ModifierType.SOLIDIFY, {"thickness": 0.15}),
                    ShapeModifierSpec(ModifierType.MIRROR, {"use_axis_x": True, "use_axis_y": True})
                ]
            },
            ShapeArchetype.BIOPHILIC_ORGANIC: {
                "name": MultilingualLookupRegistry.get_archetype_info(ShapeArchetype.BIOPHILIC_ORGANIC, lang)["name"],
                "description": MultilingualLookupRegistry.get_archetype_info(ShapeArchetype.BIOPHILIC_ORGANIC, lang)["description"],
                "default_params": {
                    "radius": 1.8,
                    "spiral_turns": 2.5,
                    "growth_rate": GOLDEN_RATIO,
                    "noise_displacement": 0.22
                },
                "modifiers": [
                    ShapeModifierSpec(ModifierType.DISPLACE, {"strength": 0.25, "midlevel": 0.5}),
                    ShapeModifierSpec(ModifierType.SUBSURF, {"levels": 2})
                ]
            },
            ShapeArchetype.KINETIC_MECHANISM: {
                "name": MultilingualLookupRegistry.get_archetype_info(ShapeArchetype.KINETIC_MECHANISM, lang)["name"],
                "description": MultilingualLookupRegistry.get_archetype_info(ShapeArchetype.KINETIC_MECHANISM, lang)["description"],
                "default_params": {
                    "radius": 2.2,
                    "thickness": 0.45,
                    "teeth_count": 16.0,
                    "inner_hole_radius": 0.6
                },
                "modifiers": [
                    ShapeModifierSpec(ModifierType.BEVEL, {"width": 0.04, "segments": 2})
                ]
            },
            ShapeArchetype.ETHEREAL_CRYSTAL: {
                "name": MultilingualLookupRegistry.get_archetype_info(ShapeArchetype.ETHEREAL_CRYSTAL, lang)["name"],
                "description": MultilingualLookupRegistry.get_archetype_info(ShapeArchetype.ETHEREAL_CRYSTAL, lang)["description"],
                "default_params": {
                    "scale": 1.5,
                    "aspect_ratio": GOLDEN_RATIO,
                    "satellite_count": 4.0
                },
                "modifiers": [
                    ShapeModifierSpec(ModifierType.SIMPLE_DEFORM, {"mode": "TWIST", "angle": 45.0}),
                    ShapeModifierSpec(ModifierType.BEVEL, {"width": 0.05, "segments": 1})
                ]
            },
            ShapeArchetype.TACTICAL_DRONE: {
                "name": MultilingualLookupRegistry.get_archetype_info(ShapeArchetype.TACTICAL_DRONE, lang)["name"],
                "description": MultilingualLookupRegistry.get_archetype_info(ShapeArchetype.TACTICAL_DRONE, lang)["description"],
                "default_params": {
                    "arm_span": 2.8,
                    "body_radius": 0.65,
                    "rotor_mount_radius": 0.25
                },
                "modifiers": [
                    ShapeModifierSpec(ModifierType.MIRROR, {"use_axis_x": True, "use_axis_y": True}),
                    ShapeModifierSpec(ModifierType.BEVEL, {"width": 0.03, "segments": 2})
                ]
            }
        }


class ShapeEconomicObserver:
    """
    Evaluates resource economics, structural health, and complexity budgets
    for any instantiated polymorphic shape, strictly obeying VITAL_MAX_HP == 6.
    """

    @staticmethod
    def calculate_plan(
        shape_id: str,
        archetype: ShapeArchetype,
        params: Dict[str, float],
        modifiers: List[ShapeModifierSpec]
    ) -> ShapeEconomicPlan:
        h = params.get("height", params.get("scale", 2.0))
        r = params.get("base_radius", params.get("radius", 1.0))
        volume = math.pi * (r ** 2) * h * 0.5

        # Base material multipliers
        crystal_cost = round(max(15.0, volume * 18.5), 1)
        mana_cost = round(max(25.0, len(modifiers) * 35.0 + volume * 8.2), 1)
        gold_cost = round(max(10.0, volume * 12.0), 1)

        # Manufacturing duration
        build_time = round(max(5.0, volume * 4.5 + len(modifiers) * 3.0), 1)

        # Structural integrity (HP clamped strictly to VITAL_MAX_HP == 6)
        density = max(0.08, min(1.0, volume / 10.0))
        structural_hp = min(VITAL_MAX_HP, max(1, int(round(density * VITAL_MAX_HP))))

        # Complexity score
        complexity = round(volume * 1.5 + len(modifiers) * 4.0, 2)
        upkeep_mana = round(complexity * 0.12, 2)

        return ShapeEconomicPlan(
            shape_id=shape_id,
            archetype=archetype,
            crystal_cost=crystal_cost,
            mana_cost=mana_cost,
            gold_cost=gold_cost,
            build_time_seconds=build_time,
            structural_hp=structural_hp,
            complexity_score=complexity,
            upkeep_mana_per_cycle=upkeep_mana,
            vital_max_hp_rule=VITAL_MAX_HP
        )


# ==============================================================================
# BLENDER SCRIPT & MESH GENERATOR
# ==============================================================================

class BlenderScriptGenerator:
    """
    Generates standalone, executable Blender 4.x Python scripts (`bpy`) with
    explicit metric Cartesian coordinate system setup, empty coordinate gizmos,
    custom properties, and multilingual step annotations.
    """

    @staticmethod
    def generate_blender_script(
        instance: PolymorphicShapeInstance,
        lang: LanguageMode = LanguageMode.SK
    ) -> str:
        s = instance
        arch = s.archetype
        p = s.parameters
        coord = s.coordinate_descriptor

        # Multilingual step headers
        headers = {
            LanguageMode.SK: [
                "1. Nastavenie Súradnicového Systému (Blender Pravotočivá Karteziánska, Metrické)",
                "2. Vyčistenie predchádzajúcich objektov v scéne",
                "3. Vytvorenie orientačného súradnicového gizma v počiatku (0,0,0)",
                "4. Konštrukcia základnej geometrie",
                "5. Aplikácia procedurálneho zásobníka modifikátorov (Modifier Stack)",
                "6. Nastavenie kybernetického materiálu a krystalického BSDF tieňovania",
                "7. Priradenie vlastných atribútov (Custom Properties & Metadata)"
            ],
            LanguageMode.EN: [
                "1. Coordinate System & Unit Setup (Blender Right-Handed Cartesian, Metric)",
                "2. Clear previous scene objects",
                "3. Create reference coordinate gizmo at origin (0,0,0)",
                "4. Construct base mesh geometry",
                "5. Apply procedural modifier stack",
                "6. Configure cybernetic crystal material and BSDF shading",
                "7. Assign custom properties & coordinate metadata"
            ],
            LanguageMode.CZ: [
                "1. Nastavení Souřadnicového Systému (Blender Pravotočivá Kartézská, Metrické)",
                "2. Vyčištění předchozích objektů ve scéně",
                "3. Vytvoření orientačního souřadnicového gizma v počátku (0,0,0)",
                "4. Konstrukce základní geometrie sítě",
                "5. Aplikace procedurálního zásobníku modifikátorů",
                "6. Nastavení kybernetického materiálu a krystalického BSDF stínování",
                "7. Přiřazení vlastních atributů (Custom Properties & Metadata)"
            ],
            LanguageMode.DE: [
                "1. Koordinatensystem- und Einheiten-Setup (Blender Rechtshändig Kartesisch, Metrisch)",
                "2. Vorherige Szenenobjekte löschen",
                "3. Erstellung des Koordinaten-Gizmos im Ursprung (0,0,0)",
                "4. Konstruktion der Basisgeometrie",
                "5. Anwenden des prozeduralen Modifikator-Stapels",
                "6. Cybernetisches Kristallmaterial und BSDF-Schattierung konfigurieren",
                "7. Benutzerdefinierte Eigenschaften und Koordinaten-Metadaten zuweisen"
            ]
        }
        steps = headers.get(lang, headers[LanguageMode.EN])

        script_lines = [
            '# ============================================================================== #',
            f'# KRYSTAL-STACK BLENDER 4.x SCRIPT: {s.name.upper()}',
            f'# Archetype: {arch.value} | Shape ID: {s.shape_id}',
            f'# Language Mode: {lang.value} | Structural HP: {s.economic_plan.structural_hp} / {VITAL_MAX_HP}',
            f'# Coordinate Frame: BLENDER_Z_UP (Right-Handed Cartesian, Metric)',
            f'# Centroid: ({coord.centroid[0]:.3f}, {coord.centroid[1]:.3f}, {coord.centroid[2]:.3f})',
            '# ============================================================================== #',
            'import bpy',
            'import math',
            '',
            f'# {steps[0]}',
            'bpy.context.scene.unit_settings.system = "METRIC"',
            'bpy.context.scene.unit_settings.scale_length = 1.0',
            '',
            f'# {steps[1]}',
            'bpy.ops.object.select_all(action="SELECT")',
            'bpy.ops.object.delete(use_global=False)',
            '',
            f'# {steps[2]}',
            'bpy.ops.object.empty_add(type="ARROWS", location=(0.0, 0.0, 0.0))',
            'gizmo = bpy.context.active_object',
            'gizmo.name = "Krystal_Cartesian_Gizmo"',
            'gizmo.empty_display_size = 1.0',
            '',
            f'# {steps[3]}',
        ]

        if arch == ShapeArchetype.TACTICAL_OBELISK:
            h = p.get("height", 4.5)
            r = p.get("base_radius", 1.2)
            script_lines.extend([
                f'bpy.ops.mesh.primitive_cone_add(vertices=6, radius1={r}, radius2={r*0.2}, depth={h}, location=(0, 0, {h/2}))',
                'obj = bpy.context.active_object',
                f'obj.name = "{s.shape_id}_Obelisk"'
            ])
        elif arch == ShapeArchetype.CITADEL_SPIRE:
            h = p.get("height", 6.0)
            r = p.get("base_radius", 2.0)
            script_lines.extend([
                f'bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius={r}, depth={h}, location=(0, 0, {h/2}))',
                'obj = bpy.context.active_object',
                f'obj.name = "{s.shape_id}_CitadelSpire"'
            ])
        elif arch == ShapeArchetype.BIOPHILIC_ORGANIC:
            r = p.get("radius", 1.8)
            script_lines.extend([
                f'bpy.ops.mesh.primitive_torus_add(major_radius={r}, minor_radius={r*0.35}, location=(0, 0, {r}))',
                'obj = bpy.context.active_object',
                f'obj.name = "{s.shape_id}_BiophilicShrine"'
            ])
        elif arch == ShapeArchetype.KINETIC_MECHANISM:
            r = p.get("radius", 2.2)
            th = p.get("thickness", 0.45)
            script_lines.extend([
                f'bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius={r}, depth={th}, location=(0, 0, {th/2}))',
                'obj = bpy.context.active_object',
                f'obj.name = "{s.shape_id}_KineticGear"'
            ])
        elif arch == ShapeArchetype.ETHEREAL_CRYSTAL:
            sc = p.get("scale", 1.5)
            script_lines.extend([
                f'bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius={sc}, location=(0, 0, {sc*1.2}))',
                'obj = bpy.context.active_object',
                f'obj.name = "{s.shape_id}_CrystalCluster"'
            ])
        else:
            # Tactical drone
            script_lines.extend([
                'bpy.ops.mesh.primitive_cube_add(size=1.2, location=(0, 0, 0.6))',
                'obj = bpy.context.active_object',
                f'obj.name = "{s.shape_id}_DroneChassis"'
            ])

        script_lines.append('')
        script_lines.append(f'# {steps[4]}')

        for idx, mod in enumerate(s.modifiers):
            m_type = mod.modifier_type
            params = mod.parameters
            m_name = f"{m_type.value}_{idx+1}"

            if m_type == ModifierType.BEVEL:
                w = params.get("width", 0.05)
                seg = params.get("segments", 2)
                script_lines.extend([
                    f'mod = obj.modifiers.new(name="{m_name}", type="BEVEL")',
                    f'mod.width = {w}',
                    f'mod.segments = {seg}'
                ])
            elif m_type == ModifierType.SOLIDIFY:
                th = params.get("thickness", 0.1)
                script_lines.extend([
                    f'mod = obj.modifiers.new(name="{m_name}", type="SOLIDIFY")',
                    f'mod.thickness = {th}'
                ])
            elif m_type == ModifierType.ARRAY:
                cnt = int(params.get("count", 3))
                script_lines.extend([
                    f'mod = obj.modifiers.new(name="{m_name}", type="ARRAY")',
                    f'mod.count = {cnt}',
                    'mod.relative_offset_displace[0] = 1.2'
                ])
            elif m_type == ModifierType.MIRROR:
                script_lines.extend([
                    f'mod = obj.modifiers.new(name="{m_name}", type="MIRROR")',
                    'mod.use_axis[0] = True',
                    'mod.use_axis[1] = True'
                ])
            elif m_type == ModifierType.DISPLACE:
                st = params.get("strength", 0.2)
                script_lines.extend([
                    f'mod = obj.modifiers.new(name="{m_name}", type="DISPLACE")',
                    f'mod.strength = {st}'
                ])
            elif m_type == ModifierType.SUBSURF:
                lv = int(params.get("levels", 1))
                script_lines.extend([
                    f'mod = obj.modifiers.new(name="{m_name}", type="SUBSURF")',
                    f'mod.levels = {lv}',
                    f'mod.render_levels = {lv+1}'
                ])
            elif m_type == ModifierType.SIMPLE_DEFORM:
                ang = params.get("angle", 30.0)
                script_lines.extend([
                    f'mod = obj.modifiers.new(name="{m_name}", type="SIMPLE_DEFORM")',
                    'mod.deform_method = "TWIST"',
                    f'mod.angle = math.radians({ang})'
                ])

        script_lines.extend([
            '',
            f'# {steps[5]}',
            'mat = bpy.data.materials.new(name="Krystal_Material")',
            'mat.use_nodes = True',
            'nodes = mat.node_tree.nodes',
            'bsdf = nodes.get("Principled BSDF")',
            'if bsdf:',
            '    bsdf.inputs["Roughness"].default_value = 0.15',
            '    bsdf.inputs["Metallic"].default_value = 0.85',
            '    bsdf.inputs["Base Color"].default_value = (0.0, 0.94, 1.0, 1.0) # Neon Cyan',
            'obj.data.materials.append(mat)',
            'for poly in obj.data.polygons:',
            '    poly.use_smooth = True',
            '',
            f'# {steps[6]}',
            f'obj["krystal_shape_id"] = "{s.shape_id}"',
            f'obj["krystal_archetype"] = "{arch.value}"',
            f'obj["krystal_structural_hp"] = {s.economic_plan.structural_hp}',
            f'obj["krystal_vital_max_hp"] = {VITAL_MAX_HP}',
            'obj["krystal_coordinate_system"] = "CARTESIAN_RIGHT_HANDED_Z_UP"',
            f'obj["krystal_centroid"] = [{coord.centroid[0]}, {coord.centroid[1]}, {coord.centroid[2]}]',
            f'obj["krystal_bounding_box"] = [{coord.bounding_box["min_x"]}, {coord.bounding_box["max_x"]}, {coord.bounding_box["min_y"]}, {coord.bounding_box["max_y"]}, {coord.bounding_box["min_z"]}, {coord.bounding_box["max_z"]}]',
            '',
            f'print(f"Krystal-Stack: Successfully instantiated {{obj.name}} with {{len(obj.data.vertices)}} vertices (HP: {s.economic_plan.structural_hp}/{VITAL_MAX_HP}).")'
        ])

        return "\n".join(script_lines)

    @staticmethod
    def generate_obj_mesh(instance: PolymorphicShapeInstance) -> str:
        """Generates standard Wavefront .OBJ mesh string in Blender Cartesian coordinates."""
        s = instance
        h = s.parameters.get("height", s.parameters.get("scale", 2.0))
        r = s.parameters.get("base_radius", s.parameters.get("radius", 1.0))

        segments = 8
        verts = []
        for i in range(segments):
            angle = (2 * math.pi / segments) * i
            verts.append((r * math.cos(angle), r * math.sin(angle), 0.0))
        top_r = r * 0.4
        for i in range(segments):
            angle = (2 * math.pi / segments) * i
            verts.append((top_r * math.cos(angle), top_r * math.sin(angle), h))
        verts.append((0.0, 0.0, h + r * 0.3))

        lines = [
            f"# Krystal-Stack Wavefront OBJ: {s.name}",
            f"# Archetype: {s.archetype.value}",
            f"# Coordinate System: BLENDER_Z_UP (Cartesian Right-Handed)",
            f"# Max HP: {s.vital_max_hp_rule}"
        ]
        for v in verts:
            lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")

        lines.append("vn 0.0 0.0 1.0")
        lines.append("vn 0.0 0.0 -1.0")

        for i in range(segments):
            nxt = (i + 1) % segments
            b1 = i + 1
            b2 = nxt + 1
            t1 = segments + i + 1
            t2 = segments + nxt + 1
            lines.append(f"f {b1} {b2} {t2} {t1}")

        apex_idx = len(verts)
        for i in range(segments):
            nxt = (i + 1) % segments
            t1 = segments + i + 1
            t2 = segments + nxt + 1
            lines.append(f"f {t1} {t2} {apex_idx}")

        return "\n".join(lines)


# ==============================================================================
# COMMAND SHAPE PLATFORM (CENTRAL ENGINE)
# ==============================================================================

class CommandShapePlatform:
    """
    Central shape management and execution platform implementing the Command Pattern,
    linking Template Factory, Economic Observer, Multilingual Registry, and Blender Exporters.
    """

    def __init__(self, default_lang: LanguageMode = LanguageMode.SK):
        self._library: Dict[str, PolymorphicShapeInstance] = {}
        self._history: List[Dict[str, Any]] = []
        self._language_mode: LanguageMode = default_lang
        self._lock = threading.RLock()
        self._init_starter_shapes()

    def get_language_mode(self) -> LanguageMode:
        with self._lock:
            return self._language_mode

    def set_language_mode(self, mode: LanguageMode) -> LanguageMode:
        with self._lock:
            self._language_mode = mode
            self._record_history(ShapeCommandType.SET_LANGUAGE_MODE, {"language_mode": mode.value})
            return self._language_mode

    def _init_starter_shapes(self):
        """Instantiates default canonical starter library."""
        templates = ShapeTemplateFactory.get_canonical_templates(self._language_mode)
        for arch, tpl in templates.items():
            self.create_shape(
                archetype=arch,
                name=tpl["name"],
                parameters=dict(tpl["default_params"]),
                modifiers=list(tpl["modifiers"])
            )

    def create_shape(
        self,
        archetype: ShapeArchetype,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, float]] = None,
        modifiers: Optional[List[ShapeModifierSpec]] = None
    ) -> PolymorphicShapeInstance:
        with self._lock:
            tpls = ShapeTemplateFactory.get_canonical_templates(self._language_mode)
            default_tpl = tpls.get(archetype, {})

            shape_id = f"shape_{archetype.value.lower()}_{uuid.uuid4().hex[:6]}"
            s_name = name or default_tpl.get("name", str(archetype.value))
            s_params = parameters if parameters is not None else dict(default_tpl.get("default_params", {}))
            s_mods = modifiers if modifiers is not None else list(default_tpl.get("modifiers", []))

            plan = ShapeEconomicObserver.calculate_plan(shape_id, archetype, s_params, s_mods)
            coord_desc = MeshGeometryDescriptor.build_coordinate_descriptor(
                archetype, s_params, CoordinateSystemType.BLENDER_Z_UP, self._language_mode
            )

            v_count = 17 + len(s_mods) * 8
            f_count = 16 + len(s_mods) * 8

            instance = PolymorphicShapeInstance(
                shape_id=shape_id,
                name=s_name,
                archetype=archetype,
                parameters=s_params,
                modifiers=s_mods,
                economic_plan=plan,
                coordinate_descriptor=coord_desc,
                vertex_count=v_count,
                face_count=f_count,
                bounding_box=coord_desc.bounding_box,
                vital_max_hp_rule=VITAL_MAX_HP
            )

            self._library[shape_id] = instance
            self._record_history(ShapeCommandType.CREATE_SHAPE, {"shape_id": shape_id, "archetype": archetype.value})
            return instance

    def morph_shape(self, shape_id: str, new_parameters: Dict[str, float]) -> Optional[PolymorphicShapeInstance]:
        with self._lock:
            instance = self._library.get(shape_id)
            if not instance:
                return None

            instance.parameters.update(new_parameters)
            instance.economic_plan = ShapeEconomicObserver.calculate_plan(
                shape_id, instance.archetype, instance.parameters, instance.modifiers
            )
            instance.coordinate_descriptor = MeshGeometryDescriptor.build_coordinate_descriptor(
                instance.archetype, instance.parameters, CoordinateSystemType.BLENDER_Z_UP, self._language_mode
            )
            instance.bounding_box = instance.coordinate_descriptor.bounding_box

            self._record_history(ShapeCommandType.MORPH_PARAMETERS, {"shape_id": shape_id, "params": new_parameters})
            return instance

    def apply_modifier(self, shape_id: str, modifier: ShapeModifierSpec) -> Optional[PolymorphicShapeInstance]:
        with self._lock:
            instance = self._library.get(shape_id)
            if not instance:
                return None

            instance.modifiers.append(modifier)
            instance.economic_plan = ShapeEconomicObserver.calculate_plan(
                shape_id, instance.archetype, instance.parameters, instance.modifiers
            )
            instance.vertex_count += 8
            instance.face_count += 8

            self._record_history(ShapeCommandType.APPLY_MODIFIER, {"shape_id": shape_id, "mod": modifier.to_dict()})
            return instance

    def export_blender_script(self, shape_id: str, lang: Optional[LanguageMode] = None) -> Optional[str]:
        with self._lock:
            instance = self._library.get(shape_id)
            if not instance:
                return None
            target_lang = lang or self._language_mode
            return BlenderScriptGenerator.generate_blender_script(instance, lang=target_lang)

    def export_obj_mesh(self, shape_id: str) -> Optional[str]:
        with self._lock:
            instance = self._library.get(shape_id)
            if not instance:
                return None
            return BlenderScriptGenerator.generate_obj_mesh(instance)

    def get_coordinate_analysis(
        self,
        shape_id: str,
        target_system: CoordinateSystemType = CoordinateSystemType.BLENDER_Z_UP,
        lang: Optional[LanguageMode] = None
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            instance = self._library.get(shape_id)
            if not instance:
                return None
            target_lang = lang or self._language_mode
            desc = MeshGeometryDescriptor.build_coordinate_descriptor(
                instance.archetype, instance.parameters, target_system, target_lang
            )
            return desc.to_dict()

    def get_library(self, lang: Optional[LanguageMode] = None) -> List[Dict[str, Any]]:
        with self._lock:
            target_lang = lang or self._language_mode
            return [s.get_localized(target_lang) for s in self._library.values()]

    def get_shape(self, shape_id: str, lang: Optional[LanguageMode] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            instance = self._library.get(shape_id)
            if not instance:
                return None
            target_lang = lang or self._language_mode
            return instance.get_localized(target_lang)

    def _record_history(self, command: ShapeCommandType, details: Dict[str, Any]):
        self._history.append({
            "command": command.value,
            "timestamp": time.time(),
            "details": details
        })
        if len(self._history) > 100:
            self._history.pop(0)

    def get_status(self, lang: Optional[LanguageMode] = None) -> Dict[str, Any]:
        with self._lock:
            target_lang = lang or self._language_mode
            total_cost = sum(s.economic_plan.crystal_cost for s in self._library.values())
            total_hp = sum(s.economic_plan.structural_hp for s in self._library.values())
            return {
                "platform_name": MultilingualLookupRegistry.get_term("platform_title", target_lang),
                "active_language": self._language_mode.value,
                "supported_languages": [m.value for m in LanguageMode],
                "coordinate_convention": "BLENDER_Z_UP_RIGHT_HANDED_METRIC",
                "registered_shapes_count": len(self._library),
                "total_crystal_valuation": round(total_cost, 2),
                "total_structural_hp": total_hp,
                "vital_max_hp_rule": VITAL_MAX_HP,
                "history_length": len(self._history),
                "shapes": [s.get_localized(target_lang) for s in self._library.values()]
            }


# ==============================================================================
# GLOBAL SINGLETON INSTANCE
# ==============================================================================

GLOBAL_SHAPE_PLATFORM = CommandShapePlatform()


if __name__ == "__main__":
    print("==============================================================================")
    print("KRYSTAL-STACK: POLYMORPHIC SHAPE & BLENDER PLATFORM (MULTILINGUAL & CARTESIAN)")
    print("==============================================================================")
    plat = GLOBAL_SHAPE_PLATFORM
    print("Language Mode:", plat.get_language_mode().value)
    print("Registered Shapes:", len(plat._library))
    obelisk = plat.create_shape(ShapeArchetype.TACTICAL_OBELISK, parameters={"height": 5.0, "base_radius": 1.5})
    print(f"Created: {obelisk.name} | HP: {obelisk.economic_plan.structural_hp} / {VITAL_MAX_HP}")
    print("Centroid:", obelisk.coordinate_descriptor.centroid)
    print("Octants:", [f"{k}: {v['percentage']}%" for k, v in obelisk.coordinate_descriptor.octant_distribution.items() if v['percentage'] > 0])
    script = plat.export_blender_script(obelisk.shape_id, lang=LanguageMode.EN)
    print("Blender Script Preview (EN):\n", script[:380], "...\n")
