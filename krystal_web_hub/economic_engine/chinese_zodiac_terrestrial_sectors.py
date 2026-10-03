"""
KRYSTAL-STACK // CHINESE ZODIAC TERRESTRIAL PHENOMENA & SECTOR TOPOLOGY
=======================================================================
Implements the 12 Earthly Branches (Shēngxiào / 地支) mapped directly to
terrestrial phenomena (geological, meteorological, geomagnetic, hydrologic)
and dedicated spatial sectors across the platform world grid.

Invariant: VITAL_MAX_HP = 6
Golden Mean: phi = 1.61803398875
"""

import math
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 0.61803398875


class ZodiacElement(str, Enum):
    WOOD = "Wood (Drevo / 木)"
    FIRE = "Fire (Oheň / 火)"
    EARTH = "Earth (Zem / 土)"
    METAL = "Metal (Kov / 金)"
    WATER = "Water (Voda / 水)"


class YinYang(str, Enum):
    YIN = "Yin (Temná/Chladná/Prijímajúca polarita)"
    YANG = "Yang (Svetlá/Teplá/Aktívna polarita)"


@dataclass
class TerrestrialPhenomenon:
    """Represents an active earthly/physical phenomenon localized in a sector."""
    code: str
    name: str
    description: str
    manifestation: str
    element: ZodiacElement
    polarity: YinYang
    shader_effect: str
    resource_modifier: Dict[str, float]
    vital_max_hp_rule: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "description": self.description,
            "manifestation": self.manifestation,
            "element": self.element.value,
            "polarity": self.polarity.value,
            "shader_effect": self.shader_effect,
            "resource_modifier": self.resource_modifier,
            "vital_max_hp_rule": self.vital_max_hp_rule
        }


@dataclass
class ZodiacSector:
    """Represents a dedicated spatial sector in the world grid corresponding to a Zodiac sign."""
    sector_id: str
    zodiac_name: str
    chinese_glyph: str
    earthly_branch: str
    element: ZodiacElement
    polarity: YinYang
    cardinal_direction: str
    hex_coordinates: List[List[int]]  # list of [q, r]
    phenomenon: TerrestrialPhenomenon
    fortification_hp: int = VITAL_MAX_HP
    max_fortification_hp: int = VITAL_MAX_HP
    resonance_intensity: float = 1.0
    controlling_faction: str = "neutral"
    terrain_type: str = "alchemical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sector_id": self.sector_id,
            "zodiac_name": self.zodiac_name,
            "chinese_glyph": self.chinese_glyph,
            "earthly_branch": self.earthly_branch,
            "element": self.element.value,
            "polarity": self.polarity.value,
            "cardinal_direction": self.cardinal_direction,
            "hex_coordinates": self.hex_coordinates,
            "phenomenon": self.phenomenon.to_dict(),
            "fortification_hp": self.fortification_hp,
            "max_fortification_hp": self.max_fortification_hp,
            "resonance_intensity": round(self.resonance_intensity, 3),
            "controlling_faction": self.controlling_faction,
            "terrain_type": self.terrain_type,
            "vital_max_hp_rule": VITAL_MAX_HP
        }


# ── The 12 Canonical Chinese Zodiac Terrestrial Sectors ──────────────────────
CANONICAL_ZODIAC_SECTORS: Dict[str, ZodiacSector] = {
    "sector_01_rat": ZodiacSector(
        sector_id="sector_01_rat",
        zodiac_name="Potkan (Rat / 鼠)",
        chinese_glyph="子 (Zǐ)",
        earthly_branch="1. Zǐ (23:00 - 01:00)",
        element=ZodiacElement.WATER,
        polarity=YinYang.YANG,
        cardinal_direction="Sever (0° / North)",
        hex_coordinates=[[0, -3], [0, -2], [1, -3]],
        phenomenon=TerrestrialPhenomenon(
            code="PHEN_SUBTERRANEAN_SPRINGS",
            name="Nočná Hydrologická Spodná Voda & Infiltrácia",
            description="Chladné pramene a podzemné artézske toky filtrujúce aéter cez podložie.",
            manifestation="Prúdenie kryštalickej vody, nočná rosa, hydrologické chladenie čipov.",
            element=ZodiacElement.WATER,
            polarity=YinYang.YANG,
            shader_effect="water_caustics_condensation",
            resource_modifier={"aether_crystals": 1.4, "cache_speed": 1.25}
        ),
        terrain_type="Kryštálové Pramene Hlbín"
    ),
    "sector_02_ox": ZodiacSector(
        sector_id="sector_02_ox",
        zodiac_name="Byvol (Ox / 牛)",
        chinese_glyph="丑 (Chǒu)",
        earthly_branch="2. Chǒu (01:00 - 03:00)",
        element=ZodiacElement.EARTH,
        polarity=YinYang.YIN,
        cardinal_direction="Severo-Severovýchod (30° / NNE)",
        hex_coordinates=[[1, -2], [2, -3], [1, -1]],
        phenomenon=TerrestrialPhenomenon(
            code="PHEN_TECTONIC_BEDROCK",
            name="Kryo-Tektonické Vrstvenie & Pôdny Masív",
            description="Extrémna geomorfológia stlačenej žuly a odolnosť voči otrasom.",
            manifestation="Husté skalné pukliny, nulová fragmentácia terénu, stabilita zbernice.",
            element=ZodiacElement.EARTH,
            polarity=YinYang.YIN,
            shader_effect="granite_crack_parallax",
            resource_modifier={"defense_bonus": 1.5, "structural_integrity": 1.35}
        ),
        terrain_type="Pevninský Masív Žuly"
    ),
    "sector_03_tiger": ZodiacSector(
        sector_id="sector_03_tiger",
        zodiac_name="Tiger (Tiger / 虎)",
        chinese_glyph="寅 (Yín)",
        earthly_branch="3. Yín (03:00 - 05:00)",
        element=ZodiacElement.WOOD,
        polarity=YinYang.YANG,
        cardinal_direction="Východo-Severovýchod (60° / ENE)",
        hex_coordinates=[[2, -2], [2, -1], [3, -2]],
        phenomenon=TerrestrialPhenomenon(
            code="PHEN_FOREST_LIGHTNING",
            name="Ranné Blesky & Kinetická Lesná Búrka",
            description="Náhle elektrostatické výboje prechádzajúce korunami ihličnanov.",
            manifestation="Elektrostatické iskrenie, ozón, bleskový nárast L1 inštrukčných burstov.",
            element=ZodiacElement.WOOD,
            polarity=YinYang.YANG,
            shader_effect="lightning_arcing_foliage",
            resource_modifier={"attack_burst": 1.6, "l1_bandwidth": 1.4}
        ),
        terrain_type="Hromová Tajga & Les Šeliem"
    ),
    "sector_04_rabbit": ZodiacSector(
        sector_id="sector_04_rabbit",
        zodiac_name="Zajac (Rabbit / 兔)",
        chinese_glyph="卯 (Mǎo)",
        earthly_branch="4. Mǎo (05:00 - 07:00)",
        element=ZodiacElement.WOOD,
        polarity=YinYang.YIN,
        cardinal_direction="Východ (90° / East)",
        hex_coordinates=[[2, 0], [3, -1], [3, 0]],
        phenomenon=TerrestrialPhenomenon(
            code="PHEN_PHYTO_SPORE_MIST",
            name="Ranná Hmla & Rast Fyto-Spór",
            description="Hustý jantárový opar podporujúci rýchlu bunečnú regeneráciu.",
            manifestation="Svetielkujúci peľ, biosféra, neustále liečenie organických jednotiek.",
            element=ZodiacElement.WOOD,
            polarity=YinYang.YIN,
            shader_effect="amber_spore_subsurface",
            resource_modifier={"amber_runes": 1.5, "healing_rate": 1.3}
        ),
        terrain_type="Miazgový Háj Jantáru"
    ),
    "sector_05_dragon": ZodiacSector(
        sector_id="sector_05_dragon",
        zodiac_name="Drak (Dragon / 龙)",
        chinese_glyph="辰 (Chén)",
        earthly_branch="5. Chén (07:00 - 09:00)",
        element=ZodiacElement.EARTH,
        polarity=YinYang.YANG,
        cardinal_direction="Východo-Juhovýchod (120° / ESE)",
        hex_coordinates=[[2, 1], [1, 1], [2, 2]],
        phenomenon=TerrestrialPhenomenon(
            code="PHEN_GEOMAGNETIC_CALDERA",
            name="Geomagnetická Búrka & Magmatický Gejzír",
            description="Polárna geomagnetická žiara zvírená hlbokým aéterovým lávovým jazerom.",
            manifestation="Pulzujúca magnetosféra, ionizovaný dym, masívna akumulácia energie.",
            element=ZodiacElement.EARTH,
            polarity=YinYang.YANG,
            shader_effect="aurora_volcanic_glow",
            resource_modifier={"mana_surge": 1.8, "fire_damage": 1.5}
        ),
        terrain_type="Dračia Kaldera & Sopka Aéteru"
    ),
    "sector_06_snake": ZodiacSector(
        sector_id="sector_06_snake",
        zodiac_name="Had (Snake / 蛇)",
        chinese_glyph="巳 (Sì)",
        earthly_branch="6. Sì (09:00 - 11:00)",
        element=ZodiacElement.FIRE,
        polarity=YinYang.YIN,
        cardinal_direction="Juho-Juhovýchod (150° / SSE)",
        hex_coordinates=[[1, 2], [0, 2], [1, 3]],
        phenomenon=TerrestrialPhenomenon(
            code="PHEN_GEOTHERMAL_ACID_GAS",
            name="Zemný Plyn & Geotermálna Kyselinová Štrbina",
            description="Korozívne sírne a toxické výpary prestupujúce z hĺbky zemskej kôry.",
            manifestation="Bublajúce bahno, kaustický opar, permanentné toxické poškodenie.",
            element=ZodiacElement.FIRE,
            polarity=YinYang.YIN,
            shader_effect="toxic_bubble_heat_distortion",
            resource_modifier={"toxic_slime": 1.6, "corrosion_rate": 1.4}
        ),
        terrain_type="Geotermálna Štrbina Zeme"
    ),
    "sector_07_horse": ZodiacSector(
        sector_id="sector_07_horse",
        zodiac_name="Kôň (Horse / 马)",
        chinese_glyph="午 (Wǔ)",
        earthly_branch="7. Wǔ (11:00 - 13:00)",
        element=ZodiacElement.FIRE,
        polarity=YinYang.YANG,
        cardinal_direction="Juh (180° / South)",
        hex_coordinates=[[0, 3], [-1, 3], [0, 2]],
        phenomenon=TerrestrialPhenomenon(
            code="PHEN_ZENITH_SOLAR_FLARE",
            name="Poludňajší Solárny Žiar & Termálna Púšť",
            description="Zenitové slnečné lúče dosahujúce maximálny tepelný flux na stepi.",
            manifestation="Mirage / zrkadlenie vzduchu, prehriatie pasívnych chladičov, maximálny boost CPU.",
            element=ZodiacElement.FIRE,
            polarity=YinYang.YANG,
            shader_effect="mirage_heat_refraction",
            resource_modifier={"core_clock_boost": 1.5, "solar_power": 1.75}
        ),
        terrain_type="Solárna Step & Slnečný Megalit"
    ),
    "sector_08_goat": ZodiacSector(
        sector_id="sector_08_goat",
        zodiac_name="Koza / Ovca (Goat / 羊)",
        chinese_glyph="未 (Wèi)",
        earthly_branch="8. Wèi (13:00 - 15:00)",
        element=ZodiacElement.EARTH,
        polarity=YinYang.YIN,
        cardinal_direction="Juho-Juhozápad (210° / SSW)",
        hex_coordinates=[[-1, 2], [-2, 3], [-1, 1]],
        phenomenon=TerrestrialPhenomenon(
            code="PHEN_MINERAL_ALLUVIUM",
            name="Sprašová Usadenina & Minerálne Ložisko",
            description="Jemné aluviálne prachy a vápencové terasy s harmonickou alchýmiou.",
            manifestation="Suchý minerálny prach, kriedové skaliská, stabilné chemické rovnováhy.",
            element=ZodiacElement.EARTH,
            polarity=YinYang.YIN,
            shader_effect="chalk_dust_ambient_occlusion",
            resource_modifier={"crafting_stability": 1.45, "alchemy_yield": 1.3}
        ),
        terrain_type="Minerálna Náhorná Plošina"
    ),
    "sector_09_monkey": ZodiacSector(
        sector_id="sector_09_monkey",
        zodiac_name="Opica (Monkey / 猴)",
        chinese_glyph="申 (Shēn)",
        earthly_branch="9. Shēn (15:00 - 17:00)",
        element=ZodiacElement.METAL,
        polarity=YinYang.YANG,
        cardinal_direction="Západo-Juhozápad (240° / WSW)",
        hex_coordinates=[[-2, 1], [-3, 2], [-2, 0]],
        phenomenon=TerrestrialPhenomenon(
            code="PHEN_IONIC_MOUNTAIN_VORTEX",
            name="Horský Vzdušný Vír & Ionosférická Rezonancia",
            description="Turbulentné vertikálne prúdy vzduchu obtekajúce ihly skalných veží.",
            manifestation="Aerodynamické turbulencie, ionizovaný vietor, akcelerácia prenosu paketov.",
            element=ZodiacElement.METAL,
            polarity=YinYang.YANG,
            shader_effect="ionic_vortex_specular",
            resource_modifier={"aerostat_glide": 1.55, "swap_throughput": 1.35}
        ),
        terrain_type="Ionizačná Skalná Veža"
    ),
    "sector_10_rooster": ZodiacSector(
        sector_id="sector_10_rooster",
        zodiac_name="Kohút (Rooster / 鸡)",
        chinese_glyph="酉 (Yǒu)",
        earthly_branch="10. Yǒu (17:00 - 19:00)",
        element=ZodiacElement.METAL,
        polarity=YinYang.YIN,
        cardinal_direction="Západ (270° / West)",
        hex_coordinates=[[-3, 0], [-2, -1], [-3, 1]],
        phenomenon=TerrestrialPhenomenon(
            code="PHEN_MAGNETITE_QUARRY",
            name="Zrkadlenie Rudných Žíl & Magnetitové Polia",
            description="Magnetické anomálie z ložísk čierneho magnetitu a kovových zrkadiel.",
            manifestation="Polarizovaný magnetický odlesk, kalenie ocele, odklon projektilov.",
            element=ZodiacElement.METAL,
            polarity=YinYang.YIN,
            shader_effect="metallic_anisotropy_reflection",
            resource_modifier={"armor_hardening": 1.6, "ballistic_deflect": 1.3}
        ),
        terrain_type="Magnetitový Lom & Ruda"
    ),
    "sector_11_dog": ZodiacSector(
        sector_id="sector_11_dog",
        zodiac_name="Pes (Dog / 狗)",
        chinese_glyph="戌 (Xū)",
        earthly_branch="11. Xū (19:00 - 21:00)",
        element=ZodiacElement.EARTH,
        polarity=YinYang.YANG,
        cardinal_direction="Západo-Severozápad (300° / WNW)",
        hex_coordinates=[[-2, -1], [-1, -1], [-2, -2]],
        phenomenon=TerrestrialPhenomenon(
            code="PHEN_SEISMIC_CRUST_BARRIER",
            name="Seizmická Hliadka & Pôdna Odolnosť Kôry",
            description="Vysokofrekvenčné mikro-otrasy detegujúce akýkoľvek pohyb pod zemou.",
            manifestation="Seizmické kruhy na pôde, včasná detekcia podkopania, ochranný val.",
            element=ZodiacElement.EARTH,
            polarity=YinYang.YANG,
            shader_effect="seismic_ring_displacement",
            resource_modifier={"recon_detection": 1.7, "stealth_counter": 1.5}
        ),
        terrain_type="Pohraničná Seizmická Hliadka"
    ),
    "sector_12_pig": ZodiacSector(
        sector_id="sector_12_pig",
        zodiac_name="Prasa (Pig / 猪)",
        chinese_glyph="亥 (Hài)",
        earthly_branch="12. Hài (21:00 - 23:00)",
        element=ZodiacElement.WATER,
        polarity=YinYang.YIN,
        cardinal_direction="Severo-Severozápad (330° / NNW)",
        hex_coordinates=[[-1, -2], [0, -3], [-1, -3]],
        phenomenon=TerrestrialPhenomenon(
            code="PHEN_ALLUVIAL_PEAT_BOG",
            name="Hlboká Aluviálna Bažina & Sedimentárne Úložisko",
            description="Starodávne rašeliniská uchovávajúce vrstvy sedimentov a aluviálneho bahna.",
            manifestation="Pomalé usadzovanie kalov, zachytávanie ťažkých častíc, trvalý archív.",
            element=ZodiacElement.WATER,
            polarity=YinYang.YIN,
            shader_effect="murky_peat_absorption",
            resource_modifier={"archive_retention": 1.65, "sludge_density": 1.4}
        ),
        terrain_type="Sedimentárny Močiar & Rašelinisko"
    )
}


class ChineseZodiacSectorEngine:
    """
    Manages the 12 Chinese Zodiac terrestrial sectors, calculates phenomenon dynamics,
    and coordinates elemental synergies with platform combat and memory subsystems.
    """

    def __init__(self):
        self.sectors: Dict[str, ZodiacSector] = {k: v for k, v in CANONICAL_ZODIAC_SECTORS.items()}

    def get_all_sectors(self) -> List[Dict[str, Any]]:
        """Returns full serialization of all 12 Zodiac sectors."""
        return [s.to_dict() for s in self.sectors.values()]

    def get_sector(self, sector_id: str) -> Optional[Dict[str, Any]]:
        s = self.sectors.get(sector_id)
        return s.to_dict() if s else None

    def trigger_phenomenon(self, sector_id: str, intensity_delta: float = 0.25) -> Dict[str, Any]:
        """Modulates resonance intensity of the earthly phenomenon in a sector."""
        if sector_id not in self.sectors:
            raise ValueError(f"Sektor {sector_id} nebol nájdený v čínskom zverokruhu.")

        sec = self.sectors[sector_id]
        sec.resonance_intensity = min(3.0, max(0.2, sec.resonance_intensity + intensity_delta))

        # Fortification HP remains strictly bounded by VITAL_MAX_HP
        sec.fortification_hp = min(VITAL_MAX_HP, sec.fortification_hp)

        return {
            "sector_id": sec.sector_id,
            "zodiac_name": sec.zodiac_name,
            "phenomenon_name": sec.phenomenon.name,
            "new_resonance_intensity": round(sec.resonance_intensity, 3),
            "fortification_hp": sec.fortification_hp,
            "vital_max_hp_rule": VITAL_MAX_HP,
            "status": "PHENOMENON_SURGED"
        }

    def evaluate_global_terrestrial_cycle(self, turn_number: int = 1) -> Dict[str, Any]:
        """
        Calculates the active ruling Earthly Branch for the given cycle/turn
        and returns active global terrestrial bonuses.
        """
        active_index = (turn_number - 1) % 12
        sector_keys = list(self.sectors.keys())
        ruling_key = sector_keys[active_index]
        ruling_sector = self.sectors[ruling_key]

        return {
            "cycle_turn": turn_number,
            "vital_max_hp_rule": VITAL_MAX_HP,
            "ruling_earthly_branch": ruling_sector.earthly_branch,
            "ruling_zodiac_sign": ruling_sector.zodiac_name,
            "ruling_glyph": ruling_sector.chinese_glyph,
            "ruling_phenomenon": ruling_sector.phenomenon.to_dict(),
            "dominant_element": ruling_sector.element.value,
            "dominant_polarity": ruling_sector.polarity.value,
            "active_sectors_count": len(self.sectors),
            "golden_ratio_stride": round(turn_number * INV_GOLDEN_RATIO, 4)
        }


# Global Singleton Instance
GLOBAL_CHINESE_ZODIAC_SECTOR_ENGINE = ChineseZodiacSectorEngine()
