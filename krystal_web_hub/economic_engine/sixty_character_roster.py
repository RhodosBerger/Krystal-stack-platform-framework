# ==============================================================================
# KRYSTAL-STACK: 60 CANONICAL CHARACTER ARCHETYPES (20 PER RACE) & HUD PANELS
# ==============================================================================
# Implements:
#   1. 20 Character types for Kryštálový Kmeň (Crystal Tribe).
#   2. 20 Character types for Jedovatý Kmeň (Toxic Tribe).
#   3. 20 Character types for Prastarý Druidský Kmeň (Druid Tribe).
#   4. Strict 5-tier hierarchical progression (Tier 1 Recruit -> Tier 5 Apex Sovereign).
#   5. Dual attributes: Warhammer combat + The West duel algebra (Toughness, Reflexes, etc.).
#   6. Custom Skill Control Panel HUD definitions for each archetype.
# ==============================================================================

from typing import Dict, List, Any, Optional

def _make_char(
    char_id: str,
    name: str,
    race: str,
    tier: int,
    role: str,
    weapon_favored: str,
    warhammer_stats: Dict[str, Any],
    duel_stats: Dict[str, int],
    signature_skill: Dict[str, Any],
    hud_panel: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "id": char_id,
        "name": name,
        "race": race,
        "tier": tier,
        "role": role,
        "weapon_favored": weapon_favored,
        "warhammer_stats": warhammer_stats,
        "duel_stats": duel_stats,
        "signature_skill": signature_skill,
        "hud_panel": hud_panel
    }

# ------------------------------------------------------------------------------
# 1. KRYŠTÁLOVÝ KMEŇ (CRYSTAL TRIBE) - 20 ARCHETYPES
# ------------------------------------------------------------------------------
CRYSTAL_ROSTER: List[Dict[str, Any]] = [
    # Tier 1: Initiates & Scouts
    _make_char("c_scout_01", "Kryštálový Prieskumník", "crystal", 1, "Scout", "Aether Dagger",
               {"WS": 4, "BS": 3, "S": 3, "T": 3, "W": 3, "A": 2, "Ld": 7, "Sv": "5+"},
               {"toughness": 8, "reflexes": 20, "aim": 16, "dodge": 22, "appearance": 10, "tactics": 12, "mobility": 25},
               {"name": "Aéterový Šprint", "type": "mobility", "cost_mana": 1, "desc": "Zvýši dosah pohybu o 2 hexové polia."},
               {"accent": "#66fcf1", "aim_reticle": "cyan_crosshair", "orbital_node": "satellite_node", "panel_layout": "compact_scout"}),

    _make_char("c_cadet_02", "Spírový Kadet", "crystal", 1, "Infantry", "Crystal Shortsword",
               {"WS": 4, "BS": 4, "S": 3, "T": 3, "W": 3, "A": 1, "Ld": 7, "Sv": "5+"},
               {"toughness": 12, "reflexes": 12, "aim": 14, "dodge": 12, "appearance": 10, "tactics": 14, "mobility": 14},
               {"name": "Základný Výpad", "type": "melee", "cost_mana": 1, "desc": "Úder za 2 body poškodenia."},
               {"accent": "#66fcf1", "aim_reticle": "standard_diamond", "orbital_node": "satellite_node", "panel_layout": "standard_frontline"}),

    _make_char("c_slinger_03", "Aéterový Prakovník", "crystal", 1, "Ranged", "Resonance Sling",
               {"WS": 5, "BS": 3, "S": 2, "T": 3, "W": 3, "A": 1, "Ld": 7, "Sv": "6+"},
               {"toughness": 6, "reflexes": 18, "aim": 20, "dodge": 16, "appearance": 8, "tactics": 12, "mobility": 18},
               {"name": "Kryštálová Salva", "type": "ranged", "cost_mana": 1, "desc": "Diaľkový zásah do ramena cieľa."},
               {"accent": "#66fcf1", "aim_reticle": "long_range_pip", "orbital_node": "satellite_node", "panel_layout": "ranged_sniper"}),

    _make_char("c_shieldbearer_04", "Krištáľový Štítonoš", "crystal", 1, "Tank", "Light Quartz Pavise",
               {"WS": 4, "BS": 5, "S": 3, "T": 4, "W": 4, "A": 1, "Ld": 8, "Sv": "4+"},
               {"toughness": 18, "reflexes": 10, "aim": 10, "dodge": 8, "appearance": 12, "tactics": 18, "mobility": 10},
               {"name": "Quartzová Hradba", "type": "defense", "cost_mana": 1, "desc": "Generuje +1 Ward štít pre seba."},
               {"accent": "#66fcf1", "aim_reticle": "shield_bracket", "orbital_node": "satellite_node", "panel_layout": "heavy_ward"}),

    # Tier 2: Frontline & Mortar Specialists
    _make_char("c_phalanx_05", "Falangový Bojovník", "crystal", 2, "Frontline", "Crystal Halberd",
               {"WS": 3, "BS": 4, "S": 4, "T": 4, "W": 4, "A": 2, "Ld": 8, "Sv": "4+"},
               {"toughness": 20, "reflexes": 14, "aim": 16, "dodge": 12, "appearance": 14, "tactics": 20, "mobility": 12},
               {"name": "Falangový Blok", "type": "parry", "cost_mana": 2, "desc": "Znižuje zranenie z čelného útoku o 2."},
               {"accent": "#66fcf1", "aim_reticle": "phalanx_square", "orbital_node": "resonator_core", "panel_layout": "standard_frontline"}),

    _make_char("c_mortar_crew_06", "Spírový Moždiarnik", "crystal", 2, "Artillery", "Aether Mortar Cannon",
               {"WS": 5, "BS": 3, "S": 3, "T": 4, "W": 4, "A": 1, "Ld": 8, "Sv": "4+"},
               {"toughness": 14, "reflexes": 16, "aim": 24, "dodge": 10, "appearance": 12, "tactics": 22, "mobility": 8},
               {"name": "Strmá Plunging Salva", "type": "mortar", "cost_mana": 2, "desc": "Ignoruje nízke krytie a zasahuje plošne."},
               {"accent": "#66fcf1", "aim_reticle": "mortar_elevation_grid", "orbital_node": "resonator_core", "panel_layout": "mortar_tactical"}),

    _make_char("c_marksman_07", "Kryštálový Odstreľovač", "crystal", 2, "Sniper", "Refracted Laser Carbine",
               {"WS": 4, "BS": 2, "S": 3, "T": 3, "W": 4, "A": 1, "Ld": 8, "Sv": "4+"},
               {"toughness": 10, "reflexes": 24, "aim": 26, "dodge": 20, "appearance": 14, "tactics": 18, "mobility": 16},
               {"name": "Refrakčný Headshot", "type": "duel_ranged", "cost_mana": 2, "desc": "Cieli zónu HEAD s 85% presnosťou."},
               {"accent": "#66fcf1", "aim_reticle": "sniper_cross_zone", "orbital_node": "resonator_core", "panel_layout": "ranged_sniper"}),

    _make_char("c_mendweaver_08", "Aéterový Tkáč Života", "crystal", 2, "Healer", "Prismatic Scepter",
               {"WS": 4, "BS": 3, "S": 3, "T": 3, "W": 4, "A": 1, "Ld": 8, "Sv": "4+"},
               {"toughness": 12, "reflexes": 18, "aim": 16, "dodge": 16, "appearance": 20, "tactics": 18, "mobility": 14},
               {"name": "Prizmatické Tkanie", "type": "heal", "cost_mana": 2, "desc": "Obnoví 2 HP (zarovnané na limit 6)."},
               {"accent": "#66fcf1", "aim_reticle": "healing_ring", "orbital_node": "resonator_core", "panel_layout": "support_healer"}),

    # Tier 3: Specialists & Spellblades
    _make_char("c_spellblade_09", "Aéterový Čepeliar", "crystal", 3, "Duelist", "Twin Crystal Blades",
               {"WS": 3, "BS": 3, "S": 4, "T": 4, "W": 5, "A": 3, "Ld": 8, "Sv": "3+"},
               {"toughness": 18, "reflexes": 26, "aim": 22, "dodge": 24, "appearance": 22, "tactics": 24, "mobility": 22},
               {"name": "Kadencia Dvojitých Čepelí", "type": "cadence", "cost_mana": 2, "desc": "Spúšťa dopamínový lúč a 2 rýchle seky."},
               {"accent": "#66fcf1", "aim_reticle": "duel_zone_four", "orbital_node": "orbital_duo", "panel_layout": "duel_master"}),

    _make_char("c_astromancer_10", "Astromant Zverokruhu", "crystal", 3, "Mage", "Celestite Armillary",
               {"WS": 4, "BS": 3, "S": 3, "T": 3, "W": 5, "A": 2, "Ld": 9, "Sv": "4+"},
               {"toughness": 10, "reflexes": 22, "aim": 24, "dodge": 18, "appearance": 28, "tactics": 22, "mobility": 14},
               {"name": "Orbitálny Gravitačný Príťah", "type": "orbital_spell", "cost_mana": 3, "desc": "Aktivuje 3 rotujúce satelity."},
               {"accent": "#66fcf1", "aim_reticle": "orbital_compass", "orbital_node": "orbital_duo", "panel_layout": "orbital_celestial"}),

    _make_char("c_infiltrator_11", "Kryštálový Infiltrátor", "crystal", 3, "Assassin", "Vibrational Glass Kunai",
               {"WS": 3, "BS": 2, "S": 4, "T": 3, "W": 4, "A": 4, "Ld": 8, "Sv": "4+"},
               {"toughness": 10, "reflexes": 30, "aim": 26, "dodge": 30, "appearance": 16, "tactics": 20, "mobility": 32},
               {"name": "Skrytý Úder v Tieni", "type": "backstab", "cost_mana": 2, "desc": "Ignoruje brnenie v zóne chrbta."},
               {"accent": "#66fcf1", "aim_reticle": "shadow_triad", "orbital_node": "orbital_duo", "panel_layout": "duel_master"}),

    _make_char("c_fortress_guard_12", "Strážca Pevnosti", "crystal", 3, "Defender", "Heavy Bastion Shield",
               {"WS": 3, "BS": 4, "S": 4, "T": 5, "W": 5, "A": 2, "Ld": 9, "Sv": "2+"},
               {"toughness": 30, "reflexes": 14, "aim": 14, "dodge": 8, "appearance": 18, "tactics": 28, "mobility": 8},
               {"name": "Nezlomná Pevnosť", "type": "soak", "cost_mana": 2, "desc": "Znižuje všetky prichádzajúce zranenia o 2."},
               {"accent": "#66fcf1", "aim_reticle": "heavy_shield_lock", "orbital_node": "orbital_duo", "panel_layout": "heavy_ward"}),

    # Tier 4: Elite Champions & Battlemages
    _make_char("c_inquisitor_13", "Veľký Kryštálový Inkvizítor", "crystal", 4, "Elite Duelist", "Inquisition Greatsword",
               {"WS": 2, "BS": 3, "S": 5, "T": 4, "W": 5, "A": 3, "Ld": 9, "Sv": "3+"},
               {"toughness": 22, "reflexes": 24, "aim": 28, "dodge": 22, "appearance": 34, "tactics": 28, "mobility": 18},
               {"name": "Súd Čistého Svetla", "type": "duel_execution", "cost_mana": 3, "desc": "Vystupovanie znižuje taktiku súpera na 0."},
               {"accent": "#66fcf1", "aim_reticle": "inquisition_cross", "orbital_node": "triad_resonance", "panel_layout": "duel_master"}),

    _make_char("c_siege_golem_14", "Rezonančný Obliehací Golem", "crystal", 4, "Siege Construct", "Hydraulic Piston Fists",
               {"WS": 3, "BS": 4, "S": 6, "T": 6, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
               {"toughness": 35, "reflexes": 10, "aim": 16, "dodge": 6, "appearance": 24, "tactics": 18, "mobility": 10},
               {"name": "Seizmický Drvič", "type": "ground_slam", "cost_mana": 3, "desc": "Rozbíja krytie v okruhu 2 polí."},
               {"accent": "#66fcf1", "aim_reticle": "golem_slam_zone", "orbital_node": "triad_resonance", "panel_layout": "heavy_ward"}),

    _make_char("c_chrono_mage_15", "Časomág Spíry", "crystal", 4, "Controller", "Temporal Chronometer",
               {"WS": 4, "BS": 3, "S": 3, "T": 4, "W": 5, "A": 2, "Ld": 9, "Sv": "3+"},
               {"toughness": 14, "reflexes": 32, "aim": 24, "dodge": 28, "appearance": 26, "tactics": 28, "mobility": 24},
               {"name": "Časová Dilation Slučka", "type": "time_slow", "cost_mana": 3, "desc": "Aktivuje Bullet Time na 2 kolá."},
               {"accent": "#66fcf1", "aim_reticle": "chrono_dial", "orbital_node": "triad_resonance", "panel_layout": "orbital_celestial"}),

    _make_char("c_artillery_marshal_16", "Maršál Moždiarovej Batérie", "crystal", 4, "Commander", "Twin Heavy Battery",
               {"WS": 3, "BS": 2, "S": 4, "T": 4, "W": 5, "A": 2, "Ld": 9, "Sv": "3+"},
               {"toughness": 20, "reflexes": 20, "aim": 32, "dodge": 14, "appearance": 28, "tactics": 32, "mobility": 12},
               {"name": "Koordinovaná Baráž", "type": "battery_burst", "cost_mana": 4, "desc": "Vystrelí 4 granáty s minimálnym rozptylom."},
               {"accent": "#66fcf1", "aim_reticle": "artillery_command_grid", "orbital_node": "triad_resonance", "panel_layout": "mortar_tactical"}),

    # Tier 5: Apex Sovereigns & Avatars
    _make_char("c_archon_17", "Kryštálový Archón", "crystal", 5, "Apex Champion", "Aether Sunderer Blade",
               {"WS": 2, "BS": 2, "S": 5, "T": 5, "W": 6, "A": 4, "Ld": 10, "Sv": "2+"},
               {"toughness": 28, "reflexes": 28, "aim": 30, "dodge": 26, "appearance": 38, "tactics": 34, "mobility": 20},
               {"name": "Sunder Aéteru", "type": "apex_strike", "cost_mana": 4, "desc": "Devastačný zásah za 5 DMG ignorujúci štíty."},
               {"accent": "#66fcf1", "aim_reticle": "archon_sovereign_ring", "orbital_node": "celestial_tetractys", "panel_layout": "apex_control"}),

    _make_char("c_celestial_sovereign_18", "Zvrchovaný Astromant", "crystal", 5, "Apex Mage", "Cosmic Core Orb",
               {"WS": 3, "BS": 2, "S": 4, "T": 5, "W": 6, "A": 3, "Ld": 10, "Sv": "3+"},
               {"toughness": 22, "reflexes": 34, "aim": 32, "dodge": 30, "appearance": 40, "tactics": 32, "mobility": 22},
               {"name": "Kozmická Konjunkcia", "type": "planetary_orbit", "cost_mana": 5, "desc": "Vyvolá 4 obiehajúce planéty s harmóniou 1:2:3:5."},
               {"accent": "#66fcf1", "aim_reticle": "celestial_quad_orbit", "orbital_node": "celestial_tetractys", "panel_layout": "orbital_celestial"}),

    _make_char("c_titan_colossus_19", "Aéterový Kolos Titán", "crystal", 5, "Apex Colossus", "Colossal Aether Cannons",
               {"WS": 3, "BS": 2, "S": 7, "T": 6, "W": 6, "A": 4, "Ld": 10, "Sv": "2+"},
               {"toughness": 40, "reflexes": 15, "aim": 26, "dodge": 10, "appearance": 35, "tactics": 30, "mobility": 10},
               {"name": "Planetárny Drvič", "type": "colossus_blast", "cost_mana": 5, "desc": "Zmetie celú líniu hexov."},
               {"accent": "#66fcf1", "aim_reticle": "colossus_wide_arc", "orbital_node": "celestial_tetractys", "panel_layout": "heavy_ward"}),

    _make_char("c_grand_patriarch_20", "Prastarý Patriarcha Spíry", "crystal", 5, "Supreme Leader", "Staff of Prime Aether",
               {"WS": 3, "BS": 3, "S": 4, "T": 5, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
               {"toughness": 26, "reflexes": 30, "aim": 28, "dodge": 28, "appearance": 45, "tactics": 40, "mobility": 18},
               {"name": "Absolútna Rezolúcia Spíry", "type": "supreme_command", "cost_mana": 5, "desc": "Doplní plnú manu a Ward všetkým spojencom."},
               {"accent": "#66fcf1", "aim_reticle": "prime_aether_seal", "orbital_node": "celestial_tetractys", "panel_layout": "apex_control"})
]

# ------------------------------------------------------------------------------
# 2. JEDOVATÝ KMEŇ (TOXIC TRIBE) - 20 ARCHETYPES
# ------------------------------------------------------------------------------
TOXIC_ROSTER: List[Dict[str, Any]] = [
    # Tier 1
    _make_char("t_skulker_01", "Morový Plíživec", "toxic", 1, "Scout", "Acid Dagger",
               {"WS": 4, "BS": 4, "S": 3, "T": 3, "W": 3, "A": 2, "Ld": 6, "Sv": "6+"},
               {"toughness": 10, "reflexes": 22, "aim": 16, "dodge": 24, "appearance": 8, "tactics": 12, "mobility": 26},
               {"name": "Kyselý Výskok", "type": "mobility", "cost_mana": 1, "desc": "Preskočí prekážku a otrávi terén."},
               {"accent": "#8a2be2", "aim_reticle": "toxic_cross", "orbital_node": "miasma_bubble", "panel_layout": "compact_scout"}),

    _make_char("t_spitter_02", "Žieravý Pľuvač", "toxic", 1, "Ranged", "Caustic Bile Gland",
               {"WS": 5, "BS": 3, "S": 2, "T": 3, "W": 3, "A": 1, "Ld": 6, "Sv": "6+"},
               {"toughness": 8, "reflexes": 16, "aim": 20, "dodge": 18, "appearance": 10, "tactics": 10, "mobility": 18},
               {"name": "Žieravé Pľuvnutie", "type": "ranged", "cost_mana": 1, "desc": "Aplikuje 1 stack kyseliny."},
               {"accent": "#8a2be2", "aim_reticle": "droplet_aim", "orbital_node": "miasma_bubble", "panel_layout": "ranged_sniper"}),

    _make_char("t_slugger_03", "Bahenný Pešiak", "toxic", 1, "Infantry", "Rusty Cleaver",
               {"WS": 4, "BS": 4, "S": 3, "T": 4, "W": 3, "A": 1, "Ld": 7, "Sv": "5+"},
               {"toughness": 16, "reflexes": 10, "aim": 14, "dodge": 10, "appearance": 10, "tactics": 14, "mobility": 12},
               {"name": "Hnilobný Sek", "type": "melee", "cost_mana": 1, "desc": "Úder za 2 poškodenia."},
               {"accent": "#8a2be2", "aim_reticle": "cleaver_slash", "orbital_node": "miasma_bubble", "panel_layout": "standard_frontline"}),

    _make_char("t_censer_initiate_04", "Učeň Kadidelnice", "toxic", 1, "Support", "Mini-Censer",
               {"WS": 4, "BS": 4, "S": 3, "T": 3, "W": 3, "A": 1, "Ld": 7, "Sv": "5+"},
               {"toughness": 12, "reflexes": 14, "aim": 12, "dodge": 14, "appearance": 14, "tactics": 16, "mobility": 16},
               {"name": "Opar Zmätku", "type": "debuff", "cost_mana": 1, "desc": "Znižuje presnosť súpera o 10%."},
               {"accent": "#8a2be2", "aim_reticle": "vapor_cone", "orbital_node": "miasma_bubble", "panel_layout": "support_healer"}),

    # Tier 2
    _make_char("t_hound_05", "Morový Chrt", "toxic", 2, "Beast Skirmisher", "Infected Fangs",
               {"WS": 3, "BS": 5, "S": 4, "T": 4, "W": 4, "A": 3, "Ld": 7, "Sv": "5+"},
               {"toughness": 14, "reflexes": 26, "aim": 18, "dodge": 26, "appearance": 14, "tactics": 16, "mobility": 32},
               {"name": "Zúrivý Výpad", "type": "charge", "cost_mana": 2, "desc": "Rýchly prepad na cieľ s +1 zásahom."},
               {"accent": "#8a2be2", "aim_reticle": "fang_snarl", "orbital_node": "fume_swirl", "panel_layout": "compact_scout"}),

    _make_char("t_mortar_launcher_06", "Slizový Hádzač", "toxic", 2, "Artillery", "Caustic Mortar Barrel",
               {"WS": 5, "BS": 3, "S": 3, "T": 4, "W": 4, "A": 1, "Ld": 7, "Sv": "4+"},
               {"toughness": 16, "reflexes": 14, "aim": 24, "dodge": 12, "appearance": 12, "tactics": 20, "mobility": 10},
               {"name": "Slizová Mína", "type": "mortar", "cost_mana": 2, "desc": "Vytvorí na cieli pole toxického slizu."},
               {"accent": "#8a2be2", "aim_reticle": "caustic_lob_arc", "orbital_node": "fume_swirl", "panel_layout": "mortar_tactical"}),

    _make_char("t_corrosive_guard_07", "Korozívny Strážca", "toxic", 2, "Tank", "Rust-Etched Bulwark",
               {"WS": 4, "BS": 4, "S": 4, "T": 5, "W": 4, "A": 2, "Ld": 8, "Sv": "4+"},
               {"toughness": 24, "reflexes": 12, "aim": 12, "dodge": 10, "appearance": 16, "tactics": 22, "mobility": 10},
               {"name": "Kyselý Odraz", "type": "reflect", "cost_mana": 2, "desc": "Odráža 1 bod utrpeného poškodenia útočníkovi."},
               {"accent": "#8a2be2", "aim_reticle": "corrosive_buckler", "orbital_node": "fume_swirl", "panel_layout": "heavy_ward"}),

    _make_char("t_alchemical_purifier_08", "Toxický Alchymista", "toxic", 2, "Healer/Buffer", "Acid Injector Staff",
               {"WS": 4, "BS": 3, "S": 3, "T": 4, "W": 4, "A": 1, "Ld": 8, "Sv": "4+"},
               {"toughness": 16, "reflexes": 16, "aim": 18, "dodge": 14, "appearance": 18, "tactics": 20, "mobility": 14},
               {"name": "Toxický Balzam", "type": "heal", "cost_mana": 2, "desc": "Regeneruje 2 HP za cenu slabého DoT."},
               {"accent": "#8a2be2", "aim_reticle": "injector_cross", "orbital_node": "fume_swirl", "panel_layout": "support_healer"}),

    # Tier 3
    _make_char("t_censer_bearer_09", "Nosič Toxickej Kadidelnice", "toxic", 3, "Duelist", "Heavy Toxic Censer Flail",
               {"WS": 3, "BS": 4, "S": 4, "T": 5, "W": 5, "A": 3, "Ld": 8, "Sv": "3+"},
               {"toughness": 24, "reflexes": 18, "aim": 20, "dodge": 20, "appearance": 26, "tactics": 24, "mobility": 16},
               {"name": "Cenzerový Korbáč", "type": "duel_melee", "cost_mana": 2, "desc": "Parírovanie rozprašuje morový oblak."},
               {"accent": "#8a2be2", "aim_reticle": "censer_arc", "orbital_node": "twin_rot_orbs", "panel_layout": "duel_master"}),

    _make_char("t_blight_sniper_10", "Záhubový Odstreľovač", "toxic", 3, "Sniper", "Needle Gun with Neurotoxin",
               {"WS": 4, "BS": 2, "S": 3, "T": 3, "W": 4, "A": 2, "Ld": 8, "Sv": "4+"},
               {"toughness": 10, "reflexes": 28, "aim": 30, "dodge": 24, "appearance": 16, "tactics": 22, "mobility": 20},
               {"name": "Neurotoxický Zásah", "type": "sniper", "cost_mana": 3, "desc": "Zasahuje HEAD a omráči cieľ na 1 kolo."},
               {"accent": "#8a2be2", "aim_reticle": "needle_target", "orbital_node": "twin_rot_orbs", "panel_layout": "ranged_sniper"}),

    _make_char("t_bio_brute_11", "Mutovaný Bio-Násilník", "toxic", 3, "Bruiser", "Bone Spikes & Claws",
               {"WS": 3, "BS": 5, "S": 5, "T": 5, "W": 5, "A": 3, "Ld": 8, "Sv": "4+"},
               {"toughness": 28, "reflexes": 14, "aim": 16, "dodge": 12, "appearance": 24, "tactics": 16, "mobility": 14},
               {"name": "Kostné Bodáky", "type": "brawl", "cost_mana": 2, "desc": "Spôsobuje krvácanie a prieraznosť +1 AP."},
               {"accent": "#8a2be2", "aim_reticle": "spiky_cross", "orbital_node": "twin_rot_orbs", "panel_layout": "standard_frontline"}),

    _make_char("t_venom_mage_12", "Mág Čierneho Slizu", "toxic", 3, "Mage", "Venom Scepter",
               {"WS": 4, "BS": 3, "S": 3, "T": 4, "W": 5, "A": 2, "Ld": 8, "Sv": "4+"},
               {"toughness": 16, "reflexes": 20, "aim": 24, "dodge": 18, "appearance": 26, "tactics": 22, "mobility": 16},
               {"name": "Čierna Žieravina", "type": "spell", "cost_mana": 3, "desc": "Rozpúšťa Ward štít a brnenie cieľa."},
               {"accent": "#8a2be2", "aim_reticle": "toxic_hexagram", "orbital_node": "twin_rot_orbs", "panel_layout": "orbital_celestial"}),

    # Tier 4
    _make_char("t_warmaster_13", "Bojový Majster Moru", "toxic", 4, "Warmaster", "Plague Halberd",
               {"WS": 2, "BS": 3, "S": 5, "T": 5, "W": 5, "A": 4, "Ld": 9, "Sv": "3+"},
               {"toughness": 30, "reflexes": 20, "aim": 24, "dodge": 20, "appearance": 32, "tactics": 30, "mobility": 16},
               {"name": "Vlna Rozkladu", "type": "cleave", "cost_mana": 3, "desc": "Seká až 3 ciele v prednom oblúku."},
               {"accent": "#8a2be2", "aim_reticle": "plague_halberd_arc", "orbital_node": "noxious_triad", "panel_layout": "duel_master"}),

    _make_char("t_siege_slimer_14", "Ťažký Katapultovací Slizák", "toxic", 4, "Siege Engine", "Dual Bile Lobbers",
               {"WS": 4, "BS": 2, "S": 5, "T": 6, "W": 6, "A": 2, "Ld": 9, "Sv": "3+"},
               {"toughness": 32, "reflexes": 14, "aim": 28, "dodge": 10, "appearance": 26, "tactics": 28, "mobility": 8},
               {"name": "Katastrofický Dážď Kyseliny", "type": "mortar_barrage", "cost_mana": 4, "desc": "Masívna 3-ranová salva s toxickým spádom."},
               {"accent": "#8a2be2", "aim_reticle": "heavy_bile_grid", "orbital_node": "noxious_triad", "panel_layout": "mortar_tactical"}),

    _make_char("t_flesh_abomination_15", "Ohavnosť z Tiel", "toxic", 4, "Monstrosity", "Fleshy Appendages",
               {"WS": 3, "BS": 5, "S": 6, "T": 6, "W": 6, "A": 4, "Ld": 9, "Sv": "3+"},
               {"toughness": 36, "reflexes": 12, "aim": 16, "dodge": 10, "appearance": 34, "tactics": 18, "mobility": 12},
               {"name": "Pohltenie a Regenerácia", "type": "vampiric", "cost_mana": 3, "desc": "Vysaje 2 HP z porazeného cieľa."},
               {"accent": "#8a2be2", "aim_reticle": "maw_circle", "orbital_node": "noxious_triad", "panel_layout": "heavy_ward"}),

    _make_char("t_corrosive_assassin_16", "Tieňový Toxický Vrah", "toxic", 4, "Master Assassin", "Poison Needle & Garrote",
               {"WS": 2, "BS": 2, "S": 4, "T": 4, "W": 5, "A": 4, "Ld": 9, "Sv": "3+"},
               {"toughness": 16, "reflexes": 34, "aim": 32, "dodge": 32, "appearance": 24, "tactics": 26, "mobility": 32},
               {"name": "Smrtiaci Zásek do Hrdla", "type": "duel_execute", "cost_mana": 3, "desc": "3.0x kritický násobiteľ v zóne HEAD."},
               {"accent": "#8a2be2", "aim_reticle": "assassin_dual_x", "orbital_node": "noxious_triad", "panel_layout": "duel_master"}),

    # Tier 5
    _make_char("t_defiler_17", "Toxický Hnilobník", "toxic", 5, "Apex Champion", "Corrosive Cleaver & Censer",
               {"WS": 2, "BS": 2, "S": 5, "T": 6, "W": 6, "A": 4, "Ld": 10, "Sv": "2+"},
               {"toughness": 34, "reflexes": 22, "aim": 28, "dodge": 22, "appearance": 40, "tactics": 32, "mobility": 18},
               {"name": "Morová Epidémia", "type": "apex_strike", "cost_mana": 4, "desc": "Zamorí všetky nepriateľské jednotky."},
               {"accent": "#8a2be2", "aim_reticle": "defiler_pentagram", "orbital_node": "quad_decay_spheres", "panel_layout": "apex_control"}),

    _make_char("t_plague_monarch_18", "Monarcha Rozkladu", "toxic", 5, "Apex Mage", "Staff of Putrefaction",
               {"WS": 3, "BS": 2, "S": 4, "T": 6, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
               {"toughness": 30, "reflexes": 28, "aim": 30, "dodge": 24, "appearance": 42, "tactics": 36, "mobility": 18},
               {"name": "Kyselý Kataklizmus", "type": "apex_magic", "cost_mana": 5, "desc": "Premení 5 hexových polí na kyselinové jazero."},
               {"accent": "#8a2be2", "aim_reticle": "monarch_crown_sigil", "orbital_node": "quad_decay_spheres", "panel_layout": "orbital_celestial"}),

    _make_char("t_bio_titan_19", "Mutagénny Titán", "toxic", 5, "Apex Colossus", "Colossal Bone Claws",
               {"WS": 2, "BS": 3, "S": 8, "T": 7, "W": 6, "A": 4, "Ld": 10, "Sv": "2+"},
               {"toughness": 42, "reflexes": 14, "aim": 24, "dodge": 10, "appearance": 38, "tactics": 26, "mobility": 12},
               {"name": "Titánska Pôdna Kontaminácia", "type": "apex_colossus", "cost_mana": 5, "desc": "Zasiahne celú arénu shockwave vlnou."},
               {"accent": "#8a2be2", "aim_reticle": "titan_hazard_ring", "orbital_node": "quad_decay_spheres", "panel_layout": "heavy_ward"}),

    _make_char("t_sovereign_alchemist_20", "Zvrchovaný Primus Alchýmie", "toxic", 5, "Supreme Leader", "Vial of Ultimate Solvent",
               {"WS": 3, "BS": 2, "S": 4, "T": 6, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
               {"toughness": 32, "reflexes": 30, "aim": 32, "dodge": 26, "appearance": 46, "tactics": 38, "mobility": 20},
               {"name": "Univerzálne Rozpúšťadlo", "type": "supreme_dissolve", "cost_mana": 5, "desc": "Úplne vymaže všetky nepriateľské štíty a opevnenia."},
               {"accent": "#8a2be2", "aim_reticle": "alchemist_ouroboros", "orbital_node": "quad_decay_spheres", "panel_layout": "apex_control"})
]

# ------------------------------------------------------------------------------
# 3. PRASTARÝ DRUIDSKÝ KMEŇ (DRUID TRIBE) - 20 ARCHETYPES
# ------------------------------------------------------------------------------
DRUID_ROSTER: List[Dict[str, Any]] = [
    # Tier 1
    _make_char("d_acorn_scout_01", "Žaluďový Prieskumník", "druid", 1, "Scout", "Yew Shortbow",
               {"WS": 4, "BS": 3, "S": 3, "T": 3, "W": 3, "A": 2, "Ld": 8, "Sv": "5+"},
               {"toughness": 10, "reflexes": 22, "aim": 18, "dodge": 24, "appearance": 10, "tactics": 14, "mobility": 26},
               {"name": "Lesný Krok", "type": "mobility", "cost_mana": 1, "desc": "Ignoruje pohybový postih v lesnom teréne."},
               {"accent": "#b8860b", "aim_reticle": "oak_leaf_reticle", "orbital_node": "spore_satellite", "panel_layout": "compact_scout"}),

    _make_char("d_wood_cadet_02", "Učeň Hvozdu", "druid", 1, "Infantry", "Living Wood Spear",
               {"WS": 4, "BS": 4, "S": 3, "T": 4, "W": 3, "A": 1, "Ld": 8, "Sv": "5+"},
               {"toughness": 14, "reflexes": 12, "aim": 14, "dodge": 14, "appearance": 12, "tactics": 16, "mobility": 14},
               {"name": "Bodnutie Vetvou", "type": "melee", "cost_mana": 1, "desc": "Úder za 2 poškodenia."},
               {"accent": "#b8860b", "aim_reticle": "spear_thrust", "orbital_node": "spore_satellite", "panel_layout": "standard_frontline"}),

    _make_char("d_root_trapper_03", "Koreňový Pasciar", "druid", 1, "Controller", "Bramble Slingshot",
               {"WS": 5, "BS": 3, "S": 2, "T": 3, "W": 3, "A": 1, "Ld": 8, "Sv": "6+"},
               {"toughness": 8, "reflexes": 18, "aim": 18, "dodge": 18, "appearance": 10, "tactics": 18, "mobility": 18},
               {"name": "Trávové Putá", "type": "root", "cost_mana": 1, "desc": "Znehybní cieľ na 1 kolo."},
               {"accent": "#b8860b", "aim_reticle": "root_tangle_zone", "orbital_node": "spore_satellite", "panel_layout": "ranged_sniper"}),

    _make_char("d_bark_tender_04", "Ošetrovateľ Kôry", "druid", 1, "Healer", "Herbal Pouch",
               {"WS": 4, "BS": 4, "S": 3, "T": 3, "W": 3, "A": 1, "Ld": 8, "Sv": "5+"},
               {"toughness": 12, "reflexes": 14, "aim": 12, "dodge": 16, "appearance": 16, "tactics": 18, "mobility": 14},
               {"name": "Bylinný Zápar", "type": "heal", "cost_mana": 1, "desc": "Vylieči 1 HP spojencovi."},
               {"accent": "#b8860b", "aim_reticle": "leaf_healing", "orbital_node": "spore_satellite", "panel_layout": "support_healer"}),

    # Tier 2
    _make_char("d_thorn_warden_05", "Tŕňový Strážca", "druid", 2, "Frontline", "Bramble Morningstar",
               {"WS": 3, "BS": 4, "S": 4, "T": 4, "W": 4, "A": 2, "Ld": 8, "Sv": "4+"},
               {"toughness": 22, "reflexes": 14, "aim": 16, "dodge": 16, "appearance": 14, "tactics": 22, "mobility": 14},
               {"name": "Tŕňový Odraz", "type": "parry", "cost_mana": 2, "desc": "Odráža 50% kontaktného poškodenia."},
               {"accent": "#b8860b", "aim_reticle": "bramble_cross", "orbital_node": "amber_orb", "panel_layout": "standard_frontline"}),

    _make_char("d_earthen_lobber_06", "Hádzač Hlinených Búrok", "druid", 2, "Artillery", "Earth Trebuchet",
               {"WS": 5, "BS": 3, "S": 4, "T": 4, "W": 4, "A": 1, "Ld": 8, "Sv": "4+"},
               {"toughness": 18, "reflexes": 12, "aim": 24, "dodge": 10, "appearance": 12, "tactics": 20, "mobility": 10},
               {"name": "Kamenitá Húfnica", "type": "mortar", "cost_mana": 2, "desc": "Balistický úder drviaci nízke krytie."},
               {"accent": "#b8860b", "aim_reticle": "rock_drop_grid", "orbital_node": "amber_orb", "panel_layout": "mortar_tactical"}),

    _make_char("d_canopy_archer_07", "Korunový Lukostrelec", "druid", 2, "Sniper", "Ancient Longbow",
               {"WS": 4, "BS": 2, "S": 3, "T": 3, "W": 4, "A": 2, "Ld": 8, "Sv": "4+"},
               {"toughness": 10, "reflexes": 26, "aim": 28, "dodge": 22, "appearance": 14, "tactics": 20, "mobility": 20},
               {"name": "Zásah z Koruny", "type": "sniper", "cost_mana": 2, "desc": "+2 k poškodeniu pri streľbe z vyvýšeného hvozdu."},
               {"accent": "#b8860b", "aim_reticle": "canopy_cross", "orbital_node": "amber_orb", "panel_layout": "ranged_sniper"}),

    _make_char("d_grove_vitalist_08", "Vitalista Hvozdu", "druid", 2, "Healer", "Oak Staff with Dew",
               {"WS": 4, "BS": 3, "S": 3, "T": 4, "W": 4, "A": 1, "Ld": 9, "Sv": "4+"},
               {"toughness": 14, "reflexes": 18, "aim": 16, "dodge": 18, "appearance": 22, "tactics": 22, "mobility": 16},
               {"name": "Prameň Života", "type": "heal", "cost_mana": 2, "desc": "Lieči 2 HP cieľa a pridáva +1 Ward."},
               {"accent": "#b8860b", "aim_reticle": "dew_bloom", "orbital_node": "amber_orb", "panel_layout": "support_healer"}),

    # Tier 3
    _make_char("d_bear_shifter_09", "Menic Medveďa", "druid", 3, "Bruiser", "Ursine Paws",
               {"WS": 3, "BS": 5, "S": 5, "T": 5, "W": 5, "A": 3, "Ld": 9, "Sv": "3+"},
               {"toughness": 30, "reflexes": 14, "aim": 18, "dodge": 12, "appearance": 24, "tactics": 20, "mobility": 14},
               {"name": "Medvedí Rev a Drvenie", "type": "brawl", "cost_mana": 2, "desc": "Zastraší súpera a udelí 3 body poškodenia."},
               {"accent": "#b8860b", "aim_reticle": "bear_claw_arc", "orbital_node": "triad_sap_runes", "panel_layout": "heavy_ward"}),

    _make_char("d_spirit_wolf_10", "Duchovný Vlk", "druid", 3, "Beast Duelist", "Spectral Fangs",
               {"WS": 2, "BS": 4, "S": 4, "T": 4, "W": 5, "A": 4, "Ld": 9, "Sv": "4+"},
               {"toughness": 14, "reflexes": 30, "aim": 26, "dodge": 32, "appearance": 20, "tactics": 26, "mobility": 35},
               {"name": "Útok Prvej Voľby", "type": "first_strike", "cost_mana": 2, "desc": "Útočí vždy ako prvý bez ohľadu na iniciatívu."},
               {"accent": "#b8860b", "aim_reticle": "wolf_instinct", "orbital_node": "triad_sap_runes", "panel_layout": "duel_master"}),

    _make_char("d_ent_sapling_11", "Mladý Lesný Ent", "druid", 3, "Tank Construct", "Living Trunk",
               {"WS": 3, "BS": 5, "S": 5, "T": 6, "W": 5, "A": 2, "Ld": 9, "Sv": "2+"},
               {"toughness": 34, "reflexes": 10, "aim": 14, "dodge": 8, "appearance": 22, "tactics": 24, "mobility": 8},
               {"name": "Koreňové Zakotvenie", "type": "anchor", "cost_mana": 2, "desc": "Imúnny voči posunutiu a znižuje poškodenie o 2."},
               {"accent": "#b8860b", "aim_reticle": "tree_rings_lock", "orbital_node": "triad_sap_runes", "panel_layout": "heavy_ward"}),

    _make_char("d_storm_druid_12", "Búrlivý Druid", "druid", 3, "Mage", "Charged Lightning Rod",
               {"WS": 4, "BS": 2, "S": 3, "T": 4, "W": 5, "A": 2, "Ld": 9, "Sv": "4+"},
               {"toughness": 14, "reflexes": 26, "aim": 28, "dodge": 22, "appearance": 26, "tactics": 22, "mobility": 18},
               {"name": "Bleskový Úder Hromu", "type": "spell", "cost_mana": 3, "desc": "Zasiahne cieľ s ignorovaním brnenia."},
               {"accent": "#b8860b", "aim_reticle": "lightning_fork", "orbital_node": "triad_sap_runes", "panel_layout": "orbital_celestial"}),

    # Tier 4
    _make_char("d_arch_ranger_13", "Veľký Hraničiar Hvozdu", "druid", 4, "Elite Duelist", "Composite Greatbow & Scimitar",
               {"WS": 2, "BS": 2, "S": 4, "T": 4, "W": 5, "A": 3, "Ld": 9, "Sv": "3+"},
               {"toughness": 20, "reflexes": 32, "aim": 32, "dodge": 30, "appearance": 28, "tactics": 30, "mobility": 26},
               {"name": "Majstrovský Dvojstrel", "type": "duel_combo", "cost_mana": 3, "desc": "Vystrelí dva šípy do rôznych zón (HEAD + TORSO)."},
               {"accent": "#b8860b", "aim_reticle": "dual_flight_arc", "orbital_node": "four_leaf_bloom", "panel_layout": "duel_master"}),

    _make_char("d_ancient_treant_14", "Prastarý Dubový Ent", "druid", 4, "Siege Construct", "Massive Oak Limbs",
               {"WS": 3, "BS": 4, "S": 7, "T": 6, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
               {"toughness": 38, "reflexes": 12, "aim": 20, "dodge": 8, "appearance": 30, "tactics": 26, "mobility": 8},
               {"name": "Zemský Moždiarový Dupot", "type": "earth_mortar", "cost_mana": 3, "desc": "Vyvrhne kamenné balvany na nekryté ciele."},
               {"accent": "#b8860b", "aim_reticle": "treant_stomp_grid", "orbital_node": "four_leaf_bloom", "panel_layout": "mortar_tactical"}),

    _make_char("d_shaman_elder_15", "Šamanský Starešina", "druid", 4, "Elite Healer", "Totem of Life & Decay",
               {"WS": 4, "BS": 3, "S": 3, "T": 5, "W": 5, "A": 2, "Ld": 9, "Sv": "3+"},
               {"toughness": 20, "reflexes": 24, "aim": 24, "dodge": 22, "appearance": 34, "tactics": 30, "mobility": 18},
               {"name": "Totemický Reťazový Heal", "type": "chain_heal", "cost_mana": 3, "desc": "Vylieči až 3 spojencov o 2 HP."},
               {"accent": "#b8860b", "aim_reticle": "totem_triad_pulse", "orbital_node": "four_leaf_bloom", "panel_layout": "support_healer"}),

    _make_char("d_wild_hunt_master_16", "Pán Divokého Honu", "druid", 4, "Commander", "Horn of the Wild Hunt",
               {"WS": 2, "BS": 3, "S": 5, "T": 5, "W": 5, "A": 4, "Ld": 10, "Sv": "3+"},
               {"toughness": 26, "reflexes": 28, "aim": 26, "dodge": 28, "appearance": 36, "tactics": 32, "mobility": 28},
               {"name": "Hlas Divokého Honu", "type": "hunt_frenzy", "cost_mana": 4, "desc": "Všetky zvieratá v aréne získavajú +2 k útoku."},
               {"accent": "#b8860b", "aim_reticle": "stag_horn_compass", "orbital_node": "four_leaf_bloom", "panel_layout": "apex_control"}),

    # Tier 5
    _make_char("d_druid_elder_17", "Prastarý Druid", "druid", 5, "Apex Champion", "Elder Oak Staff & Living Vines",
               {"WS": 2, "BS": 2, "S": 4, "T": 5, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
               {"toughness": 30, "reflexes": 30, "aim": 30, "dodge": 28, "appearance": 42, "tactics": 38, "mobility": 20},
               {"name": "Hnev Matky Zeme", "type": "apex_strike", "cost_mana": 4, "desc": "Zviaže celú nepriateľskú armádu koreňmi a vylieči spojencov."},
               {"accent": "#b8860b", "aim_reticle": "world_tree_mandala", "orbital_node": "celestial_druidic_spheres", "panel_layout": "apex_control"}),

    _make_char("d_solar_archdruid_18", "Solárny Archdruid", "druid", 5, "Apex Mage", "Sunstone Orb",
               {"WS": 3, "BS": 2, "S": 4, "T": 5, "W": 6, "A": 3, "Ld": 10, "Sv": "2+"},
               {"toughness": 24, "reflexes": 34, "aim": 34, "dodge": 30, "appearance": 44, "tactics": 34, "mobility": 22},
               {"name": "Solárna Supernova", "type": "apex_solar", "cost_mana": 5, "desc": "Vyvolá solárny výbuch so spaľujúcim aéterom."},
               {"accent": "#b8860b", "aim_reticle": "solar_flame_ring", "orbital_node": "celestial_druidic_spheres", "panel_layout": "orbital_celestial"}),

    _make_char("d_world_tree_colossus_19", "Kolos Svetového Stromu", "druid", 5, "Apex Colossus", "Colossal Bramble Club",
               {"WS": 3, "BS": 3, "S": 8, "T": 7, "W": 6, "A": 4, "Ld": 10, "Sv": "2+"},
               {"toughness": 44, "reflexes": 12, "aim": 22, "dodge": 8, "appearance": 36, "tactics": 28, "mobility": 8},
               {"name": "Drtivý Pád Kmeňa", "type": "apex_colossus", "cost_mana": 5, "desc": "Zničí všetky nepriateľské štíty a budovy v dosahu."},
               {"accent": "#b8860b", "aim_reticle": "world_trunk_smash", "orbital_node": "celestial_druidic_spheres", "panel_layout": "heavy_ward"}),

    _make_char("d_avatar_of_gaia_20", "Avatar Života a Rovnováhy", "druid", 5, "Supreme Leader", "Heart of the Primeval Forest",
               {"WS": 2, "BS": 2, "S": 5, "T": 6, "W": 6, "A": 4, "Ld": 10, "Sv": "2+"},
               {"toughness": 36, "reflexes": 32, "aim": 32, "dodge": 30, "appearance": 48, "tactics": 42, "mobility": 24},
               {"name": "Harmonická Obnova Všehomíra", "type": "supreme_gaia", "cost_mana": 5, "desc": "Vylieči všetkých spojencov na maximum 6 HP a vytvorí nedobytný Ward."},
               {"accent": "#b8860b", "aim_reticle": "gaia_harmony_sigil", "orbital_node": "celestial_druidic_spheres", "panel_layout": "apex_control"})
]

# ------------------------------------------------------------------------------
# 4. UNIFIED 60-CHARACTER REGISTRY
# ------------------------------------------------------------------------------
ALL_SIXTY_CHARACTERS: Dict[str, Dict[str, Any]] = {
    c["id"]: c for c in (CRYSTAL_ROSTER + TOXIC_ROSTER + DRUID_ROSTER)
}

class SixtyCharacterRosterEngine:
    """
    Query, filter, and inspect the 60 canonical character archetypes
    and retrieve their specialized skill control panel configurations.
    """

    @classmethod
    def get_character(cls, char_id: str) -> Optional[Dict[str, Any]]:
        return ALL_SIXTY_CHARACTERS.get(char_id)

    @classmethod
    def get_characters_by_race(cls, race: str) -> List[Dict[str, Any]]:
        return [c for c in ALL_SIXTY_CHARACTERS.values() if c["race"] == race.lower()]

    @classmethod
    def get_characters_by_tier(cls, tier: int) -> List[Dict[str, Any]]:
        return [c for c in ALL_SIXTY_CHARACTERS.values() if c["tier"] == tier]

    @classmethod
    def get_total_roster_count(cls) -> int:
        return len(ALL_SIXTY_CHARACTERS)
