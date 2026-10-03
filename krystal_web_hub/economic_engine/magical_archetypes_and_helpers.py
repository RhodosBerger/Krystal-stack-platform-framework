# ==============================================================================
# KRYSTAL-STACK: 20 MALE & 20 FEMALE MAGICAL ARCHETYPES & 120 HELPERS CATALOG
# ==============================================================================
# Implements:
#   1. 20 Male magical representatives (Veštci, Čarodejníci, Mágovia, Kúzelníci,
#      Iluzionisti, Chladní Vystupovači, Odolávači, Ostreľovači, Šamani, Nekromanti).
#   2. 20 Female magical representatives (Veštice, Čarodejnice, Mágky, Kúzelníčky,
#      Iluzionistky, Chladné Vystupovačky, Odolávačky, Uhýbačky, Druidky, Banshee).
#   3. Complete dual-system stats:
#      - Warhammer profiles (WS, BS, S, T, W <= 6, A, Ld, Sv).
#      - The West duel attributes (Toughness, Reflexes, Aim, Dodge, Appearance, Tactics, Mobility).
#   4. 12 Canonical fantasy/sci-fi races with racial affinities and passives.
#   5. 120 Unique helpers/familiars (10 per race x 12 races = 120) with duel auras.
#   6. Continuous interactive slider blending engine (Hero stats + Slider deltas + Helper synergy).
#   7. Strict 6 Max HP vital invariant enforcement across all calculations.
# ==============================================================================

import math
from typing import Dict, List, Any, Optional, Tuple
from .the_west_duel_algebra import TheWestDuelAlgebra, DuelTargetZone, DuelDodgeStance, DuelWeaponCategory

# ── 1. TWELVE CANONICAL RACES ────────────────────────────────────────────────
RACES_CATALOG: Dict[str, Dict[str, Any]] = {
    "crystal": {
        "id": "crystal",
        "name": "Kryštálový Kmeň",
        "element": "Aéter & Svetlo",
        "icon": "💎",
        "racial_passives": "+15% Aéterová Rezonancia, +10 Reflexy proti strelám, Pasívna regenerácia štítu.",
        "synergy_bonus": {"reflexes": 8, "aim": 6, "appearance": 4},
        "color_accent": "#66fcf1"
    },
    "toxic": {
        "id": "toxic",
        "name": "Jedovatý Kmeň",
        "element": "Kyselina & Korózia",
        "icon": "🧪",
        "racial_passives": "+20% Odolnosť voči jedu, Kyselinový DoT poškodenie, +12 Húževnatosť.",
        "synergy_bonus": {"toughness": 12, "tactics": 6, "appearance": 4},
        "color_accent": "#39ff14"
    },
    "druid": {
        "id": "druid",
        "name": "Prastarí Druidi",
        "element": "Hvozd & Príroda",
        "icon": "🌿",
        "racial_passives": "+15% Mykorhízne prepojenie, Spojenie s koreňmi, +10 Taktika a liečenie.",
        "synergy_bonus": {"toughness": 8, "tactics": 10, "mobility": 6},
        "color_accent": "#55aa55"
    },
    "human": {
        "id": "human",
        "name": "Hraniční Ľudia",
        "element": "Oceľ & Pušný Prach",
        "icon": "🤠",
        "racial_passives": "+10% Univerzálnosť, Rýchle tasenie zbraní, +12 Presnosť a morálka.",
        "synergy_bonus": {"aim": 10, "reflexes": 6, "tactics": 8},
        "color_accent": "#d4af37"
    },
    "dwarf": {
        "id": "dwarf",
        "name": "Hlbinní Trpaslíci",
        "element": "Žula & Kov",
        "icon": "⚒️",
        "racial_passives": "+25% Absorpcia chladných zbraní, Pevný postoj bez knockbacku, +14 Húževnatosť.",
        "synergy_bonus": {"toughness": 14, "tactics": 8, "aim": 4},
        "color_accent": "#cd7f32"
    },
    "elf": {
        "id": "elf",
        "name": "Hviezdni Elfovia",
        "element": "Hviezdy & Kozmos",
        "icon": "✨",
        "racial_passives": "+20% Presnosť ostreľovania, Astrálna elegancia, +12 Vystupovanie a Uhýbanie.",
        "synergy_bonus": {"aim": 12, "dodge": 10, "appearance": 8},
        "color_accent": "#a8e6cf"
    },
    "infernal": {
        "id": "infernal",
        "name": "Pekelní Démoni",
        "element": "Oheň & Síra",
        "icon": "🔥",
        "racial_passives": "+20% Zastrašovanie súpera, Plamenná aura, +14 Vystupovanie a útočná sila.",
        "synergy_bonus": {"appearance": 14, "aim": 8, "toughness": 6},
        "color_accent": "#ff3333"
    },
    "celestial": {
        "id": "celestial",
        "name": "Nebeskí Anjeli",
        "element": "Sväté Svetlo",
        "icon": "🕊️",
        "racial_passives": "+20% Posvätný Ward, Žiarivá prítomnosť, Ochrana pred kritickými zásahmi.",
        "synergy_bonus": {"appearance": 12, "tactics": 10, "dodge": 6},
        "color_accent": "#ffd700"
    },
    "spectral": {
        "id": "spectral",
        "name": "Prízrační Nemŕtvi",
        "element": "Prázdnota & Chlad",
        "icon": "👻",
        "racial_passives": "+25% Fázový posun (Phase Shift), Imunita voči strachu, +14 Uhýbanie a Reflexy.",
        "synergy_bonus": {"dodge": 14, "reflexes": 10, "mobility": 6},
        "color_accent": "#7b68ee"
    },
    "elemental": {
        "id": "elemental",
        "name": "Elementáli Živlov",
        "element": "Búrka & Magma",
        "icon": "⚡",
        "racial_passives": "+15% Elementálna nestabilita, Šokové výboje pri zásahu, +10 Pohyblivosť.",
        "synergy_bonus": {"reflexes": 10, "aim": 8, "mobility": 10},
        "color_accent": "#00e5ff"
    },
    "fae": {
        "id": "fae",
        "name": "Lesné Víly",
        "element": "Preludy & Ilúzie",
        "icon": "🦋",
        "racial_passives": "+25% Zmätenie cieľa (Glamour), Zrkadlové klamy, +14 Uhýbanie a Taktika.",
        "synergy_bonus": {"dodge": 12, "tactics": 12, "appearance": 8},
        "color_accent": "#ff69b4"
    },
    "beastkin": {
        "id": "beastkin",
        "name": "Zvierací Bojovníci",
        "element": "Krv & Inštinkt",
        "icon": "🐺",
        "racial_passives": "+20% Feralny inštinkt, Bleskový výpad, +12 Pohyblivosť a Húževnatosť.",
        "synergy_bonus": {"mobility": 12, "toughness": 8, "reflexes": 8},
        "color_accent": "#e67e22"
    }
}

# ── 2. 20 MALE MAGICAL REPRESENTATIVES ───────────────────────────────────────
MALE_ARCHETYPES_20: List[Dict[str, Any]] = [
    {
        "id": "m_oracle_01",
        "name": "Veštec Hviezdnych Dráh",
        "gender": "male",
        "archetype_class": "Veštec",
        "title": "Astrological Star Seer",
        "lore": "Číta osud v konšteláciách a dráhach planét. Predvída výstrely súpera ešte pred stlačením spúšte.",
        "warhammer_stats": {"WS": 4, "BS": 6, "S": 3, "T": 4, "W": 4, "A": 2, "Ld": 9, "Sv": "5+"},
        "duel_stats": {"toughness": 14, "reflexes": 18, "aim": 34, "dodge": 16, "appearance": 18, "tactics": 26, "mobility": 15},
        "signature_ability": {"name": "Hviezdne Orákulum", "cost_mana": 2, "type": "precision", "effect": "+25% Presnosť a odhalenie zóny obrancu."},
        "favored_weapon": "Kryštálová Astrolábová Palica",
        "visual_aura": "#4169e1"
    },
    {
        "id": "m_oracle_02",
        "name": "Orákulum Zrkadlových Hlbín",
        "gender": "male",
        "archetype_class": "Veštec",
        "title": "Mirror Depth Oracle",
        "lore": "Ponára svoju myseľ do jazerných zrkadiel, kde vidí alternatívne časové línie a slabiny v taktike súpera.",
        "warhammer_stats": {"WS": 4, "BS": 5, "S": 3, "T": 4, "W": 4, "A": 2, "Ld": 9, "Sv": "5+"},
        "duel_stats": {"toughness": 16, "reflexes": 20, "aim": 22, "dodge": 24, "appearance": 15, "tactics": 32, "mobility": 12},
        "signature_ability": {"name": "Odraz Osudu", "cost_mana": 2, "type": "tactics", "effect": "Odrazí 30% poškodenia späť na útočníka."},
        "favored_weapon": "Zrkadlový Žezlový Kryštál",
        "visual_aura": "#00bfff"
    },
    {
        "id": "m_witch_03",
        "name": "Tieňový Čarodejník",
        "gender": "male",
        "archetype_class": "Čarodejník",
        "title": "Shadow Hexer",
        "lore": "Manipuluje s tieňmi a nočnou hmlou. Jeho temná prítomnosť láme odvahu protivníka.",
        "warhammer_stats": {"WS": 5, "BS": 5, "S": 4, "T": 4, "W": 5, "A": 3, "Ld": 8, "Sv": "4+"},
        "duel_stats": {"toughness": 18, "reflexes": 18, "aim": 24, "dodge": 16, "appearance": 30, "tactics": 20, "mobility": 16},
        "signature_ability": {"name": "Tieňová Kliatba", "cost_mana": 3, "type": "intimidation", "effect": "Zníži súperovu Taktiku o 15 na 2 kolá."},
        "favored_weapon": "Obsidiánová Čarodejnícka Dýka",
        "visual_aura": "#4b0082"
    },
    {
        "id": "m_witch_04",
        "name": "Rituálny Hexer Krvi",
        "gender": "male",
        "archetype_class": "Čarodejník",
        "title": "Blood Ritualist",
        "lore": "Uzatvára krvavé pakty a spevňuje svoje telo bolesťou. Znáša rany, ktoré by iných zabili.",
        "warhammer_stats": {"WS": 5, "BS": 4, "S": 5, "T": 5, "W": 6, "A": 3, "Ld": 9, "Sv": "4+"},
        "duel_stats": {"toughness": 26, "reflexes": 14, "aim": 16, "dodge": 12, "appearance": 28, "tactics": 22, "mobility": 12},
        "signature_ability": {"name": "Krvavá Pečať Odplaty", "cost_mana": 2, "type": "soak", "effect": "Prevedie 2 body utŕženého zranenia do liečenia."},
        "favored_weapon": "Rituálny Kosák a Reťaz",
        "visual_aura": "#8b0000"
    },
    {
        "id": "m_mage_05",
        "name": "Pyromantický Mág Plameňa",
        "gender": "male",
        "archetype_class": "Mág",
        "title": "Pyromancer Archmage",
        "lore": "Vládca žiarivého plameňa. Súperov spaľuje na popol skôr, než sa stihnú priblížiť.",
        "warhammer_stats": {"WS": 4, "BS": 6, "S": 4, "T": 4, "W": 5, "A": 2, "Ld": 9, "Sv": "4+"},
        "duel_stats": {"toughness": 15, "reflexes": 22, "aim": 32, "dodge": 18, "appearance": 26, "tactics": 18, "mobility": 16},
        "signature_ability": {"name": "Slnečná Supernova", "cost_mana": 3, "type": "burst", "effect": "Zásah spôsobí plošné horenie za 2 dodatočné zranenia."},
        "favored_weapon": "Plamenná Aéterová Guľa",
        "visual_aura": "#ff4500"
    },
    {
        "id": "m_mage_06",
        "name": "Kryomantický Mág Ľadových Hrotov",
        "gender": "male",
        "archetype_class": "Mág",
        "title": "Cryomancer Stalker",
        "lore": "Taktický majster absolútneho mrazu. Spomaľuje pohyby a zamŕza strelné mechanizmy protivníka.",
        "warhammer_stats": {"WS": 5, "BS": 5, "S": 4, "T": 5, "W": 5, "A": 3, "Ld": 9, "Sv": "3+"},
        "duel_stats": {"toughness": 20, "reflexes": 26, "aim": 24, "dodge": 20, "appearance": 16, "tactics": 28, "mobility": 15},
        "signature_ability": {"name": "Glaciálna Hradba", "cost_mana": 2, "type": "mitigation", "effect": "+10 k Húževnatosti a zmrazenie nepriateľského úderu."},
        "favored_weapon": "Ľadový Kryštálový Špic",
        "visual_aura": "#afeeee"
    },
    {
        "id": "m_mage_07",
        "name": "Bleskový Mág Búrok",
        "gender": "male",
        "archetype_class": "Mág",
        "title": "Storm Lightning Mage",
        "lore": "Elektrifikuje bojisko s bleskovou rýchlosťou. Zásahy do ramena paralyzujú mierenie súpera.",
        "warhammer_stats": {"WS": 4, "BS": 6, "S": 4, "T": 3, "W": 4, "A": 3, "Ld": 8, "Sv": "5+"},
        "duel_stats": {"toughness": 12, "reflexes": 28, "aim": 26, "dodge": 22, "appearance": 20, "tactics": 14, "mobility": 30},
        "signature_ability": {"name": "Reťazový Bleskový Výboj", "cost_mana": 3, "type": "speed", "effect": "Ignoruje 50% súperových Reflexov pri výpočte obrany."},
        "favored_weapon": "Búrková Rezonančná Tyč",
        "visual_aura": "#00ffff"
    },
    {
        "id": "m_mage_08",
        "name": "Aéterový Archmág Prúdenia",
        "gender": "male",
        "archetype_class": "Mág",
        "title": "Aether Arcanist",
        "lore": "Ovláda čistú esenciu kozmického aéteru. Harmonizuje priestor a rozptyľuje projektily.",
        "warhammer_stats": {"WS": 4, "BS": 6, "S": 3, "T": 4, "W": 5, "A": 2, "Ld": 10, "Sv": "3+"},
        "duel_stats": {"toughness": 14, "reflexes": 22, "aim": 28, "dodge": 18, "appearance": 30, "tactics": 26, "mobility": 18},
        "signature_ability": {"name": "Aéterové Rozpustenie Hmoty", "cost_mana": 3, "type": "disruption", "effect": "Deaktivuje nepriateľské špeciálne munície a kúzla."},
        "favored_weapon": "Zlaté Aéterové Žezlo",
        "visual_aura": "#9370db"
    },
    {
        "id": "m_wizard_09",
        "name": "Runový Kúzelník Pečatí",
        "gender": "male",
        "archetype_class": "Kúzelník",
        "title": "Runic Enchanter",
        "lore": "Vyrezáva starobylé rúnické glyfy priamo do zbroje a pokožky. Neotrasiteľný pevný obranca.",
        "warhammer_stats": {"WS": 5, "BS": 4, "S": 4, "T": 5, "W": 6, "A": 2, "Ld": 9, "Sv": "3+"},
        "duel_stats": {"toughness": 24, "reflexes": 18, "aim": 20, "dodge": 16, "appearance": 16, "tactics": 30, "mobility": 14},
        "signature_ability": {"name": "Runa Nedobytnosti", "cost_mana": 2, "type": "shield", "effect": "Poskytuje 4 body absorpčného bariérového štítu."},
        "favored_weapon": "Rúnami Kovaný Palcát",
        "visual_aura": "#d2691e"
    },
    {
        "id": "m_wizard_10",
        "name": "Spektrálny Vyvolávač",
        "gender": "male",
        "archetype_class": "Kúzelník",
        "title": "Spectral Conjuror",
        "lore": "Otvára štrbiny do ríše duchov a privoláva astrálne fantómy na rozptýlenie nepriateľskej paľby.",
        "warhammer_stats": {"WS": 4, "BS": 5, "S": 3, "T": 4, "W": 5, "A": 2, "Ld": 9, "Sv": "4+"},
        "duel_stats": {"toughness": 14, "reflexes": 20, "aim": 22, "dodge": 24, "appearance": 26, "tactics": 24, "mobility": 18},
        "signature_ability": {"name": "Vyvolanie Spektrálneho Prízraku", "cost_mana": 2, "type": "decoy", "effect": "Presmeruje prvý nepriateľský útok na prízrak."},
        "favored_weapon": "Šepkajúci Duchovný Prút",
        "visual_aura": "#48d1cc"
    },
    {
        "id": "m_wizard_11",
        "name": "Rezonančný Kúzelník Zvuku",
        "gender": "male",
        "archetype_class": "Kúzelník",
        "title": "Sonic Resonator",
        "lore": "Láme sklo a kosti pomocou frekvenčných vĺn a ladičiek. Zvukové impulzy prerazia každé brnenie.",
        "warhammer_stats": {"WS": 4, "BS": 5, "S": 4, "T": 4, "W": 5, "A": 2, "Ld": 8, "Sv": "4+"},
        "duel_stats": {"toughness": 16, "reflexes": 26, "aim": 24, "dodge": 22, "appearance": 18, "tactics": 26, "mobility": 18},
        "signature_ability": {"name": "Sonický Pulz Omráčenia", "cost_mana": 2, "type": "stun", "effect": "Omráči cieľ a zníži jeho Uhýbanie na polovicu."},
        "favored_weapon": "Akustická Rezonančná Ladička",
        "visual_aura": "#7fffd4"
    },
    {
        "id": "m_wizard_12",
        "name": "Alchymistický Kúzelník Elixírov",
        "gender": "male",
        "archetype_class": "Kúzelník",
        "title": "Alchemical Evoker",
        "lore": "Vytvára nestabilné elixíry a vrhacie banky. V boji adaptuje svoje štatistiky podľa zraniteľnosti súpera.",
        "warhammer_stats": {"WS": 4, "BS": 5, "S": 4, "T": 4, "W": 5, "A": 2, "Ld": 8, "Sv": "4+"},
        "duel_stats": {"toughness": 22, "reflexes": 18, "aim": 22, "dodge": 18, "appearance": 14, "tactics": 28, "mobility": 16},
        "signature_ability": {"name": "Transmutačný Elixír Zúrivosti", "cost_mana": 2, "type": "buff", "effect": "+15 k Húževnatosti a +1 k základnému poškodeniu zbrane."},
        "favored_weapon": "Fosforový Mažiarový Pištoľ",
        "visual_aura": "#9acd32"
    },
    {
        "id": "m_illusion_13",
        "name": "Majster Zrkadlových Ilúzií",
        "gender": "male",
        "archetype_class": "Iluzionista",
        "title": "Phantasm Mirage Master",
        "lore": "Premieta desiatky falošných obrazov svojho tela. Nepriateľ strieľa do prázdneho vzduchu.",
        "warhammer_stats": {"WS": 5, "BS": 4, "S": 3, "T": 3, "W": 4, "A": 3, "Ld": 9, "Sv": "5+"},
        "duel_stats": {"toughness": 10, "reflexes": 26, "aim": 20, "dodge": 34, "appearance": 22, "tactics": 30, "mobility": 24},
        "signature_ability": {"name": "Zrkadlový Dvojník", "cost_mana": 2, "type": "evasion", "effect": "100% šanca na úplné vyhnutie sa jednému útoku."},
        "favored_weapon": "Strieborný Kord Iluzionistu",
        "visual_aura": "#dda0dd"
    },
    {
        "id": "m_illusion_14",
        "name": "Chronos Iluzionista",
        "gender": "male",
        "archetype_class": "Iluzionista",
        "title": "Chronos Illusionist",
        "lore": "Manipuluje s vnímaním toku času. Súper vidí guľku spomalene, no v skutočnosti už dopadla.",
        "warhammer_stats": {"WS": 5, "BS": 5, "S": 3, "T": 3, "W": 4, "A": 3, "Ld": 9, "Sv": "5+"},
        "duel_stats": {"toughness": 12, "reflexes": 30, "aim": 22, "dodge": 28, "appearance": 16, "tactics": 24, "mobility": 32},
        "signature_ability": {"name": "Časová Distorzia (Bullet Time)", "cost_mana": 3, "type": "time_warp", "effect": "Umožňuje okamžitý dodatočný protiútok po úspešnom úhybe."},
        "favored_weapon": "Chronometrický Revolver",
        "visual_aura": "#ba55d3"
    },
    {
        "id": "m_cold_15",
        "name": "Chladný Vystupovač - Ostrý Pohľad",
        "gender": "male",
        "archetype_class": "Chladný Vystupovač",
        "title": "Cold Stare Gunslinger",
        "lore": "Kultový duelant Západu. Jeho kamenná tvár a chladnokrvné vystupovanie paralyzujú súperovu taktiku.",
        "warhammer_stats": {"WS": 4, "BS": 6, "S": 4, "T": 4, "W": 5, "A": 3, "Ld": 9, "Sv": "4+"},
        "duel_stats": {"toughness": 14, "reflexes": 24, "aim": 30, "dodge": 16, "appearance": 36, "tactics": 12, "mobility": 18},
        "signature_ability": {"name": "Ľadový Pohľad Duelanta", "cost_mana": 1, "type": "intimidation", "effect": "Rozdiel vo Vystupovaní znižuje súperovu Presnosť na polovicu."},
        "favored_weapon": "Dvojhlavňový Kolt s Dlhým Doletom",
        "visual_aura": "#708090"
    },
    {
        "id": "m_cold_16",
        "name": "Chladný Povýšenecký Duelant",
        "gender": "male",
        "archetype_class": "Chladný Vystupovač",
        "title": "Arrogant Cold Duelist",
        "lore": "Šľachtický šermiar a pištoľník. Vystupuje s totálnym pohŕdaním, čím núti súpera k unáhleným chybám.",
        "warhammer_stats": {"WS": 6, "BS": 5, "S": 4, "T": 4, "W": 5, "A": 4, "Ld": 9, "Sv": "3+"},
        "duel_stats": {"toughness": 14, "reflexes": 28, "aim": 28, "dodge": 22, "appearance": 34, "tactics": 16, "mobility": 20},
        "signature_ability": {"name": "Bravúrne Vystúpenie", "cost_mana": 2, "type": "crit", "effect": "+35% šanca na kritický zásah do hlavy."},
        "favored_weapon": "Gravírovaný Kord a Derringer",
        "visual_aura": "#2f4f4f"
    },
    {
        "id": "m_soak_17",
        "name": "Železný Odolávač Bunkra",
        "gender": "male",
        "archetype_class": "Odolávač",
        "title": "Iron Soak Bulwark",
        "lore": "Ťažkoodenec zameraný na čistú Húževnatosť. Rany chladnou zbraňou sa odrážajú bez ujmy na zdraví.",
        "warhammer_stats": {"WS": 5, "BS": 3, "S": 5, "T": 6, "W": 6, "A": 2, "Ld": 10, "Sv": "2+"},
        "duel_stats": {"toughness": 38, "reflexes": 14, "aim": 16, "dodge": 10, "appearance": 12, "tactics": 26, "mobility": 10},
        "signature_ability": {"name": "Žulový Postoj Bunkra", "cost_mana": 1, "type": "tank", "effect": "Pohltí 50% celkového zranenia cez Húževnatosť."},
        "favored_weapon": "Zákopové Kladivo a Masívny Pavézny Štít",
        "visual_aura": "#696969"
    },
    {
        "id": "m_sniper_18",
        "name": "Reflexný Ostreľovač Vetra",
        "gender": "male",
        "archetype_class": "Ostreľovač",
        "title": "Wind Reflex Sniper",
        "lore": "Zložený z čistých reflexov a chirurgickej presnosti. Mitiguje strely a sám zasahuje z najväčšej diaľky.",
        "warhammer_stats": {"WS": 3, "BS": 7, "S": 3, "T": 3, "W": 4, "A": 2, "Ld": 8, "Sv": "5+"},
        "duel_stats": {"toughness": 10, "reflexes": 32, "aim": 36, "dodge": 24, "appearance": 10, "tactics": 16, "mobility": 28},
        "signature_ability": {"name": "Prerážajúci Výstrel do Hlavy", "cost_mana": 2, "type": "headshot", "effect": "Garantuje zásah do hlavy s násobiteľom 1.5x damage."},
        "favored_weapon": "Dlhodosahová Aéterová Puška Sharps",
        "visual_aura": "#87ceeb"
    },
    {
        "id": "m_shaman_19",
        "name": "Šaman Zemných Koreňov",
        "gender": "male",
        "archetype_class": "Šaman",
        "title": "Earthroot Shaman",
        "lore": "Vytvára symbiózu so zemským jadrom. Korene zachytávajú nohy nepriateľa a rušia jeho úhybné postoje.",
        "warhammer_stats": {"WS": 5, "BS": 4, "S": 5, "T": 5, "W": 6, "A": 2, "Ld": 9, "Sv": "3+"},
        "duel_stats": {"toughness": 28, "reflexes": 18, "aim": 18, "dodge": 14, "appearance": 18, "tactics": 28, "mobility": 14},
        "signature_ability": {"name": "Koreňové Zovretie", "cost_mana": 2, "type": "root", "effect": "Znehybní cieľ, znemožní mu postoj Zohnutie (Duck Down)."},
        "favored_weapon": "Totemová Hlohovú Palica",
        "visual_aura": "#8b4513"
    },
    {
        "id": "m_necro_20",
        "name": "Nekromant Prázdnoty",
        "gender": "male",
        "archetype_class": "Nekromant",
        "title": "Void Necromancer",
        "lore": "Čerpá silu z padlých duší v Tartare. Jeho chladný pohľad vysáva životnú silu zo súperovho tela.",
        "warhammer_stats": {"WS": 4, "BS": 5, "S": 4, "T": 4, "W": 5, "A": 2, "Ld": 9, "Sv": "4+"},
        "duel_stats": {"toughness": 20, "reflexes": 16, "aim": 22, "dodge": 14, "appearance": 32, "tactics": 26, "mobility": 14},
        "signature_ability": {"name": "Pohltenie Duše", "cost_mana": 3, "type": "drain", "effect": "Odoberie 1 HP súperovi a obnoví 1 HP sebe (strop 6 HP)."},
        "favored_weapon": "Kosák z Čiernej Ocele a Kadidelnica",
        "visual_aura": "#191970"
    }
]

# ── 3. 20 FEMALE MAGICAL REPRESENTATIVES ─────────────────────────────────────
FEMALE_ARCHETYPES_20: List[Dict[str, Any]] = [
    {
        "id": "f_oracle_01",
        "name": "Veštica Osudových Nití",
        "gender": "female",
        "archetype_class": "Veštica",
        "title": "Fate Weaver Oracle",
        "lore": "Rozpletá a spriada zlaté nite osudu. Pozná presnú trajektóriu každého projektilu v aréne.",
        "warhammer_stats": {"WS": 4, "BS": 6, "S": 3, "T": 4, "W": 4, "A": 2, "Ld": 10, "Sv": "5+"},
        "duel_stats": {"toughness": 12, "reflexes": 22, "aim": 30, "dodge": 20, "appearance": 24, "tactics": 34, "mobility": 16},
        "signature_ability": {"name": "Prestrihnutie Nite Osudu", "cost_mana": 3, "type": "fate", "effect": "Zruší súperov bonus k mierenou a zaručí zásah."},
        "favored_weapon": "Zlaté Nožnice Osudu a Vreteno",
        "visual_aura": "#e6e6fa"
    },
    {
        "id": "f_oracle_02",
        "name": "Lunárna Veštica Prílivu",
        "gender": "female",
        "archetype_class": "Veštica",
        "title": "Lunar Tide Seeress",
        "lore": "Ovládaná mesačnými fázami a nočným prílivom. Mätie protivníka odrazmi na hladine vody.",
        "warhammer_stats": {"WS": 4, "BS": 5, "S": 3, "T": 4, "W": 4, "A": 2, "Ld": 9, "Sv": "5+"},
        "duel_stats": {"toughness": 14, "reflexes": 24, "aim": 30, "dodge": 24, "appearance": 22, "tactics": 30, "mobility": 18},
        "signature_ability": {"name": "Lunárne Zrkadlenie", "cost_mana": 2, "type": "counter", "effect": "Zvyšuje Uhýbanie o +20 pri nočných a šerých podmienkach."},
        "favored_weapon": "Strieborné Lunárne Zrkadlo",
        "visual_aura": "#b0e0e6"
    },
    {
        "id": "f_witch_03",
        "name": "Nočná Čarodejnica Hviezdokop",
        "gender": "female",
        "archetype_class": "Čarodejnica",
        "title": "Night Star Witch",
        "lore": "Známa z nočných rituálov pod Polárkou. Jej aury zastrašujú aj najodvážnejších pištoľníkov.",
        "warhammer_stats": {"WS": 5, "BS": 5, "S": 4, "T": 4, "W": 5, "A": 3, "Ld": 9, "Sv": "4+"},
        "duel_stats": {"toughness": 16, "reflexes": 24, "aim": 26, "dodge": 20, "appearance": 32, "tactics": 22, "mobility": 18},
        "signature_ability": {"name": "Nočný Opar Zmätenia", "cost_mana": 2, "type": "debuff", "effect": "Znižuje nepriateľskú Taktiku a zamedzuje presnému mierenou do ramien."},
        "favored_weapon": "Metla z Čierneho Hlohu a Dýka",
        "visual_aura": "#483d8b"
    },
    {
        "id": "f_witch_04",
        "name": "Močiarna Bylinkárka a Travička",
        "gender": "female",
        "archetype_class": "Čarodejnica",
        "title": "Venomous Swamp Witch",
        "lore": "Majsterka bylín, toxínov a jedovatých extraktov. Každý jej zásah infikuje ranu leptavým jedom.",
        "warhammer_stats": {"WS": 4, "BS": 5, "S": 4, "T": 5, "W": 6, "A": 2, "Ld": 9, "Sv": "4+"},
        "duel_stats": {"toughness": 26, "reflexes": 18, "aim": 20, "dodge": 16, "appearance": 24, "tactics": 28, "mobility": 14},
        "signature_ability": {"name": "Jedovatá Toxická Infúzia", "cost_mana": 2, "type": "dot", "effect": "Zásah spôsobí 1 poškodenie navyše v nasledujúcom kole."},
        "favored_weapon": "Močiarna Fúkačka a Kyselina",
        "visual_aura": "#2e8b57"
    },
    {
        "id": "f_mage_05",
        "name": "Solárna Mágka Spálenia",
        "gender": "female",
        "archetype_class": "Mágka",
        "title": "Solar Flare Pyromanceress",
        "lore": "Koncentruje slnečné svetlo do spaľujúcich lúčov. Jej presnosť pri mierení je absolútne smrtiaca.",
        "warhammer_stats": {"WS": 4, "BS": 6, "S": 4, "T": 4, "W": 5, "A": 2, "Ld": 9, "Sv": "4+"},
        "duel_stats": {"toughness": 14, "reflexes": 24, "aim": 34, "dodge": 18, "appearance": 28, "tactics": 18, "mobility": 18},
        "signature_ability": {"name": "Koronálny Solárny Výbuch", "cost_mana": 3, "type": "solar_burst", "effect": "Trojitý ohnivý lúč s ignorovaním reflexného brnenia."},
        "favored_weapon": "Solárna Prsteňová Koruna",
        "visual_aura": "#ffa500"
    },
    {
        "id": "f_mage_06",
        "name": "Kryomantická Mágka Absolútnej Nuly",
        "gender": "female",
        "archetype_class": "Mágka",
        "title": "Absolute Zero Cryomanceress",
        "lore": "Ochladzuje teplotu okolo seba na bod mrazu. Vytvára ľadové brnenie pohlcujúce náboje.",
        "warhammer_stats": {"WS": 5, "BS": 5, "S": 4, "T": 5, "W": 5, "A": 2, "Ld": 9, "Sv": "3+"},
        "duel_stats": {"toughness": 24, "reflexes": 22, "aim": 26, "dodge": 18, "appearance": 20, "tactics": 30, "mobility": 14},
        "signature_ability": {"name": "Permafrostová Zmrazená Zóna", "cost_mana": 2, "type": "freeze", "effect": "Znižuje súperovu Pohyblivosť o -15."},
        "favored_weapon": "Krištáľový Ľadový Rapír",
        "visual_aura": "#e0ffff"
    },
    {
        "id": "f_mage_07",
        "name": "Búrková Mágka Bleskových Výbojov",
        "gender": "female",
        "archetype_class": "Mágka",
        "title": "Thunderclap Sorceress",
        "lore": "Vyvoláva blesky priamo z búrkových mrakov. Jej rýchlosť pohybu v aréne je neprekonateľná.",
        "warhammer_stats": {"WS": 4, "BS": 6, "S": 3, "T": 3, "W": 4, "A": 3, "Ld": 8, "Sv": "5+"},
        "duel_stats": {"toughness": 12, "reflexes": 32, "aim": 28, "dodge": 24, "appearance": 18, "tactics": 16, "mobility": 30},
        "signature_ability": {"name": "Búrkový Elektro-Skok", "cost_mana": 2, "type": "teleport", "effect": "Bleskový posun z dosahu nepriateľa pri úspešnom úhybe."},
        "favored_weapon": "Elektro-Magnetický Bič",
        "visual_aura": "#1e90ff"
    },
    {
        "id": "f_mage_08",
        "name": "Kozmická Aéterová Mágka",
        "gender": "female",
        "archetype_class": "Mágka",
        "title": "Cosmic Aether Mage",
        "lore": "Harmonizuje tok energie z Empyrean Neba. Vládne aéterickým poliam lámajúcim gravitáciu.",
        "warhammer_stats": {"WS": 4, "BS": 6, "S": 3, "T": 4, "W": 5, "A": 2, "Ld": 10, "Sv": "3+"},
        "duel_stats": {"toughness": 14, "reflexes": 22, "aim": 28, "dodge": 20, "appearance": 32, "tactics": 28, "mobility": 18},
        "signature_ability": {"name": "Aéterická Singularity Gravitácie", "cost_mana": 3, "type": "gravity", "effect": "Stiahne súpera a znemožní mu zmenu obranného postoja."},
        "favored_weapon": "Kozmický Prismatický Diadém",
        "visual_aura": "#8a2be2"
    },
    {
        "id": "f_wizard_09",
        "name": "Runová Kúzelníčka Ochranných Pečatí",
        "gender": "female",
        "archetype_class": "Kúzelníčka",
        "title": "Ward Weaver Enchantress",
        "lore": "Tká ochranné rúnické bariéry priamo v priestore. Pevnosť jej obrany chráni celú skupinu.",
        "warhammer_stats": {"WS": 5, "BS": 4, "S": 4, "T": 5, "W": 6, "A": 2, "Ld": 10, "Sv": "2+"},
        "duel_stats": {"toughness": 26, "reflexes": 20, "aim": 20, "dodge": 18, "appearance": 16, "tactics": 32, "mobility": 14},
        "signature_ability": {"name": "Aegis Bariéra Nepreniknuteľnosti", "cost_mana": 2, "type": "barrier", "effect": "Pohlcuje až 3 body poškodenia z akéhokoľvek typu zbrane."},
        "favored_weapon": "Rúnami Vykladaný Trojzubec",
        "visual_aura": "#4682b4"
    },
    {
        "id": "f_wizard_10",
        "name": "Kúzelníčka Astrálnych Sfér",
        "gender": "female",
        "archetype_class": "Kúzelníčka",
        "title": "Astral Conjuror",
        "lore": "Vyvoláva bytosti svetla a astrálne entity, ktoré oslňujú mierenie protivníkov.",
        "warhammer_stats": {"WS": 4, "BS": 5, "S": 3, "T": 4, "W": 5, "A": 2, "Ld": 9, "Sv": "4+"},
        "duel_stats": {"toughness": 14, "reflexes": 22, "aim": 24, "dodge": 26, "appearance": 28, "tactics": 26, "mobility": 20},
        "signature_ability": {"name": "Brána Astrálnych Bytostí", "cost_mana": 2, "type": "summon", "effect": "Astrálny spoločník absorbuje 1 ranu na kolo."},
        "favored_weapon": "Astrálna Krištáľová Harfa",
        "visual_aura": "#00fa9a"
    },
    {
        "id": "f_wizard_11",
        "name": "Kúzelníčka Prizmatického Svetla",
        "gender": "female",
        "archetype_class": "Kúzelníčka",
        "title": "Prismatic Light Weaver",
        "lore": "Láme biele svetlo na spektrálne lúče, ktoré oslepujú oči a znižujú súperovu šancu na zásah.",
        "warhammer_stats": {"WS": 4, "BS": 6, "S": 3, "T": 4, "W": 4, "A": 2, "Ld": 9, "Sv": "4+"},
        "duel_stats": {"toughness": 14, "reflexes": 28, "aim": 28, "dodge": 22, "appearance": 24, "tactics": 24, "mobility": 20},
        "signature_ability": {"name": "Oslepujúci Prizmatický Lúč", "cost_mana": 2, "type": "blind", "effect": "-20 k Presnosti súpera na 1 kolo."},
        "favored_weapon": "Spektrálny Prizmatický Prútik",
        "visual_aura": "#ff1493"
    },
    {
        "id": "f_wizard_12",
        "name": "Alchymistická Transmutátorka",
        "gender": "female",
        "archetype_class": "Kúzelníčka",
        "title": "Transmutation Evokeress",
        "lore": "Premieňa tuhé kovy na tekutý oheň a neutralizuje nepriateľské projektily vo vzduchu.",
        "warhammer_stats": {"WS": 4, "BS": 5, "S": 4, "T": 4, "W": 5, "A": 2, "Ld": 8, "Sv": "4+"},
        "duel_stats": {"toughness": 24, "reflexes": 20, "aim": 24, "dodge": 18, "appearance": 16, "tactics": 28, "mobility": 16},
        "signature_ability": {"name": "Transmutácia Olova na Aéter", "cost_mana": 2, "type": "convert", "effect": "Premení nepriateľskú guľku na neškodný záblesk."},
        "favored_weapon": "Transmutačná Banka a Pištoľ",
        "visual_aura": "#32cd32"
    },
    {
        "id": "f_illusion_13",
        "name": "Tkáčka Zrkadlových Preludov",
        "gender": "female",
        "archetype_class": "Iluzionistka",
        "title": "Mirage Phantasmist",
        "lore": "Kráľovná optických klamov. Jej telo splýva s priestorom a útoky cez ňu prechádzajú bez dotyku.",
        "warhammer_stats": {"WS": 5, "BS": 4, "S": 3, "T": 3, "W": 4, "A": 3, "Ld": 9, "Sv": "5+"},
        "duel_stats": {"toughness": 10, "reflexes": 30, "aim": 18, "dodge": 36, "appearance": 24, "tactics": 28, "mobility": 26},
        "signature_ability": {"name": "Zrkadlové Bludisko", "cost_mana": 2, "type": "mirage", "effect": "+30 k Uhýbaniu proti všetkým zónam zásahu."},
        "favored_weapon": "Krištáľové Vejáre s Ostnatým Lemom",
        "visual_aura": "#ee82ee"
    },
    {
        "id": "f_illusion_14",
        "name": "Iluzionistka Zmyslových Falší",
        "gender": "female",
        "archetype_class": "Iluzionistka",
        "title": "Sensory Hypnotist",
        "lore": "Očaruje myseľ protivníka hypnotickým hlasom a pohľadom. Núti súpera mieriť do nesprávnej zóny.",
        "warhammer_stats": {"WS": 4, "BS": 5, "S": 3, "T": 3, "W": 4, "A": 2, "Ld": 9, "Sv": "5+"},
        "duel_stats": {"toughness": 12, "reflexes": 24, "aim": 20, "dodge": 28, "appearance": 32, "tactics": 30, "mobility": 22},
        "signature_ability": {"name": "Hypnotická Parýza Zmyslov", "cost_mana": 2, "type": "hypnosis", "effect": "Presmeruje súperov úder z Hlavy do Trupu."},
        "favored_weapon": "Hypnotické Kyvadlo a Ihlica",
        "visual_aura": "#da70d6"
    },
    {
        "id": "f_cold_15",
        "name": "Chladná Vystupovačka - Ľadová Kráľovná",
        "gender": "female",
        "archetype_class": "Chladná Vystupovačka",
        "title": "Ice Queen Intimidator",
        "lore": "Najchladnejšie vystupovanie na celom Západnom pohraničí. Jej povýšenecká prítomnosť drví súperovu taktiku.",
        "warhammer_stats": {"WS": 5, "BS": 5, "S": 4, "T": 4, "W": 5, "A": 3, "Ld": 10, "Sv": "3+"},
        "duel_stats": {"toughness": 14, "reflexes": 26, "aim": 28, "dodge": 18, "appearance": 38, "tactics": 14, "mobility": 18},
        "signature_ability": {"name": "Mrazivý Povýšenecký Pohľad", "cost_mana": 1, "type": "freeze_stare", "effect": "Extrémny bonus k zastrašeniu: zníži súperovu obranu o 20%."},
        "favored_weapon": "Ľadový Gravírovaný Revolver Colt",
        "visual_aura": "#b0c4de"
    },
    {
        "id": "f_cold_16",
        "name": "Chladná Vystupovačka - Nemilosrdná Exekútorka",
        "gender": "female",
        "archetype_class": "Chladná Vystupovačka",
        "title": "Cold Executioneress",
        "lore": "Lovkyňa odmien, ktorá nikdy neminie cieľ. Jej chladný kľud v paľbe zaručuje okamžitú exekúciu.",
        "warhammer_stats": {"WS": 5, "BS": 6, "S": 4, "T": 4, "W": 5, "A": 3, "Ld": 9, "Sv": "3+"},
        "duel_stats": {"toughness": 16, "reflexes": 28, "aim": 32, "dodge": 20, "appearance": 34, "tactics": 16, "mobility": 18},
        "signature_ability": {"name": "Smrtiaci Výstrel Bez Mihnutia Oka", "cost_mana": 2, "type": "execute", "effect": "+2 poškodenie ak má súper menej ako 3 HP."},
        "favored_weapon": "Upravená Winchesterovka s Optikou",
        "visual_aura": "#778899"
    },
    {
        "id": "f_soak_17",
        "name": "Titánska Odolávačka Kryštálu",
        "gender": "female",
        "archetype_class": "Odolávačka",
        "title": "Crystal Shield Soak Ward",
        "lore": "Maximálna absorpcia zranenia. Pokrytá kryštálovými platňami ignoruje údery chladných zbraní.",
        "warhammer_stats": {"WS": 5, "BS": 3, "S": 5, "T": 6, "W": 6, "A": 2, "Ld": 10, "Sv": "2+"},
        "duel_stats": {"toughness": 40, "reflexes": 14, "aim": 16, "dodge": 12, "appearance": 14, "tactics": 24, "mobility": 10},
        "signature_ability": {"name": "Kryštálová Hradba Absolútneho Soaku", "cost_mana": 1, "type": "max_soak", "effect": "Mitigácia cez Húževnatosť sa zdvojnásobí proti chladným zbraniam."},
        "favored_weapon": "Veľký Kryštálový Pavézny Štít a Sekera",
        "visual_aura": "#5f9ea0"
    },
    {
        "id": "f_dodge_18",
        "name": "Blesková Uhýbačka a Akrobatka",
        "gender": "female",
        "archetype_class": "Uhýbačka",
        "title": "Agile Bullet Dodgeress",
        "lore": "Neuveriteľne pružná akrobatka. Dokáže sa zohnúť pod akúkoľvek guľku a uskočiť pred zásahom do ramien.",
        "warhammer_stats": {"WS": 5, "BS": 5, "S": 3, "T": 3, "W": 4, "A": 4, "Ld": 9, "Sv": "5+"},
        "duel_stats": {"toughness": 10, "reflexes": 34, "aim": 24, "dodge": 38, "appearance": 16, "tactics": 16, "mobility": 36},
        "signature_ability": {"name": "Akrobatický Výkrut (100% Úhyb)", "cost_mana": 1, "type": "acrobatic", "effect": "Automatický úspech postoja Kačica (Duck Down) a Úklon."},
        "favored_weapon": "Dve Ľahké Aéterové Dýky",
        "visual_aura": "#00fa9a"
    },
    {
        "id": "f_druid_19",
        "name": "Druidská Panej Divokých Hvozodv",
        "gender": "female",
        "archetype_class": "Druidka",
        "title": "Arborial Wild Wardeness",
        "lore": "Strážkyňa prastarých stromov a lesných zvierat. Čerpá regeneráciu priamo z mykorhíznej siete.",
        "warhammer_stats": {"WS": 5, "BS": 4, "S": 5, "T": 5, "W": 6, "A": 3, "Ld": 9, "Sv": "3+"},
        "duel_stats": {"toughness": 28, "reflexes": 20, "aim": 20, "dodge": 18, "appearance": 20, "tactics": 30, "mobility": 18},
        "signature_ability": {"name": "Hnev Posvätného Hvozdu", "cost_mana": 2, "type": "forest_fury", "effect": "+10 k Taktike a obnova 1 HP pri úspešnom bloku."},
        "favored_weapon": "Kvitnúca Hlohovú Kosa a Zrkadlový List",
        "visual_aura": "#228b22"
    },
    {
        "id": "f_banshee_20",
        "name": "Astrálna Banshee Šepotov",
        "gender": "female",
        "archetype_class": "Banshee",
        "title": "Whispering Astral Banshee",
        "lore": "Éterický hlas zo záhrobia. Jej výkrik rozochveje bubienky a zničí koncentráciu každého strelca.",
        "warhammer_stats": {"WS": 4, "BS": 5, "S": 3, "T": 4, "W": 5, "A": 3, "Ld": 9, "Sv": "4+"},
        "duel_stats": {"toughness": 14, "reflexes": 26, "aim": 22, "dodge": 28, "appearance": 34, "tactics": 24, "mobility": 24},
        "signature_ability": {"name": "Smrtonosný Výkrik Záhrobia", "cost_mana": 3, "type": "screech", "effect": "Zníži súperovu Húževnatosť aj Reflexy o 10 na 2 kolá."},
        "favored_weapon": "Astrálny Flautový Prútik a Tieňový Závoj",
        "visual_aura": "#9932cc"
    }
]

# ── 4. CATALOG OF 120 UNIQUE HELPERS (10 PER RACE x 12 RACES) ───────────────
def _generate_120_helpers() -> List[Dict[str, Any]]:
    helpers: List[Dict[str, Any]] = []

    # Definitions of names, roles and abilities per race
    race_helper_blueprints = {
        "crystal": [
            ("Kryštálový Kolibrík", "Familiár", {"reflexes": 8, "aim": 5}, "Rýchly aéterový prieskum"),
            ("Aéterový Motýľ", "Astrálny sprievodca", {"dodge": 7, "appearance": 6}, "Optické trblietanie"),
            ("Rezonančný Chrobák", "Bojový minion", {"toughness": 8, "tactics": 5}, "Frekvenčný štít"),
            ("Prisma Hadík", "Familiár", {"aim": 9, "reflexes": 6}, "Laserové značenie zóny"),
            ("Žiarivý Karbunkul", "Rituálny nosič", {"appearance": 10, "tactics": 6}, "Svetelná aura"),
            ("Kryštálový Havran", "Prieskumník", {"aim": 8, "dodge": 8}, "Nočné varovanie"),
            ("Kremenný Strážca", "Bojový minion", {"toughness": 12, "reflexes": 4}, "Kremenný pancier"),
            ("Aéterový Rys", "Dravec", {"mobility": 10, "aim": 7}, "Tichý výpad z boku"),
            ("Hviezdny Homunkulus", "Učeň", {"tactics": 10, "appearance": 8}, "Katalyzátor kúziel"),
            ("Kryštálový Drakoliah", "Apex spoločník", {"toughness": 10, "appearance": 12}, "Prismatický dych")
        ],
        "toxic": [
            ("Jedovatý Škorpiónik", "Familiár", {"toughness": 6, "aim": 7}, "Jedový osteň"),
            ("Močiarny Pavúčik", "Prieskumník", {"dodge": 8, "tactics": 6}, "Lepkavá pasca"),
            ("Kyslý Slimák", "Bojový minion", {"toughness": 10, "tactics": 5}, "Kyselinový sliz"),
            ("Zelený Bazilištek", "Dravec", {"appearance": 9, "reflexes": 6}, "Ochromujúci pohľad"),
            ("Viperí Plaz", "Familiár", {"reflexes": 8, "aim": 8}, "Bleskové uštipnutie"),
            ("Spórový Hríbik", "Rituálny nosič", {"toughness": 9, "appearance": 6}, "Oblak toxických spór"),
            ("Toxický Mlok", "Familiár", {"dodge": 9, "toughness": 6}, "Regeneračný sliz"),
            ("Kyselinový Chrlič", "Bojový minion", {"aim": 10, "toughness": 8}, "Žieravý výron"),
            ("Jedová Húsenica", "Familiár", {"tactics": 8, "toughness": 7}, "Toxický kokón"),
            ("Močiarny Hydra-Zárodok", "Apex spoločník", {"toughness": 12, "appearance": 10}, "Viacnásobné uštipnutie")
        ],
        "druid": [
            ("Dubový Dendroidík", "Familiár", {"toughness": 10, "tactics": 6}, "Koreňová hradba"),
            ("Mesačný Vĺčik", "Bojový spoločník", {"mobility": 8, "reflexes": 7}, "Nočné zavytie"),
            ("Sova Hvozdu", "Astrálny sprievodca", {"aim": 9, "tactics": 8}, "Múdrosť lesa"),
            ("Lesný Puk", "Škriatok", {"dodge": 10, "mobility": 7}, "Lesný klam"),
            ("Machový Golemík", "Bojový minion", {"toughness": 12, "tactics": 5}, "Machový absorbér"),
            ("Diviak Štetináč", "Bojový spoločník", {"toughness": 9, "mobility": 8}, "Húževnatý náraz"),
            ("Šepkajúci Žalud", "Rituálny nosič", {"tactics": 9, "appearance": 7}, "Telepatický šepot"),
            ("Prameňový Nymf", "Familiár", {"dodge": 8, "reflexes": 8}, "Hojivá kvapka"),
            ("Hlohový Sprite", "Familiár", {"aim": 8, "appearance": 8}, "Trnistá clona"),
            ("Prastarý Lesný Medveď", "Apex spoločník", {"toughness": 14, "appearance": 9}, "Úder labou a rev")
        ],
        "human": [
            ("Stopársky Pes", "Stopár", {"tactics": 8, "aim": 7}, "Ostrý čuch na stopy"),
            ("Poštový Sokol", "Prieskumník", {"reflexes": 9, "aim": 8}, "Blesková správa"),
            ("Verný Poni", "Nosič", {"toughness": 8, "mobility": 6}, "Dodatočná munícia"),
            ("Učeň Pištoľníka", "Pomocník", {"aim": 9, "reflexes": 7}, "Rýchle nabíjanie"),
            ("Hliadkový Bubeník", "Bojový bubeník", {"appearance": 8, "tactics": 8}, "Bojový pochod"),
            ("Kovbojský Kokeršpaniel", "Spoločník", {"dodge": 8, "mobility": 7}, "Rozptýlenie súpera"),
            ("Pustinný Rys", "Dravec", {"mobility": 9, "aim": 8}, "Nočný lov"),
            ("Táborový Mastif", "Obranca", {"toughness": 10, "appearance": 7}, "Hrdelný štekot"),
            ("Polný Felčiarik", "Zdravotník", {"tactics": 9, "toughness": 6}, "Bandážovanie v kryte"),
            ("Hraničiarsky Mustang", "Apex zviera", {"mobility": 12, "reflexes": 9}, "Úniková rýchlosť")
        ],
        "dwarf": [
            ("Kováčsky Piadimužík", "Kováč", {"toughness": 9, "tactics": 7}, "Oprava brnenia za pochodu"),
            ("Bronzový Mechanoid", "Bojový robot", {"toughness": 12, "aim": 5}, "Parný výfuk"),
            ("Banský Jazvec", "Familiár", {"toughness": 8, "dodge": 7}, "Hĺbenie zákopu"),
            ("Kamenný Krt", "Prieskumník", {"tactics": 8, "toughness": 7}, "Podkopanie súpera"),
            ("Parný Droník", "Lietajúci dron", {"reflexes": 8, "aim": 8}, "Monitorovanie arény"),
            ("Nákovový Golem", "Obranný kolos", {"toughness": 14, "appearance": 6}, "Absolútny odraz guliek"),
            ("Lávový Salamandrík", "Familiár", {"aim": 8, "appearance": 8}, "Žeravý dych"),
            ("Železný Baranič", "Bojový minion", {"toughness": 11, "mobility": 6}, "Prerazenie dverí"),
            ("Rúnostrojný Pomocník", "Učeň", {"tactics": 10, "aim": 7}, "Kalibrácia rún"),
            ("Hlbinný Bazaltový Titaník", "Apex spoločník", {"toughness": 15, "appearance": 8}, "Zemetrasenie")
        ],
        "elf": [
            ("Hviezdny Fénixík", "Familiár", {"appearance": 10, "aim": 8}, "Znovuzrodenie z popola"),
            ("Mesačná Laň", "Astrálny sprievodca", {"dodge": 11, "mobility": 8}, "Ľahký krok bez hluku"),
            ("Strieborný Jastrab", "Dravec", {"aim": 12, "reflexes": 8}, "Presný zásah oka"),
            ("Aéterická Víla", "Familiár", {"appearance": 9, "dodge": 9}, "Závoj neviditeľnosti"),
            ("Hviezdna Líška", "Prieskumník", {"reflexes": 10, "tactics": 7}, "Rýchle predvídanie"),
            ("Slnečný Sokol", "Dravec", {"aim": 11, "appearance": 8}, "Oslepujúci strmhlavý let"),
            ("Žiarivý Jednorožček", "Astrálny sprievodca", {"appearance": 12, "tactics": 9}, "Očistenie od kliatob"),
            ("Hviezdokopný Duch", "Mág", {"tactics": 10, "aim": 9}, "Hviezdne zameriavanie"),
            ("Zrkadlový Swan", "Familiár", {"dodge": 10, "appearance": 9}, "Kozmetický klam"),
            ("Nebeský Pegasík", "Apex spoločník", {"mobility": 14, "appearance": 11}, "Let ponad prekážky")
        ],
        "infernal": [
            ("Ohnivý Imp", "Familiár", {"appearance": 8, "aim": 7}, "Zákerný plamienok"),
            ("Sírový Chrt", "Bojový spoločník", {"appearance": 9, "toughness": 8}, "Pálivý dych"),
            ("Plamenný Salamander", "Familiár", {"toughness": 9, "appearance": 7}, "Lávová stopa"),
            ("Popolavý Netopier", "Prieskumník", {"dodge": 9, "reflexes": 7}, "Popolavá clona"),
            ("Pekelný Škorpión", "Familiár", {"aim": 8, "toughness": 8}, "Žeravý bodec"),
            ("Lávový Krab", "Bojový minion", {"toughness": 12, "appearance": 6}, "Odolnosť voči ohňu"),
            ("Démonický Havran", "Posol", {"appearance": 10, "tactics": 7}, "Krákanie záhuby"),
            ("Ohnivý Kobold", "Bojový minion", {"mobility": 9, "aim": 8}, "Podpaľačský zápal"),
            ("Žeravý Chrlič", "Obranca", {"toughness": 11, "appearance": 9}, "Kamenný žeravý pohľad"),
            ("Menší Pán Pekla", "Apex spoločník", {"appearance": 14, "toughness": 10}, "Pekelné plamene a strach")
        ],
        "celestial": [
            ("Zlatý Cherubín", "Familiár", {"appearance": 10, "tactics": 8}, "Spev anjelov"),
            ("Svetelná Holubica", "Posol", {"dodge": 9, "reflexes": 8}, "Požehnanie mieru"),
            ("Žiarivý Halo-Duch", "Astrálny sprievodca", {"appearance": 11, "dodge": 8}, "Oslnivá žiara"),
            ("Nebeský Baránok", "Familiár", {"tactics": 9, "toughness": 7}, "Ochranné rúcho"),
            ("Strieborná Lutna", "Totem", {"appearance": 10, "tactics": 8}, "Harmonické tóny"),
            ("Aureolový Motýľ", "Familiár", {"dodge": 10, "appearance": 8}, "Zlatý prach"),
            ("Anjelský Panoš", "Bojový minion", {"aim": 9, "tactics": 9}, "Svetelná kuša"),
            ("Trónový Lúčik", "Energia", {"aim": 11, "appearance": 9}, "Lúč očisty"),
            ("Prizmatický Sprievodca", "Astrálny sprievodca", {"tactics": 10, "dodge": 9}, "Ochrana pred zlom"),
            ("Menší Seraf", "Apex spoločník", {"appearance": 13, "tactics": 11}, "Šesťkrídle svetlo")
        ],
        "spectral": [
            ("Bludička Močiarna", "Familiár", {"dodge": 10, "appearance": 7}, "Zavádzajúce svetielko"),
            ("Kostlivý Papagáj", "Prieskumník", {"reflexes": 9, "dodge": 8}, "Chraplavé varovanie"),
            ("Prízračná Ruka", "Familiár", {"aim": 8, "tactics": 8}, "Krádež munície zo vzduchu"),
            ("Astrálny Tieň", "Familiár", {"dodge": 11, "reflexes": 8}, "Tiché splynutie"),
            ("Polnočný Netopierik", "Prieskumník", {"mobility": 9, "reflexes": 8}, "Echolokácia v tme"),
            ("Éterická Lebka", "Totem", {"appearance": 10, "tactics": 7}, "Mrazivý smiech"),
            ("Cintorínsky Čierny Kocúr", "Familiár", {"dodge": 10, "reflexes": 9}, "Nešťastie pre súpera"),
            ("Spektrálny Prízrak", "Bojový minion", {"appearance": 11, "dodge": 9}, "Chladivý dotyk smrti"),
            ("Prachový Duch", "Familiár", {"tactics": 9, "dodge": 9}, "Rozptýlenie do prachu"),
            ("Prízračný Jazdec-Zárodok", "Apex spoločník", {"mobility": 13, "appearance": 11}, "Bezhlavý výpad")
        ],
        "elemental": [
            ("Vzdušný Vírnik", "Familiár", {"reflexes": 9, "mobility": 9}, "Veterný poryv"),
            ("Blesková Iskra", "Familiár", {"aim": 9, "reflexes": 9}, "Mikro-výboj"),
            ("Vodná Kvapka", "Familiár", {"dodge": 9, "toughness": 7}, "Tekutý úhyb"),
            ("Zemský Okruhliak", "Bojový minion", {"toughness": 11, "tactics": 6}, "Tvrdá škrupina"),
            ("Ohnivý Uhlík", "Familiár", {"aim": 8, "appearance": 8}, "Tlejúci plamienok"),
            ("Búrkový Obláčik", "Astrálny sprievodca", {"reflexes": 10, "aim": 8}, "Miniatúrny blesk"),
            ("Piesočný Diablik", "Familiár", {"dodge": 10, "mobility": 8}, "Piesočná búrka v očiach"),
            ("Magmatický Výron", "Bojový minion", {"toughness": 10, "aim": 8}, "Horúca troska"),
            ("Ľadový Kryštálik", "Familiár", {"tactics": 9, "toughness": 8}, "Mrazivý kryštál"),
            ("Prvotný Živlový Zhluk", "Apex spoločník", {"mobility": 11, "reflexes": 11}, "Vír všetkých 4 živlov")
        ],
        "fae": [
            ("Lesná Rusalka", "Familiár", {"appearance": 9, "dodge": 9}, "Zvádzanie k rieke"),
            ("Kvetinový Škriatok", "Familiár", {"dodge": 9, "tactics": 8}, "Peľový spánok"),
            ("Hubová Víla", "Rituálny nosič", {"tactics": 9, "toughness": 7}, "Halucinogénne spóry"),
            ("Trblietavý Chrobáčik", "Familiár", {"appearance": 8, "dodge": 8}, "Nočné svetielkovanie"),
            ("Šantivý Faun", "Bojový spoločník", {"mobility": 10, "appearance": 8}, "Hravý skok"),
            ("Zlatý Pavúk Tkáč", "Familiár", {"tactics": 10, "aim": 8}, "Zlatá pavučina"),
            ("Svetielkujúca Vážka", "Prieskumník", {"reflexes": 11, "dodge": 9}, "Akrobatický let"),
            ("Rosný Duch", "Familiár", {"dodge": 10, "tactics": 8}, "Ranná osviežujúca rosa"),
            ("Melodický Drozd", "Posol", {"appearance": 9, "tactics": 8}, "Pieseň zmätenia"),
            ("Kráľovnin Panoš Fae", "Apex spoločník", {"appearance": 13, "dodge": 11}, "Vília kráľovská pečať")
        ],
        "beastkin": [
            ("Pustinný Šakal", "Familiár", {"tactics": 8, "mobility": 8}, "Svorkový inštinkt"),
            ("Horský Kozorožec", "Bojový spoločník", {"toughness": 9, "mobility": 8}, "Skalný náraz rohmi"),
            ("Nočný Leopardík", "Dravec", {"mobility": 11, "aim": 8}, "Plíženie v tieni"),
            ("Bojový Jazvec", "Obranca", {"toughness": 11, "tactics": 7}, "Húževnaté zahryznutie"),
            ("Rýchla Lasica", "Familiár", {"reflexes": 10, "dodge": 9}, "Bleskový úskok"),
            ("Zúrivý Vlčisko", "Bojový spoločník", {"mobility": 10, "toughness": 8}, "Trhanie koristi"),
            ("Pralesy Pavián", "Bojový minion", {"aim": 9, "mobility": 8}, "Vrh kameňom z výšky"),
            ("Skalný Orol", "Dravec", {"aim": 11, "reflexes": 9}, "Pazúry z oblakov"),
            ("Riečna Vydra", "Familiár", {"dodge": 10, "mobility": 9}, "Vynorenie a úder"),
            ("Feralny Alfa-Tieň", "Apex spoločník", {"mobility": 13, "toughness": 10}, "Hrdelný rev predátorov")
        ]
    }

    helper_idx = 1
    for race_id, blueprints in race_helper_blueprints.items():
        race_info = RACES_CATALOG[race_id]
        for name, role, aura_stats, special in blueprints:
            tier = 1 + ((helper_idx - 1) % 10) // 2  # 1 to 5 tier
            h_obj = {
                "id": f"helper_{helper_idx:03d}",
                "index": helper_idx,
                "name": name,
                "race": race_id,
                "race_name": race_info["name"],
                "tier": tier,
                "role": role,
                "aura_bonus": aura_stats,
                "special_ability": special,
                "synergy_multiplier": 1.40,
                "icon": race_info["icon"],
                "color": race_info["color_accent"]
            }
            helpers.append(h_obj)
            helper_idx += 1

    return helpers

HELPERS_120_CATALOG: List[Dict[str, Any]] = _generate_120_helpers()
HELPERS_BY_ID: Dict[str, Dict[str, Any]] = {h["id"]: h for h in HELPERS_120_CATALOG}
HELPERS_BY_INDEX: Dict[int, Dict[str, Any]] = {h["index"]: h for h in HELPERS_120_CATALOG}

ALL_40_REPRESENTATIVES: Dict[str, Dict[str, Any]] = {
    c["id"]: c for c in (MALE_ARCHETYPES_20 + FEMALE_ARCHETYPES_20)
}

# ── 5. ARCHETYPE AND HELPER ENGINE ───────────────────────────────────────────
class ArchetypeAndHelperEngine:
    """
    Coordinates 20 Male / 20 Female magical archetypes, 12 races,
    120 helpers, continuous slider blending, and dual-system duel resolution.
    """

    @staticmethod
    def get_character(char_id: str) -> Optional[Dict[str, Any]]:
        return ALL_40_REPRESENTATIVES.get(char_id)

    @staticmethod
    def list_characters(gender: Optional[str] = None, archetype_class: Optional[str] = None) -> List[Dict[str, Any]]:
        chars = list(ALL_40_REPRESENTATIVES.values())
        if gender:
            chars = [c for c in chars if c["gender"].lower() == gender.lower()]
        if archetype_class:
            chars = [c for c in chars if archetype_class.lower() in c["archetype_class"].lower()]
        return chars

    @staticmethod
    def get_helper(index_or_id: Any) -> Optional[Dict[str, Any]]:
        if isinstance(index_or_id, int):
            return HELPERS_BY_INDEX.get(index_or_id)
        if isinstance(index_or_id, str):
            if index_or_id.isdigit():
                return HELPERS_BY_INDEX.get(int(index_or_id))
            return HELPERS_BY_ID.get(index_or_id)
        return None

    @staticmethod
    def list_helpers(race: Optional[str] = None, tier: Optional[int] = None) -> List[Dict[str, Any]]:
        h_list = HELPERS_120_CATALOG
        if race:
            h_list = [h for h in h_list if h["race"].lower() == race.lower()]
        if tier:
            h_list = [h for h in h_list if h["tier"] == tier]
        return h_list

    @staticmethod
    def calculate_composite_build(
        hero_id: str,
        helper_index: int,
        race_id: str,
        slider_adjustments: Optional[Dict[str, int]] = None,
        synergy_scale: float = 1.0
    ) -> Dict[str, Any]:
        """
        Combines hero baseline stats with chosen race, 1-120 helper aura,
        and interactive slider adjustments. Strictly enforces vital invariants.
        """
        hero = ArchetypeAndHelperEngine.get_character(hero_id)
        if not hero:
            # Fallback to first representative
            hero = MALE_ARCHETYPES_20[0]

        helper = ArchetypeAndHelperEngine.get_helper(helper_index)
        if not helper:
            helper = HELPERS_BY_INDEX.get(1, HELPERS_120_CATALOG[0])

        race_info = RACES_CATALOG.get(race_id, RACES_CATALOG["crystal"])

        # Check racial synergy match
        is_racial_match = (race_id == helper["race"])
        base_match_mult = helper["synergy_multiplier"] if is_racial_match else 1.0
        final_synergy = base_match_mult * max(0.5, min(2.0, synergy_scale))

        # Base duel stats
        base_duel = dict(hero["duel_stats"])
        composite_duel = {}

        # Apply race bonus
        race_bonuses = race_info.get("synergy_bonus", {})
        helper_auras = helper.get("aura_bonus", {})
        sliders = slider_adjustments or {}

        stat_keys = ["toughness", "reflexes", "aim", "dodge", "appearance", "tactics", "mobility"]
        for k in stat_keys:
            base_val = base_duel.get(k, 10)
            r_val = race_bonuses.get(k, 0)
            h_val = int(round(helper_auras.get(k, 0) * final_synergy))
            s_val = sliders.get(k, 0)  # delta from slider or direct slider adjustment

            # Combined total
            combined = max(1, base_val + r_val + h_val + s_val)
            composite_duel[k] = combined

        # Enforce Warhammer vital invariant (Wounds <= 6)
        base_wh = dict(hero["warhammer_stats"])
        clamped_wounds = max(1, min(6, base_wh.get("W", 4)))
        base_wh["W"] = clamped_wounds

        return {
            "hero_id": hero["id"],
            "hero_name": hero["name"],
            "hero_gender": hero["gender"],
            "archetype_class": hero["archetype_class"],
            "selected_race": race_info,
            "selected_helper": helper,
            "racial_synergy_match": is_racial_match,
            "synergy_multiplier_applied": round(final_synergy, 2),
            "composite_duel_stats": composite_duel,
            "warhammer_stats": base_wh,
            "max_hp_vital_invariant": 6,
            "favored_weapon": hero.get("favored_weapon", "Aéterová Zbraň"),
            "signature_ability": hero.get("signature_ability", {})
        }

    @staticmethod
    def resolve_interactive_duel_round(
        composite_build: Dict[str, Any],
        defender_stats: Optional[Dict[str, int]],
        attack_zone: str,
        defense_stance: str,
        weapon_type: DuelWeaponCategory = DuelWeaponCategory.COLD_MELEE,
        base_damage: int = 4,
        defender_current_hp: int = 6
    ) -> Dict[str, Any]:
        """
        Executes a duel round using TheWestDuelAlgebra with the composite build.
        """
        attacker_duel_stats = composite_build.get("composite_duel_stats", {})
        if not defender_stats:
            defender_stats = {
                "toughness": 20, "reflexes": 20, "aim": 20,
                "dodge": 20, "appearance": 15, "tactics": 20, "mobility": 15
            }

        return TheWestDuelAlgebra.resolve_duel_round(
            attacker_stats=attacker_duel_stats,
            defender_stats=defender_stats,
            attack_zone=attack_zone,
            defense_stance=defense_stance,
            weapon_type=weapon_type,
            base_weapon_damage=base_damage,
            defender_current_hp=defender_current_hp
        )
