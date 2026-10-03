# ==============================================================================
# KRYSTAL-STACK: EPIC CAMPAIGN SAGA & INTERACTIVE CONTENT ENGINE
# ==============================================================================
# Implements:
#   1. Canonical Multi-Stage Campaign Saga:
#      - Chapter 1: Krádež Aéterového Plameňa (Desert Canyon Caravan Ambush)
#      - Chapter 2: Búrka Totémov a Rozštiepenie Sektora (Rogallo Tempest Dogfight)
#      - Chapter 3: Obliehanie Balónovej Citadely (240mm Plunging Mortar Siege)
#   2. Procedural Mission & Dynamic Encounter Generator:
#      - Multi-tier altitudes (Canyon, Skybridge Nexus, Stratospheric Bastion).
#      - Dynamic weather (Sandstorms, Totem Lightning Rifts, Thermal Gales).
#      - Procedural enemy fleets & tactical rewards.
#   3. Full Lore Codex & Bestiary:
#      - Tribal factions, floating balloon citadels, Rogallo delta wings, cyber camels.
#   4. Strict 6 Max HP Vital Invariant compliance across all battle simulations.
# ==============================================================================

import math
import time
import uuid
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Strict vital invariant
VITAL_MAX_HP: int = 6


@dataclass
class CampaignChoice:
    id: str
    label: str
    description: str
    tactical_modifier: str
    risk_level: str
    reward_bonus_pct: int


@dataclass
class CampaignChapter:
    id: str
    chapter_number: int
    title: str
    subtitle: str
    altitude_tier: str
    location_name: str
    narrative_intro: str
    primary_objective: str
    secondary_objective: str
    recommended_deck_archetype: str
    base_nuggets_reward: int
    base_xp_reward: int
    choices: List[CampaignChoice]
    enemy_leader_name: str
    enemy_leader_hp: int  # Strict 6 Max HP


class EpicCampaignEngine:
    """
    Manages the overarching narrative campaign saga, procedural tactical missions,
    interactive branching operations, and lore codex.
    """

    def __init__(self):
        self.chapters: Dict[str, CampaignChapter] = {}
        self.codex: Dict[str, Dict[str, Any]] = {}
        self._initialize_canonical_chapters()
        self._initialize_lore_codex()

    def _initialize_canonical_chapters(self):
        c1 = CampaignChapter(
            id="chapter_1_desert_caravan",
            chapter_number=1,
            title="Kapitola 1: Krádež Aéterového Plameňa",
            subtitle="Nočný výsadok na padákoch do kaňonov a prepad karavány",
            altitude_tier="canyon_dunes",
            location_name="Zlatý Piesočný Kaňon Korrado (Y=45m)",
            narrative_intro=(
                "Púštna karavána kybernetických tiav klanu Korrado preváža vzácny 4B IBM Granite LLM "
                "kognitívny procesor. Toxickí nájazdníci obsadili kaňonové úžiny. Váš oddiel musí "
                "vykonať nočný výsadok na riaditeľných padákoch, zlikvidovať hliadky a zabezpečiť konvoj."
            ),
            primary_objective="Zabezpečiť procesorové jadro a zachrániť aspoň 2 kyber-ťavy",
            secondary_objective="Zničiť strážnu vežu bez spustenia svetlicového poplachu",
            recommended_deck_archetype="Toxický Infiltrátor / Tichý Parížsky Padáčkar",
            base_nuggets_reward=250,
            base_xp_reward=600,
            choices=[
                CampaignChoice(
                    id="choice_silent_drop",
                    label="Tichý Nočný Výsadok na Padákoch",
                    description="Otvorte padáky vo výške 200m a pristaňte priamo na chrbtoch strážnych veží.",
                    tactical_modifier="STEALTH_BONUS_CRIT_PLUS_35",
                    risk_level="MEDIUM",
                    reward_bonus_pct=15
                ),
                CampaignChoice(
                    id="choice_mortar_diversion",
                    label="Krycia Delostrelecká Paľba z Rogala",
                    description="Vystreľte z výšky signalizačný mínomet a preleťte kaňon na rogalách.",
                    tactical_modifier="EXPLOSIVE_SUPPRESSION_AOE",
                    risk_level="HIGH",
                    reward_bonus_pct=30
                )
            ],
            enemy_leader_name="Náčelník Kaňonových Hrdlorezov Vane",
            enemy_leader_hp=VITAL_MAX_HP
        )

        c2 = CampaignChapter(
            id="chapter_2_totem_tempest",
            chapter_number=2,
            title="Kapitola 2: Búrka Totémov a Rozštiepenie Sektora",
            subtitle="Letecký súboj na rogalách cez pole rozrezaných anomálií",
            altitude_tier="skybridge_midways",
            location_name="Rezonančné Totemové Pásmo 'Severný Kryštál' (Y=240m)",
            narrative_intro=(
                "Starobylé kryštálové totemy v sektore sa rozkmitali na chaotickej frekvencii 918 Hz. "
                "Dimenzionálne trhliny chrlia azúrové blesky a vytvárajú gravitačné vlny. "
                "Bojovníci vzlietli na delta-krídlach Rogalo, aby pomocou aéterových projektorov "
                "preladili rezonančné jadro skôr, než zrúti visuté mosty."
            ),
            primary_objective="Naladiť 3 rezonančné totemy na harmonickú frekvenciu 432 Hz",
            secondary_objective="Preletieť cez 2 stúpavé termické prúdy bez zásahu anomáliou",
            recommended_deck_archetype="Kryštálový Astromant / Rýchly Letec Rogala",
            base_nuggets_reward=450,
            base_xp_reward=1100,
            choices=[
                CampaignChoice(
                    id="choice_totem_attune",
                    label="Harmonické Prepojenie Totémov Kúzlom",
                    description="Použite lúčové projekcie na zjednotenie frekvencií totémov.",
                    tactical_modifier="TOTEM_HARMONIC_WARD_SHIELD",
                    risk_level="LOW",
                    reward_bonus_pct=10
                ),
                CampaignChoice(
                    id="choice_rift_overcharge",
                    label="Preťaženie Dimenzionálnej Trhliny",
                    description="Využite anomáliu na vytvorenie rázovej vlny, ktorá oslepí nepriateľské balóny.",
                    tactical_modifier="RIFT_BLINDNESS_ELECTRO_BURST",
                    risk_level="HIGH",
                    reward_bonus_pct=40
                )
            ],
            enemy_leader_name="Anomálny Prízrak 'Chrono-Blesk'",
            enemy_leader_hp=VITAL_MAX_HP
        )

        c3 = CampaignChapter(
            id="chapter_3_citadel_siege",
            chapter_number=3,
            title="Kapitola 3: Obliehanie Balónovej Citadely",
            subtitle="Strmá paľba 240mm mínometu z vznášajúceho sa ostrova a hák",
            altitude_tier="stratospheric_citadel",
            location_name="Železná Plávajúca Citadela 'Alpha Bastion' (Y=380m)",
            narrative_intro=(
                "Vrcholné stretnutie vo výške 380 metrov. Pevnostný ostrov nepriateľa je chránený "
                "štyrmi nosnými aéterickými aerostatmi a 240mm mínometnou batériou. "
                "Váš tím musí vystreliť Just Cause háky, vyšplhať sa cez kymácajúce sa visuté lávky "
                "a v súboji na život a na smrť neutralizovať obliehací mínomet."
            ),
            primary_objective="Vyhodiť do vzduchu prívod plynu nosného balóna a obsadiť mínomet",
            secondary_objective="Vykonať úspešný Slingshot skok z mosta na nepriateľskú palubu",
            recommended_deck_archetype="Druidský Titán / Just Cause Kinetický Majster",
            base_nuggets_reward=1200,
            base_xp_reward=2800,
            choices=[
                CampaignChoice(
                    id="choice_counter_battery",
                    label="Plunging Mortar Súboj z Vlastného Ostrova",
                    description="Odpáľte 240mm moždiarový granát s gravitačným urýchlením z výšky 280m.",
                    tactical_modifier="MORTAR_GRAVITY_PLUNGE_263MPS",
                    risk_level="HIGH",
                    reward_bonus_pct=35
                ),
                CampaignChoice(
                    id="choice_grapple_assault",
                    label="Kinetický Výsadok Hákom a Rogalom",
                    description="Pritiahnite sa hákom priamo k spodnej konzole nepriateľského ostrova.",
                    tactical_modifier="JUST_CAUSE_BOARDING_SLINGSHOT",
                    risk_level="VERY_HIGH",
                    reward_bonus_pct=50
                )
            ],
            enemy_leader_name="Lord Generál Korrado 'Železný Balón'",
            enemy_leader_hp=VITAL_MAX_HP
        )

        self.chapters = {c1.id: c1, c2.id: c2, c3.id: c3}

    def _initialize_lore_codex(self):
        self.codex = {
            "factions": {
                "crystal_tribe": {
                    "name": "Kryštálový Kmeň (Aéteroví Inžinieri)",
                    "philosophy": "Presnosť, astromantická geometria, fotonické kryštály a solárne aerostaty.",
                    "favored_weapon": "Kryštálové meče a lúčové projekčné žiariče.",
                    "aerial_doctrine": "Vysoko-orbitálne vznášajúce sa platformy s aéterickými plynovými bunkami."
                },
                "toxic_tribe": {
                    "name": "Toxický Kmeň (Púštni Vlci z Pustatín)",
                    "philosophy": "Kyselina, prežitie v piesočných búrkach, rádioaktívny sliz a rýchle přepady.",
                    "favored_weapon": "Cenzerové korbáče, rozprašovače spór a zubaté harpuny.",
                    "aerial_doctrine": "Agresívne Rogalo delta-krídla so zubami a dymovými stopami."
                },
                "druid_tribe": {
                    "name": "Druidský Kmeň (Strážcovia Zemských Koreňov)",
                    "philosophy": "Mykorhízne siete, živá kôra, liečenie a prastaré kamenné totemy.",
                    "favored_weapon": "Dubové palice, drevené pavézy a koreňové pasce.",
                    "aerial_doctrine": "Ťažké dreveno-kamenné ostrovy spojené lianovými a lanovými mostami."
                }
            },
            "vehicles_and_tech": {
                "ironclad_mortar_island": {
                    "name": "Železný Vznášajúci sa Ostrov s 240mm Mínometom",
                    "description": "24 000 m³ aéterický aerostat nesúci pevnostnú platformu. Pádová kinetická energia granátu presahuje 9 900 kJ.",
                    "vital_garrison_hp": 6
                },
                "rogallo_delta_glider": {
                    "name": "Rogalo Delta-Krídlo 'Skag'",
                    "description": "Závesný klzák s pomerom kĺzania 7.5:1 a zosilneným titánovým nosníkom. Schopný využívať termické prúdy aj anomálie.",
                    "vital_pilot_hp": 6
                },
                "cyber_camel_caravan": {
                    "name": "Kybernetická Púštna Ťava Korrado",
                    "description": "Bionická ťava s chladeným nákladným priestorom na prenos 4B LLM procesorov cez púšte s teplotou 55°C.",
                    "vital_hp": 6
                },
                "ncon_vr_tactical_hud": {
                    "name": "Ncon by Korrado 130° FOV Headset",
                    "description": "Zobrazuje v reálnom čase balistické paraboly mínometu, napätie Just Cause lana a zdravotný stav (6 Max HP)."
                }
            }
        }

    def get_all_chapters(self) -> List[Dict[str, Any]]:
        """Returns the full list of canonical campaign chapters."""
        return [asdict(c) for c in self.chapters.values()]

    def get_chapter(self, chapter_id: str) -> Optional[Dict[str, Any]]:
        """Returns a single campaign chapter by ID."""
        chapter = self.chapters.get(chapter_id)
        return asdict(chapter) if chapter else None

    def get_codex(self) -> Dict[str, Any]:
        """Returns the complete Lore Codex and Bestiary."""
        return self.codex

    def generate_procedural_mission(
        self,
        altitude_tier: str = "skybridge_midways",
        threat_level: int = 3,
        weather_condition: str = "totem_rift_lightning",
        enemy_faction: str = "toxic_tribe"
    ) -> Dict[str, Any]:
        """
        Procedurally generates a unique high-stakes tactical mission with custom
        objectives, battle conditions, and balanced reward scaling.
        """
        threat_level = max(1, min(5, threat_level))
        mission_id = f"op_{uuid.uuid4().hex[:8]}"

        alt_map = {
            "canyon_dunes": {"name": "Púštny Kaňon Dunes", "alt_m": 45.0, "glide_penalty": 0.8},
            "skybridge_midways": {"name": "Nebeský Mostový Uzol", "alt_m": 220.0, "glide_penalty": 1.0},
            "stratospheric_citadel": {"name": "Stratosférická Citadela", "alt_m": 380.0, "glide_penalty": 1.3}
        }
        alt_info = alt_map.get(altitude_tier, alt_map["skybridge_midways"])

        weather_map = {
            "sandstorm_zero_vis": {"name": "Piesočná Búrka s Nulovou Viditeľnosťou", "wind_mps": 14.0, "mortar_dispersion_mult": 1.6},
            "totem_rift_lightning": {"name": "Totemová Búrka s Azúrovými Bleskami", "wind_mps": 6.5, "mortar_dispersion_mult": 1.1},
            "thermal_updraft_gale": {"name": "Horúci Termický Víchor", "wind_mps": 9.0, "mortar_dispersion_mult": 1.25},
            "calm_golden_hour": {"name": "Pokojná Zlatá Hodina", "wind_mps": 2.0, "mortar_dispersion_mult": 0.9}
        }
        w_info = weather_map.get(weather_condition, weather_map["totem_rift_lightning"])

        base_nuggets = 150 * threat_level
        base_xp = 350 * threat_level

        # Procedural enemies strictly obeying 6 Max HP
        enemy_squads = [
            {"id": f"squad_{i+1}", "type": "Rogalo Raider", "count": 2 + threat_level, "squad_hp": VITAL_MAX_HP}
            for i in range(min(3, threat_level))
        ]
        if threat_level >= 3:
            enemy_squads.append({"id": "boss_island", "type": "Floating Mortar Battery", "count": 1, "squad_hp": VITAL_MAX_HP})

        mission_name = f"Operácia '{random.choice(['Zlatý Blesk', 'Shatterpoint', 'Železný Víchor', 'Kryštálový Úsvit', 'Púštny Orol'])}'"

        return {
            "mission_id": mission_id,
            "mission_name": mission_name,
            "altitude_tier": altitude_tier,
            "altitude_m": alt_info["alt_m"],
            "location_name": alt_info["name"],
            "weather": w_info["name"],
            "wind_speed_mps": w_info["wind_mps"],
            "threat_level": threat_level,
            "enemy_faction": enemy_faction,
            "enemy_squads": enemy_squads,
            "rewards": {
                "aether_nuggets": base_nuggets,
                "gold": base_nuggets * 4,
                "experience_xp": base_xp
            },
            "tactical_briefing": (
                f"V sektore {alt_info['name']} zúri {w_info['name']}. "
                f"Rozpoznané nepriateľské oddiely klanu {enemy_faction}. "
                f"Odporúčaná kombinácia: Strmá paľba mínometu + výsadok na rogalách."
            ),
            "vital_max_hp_rule_observed": True
        }

    def simulate_combat_operation(
        self,
        chapter_id: str,
        chosen_choice_id: str,
        player_squad_hp: int = VITAL_MAX_HP,
        artillery_active: bool = True,
        flight_support_active: bool = True
    ) -> Dict[str, Any]:
        """
        Simulates execution of a campaign chapter operation with player tactical choices.
        Strictly observes 6 Max HP invariant.
        """
        chapter = self.chapters.get(chapter_id)
        if not chapter:
            chapter = list(self.chapters.values())[0]

        choice = next((c for c in chapter.choices if c.id == chosen_choice_id), chapter.choices[0])

        player_squad_hp = max(1, min(VITAL_MAX_HP, player_squad_hp))
        enemy_hp = chapter.enemy_leader_hp

        # Calculate combat rounds
        rounds_log = []
        round_num = 1

        while player_squad_hp > 0 and enemy_hp > 0 and round_num <= 4:
            # Player attack
            base_hit = 2
            if artillery_active:
                base_hit += 1  # Plunging mortar bonus
            if "CRIT" in choice.tactical_modifier or "PLUNGE" in choice.tactical_modifier:
                base_hit += 1

            dmg_to_enemy = min(enemy_hp, min(4, base_hit))
            enemy_hp -= dmg_to_enemy

            # Enemy counter-attack
            incoming_dmg = 1 if flight_support_active else 2
            if choice.risk_level == "VERY_HIGH":
                incoming_dmg += 1

            dmg_to_player = min(player_squad_hp, incoming_dmg)
            player_squad_hp -= dmg_to_player

            rounds_log.append({
                "round": round_num,
                "player_action": f"Kombinovaný úder ({choice.label})",
                "damage_dealt_to_enemy": dmg_to_enemy,
                "enemy_hp_remaining": enemy_hp,
                "damage_taken_by_player": dmg_to_player,
                "player_hp_remaining": player_squad_hp
            })
            round_num += 1

        victory = enemy_hp <= 0 and player_squad_hp > 0

        # Calculate final reward
        bonus_mult = 1.0 + (choice.reward_bonus_pct / 100.0)
        final_nuggets = int(chapter.base_nuggets_reward * bonus_mult) if victory else int(chapter.base_nuggets_reward * 0.25)
        final_xp = int(chapter.base_xp_reward * bonus_mult) if victory else int(chapter.base_xp_reward * 0.25)

        return {
            "chapter_id": chapter.id,
            "chapter_title": chapter.title,
            "choice_made": choice.label,
            "tactical_modifier_used": choice.tactical_modifier,
            "victory": victory,
            "player_final_hp": player_squad_hp,
            "enemy_final_hp": max(0, enemy_hp),
            "rounds_executed": len(rounds_log),
            "combat_log": rounds_log,
            "rewards_awarded": {
                "aether_nuggets": final_nuggets,
                "experience_xp": final_xp,
                "vital_invariant_honored": player_squad_hp <= VITAL_MAX_HP and enemy_hp <= VITAL_MAX_HP
            },
            "summary_text": (
                f"{'VÍŤAZSTVO!' if victory else 'PORÁŽKA!'} Oddiel úspešne vykonal "
                f"taktiku '{choice.label}'. Zostávajúce HP: {player_squad_hp}/{VITAL_MAX_HP}."
            )
        }
