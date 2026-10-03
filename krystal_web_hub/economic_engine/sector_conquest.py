# ==============================================================================
# KRYSTAL-STACK: SECTOR CONQUEST & TERRITORIAL FIGHT SYSTEM
# ==============================================================================
# Implements strategic sector warfare across the tactical hex map:
#   1. Sector Generation: North Crystals, South Slime Bog, East Sacred Grove,
#      West Sulfur Pass, and the Central Aether Citadel.
#   2. Sector Assault & Siege Logic: Fortification breaches, garrison combat,
#      capture progress, and territorial annexation.
#   3. Fortification & Reinforcement: Building walls, upgrading defenses.
#   4. Economic Annexation: Adding sector resource bonuses into the ledger.
# ==============================================================================

import time
from typing import List, Dict, Any, Tuple, Optional
from .models import (
    Tribe, Sector, SectorContestStatus, GarrisonUnit, CombatantState,
    MatchState, LedgerEntry, EscalationStage
)

# ------------------------------------------------------------------------------
# 1. DEFAULT SECTORS GENERATOR
# ------------------------------------------------------------------------------
def generate_default_sectors() -> List[Sector]:
    """Divides the 19-hex board into 5 contested strategic sectors."""
    return [
        Sector(
            id="sector_north_crystal",
            name="Severné Ľadovce (Kryštálový Hrebeň)",
            owner="player",
            tribe_alignment=Tribe.CRYSTAL,
            hex_tiles=[[0, -2], [1, -2], [0, -1], [1, -1]],
            fortification_level=2,
            fortification_hp=8,
            max_fortification_hp=8,
            garrison=[
                GarrisonUnit(id="g_cryst_1", name="Kryštálový Strážca", tribe=Tribe.CRYSTAL, count=2, attack=3, defense=3, hp=4, max_hp=4)
            ],
            resource_bonus={"mana": 2, "aether_crystal": 2},
            status=SectorContestStatus.UNCONTESTED,
            capture_progress=0,
            color="#00ffff",
            description="Severná bašta chrániaca primárne aéterové žily. Bohatá na kryštály."
        ),
        Sector(
            id="sector_south_toxic",
            name="Toxická Kotlina (Kyslé Močiare)",
            owner="enemy",
            tribe_alignment=Tribe.TOXIC,
            hex_tiles=[[-1, 1], [0, 1], [-1, 2], [0, 2]],
            fortification_level=2,
            fortification_hp=7,
            max_fortification_hp=7,
            garrison=[
                GarrisonUnit(id="g_tox_1", name="Slizový Pľuvač", tribe=Tribe.TOXIC, count=3, attack=3, defense=2, hp=3, max_hp=3)
            ],
            resource_bonus={"mana": 2, "toxic_slime": 2},
            status=SectorContestStatus.UNCONTESTED,
            capture_progress=0,
            color="#39ff14",
            description="Rozsiahle nádrže leptavého slizu. Nepriateľská základňa."
        ),
        Sector(
            id="sector_east_druid",
            name="Svätyňa Hlbokého Lesa (Posvätné Duby)",
            owner="neutral",
            tribe_alignment=Tribe.DRUID,
            hex_tiles=[[1, 0], [2, -1], [2, 0]],
            fortification_level=1,
            fortification_hp=6,
            max_fortification_hp=6,
            garrison=[
                GarrisonUnit(id="g_druid_1", name="Prastarý Ent", tribe=Tribe.DRUID, count=1, attack=4, defense=4, hp=6, max_hp=6)
            ],
            resource_bonus={"mana": 2, "amber_rune": 2},
            status=SectorContestStatus.UNCONTESTED,
            capture_progress=0,
            color="#2e8b57",
            description="Posvätný háj s vysokou regeneráciou a zásobami jantáru."
        ),
        Sector(
            id="sector_west_sulfur",
            name="Sírny Priesmyk (Pustinný Kaňon)",
            owner="neutral",
            tribe_alignment=Tribe.TOXIC,
            hex_tiles=[[-2, 0], [-2, 1], [-1, 0]],
            fortification_level=1,
            fortification_hp=5,
            max_fortification_hp=5,
            garrison=[
                GarrisonUnit(id="g_sulfur_1", name="Púštny Náhončí", tribe=Tribe.TOXIC, count=2, attack=2, defense=1, hp=3, max_hp=3)
            ],
            resource_bonus={"mana": 1, "toxic_slime": 1},
            status=SectorContestStatus.UNCONTESTED,
            capture_progress=0,
            color="#9d4edd",
            description="Úzky koridor spájajúci kotlinu so severom. Taktická výhoda."
        ),
        Sector(
            id="sector_center_citadel",
            name="Aéterová Citadela (Centrálny Nexus)",
            owner="neutral",
            tribe_alignment=Tribe.NEUTRAL,
            hex_tiles=[[0, 0]],
            fortification_level=3,
            fortification_hp=12,
            max_fortification_hp=12,
            garrison=[
                GarrisonUnit(id="g_citadel_guard", name="Strážca Nexu", tribe=Tribe.NEUTRAL, count=2, attack=4, defense=5, hp=8, max_hp=8)
            ],
            resource_bonus={"mana": 4, "aether_crystal": 1, "amber_rune": 1},
            status=SectorContestStatus.UNCONTESTED,
            capture_progress=0,
            color="#ffd700",
            description="Centrálna pevnosť na bojisku. Kontrola zaručuje masívny prílev Many a víťaznú dominanciu!"
        )
    ]

# ------------------------------------------------------------------------------
# 2. SECTOR COMBAT & ASSAULT ENGINE
# ------------------------------------------------------------------------------
class SectorConquestEngine:
    @staticmethod
    def get_sector_by_id(match: MatchState, sector_id: str) -> Optional[Sector]:
        for s in match.sectors:
            if s.id == sector_id:
                return s
        return None

    @staticmethod
    def resolve_sector_assault(
        match: MatchState,
        sector_id: str,
        attacker_side: str, # "player" or "enemy"
        attack_power: int,
        mana_committed: int = 2
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Executes an assault against a sector, resolving fortification and garrison defense."""
        sector = SectorConquestEngine.get_sector_by_id(match, sector_id)
        if not sector:
            return False, {}, f"Sektor '{sector_id}' neexistuje!"

        attacker = match.player if attacker_side == "player" else match.enemy
        defender = match.enemy if attacker_side == "player" else match.player

        # Check if already owned
        if sector.owner == attacker_side:
            return False, {}, f"Sektor '{sector.name}' už vlastníte!"

        # Mana cost for mobilizing troops
        if attacker.mana < mana_committed:
            return False, {}, f"Nedostatok Many na útok na sektor: potrebné {mana_committed}, k dispozícii {attacker.mana}"

        attacker.mana -= mana_committed

        # Escalation bonus
        effective_power = attack_power
        if match.escalation == EscalationStage.ROUND_2_SURGE:
            effective_power = int(effective_power * 1.25)
        elif match.escalation in [EscalationStage.ROUND_3_APEX, EscalationStage.ROUND_4_CATACLYSM]:
            effective_power = int(effective_power * 1.5)

        fort_damage = 0
        garrison_damage = 0
        captured = False
        remaining_power = effective_power

        # 1. Breach Fortification Walls
        if sector.fortification_hp > 0:
            fort_damage = min(sector.fortification_hp, remaining_power)
            sector.fortification_hp -= fort_damage
            remaining_power -= fort_damage
            sector.status = SectorContestStatus.UNDER_SIEGE

        # 2. Damage Garrison Units
        if remaining_power > 0 and sector.garrison:
            for g in sector.garrison:
                if remaining_power <= 0:
                    break
                g_dmg = min(g.hp, remaining_power)
                g.hp -= g_dmg
                remaining_power -= g_dmg
                garrison_damage += g_dmg

            # Remove slain garrison units
            sector.garrison = [g for g in sector.garrison if g.hp > 0]

        # 3. Capture Progress Calculation
        if sector.fortification_hp == 0 and not sector.garrison:
            # Full capture
            sector.capture_progress = 100
            captured = True
            old_owner = sector.owner
            sector.owner = attacker_side
            sector.status = SectorContestStatus.CAPTURED
            sector.fortification_hp = sector.max_fortification_hp // 2 # Partial repair on capture

            # Award conquest bounty
            attacker.mana = min(attacker.max_mana, attacker.mana + 4)
            attacker.aether_crystals += 1

            # Log conquest
            entry = LedgerEntry(
                timestamp_turn=int(time.time()),
                round_number=match.round_number,
                phase="sector_conquest",
                event_type="SECTOR_ANNEXED",
                description=f"SEKTOR DOBYTÝ: {attacker.name} ovládol {sector.name}! (Pôvodný vlastník: {old_owner}). Prílev surovín: {sector.resource_bonus}.",
                resource_delta={"mana": 4, "aether_crystal": 1},
                balance_after={"mana": attacker.mana, "aether_crystal": attacker.aether_crystals}
            )
            match.ledger.append(entry)
        else:
            # Partial siege progress
            progress_inc = int((effective_power / 10.0) * 35)
            sector.capture_progress = min(99, sector.capture_progress + progress_inc)
            sector.status = SectorContestStatus.CONTESTED

        return True, {
            "sector_id": sector.id,
            "sector_name": sector.name,
            "attacker": attacker_side,
            "attack_power_used": effective_power,
            "fort_damage": fort_damage,
            "fort_hp_left": sector.fortification_hp,
            "garrison_damage": garrison_damage,
            "garrison_alive": len(sector.garrison),
            "capture_progress": sector.capture_progress,
            "captured": captured,
            "new_owner": sector.owner,
            "status": sector.status.value
        }, "SUCCESS"

    @staticmethod
    def fortify_sector(
        match: MatchState,
        sector_id: str,
        side: str = "player"
    ) -> Tuple[bool, Dict[str, Any], str]:
        """Repairs walls and stations reinforcement garrison in an owned sector."""
        sector = SectorConquestEngine.get_sector_by_id(match, sector_id)
        if not sector:
            return False, {}, "Sektor nenájdený!"

        combatant = match.player if side == "player" else match.enemy

        if sector.owner != side:
            return False, {}, "Môžete opevňovať iba sektory pod vašou kontrolou!"

        # Cost: 3 Mana + 1 tribal resource
        cost_mana = 3
        if combatant.mana < cost_mana:
            return False, {}, f"Nedostatok Many: potrebné {cost_mana}, k dispozícii {combatant.mana}"

        combatant.mana -= cost_mana
        sector.fortification_level = min(5, sector.fortification_level + 1)
        sector.max_fortification_hp += 3
        sector.fortification_hp = sector.max_fortification_hp
        sector.status = SectorContestStatus.UNCONTESTED

        # Add reinforcement garrison unit
        new_unit = GarrisonUnit(
            id=f"guard_{sector.id}_{len(sector.garrison) + 1}",
            name=f"Opevnená Hliadka ({sector.name[:12]})",
            tribe=combatant.tribe,
            count=1,
            attack=2,
            defense=3,
            hp=4,
            max_hp=4
        )
        sector.garrison.append(new_unit)

        entry = LedgerEntry(
            timestamp_turn=int(time.time()),
            round_number=match.round_number,
            phase="sector_fortify",
            event_type="SECTOR_FORTIFIED",
            description=f"Opevnenie sektoru {sector.name} zvýšené na úroveň {sector.fortification_level} (HP: {sector.fortification_hp}). Pridána hliadka.",
            resource_delta={"mana": -cost_mana},
            balance_after={"mana": combatant.mana}
        )
        match.ledger.append(entry)

        return True, {
            "sector_id": sector.id,
            "fortification_level": sector.fortification_level,
            "fortification_hp": sector.fortification_hp,
            "garrison_count": len(sector.garrison)
        }, "SUCCESS"

    @staticmethod
    def apply_sector_round_yields(match: MatchState) -> Dict[str, Dict[str, int]]:
        """Collects resource yields from all controlled sectors for each combatant."""
        player_gains = {"mana": 0, "aether_crystal": 0, "toxic_slime": 0, "amber_rune": 0}
        enemy_gains = {"mana": 0, "aether_crystal": 0, "toxic_slime": 0, "amber_rune": 0}

        for s in match.sectors:
            if s.owner == "player":
                for res, amt in s.resource_bonus.items():
                    if res in player_gains:
                        player_gains[res] += amt
            elif s.owner == "enemy":
                for res, amt in s.resource_bonus.items():
                    if res in enemy_gains:
                        enemy_gains[res] += amt

        # Apply to combatant pools
        match.player.mana = min(match.player.max_mana, match.player.mana + player_gains["mana"])
        match.player.aether_crystals += player_gains["aether_crystal"]
        match.player.toxic_slime += player_gains["toxic_slime"]
        match.player.amber_runes += player_gains["amber_rune"]

        match.enemy.mana = min(match.enemy.max_mana, match.enemy.mana + enemy_gains["mana"])
        match.enemy.aether_crystals += enemy_gains["aether_crystal"]
        match.enemy.toxic_slime += enemy_gains["toxic_slime"]
        match.enemy.amber_runes += enemy_gains["amber_rune"]

        return {
            "player": player_gains,
            "enemy": enemy_gains
        }
