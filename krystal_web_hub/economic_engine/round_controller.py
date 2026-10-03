# ==============================================================================
# KRYSTAL-STACK: ROUND CONTROLLER & ESCALATION STATE MACHINE
# ==============================================================================
# Manages multi-phase gradated rounds:
#   Phase 1: Economy & Resource Yields
#   Phase 2: Build & Infrastructure
#   Phase 3: Skirmish & Combos
#   Phase 4: Ledger Audit & Escalation Gradation (Skirmish -> Surge -> Apex -> Cataclysm)
# ==============================================================================

import time
from typing import Dict, Any, Tuple
from .models import (
    MatchState, RoundPhase, EscalationStage, LedgerEntry, CombatantState, Tribe
)
from .economic_rules import EconomicLedger
from .ability_framework import AbilityEngine, ABILITY_REGISTRY
from .sector_conquest import generate_default_sectors, SectorConquestEngine

class RoundController:
    @staticmethod
    def initialize_match(match_id: str = "match_001", player_tribe: Tribe = Tribe.CRYSTAL, enemy_tribe: Tribe = Tribe.TOXIC) -> MatchState:
        player = CombatantState(name="Hráč (Kmeň)", tribe=player_tribe, hp=6, max_hp=6, armor=0, mana=8, max_mana=15, aether_crystals=3, toxic_slime=2, amber_runes=2)
        enemy = CombatantState(name="Nepriateľský Kmeň", tribe=enemy_tribe, hp=6, max_hp=6, armor=0, mana=8, max_mana=15, aether_crystals=2, toxic_slime=3, amber_runes=2)
        
        match = MatchState(
            match_id=match_id,
            round_number=1,
            phase=RoundPhase.PHASE_1_ECONOMY,
            escalation=EscalationStage.ROUND_1_SKIRMISH,
            player=player,
            enemy=enemy,
            sectors=generate_default_sectors(),
            ledger=[
                LedgerEntry(
                    timestamp_turn=1,
                    round_number=1,
                    phase=RoundPhase.PHASE_1_ECONOMY.value,
                    event_type="MATCH_INIT",
                    description="Zápas inicializovaný. Fáza: 1. Kolo - Prieskumná potýčka (Skirmish). Sektory aktivované.",
                    resource_delta={"mana": 8},
                    balance_after=EconomicLedger.get_current_balances(player)
                )
            ]
        )
        return match

    @staticmethod
    def step_phase(match: MatchState) -> Dict[str, Any]:
        """Advances the match phase cyclically through Phase 1 -> 2 -> 3 -> 4 -> Next Round."""
        if match.winner:
            return {"status": "MATCH_OVER", "winner": match.winner}

        events_log = []

        if match.phase == RoundPhase.PHASE_1_ECONOMY:
            # ── PHASE 1: COLLECT YIELDS (BUILDINGS + SECTORS) ──────────────────
            player_yield = EconomicLedger.calculate_round_yield(match.player, match.escalation)
            enemy_yield = EconomicLedger.calculate_round_yield(match.enemy, match.escalation)

            # Collect yields from captured/controlled sectors
            sector_yields = SectorConquestEngine.apply_sector_round_yields(match)
            if any(v > 0 for v in sector_yields["player"].values()):
                p_sec = sector_yields["player"]
                events_log.append(f"Výnosy z kontrolovaných sektorov (Hráč): +{p_sec['mana']} Many, +{p_sec['aether_crystal']} Kryštálov, +{p_sec['toxic_slime']} Slizu, +{p_sec['amber_rune']} Jantáru.")

            # World tree passive heal check
            for b in match.player.buildings:
                if b.active and "world_tree" in b.id:
                    match.player.hp = min(match.player.max_hp, match.player.hp + 1)
                    events_log.append("Výhonok Stromu Sveta obnovil +1 HP hráčovi.")

            entry = LedgerEntry(
                timestamp_turn=int(time.time()),
                round_number=match.round_number,
                phase=match.phase.value,
                event_type="ECONOMY_YIELD",
                description=f"Výnosy kola {match.round_number}: +{player_yield['mana']} Many, +{player_yield['aether_crystal']} Kryštálov, +{player_yield['toxic_slime']} Slizu, +{player_yield['amber_rune']} Jantáru.",
                resource_delta=player_yield,
                balance_after=EconomicLedger.get_current_balances(match.player)
            )
            match.ledger.append(entry)

            # Move to Build Phase
            match.phase = RoundPhase.PHASE_2_BUILD
            events_log.append("Fáza 1 dokončená. Prechod do Fázy 2: Stavba infraštruktúry.")

        elif match.phase == RoundPhase.PHASE_2_BUILD:
            # Move to Skirmish Phase
            match.phase = RoundPhase.PHASE_3_SKIRMISH
            events_log.append("Fáza 2 dokončená. Prechod do Fázy 3: Taktický boj a sektorové potýčky.")

        elif match.phase == RoundPhase.PHASE_3_SKIRMISH:
            # ── PHASE 3: TACTICAL SKIRMISH & AUTONOMOUS AI SECTOR ASSAULT ──────
            # Enemy AI launches sector offensive if sufficient mana is available
            if match.enemy.mana >= 3:
                target_candidates = [s for s in match.sectors if s.owner != "enemy"]
                # Priority 1: Citadel if neutral/player, else neutral sectors, else player sector
                citadel = next((s for s in target_candidates if s.id == "sector_center_citadel"), None)
                target_sector = citadel if citadel else (target_candidates[0] if target_candidates else None)
                if target_sector:
                    ai_attack_power = 2 + match.round_number
                    success, assault_res, msg = SectorConquestEngine.resolve_sector_assault(
                        match, target_sector.id, attacker_side="enemy", attack_power=ai_attack_power, mana_committed=2
                    )
                    if success:
                        events_log.append(
                            f"⚔️ Nepriateľský výpad! Útok na sektor '{target_sector.name}' silou {assault_res['attack_power_used']}. "
                            f"(Obrana sektoru: {assault_res['fort_hp_left']} HP)."
                        )
                        if assault_res.get("captured"):
                            events_log.append(f"⚠️ Sektor '{target_sector.name}' padol do rúk nepriateľa!")

            # Move to Resolution & Escalation Audit Phase
            match.phase = RoundPhase.PHASE_4_AUDIT_ESCALATE
            events_log.append("Fáza 3 dokončená. Prechod do Fázy 4: Ledger Audit a Gradácia kola.")


        elif match.phase == RoundPhase.PHASE_4_AUDIT_ESCALATE:
            # ── PHASE 4: STATUS TICKS, AUDIT & ROUND ESCALATION ───────────────
            # Process Poison on Player
            if "poisoned" in match.player.active_statuses:
                match.player.hp = max(0, match.player.hp - 1)
                events_log.append("Jed spôsobil -1 poškodenie hráčovi.")
            if "lethal_plague" in match.player.active_statuses:
                match.player.hp = max(0, match.player.hp - 2)
                events_log.append("Smrteľná nákaza spôsobila -2 poškodenia hráčovi.")

            # Process Poison on Enemy
            if "poisoned" in match.enemy.active_statuses:
                match.enemy.hp = max(0, match.enemy.hp - 1)
                events_log.append("Jed spôsobil -1 poškodenie nepriateľovi.")
            if "lethal_plague" in match.enemy.active_statuses:
                match.enemy.hp = max(0, match.enemy.hp - 2)
                events_log.append("Smrteľná nákaza spôsobila -2 poškodenia nepriateľovi.")

            # Decrement status durations
            for target_c in [match.player, match.enemy]:
                expired = []
                for s, dur in target_c.active_statuses.items():
                    if dur <= 1:
                        expired.append(s)
                    else:
                        target_c.active_statuses[s] = dur - 1
                for s in expired:
                    del target_c.active_statuses[s]
                    events_log.append(f"Stav '{s}' vypršal.")

            # Check Win/Loss
            if match.player.hp <= 0 and match.enemy.hp <= 0:
                match.winner = "DRAW"
                events_log.append("REMIZA: Obaja bojovníci padli v rovnakom kole.")
            elif match.enemy.hp <= 0:
                match.winner = "PLAYER"
                events_log.append("VÍŤAZSTVO: Hráč porazil nepriateľský kmeň!")
            elif match.player.hp <= 0:
                match.winner = "ENEMY"
                events_log.append("PORÁŽKA: Nepriateľ ovládol arénu.")

            # Advance Round Number & Escalation
            match.round_number += 1
            if match.round_number == 2:
                match.escalation = EscalationStage.ROUND_2_SURGE
                events_log.append("GRADÁCIA: 2. Kolo - Priemyselný nárast (Industrial Surge). Výnosy +50%!")
            elif match.round_number == 3:
                match.escalation = EscalationStage.ROUND_3_APEX
                events_log.append("GRADÁCIA: 3. Kolo - Totálna vojna (Total War / Apex). Ultimátne schopnosti odomknuté!")
            elif match.round_number >= 4:
                match.escalation = EscalationStage.ROUND_4_CATACLYSM
                events_log.append("GRADÁCIA: 4. Kolo - Kataklyzma (Cataclysm)! Ceny Many sa zdvojnásobujú!")

            # Reset back to Economy Phase for the new round
            match.phase = RoundPhase.PHASE_1_ECONOMY
            events_log.append(f"Začína nové kolo: {match.round_number} ({match.escalation.value}).")

            # Ledger audit entry
            entry = LedgerEntry(
                timestamp_turn=int(time.time()),
                round_number=match.round_number - 1,
                phase="audit",
                event_type="ROUND_AUDIT",
                description=f"Audit kola {match.round_number - 1}. Hráč HP: {match.player.hp}/6, Nepriateľ HP: {match.enemy.hp}/6.",
                resource_delta={},
                balance_after=EconomicLedger.get_current_balances(match.player)
            )
            match.ledger.append(entry)

        return {
            "round_number": match.round_number,
            "phase": match.phase.value,
            "escalation": match.escalation.value,
            "events": events_log,
            "match_state": match.to_dict()
        }
