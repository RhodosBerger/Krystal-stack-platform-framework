# ==============================================================================
# KRYSTAL-STACK: DEATH AD PENALTY, ANTI-FRUSTRATION BALANCING & REWARDED QUOTAS
# ==============================================================================
# Implements:
#   1. Death Ad Penalty Evaluator:
#      - Triggers upon death if rounds_won <= 2 and has_plus_membership == False.
#      - Tripartite resolution: [Watch 3 Ads, Pay Nuggets, Buy + Membership].
#   2. Anti-Frustration Fair-Play Balancing:
#      - Early Death Mercy (<60s lifespan skips penalty).
#      - Penalty Cooldown Window (min 600s between forced ads).
#      - Consecutive Loss Protection tokens.
#   3. Rewarded Ad Quota Engine:
#      - Daily Nugget quotas with diminishing returns.
#      - Day streak multiplier (up to +50%).
#      - Daily hard caps preventing hyper-inflation.
# ==============================================================================

import time
import math
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

class PenaltyResolutionType(str, Enum):
    WATCH_3_ADS = "watch_3_ads"
    PAY_NUGGETS = "pay_nuggets"
    BUY_PLUS_MEMBERSHIP = "buy_plus_membership"
    MERCY_REVIVE = "mercy_revive"
    PLUS_EXEMPTION = "plus_exemption"
    SKILL_EXEMPTION = "skill_exemption"


class DeathPenaltyEvaluator:
    """
    Evaluates whether a player death incurs a commercial ad penalty
    and enforces anti-frustration fairness rules.
    """

    COOLDOWN_WINDOW_SEC = 600.0    # 10 minutes between mandatory ad prompts
    MERCY_LIFESPAN_SEC = 60.0      # Dying under 60 seconds triggers mercy pass
    NUGGET_PENALTY_COST = 15       # Cost in Gold Nuggets to skip 3 ads

    @staticmethod
    def evaluate_death_penalty(
        player_id: str,
        rounds_won: int,
        has_plus_membership: bool,
        lifespan_sec: float,
        last_ad_prompt_timestamp: float,
        consecutive_deaths: int = 1
    ) -> Dict[str, Any]:
        """
        Calculates whether a death penalty applies and determines available resolutions.
        """
        now = time.time()
        time_since_last_ad = now - last_ad_prompt_timestamp

        # Rule 1: Paid + Membership has absolute immunity
        if has_plus_membership:
            return {
                "penalty_required": False,
                "reason": "PLUS_MEMBERSHIP_ACTIVE",
                "resolution": PenaltyResolutionType.PLUS_EXEMPTION.value,
                "options": [],
                "nugget_cost": 0,
                "ads_to_watch": 0,
                "message": "KRYSTAL+ AKTÍVNE: Reklamy po smrti sú trvalo vypnuté!"
            }

        # Rule 2: Skill-based exemption: Won more than 2 rounds
        if rounds_won > 2:
            return {
                "penalty_required": False,
                "reason": "SKILL_EXEMPTION_ROUNDS_WON",
                "resolution": PenaltyResolutionType.SKILL_EXEMPTION.value,
                "options": [],
                "nugget_cost": 0,
                "ads_to_watch": 0,
                "message": f"VÍŤAZNÁ ŠNÚRA ({rounds_won} výhier): Výnimočný výkon oslobodzuje od reklamy!"
            }

        # Rule 3: Anti-Frustration: Early Death Mercy
        if lifespan_sec < DeathPenaltyEvaluator.MERCY_LIFESPAN_SEC:
            return {
                "penalty_required": False,
                "reason": "MERCY_SPAWN_REVIVE",
                "resolution": PenaltyResolutionType.MERCY_REVIVE.value,
                "options": [],
                "nugget_cost": 0,
                "ads_to_watch": 0,
                "message": f"MILOSRDENSTVO ARÉNY: Život trval iba {round(lifespan_sec, 1)}s. Okamžitý respawn bez reklamy."
            }

        # Rule 4: Anti-Frustration: Cooldown Window (Max 1 forced ad screen per 10 mins)
        if time_since_last_ad < DeathPenaltyEvaluator.COOLDOWN_WINDOW_SEC:
            remaining_cooldown = int(DeathPenaltyEvaluator.COOLDOWN_WINDOW_SEC - time_since_last_ad)
            return {
                "penalty_required": False,
                "reason": "COOLDOWN_ACTIVE",
                "resolution": PenaltyResolutionType.MERCY_REVIVE.value,
                "options": [],
                "nugget_cost": 0,
                "ads_to_watch": 0,
                "message": f"OCHRANA PRED FRUSTRÁCIOU: Reklamný cooldown aktívny (ešte {remaining_cooldown}s). Pokračujte v hre!"
            }

        # Rule 5: Consecutive loss protection: 3 deaths without win grants discount
        nugget_cost = DeathPenaltyEvaluator.NUGGET_PENALTY_COST
        if consecutive_deaths >= 3:
            nugget_cost = max(5, nugget_cost - 5) # Discounted to 10 nuggets

        # Penalty must be resolved
        options = [
            {
                "type": PenaltyResolutionType.WATCH_3_ADS.value,
                "name": "Pozrieť 3 Reklamy",
                "description": "Odblokuje okamžitý respawn a pridá bonusové nugety do kvóty.",
                "ads_required": 3
            },
            {
                "type": PenaltyResolutionType.PAY_NUGGETS.value,
                "name": f"Zaplatiť {nugget_cost} Nugetov",
                "description": "Okamžité preskočenie bez sledovania reklám.",
                "nuggets_required": nugget_cost
            },
            {
                "type": PenaltyResolutionType.BUY_PLUS_MEMBERSHIP.value,
                "name": "Aktivovať Krystal+ Členstvo",
                "description": "Permanentné odstránenie všetkých reklám po smrti + VIP status.",
                "price_eur": 4.99
            }
        ]

        return {
            "penalty_required": True,
            "reason": "UNPAID_AND_LOW_ROUNDS",
            "consecutive_deaths": consecutive_deaths,
            "rounds_won": rounds_won,
            "options": options,
            "ads_to_watch": 3,
            "nugget_cost": nugget_cost,
            "message": "PENALIZÁCIA ZA PORÁŽKU: Vyberte si spôsob oživenia [3 Reklamy / Nugety / Krystal+]."
        }


class RewardedAdQuotaManager:
    """
    Manages daily quotas, diminishing returns, and Nugget bonuses for watching ads.
    """

    MAX_DAILY_ADS = 15            # Daily maximum rewarded ads
    BASE_NUGGETS_TIER_1 = 10      # Ads 1-5
    BASE_NUGGETS_TIER_2 = 6       # Ads 6-10
    BASE_NUGGETS_TIER_3 = 3       # Ads 11-15

    def __init__(self):
        # player_id -> { "day_key": "YYYY-MM-DD", "ads_watched_today": int, "streak_days": int }
        self.player_quotas: Dict[str, Dict[str, Any]] = {}

    def _get_current_day_key(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def get_player_status(self, player_id: str) -> Dict[str, Any]:
        day_key = self._get_current_day_key()
        record = self.player_quotas.get(player_id)
        if not record or record.get("day_key") != day_key:
            streak = record.get("streak_days", 1) if record else 1
            record = {
                "day_key": day_key,
                "ads_watched_today": 0,
                "streak_days": streak,
                "total_nuggets_earned_today": 0
            }
            self.player_quotas[player_id] = record
        return record

    def calculate_ad_reward(self, player_id: str) -> Dict[str, Any]:
        """
        Calculates the Gold Nugget payout for the NEXT ad watched today.
        Incorporates diminishing returns and day-streak multipliers.
        """
        status = self.get_player_status(player_id)
        watched = status["ads_watched_today"]

        if watched >= self.MAX_DAILY_ADS:
            return {
                "eligible": False,
                "base_nuggets": 0,
                "multiplier": 1.0,
                "final_nuggets": 0,
                "ads_remaining_today": 0,
                "message": f"DENNÝ LIMIT DOSIAHNUTÝ: Maximálna kvóta {self.MAX_DAILY_ADS} reklám na dnes vyčerpaná."
            }

        # Diminishing returns curve
        if watched < 5:
            base = self.BASE_NUGGETS_TIER_1
            tier = "Tier 1: Plná Odmena"
        elif watched < 10:
            base = self.BASE_NUGGETS_TIER_2
            tier = "Tier 2: Mierny Útlm"
        else:
            base = self.BASE_NUGGETS_TIER_3
            tier = "Tier 3: Minimálny Útlm"

        # Streak multiplier (+10% per streak day up to +50%)
        streak = status.get("streak_days", 1)
        streak_bonus = min(0.50, (streak - 1) * 0.10)
        multiplier = round(1.0 + streak_bonus, 2)

        final_payout = int(math.floor(base * multiplier))

        return {
            "eligible": True,
            "ad_index_today": watched + 1,
            "ads_remaining_today": self.MAX_DAILY_ADS - (watched + 1),
            "tier_name": tier,
            "base_nuggets": base,
            "streak_days": streak,
            "streak_multiplier": multiplier,
            "final_nuggets": final_payout,
            "message": f"REKLAMA #{watched + 1}: Získate +{final_payout} Nugetov (Streak: {streak} dní, {int(streak_bonus * 100)}% bonus)."
        }

    def record_watched_ad(self, player_id: str) -> Dict[str, Any]:
        """
        Registers an ad completion, updates daily quota, and awards Gold Nuggets.
        """
        reward_info = self.calculate_ad_reward(player_id)
        if not reward_info["eligible"]:
            return reward_info

        status = self.get_player_status(player_id)
        status["ads_watched_today"] += 1
        status["total_nuggets_earned_today"] += reward_info["final_nuggets"]

        return {
            "success": True,
            "awarded_nuggets": reward_info["final_nuggets"],
            "ads_watched_today": status["ads_watched_today"],
            "ads_remaining_today": self.MAX_DAILY_ADS - status["ads_watched_today"],
            "total_nuggets_today": status["total_nuggets_earned_today"],
            "streak_days": status["streak_days"]
        }
