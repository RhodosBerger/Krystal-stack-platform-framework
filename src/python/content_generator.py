"""
Game Content Generator - Dynamic Content from Unified Feature Pipeline

Links all systems together:
- Telemetry → Feature Engine → Signals
- LLM Bridge → Content Generation
- Economic Engine → Resource-aware generation
- Allocation → Budget-constrained assets
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from .runtime import Runtime
from .feature_engine import FeatureEngine, ScaleParams
from .signals import SignalScheduler, Signal, SignalKind, telemetry_signal
from .allocation import Allocator, AllocationRequest, ResourceType, Priority, create_default_allocator
from .llm_bridge import LLMBridge, create_llm_bridge

logger = logging.getLogger(__name__)


class ContentType(Enum):
    LEVEL = "level"
    QUEST = "quest"
    DIALOGUE = "dialogue"
    ITEM = "item"
    ENEMY = "enemy"
    EVENT = "event"
    TEXTURE = "texture"
    AUDIO = "audio"


@dataclass
class GameContext:
    """Current game state for content generation."""
    player_level: int = 1
    difficulty: float = 0.5
    biome: str = "forest"
    time_of_day: float = 0.5  # 0=midnight, 0.5=noon
    weather: str = "clear"
    active_quests: List[str] = field(default_factory=list)
    inventory_value: float = 0.0
    playtime_hours: float = 0.0
    session_fps: float = 60.0
    thermal_state: float = 0.5  # 0=cool, 1=hot


@dataclass
class GeneratedContent:
    """Generated game content."""
    content_type: ContentType
    data: Dict[str, Any]
    quality_tier: int  # 1-5, based on available resources
    generation_cost: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContentGenerator:
    """
    Unified content generation pipeline.

    Flow:
    1. Fetch game context → Runtime variables
    2. Transform features → Scaled/derived values
    3. Generate signals → Priority scheduling
    4. Allocate resources → Budget for generation
    5. Generate content → LLM-powered creation
    6. Post-process → Quality scaling based on resources
    """

    def __init__(
        self,
        llm: Optional[LLMBridge] = None,
        allocator: Optional[Allocator] = None,
    ):
        self.runtime = Runtime()
        self.features = FeatureEngine()
        self.scheduler = SignalScheduler()
        self.allocator = allocator or create_default_allocator()
        self.llm = llm or create_llm_bridge()

        # Register game-specific computed variables
        self._setup_computed_vars()

    def _setup_computed_vars(self):
        """Register game-specific feature computations."""
        self.runtime.register_computed_var("challenge_factor",
            "difficulty * (1 + player_level * 0.1)")
        self.runtime.register_computed_var("resource_pressure",
            "(1 - session_fps / 60) * thermal_state")
        self.runtime.register_computed_var("content_budget",
            "clamp(1 - resource_pressure, 0.3, 1.0)")
        self.runtime.register_computed_var("time_factor",
            "sin(time_of_day * 3.14159)")

    def update_context(self, ctx: GameContext):
        """Update runtime with game context."""
        self.runtime.update_telemetry_dict({
            "player_level": ctx.player_level,
            "difficulty": ctx.difficulty,
            "time_of_day": ctx.time_of_day,
            "session_fps": ctx.session_fps,
            "thermal_state": ctx.thermal_state,
            "inventory_value": ctx.inventory_value,
            "playtime_hours": ctx.playtime_hours,
        })

        # Generate signals based on context
        if ctx.session_fps < 30:
            self.scheduler.enqueue(telemetry_signal(
                SignalKind.FRAMETIME_SPIKE, 0.8, fps=ctx.session_fps))
        if ctx.thermal_state > 0.8:
            self.scheduler.enqueue(telemetry_signal(
                SignalKind.THERMAL_WARNING, 0.7, thermal=ctx.thermal_state))

    def _allocate_generation_budget(self, content_type: ContentType) -> Dict[str, float]:
        """Allocate resources for content generation."""
        # Request CPU for generation
        cpu_req = AllocationRequest(
            id=f"gen_{content_type.value}",
            resource_type=ResourceType.CPU_TIME,
            amount=100,  # 100μs base
            priority=Priority.NORMAL,
        )
        cpu_alloc = self.allocator.allocate(cpu_req)

        # Request memory for assets
        mem_req = AllocationRequest(
            id=f"mem_{content_type.value}",
            resource_type=ResourceType.MEMORY,
            amount=64,  # 64MB base
            priority=Priority.NORMAL,
        )
        mem_alloc = self.allocator.allocate(mem_req)

        return {
            "cpu_budget": cpu_alloc.amount if cpu_alloc else 50,
            "memory_budget": mem_alloc.amount if mem_alloc else 32,
        }

    def _determine_quality_tier(self, budget: Dict[str, float]) -> int:
        """Determine content quality based on resource budget."""
        resource_pressure = self.runtime.fetch_var("resource_pressure")
        content_budget = self.runtime.fetch_var("content_budget")

        # Scale 1-5 based on available resources
        score = content_budget * (budget.get("cpu_budget", 50) / 100)

        if score > 0.8:
            return 5  # Ultra quality
        elif score > 0.6:
            return 4  # High quality
        elif score > 0.4:
            return 3  # Medium quality
        elif score > 0.2:
            return 2  # Low quality
        return 1  # Minimal quality

    def generate_level(self, ctx: GameContext) -> GeneratedContent:
        """Generate a procedural level."""
        self.update_context(ctx)
        budget = self._allocate_generation_budget(ContentType.LEVEL)
        quality = self._determine_quality_tier(budget)

        challenge = self.runtime.fetch_var("challenge_factor")
        time_factor = self.runtime.fetch_var("time_factor")

        prompt = f"""Generate a {ctx.biome} game level for player level {ctx.player_level}.
Challenge factor: {challenge:.2f}
Time of day: {"day" if time_factor > 0 else "night"}
Weather: {ctx.weather}
Quality tier: {quality}/5

Return JSON with: name, description, enemy_count, treasure_count, hazards, layout_hints"""

        response = self.llm.generate(prompt, system="You are a game level designer. Return valid JSON only.")

        try:
            data = json.loads(response.content)
        except:
            data = {"name": f"{ctx.biome.title()} Area", "description": response.content}

        # Scale content based on quality
        if "enemy_count" in data:
            data["enemy_count"] = int(data["enemy_count"] * (quality / 3))

        return GeneratedContent(
            content_type=ContentType.LEVEL,
            data=data,
            quality_tier=quality,
            generation_cost=budget,
            metadata={"challenge": challenge, "biome": ctx.biome}
        )

    def generate_quest(self, ctx: GameContext, theme: str = "exploration") -> GeneratedContent:
        """Generate a dynamic quest."""
        self.update_context(ctx)
        budget = self._allocate_generation_budget(ContentType.QUEST)
        quality = self._determine_quality_tier(budget)

        prompt = f"""Generate a {theme} quest for a level {ctx.player_level} player.
Current biome: {ctx.biome}
Active quests: {len(ctx.active_quests)}
Playtime: {ctx.playtime_hours:.1f} hours
Quality tier: {quality}/5

Return JSON with: title, description, objectives, rewards, difficulty, estimated_time"""

        response = self.llm.generate(prompt, system="You are a quest designer. Return valid JSON only.")

        try:
            data = json.loads(response.content)
        except:
            data = {"title": f"{theme.title()} Quest", "description": response.content}

        return GeneratedContent(
            content_type=ContentType.QUEST,
            data=data,
            quality_tier=quality,
            generation_cost=budget,
        )

    def generate_dialogue(self, ctx: GameContext, npc_type: str, situation: str) -> GeneratedContent:
        """Generate NPC dialogue."""
        self.update_context(ctx)
        budget = self._allocate_generation_budget(ContentType.DIALOGUE)
        quality = self._determine_quality_tier(budget)

        prompt = f"""Generate dialogue for a {npc_type} NPC in situation: {situation}
Player level: {ctx.player_level}
Time: {"day" if self.runtime.fetch_var("time_factor") > 0 else "night"}
Quality tier: {quality}/5

Return JSON with: greeting, options (array of player choices), responses (for each option)"""

        response = self.llm.generate(prompt, system="You are a game writer. Return valid JSON only.")

        try:
            data = json.loads(response.content)
        except:
            data = {"greeting": response.content, "options": []}

        return GeneratedContent(
            content_type=ContentType.DIALOGUE,
            data=data,
            quality_tier=quality,
            generation_cost=budget,
        )

    def generate_item(self, ctx: GameContext, item_class: str = "weapon") -> GeneratedContent:
        """Generate a procedural item."""
        self.update_context(ctx)
        budget = self._allocate_generation_budget(ContentType.ITEM)
        quality = self._determine_quality_tier(budget)

        # Use feature scaling for item stats
        base_power = ctx.player_level * 10
        scaled_power = self.features.scale_abt(
            base_power / 100,
            ScaleParams(alpha=1.2, beta=0.1 * ctx.difficulty, theta=0.0)
        ) * 100

        prompt = f"""Generate a {item_class} for player level {ctx.player_level}.
Base power: {scaled_power:.0f}
Rarity tier: {quality}/5
Biome theme: {ctx.biome}

Return JSON with: name, description, stats, rarity, special_effects"""

        response = self.llm.generate(prompt, system="You are an item designer. Return valid JSON only.")

        try:
            data = json.loads(response.content)
        except:
            data = {"name": f"Mysterious {item_class.title()}", "stats": {"power": scaled_power}}

        return GeneratedContent(
            content_type=ContentType.ITEM,
            data=data,
            quality_tier=quality,
            generation_cost=budget,
            metadata={"scaled_power": scaled_power}
        )

    def generate_enemy(self, ctx: GameContext, enemy_type: str = "creature") -> GeneratedContent:
        """Generate a procedural enemy."""
        self.update_context(ctx)
        budget = self._allocate_generation_budget(ContentType.ENEMY)
        quality = self._determine_quality_tier(budget)

        challenge = self.runtime.fetch_var("challenge_factor")

        prompt = f"""Generate a {enemy_type} enemy for player level {ctx.player_level}.
Challenge factor: {challenge:.2f}
Biome: {ctx.biome}
Quality tier: {quality}/5

Return JSON with: name, description, health, damage, abilities, drops, behavior_hints"""

        response = self.llm.generate(prompt, system="You are a creature designer. Return valid JSON only.")

        try:
            data = json.loads(response.content)
        except:
            data = {"name": f"Wild {enemy_type.title()}", "health": ctx.player_level * 50}

        # Scale stats based on challenge
        if "health" in data:
            data["health"] = int(data["health"] * challenge)
        if "damage" in data:
            data["damage"] = int(data["damage"] * challenge)

        return GeneratedContent(
            content_type=ContentType.ENEMY,
            data=data,
            quality_tier=quality,
            generation_cost=budget,
            metadata={"challenge": challenge}
        )

    def generate_event(self, ctx: GameContext, event_trigger: str = "exploration") -> GeneratedContent:
        """Generate a random event."""
        self.update_context(ctx)
        budget = self._allocate_generation_budget(ContentType.EVENT)
        quality = self._determine_quality_tier(budget)

        time_factor = self.runtime.fetch_var("time_factor")

        prompt = f"""Generate a {event_trigger} event for the game.
Player level: {ctx.player_level}
Biome: {ctx.biome}
Time: {"day" if time_factor > 0 else "night"}, weather: {ctx.weather}
Quality tier: {quality}/5

Return JSON with: title, description, choices, outcomes, rewards_risk"""

        response = self.llm.generate(prompt, system="You are an event designer. Return valid JSON only.")

        try:
            data = json.loads(response.content)
        except:
            data = {"title": f"Mysterious {event_trigger.title()}", "description": response.content}

        return GeneratedContent(
            content_type=ContentType.EVENT,
            data=data,
            quality_tier=quality,
            generation_cost=budget,
        )

    def batch_generate(self, ctx: GameContext, content_types: List[ContentType]) -> List[GeneratedContent]:
        """Generate multiple content pieces efficiently."""
        results = []

        for ct in content_types:
            if ct == ContentType.LEVEL:
                results.append(self.generate_level(ctx))
            elif ct == ContentType.QUEST:
                results.append(self.generate_quest(ctx))
            elif ct == ContentType.ITEM:
                results.append(self.generate_item(ctx))
            elif ct == ContentType.ENEMY:
                results.append(self.generate_enemy(ctx))
            elif ct == ContentType.EVENT:
                results.append(self.generate_event(ctx))

        return results

    def get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about content generation."""
        return {
            "allocator_stats": {k.value: v.__dict__ for k, v in self.allocator.stats().items()},
            "llm_cache": self.llm.cache.stats(),
            "pending_signals": len(self.scheduler._queue),
        }


# Factory
def create_content_generator(llm_url: str = "http://localhost:1234/v1") -> ContentGenerator:
    """Create content generator with default configuration."""
    llm = create_llm_bridge(base_url=llm_url)
    allocator = create_default_allocator()
    return ContentGenerator(llm=llm, allocator=allocator)
