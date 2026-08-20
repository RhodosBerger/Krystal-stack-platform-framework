"""
GAMESA Generators - Algorithmic Content Generation

New generators that branch from existing systems:
1. PolicyGenerator - Generate optimization policies from patterns
2. RuleGenerator - Auto-generate MicroInferenceRules from data
3. ConfigGenerator - Generate optimal configurations
4. ProfileGenerator - Generate hardware profiles from telemetry
5. ActionSequenceGenerator - Generate action sequences for scenarios
"""

import random
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from enum import Enum


# =============================================================================
# 1. POLICY GENERATOR
# =============================================================================

@dataclass
class GeneratedPolicy:
    """Auto-generated optimization policy."""
    policy_id: str
    name: str
    conditions: List[str]
    actions: List[str]
    priority: int
    confidence: float
    source: str  # pattern/learned/template
    metadata: Dict[str, Any] = field(default_factory=dict)


class PolicyGenerator:
    """
    Generate optimization policies from observed patterns.

    Branches from: cognitive_engine + recurrent_logic
    """

    def __init__(self):
        self.observation_buffer: deque = deque(maxlen=1000)
        self.generated_policies: List[GeneratedPolicy] = []
        self.policy_counter = 0

        # Pattern templates
        self.templates = {
            "thermal_response": {
                "condition_pattern": "thermal_headroom < {threshold}",
                "action_pattern": "{action}",
                "actions": ["reduce_power", "throttle", "disable_smt"],
            },
            "performance_boost": {
                "condition_pattern": "cpu_util < {threshold} and thermal_headroom > {margin}",
                "action_pattern": "{action}",
                "actions": ["enable_boost", "increase_power", "enable_smt"],
            },
            "power_save": {
                "condition_pattern": "cpu_util < {threshold} and gpu_util < {threshold}",
                "action_pattern": "{action}",
                "actions": ["enter_powersave", "reduce_clocks", "deep_sleep"],
            },
        }

    def observe(self, telemetry: Dict[str, float], action_taken: str, outcome: float):
        """Observe state-action-outcome triplet."""
        self.observation_buffer.append({
            "timestamp": time.time(),
            "telemetry": telemetry.copy(),
            "action": action_taken,
            "outcome": outcome,
        })

    def generate(self, strategy: str = "pattern") -> Optional[GeneratedPolicy]:
        """Generate new policy based on observations."""
        if strategy == "pattern":
            return self._generate_from_patterns()
        elif strategy == "template":
            return self._generate_from_template()
        elif strategy == "genetic":
            return self._generate_genetic()
        return None

    def _generate_from_patterns(self) -> Optional[GeneratedPolicy]:
        """Find patterns in successful actions."""
        if len(self.observation_buffer) < 50:
            return None

        # Find actions with positive outcomes
        successful = [o for o in self.observation_buffer if o["outcome"] > 0.7]

        if len(successful) < 10:
            return None

        # Cluster by action
        action_stats: Dict[str, List[Dict]] = {}
        for obs in successful:
            action = obs["action"]
            if action not in action_stats:
                action_stats[action] = []
            action_stats[action].append(obs["telemetry"])

        # Find most successful action
        best_action = max(action_stats, key=lambda a: len(action_stats[a]))
        telemetries = action_stats[best_action]

        # Extract common conditions
        conditions = self._extract_conditions(telemetries)

        self.policy_counter += 1
        policy = GeneratedPolicy(
            policy_id=f"gen_pattern_{self.policy_counter}",
            name=f"Pattern-based {best_action}",
            conditions=conditions,
            actions=[best_action],
            priority=5,
            confidence=len(telemetries) / len(successful),
            source="pattern",
            metadata={"sample_size": len(telemetries)},
        )

        self.generated_policies.append(policy)
        return policy

    def _generate_from_template(self) -> Optional[GeneratedPolicy]:
        """Generate from predefined templates."""
        template_name = random.choice(list(self.templates.keys()))
        template = self.templates[template_name]

        # Fill template with learned thresholds
        threshold = self._learn_threshold(template_name)
        margin = random.uniform(10, 20)
        action = random.choice(template["actions"])

        condition = template["condition_pattern"].format(
            threshold=threshold,
            margin=margin,
        )

        self.policy_counter += 1
        policy = GeneratedPolicy(
            policy_id=f"gen_template_{self.policy_counter}",
            name=f"Template {template_name}",
            conditions=[condition],
            actions=[action],
            priority=4,
            confidence=0.6,
            source="template",
            metadata={"template": template_name},
        )

        self.generated_policies.append(policy)
        return policy

    def _generate_genetic(self) -> Optional[GeneratedPolicy]:
        """Genetic algorithm: mutate existing policies."""
        if not self.generated_policies:
            return self._generate_from_template()

        # Select parent
        parent = random.choice(self.generated_policies)

        # Mutate
        new_conditions = []
        for cond in parent.conditions:
            if random.random() < 0.3:
                # Mutate threshold
                cond = self._mutate_condition(cond)
            new_conditions.append(cond)

        new_actions = parent.actions.copy()
        if random.random() < 0.2:
            # Add/remove action
            all_actions = ["throttle", "boost", "reduce_power", "increase_power"]
            new_actions = [random.choice(all_actions)]

        self.policy_counter += 1
        policy = GeneratedPolicy(
            policy_id=f"gen_genetic_{self.policy_counter}",
            name=f"Evolved from {parent.policy_id}",
            conditions=new_conditions,
            actions=new_actions,
            priority=parent.priority,
            confidence=parent.confidence * 0.9,
            source="genetic",
            metadata={"parent": parent.policy_id},
        )

        self.generated_policies.append(policy)
        return policy

    def _extract_conditions(self, telemetries: List[Dict]) -> List[str]:
        """Extract common conditions from telemetry samples."""
        conditions = []

        # Find average values
        keys = ["thermal_headroom", "cpu_util", "gpu_util", "power_draw"]
        for key in keys:
            values = [t.get(key, 0) for t in telemetries if key in t]
            if values:
                avg = sum(values) / len(values)
                std = (sum((v - avg) ** 2 for v in values) / len(values)) ** 0.5

                if std < avg * 0.3:  # Low variance = good condition
                    if key in ("thermal_headroom",):
                        conditions.append(f"{key} < {avg + std:.1f}")
                    else:
                        conditions.append(f"{key} > {avg - std:.1f}")

        return conditions[:3]  # Max 3 conditions

    def _learn_threshold(self, template_name: str) -> float:
        """Learn optimal threshold from observations."""
        if template_name == "thermal_response":
            return random.uniform(5, 15)
        elif template_name == "performance_boost":
            return random.uniform(0.2, 0.4)
        else:
            return random.uniform(0.1, 0.3)

    def _mutate_condition(self, condition: str) -> str:
        """Mutate a condition string."""
        import re
        # Find numbers and mutate
        numbers = re.findall(r"[\d.]+", condition)
        for num in numbers:
            try:
                val = float(num)
                mutated = val * random.uniform(0.8, 1.2)
                condition = condition.replace(num, f"{mutated:.1f}", 1)
            except ValueError:
                pass
        return condition


# =============================================================================
# 2. RULE GENERATOR
# =============================================================================

@dataclass
class GeneratedRule:
    """Auto-generated inference rule."""
    rule_id: str
    condition: str
    action: str
    priority: int
    safety_tier: str
    shadow_mode: bool
    confidence: float
    generation_method: str


class RuleGenerator:
    """
    Auto-generate MicroInferenceRules from telemetry patterns.

    Branches from: cognitive_engine.RuleEngine
    """

    def __init__(self):
        self.rule_counter = 0
        self.generated_rules: List[GeneratedRule] = []
        self.condition_templates = [
            "{var} < {low}",
            "{var} > {high}",
            "{var} < {low} and {var2} > {high2}",
            "{var} > {high} or {var2} < {low2}",
        ]
        self.variables = [
            ("thermal_headroom", 5, 20),
            ("cpu_util", 0.3, 0.8),
            ("gpu_util", 0.3, 0.8),
            ("power_draw", 15, 25),
            ("fps", 30, 60),
            ("latency", 10, 30),
        ]
        self.actions = [
            ("throttle", "thermal", 9),
            ("boost", "performance", 5),
            ("reduce_power", "power", 7),
            ("reduce_quality", "latency", 6),
            ("enable_smt", "performance", 4),
            ("disable_smt", "thermal", 8),
        ]

    def generate_rule(self, method: str = "random") -> GeneratedRule:
        """Generate a new rule."""
        if method == "random":
            return self._generate_random()
        elif method == "threshold":
            return self._generate_threshold_based()
        elif method == "composite":
            return self._generate_composite()
        return self._generate_random()

    def _generate_random(self) -> GeneratedRule:
        """Generate random rule."""
        var, low, high = random.choice(self.variables)
        action, category, base_priority = random.choice(self.actions)

        if random.random() < 0.5:
            condition = f"{var} < {low}"
        else:
            condition = f"{var} > {high}"

        self.rule_counter += 1
        rule = GeneratedRule(
            rule_id=f"gen_rule_{self.rule_counter}",
            condition=condition,
            action=action,
            priority=base_priority + random.randint(-2, 2),
            safety_tier="EXPERIMENTAL",
            shadow_mode=True,  # Always shadow for generated
            confidence=0.5,
            generation_method="random",
        )

        self.generated_rules.append(rule)
        return rule

    def _generate_threshold_based(self) -> GeneratedRule:
        """Generate rule at critical thresholds."""
        # Pick a critical threshold scenario
        scenarios = [
            ("thermal_headroom", "<", 5, "emergency_throttle", 10),
            ("thermal_headroom", "<", 10, "reduce_power", 8),
            ("cpu_util", ">", 0.95, "throttle", 7),
            ("power_draw", ">", 27, "reduce_clocks", 8),
            ("fps", "<", 30, "reduce_quality", 6),
        ]

        var, op, threshold, action, priority = random.choice(scenarios)
        condition = f"{var} {op} {threshold}"

        self.rule_counter += 1
        rule = GeneratedRule(
            rule_id=f"gen_threshold_{self.rule_counter}",
            condition=condition,
            action=action,
            priority=priority,
            safety_tier="STRICT",
            shadow_mode=False,
            confidence=0.8,
            generation_method="threshold",
        )

        self.generated_rules.append(rule)
        return rule

    def _generate_composite(self) -> GeneratedRule:
        """Generate composite rule with multiple conditions."""
        var1, low1, high1 = random.choice(self.variables)
        var2, low2, high2 = random.choice(self.variables)

        while var2 == var1:
            var2, low2, high2 = random.choice(self.variables)

        condition = f"{var1} < {low1} and {var2} > {high2}"
        action, _, priority = random.choice(self.actions)

        self.rule_counter += 1
        rule = GeneratedRule(
            rule_id=f"gen_composite_{self.rule_counter}",
            condition=condition,
            action=action,
            priority=priority,
            safety_tier="EXPERIMENTAL",
            shadow_mode=True,
            confidence=0.4,
            generation_method="composite",
        )

        self.generated_rules.append(rule)
        return rule

    def generate_bundle(self, count: int = 5) -> List[GeneratedRule]:
        """Generate a bundle of rules."""
        methods = ["random", "threshold", "composite"]
        rules = []
        for _ in range(count):
            method = random.choice(methods)
            rules.append(self.generate_rule(method))
        return rules


# =============================================================================
# 3. CONFIG GENERATOR
# =============================================================================

@dataclass
class GeneratedConfig:
    """Auto-generated system configuration."""
    config_id: str
    name: str
    parameters: Dict[str, Any]
    target_scenario: str
    estimated_impact: Dict[str, float]


class ConfigGenerator:
    """
    Generate optimal configurations for different scenarios.

    Branches from: platform_hal + derived_features
    """

    def __init__(self):
        self.config_counter = 0
        self.base_configs = {
            "gaming_performance": {
                "smt": True,
                "governor": "performance",
                "turbo": True,
                "power_limit": 28,
                "thermal_target": 85,
            },
            "gaming_quiet": {
                "smt": True,
                "governor": "powersave",
                "turbo": False,
                "power_limit": 20,
                "thermal_target": 70,
            },
            "productivity": {
                "smt": True,
                "governor": "schedutil",
                "turbo": True,
                "power_limit": 25,
                "thermal_target": 75,
            },
            "battery": {
                "smt": False,
                "governor": "powersave",
                "turbo": False,
                "power_limit": 15,
                "thermal_target": 65,
            },
        }

    def generate(self, scenario: str, constraints: Dict[str, Any] = None) -> GeneratedConfig:
        """Generate config for scenario with constraints."""
        constraints = constraints or {}

        # Start from base
        base = self.base_configs.get(scenario, self.base_configs["productivity"]).copy()

        # Apply constraints
        if "max_thermal" in constraints:
            base["thermal_target"] = min(base["thermal_target"], constraints["max_thermal"])
        if "max_power" in constraints:
            base["power_limit"] = min(base["power_limit"], constraints["max_power"])
        if "force_quiet" in constraints and constraints["force_quiet"]:
            base["turbo"] = False
            base["governor"] = "powersave"

        # Estimate impact
        impact = self._estimate_impact(base, scenario)

        self.config_counter += 1
        return GeneratedConfig(
            config_id=f"gen_config_{self.config_counter}",
            name=f"{scenario}_optimized",
            parameters=base,
            target_scenario=scenario,
            estimated_impact=impact,
        )

    def generate_adaptive(self, telemetry: Dict[str, float]) -> GeneratedConfig:
        """Generate config adapted to current telemetry."""
        temp = telemetry.get("temperature", 70)
        power = telemetry.get("power_draw", 20)
        cpu = telemetry.get("cpu_util", 0.5)

        params = {
            "smt": temp < 80,
            "governor": "performance" if temp < 75 else "schedutil",
            "turbo": temp < 70 and cpu > 0.5,
            "power_limit": min(28, 28 - max(0, temp - 70)),
            "thermal_target": min(85, 90 - (temp - 60) * 0.5),
        }

        self.config_counter += 1
        return GeneratedConfig(
            config_id=f"gen_adaptive_{self.config_counter}",
            name="adaptive_realtime",
            parameters=params,
            target_scenario="adaptive",
            estimated_impact=self._estimate_impact(params, "adaptive"),
        )

    def _estimate_impact(self, params: Dict, scenario: str) -> Dict[str, float]:
        """Estimate performance/thermal/power impact."""
        perf_score = 0.5
        thermal_score = 0.5
        power_score = 0.5

        if params.get("turbo"):
            perf_score += 0.2
            thermal_score -= 0.15
        if params.get("smt"):
            perf_score += 0.15
            thermal_score -= 0.1
        if params.get("governor") == "performance":
            perf_score += 0.1
            power_score -= 0.2

        power_limit = params.get("power_limit", 25)
        power_score += (28 - power_limit) * 0.02

        return {
            "performance": min(1.0, perf_score),
            "thermal": min(1.0, thermal_score),
            "power_efficiency": min(1.0, power_score),
        }


# =============================================================================
# 4. PROFILE GENERATOR
# =============================================================================

@dataclass
class GeneratedProfile:
    """Auto-generated hardware profile."""
    profile_id: str
    name: str
    thermal_limits: Dict[str, float]
    power_limits: Dict[str, float]
    performance_targets: Dict[str, float]
    learned_from: str


class ProfileGenerator:
    """
    Generate hardware profiles from observed telemetry.

    Branches from: platform_hal.HardwareSafetyProfile
    """

    def __init__(self):
        self.profile_counter = 0
        self.telemetry_history: deque = deque(maxlen=5000)
        self.generated_profiles: List[GeneratedProfile] = []

    def observe(self, telemetry: Dict[str, float]):
        """Collect telemetry for profile generation."""
        self.telemetry_history.append({
            "timestamp": time.time(),
            "data": telemetry.copy(),
        })

    def generate_from_observations(self) -> Optional[GeneratedProfile]:
        """Generate profile from collected telemetry."""
        if len(self.telemetry_history) < 100:
            return None

        # Analyze thermal behavior
        temps = [t["data"].get("temperature", 70) for t in self.telemetry_history]
        max_temp = max(temps)
        avg_temp = sum(temps) / len(temps)
        temp_std = (sum((t - avg_temp) ** 2 for t in temps) / len(temps)) ** 0.5

        # Analyze power
        powers = [t["data"].get("power_draw", 20) for t in self.telemetry_history]
        max_power = max(powers)
        avg_power = sum(powers) / len(powers)

        # Generate conservative profile
        self.profile_counter += 1
        profile = GeneratedProfile(
            profile_id=f"gen_profile_{self.profile_counter}",
            name=f"Learned Profile #{self.profile_counter}",
            thermal_limits={
                "critical": min(100, max_temp + 10),
                "throttle": min(90, max_temp + 5),
                "warning": avg_temp + temp_std,
                "target": avg_temp,
            },
            power_limits={
                "max": min(28, max_power * 1.1),
                "sustained": avg_power,
                "burst": max_power,
            },
            performance_targets={
                "fps_min": 30,
                "fps_target": 60,
                "latency_max": 20,
            },
            learned_from=f"{len(self.telemetry_history)} observations",
        )

        self.generated_profiles.append(profile)
        return profile

    def generate_for_workload(self, workload: str) -> GeneratedProfile:
        """Generate profile for specific workload type."""
        presets = {
            "gaming": {
                "thermal": {"critical": 95, "throttle": 85, "warning": 75, "target": 70},
                "power": {"max": 28, "sustained": 25, "burst": 28},
                "perf": {"fps_min": 30, "fps_target": 60, "latency_max": 16},
            },
            "rendering": {
                "thermal": {"critical": 95, "throttle": 90, "warning": 80, "target": 75},
                "power": {"max": 28, "sustained": 28, "burst": 28},
                "perf": {"fps_min": 1, "fps_target": 10, "latency_max": 1000},
            },
            "office": {
                "thermal": {"critical": 90, "throttle": 75, "warning": 65, "target": 55},
                "power": {"max": 20, "sustained": 15, "burst": 25},
                "perf": {"fps_min": 30, "fps_target": 60, "latency_max": 50},
            },
        }

        preset = presets.get(workload, presets["office"])

        self.profile_counter += 1
        return GeneratedProfile(
            profile_id=f"gen_workload_{self.profile_counter}",
            name=f"{workload.title()} Profile",
            thermal_limits=preset["thermal"],
            power_limits=preset["power"],
            performance_targets=preset["perf"],
            learned_from=f"workload:{workload}",
        )


# =============================================================================
# 5. ACTION SEQUENCE GENERATOR
# =============================================================================

@dataclass
class ActionSequence:
    """Generated sequence of actions."""
    sequence_id: str
    name: str
    actions: List[Dict[str, Any]]
    trigger_condition: str
    estimated_duration_ms: float
    rollback_actions: List[str]


class ActionSequenceGenerator:
    """
    Generate action sequences for complex scenarios.

    Branches from: recurrent_logic + derived_features
    """

    def __init__(self):
        self.sequence_counter = 0
        self.action_library = {
            "thermal_emergency": [
                {"action": "disable_turbo", "delay_ms": 0},
                {"action": "reduce_power_50", "delay_ms": 100},
                {"action": "disable_smt", "delay_ms": 200},
                {"action": "max_fans", "delay_ms": 0},
            ],
            "performance_boost": [
                {"action": "enable_turbo", "delay_ms": 0},
                {"action": "enable_smt", "delay_ms": 50},
                {"action": "set_performance_governor", "delay_ms": 100},
            ],
            "power_save_transition": [
                {"action": "reduce_clocks", "delay_ms": 0},
                {"action": "disable_turbo", "delay_ms": 100},
                {"action": "set_powersave_governor", "delay_ms": 200},
                {"action": "reduce_refresh_rate", "delay_ms": 300},
            ],
            "game_start": [
                {"action": "load_game_profile", "delay_ms": 0},
                {"action": "set_cpu_affinity", "delay_ms": 50},
                {"action": "enable_turbo", "delay_ms": 100},
                {"action": "prioritize_gpu", "delay_ms": 150},
            ],
        }

    def generate(self, scenario: str) -> ActionSequence:
        """Generate action sequence for scenario."""
        if scenario in self.action_library:
            actions = self.action_library[scenario]
        else:
            actions = self._generate_custom(scenario)

        rollback = self._generate_rollback(actions)
        duration = sum(a.get("delay_ms", 0) for a in actions) + 100

        self.sequence_counter += 1
        return ActionSequence(
            sequence_id=f"seq_{self.sequence_counter}",
            name=f"{scenario}_sequence",
            actions=actions,
            trigger_condition=self._infer_trigger(scenario),
            estimated_duration_ms=duration,
            rollback_actions=rollback,
        )

    def generate_adaptive(self, current_state: Dict[str, float], target_state: Dict[str, float]) -> ActionSequence:
        """Generate sequence to transition from current to target state."""
        actions = []
        delay = 0

        # Temperature transition
        current_temp = current_state.get("temperature", 70)
        target_temp = target_state.get("temperature", 70)

        if target_temp < current_temp - 5:
            actions.append({"action": "reduce_power", "delay_ms": delay})
            delay += 100
            if target_temp < current_temp - 10:
                actions.append({"action": "disable_turbo", "delay_ms": delay})
                delay += 100

        elif target_temp > current_temp + 5:
            actions.append({"action": "enable_turbo", "delay_ms": delay})
            delay += 100

        # Performance transition
        current_perf = current_state.get("cpu_util", 0.5)
        target_perf = target_state.get("cpu_util", 0.5)

        if target_perf > current_perf + 0.2:
            actions.append({"action": "boost_priority", "delay_ms": delay})
            delay += 50

        self.sequence_counter += 1
        return ActionSequence(
            sequence_id=f"seq_adaptive_{self.sequence_counter}",
            name="adaptive_transition",
            actions=actions if actions else [{"action": "maintain", "delay_ms": 0}],
            trigger_condition="state_transition",
            estimated_duration_ms=delay + 50,
            rollback_actions=["restore_previous_state"],
        )

    def _generate_custom(self, scenario: str) -> List[Dict]:
        """Generate custom sequence for unknown scenario."""
        return [
            {"action": "analyze_scenario", "delay_ms": 0},
            {"action": "apply_safe_defaults", "delay_ms": 100},
            {"action": "monitor", "delay_ms": 200},
        ]

    def _generate_rollback(self, actions: List[Dict]) -> List[str]:
        """Generate rollback actions."""
        rollback_map = {
            "disable_turbo": "enable_turbo",
            "enable_turbo": "disable_turbo",
            "disable_smt": "enable_smt",
            "enable_smt": "disable_smt",
            "reduce_power_50": "restore_power",
            "set_powersave_governor": "restore_governor",
            "set_performance_governor": "restore_governor",
        }

        rollback = []
        for action in reversed(actions):
            act = action.get("action", "")
            if act in rollback_map:
                rollback.append(rollback_map[act])

        return rollback

    def _infer_trigger(self, scenario: str) -> str:
        """Infer trigger condition for scenario."""
        triggers = {
            "thermal_emergency": "thermal_headroom < 3",
            "performance_boost": "cpu_util > 0.8 and thermal_headroom > 15",
            "power_save_transition": "cpu_util < 0.2 and gpu_util < 0.2",
            "game_start": "game_detected == True",
        }
        return triggers.get(scenario, "manual_trigger")


# =============================================================================
# UNIFIED GENERATOR SYSTEM
# =============================================================================

class GeneratorSystem:
    """Unified system combining all generators."""

    def __init__(self):
        self.policy_gen = PolicyGenerator()
        self.rule_gen = RuleGenerator()
        self.config_gen = ConfigGenerator()
        self.profile_gen = ProfileGenerator()
        self.sequence_gen = ActionSequenceGenerator()

    def observe(self, telemetry: Dict[str, float], action: str = "", outcome: float = 0.5):
        """Feed telemetry to all generators."""
        self.policy_gen.observe(telemetry, action, outcome)
        self.profile_gen.observe(telemetry)

    def generate_all(self, scenario: str = "gaming") -> Dict[str, Any]:
        """Generate outputs from all generators."""
        return {
            "policy": self.policy_gen.generate("template"),
            "rules": self.rule_gen.generate_bundle(3),
            "config": self.config_gen.generate(scenario),
            "profile": self.profile_gen.generate_for_workload(scenario),
            "sequence": self.sequence_gen.generate(scenario if scenario in self.sequence_gen.action_library else "game_start"),
        }


def create_generator_system() -> GeneratorSystem:
    """Factory function."""
    return GeneratorSystem()


if __name__ == "__main__":
    system = GeneratorSystem()

    print("=== GAMESA Generators ===\n")

    # Generate for gaming scenario
    output = system.generate_all("gaming")

    print("Generated Policy:")
    if output["policy"]:
        print(f"  ID: {output['policy'].policy_id}")
        print(f"  Conditions: {output['policy'].conditions}")
        print(f"  Actions: {output['policy'].actions}")

    print("\nGenerated Rules:")
    for rule in output["rules"]:
        print(f"  {rule.rule_id}: {rule.condition} -> {rule.action}")

    print("\nGenerated Config:")
    print(f"  {output['config'].name}: {output['config'].parameters}")

    print("\nGenerated Profile:")
    print(f"  {output['profile'].name}")
    print(f"  Thermal: {output['profile'].thermal_limits}")

    print("\nGenerated Sequence:")
    print(f"  {output['sequence'].name}: {len(output['sequence'].actions)} actions")
    for a in output['sequence'].actions:
        print(f"    +{a['delay_ms']}ms: {a['action']}")
