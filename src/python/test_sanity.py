"""
GAMESA/KrystalStack Test Suite - Sanity Tests

Smoke tests for:
1. CognitiveOrchestrator pipeline
2. Safety guardrails validation
3. Rule engine with shadow-mode
4. Economic validation
5. Invention engine components
6. Emergent system behaviors
"""

import time
import unittest
from typing import Dict, Any


class TestSafetyGuardrails(unittest.TestCase):
    """Test safety guardrails enforcement."""

    def setUp(self):
        from cognitive_engine import SafetyGuardrails, SafetyConstraint
        self.guardrails = SafetyGuardrails()

    def test_thermal_critical_violation(self):
        """Thermal headroom <= 0 triggers emergency throttle."""
        context = {"thermal_headroom": -1, "power_draw": 100, "power_limit": 150,
                   "memory_util": 0.5, "latency_ms": 10, "latency_target": 16}
        violations = self.guardrails.validate(context)
        self.assertTrue(any(v.name == "thermal_critical" for v in violations))

    def test_thermal_safe(self):
        """Thermal headroom > 0 passes."""
        context = {"thermal_headroom": 10, "power_draw": 100, "power_limit": 150,
                   "memory_util": 0.5, "latency_ms": 10, "latency_target": 16}
        violations = self.guardrails.validate(context)
        thermal_violations = [v for v in violations if v.name == "thermal_critical"]
        self.assertEqual(len(thermal_violations), 0)

    def test_power_limit_violation(self):
        """Power draw > 110% limit triggers reduce_power."""
        context = {"thermal_headroom": 10, "power_draw": 180, "power_limit": 150,
                   "memory_util": 0.5, "latency_ms": 10, "latency_target": 16}
        violations = self.guardrails.validate(context)
        self.assertTrue(any(v.name == "power_limit" for v in violations))

    def test_memory_pressure_warning(self):
        """Memory util >= 95% triggers gc_trigger."""
        context = {"thermal_headroom": 10, "power_draw": 100, "power_limit": 150,
                   "memory_util": 0.98, "latency_ms": 10, "latency_target": 16}
        violations = self.guardrails.validate(context)
        self.assertTrue(any(v.name == "memory_pressure" for v in violations))

    def test_violation_rate_tracking(self):
        """Violation rate computed correctly."""
        context = {"thermal_headroom": -1, "power_draw": 100, "power_limit": 150,
                   "memory_util": 0.5, "latency_ms": 10, "latency_target": 16}
        for _ in range(5):
            self.guardrails.validate(context)
        rate = self.guardrails.get_violation_rate(window_seconds=60)
        self.assertGreater(rate, 0)


class TestRuleEngine(unittest.TestCase):
    """Test rule engine with shadow-mode."""

    def setUp(self):
        from cognitive_engine import RuleEngine, MicroInferenceRule, SafetyTier
        self.engine = RuleEngine()
        self.SafetyTier = SafetyTier
        self.MicroInferenceRule = MicroInferenceRule

    def test_rule_triggers(self):
        """Rule triggers when condition met."""
        rule = self.MicroInferenceRule(
            rule_id="test_throttle",
            condition="thermal_headroom < 5",
            action="throttle",
            priority=10
        )
        self.engine.add_rule(rule)
        actions = self.engine.execute({"thermal_headroom": 3})
        self.assertIn("throttle", actions)

    def test_rule_not_triggers(self):
        """Rule does not trigger when condition not met."""
        rule = self.MicroInferenceRule(
            rule_id="test_throttle",
            condition="thermal_headroom < 5",
            action="throttle",
            priority=10
        )
        self.engine.add_rule(rule)
        actions = self.engine.execute({"thermal_headroom": 10})
        self.assertNotIn("throttle", actions)

    def test_shadow_mode_logs_only(self):
        """Shadow mode logs but does not execute."""
        rule = self.MicroInferenceRule(
            rule_id="shadow_test",
            condition="cpu_util > 0.8",
            action="reduce_quality",
            priority=5,
            shadow_mode=True
        )
        self.engine.add_rule(rule)
        actions = self.engine.execute({"cpu_util": 0.9})
        self.assertNotIn("reduce_quality", actions)
        self.assertEqual(len(self.engine.execution_log), 1)
        self.assertTrue(self.engine.execution_log[0]["shadow"])

    def test_priority_ordering(self):
        """Higher priority rules execute first."""
        rule_low = self.MicroInferenceRule(
            rule_id="low", condition="True", action="low_action", priority=1)
        rule_high = self.MicroInferenceRule(
            rule_id="high", condition="True", action="high_action", priority=10)
        self.engine.add_rule(rule_low)
        self.engine.add_rule(rule_high)
        actions = self.engine.execute({})
        self.assertEqual(actions[0], "high_action")


class TestEconomicEngine(unittest.TestCase):
    """Test economic validation and budget tracking."""

    def setUp(self):
        from cognitive_engine import EconomicEngine, ActionEconomicProfile, ResourceBudgets
        self.engine = EconomicEngine()
        self.ActionEconomicProfile = ActionEconomicProfile

    def test_can_afford_within_budget(self):
        """Action within budget is affordable."""
        profile = self.ActionEconomicProfile(
            action_id="boost",
            costs={"cpu_mw": 1000, "latency_ms": 1},
            payoffs={"performance": 10},
            risks={"thermal": 0.1}
        )
        self.assertTrue(self.engine.can_afford(profile))

    def test_cannot_afford_over_budget(self):
        """Action over budget is not affordable."""
        profile = self.ActionEconomicProfile(
            action_id="mega_boost",
            costs={"cpu_mw": 100000},  # Over 45W budget
            payoffs={},
            risks={}
        )
        self.assertFalse(self.engine.can_afford(profile))

    def test_trade_deducts_budget(self):
        """Successful trade deducts from budget."""
        initial_cpu = self.engine.budgets.cpu_budget_mw
        profile = self.ActionEconomicProfile(
            action_id="boost",
            costs={"cpu_mw": 5000},
            payoffs={},
            risks={}
        )
        self.engine.execute_trade(profile)
        self.assertEqual(self.engine.budgets.cpu_budget_mw, initial_cpu - 5000)

    def test_replenish_adds_budget(self):
        """Replenish adds to budget."""
        self.engine.budgets.cpu_budget_mw = 10000
        self.engine.replenish({"cpu_mw": 5000})
        self.assertEqual(self.engine.budgets.cpu_budget_mw, 15000)


class TestCognitiveOrchestrator(unittest.TestCase):
    """Test full orchestrator pipeline."""

    def setUp(self):
        from cognitive_engine import create_cognitive_orchestrator
        self.orchestrator = create_cognitive_orchestrator()

    def test_normal_processing(self):
        """Normal telemetry produces valid action."""
        telemetry = {
            "frametime_ms": 16.6,
            "thermal_headroom": 10,
            "cpu_util": 0.5,
            "gpu_util": 0.6,
            "power_draw": 100,
            "power_limit": 150,
            "memory_util": 0.5,
            "latency_ms": 10,
            "latency_target": 16
        }
        result = self.orchestrator.process(telemetry)
        self.assertIn("action", result)
        self.assertIn(result["action"], ["boost", "throttle", "migrate", "idle", "noop"])

    def test_safety_violation_override(self):
        """Critical safety violation overrides normal action."""
        telemetry = {
            "frametime_ms": 16.6,
            "thermal_headroom": -5,  # Critical violation
            "cpu_util": 0.5,
            "gpu_util": 0.6,
            "power_draw": 100,
            "power_limit": 150,
            "memory_util": 0.5,
            "latency_ms": 10,
            "latency_target": 16
        }
        result = self.orchestrator.process(telemetry)
        self.assertEqual(result["action"], "emergency_throttle")
        self.assertEqual(result["reason"], "safety_violation")

    def test_economic_tracking(self):
        """Economic budgets are tracked in result."""
        telemetry = {
            "frametime_ms": 16.6,
            "thermal_headroom": 10,
            "cpu_util": 0.5,
            "gpu_util": 0.6,
            "power_draw": 100,
            "power_limit": 150,
            "memory_util": 0.5,
            "latency_ms": 10,
            "latency_target": 16
        }
        result = self.orchestrator.process(telemetry)
        self.assertIn("economic", result)
        self.assertIn("budgets", result["economic"])

    def test_metacog_insights(self):
        """Metacognitive insights available."""
        telemetry = {
            "frametime_ms": 16.6,
            "thermal_headroom": 10,
            "cpu_util": 0.5,
            "gpu_util": 0.6,
            "power_draw": 100,
            "power_limit": 150,
            "memory_util": 0.5,
            "latency_ms": 10,
            "latency_target": 16
        }
        # Process multiple times to build experience
        for _ in range(15):
            result = self.orchestrator.process(telemetry)
        self.assertIn("metacog", result)


class TestInventionEngine(unittest.TestCase):
    """Smoke tests for invention engine components."""

    def test_quantum_scheduler(self):
        """Quantum scheduler creates and measures superposition."""
        from invention_engine import SuperpositionScheduler
        scheduler = SuperpositionScheduler()
        task = scheduler.create_superposition("task1", ["boost", "throttle", "idle"])
        self.assertEqual(len(task.amplitudes), 3)
        result = scheduler.measure("task1")
        self.assertIn(result, ["boost", "throttle", "idle"])

    def test_causal_inference(self):
        """Causal inference observes and discovers."""
        from invention_engine import CausalInferenceEngine
        engine = CausalInferenceEngine()
        # Generate correlated data
        for i in range(100):
            engine.observe({"temp": 50 + i * 0.5, "power": 100 + i * 0.3})
        strength, lag = engine.granger_causality("temp", "power")
        self.assertIsInstance(strength, float)

    def test_hyperdimensional_encoder(self):
        """HD encoder stores and queries states."""
        from invention_engine import HyperdimensionalEncoder
        hd = HyperdimensionalEncoder(dimensions=1000)
        state = {"cpu": 0.5, "gpu": 0.7, "temp": 0.3}
        hd.store("state1", state)
        similar = hd.query(state, top_k=1)
        self.assertEqual(similar[0][0], "state1")

    def test_invention_engine_process(self):
        """Full invention engine processes telemetry."""
        from invention_engine import create_invention_engine
        engine = create_invention_engine()
        result = engine.process({"cpu_util": 0.5, "gpu_util": 0.6, "npu_util": 0.3})
        self.assertIn("action", result)
        self.assertIn("allocation", result)


class TestEmergentSystem(unittest.TestCase):
    """Smoke tests for emergent system."""

    def test_cellular_automata(self):
        """CA steps and produces metrics."""
        from emergent_system import CellularAutomata
        ca = CellularAutomata(width=20, height=20)
        ca.seed_from_state({"cpu": 0.5, "gpu": 0.7})
        for _ in range(5):
            ca.step()
        metrics = ca.get_metrics()
        self.assertIn("density", metrics)
        self.assertIn("clusters", metrics)

    def test_self_organizing_map(self):
        """SOM trains and clusters."""
        from emergent_system import SelfOrganizingMap
        som = SelfOrganizingMap(map_size=5)
        for _ in range(10):
            som.train({"dim_0": 0.5, "dim_1": 0.3})
        cluster = som.get_cluster_id({"dim_0": 0.5, "dim_1": 0.3})
        self.assertIsInstance(cluster, int)

    def test_criticality_detector(self):
        """Criticality detector computes metrics."""
        from emergent_system import CriticalityDetector
        detector = CriticalityDetector()
        for i in range(200):
            detector.observe(0.5 + 0.3 * (i % 10) / 10)
        is_critical, metrics = detector.is_critical()
        self.assertIn("branching_ratio", metrics)

    def test_emergent_system_process(self):
        """Full emergent system processes telemetry."""
        from emergent_system import create_emergent_system
        system = create_emergent_system()
        result = system.process({"cpu_util": 0.5, "gpu_util": 0.6})
        self.assertIn("cellular_automata", result)
        self.assertIn("cluster_id", result)


# =============================================================================
# Default Rule Bundle
# =============================================================================

def get_default_rules():
    """Get default safe MicroInferenceRule bundle."""
    from cognitive_engine import MicroInferenceRule, SafetyTier

    return [
        # Thermal management
        MicroInferenceRule(
            rule_id="thermal_warning",
            condition="thermal_headroom < 8",
            action="reduce_power",
            priority=8,
            safety_tier=SafetyTier.STRICT,
            description="Reduce power when thermal headroom low"
        ),
        MicroInferenceRule(
            rule_id="thermal_critical",
            condition="thermal_headroom < 3",
            action="emergency_throttle",
            priority=10,
            safety_tier=SafetyTier.STRICT,
            description="Emergency throttle on critical thermal"
        ),

        # Latency management
        MicroInferenceRule(
            rule_id="latency_cutback",
            condition="latency_ms > latency_target * 1.5",
            action="reduce_quality",
            priority=7,
            safety_tier=SafetyTier.STRICT,
            description="Reduce quality when latency exceeds target"
        ),

        # Memory management
        MicroInferenceRule(
            rule_id="memory_demotion",
            condition="memory_util > 0.85",
            action="demote_cold_pages",
            priority=5,
            safety_tier=SafetyTier.STRICT,
            description="Demote cold pages on memory pressure"
        ),

        # Power management
        MicroInferenceRule(
            rule_id="power_limit",
            condition="power_draw > tdp_max * 1.1",
            action="reduce_clocks",
            priority=9,
            safety_tier=SafetyTier.STRICT,
            description="Reduce clocks when exceeding TDP"
        ),
        MicroInferenceRule(
            rule_id="idle_powersave",
            condition="cpu_util < 0.15 and gpu_util < 0.1",
            action="enter_powersave",
            priority=2,
            safety_tier=SafetyTier.STRICT,
            description="Enter powersave on sustained idle"
        ),

        # Gaming performance
        MicroInferenceRule(
            rule_id="combat_boost_affinity",
            condition="game_state == 'combat' and thermal_headroom > 10",
            action="boost_priority",
            priority=6,
            safety_tier=SafetyTier.STRICT,
            description="Boost priority during combat if thermal OK"
        ),
        MicroInferenceRule(
            rule_id="fps_recovery",
            condition="fps < fps_target * 0.9 and thermal_headroom > 5",
            action="increase_gpu_power",
            priority=5,
            safety_tier=SafetyTier.STRICT,
            description="Increase GPU power to recover FPS"
        ),

        # Stability
        MicroInferenceRule(
            rule_id="anomaly_response",
            condition="anomaly_count > 3",
            action="conservative_mode",
            priority=7,
            safety_tier=SafetyTier.STRICT,
            description="Enter conservative mode on anomalies"
        ),

        # Shadow-mode experiments
        MicroInferenceRule(
            rule_id="experimental_boost",
            condition="cpu_util < 0.3 and gpu_util < 0.3 and thermal_headroom > 15",
            action="opportunistic_boost",
            priority=3,
            safety_tier=SafetyTier.EXPERIMENTAL,
            shadow_mode=True,
            description="[SHADOW] Test opportunistic boost in idle"
        ),
        MicroInferenceRule(
            rule_id="experimental_prefetch",
            condition="memory_util < 0.5 and cache_miss_rate > 0.1",
            action="aggressive_prefetch",
            priority=2,
            safety_tier=SafetyTier.EXPERIMENTAL,
            shadow_mode=True,
            description="[SHADOW] Test aggressive prefetching"
        ),
    ]


def install_default_rules(orchestrator):
    """Install default rules into orchestrator."""
    for rule in get_default_rules():
        orchestrator.rules.add_rule(rule)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
