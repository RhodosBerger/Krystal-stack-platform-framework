"""
Tests for KrystalSDK

Run with: pytest tests/test_krystal_sdk.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


class TestMicroLearner:
    """Tests for MicroLearner TD component."""

    def test_init(self):
        from src.python.krystal_sdk import MicroLearner
        learner = MicroLearner(dim=4)
        assert len(learner.weights) == 4

    def test_predict(self):
        from src.python.krystal_sdk import MicroLearner
        learner = MicroLearner(dim=4)
        pred = learner.predict([0.5, 0.5, 0.5, 0.5])
        assert isinstance(pred, float)

    def test_update(self):
        from src.python.krystal_sdk import MicroLearner
        learner = MicroLearner(dim=4, lr=0.1)
        state = [0.5] * 4
        next_state = [0.6] * 4
        error = learner.update(state, 1.0, next_state)
        assert isinstance(error, float)


class TestMicroPhase:
    """Tests for phase transition engine."""

    def test_init(self):
        from src.python.krystal_sdk import MicroPhase, Phase
        phase = MicroPhase()
        assert phase.phase == Phase.SOLID

    def test_update(self):
        from src.python.krystal_sdk import MicroPhase, Phase
        phase = MicroPhase()
        result = phase.update(0.1, 0.9)
        assert isinstance(result, Phase)

    def test_exploration_rate(self):
        from src.python.krystal_sdk import MicroPhase
        phase = MicroPhase()
        rate = phase.exploration_rate()
        assert 0 <= rate <= 1


class TestMicroSwarm:
    """Tests for particle swarm optimizer."""

    def test_init(self):
        from src.python.krystal_sdk import MicroSwarm
        swarm = MicroSwarm(n_particles=5, dim=3)
        assert len(swarm.particles) == 5
        assert len(swarm.particles[0]) == 3

    def test_step(self):
        from src.python.krystal_sdk import MicroSwarm
        swarm = MicroSwarm(n_particles=5, dim=3)

        def objective(x):
            return -sum(xi**2 for xi in x)

        best = swarm.step(objective)
        assert len(best) == 3


class TestMicroController:
    """Tests for PID controller."""

    def test_init(self):
        from src.python.krystal_sdk import MicroController
        ctrl = MicroController(kp=1.0, ki=0.1, kd=0.05)
        assert ctrl.kp == 1.0

    def test_update(self):
        from src.python.krystal_sdk import MicroController
        ctrl = MicroController()
        output = ctrl.update(1.0, dt=0.016)
        assert isinstance(output, float)


class TestKrystal:
    """Tests for main Krystal class."""

    def test_init(self):
        from src.python.krystal_sdk import Krystal
        k = Krystal()
        assert k.cycle == 0

    def test_observe(self):
        from src.python.krystal_sdk import Krystal
        k = Krystal()
        result = k.observe({"cpu": 0.5, "mem": 0.3})
        assert result is k  # Returns self for chaining

    def test_decide(self):
        from src.python.krystal_sdk import Krystal
        k = Krystal()
        k.observe({"cpu": 0.5})
        action = k.decide()
        assert isinstance(action, list)
        assert all(0 <= a <= 1 for a in action)

    def test_reward(self):
        from src.python.krystal_sdk import Krystal
        k = Krystal()
        k.observe({"x": 0.5})
        k.decide()
        k.observe({"x": 0.6})
        td_error = k.reward(1.0)
        assert isinstance(td_error, float)

    def test_optimize(self):
        from src.python.krystal_sdk import Krystal
        k = Krystal()

        def objective(x):
            return -sum(xi**2 for xi in x)

        best, score = k.optimize(objective, iterations=10)
        assert isinstance(best, list)
        assert isinstance(score, float)

    def test_control(self):
        from src.python.krystal_sdk import Krystal
        k = Krystal()
        output = k.control(setpoint=1.0, current=0.5)
        assert isinstance(output, float)

    def test_get_phase(self):
        from src.python.krystal_sdk import Krystal
        k = Krystal()
        phase = k.get_phase()
        assert phase in ["SOLID", "LIQUID", "GAS", "PLASMA"]

    def test_get_metrics(self):
        from src.python.krystal_sdk import Krystal
        k = Krystal()
        k.observe({"x": 0.5})
        k.decide()
        k.reward(0.5)
        metrics = k.get_metrics()
        assert "cycles" in metrics
        assert "total_reward" in metrics

    def test_save_load(self, tmp_path):
        from src.python.krystal_sdk import Krystal
        k1 = Krystal()
        for _ in range(10):
            k1.observe({"x": 0.5})
            k1.decide()
            k1.reward(0.5)

        path = tmp_path / "krystal.json"
        k1.save(str(path))

        k2 = Krystal()
        k2.load(str(path))
        assert k2.learner.weights == k1.learner.weights


class TestFactories:
    """Tests for factory functions."""

    def test_create_game_optimizer(self):
        from src.python.krystal_sdk import create_game_optimizer
        k = create_game_optimizer()
        assert k.config.state_dim == 8
        assert k.config.action_dim == 4

    def test_create_server_optimizer(self):
        from src.python.krystal_sdk import create_server_optimizer
        k = create_server_optimizer()
        assert k.config.state_dim == 6
        assert k.config.action_dim == 3

    def test_create_ml_optimizer(self):
        from src.python.krystal_sdk import create_ml_optimizer
        k = create_ml_optimizer()
        assert k.config.state_dim == 4

    def test_create_iot_optimizer(self):
        from src.python.krystal_sdk import create_iot_optimizer
        k = create_iot_optimizer()
        assert k.config.action_dim == 2


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check(self):
        from src.python.krystal_sdk import health_check
        result = health_check()
        assert result["status"] in ["healthy", "degraded"]
        assert "components" in result
        assert "learner" in result["components"]


class TestIntegration:
    """Integration tests."""

    def test_full_cycle(self):
        from src.python.krystal_sdk import Krystal
        k = Krystal()

        for _ in range(100):
            k.observe({"metric": 0.5})
            action = k.decide()
            k.reward(sum(action) / len(action))

        assert k.cycle == 100
        assert k.total_reward != 0

    def test_chained_api(self):
        from src.python.krystal_sdk import Krystal
        k = Krystal()

        # Chained observe
        result = k.observe({"a": 0.1}).observe({"b": 0.2})
        assert result is k


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
