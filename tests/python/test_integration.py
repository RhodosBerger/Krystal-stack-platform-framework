"""
Integration Tests - Full System Verification

Tests all layers working together.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestUnifiedSystem:
    """Test unified system integration."""

    def test_create_system(self):
        """Test system creation."""
        from src.python.unified_system import create_unified_system

        system = create_unified_system()
        assert system is not None
        assert system.mode.name == "INIT"

    def test_system_start_stop(self):
        """Test system lifecycle."""
        from src.python.unified_system import create_unified_system

        system = create_unified_system()
        system.start()
        assert system._running

        system.stop()
        assert not system._running

    def test_single_tick(self):
        """Test single tick execution."""
        from src.python.unified_system import create_unified_system

        system = create_unified_system()
        system.start()

        result = system.tick()

        assert "cycle" in result
        assert "telemetry" in result
        assert "signals" in result
        assert "decision" in result
        assert "predictions" in result
        assert "emergence" in result
        assert "reward" in result
        assert "mode" in result

        system.stop()

    def test_multiple_cycles(self):
        """Test multiple cycle execution."""
        from src.python.unified_system import create_unified_system

        system = create_unified_system()
        system.start()

        results = system.run_cycles(50)

        assert len(results) == 50
        assert results[-1]["cycle"] == 50

        # Verify learning occurred
        rewards = [r["reward"] for r in results]
        assert len(set(rewards)) > 1  # Not all same

        system.stop()

    def test_state_summary(self):
        """Test state summary generation."""
        from src.python.unified_system import create_unified_system

        system = create_unified_system()
        system.start()
        system.run_cycles(20)

        summary = system.get_state_summary()

        assert "mode" in summary
        assert "cycle_count" in summary
        assert "metrics" in summary
        assert "consciousness" in summary
        assert "learning" in summary
        assert "emergence" in summary

        system.stop()


class TestLevelIntegration:
    """Test individual level integration."""

    def test_level0_hardware(self):
        """Test hardware level."""
        from src.python.unified_system import HardwareLevel

        hw = HardwareLevel()
        telemetry = hw.read_telemetry()

        assert "cpu_util" in telemetry
        assert "gpu_util" in telemetry
        assert "gpu_temp" in telemetry
        assert 0 <= telemetry["cpu_util"] <= 1
        assert 0 <= telemetry["gpu_util"] <= 1

    def test_level1_signals(self):
        """Test signal processing level."""
        from src.python.unified_system import HardwareLevel, SignalLevel

        hw = HardwareLevel()
        signal = SignalLevel(hw)

        telemetry = hw.read_telemetry()
        signals = signal.process_signals(telemetry)

        assert "thermal_control" in signals
        assert "performance_control" in signals
        assert "thermal_headroom" in signals

    def test_level2_learning(self):
        """Test learning level."""
        from src.python.unified_system import LearningLevel
        import numpy as np

        learning = LearningLevel()

        state = np.random.random(8)
        action, confidence = learning.decide(state)

        assert len(action) == 4
        assert 0 <= confidence <= 1

    def test_level3_prediction(self):
        """Test prediction level."""
        from src.python.unified_system import PredictionLevel

        prediction = PredictionLevel()

        telemetry = {"gpu_util": 0.5, "gpu_temp": 60, "gpu_power": 150}
        predictions = prediction.predict(telemetry)

        assert "future_states" in predictions
        assert "neural_settings" in predictions
        assert "pre_execution" in predictions

    def test_level4_emergence(self):
        """Test emergence level."""
        from src.python.unified_system import EmergenceLevel

        emergence = EmergenceLevel()

        telemetry = {
            "cpu_util": 0.5, "gpu_util": 0.5,
            "gpu_temp": 60, "fps_variance": 2
        }

        def objective(s):
            return -np.sum((s - 0.7) ** 2)

        result = emergence.evolve(telemetry, objective)

        assert "evolution" in result
        assert "consciousness" in result
        assert "enhancements" in result
        assert "phase" in result

    def test_level5_generation(self):
        """Test generation level."""
        from src.python.unified_system import GenerationLevel

        gen = GenerationLevel()
        preset = gen.generate_preset()

        assert "name" in preset
        assert "cpu_freq" in preset
        assert "gpu_clock" in preset
        assert "power_limit" in preset
        assert "quality_score" in preset


class TestEmergentBehavior:
    """Test emergent behavior properties."""

    def test_attractor_convergence(self):
        """Test that system converges to attractors."""
        from src.python.emergent_intelligence import AttractorLandscape
        import numpy as np

        landscape = AttractorLandscape()

        # Start far from attractors
        state = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        # Evolve
        for _ in range(100):
            state = landscape.update(state)

        # Should have moved toward an attractor
        nearest = landscape.get_nearest_attractor()
        distance = np.linalg.norm(state - nearest.center)
        assert distance < nearest.basin_radius * 2

    def test_phase_transitions(self):
        """Test phase transition behavior."""
        from src.python.emergent_intelligence import PhaseTransitionEngine, PhaseState

        engine = PhaseTransitionEngine()

        # Low gradient = solid
        for _ in range(10):
            engine.update(0.01, 0.9)
        assert engine.current_phase == PhaseState.SOLID

        # High gradient = liquid or higher
        for _ in range(20):
            engine.update(0.5, 0.5)
        assert engine.current_phase != PhaseState.SOLID

    def test_collective_convergence(self):
        """Test swarm convergence."""
        from src.python.emergent_intelligence import CollectiveIntelligence
        import numpy as np

        swarm = CollectiveIntelligence(n_agents=10)

        def objective(state):
            # Maximum at [0.7, 0.7, ...]
            return -np.sum((state - 0.7) ** 2)

        # Evolve swarm
        for _ in range(50):
            best = swarm.update(objective)

        # Should have found near-optimal
        assert objective(best) > -0.5


class TestBreakthroughEngine:
    """Test breakthrough engine components."""

    def test_temporal_prediction(self):
        """Test temporal predictor."""
        from src.python.breakthrough_engine import TemporalPredictor

        predictor = TemporalPredictor()

        # Add observations
        for i in range(20):
            predictor.observe({
                "cpu_util": 0.5 + 0.1 * np.sin(i / 5),
                "gpu_util": 0.6,
                "thermal": 60,
                "fps": 60
            })

        predictions = predictor.predict(steps_ahead=3)
        assert len(predictions) == 3

    def test_neural_fabric(self):
        """Test neural hardware fabric."""
        from src.python.breakthrough_engine import NeuralHardwareFabric

        fabric = NeuralHardwareFabric()

        output = fabric.forward({
            "workload": 0.7,
            "thermal_sensor": 0.6,
            "power_sensor": 0.5
        })

        assert "fps" in output
        assert "cpu_clock" in output
        assert "gpu_clock" in output

    def test_quantum_optimizer(self):
        """Test quantum-inspired optimizer."""
        from src.python.breakthrough_engine import QuantumInspiredOptimizer

        optimizer = QuantumInspiredOptimizer()

        def objective(params):
            return params["clock"] - params["power"] * 0.5

        state, params, score = optimizer.anneal(objective, iterations=50)

        assert state in ["powersave", "balanced", "performance", "max_perf"]
        assert score > -1


class TestCognitiveEngine:
    """Test cognitive engine components."""

    def test_feedback_controller(self):
        """Test PID feedback controller."""
        from src.python.cognitive_engine import FeedbackController

        controller = FeedbackController(kp=1.0, ki=0.1, kd=0.05)

        # Simulate control loop
        setpoint = 1.0
        value = 0.0

        for _ in range(50):
            error = setpoint - value
            output = controller.update(error, value)
            value += output * 0.1  # Simulated plant

        # Should have converged near setpoint
        assert abs(value - setpoint) < 0.2

    def test_td_learner(self):
        """Test TD learner."""
        from src.python.cognitive_engine import TDLearner

        learner = TDLearner(state_dim=4, action_dim=2)

        # Learn from experiences
        for _ in range(100):
            state = np.random.random(4)
            action = np.random.random(2)
            reward = np.random.random()
            next_state = np.random.random(4)

            td_error = learner.update(state, action, reward, next_state)

        assert len(learner.experience_buffer) == 100


class TestGPUOptimizers:
    """Test GPU optimizer interfaces."""

    def test_unified_optimizer_creation(self):
        """Test unified GPU optimizer creation."""
        from src.python.gpu_optimizer import create_gpu_optimizer

        optimizer = create_gpu_optimizer()
        assert optimizer is not None

    def test_gpu_detector(self):
        """Test GPU detection."""
        from src.python.gpu_optimizer import GpuDetector

        gpus = GpuDetector.detect_all()
        # May be empty in test environment
        assert isinstance(gpus, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
