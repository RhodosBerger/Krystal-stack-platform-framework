"""
Unit Tests for Symplectic Hamiltonian Cyclic Organism Engine
=============================================================
Tests:
1. Symplectic energy conservation under zero damping
2. Restoring forces toward homeostatic equilibrium
3. Cognitive Brainwave transitions (ALPHA, BETA, GAMMA, OMEGA)
4. Non-linear visual entropy backpressure regulation
5. JSON Schema validation against schemas/cyclic_architecture_schema.json
"""

import unittest
import os
import sys
import json

try:
    import jsonschema
    has_jsonschema = True
except ImportError:
    has_jsonschema = False

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.python.cyclic_organism_kernel import SymplecticCyclicEngine

class TestCyclicOrganism(unittest.TestCase):

    def setUp(self):
        self.engine = SymplecticCyclicEngine(organism_id="Test-Organism-01")
        schema_path = os.path.join(REPO_ROOT, "schemas", "cyclic_architecture_schema.json")
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)

    def test_01_symplectic_energy_conservation(self):
        """Under zero dissipation and zero external force, Hamiltonian energy H is conserved."""
        engine_ideal = SymplecticCyclicEngine(
            organism_id="Ideal-Engine",
            linear_damping_gamma=0.0,
            nonlinear_entropy_beta=0.0,
            enable_economic_forcing=False
        )
        # Give initial non-zero displacement and momentum
        engine_ideal.q[0] = 1.15
        engine_ideal.p[0] = 0.3
        h_initial = engine_ideal.total_hamiltonian()

        # Step 200 cycles with zero entropy
        for _ in range(200):
            engine_ideal.symplectic_step(dt=0.005, visual_entropy=0.0, gpu_temp=45.0)

        h_final = engine_ideal.total_hamiltonian()
        relative_drift = abs(h_final - h_initial) / max(0.1, h_initial)
        # Symplectic Velocity-Verlet guarantees bounded energy oscillations (< 10%)
        self.assertLess(relative_drift, 0.10)

    def test_02_homeostatic_restoring_force(self):
        """Displacing coordinates from q* generates restoring force back toward setpoint."""
        engine = SymplecticCyclicEngine()
        # Displace q[0] far to the right of setpoint
        engine.q[0] = engine.q_star[0] + 0.5
        forces = engine.potential_gradient(engine.q)
        # Force must point in the negative direction (restoring)
        self.assertLess(forces[0], 0.0)

    def test_03_omega_phase_backpressure_transition(self):
        """High visual entropy (E > 0.70) must immediately trigger OMEGA phase and throttled step budget."""
        engine = SymplecticCyclicEngine()
        # Normal step
        engine.symplectic_step(dt=0.033, visual_entropy=0.30, gpu_temp=50.0)
        self.assertNotEqual(engine.cognitive_phase, "OMEGA")

        # Inject high entropy spike
        engine.symplectic_step(dt=0.033, visual_entropy=0.82, gpu_temp=50.0)
        self.assertEqual(engine.cognitive_phase, "OMEGA")

        # Check actuation
        act = engine.get_control_actuation()
        self.assertEqual(act["thread_priority"], "LOW")
        self.assertLessEqual(act["raymarch_step_budget"], 24)

    def test_04_gamma_phase_hyperfocus(self):
        """High computational momentum under low entropy triggers GAMMA phase."""
        engine = SymplecticCyclicEngine()
        engine.p = [0.8, 0.8, 0.5, 0.5]  # High momentum norm > 1.2
        engine.symplectic_step(dt=0.033, visual_entropy=0.20, gpu_temp=45.0)
        self.assertEqual(engine.cognitive_phase, "GAMMA")

        act = engine.get_control_actuation()
        self.assertEqual(act["thread_priority"], "REALTIME")
        self.assertGreaterEqual(act["raymarch_step_budget"], 32)

    def test_05_schema_validation(self):
        """Generated state dictionary must fully validate against cyclic_architecture_schema.json."""
        engine = SymplecticCyclicEngine(organism_id="Alpha-Validator")
        engine.symplectic_step(dt=0.033, visual_entropy=0.35, gpu_temp=52.0)

        data = engine.to_schema_dict()

        # Validate schema
        if has_jsonschema:
            jsonschema.validate(instance=data, schema=self.schema)
        else:
            for req in self.schema["required"]:
                self.assertIn(req, data)
            for req in self.schema["properties"]["hamiltonian_parameters"]["required"]:
                self.assertIn(req, data["hamiltonian_parameters"])
            for req in self.schema["properties"]["state_vector"]["required"]:
                self.assertIn(req, data["state_vector"])
            for req in self.schema["properties"]["telemetry"]["required"]:
                self.assertIn(req, data["telemetry"])

        self.assertEqual(data["organism_id"], "Alpha-Validator")
        self.assertIn(data["cognitive_phase"], ["ALPHA", "BETA", "GAMMA", "OMEGA"])
        self.assertIn("total_energy_h", data["telemetry"])

    def test_06_phase_portrait_generation(self):
        """Phase portrait trajectory generator returns valid coordinates."""
        engine = SymplecticCyclicEngine()
        trajectory = engine.generate_phase_trajectory(num_points=16, dt=0.033)
        self.assertEqual(len(trajectory), 16)
        for pt in trajectory:
            self.assertEqual(len(pt), 2)
            self.assertIsInstance(pt[0], float)
            self.assertIsInstance(pt[1], float)


if __name__ == "__main__":
    unittest.main()
