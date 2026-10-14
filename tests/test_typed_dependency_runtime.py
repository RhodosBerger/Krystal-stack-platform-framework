import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.typed_dependency_runtime import (
    DependencyCycleError, DependencyRuntimeError, DependencyValue, EpistemicState,
    StaleEvidenceError, TypedDependencyRuntime, UnitError, convert_physical,
)

T0 = 1_000_000_000


class TestTypedDependencyRuntime(unittest.TestCase):
    def setUp(self):
        assert VITAL_MAX_HP == 6
        self.rt = TypedDependencyRuntime(clock=lambda: T0)
        self.rt.observe("OBS_RAM_USED", 12 * 2**30, "Bytes", uncertainty=2**20, now_ns=T0)
        self.rt.observe("OBS_RAM_TOTAL", 16 * 2**30, "Bytes", now_ns=T0)

    def test_derive_pressure_with_provenance(self):
        d = self.rt.derive("DER_RAM_PRESSURE", "divide", "OBS_RAM_USED", "OBS_RAM_TOTAL", now_ns=T0)
        self.assertAlmostEqual(d.value, 0.75)
        self.assertEqual(d.unit, "Ratio")
        self.assertEqual(d.epistemic_state, EpistemicState.DER)
        self.assertEqual(d.parent_ids, ("OBS_RAM_USED", "OBS_RAM_TOTAL"))
        self.assertGreater(d.uncertainty, 0)
        self.assertEqual(self.rt.read_set("DER_RAM_PRESSURE"), ["OBS_RAM_TOTAL", "OBS_RAM_USED"])
        self.assertAlmostEqual(self.rt.replay("DER_RAM_PRESSURE"), 0.75)

    def test_unit_checks_and_conversion(self):
        self.assertAlmostEqual(convert_physical(0, "degC", "K"), 273.15)
        self.assertAlmostEqual(convert_physical(1500, "ms", "s"), 1.5)
        with self.assertRaises(UnitError):
            convert_physical(1, "ms", "K")
        self.rt.observe("OBS_T", 70, "degC", now_ns=T0)
        with self.assertRaises(UnitError):
            self.rt.derive("X", "add", "OBS_T", "OBS_RAM_USED", now_ns=T0)
        with self.assertRaises(UnitError):
            self.rt.derive("X", "complement", "OBS_RAM_USED", now_ns=T0)

    def test_der_never_obs(self):
        with self.assertRaises(DependencyRuntimeError):
            DependencyValue("A", 1.0, "Ratio", EpistemicState.DER, T0, T0 + 1)
        with self.assertRaises(DependencyRuntimeError):
            DependencyValue("A", 1.0, "Ratio", EpistemicState.OBS, T0, T0 + 1, parent_ids=("p",))

    def test_expired_evidence_rejected(self):
        later = T0 + 10_000_000_000
        with self.assertRaises(StaleEvidenceError):
            self.rt.derive("D", "divide", "OBS_RAM_USED", "OBS_RAM_TOTAL", now_ns=later)
        self.assertEqual(self.rt.trace("OBS_RAM_USED", now_ns=later)["temporal"], "EXPIRED")

    def test_divide_by_zero_guarded_and_cycle_error(self):
        self.rt.observe("OBS_Z", 0.0, "Bytes", now_ns=T0)
        d = self.rt.derive("D0", "divide", "OBS_RAM_USED", "OBS_Z", now_ns=T0)
        self.assertTrue(d.value == d.value and abs(d.value) != float("inf"))
        with self.assertRaises(DependencyCycleError):
            self.rt.derive("OBS_RAM_USED", "divide", "OBS_RAM_USED", "OBS_RAM_TOTAL", now_ns=T0)

    def test_latent_projection_is_labelled_model_not_physics(self):
        self.rt.observe("OBS_TEMP", 70, "degC", now_ns=T0)
        p = self.rt.project_latent(
            "DER_LATENCY", "OBS_TEMP", lambda t: t / 100.0, lambda x: 40 + 60 * x, "ns",
            "quad_manifold_v1", "calib_2026_10", 4.0, "thermal", "memory_latency", now_ns=T0)
        self.assertEqual(p.epistemic_state, EpistemicState.DER_LATENT_PROJECTION)
        self.assertAlmostEqual(p.value, 82.0)
        self.assertFalse(p.metadata["physical_law"])
        self.assertEqual(p.metadata["projection_model_id"], "quad_manifold_v1")
        self.assertGreaterEqual(p.uncertainty, 4.0)

    def test_trace_tree(self):
        self.rt.derive("DER_P", "divide", "OBS_RAM_USED", "OBS_RAM_TOTAL", now_ns=T0)
        self.rt.derive("DER_HEADROOM", "complement", "DER_P", now_ns=T0)
        tr = self.rt.trace("DER_HEADROOM", now_ns=T0)
        self.assertAlmostEqual(tr["value"], 0.25)
        self.assertEqual(tr["parents"][0]["id"], "DER_P")
        self.assertEqual(len(tr["parents"][0]["parents"]), 2)
        self.assertAlmostEqual(self.rt.replay("DER_HEADROOM"), 0.25)


if __name__ == "__main__":
    unittest.main()
