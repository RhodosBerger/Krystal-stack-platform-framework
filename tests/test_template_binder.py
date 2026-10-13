import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.template_registry import TemplateRegistry
from krystal_kernel.karaf.template_binder import TemplateBinder, template_input_symbols
from krystal_kernel.verification.evidence_ledger import EvidenceLedger, HardwareDependencyBridge

assert VITAL_MAX_HP == 6
S = 1_000_000_000
ALIASES = {"OBS_TEMP_C": "OBS_CORE_TEMP", "OBS_POWER_W": "OBS_PACKAGE_POWER",
           "OBS_CPU_LOAD": "OBS_CPU_UTIL"}


class BinderTests(unittest.TestCase):
    def setUp(self):
        self.tpl = TemplateRegistry().get_template("thermal_predictor")
        self.assertIsNotNone(self.tpl)

    def test_symbols_parsed(self):
        syms = dict(template_input_symbols(self.tpl))
        self.assertEqual(syms["current_temp_c"], "OBS_TEMP_C")

    def test_missing_reported_not_invented(self):
        b = TemplateBinder(EvidenceLedger()).bind(self.tpl, 0)
        self.assertFalse(b.runnable)
        self.assertIn("OBS_TEMP_C", b.missing)
        self.assertEqual(b.bound, {})

    def test_bind_with_aliases_and_provenance(self):
        br = HardwareDependencyBridge()
        br.ingest({"core_temp": 60.0, "tjmax": 100.0, "package_power": 20.0, "cpu_util": 0.5,
                   "cpu_freq": 3000.0}, now_ns=0)
        L = br.ledger
        L.observe("OBS_CPU_FREQ_ALIAS_UNUSED", 1.0, now_ns=0)
        L.derive("DER_ZTH_SILICON", "div", "OBS_CORE_TEMP", "OBS_PACKAGE_POWER", "C/W", 0)
        # template wants OBS_CPU_FREQ: present directly
        b = TemplateBinder(L, ALIASES).bind(self.tpl, S)
        self.assertTrue(b.runnable, (b.missing, b.stale, b.rejected))
        zth = b.bound["thermal_impedance_zth"]
        self.assertEqual(zth.kind, "DER")
        self.assertEqual(zth.op, "div")
        self.assertEqual(set(zth.lineage), {"OBS_CORE_TEMP", "OBS_PACKAGE_POWER"})
        self.assertAlmostEqual(zth.value, 3.0)

    def test_stale_blocks_run(self):
        L = EvidenceLedger()
        for sym in ("OBS_CORE_TEMP", "OBS_PACKAGE_POWER", "OBS_CPU_UTIL", "OBS_CPU_FREQ"):
            L.observe(sym, 1.0, now_ns=0, ttl_ns=S)
        L.derive("DER_ZTH_SILICON", "div", "OBS_CORE_TEMP", "OBS_PACKAGE_POWER", now_ns=0)
        b = TemplateBinder(L, ALIASES).bind(self.tpl, 10 * S)
        self.assertFalse(b.runnable)
        self.assertTrue(b.stale)

    def test_pred_rejected(self):
        L = EvidenceLedger()
        L.predict("PRED_X", 1.0, now_ns=0)
        self.tpl.inputs = [{"x": "PRED_X"}]
        b = TemplateBinder(L).bind(self.tpl, 0)
        self.assertEqual(b.rejected, ("PRED_X",))


if __name__ == "__main__":
    unittest.main()
