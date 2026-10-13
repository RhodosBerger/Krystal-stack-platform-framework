import unittest

from krystal_kernel.verification.evidence_ledger import (
    EvidenceLedger, HardwareDependencyBridge, LedgerError, PerfClaim, claim_evidence_audit,
)

S = 1_000_000_000


class LedgerTests(unittest.TestCase):
    def test_derive_has_provenance_and_does_not_overwrite(self):
        L = EvidenceLedger()
        L.observe("OBS_A", 2, now_ns=0); L.observe("OBS_B", 3, now_ns=0); L.observe("OBS_C", 5, now_ns=0)
        c = L.derive("DER_C", "add", "OBS_A", "OBS_B", now_ns=0)
        self.assertEqual((c.value, c.op, c.parents), (5, "add", ("OBS_A", "OBS_B")))
        self.assertEqual(L.nodes["OBS_C"].value, 5)
        self.assertEqual(L.check_consistency("DER_C", "OBS_C").verdict, "CONSISTENT")
        L.observe("OBS_D", 9, now_ns=0)
        self.assertEqual(L.check_consistency("DER_C", "OBS_D").verdict, "CONTRADICTORY")
        with self.assertRaises(LedgerError):
            L.observe("OBS_A", 1)

    def test_ops_and_div_guard(self):
        L = EvidenceLedger()
        L.observe("OBS_X", 6, now_ns=0); L.observe("OBS_Y", 0, now_ns=0)
        self.assertEqual(L.derive("DER_S", "sub", "OBS_X", "OBS_Y", now_ns=0).value, 6)
        self.assertEqual(L.derive("DER_M", "mul", "OBS_X", "OBS_Y", now_ns=0).value, 0)
        self.assertTrue(L.derive("DER_D", "div", "OBS_X", "OBS_Y", now_ns=0).value > 1e11)
        self.assertAlmostEqual(L.derive("DER_K", "complement", "OBS_Y", now_ns=0).value, 1.0)

    def test_pred_cannot_feed_der(self):
        L = EvidenceLedger()
        L.observe("OBS_A", 1, now_ns=0); L.predict("PRED_A", 2, ["OBS_A"], now_ns=0)
        with self.assertRaises(LedgerError):
            L.derive("DER_Z", "add", "OBS_A", "PRED_A")

    def test_freshness_inherits_stalest_parent(self):
        L = EvidenceLedger()
        L.observe("OBS_A", 1, now_ns=0, ttl_ns=10 * S); L.observe("OBS_B", 1, now_ns=5 * S, ttl_ns=10 * S)
        d = L.derive("DER_S", "add", "OBS_A", "OBS_B", now_ns=6 * S)
        self.assertTrue(d.fresh(8 * S))
        self.assertFalse(d.fresh(11 * S))

    def test_uncertainty_propagates(self):
        L = EvidenceLedger()
        L.observe("OBS_A", 1, now_ns=0, uncertainty=3); L.observe("OBS_B", 1, now_ns=0, uncertainty=4)
        self.assertAlmostEqual(L.derive("DER_S", "add", "OBS_A", "OBS_B", now_ns=0).uncertainty, 5.0)

    def test_hardware_bridge(self):
        br = HardwareDependencyBridge()
        r = br.ingest({"ram_total": 16e9, "ram_used": 4e9, "core_temp": 60, "tjmax": 100}, now_ns=0)
        self.assertEqual(r["free"].value, 12e9)
        self.assertAlmostEqual(r["pressure"].value, 0.25)
        self.assertEqual(r["headroom"].value, 40)
        self.assertEqual(br.ledger.lineage("DER_THERMAL_HEADROOM"), {"OBS_TJMAX", "OBS_CORE_TEMP"})


class ClaimAuditTests(unittest.TestCase):
    def test_audit(self):
        cl = [PerfClaim("AMX 0.52 ms", "AMX", False, 0), PerfClaim("AVX2 ok", "AVX2", True, 10),
              PerfClaim("redraw 239x", None, False, 1), PerfClaim("x", "AMX", True, 9)]
        v = claim_evidence_audit(cl, host_isa={"AVX2"})
        self.assertEqual([x.status for x in v], ["ISA_MISMATCH", "VERIFIED", "UNVERIFIED", "ISA_MISMATCH"])
        self.assertEqual(claim_evidence_audit(cl[:1], detect=False)[0].status, "ISA_UNKNOWN")


if __name__ == "__main__":
    unittest.main()
