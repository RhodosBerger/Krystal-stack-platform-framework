import ctypes
import importlib.util
import json
import unittest
from pathlib import Path

from tools import native_abi_gen as g

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "native" / "generated"
SPEC_DIR = ROOT / "native" / "spec"


def _ctypes_mod():
    p = GEN / "kr_resource_evidence_envelope_ctypes.py"
    sp = importlib.util.spec_from_file_location("env_ct", p)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


class TestNativeAbi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g.main()
        cls.spec = g.load_spec()
        cls.s = cls.spec["structs"][0]

    def test_spec_has_no_implicit_padding(self):
        g.validate_struct(self.s)

    def test_validator_rejects_gap(self):
        bad = json.loads(json.dumps(self.s))
        bad["fields"][1]["offset"] = 6  # overlap/gap
        with self.assertRaises(ValueError):
            g.validate_struct(bad)

    def test_ctypes_layout_matches_spec(self):
        m = _ctypes_mod()
        cls = m.kr_resource_evidence_envelope
        self.assertEqual(ctypes.sizeof(cls), self.s["size"])
        for f in self.s["fields"]:
            self.assertEqual(getattr(cls, f["name"]).offset, f["offset"], f["name"])

    def test_c_and_rust_carry_static_asserts(self):
        h = (GEN / "kr_resource_evidence_envelope.h").read_text()
        r = (GEN / "kr_resource_evidence_envelope.rs").read_text()
        for f in self.s["fields"]:
            self.assertIn(f"offsetof(kr_resource_evidence_envelope_t, {f['name']}) == {f['offset']}", h)
            self.assertIn(f"offset_of!(KrResourceEvidenceEnvelope, {f['name']}) == {f['offset']}", r)

    def test_golden_roundtrip_and_ctypes_agree(self):
        data = (GEN / "kr_resource_evidence_envelope.golden.bin").read_bytes()
        self.assertEqual(len(data), 64)
        d = g.decode(self.s, data)
        self.assertEqual(d["state_version"], 42)
        self.assertEqual(d["value"], 71.5)
        self.assertEqual(g.encode(self.s, d), data)
        c = _ctypes_mod().kr_resource_evidence_envelope.from_buffer_copy(data)
        self.assertEqual(c.state_version, 42)
        self.assertEqual(c.lease_ttl_ns, 250_000_000)
        self.assertEqual(c.crc32, d["crc32"])

    def test_crc_detects_corruption(self):
        data = bytearray((GEN / "kr_resource_evidence_envelope.golden.bin").read_bytes())
        data[33] ^= 0xFF
        with self.assertRaises(ValueError):
            g.decode(self.s, bytes(data))

    def test_epistemic_transitions_spec(self):
        e = json.loads((SPEC_DIR / "epistemic_types.yaml").read_text())
        self.assertIn(["PREDICTED", "OBSERVED"], e["forbidden_conversions"])
        self.assertEqual(e["actuation_allowed_from"], ["VALIDATED"])
        self.assertFalse(e["authority"]["learning_may_mutate_authority_policy"])

    def test_claims_not_encoded_as_truth(self):
        c = json.loads((SPEC_DIR / "claims_ledger.json").read_text())
        for cl in c["claims"]:
            self.assertIn(cl["status"], {"DECLARED", "MEASURED", "REPRODUCED", "UNVERIFIED"})
            if cl["status"] == "REPRODUCED":
                self.assertIsNotNone(cl["evidence"])


if __name__ == "__main__":
    unittest.main()
