import unittest

from krystal_kernel.runtime.klang_value import (
    AuthorityBoundary, ExecutionDomain as D, KLangError, KValue, Lifetime, evidence_meet,
)
from krystal_kernel.verification.semantic_protocol import AuthorityState as A, EpistemicState as E

OK = AuthorityBoundary(A.AUTH_COMMIT_APPROVED, D.USER, frozenset({"write", "promote"}))


def kv(v=1, ev=E.EPI_VALIDATED, life=None, dom=D.USER, b=OK):
    return KValue(v, int, ev, life or Lifetime.static(), dom, b)


class KLangValueTests(unittest.TestCase):
    def test_type_enforced(self):
        with self.assertRaises(KLangError):
            KValue("x", int)

    def test_combine_takes_weakest_facets(self):
        a = kv(2, E.EPI_VALIDATED, Lifetime(0, 100), D.KERNEL)
        b = kv(3, E.EPI_OBSERVED, Lifetime(0, 50), D.NETWORK, AuthorityBoundary(A.AUTH_DEFERRED, D.NETWORK))
        c = a.combine(b, lambda x, y: x * y, int)
        self.assertEqual(c.value, 6)
        self.assertEqual(c.evidence, E.EPI_DERIVED)
        self.assertEqual(c.lifetime.expires_ns, 50)
        self.assertEqual(c.domain, D.NETWORK)
        self.assertEqual(c.boundary.authority, A.AUTH_DEFERRED)
        self.assertEqual(c.boundary.effects, frozenset())

    def test_poison_dominates(self):
        self.assertEqual(evidence_meet(E.EPI_VALIDATED, E.EPI_FALSIFIED), E.EPI_FALSIFIED)
        with self.assertRaises(KLangError):
            kv(ev=E.EPI_QUARANTINED).cross(D.SHADOW)

    def test_commit_requires_all_facets(self):
        self.assertEqual(kv(7).commit("write", now_ns=1), 7)
        with self.assertRaises(KLangError):
            kv(ev=E.EPI_PREDICTED).commit("write")           # evidence
        with self.assertRaises(KLangError):
            kv(life=Lifetime(0, 10)).commit("write", now_ns=11)   # expired
        with self.assertRaises(KLangError):
            kv(life=Lifetime(0, 10)).commit("write", now_ns=9)    # stale (>=80%)
        with self.assertRaises(KLangError):
            kv().commit("delete")                              # effect not granted
        with self.assertRaises(KLangError):
            kv(b=AuthorityBoundary()).commit("write")          # not approved

    def test_cross_rules(self):
        self.assertEqual(kv().cross(D.USER).domain, D.USER)
        with self.assertRaises(KLangError):
            kv().cross(D.NETWORK)                              # beyond ceiling
        self.assertEqual(kv().cross(D.KERNEL).domain, D.KERNEL)  # validated + promote permit
        with self.assertRaises(KLangError):
            kv(ev=E.EPI_CALIBRATED).cross(D.KERNEL)            # promotion needs VALIDATED
        with self.assertRaises(KLangError):
            kv(life=Lifetime(0, 5)).cross(D.USER, now_ns=6)    # expired

    def test_map_never_stronger_than_derived(self):
        self.assertEqual(kv(2).map(lambda x: x + 1, int).evidence, E.EPI_DERIVED)


if __name__ == "__main__":
    unittest.main()
