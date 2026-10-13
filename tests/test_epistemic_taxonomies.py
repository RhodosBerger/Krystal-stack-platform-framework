#!/usr/bin/env python3
"""
Unit tests for Epistemic Taxonomies (Boolean, List) and Predictive Dependency Graph.
"""

import time
import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.epistemic_taxonomies import (
    BooleanActionKind,
    BooleanInvariantKind,
    BooleanModalKind,
    BooleanTemporalKind,
    DependencyEdgeKind,
    EpistemicBoolean,
    EpistemicList,
    InvariantViolationError,
    ListAlgebraicKind,
    ListCapacityKind,
    ListIntegrityKind,
    ListMemoryKind,
    PredictiveDependencyGraph,
    PredictiveDependencyNode,
    create_vital_hearts_list,
)


class TestEpistemicTaxonomies(unittest.TestCase):

    def test_boolean_taxonomies_committable_and_veto(self):
        # 1. Verified True is committable
        b_true = EpistemicBoolean(
            value=True,
            modal_kind=BooleanModalKind.VERIFIED_TRUE,
            confidence=0.95,
        )
        self.assertTrue(b_true.is_committable())

        # 2. Vetoed False cannot participate in commit
        b_veto = EpistemicBoolean(
            value=False,
            modal_kind=BooleanModalKind.VETOED_FALSE,
            confidence=1.0,
        )
        self.assertFalse(b_veto.is_committable())

        # 3. Quarantined or Indeterminate cannot commit
        b_quarantine = EpistemicBoolean(
            value=True,
            modal_kind=BooleanModalKind.QUARANTINED,
            confidence=0.99,
        )
        self.assertFalse(b_quarantine.is_committable())

        b_pending = EpistemicBoolean(
            value=True,
            modal_kind=BooleanModalKind.INDETERMINATE_PENDING,
            confidence=0.50,
        )
        self.assertFalse(b_pending.is_committable())

    def test_boolean_temporal_lease_expiry(self):
        t0 = 1_000_000_000
        ttl = 100_000_000  # 100ms
        b_leased = EpistemicBoolean(
            value=True,
            modal_kind=BooleanModalKind.VERIFIED_TRUE,
            temporal_kind=BooleanTemporalKind.EPOCH_LEASED,
            valid_from_ns=t0,
            lease_ttl_ns=ttl,
        )

        # Before expiry
        self.assertFalse(b_leased.is_expired(now_ns=t0 + 50_000_000))
        self.assertTrue(b_leased.is_committable(now_ns=t0 + 50_000_000))

        # After expiry
        self.assertTrue(b_leased.is_expired(now_ns=t0 + 150_000_000))
        self.assertFalse(b_leased.is_committable(now_ns=t0 + 150_000_000))

    def test_boolean_constitutional_invariant_assertion(self):
        # Sound constitutional invariant
        b_const = EpistemicBoolean(
            value=True,
            modal_kind=BooleanModalKind.VERIFIED_TRUE,
            invariant_kind=BooleanInvariantKind.CONSTITUTIONAL_INVARIANT,
            domain="vital.hp.guard",
        )
        # Should not raise
        b_const.assert_constitutional()

        # Adversarial attempt to violate constitutional invariant
        b_adversarial = EpistemicBoolean(
            value=False,
            modal_kind=BooleanModalKind.FALSIFIED_FALSE,
            invariant_kind=BooleanInvariantKind.CONSTITUTIONAL_INVARIANT,
            domain="vital.hp.guard",
        )
        with self.assertRaises(InvariantViolationError):
            b_adversarial.assert_constitutional()

    def test_epistemic_list_vital_bounded_invariant(self):
        vital_list: EpistemicList[int] = EpistemicList(
            capacity_kind=ListCapacityKind.VITAL_BOUNDED_LIST,
        )
        self.assertEqual(vital_list.max_capacity, VITAL_MAX_HP)
        self.assertEqual(vital_list.max_capacity, 6)

        # Appending up to 6 succeeds
        for i in range(6):
            vital_list.append(i + 1)
        self.assertEqual(len(vital_list), 6)

        # Appending a 7th item strictly breaches invariant
        with self.assertRaises(InvariantViolationError):
            vital_list.append(7)
        self.assertEqual(len(vital_list), 6)

    def test_create_vital_hearts_list(self):
        hearts = create_vital_hearts_list(4)
        self.assertEqual(len(hearts), 6)
        active_count = sum(1 for h in hearts if h.value)
        inactive_count = sum(1 for h in hearts if not h.value)
        self.assertEqual(active_count, 4)
        self.assertEqual(inactive_count, 2)
        self.assertTrue(hearts.merkle_root != "0" * 64)

    def test_epistemic_list_circular_ring_buffer(self):
        ring: EpistemicList[str] = EpistemicList(
            capacity_kind=ListCapacityKind.UNBOUNDED_EVIDENCE_STREAM,
            algebraic_kind=ListAlgebraicKind.CIRCULAR_RING_BUFFER,
            max_capacity=3,
        )
        ring.append("A")
        ring.append("B")
        ring.append("C")
        self.assertEqual(ring.items, ["A", "B", "C"])

        # Overwrite oldest
        ring.append("D")
        self.assertEqual(ring.items, ["B", "C", "D"])
        self.assertEqual(len(ring), 3)

    def test_epistemic_list_merkle_tamper_detection(self):
        list1: EpistemicList[int] = EpistemicList(capacity_kind=ListCapacityKind.UNBOUNDED_EVIDENCE_STREAM)
        list1.append(10)
        list1.append(20)

        list2: EpistemicList[int] = EpistemicList(capacity_kind=ListCapacityKind.UNBOUNDED_EVIDENCE_STREAM)
        list2.append(10)
        list2.append(20)

        # Identical insertion sequences produce identical Merkle roots
        self.assertEqual(list1.merkle_root, list2.merkle_root)

        # Inserting a differing item diverges Merkle roots
        list1.append(30)
        list2.append(31)
        self.assertNotEqual(list1.merkle_root, list2.merkle_root)

    def test_predictive_dependency_graph_invalidation(self):
        graph = PredictiveDependencyGraph()

        # Nodes
        graph.add_node(PredictiveDependencyNode("hw_thermal", "HARDWARE_SENSOR", "MEASURED"))
        graph.add_node(PredictiveDependencyNode("thermal_forecast", "BACKEND_KERNEL", "PREDICTED"))
        graph.add_node(PredictiveDependencyNode("ui_thermal_hud", "FRONTEND_UI", "VIEW_MODEL"))
        graph.add_node(PredictiveDependencyNode("ui_unrelated", "FRONTEND_UI", "VIEW_MODEL"))

        # Edges
        graph.add_edge("hw_thermal", "thermal_forecast", DependencyEdgeKind.PREDICTS)
        graph.add_edge("thermal_forecast", "ui_thermal_hud", DependencyEdgeKind.INVALIDATES)

        # When hw_thermal changes, both thermal_forecast and ui_thermal_hud must be invalidated
        invalidation_set = graph.get_invalidation_set("hw_thermal")
        self.assertIn("hw_thermal", invalidation_set)
        self.assertIn("thermal_forecast", invalidation_set)
        self.assertIn("ui_thermal_hud", invalidation_set)
        self.assertNotIn("ui_unrelated", invalidation_set)


if __name__ == "__main__":
    unittest.main()
