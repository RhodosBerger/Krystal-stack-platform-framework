#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: UNIT TEST SUITE FOR CAUSAL EVENT LOG & REPLAY (FAMILIES P & Q)
==============================================================================
Module: tests/test_causal_event_log.py
==============================================================================
"""

import time
import unittest

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.verification.state_version import StateVersion
from krystal_kernel.verification.causal_event_log import (
    CausalEvent,
    CausalEventClass,
    CausalEventLog
)


class TestCausalEventLog(unittest.TestCase):

    def setUp(self):
        self.log = CausalEventLog(capacity=100)
        self.sv = StateVersion(major=1, minor=0, sequence=1, wall_timestamp_utc=time.time())

    def test_emit_and_retrieve_event(self):
        """Events are appended and indexed correctly."""
        ev = self.log.emit(
            event_class=CausalEventClass.OBSERVATION,
            state_version=self.sv,
            payload={"metric": "cpu", "value": 42.0},
            clock_domain="host_clock_0"
        )
        self.assertIsNotNone(ev.event_id)
        retrieved = self.log.get_event(ev.event_id)
        self.assertEqual(retrieved.event_id, ev.event_id)
        self.assertEqual(retrieved.event_class, CausalEventClass.OBSERVATION)
        self.assertEqual(retrieved.vital_hp, VITAL_MAX_HP)

    def test_causal_lineage_graph_reconstruction(self):
        """Lineage DAG correctly walks backwards from authority verdict to raw observation."""
        now_ns = int(time.time() * 1e9)
        
        # 1. Observation
        ev1 = self.log.emit(
            event_class=CausalEventClass.OBSERVATION,
            state_version=self.sv,
            payload={"sensor": "cgroup"},
            now_ns=now_ns
        )

        # 2. Prediction (parent = ev1)
        ev2 = self.log.emit(
            event_class=CausalEventClass.PREDICTION,
            state_version=self.sv,
            causal_parent=ev1.event_id,
            payload={"expected_delta": -4.0},
            now_ns=now_ns + 1000
        )

        # 3. CandidateAction (parent = ev2)
        ev3 = self.log.emit(
            event_class=CausalEventClass.CANDIDATE,
            state_version=self.sv,
            causal_parent=ev2.event_id,
            payload={"action": "RESIZE_BALLOON"},
            now_ns=now_ns + 2000
        )

        # 4. AuthorityVerdict (parent = ev3)
        ev4 = self.log.emit(
            event_class=CausalEventClass.AUTHORITY_VERDICT,
            state_version=self.sv,
            causal_parent=ev3.event_id,
            payload={"verdict_type": "COMMIT_APPROVED", "candidate_id": "ACT-01"},
            now_ns=now_ns + 3000
        )

        lineage = self.log.get_lineage(ev4.event_id)
        self.assertEqual(len(lineage), 4)
        self.assertEqual(lineage[0].event_id, ev1.event_id)
        self.assertEqual(lineage[1].event_id, ev2.event_id)
        self.assertEqual(lineage[2].event_id, ev3.event_id)
        self.assertEqual(lineage[3].event_id, ev4.event_id)

    def test_causal_integrity_monotonic_timestamps(self):
        """Validates that parent timestamp <= child timestamp within same clock domain."""
        now_ns = int(time.time() * 1e9)
        ev1 = self.log.emit(
            event_class=CausalEventClass.OBSERVATION,
            state_version=self.sv,
            payload={},
            clock_domain="host_clock_0",
            now_ns=now_ns
        )
        ev2 = self.log.emit(
            event_class=CausalEventClass.PREDICTION,
            state_version=self.sv,
            causal_parent=ev1.event_id,
            payload={},
            clock_domain="host_clock_0",
            now_ns=now_ns + 500
        )

        ok, errors = self.log.verify_causal_integrity()
        self.assertTrue(ok)
        self.assertEqual(len(errors), 0)

    def test_causal_integrity_detects_inversion(self):
        """Detects backwards causal timestamp inversions within the same clock domain."""
        now_ns = int(time.time() * 1e9)
        ev1 = self.log.emit(
            event_class=CausalEventClass.OBSERVATION,
            state_version=self.sv,
            payload={},
            clock_domain="host_clock_0",
            now_ns=now_ns + 5000 # Child in future relative to parent? No, parent is ahead!
        )
        ev2 = self.log.emit(
            event_class=CausalEventClass.PREDICTION,
            state_version=self.sv,
            causal_parent=ev1.event_id,
            payload={},
            clock_domain="host_clock_0",
            now_ns=now_ns # Timestamp inversion!
        )

        ok, errors = self.log.verify_causal_integrity()
        self.assertFalse(ok)
        self.assertGreater(len(errors), 0)
        self.assertIn("Causal inversion", errors[0])

    def test_deterministic_replay_verdict(self):
        """Replay confirms exact verdict reproduction."""
        ev = self.log.emit(
            event_class=CausalEventClass.AUTHORITY_VERDICT,
            state_version=self.sv,
            payload={"verdict_type": "COMMIT_APPROVED", "candidate_id": "ACT-REPLAY-01"}
        )
        res = self.log.replay_verdict(ev.event_id)
        self.assertTrue(res["success"])
        self.assertTrue(res["deterministic_match"])
        self.assertEqual(res["recorded_verdict"], "COMMIT_APPROVED")
        self.assertEqual(res["vital_max_hp_rule"], 6)


if __name__ == "__main__":
    unittest.main()
