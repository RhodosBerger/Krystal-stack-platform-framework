#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: CASCADE TRACE & CAUSAL EVENT LIFECYCLE
==============================================================================
Module: tests/test_cascade_trace.py
Description:
    Tests for CascadeTraceEvent, CascadeNodeResult, and CascadeFinalResult:
    - 10 canonical lifecycle trace events
    - Causal DAG chain reconstruction (parent_event linkage)
    - Full serialization / deserialization to dict
    - Backward-compatibility of CascadeExecutionTrace
    - Invariant: assert VITAL_MAX_HP == 6
==============================================================================
"""

import unittest
import time
from typing import Dict, Any, List

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.cascade.cascade_trace import (
    CascadeTraceEvent,
    CascadeNodeResult,
    CascadeFinalResult,
    CascadeExecutionTrace
)


class TestCascadeTrace(unittest.TestCase):

    def test_01_vital_max_hp_invariant(self):
        """Universal platform invariant check across trace models."""
        self.assertEqual(VITAL_MAX_HP, 6)
        evt = CascadeTraceEvent(
            event_id="EVT_01",
            request_id="REQ_01",
            node_id="test_node",
            timestamp=time.time(),
            state_version=1,
            parent_event=None,
            latency=1.5,
            memory=32.0,
            token_count=10,
            status="SUCCESS",
            event_type="NODE_START"
        )
        self.assertEqual(evt.vital_max_hp, 6)

        res = CascadeNodeResult(
            node_id="test",
            output="ok",
            latency_ms=1.0,
            input_tokens=2,
            output_tokens=2
        )
        self.assertEqual(res.vital_max_hp, 6)

        final_res = CascadeFinalResult(
            request_id="REQ_01",
            plan_id="PLAN_01",
            final_output="out",
            node_results=[res]
        )
        self.assertEqual(final_res.vital_max_hp, 6)

    def test_02_ten_canonical_event_types(self):
        """Verify all 10 canonical event types can be recorded."""
        canonical_types = [
            "NODE_START", "NODE_END", "ROUTE_DECISION", "RETRIEVAL",
            "MODEL_INFERENCE", "TOOL_CALL", "VERIFY", "FALLBACK",
            "MERGE", "FINAL_SYNTHESIS"
        ]
        for t in canonical_types:
            evt = CascadeTraceEvent(
                event_id=f"EVT_{t}",
                request_id="REQ_CANONICAL",
                node_id="test_node",
                timestamp=time.time(),
                state_version=1,
                parent_event=None,
                latency=0.5,
                memory=16.0,
                token_count=5,
                status="SUCCESS",
                event_type=t
            )
            self.assertEqual(evt.event_type, t)
            d = evt.to_dict()
            self.assertEqual(d["event_type"], t)

    def test_03_causal_dag_chain_reconstruction(self):
        """Verify that parent_event IDs form an unbroken causal DAG from leaf to root."""
        root_evt = CascadeTraceEvent(
            event_id="EVT_ROOT_ROUTE",
            request_id="REQ_CHAIN",
            node_id="router",
            timestamp=100.0,
            state_version=1,
            parent_event=None,
            latency=1.0,
            memory=16.0,
            token_count=10,
            status="SUCCESS",
            event_type="ROUTE_DECISION"
        )
        node_start = CascadeTraceEvent(
            event_id="EVT_NODE_START",
            request_id="REQ_CHAIN",
            node_id="node_a",
            timestamp=101.0,
            state_version=1,
            parent_event=root_evt.event_id,
            latency=0.2,
            memory=16.0,
            token_count=10,
            status="STARTED",
            event_type="NODE_START"
        )
        node_infer = CascadeTraceEvent(
            event_id="EVT_NODE_INFER",
            request_id="REQ_CHAIN",
            node_id="node_a",
            timestamp=102.0,
            state_version=1,
            parent_event=node_start.event_id,
            latency=5.0,
            memory=32.0,
            token_count=20,
            status="SUCCESS",
            event_type="MODEL_INFERENCE"
        )
        node_end = CascadeTraceEvent(
            event_id="EVT_NODE_END",
            request_id="REQ_CHAIN",
            node_id="node_a",
            timestamp=107.0,
            state_version=1,
            parent_event=node_infer.event_id,
            latency=0.1,
            memory=32.0,
            token_count=20,
            status="SUCCESS",
            event_type="NODE_END"
        )

        events_map = {
            e.event_id: e for e in [root_evt, node_start, node_infer, node_end]
        }

        # Traverse backwards from node_end to root
        curr = node_end
        trail = [curr.event_type]
        while curr.parent_event:
            curr = events_map[curr.parent_event]
            trail.append(curr.event_type)

        self.assertEqual(trail, ["NODE_END", "MODEL_INFERENCE", "NODE_START", "ROUTE_DECISION"])

    def test_04_cascade_execution_trace_backward_compat(self):
        """Verify CascadeExecutionTrace retains all legacy and new fields."""
        trace = CascadeExecutionTrace(
            trace_id="TRC_LEGACY_01",
            plan_id="PLAN_01",
            prompt="Test prompt",
            strategy="BALANCED",
            objective="BALANCED",
            final_output="Output text",
            total_latency_ms=15.5,
            peak_memory_mb=128.0,
            final_confidence=0.95,
            dual_gate_verdict="AUTH_COMMIT_APPROVED"
        )

        d = trace.to_dict()
        self.assertEqual(d["trace_id"], "TRC_LEGACY_01")
        self.assertEqual(d["request_id"], "TRC_LEGACY_01")
        self.assertEqual(d["authority_status"], "AUTH_COMMIT_APPROVED")
        self.assertEqual(d["dual_gate_verdict"], "AUTH_COMMIT_APPROVED")
        self.assertEqual(d["total_memory_peak"], 128.0)
        self.assertEqual(d["vital_max_hp"], 6)


if __name__ == "__main__":
    unittest.main()
