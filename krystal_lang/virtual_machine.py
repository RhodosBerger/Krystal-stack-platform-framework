"""
Krystal-Lang: Topological Virtual Machine (K-TVM)
=================================================
Executes queue-partitioned bytecode programs, streams data packets
through geometric manifolds, and tracks computational entropy.
"""

import queue
import time
import math
from typing import Dict, Any, List, Optional, Union
from krystal_lang.fast_queue import FastRingBuffer

class TopologicalVM:
    """
    Virtual machine executing Krystal-Lang queue pipelines and bytecode instructions.
    Supports standard thread-safe queues or zero-lock FastRingBuffer channels.
    """
    def __init__(self, compilation_result: Dict[str, Any], use_fast_queue: bool = False):
        self.comp = compilation_result
        self.bytecode = self.comp.get("bytecode", [])
        self.partitions = self.comp.get("queue_partitions", [])
        self.use_fast_queue = use_fast_queue
        self.channels: Dict[str, Union[queue.Queue, FastRingBuffer]] = {}
        self.metrics = {
            "packets_processed": 0,
            "cycles": 0,
            "active_queues": 0,
            "average_throughput_pps": 0.0,
            "queue_depths": {},
            "what_if_resolutions": 0,
            "engine_mode": "LOCK_FREE_RING" if use_fast_queue else "STANDARD_MUTEX"
        }
        self._init_channels()

    def _create_channel(self, name: str, capacity: int = 256):
        if self.use_fast_queue:
            self.channels[name] = FastRingBuffer(capacity)
        else:
            self.channels[name] = queue.Queue(maxsize=capacity)
        self.metrics["queue_depths"][name] = 0

    def _init_channels(self):
        for instr in self.bytecode:
            op = instr.get("op")
            if op == "OP_ALLOC_QUEUE":
                q_name = instr["name"]
                cap = instr.get("capacity", 256)
                if q_name not in self.channels:
                    self._create_channel(q_name, cap)
            elif op in ("OP_BRANCH_IF", "OP_BRANCH_ELIF", "OP_BRANCH_ELSE"):
                for qk in ("source_queue", "target_queue"):
                    qn = instr.get(qk)
                    if qn and qn not in self.channels:
                        self._create_channel(qn, 256)
            elif op == "OP_WHAT_IF_SPECULATE":
                for qk in ("source_queue", "fallback_queue"):
                    qn = instr.get(qk)
                    if qn and qn not in self.channels:
                        self._create_channel(qn, 256)
                for sc in instr.get("scenarios", []):
                    tq = sc.get("target_queue")
                    if tq and tq not in self.channels:
                        self._create_channel(tq, 256)
        self.metrics["active_queues"] = len(self.channels)

    def _eval_condition(self, pkt: Any, cond: Dict[str, Any]) -> bool:
        if not isinstance(pkt, dict):
            return True
        field = cond.get("field", "status")
        op = cond.get("op", "==")
        val = cond.get("val")

        if field not in pkt:
            return False

        pkt_val = pkt[field]
        try:
            if op == ">": return float(pkt_val) > float(val)
            elif op == "<": return float(pkt_val) < float(val)
            elif op == ">=": return float(pkt_val) >= float(val)
            elif op == "<=": return float(pkt_val) <= float(val)
            elif op == "==": return pkt_val == val or str(pkt_val) == str(val)
            elif op == "!=": return pkt_val != val and str(pkt_val) != str(val)
        except (ValueError, TypeError):
            return str(pkt_val) == str(val)
        return False

    def inject_input(self, queue_name: str, item: Any) -> bool:
        """Injects data packet into a target queue."""
        if queue_name not in self.channels:
            return False
        q = self.channels[queue_name]
        if self.use_fast_queue:
            ok = q.put_nowait(item)
            if ok:
                self.metrics["queue_depths"][queue_name] = q.qsize()
            return ok
        else:
            try:
                q.put_nowait(item)
                self.metrics["queue_depths"][queue_name] = q.qsize()
                return True
            except queue.Full:
                return False

    def step_execution(self, cycles: int = 1) -> Dict[str, Any]:
        """Executes cycles of queue pipelining across topological stages."""
        t0 = time.perf_counter()
        packets_this_step = 0

        for _ in range(cycles):
            self.metrics["cycles"] += 1

            # Dispatch instructions
            for instr in self.bytecode:
                op = instr.get("op")
                if op == "OP_QUEUE_DISPATCH":
                    src = instr["from_queue"]
                    dst = instr["to_queue"]
                    action = instr.get("action", "PASS")

                    if src in self.channels and dst in self.channels:
                        src_q = self.channels[src]
                        dst_q = self.channels[dst]

                        for _ in range(8):
                            if src_q.empty():
                                break
                            if self.use_fast_queue:
                                pkt = src_q.get_nowait()
                                if pkt is None:
                                    break
                                if action == "EVAL_SDF":
                                    out_pkt = {
                                        "source": pkt,
                                        "sdf_val": round(math.sin(float(self.metrics["cycles"]) * 0.1), 4),
                                        "status": "EVALUATED"
                                    }
                                else:
                                    out_pkt = pkt
                                if not dst_q.put_nowait(out_pkt):
                                    break
                                packets_this_step += 1
                            else:
                                try:
                                    pkt = src_q.get_nowait()
                                    if action == "EVAL_SDF":
                                        out_pkt = {
                                            "source": pkt,
                                            "sdf_val": round(math.sin(float(self.metrics["cycles"]) * 0.1), 4),
                                            "status": "EVALUATED"
                                        }
                                    else:
                                        out_pkt = pkt

                                    dst_q.put_nowait(out_pkt)
                                    packets_this_step += 1
                                except (queue.Empty, queue.Full):
                                    break

                elif op == "OP_BRANCH_IF":
                    ladder_name = instr.get("ladder_name")
                    src = instr.get("source_queue")
                    ladder_branches = [
                        b for b in self.bytecode
                        if b.get("ladder_name") == ladder_name and b.get("op") in ("OP_BRANCH_IF", "OP_BRANCH_ELIF", "OP_BRANCH_ELSE")
                    ]

                    if src in self.channels:
                        src_q = self.channels[src]
                        for _ in range(8):
                            if src_q.empty():
                                break
                            try:
                                pkt = src_q.get_nowait()
                            except Exception:
                                break
                            if pkt is None:
                                break

                            routed = False
                            for branch in ladder_branches:
                                b_op = branch.get("op")
                                b_cond = branch.get("condition")
                                b_dst = branch.get("target_queue")

                                if b_op == "OP_BRANCH_ELSE" or (b_cond and self._eval_condition(pkt, b_cond)):
                                    if b_dst in self.channels:
                                        out_pkt = dict(pkt) if isinstance(pkt, dict) else {"payload": pkt}
                                        out_pkt["branch_taken"] = b_op
                                        out_pkt["ladder"] = ladder_name
                                        self.channels[b_dst].put_nowait(out_pkt)
                                        packets_this_step += 1
                                        routed = True
                                        break

                            if not routed:
                                src_q.put_nowait(pkt)
                                break

                elif op in ("OP_BRANCH_ELIF", "OP_BRANCH_ELSE"):
                    pass

                elif op == "OP_WHAT_IF_SPECULATE":
                    src = instr.get("source_queue")
                    fb = instr.get("fallback_queue")
                    scenarios = instr.get("scenarios", [])

                    if src in self.channels:
                        src_q = self.channels[src]
                        for _ in range(8):
                            if src_q.empty():
                                break
                            try:
                                pkt = src_q.get_nowait()
                            except Exception:
                                break
                            if pkt is None:
                                break

                            # Speculative simulation across all candidate scenarios
                            best_sc = None
                            best_score = -float('inf')

                            for sc in scenarios:
                                exp_u = float(sc.get("expected_utility", 0.8))
                                cost = float(sc.get("cost_penalty", 0.1))
                                net_score = exp_u - cost
                                if net_score > best_score:
                                    best_score = net_score
                                    best_sc = sc

                            target_q_name = best_sc["target_queue"] if (best_sc and best_sc["target_queue"] in self.channels) else fb
                            if target_q_name and target_q_name in self.channels:
                                dst_q = self.channels[target_q_name]
                                out_pkt = dict(pkt) if isinstance(pkt, dict) else {"payload": pkt}
                                out_pkt["what_if_winner"] = best_sc["name"] if best_sc else "Fallback"
                                out_pkt["utility_score"] = round(best_score, 3)
                                out_pkt["hypothesis"] = best_sc["hypothesis"] if best_sc else "None"
                                dst_q.put_nowait(out_pkt)
                                packets_this_step += 1
                                self.metrics["what_if_resolutions"] = self.metrics.get("what_if_resolutions", 0) + 1

        # Update telemetry
        dt = max(1e-5, time.perf_counter() - t0)
        self.metrics["packets_processed"] += packets_this_step
        self.metrics["average_throughput_pps"] = round(packets_this_step / dt, 1)

        for q_name, q in self.channels.items():
            self.metrics["queue_depths"][q_name] = q.qsize()

        return dict(self.metrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.comp.get("module_name", "Unknown"),
            "bytecode_len": len(self.bytecode),
            "stages_count": len(self.partitions),
            "channels": list(self.channels.keys()),
            "metrics": self.metrics
        }
