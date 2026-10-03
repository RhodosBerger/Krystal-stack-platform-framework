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
            "engine_mode": "LOCK_FREE_RING" if use_fast_queue else "STANDARD_MUTEX"
        }
        self._init_channels()

    def _init_channels(self):
        for instr in self.bytecode:
            if instr.get("op") == "OP_ALLOC_QUEUE":
                q_name = instr["name"]
                cap = instr.get("capacity", 256)
                if self.use_fast_queue:
                    self.channels[q_name] = FastRingBuffer(cap)
                else:
                    self.channels[q_name] = queue.Queue(maxsize=cap)
                self.metrics["queue_depths"][q_name] = 0
        self.metrics["active_queues"] = len(self.channels)

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

            # Dispatch pipelines
            for instr in self.bytecode:
                if instr.get("op") == "OP_QUEUE_DISPATCH":
                    src = instr["from_queue"]
                    dst = instr["to_queue"]
                    action = instr.get("action", "PASS")

                    if src in self.channels and dst in self.channels:
                        src_q = self.channels[src]
                        dst_q = self.channels[dst]

                        # Process up to 8 packets per pipeline step
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
