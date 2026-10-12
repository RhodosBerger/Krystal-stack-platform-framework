#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: NETWORK TASK ARBITER & SCHEDULING DISPATCH MEDIATOR
==============================================================================
Module: krystal_kernel/network_task_arbiter.py
Description:
    Implements the architectural "Golden Mean" (Stred) between concrete
    HTTP/REST/RPC endpoints and conceptual/predictive network task scheduling:
      1. Dual-Path Dispatch:
         - Fast-Path Bypass (<0.1 ms) for Ring 0 safety & critical invariants.
         - Scheduled Frame-Budget Quantum for real-time inference and game state.
      2. Priority Bands (QoS Rings):
         - Band 0 (CRITICAL_REALTIME): Safety, thermal trip, vital invariant clamp.
         - Band 1 (HIGH_INFERENCE): PROTIX ML forward-pass (<0.4 ms).
         - Band 2 (STANDARD_GAME_STATE): Poslední Kmen ledger and turn updates.
         - Band 3 (BACKGROUND_BATCH): Asynchronous model evaluation & archiving.
      3. Physical-Aware Backpressure:
         - Incorporates Ohm's Law (I = V / R_eff <= 10A) and thermal dissipation
           (R_th = 0.72 °C/W with 100% cooling fan) into queue admission control.
      4. Frame Budget Partitioning (60 FPS / 16.6 ms cycle):
         - Allocates a deterministic 3.5 ms window per tick for network task
           execution to eliminate jitter, bufferbloat, and latency spikes.
      5. Universal Platform Invariant: VITAL_MAX_HP == 6 strictly enforced.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import time
import math
from enum import IntEnum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Tuple

VITAL_MAX_HP: int = 6


class NetworkTaskPriority(IntEnum):
    CRITICAL_REALTIME = 0      # Sub-millisecond safety, emergency fan, HP invariants
    HIGH_INFERENCE    = 1      # Real-time PROTIX ML predictions (<0.4 ms)
    STANDARD_GAME_STATE = 2    # Card/ledger turns, movement, tactical triggers
    BACKGROUND_BATCH  = 3      # Evaluation suites, model architecture queries, logging


class NetworkTaskStatus(str):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    REJECTED  = "REJECTED"
    TIMED_OUT = "TIMED_OUT"


@dataclass
class NetworkTaskDescriptor:
    """Formal network task unit bridging an external endpoint to scheduled execution."""
    task_id: str
    endpoint_path: str
    priority: NetworkTaskPriority
    payload: Dict[str, Any]
    max_latency_budget_ms: float
    created_at_epoch: float = field(default_factory=time.time)
    started_at_epoch: Optional[float] = None
    completed_at_epoch: Optional[float] = None
    status: str = NetworkTaskStatus.PENDING
    result_data: Optional[Dict[str, Any]] = None
    execution_time_us: float = 0.0
    vital_max_hp_rule: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["priority"] = int(self.priority)
        return d


@dataclass
class PhysicalNetworkEnvelope:
    """Hardware telemetry constraints governing network task admission."""
    temp_cpu_c: float = 52.1
    fan_duty_pct: float = 100.0
    thermal_resistance: float = 0.72  # °C/W at 100% fan duty
    voltage_v: float = 0.85
    effective_resistance_ohms: float = 2.0
    max_physical_current_a: float = 10.0
    queue_depth_max: int = 250

    @property
    def physical_current_a(self) -> float:
        # Ohm's Law: I = V / R_eff
        r = max(0.1, self.effective_resistance_ohms)
        return round(self.voltage_v / r, 4)

    @property
    def has_thermal_headroom(self) -> bool:
        return self.temp_cpu_c < 75.0 and self.physical_current_a <= self.max_physical_current_a


class NetworkTaskArbiter:
    """
    Architectural Mediator ("Stred") between Endpoints and Network Task Scheduling:
    Converts ad-hoc synchronous HTTP calls into physical-aware, budget-constrained
    network tasks while providing low-latency Fast-Path execution for real-time safety.
    """

    def __init__(self, frame_budget_ms: float = 3.5):
        self.frame_budget_ms = frame_budget_ms
        self.envelope = PhysicalNetworkEnvelope()
        self.queues: Dict[NetworkTaskPriority, List[NetworkTaskDescriptor]] = {
            NetworkTaskPriority.CRITICAL_REALTIME: [],
            NetworkTaskPriority.HIGH_INFERENCE: [],
            NetworkTaskPriority.STANDARD_GAME_STATE: [],
            NetworkTaskPriority.BACKGROUND_BATCH: []
        }
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.total_dispatched_tasks: int = 0
        self.total_fastpath_tasks: int = 0
        self.total_rejected_tasks: int = 0
        self.total_budget_exhaustions: int = 0

    def register_endpoint_handler(self, endpoint_path: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        """Registers execution callback for a known endpoint path."""
        self.handlers[endpoint_path] = handler

    def enqueue_or_fastpath(
        self,
        endpoint_path: str,
        payload: Dict[str, Any],
        priority: NetworkTaskPriority = NetworkTaskPriority.STANDARD_GAME_STATE,
        allow_fastpath: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        The central mediator entry point:
        - If priority is CRITICAL_REALTIME and allow_fastpath is True -> executes immediately (<0.1 ms).
        - Otherwise, enqueues into the priority budget ring for synchronized batch dispatch.
        """
        # Invariant Pre-check
        hp_candidate = payload.get("player_hp", payload.get("hp", 0))
        if isinstance(hp_candidate, (int, float)) and hp_candidate > VITAL_MAX_HP:
            payload["player_hp"] = VITAL_MAX_HP
            payload["hp"] = VITAL_MAX_HP

        task_id = f"task_{int(time.time()*1e6)}_{len(self.queues[priority])}"
        desc = NetworkTaskDescriptor(
            task_id=task_id,
            endpoint_path=endpoint_path,
            priority=priority,
            payload=payload,
            max_latency_budget_ms=16.6 if priority != NetworkTaskPriority.CRITICAL_REALTIME else 2.0
        )

        # 1. Fast-Path Bypass for Critical Safety or direct bypass
        if allow_fastpath and priority == NetworkTaskPriority.CRITICAL_REALTIME:
            return self._execute_task_direct(desc)

        # 2. Admission Control & Backpressure Check
        total_queued = sum(len(q) for q in self.queues.values())
        if total_queued >= self.envelope.queue_depth_max or not self.envelope.has_thermal_headroom:
            desc.status = NetworkTaskStatus.REJECTED
            self.total_rejected_tasks += 1
            return False, {
                "error": "NETWORK_ARBITER_CONGESTION_BACKPRESSURE",
                "queue_depth": total_queued,
                "current_temp_c": self.envelope.temp_cpu_c,
                "ohm_current_a": self.envelope.physical_current_a
            }

        # 3. Enqueue into Priority Band
        self.queues[priority].append(desc)
        return True, {
            "status": "QUEUED",
            "task_id": desc.task_id,
            "priority": int(priority),
            "estimated_wait_frames": len(self.queues[priority])
        }

    def _execute_task_direct(self, desc: NetworkTaskDescriptor) -> Tuple[bool, Dict[str, Any]]:
        """Direct zero-overhead execution for Fast-Path requests."""
        t0 = time.perf_counter()
        desc.started_at_epoch = time.time()
        desc.status = NetworkTaskStatus.RUNNING

        handler = self.handlers.get(desc.endpoint_path)
        if not handler:
            # Default echo execution
            res = {"status": "OK", "executed_default": True, "payload": desc.payload}
        else:
            try:
                res = handler(desc.payload)
            except Exception as e:
                res = {"status": "ERROR", "error": str(e)}

        desc.execution_time_us = round((time.perf_counter() - t0) * 1e6, 2)
        desc.completed_at_epoch = time.time()
        desc.status = NetworkTaskStatus.COMPLETED
        desc.result_data = res

        self.total_dispatched_tasks += 1
        self.total_fastpath_tasks += 1
        return True, res

    def dispatch_tick(self, budget_override_ms: Optional[float] = None) -> Dict[str, Any]:
        """
        Executes queued network tasks within the strict frame budget (e.g., 3.5 ms).
        Traverses priority rings (0 -> 1 -> 2 -> 3) until queue empty or budget exhausted.
        """
        t_start = time.perf_counter()
        max_budget = budget_override_ms or self.frame_budget_ms
        tasks_run_in_tick = 0
        completed_results = []

        for prio in [
            NetworkTaskPriority.CRITICAL_REALTIME,
            NetworkTaskPriority.HIGH_INFERENCE,
            NetworkTaskPriority.STANDARD_GAME_STATE,
            NetworkTaskPriority.BACKGROUND_BATCH
        ]:
            queue = self.queues[prio]
            while queue:
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                if elapsed_ms >= max_budget:
                    self.total_budget_exhaustions += 1
                    break

                desc = queue.pop(0)
                _, res = self._execute_task_direct(desc)
                tasks_run_in_tick += 1
                completed_results.append({
                    "task_id": desc.task_id,
                    "endpoint": desc.endpoint_path,
                    "priority": int(desc.priority),
                    "execution_time_us": desc.execution_time_us
                })

        total_elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 3)
        return {
            "tasks_executed": tasks_run_in_tick,
            "elapsed_ms": total_elapsed_ms,
            "budget_ms": max_budget,
            "budget_exhausted": total_elapsed_ms >= max_budget,
            "remaining_queued": sum(len(q) for q in self.queues.values()),
            "details": completed_results,
            "vital_max_hp_rule": VITAL_MAX_HP
        }

    def get_telemetry_status(self) -> Dict[str, Any]:
        """Returns runtime arbitration telemetry and queue state."""
        return {
            "arbiter_frame_budget_ms": self.frame_budget_ms,
            "total_dispatched": self.total_dispatched_tasks,
            "total_fastpath": self.total_fastpath_tasks,
            "total_rejected": self.total_rejected_tasks,
            "total_budget_exhaustions": self.total_budget_exhaustions,
            "current_queue_depth": sum(len(q) for q in self.queues.values()),
            "queue_breakdown": {
                "critical_realtime": len(self.queues[NetworkTaskPriority.CRITICAL_REALTIME]),
                "high_inference": len(self.queues[NetworkTaskPriority.HIGH_INFERENCE]),
                "standard_game": len(self.queues[NetworkTaskPriority.STANDARD_GAME_STATE]),
                "background_batch": len(self.queues[NetworkTaskPriority.BACKGROUND_BATCH])
            },
            "physical_envelope": {
                "temp_cpu_c": self.envelope.temp_cpu_c,
                "fan_duty_pct": self.envelope.fan_duty_pct,
                "thermal_resistance": self.envelope.thermal_resistance,
                "voltage_v": self.envelope.voltage_v,
                "effective_resistance_ohms": self.envelope.effective_resistance_ohms,
                "ohm_current_a": self.envelope.physical_current_a,
                "has_thermal_headroom": self.envelope.has_thermal_headroom
            },
            "vital_max_hp_rule": VITAL_MAX_HP
        }


# Global Singleton Arbiter Instance
GLOBAL_NETWORK_TASK_ARBITER = NetworkTaskArbiter()
