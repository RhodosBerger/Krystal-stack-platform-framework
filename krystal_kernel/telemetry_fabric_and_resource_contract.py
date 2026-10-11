#!/usr/bin/env python3
"""
Telemetry Fabric, Resource Contract & Communication Integrity Plane for Krystal-Stack.

Implements the architectural paradigm:
"Najprv pozorovať, potom vysvetliť, až potom zasiahnuť." (Observe -> Explain -> Intervene)

Components:
1. Canonical Telemetry Data Contract (tamper-evident SHA-256 payload).
2. Resource Contract with Desired State vs Observed State drift detection.
3. Diagnostic Hypothesis Engine with Bounded Autonomous Intervention Mode.
4. Communication Integrity Plane (strictly declared service & cloud flows).
5. Physical Mathematics as a real-time hardware telemetry sensor.
"""

import os
import sys
import time
import uuid
import math
import hashlib
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.resource_integrity_engine import (
    HealthVector,
    CompetingHypothesesEngine,
    Hypothesis,
    ResourceDegradationEvent,
    ForensicAuditLedger,
    GLOBAL_RESOURCE_INTEGRITY_GOVERNOR
)


# ── 1. CANONICAL TELEMETRY DATA CONTRACT ─────────────────────────────────────

@dataclass
class CanonicalTelemetryPacket:
    """
    Standardized, immutable telemetry contract across all Krystal Linux nodes.
    Unifies physical mathematics, cgroups v2, queue state, and health vector.
    """
    packet_id: str
    timestamp: float
    correlation_id: str
    source_node: str
    profile_active: str  # GOV_DETERMINISTIC_RT, EDGE_FLEET_HEADLESS, STUDIO_WORKSTATION_HYBRID
    
    # Resource metrics
    cpu_usage_pct: float
    cgroups_nr_throttled: int
    cgroups_throttled_time_usec: int
    memory_current_bytes: int
    memory_limit_bytes: int
    queue_depth: int
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Physical mathematics telemetry
    physical_temperature_celsius: float
    thermal_partition_q_ratio: float  # Loewen-Shaw partition ratio
    effective_resistance_ohms: float  # Ohm's law R_eff >= 0.1
    effective_current_amps: float     # I = V / R_eff
    
    # Multi-dimensional health
    health_vector: Dict[str, float]
    packet_sha256: str = ""

    def calculate_hash(self) -> str:
        data = {
            "packet_id": self.packet_id,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "source_node": self.source_node,
            "profile_active": self.profile_active,
            "cpu_usage_pct": round(self.cpu_usage_pct, 2),
            "nr_throttled": self.cgroups_nr_throttled,
            "queue_depth": self.queue_depth,
            "temp_c": round(self.physical_temperature_celsius, 2),
            "i_amps": round(self.effective_current_amps, 4),
            "health": self.health_vector
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "source_node": self.source_node,
            "profile_active": self.profile_active,
            "metrics": {
                "cpu_usage_pct": self.cpu_usage_pct,
                "cgroups_nr_throttled": self.cgroups_nr_throttled,
                "cgroups_throttled_time_usec": self.cgroups_throttled_time_usec,
                "memory_current_bytes": self.memory_current_bytes,
                "memory_limit_bytes": self.memory_limit_bytes,
                "queue_depth": self.queue_depth,
                "p95_latency_ms": self.p95_latency_ms,
                "p99_latency_ms": self.p99_latency_ms,
                "physical_temperature_celsius": self.physical_temperature_celsius,
                "thermal_partition_q_ratio": self.thermal_partition_q_ratio,
                "effective_resistance_ohms": self.effective_resistance_ohms,
                "effective_current_amps": self.effective_current_amps
            },
            "health_vector": self.health_vector,
            "packet_sha256": self.packet_sha256 or self.calculate_hash()
        }


# ── 2. RESOURCE CONTRACT: DESIRED VS OBSERVED STATE ──────────────────────────

@dataclass
class DesiredResourceState:
    """Contractual expectation of hardware and operating system resources."""
    cpu_cores_pinned: List[int]
    cgroups_cpu_max_quota_usec: int
    cgroups_cpu_period_usec: int
    memory_max_bytes: int
    memory_high_bytes: int
    max_queue_depth: int
    p95_latency_budget_ms: float
    p99_latency_budget_ms: float
    thermal_ceiling_celsius: float
    max_current_amps: float


@dataclass
class ObservedResourceState:
    """Empirical measurements gathered by the Telemetry Fabric."""
    cpu_cores_active: List[int]
    cpu_usage_pct: float
    cgroups_nr_throttled: int
    cgroups_throttled_time_usec: int
    memory_current_bytes: int
    queue_depth: int
    p95_latency_ms: float
    p99_latency_ms: float
    physical_temperature_celsius: float
    effective_current_amps: float


@dataclass
class ResourceContractDrift:
    has_drift: bool
    drift_score: float  # 0.0 = aligned, 1.0 = critical drift
    violations: List[str]
    correlation_id: str
    timestamp: float


class ResourceContract:
    """
    Manages the Desired vs Observed State contract.
    Detects resource allocation mismatch, unapproved throttling, and thermal drift.
    """

    def __init__(self, desired: DesiredResourceState):
        self.desired = desired

    def evaluate_drift(self, observed: ObservedResourceState, correlation_id: str) -> ResourceContractDrift:
        violations: List[str] = []
        drift_components: List[float] = []

        # 1. Throttling violation (Anti-silent throttling rule)
        if observed.cgroups_nr_throttled > 0 or observed.cgroups_throttled_time_usec > 5000:
            violations.append(
                f"CGROUPS_THROTTLE_DRIFT: {observed.cgroups_nr_throttled} periods throttled "
                f"({observed.cgroups_throttled_time_usec} usec)"
            )
            drift_components.append(0.85)
        else:
            drift_components.append(0.0)

        # 2. Memory threshold drift
        if observed.memory_current_bytes > self.desired.memory_high_bytes:
            pct = (observed.memory_current_bytes / self.desired.memory_max_bytes) * 100.0
            violations.append(f"MEMORY_THRESHOLD_DRIFT: Memory at {pct:.1f}% exceeds memory.high")
            drift_components.append(min(1.0, observed.memory_current_bytes / self.desired.memory_max_bytes))
        else:
            drift_components.append(0.0)

        # 3. Queue SLA drift
        if observed.queue_depth > self.desired.max_queue_depth:
            violations.append(f"QUEUE_SLA_DRIFT: Queue depth {observed.queue_depth} exceeds max {self.desired.max_queue_depth}")
            drift_components.append(min(1.0, observed.queue_depth / (self.desired.max_queue_depth * 2.0)))
        else:
            drift_components.append(0.0)

        # 4. Latency SLA drift
        if observed.p95_latency_ms > self.desired.p95_latency_budget_ms:
            violations.append(
                f"LATENCY_P95_DRIFT: p95 {observed.p95_latency_ms:.1f}ms exceeds budget {self.desired.p95_latency_budget_ms}ms"
            )
            drift_components.append(0.60)
        else:
            drift_components.append(0.0)

        # 5. Physical Thermal Drift
        if observed.physical_temperature_celsius > self.desired.thermal_ceiling_celsius:
            violations.append(
                f"THERMAL_CEILING_DRIFT: Temperature {observed.physical_temperature_celsius:.1f}C "
                f"exceeds ceiling {self.desired.thermal_ceiling_celsius:.1f}C"
            )
            drift_components.append(0.90)
        else:
            drift_components.append(0.0)

        # 6. Physical Current / Ohm Drift
        if observed.effective_current_amps > self.desired.max_current_amps:
            violations.append(
                f"PHYSICAL_CURRENT_DRIFT: Current {observed.effective_current_amps:.2f}A exceeds limit {self.desired.max_current_amps:.2f}A"
            )
            drift_components.append(0.95)
        else:
            drift_components.append(0.0)

        total_drift = max(drift_components) if drift_components else 0.0

        return ResourceContractDrift(
            has_drift=len(violations) > 0,
            drift_score=round(total_drift, 3),
            violations=violations,
            correlation_id=correlation_id,
            timestamp=time.time()
        )


# ── 3. DIAGNOSTIC HYPOTHESIS ENGINE & BOUNDED INTERVENTION GOVERNOR ──────────

class GovernorExecutionMode(str, Enum):
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"                          # Observe & Explain ONLY (Strict default)
    BOUNDED_AUTONOMOUS_INTERVENTION = "BOUNDED_AUTONOMOUS_INTERVENTION"  # Intervene within verified limits


@dataclass
class InterventionProposal:
    correlation_id: str
    winning_hypothesis: Hypothesis
    recommended_action: str  # Z_RETRACT_4MM_M07, CGROUP_QUOTA_RESET, QUEUE_SHED_PRIORITY_LOW, THROTTLE_VOLTAGE
    is_executed: bool
    execution_reason: str
    timestamp: float


class DiagnosticGovernor:
    """
    Enforces the core rule:
    "Najprv pozorovať, potom vysvetliť, až potom zasiahnuť."
    In DIAGNOSTIC_ONLY mode, interventions are strictly recorded as proposals.
    In BOUNDED_AUTONOMOUS_INTERVENTION mode, only confirmed, non-falsified hypotheses
    trigger deterministic safe actions.
    """

    def __init__(self, mode: GovernorExecutionMode = GovernorExecutionMode.DIAGNOSTIC_ONLY):
        self.mode = mode
        self.mch_engine = CompetingHypothesesEngine()
        self.proposals: List[InterventionProposal] = []
        self.executed_interventions: List[InterventionProposal] = []

    def set_execution_mode(self, new_mode: GovernorExecutionMode):
        self.mode = new_mode

    def process_anomaly(
        self,
        telemetry: Dict[str, Any],
        correlation_id: str
    ) -> InterventionProposal:
        # Phase 1 & 2: Observe and Explain (MCH Engine)
        winner, all_hypotheses = self.mch_engine.evaluate_telemetry(telemetry)

        # Determine mapped mitigation action
        action_map = {
            "H1_NETWORK_TIMEOUT": "REROUTE_LOCAL_IPC_FALLBACK",
            "H2_CGROUPS_CPU_THROTTLING": "CGROUP_QUOTA_RESET_PREEMPT_RT",
            "H3_MEMORY_PRESSURE_GC_STALL": "COMPACT_STATE_CACHE",
            "H4_ADVERSARIAL_QUEUE_FLOOD": "QUEUE_SHED_PRIORITY_LOW"
        }
        recommended_action = action_map.get(winner.code, "Z_RETRACT_4MM_M07")

        # Check physical thermal/voltage overrides
        temp = float(telemetry.get("physical_temperature_celsius", 25.0))
        current = float(telemetry.get("effective_current_amps", 0.0))
        if temp > 85.0 or current > 15.0:
            recommended_action = "Z_RETRACT_4MM_M07"

        # Phase 3: Intervene ONLY if permitted by mode and proven by evidence
        can_execute = (
            self.mode == GovernorExecutionMode.BOUNDED_AUTONOMOUS_INTERVENTION and
            not winner.is_falsified and
            winner.confidence_score >= 0.80
        )

        reason = (
            f"Autonomous action executed: Hypothesis {winner.code} confirmed with score {winner.confidence_score}"
            if can_execute else
            f"Observation & Explanation recorded. Intervention held by mode {self.mode.value}."
        )

        proposal = InterventionProposal(
            correlation_id=correlation_id,
            winning_hypothesis=winner,
            recommended_action=recommended_action,
            is_executed=can_execute,
            execution_reason=reason,
            timestamp=time.time()
        )

        self.proposals.append(proposal)
        if can_execute:
            self.executed_interventions.append(proposal)

        return proposal


# ── 4. COMMUNICATION INTEGRITY PLANE ─────────────────────────────────────────

@dataclass(frozen=True)
class CommunicationFlow:
    source_service: str
    destination_service: str
    protocol: str           # mTLS_v1_3, UNIX_DOMAIN_SOCKET, HTTPS_REST, SHM_IPC
    port_or_socket: str
    direction: str          # INGRESS, EGRESS, BIDIRECTIONAL
    max_rate_requests_sec: int


@dataclass
class FlowValidationResult:
    is_authorized: bool
    flow: CommunicationFlow
    reason: str
    correlation_id: str
    timestamp: float


class CommunicationIntegrityPlane:
    """
    Enforces declared communication topology between services and the cloud.
    Any undeclared, rogue, or out-of-spec packet is immediately rejected and logged.
    """

    def __init__(self):
        self._declared_flows: Set[CommunicationFlow] = set()
        self.violations: List[FlowValidationResult] = []
        self._seed_default_declared_flows()

    def _seed_default_declared_flows(self):
        """Standard declared communication matrix for Krystal-Stack."""
        self._declared_flows.add(CommunicationFlow(
            source_service="API_DISPATCHER",
            destination_service="WORKER_CORE",
            protocol="UNIX_DOMAIN_SOCKET",
            port_or_socket="/run/krystal/worker.sock",
            direction="BIDIRECTIONAL",
            max_rate_requests_sec=5000
        ))
        self._declared_flows.add(CommunicationFlow(
            source_service="NODE_ORCHESTRATOR",
            destination_service="API_DISPATCHER",
            protocol="mTLS_v1_3",
            port_or_socket="8090",
            direction="INGRESS",
            max_rate_requests_sec=1000
        ))
        self._declared_flows.add(CommunicationFlow(
            source_service="WORKER_CORE",
            destination_service="FLEET_CLOUD_TELEMETRY",
            protocol="HTTPS_REST",
            port_or_socket="443",
            direction="EGRESS",
            max_rate_requests_sec=100
        ))
        self._declared_flows.add(CommunicationFlow(
            source_service="FAULT_INJECTION_LAB",
            destination_service="WORKER_CORE",
            protocol="SHM_IPC",
            port_or_socket="/dev/shm/krystal_fault_ctrl",
            direction="BIDIRECTIONAL",
            max_rate_requests_sec=50
        ))

    def register_declared_flow(self, flow: CommunicationFlow):
        self._declared_flows.add(flow)

    def validate_packet_flow(
        self,
        source: str,
        destination: str,
        protocol: str,
        port_or_socket: str,
        direction: str,
        rate_req_sec: int,
        correlation_id: str
    ) -> FlowValidationResult:
        candidate = CommunicationFlow(
            source_service=source,
            destination_service=destination,
            protocol=protocol,
            port_or_socket=port_or_socket,
            direction=direction,
            max_rate_requests_sec=rate_req_sec
        )

        # Check if flow matches declared matrix (ignoring rate for matching)
        matched = None
        for declared in self._declared_flows:
            if (declared.source_service == source and
                declared.destination_service == destination and
                declared.protocol == protocol and
                declared.port_or_socket == port_or_socket and
                declared.direction == direction):
                matched = declared
                break

        if not matched:
            res = FlowValidationResult(
                is_authorized=False,
                flow=candidate,
                reason=f"UNDECLARED_COMMUNICATION_FLOW: {source} -> {destination} via {protocol}:{port_or_socket}",
                correlation_id=correlation_id,
                timestamp=time.time()
            )
            self.violations.append(res)
            return res

        # Rate check
        if rate_req_sec > matched.max_rate_requests_sec:
            res = FlowValidationResult(
                is_authorized=False,
                flow=candidate,
                reason=f"RATE_LIMIT_EXCEEDED: {rate_req_sec} req/s exceeds limit {matched.max_rate_requests_sec}",
                correlation_id=correlation_id,
                timestamp=time.time()
            )
            self.violations.append(res)
            return res

        return FlowValidationResult(
            is_authorized=True,
            flow=candidate,
            reason="AUTHORIZED_DECLARED_FLOW",
            correlation_id=correlation_id,
            timestamp=time.time()
        )


# ── 5. UNIFIED TELEMETRY FABRIC ──────────────────────────────────────────────

class TelemetryFabric:
    """
    Coordinates empirical observations, resource contract drift detection,
    diagnostic hypothesis processing, and communication flow authorization.
    """

    def __init__(self, desired_state: Optional[DesiredResourceState] = None):
        self.desired = desired_state or DesiredResourceState(
            cpu_cores_pinned=[2, 3],
            cgroups_cpu_max_quota_usec=200000,
            cgroups_cpu_period_usec=100000,
            memory_max_bytes=1024 * 1024 * 1024,      # 1024 MB
            memory_high_bytes=800 * 1024 * 1024,      # 800 MB
            max_queue_depth=200,
            p95_latency_budget_ms=25.0,
            p99_latency_budget_ms=50.0,
            thermal_ceiling_celsius=75.0,
            max_current_amps=12.0
        )
        self.resource_contract = ResourceContract(self.desired)
        self.governor = DiagnosticGovernor(mode=GovernorExecutionMode.DIAGNOSTIC_ONLY)
        self.comm_plane = CommunicationIntegrityPlane()
        self.packets_history: List[CanonicalTelemetryPacket] = []

    def ingest_observation(
        self,
        packet: CanonicalTelemetryPacket,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        corr = correlation_id or packet.correlation_id or f"CORR-{uuid.uuid4().hex[:12].upper()}"
        self.packets_history.append(packet)
        if len(self.packets_history) > 1000:
            self.packets_history.pop(0)

        # 1. Evaluate Resource Contract Drift
        observed = ObservedResourceState(
            cpu_cores_active=self.desired.cpu_cores_pinned,
            cpu_usage_pct=packet.cpu_usage_pct,
            cgroups_nr_throttled=packet.cgroups_nr_throttled,
            cgroups_throttled_time_usec=packet.cgroups_throttled_time_usec,
            memory_current_bytes=packet.memory_current_bytes,
            queue_depth=packet.queue_depth,
            p95_latency_ms=packet.p95_latency_ms,
            p99_latency_ms=packet.p99_latency_ms,
            physical_temperature_celsius=packet.physical_temperature_celsius,
            effective_current_amps=packet.effective_current_amps
        )
        drift = self.resource_contract.evaluate_drift(observed, correlation_id=corr)

        # 2. If drift or anomaly detected, feed to Diagnostic Governor (Observe -> Explain)
        intervention_proposal = None
        if drift.has_drift or packet.health_vector.get("degradation_score", 0.0) > 0.2:
            telemetry_snapshot = {
                "queue_depth": packet.queue_depth,
                "nr_throttled_delta": packet.cgroups_nr_throttled,
                "cpu_usage_pct": packet.cpu_usage_pct,
                "memory_usage_pct": (packet.memory_current_bytes / self.desired.memory_max_bytes) * 100.0,
                "loopback_rtt_ms": 0.1,
                "tcp_retransmits": 0,
                "intent_confidence": 0.95,
                "physical_temperature_celsius": packet.physical_temperature_celsius,
                "effective_current_amps": packet.effective_current_amps
            }
            intervention_proposal = self.governor.process_anomaly(telemetry_snapshot, correlation_id=corr)

        return {
            "status": "OBSERVATION_INGESTED",
            "packet_id": packet.packet_id,
            "correlation_id": corr,
            "has_drift": drift.has_drift,
            "drift_score": drift.drift_score,
            "violations": drift.violations,
            "governor_mode": self.governor.mode.value,
            "intervention": intervention_proposal.__dict__ if intervention_proposal else None
        }


GLOBAL_TELEMETRY_FABRIC = TelemetryFabric()
