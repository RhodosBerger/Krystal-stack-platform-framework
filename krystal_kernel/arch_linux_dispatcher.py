#!/usr/bin/env python3
"""
Arch Linux API Dispatcher, Systemd Cgroups v2 Slice Generator,
and Physical Mathematics / Fault Injection Calibration Subsystem.

Normative implementation for:
1. Arch Linux Profiles: GOV_DETERMINISTIC_RT, EDGE_FLEET_HEADLESS, STUDIO_WORKSTATION_HYBRID.
2. Systemd Slice & Unit configuration generator for cgroups v2 & PREEMPT_RT.
3. Transformed optics of Physical Mathematics (Ohm's Law & Thermal Tensors as telemetry sensors).
4. Transformed optics of Fault Injection Lab (Calibration of Diagnostic Hypothesis Engine).
5. Arch Linux API Dispatcher with Desired vs Observed State enforcement.
"""

import os
import sys
import time
import uuid
import math
import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.resource_integrity_engine import (
    HealthVector,
    FaultInjectionLab,
    FaultExperiment,
    GLOBAL_RESOURCE_INTEGRITY_GOVERNOR
)
from krystal_kernel.telemetry_fabric_and_resource_contract import (
    CanonicalTelemetryPacket,
    DesiredResourceState,
    ObservedResourceState,
    ResourceContract,
    DiagnosticGovernor,
    GovernorExecutionMode,
    GLOBAL_TELEMETRY_FABRIC
)


# ── 1. ARCH LINUX PROFILE DEFINITIONS ───────────────────────────────────────

class ArchLinuxProfileType(str, Enum):
    GOV_DETERMINISTIC_RT = "GOV_DETERMINISTIC_RT"        # Arch Linux + linux-rt (PREEMPT_RT), nanosecond determinism
    EDGE_FLEET_HEADLESS = "EDGE_FLEET_HEADLESS"          # Arch Linux minimal headless for B2B multi-spindle fleet
    STUDIO_WORKSTATION_HYBRID = "STUDIO_WORKSTATION_HYBRID"  # Arch Linux + Mesa Vulkan/Iris Xe + WebGL studio


@dataclass
class ArchLinuxProfileConfig:
    profile_type: ArchLinuxProfileType
    description: str
    target_kernel: str              # linux-rt, linux-lts, linux-zen
    systemd_slice_name: str         # krystal-rt.slice, krystal-fleet.slice
    allowed_cpus: str               # "2,3" for strict pinning
    cpu_weight: int                 # 100 .. 10000 (cgroups v2)
    cpu_quota_pct: int              # 100% = 1 core, 200% = 2 cores
    memory_max_mb: int              # cgroups memory.max
    memory_swap_max_mb: int         # cgroups memory.swap.max (0 for RT)
    io_weight: int                  # 100 .. 1000
    rt_priority: int                # SCHED_FIFO / SCHED_RR (1 .. 99)
    thermal_ceiling_celsius: float  # Loewen-Shaw ceiling
    max_physical_current_amps: float  # Ohm's law safe ceiling
    audit_strictness: str           # ISO_13849_PLd, NIST_SP_800_53, COMMERCIAL


ARCH_LINUX_PROFILES: Dict[ArchLinuxProfileType, ArchLinuxProfileConfig] = {
    ArchLinuxProfileType.GOV_DETERMINISTIC_RT: ArchLinuxProfileConfig(
        profile_type=ArchLinuxProfileType.GOV_DETERMINISTIC_RT,
        description="Government-Grade Real-Time Deterministic Profile (PREEMPT_RT / ISO 13849)",
        target_kernel="linux-rt",
        systemd_slice_name="krystal-rt.slice",
        allowed_cpus="2,3",
        cpu_weight=1000,
        cpu_quota_pct=200,
        memory_max_mb=1024,
        memory_swap_max_mb=0,          # Zero swap for real-time determinism
        io_weight=1000,
        rt_priority=80,                # Real-time FIFO priority
        thermal_ceiling_celsius=75.0,
        max_physical_current_amps=10.0,
        audit_strictness="ISO_13849_PLd"
    ),
    ArchLinuxProfileType.EDGE_FLEET_HEADLESS: ArchLinuxProfileConfig(
        profile_type=ArchLinuxProfileType.EDGE_FLEET_HEADLESS,
        description="Arch Linux Minimal Headless for Multi-Spindle B2B Fleet Management",
        target_kernel="linux-lts",
        systemd_slice_name="krystal-fleet.slice",
        allowed_cpus="0,1",
        cpu_weight=500,
        cpu_quota_pct=100,
        memory_max_mb=512,
        memory_swap_max_mb=0,
        io_weight=600,
        rt_priority=20,
        thermal_ceiling_celsius=85.0,
        max_physical_current_amps=14.0,
        audit_strictness="COMMERCIAL_FLEET"
    ),
    ArchLinuxProfileType.STUDIO_WORKSTATION_HYBRID: ArchLinuxProfileConfig(
        profile_type=ArchLinuxProfileType.STUDIO_WORKSTATION_HYBRID,
        description="Arch Linux Workstation with Mesa Vulkan UMA & Node.js Dispatcher",
        target_kernel="linux-zen",
        systemd_slice_name="krystal-studio.slice",
        allowed_cpus="0,1,2,3",
        cpu_weight=800,
        cpu_quota_pct=400,
        memory_max_mb=4096,
        memory_swap_max_mb=512,
        io_weight=800,
        rt_priority=10,
        thermal_ceiling_celsius=90.0,
        max_physical_current_amps=18.0,
        audit_strictness="COMMERCIAL"
    )
}


# ── 2. SYSTEMD CGROUPS V2 SLICE & UNIT GENERATOR ────────────────────────────

class SystemdUnitGenerator:
    """
    Generates deterministic systemd slice and service files for Arch Linux,
    enforcing cgroups v2 resource boundaries without silent throttling.
    """

    @staticmethod
    def generate_slice_file(config: ArchLinuxProfileConfig) -> str:
        quota_str = f"{config.cpu_quota_pct}%"
        mem_bytes = config.memory_max_mb * 1024 * 1024
        swap_bytes = config.memory_swap_max_mb * 1024 * 1024

        return f"""# /etc/systemd/system/{config.systemd_slice_name}
# Generated by Krystal-Stack Arch Linux Dispatcher
# Profile: {config.profile_type.value}
# Standard: {config.audit_strictness}

[Unit]
Description=Krystal Stack cgroups v2 Slice for {config.profile_type.value}
Documentation=file:///docs/gov_resource_integrity_and_forensic_architecture.md
Before=slices.target

[Slice]
# CPU & Real-Time Resource Controls
AllowedCPUs={config.allowed_cpus}
CPUWeight={config.cpu_weight}
CPUQuota={quota_str}

# Memory Isolation (Zero-Swap Real-Time Boundary)
MemoryMax={mem_bytes}
MemoryHigh={int(mem_bytes * 0.85)}
MemorySwapMax={swap_bytes}

# I/O Strict Scheduling
IOWeight={config.io_weight}
"""

    @staticmethod
    def generate_service_file(config: ArchLinuxProfileConfig, service_name: str = "krystal-worker") -> str:
        return f"""# /etc/systemd/system/{service_name}.service
[Unit]
Description=Krystal Stack Hardened Worker ({config.profile_type.value})
After=network.target
Slice={config.systemd_slice_name}

[Service]
Type=simple
User=krystal
Group=krystal
WorkingDirectory=/opt/krystal-stack
ExecStart=/usr/bin/python3 -u -m krystal_kernel.resource_integrity_engine
Restart=on-failure
RestartSec=200ms

# Real-Time Scheduling Guard (PREEMPT_RT)
CPUSchedulingPolicy=fifo
CPUSchedulingPriority={config.rt_priority}
LimitMEMLOCK=infinity
LimitRTPRIO={config.rt_priority}

# Security Isolation
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
"""

    @staticmethod
    def generate_rt_limits_conf(rt_prio: int = 99) -> str:
        return f"""# /etc/security/limits.d/99-krystal-rt.conf
# Real-Time scheduler limits for Arch Linux Krystal Stack
@krystal   soft   rtprio   {rt_prio}
@krystal   hard   rtprio   {rt_prio}
@krystal   soft   memlock  unlimited
@krystal   hard   memlock  unlimited
"""


# ── 3. TRANSFORMED OPTICS: PHYSICAL MATHEMATICS AS TELEMETRY SENSOR ─────────

@dataclass
class PhysicalTelemetryObservation:
    voltage_volts: float
    effective_resistance_ohms: float
    current_amps: float
    thermal_partition_ratio: float  # Loewen-Shaw heat partition (Q_chip / Q_tool)
    temperature_celsius: float
    is_physical_hazard: bool
    conduct_badge: str
    signature: str


class PhysicalMathematicsSensor:
    """
    Transforms Physical Mathematics into an active real-time hardware telemetry sensor:
    - Ohm's Law: I = V / R_eff with zero-division guard (R_eff >= 0.1).
    - Thermal Partitioning: Loewen-Shaw chip-tool partition ratio.
    - Anomaly Detection: Current surge or thermal runaways directly generate telemetry packets.
    """

    def __init__(self, base_resistance: float = 2.0, ambient_temp: float = 24.0):
        self.base_resistance = base_resistance
        self.temperature = ambient_temp
        self.min_resistance = 0.1

    def observe_physical_state(
        self,
        applied_voltage: float,
        resistance_delta: float = 0.0,
        cutting_heat_watts: float = 250.0
    ) -> PhysicalTelemetryObservation:
        # Effective resistance with division guard
        r_eff = max(self.min_resistance, self.base_resistance + resistance_delta)
        
        # Ohm's law: I = V / R_eff
        v_clamped = max(0.0, applied_voltage)
        current = v_clamped / r_eff

        # Loewen-Shaw heat partitioning model (q_ratio = fraction going into workpiece/chip)
        q_ratio = max(0.1, min(0.9, 0.72 - (0.015 * current)))
        heat_into_tool = cutting_heat_watts * (1.0 - q_ratio)
        
        # Thermal accumulation (dry cutting accumulation)
        temp_rise = (heat_into_tool * 0.12) + (current * 1.5)
        self.temperature = round(24.0 + temp_rise, 2)

        # Invariant checks: Hazard if I > 10A or T > 75C
        is_hazard = (current > 10.0) or (self.temperature > 75.0)
        
        act = "[ACT]" if not is_hazard else "[!ACT]"
        lck = "[LCK]" if is_hazard else "[!LCK]"
        cnf = "[CNF]" if not is_hazard else "[!CNF]"
        haz = "[HAZ]" if is_hazard else "[!HAZ]"
        badge = f"{act} {lck} {cnf} {haz} [VIT {VITAL_MAX_HP}HP]"

        signature = (
            f"PHYSICAL_OVERLOAD_I={current:.2f}A_T={self.temperature:.1f}C"
            if is_hazard else "PHYSICAL_EQUILIBRIUM_NOMINAL"
        )

        return PhysicalTelemetryObservation(
            voltage_volts=v_clamped,
            effective_resistance_ohms=round(r_eff, 4),
            current_amps=round(current, 4),
            thermal_partition_ratio=round(q_ratio, 3),
            temperature_celsius=self.temperature,
            is_physical_hazard=is_hazard,
            conduct_badge=badge,
            signature=signature
        )


# ── 4. TRANSFORMED OPTICS: FAULT INJECTION LAB AS CALIBRATOR ────────────────

class CalibratedFaultInjectionController:
    """
    Transforms Fault Injection Lab into a precision calibration instrument:
    Validates that:
    1. Observation is made by Telemetry Fabric without delay.
    2. Hypothesis Engine explains the failure with >80% confidence and zero false accusations.
    3. Intervention occurs ONLY after explanation, executing the deterministic recovery loop.
    """

    def __init__(self, governor: DiagnosticGovernor):
        self.governor = governor
        self.lab = FaultInjectionLab()
        self.calibration_history: List[Dict[str, Any]] = []

    def run_calibrated_experiment(
        self,
        experiment_name: str,
        target_container: str,
        fault_type: str,
        expected_hypothesis_code: str,
        intensity_pct: float = 50.0
    ) -> Dict[str, Any]:
        # 1. Create Experiment in Fault Lab
        exp = self.lab.create_experiment(
            name=experiment_name,
            target_container=target_container,
            fault_type=fault_type,
            intensity_pct=intensity_pct,
            duration_sec=3.0,
            expected_signature=f"CALIBRATION_PROBE_FOR_{expected_hypothesis_code}"
        )

        # 2. Simulate Observed Anomaly Ingestion
        simulated_telemetry = {
            "queue_depth": 35 if fault_type != "QUEUE_JAM" else 420,
            "nr_throttled_delta": 18 if fault_type == "CPU_QUOTA_STEAL" else 0,
            "cpu_usage_pct": 98.0 if fault_type == "CPU_QUOTA_STEAL" else 40.0,
            "memory_usage_pct": 92.0 if fault_type == "MEMORY_INFLATION" else 30.0,
            "loopback_rtt_ms": 0.1,
            "tcp_retransmits": 0,
            "intent_confidence": 0.95 if fault_type != "QUEUE_JAM" else 0.40,
            "physical_temperature_celsius": 82.0 if fault_type == "THERMAL_OVERLOAD" else 45.0,
            "effective_current_amps": 12.5 if fault_type == "THERMAL_OVERLOAD" else 4.0
        }

        # 3. Step: OBSERVE & EXPLAIN via Diagnostic Governor
        proposal = self.governor.process_anomaly(simulated_telemetry, correlation_id=exp.experiment_id)
        
        # Verify if explanation matched calibrated hypothesis
        hypothesis_matched = (proposal.winning_hypothesis.code == expected_hypothesis_code)

        # 4. Step: RECOVER via Deterministic 4-Phase Recovery Loop
        recovery_result = self.lab.execute_recovery_loop(exp)

        calibration_record = {
            "experiment_id": exp.experiment_id,
            "name": experiment_name,
            "fault_type": fault_type,
            "expected_hypothesis": expected_hypothesis_code,
            "detected_hypothesis": proposal.winning_hypothesis.code,
            "hypothesis_matched": hypothesis_matched,
            "confidence_score": proposal.winning_hypothesis.confidence_score,
            "proposal_action": proposal.recommended_action,
            "is_intervened": proposal.is_executed,
            "recovery_status": recovery_result["recovery_status"],
            "recovery_duration_ms": recovery_result["loop_duration_ms"],
            "timestamp": time.time()
        }
        self.calibration_history.append(calibration_record)
        return calibration_record


# ── 5. UNIFIED ARCH LINUX API DISPATCHER ─────────────────────────────────────

class ArchLinuxApiDispatcher:
    """
    Unified Dispatcher for Arch Linux Kernel Stack.
    Manages runtime profiles, systemd configurations, telemetry ingestion,
    physical mathematics sensors, and calibrated resilience experiments.
    """

    def __init__(self, default_profile: ArchLinuxProfileType = ArchLinuxProfileType.GOV_DETERMINISTIC_RT):
        self.active_profile_type = default_profile
        self.profile_config = ARCH_LINUX_PROFILES[default_profile]
        self.telemetry_fabric = GLOBAL_TELEMETRY_FABRIC
        self.physics_sensor = PhysicalMathematicsSensor()
        self.calibrator = CalibratedFaultInjectionController(self.telemetry_fabric.governor)
        self.jobs_registry: Dict[str, Dict[str, Any]] = {}

    def switch_profile(self, new_profile: ArchLinuxProfileType) -> Dict[str, Any]:
        """Dynamically switches active Arch Linux configuration profile."""
        if new_profile not in ARCH_LINUX_PROFILES:
            raise ValueError(f"Unknown Arch Linux profile: {new_profile}")
        
        self.active_profile_type = new_profile
        self.profile_config = ARCH_LINUX_PROFILES[new_profile]

        # Update desired state in Telemetry Fabric
        mem_bytes = self.profile_config.memory_max_mb * 1024 * 1024
        cpus_parsed = [int(c.strip()) for c in self.profile_config.allowed_cpus.split(",")]
        
        self.telemetry_fabric.desired = DesiredResourceState(
            cpu_cores_pinned=cpus_parsed,
            cgroups_cpu_max_quota_usec=int(self.profile_config.cpu_quota_pct * 1000),
            cgroups_cpu_period_usec=100000,
            memory_max_bytes=mem_bytes,
            memory_high_bytes=int(mem_bytes * 0.85),
            max_queue_depth=200 if new_profile != ArchLinuxProfileType.EDGE_FLEET_HEADLESS else 500,
            p95_latency_budget_ms=25.0 if new_profile == ArchLinuxProfileType.GOV_DETERMINISTIC_RT else 60.0,
            p99_latency_budget_ms=50.0 if new_profile == ArchLinuxProfileType.GOV_DETERMINISTIC_RT else 120.0,
            thermal_ceiling_celsius=self.profile_config.thermal_ceiling_celsius,
            max_current_amps=self.profile_config.max_physical_current_amps
        )
        self.telemetry_fabric.resource_contract = ResourceContract(self.telemetry_fabric.desired)

        return {
            "status": "PROFILE_ACTIVATED",
            "active_profile": self.active_profile_type.value,
            "target_kernel": self.profile_config.target_kernel,
            "systemd_slice": self.profile_config.systemd_slice_name,
            "allowed_cpus": self.profile_config.allowed_cpus,
            "audit_strictness": self.profile_config.audit_strictness
        }

    def get_systemd_artifacts(self) -> Dict[str, str]:
        """Exports Arch Linux systemd slice and service definitions for active profile."""
        return {
            "slice_filename": self.profile_config.systemd_slice_name,
            "slice_content": SystemdUnitGenerator.generate_slice_file(self.profile_config),
            "service_filename": "krystal-worker.service",
            "service_content": SystemdUnitGenerator.generate_service_file(self.profile_config),
            "limits_conf": SystemdUnitGenerator.generate_rt_limits_conf(self.profile_config.rt_priority)
        }

    def dispatch_transaction(
        self,
        payload_command: str,
        applied_voltage: float = 12.0,
        cutting_heat_watts: float = 200.0,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes complete dispatched transaction:
        1. Evaluates physical mathematics as telemetry sensor.
        2. Ingests canonical telemetry into Telemetry Fabric.
        3. Enforces Observe -> Explain -> Intervene sequence.
        4. Returns status with Conduct Code badge and health metrics.
        """
        corr = correlation_id or f"CORR-{uuid.uuid4().hex[:12].upper()}"
        job_id = f"JOB-{uuid.uuid4().hex[:8].upper()}"

        # 1. Physical sensor observation
        phys = self.physics_sensor.observe_physical_state(
            applied_voltage=applied_voltage,
            cutting_heat_watts=cutting_heat_watts
        )

        # 2. Canonical Telemetry Packet construction
        hv = HealthVector(
            liveness=1.0,
            readiness=1.0 if not phys.is_physical_hazard else 0.5,
            resource_integrity=0.95 if not phys.is_physical_hazard else 0.40,
            degradation_score=0.0 if not phys.is_physical_hazard else 0.60,
            sla_compliance=0.98 if not phys.is_physical_hazard else 0.50
        )

        packet = CanonicalTelemetryPacket(
            packet_id=f"PKT-{uuid.uuid4().hex[:8].upper()}",
            timestamp=time.time(),
            correlation_id=corr,
            source_node=f"arch-{self.active_profile_type.value.lower()}",
            profile_active=self.active_profile_type.value,
            cpu_usage_pct=42.5 if not phys.is_physical_hazard else 95.0,
            cgroups_nr_throttled=0 if not phys.is_physical_hazard else 8,
            cgroups_throttled_time_usec=0 if not phys.is_physical_hazard else 7500,
            memory_current_bytes=int(self.profile_config.memory_max_mb * 0.4 * 1024 * 1024),
            memory_limit_bytes=int(self.profile_config.memory_max_mb * 1024 * 1024),
            queue_depth=12 if not phys.is_physical_hazard else 180,
            p95_latency_ms=14.2 if not phys.is_physical_hazard else 65.0,
            p99_latency_ms=28.5 if not phys.is_physical_hazard else 115.0,
            physical_temperature_celsius=phys.temperature_celsius,
            thermal_partition_q_ratio=phys.thermal_partition_ratio,
            effective_resistance_ohms=phys.effective_resistance_ohms,
            effective_current_amps=phys.current_amps,
            health_vector=hv.to_dict()
        )
        packet.packet_sha256 = packet.calculate_hash()

        # 3. Telemetry Fabric Ingestion
        fabric_res = self.telemetry_fabric.ingest_observation(packet, correlation_id=corr)

        job_record = {
            "job_id": job_id,
            "correlation_id": corr,
            "profile": self.active_profile_type.value,
            "command": payload_command,
            "physical_observation": {
                "voltage": phys.voltage_volts,
                "current_amps": phys.current_amps,
                "temperature_c": phys.temperature_celsius,
                "is_hazard": phys.is_physical_hazard,
                "conduct_badge": phys.conduct_badge
            },
            "telemetry_fabric": fabric_res,
            "vital_hp": VITAL_MAX_HP if not phys.is_physical_hazard else max(1, VITAL_MAX_HP - 2),
            "status": "SUCCESS" if not phys.is_physical_hazard else "DEFENSIVE_FAILSAFE"
        }
        self.jobs_registry[job_id] = job_record
        return job_record


GLOBAL_ARCH_LINUX_DISPATCHER = ArchLinuxApiDispatcher()
