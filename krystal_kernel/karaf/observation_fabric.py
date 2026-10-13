#!/usr/bin/env python3
"""
==============================================================================
KARAF: HARDWARE OBSERVATION FABRIC
==============================================================================
Module: krystal_kernel/karaf/observation_fabric.py
Description:
    Observation Plane transforming raw Linux / system telemetry into
    explicit, strictly separated evidence:
      - OBS_* : Unadulterated, measured physical and kernel observations
      - DER_* : Mathematical derived metrics with explicit formulas

    Rules:
      1. Never mix OBS_* and DER_* metrics.
      2. No fixed hardware assumptions (detects topology at runtime).
      3. Enforces VITAL_MAX_HP == 6.
==============================================================================
"""

import os
import sys
import time
import math
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.types import (
    HardwareDomain,
    ObservationMetric,
    DerivedMetric,
    ObservationFabricSnapshot
)
from krystal_kernel.linux_hw_monitor_utility import LinuxHwMonitorUtility, HardwareAllocationSnapshot


class HardwareObservationFabric:
    """
    Observation plane for KARAF.
    Dissects system hardware metrics into categorized, timestamped evidence.
    """

    def __init__(self, monitor: Optional[LinuxHwMonitorUtility] = None):
        assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"
        self.vital_max_hp = VITAL_MAX_HP
        self.monitor = monitor or LinuxHwMonitorUtility()
        self._mock_overrides: Dict[str, Any] = {}

    def set_mock_override(self, metric_name: str, value: Any):
        """Allows deterministic testing by mocking specific OBS_* metrics."""
        self._mock_overrides[metric_name] = value

    def clear_mock_overrides(self):
        self._mock_overrides.clear()

    def sample_fabric(self) -> ObservationFabricSnapshot:
        """
        Samples the host hardware and produces an ObservationFabricSnapshot
        strictly partitioning OBS_* and DER_* metrics.
        """
        now_ns = time.time_ns()
        raw_snap: HardwareAllocationSnapshot = self.monitor.sample_snapshot()

        observations: Dict[str, ObservationMetric] = {}
        derived: Dict[str, DerivedMetric] = {}

        # ----------------------------------------------------------------------
        # 1. RAW OBSERVATIONS (OBS_*)
        # ----------------------------------------------------------------------

        # CPU_SCHEDULER
        cpu = raw_snap.cpu
        c_util = self._mock_overrides.get("OBS_CPU_UTILIZATION", float(cpu.cpu_utilization_pct))
        c_freq = self._mock_overrides.get("OBS_CPU_FREQ", float(cpu.average_frequency_mhz))
        c_ctx = self._mock_overrides.get("OBS_CONTEXT_SWITCH_RATE", float(cpu.context_switches_sec))
        c_cores = max(1, int(cpu.total_logical_cores))

        observations["OBS_CPU_UTILIZATION"] = ObservationMetric(
            name="OBS_CPU_UTILIZATION",
            domain=HardwareDomain.CPU_SCHEDULER,
            value=c_util,
            unit="percent",
            timestamp_ns=now_ns,
            provenance="proc:stat"
        )
        observations["OBS_CPU_FREQ"] = ObservationMetric(
            name="OBS_CPU_FREQ",
            domain=HardwareDomain.CPU_SCHEDULER,
            value=c_freq,
            unit="MHz",
            timestamp_ns=now_ns,
            provenance="sys:cpu_freq"
        )
        observations["OBS_CONTEXT_SWITCH_RATE"] = ObservationMetric(
            name="OBS_CONTEXT_SWITCH_RATE",
            domain=HardwareDomain.CPU_SCHEDULER,
            value=c_ctx,
            unit="cs/sec",
            timestamp_ns=now_ns,
            provenance="proc:stat:ctxt"
        )

        # MEMORY_UMA
        mem = raw_snap.memory
        r_total = max(1.0, float(mem.ram_total_mb))
        r_used = self._mock_overrides.get("OBS_RAM_USED", float(mem.ram_used_mb))
        sw_act = self._mock_overrides.get("OBS_SWAP_ACTIVITY", float(mem.swap_used_mb))
        uma_alloc = self._mock_overrides.get("OBS_GPU_UMA_ALLOC", float(mem.uma_shared_vram_mb))

        observations["OBS_RAM_USED"] = ObservationMetric(
            name="OBS_RAM_USED",
            domain=HardwareDomain.MEMORY_UMA,
            value=r_used,
            unit="MB",
            timestamp_ns=now_ns,
            provenance="proc:meminfo:MemTotal_MemAvailable",
            metadata={"ram_total_mb": r_total}
        )
        observations["OBS_SWAP_ACTIVITY"] = ObservationMetric(
            name="OBS_SWAP_ACTIVITY",
            domain=HardwareDomain.MEMORY_UMA,
            value=sw_act,
            unit="MB",
            timestamp_ns=now_ns,
            provenance="proc:meminfo:SwapTotal_SwapFree"
        )
        observations["OBS_GPU_UMA_ALLOC"] = ObservationMetric(
            name="OBS_GPU_UMA_ALLOC",
            domain=HardwareDomain.MEMORY_UMA,
            value=uma_alloc,
            unit="MB",
            timestamp_ns=now_ns,
            provenance="sys:kernel:drm:ttm:uma"
        )

        # GPU_ACCELERATOR
        gpu = raw_snap.gpu
        g_load = self._mock_overrides.get("OBS_GPU_LOAD", float(gpu.gpu_utilization_pct))
        observations["OBS_GPU_LOAD"] = ObservationMetric(
            name="OBS_GPU_LOAD",
            domain=HardwareDomain.GPU_ACCELERATOR,
            value=g_load,
            unit="percent",
            timestamp_ns=now_ns,
            provenance="drm:i915_perf"
        )

        # NPU_CASCADE
        npu = raw_snap.npu
        n_mem = self._mock_overrides.get("OBS_CASCADE_NODE_MEMORY", float(npu.total_model_vram_mb))
        observations["OBS_CASCADE_NODE_MEMORY"] = ObservationMetric(
            name="OBS_CASCADE_NODE_MEMORY",
            domain=HardwareDomain.NPU_CASCADE,
            value=n_mem,
            unit="MB",
            timestamp_ns=now_ns,
            provenance="openvino:runtime_registry"
        )

        # POWER_THERMAL
        thermal = raw_snap.power_thermal
        pkg_pwr = self._mock_overrides.get("OBS_PACKAGE_POWER", float(thermal.package_power_watts))
        core_tmp = self._mock_overrides.get("OBS_CORE_TEMP", float(thermal.package_temperature_c))
        tj_max = float(thermal.tj_max_c)

        observations["OBS_PACKAGE_POWER"] = ObservationMetric(
            name="OBS_PACKAGE_POWER",
            domain=HardwareDomain.POWER_THERMAL,
            value=pkg_pwr,
            unit="Watts",
            timestamp_ns=now_ns,
            provenance="sysfs:powercap:rapl"
        )
        observations["OBS_CORE_TEMP"] = ObservationMetric(
            name="OBS_CORE_TEMP",
            domain=HardwareDomain.POWER_THERMAL,
            value=core_tmp,
            unit="degC",
            timestamp_ns=now_ns,
            provenance="sysfs:thermal_zone"
        )

        # STORAGE_IO
        storage = raw_snap.storage
        st_lat = self._mock_overrides.get("OBS_STORAGE_LATENCY", float(storage.direct_storage_latency_us) / 1000.0)
        observations["OBS_STORAGE_LATENCY"] = ObservationMetric(
            name="OBS_STORAGE_LATENCY",
            domain=HardwareDomain.STORAGE_IO,
            value=st_lat,
            unit="ms",
            timestamp_ns=now_ns,
            provenance="sysfs:block:stat"
        )

        # NETWORK_SOCKETS
        net = raw_snap.network
        net_rx = self._mock_overrides.get("OBS_NETWORK_RX", float(net.rx_throughput_kb_s))
        observations["OBS_NETWORK_RX"] = ObservationMetric(
            name="OBS_NETWORK_RX",
            domain=HardwareDomain.NETWORK_SOCKETS,
            value=net_rx,
            unit="KB/s",
            timestamp_ns=now_ns,
            provenance="proc:net:dev"
        )

        # EPISTEMIC_INVARIANTS
        observations["OBS_VITAL_MAX_HP"] = ObservationMetric(
            name="OBS_VITAL_MAX_HP",
            domain=HardwareDomain.EPISTEMIC_INVARIANTS,
            value=VITAL_MAX_HP,
            unit="invariable_scalar",
            timestamp_ns=now_ns,
            provenance="kernel:scene_constructor_presets:VITAL_MAX_HP"
        )

        # ----------------------------------------------------------------------
        # 2. DERIVED METRICS (DER_*)
        # ----------------------------------------------------------------------

        # DER_CPU_HEADROOM = max(0, 100.0 - OBS_CPU_UTILIZATION)
        der_cpu_headroom = max(0.0, min(100.0, 100.0 - c_util))
        derived["DER_CPU_HEADROOM"] = DerivedMetric(
            name="DER_CPU_HEADROOM",
            domain=HardwareDomain.CPU_SCHEDULER,
            value=round(der_cpu_headroom, 2),
            formula="max(0.0, 100.0 - OBS_CPU_UTILIZATION)",
            inputs_used=["OBS_CPU_UTILIZATION"],
            timestamp_ns=now_ns,
            confidence=1.0
        )

        # DER_MEMORY_PRESSURE = OBS_RAM_USED / ram_total_mb
        der_mem_pressure = min(1.0, max(0.0, r_used / r_total))
        derived["DER_MEMORY_PRESSURE"] = DerivedMetric(
            name="DER_MEMORY_PRESSURE",
            domain=HardwareDomain.MEMORY_UMA,
            value=round(der_mem_pressure, 4),
            formula="OBS_RAM_USED / ram_total_mb",
            inputs_used=["OBS_RAM_USED"],
            timestamp_ns=now_ns,
            confidence=1.0
        )

        # DER_UMA_PRESSURE = (OBS_RAM_USED + OBS_GPU_UMA_ALLOC) / ram_total_mb
        der_uma_pressure = min(1.0, max(0.0, (r_used + uma_alloc) / r_total))
        derived["DER_UMA_PRESSURE"] = DerivedMetric(
            name="DER_UMA_PRESSURE",
            domain=HardwareDomain.MEMORY_UMA,
            value=round(der_uma_pressure, 4),
            formula="(OBS_RAM_USED + OBS_GPU_UMA_ALLOC) / ram_total_mb",
            inputs_used=["OBS_RAM_USED", "OBS_GPU_UMA_ALLOC"],
            timestamp_ns=now_ns,
            confidence=0.98
        )

        # DER_THERMAL_HEADROOM = max(0.0, tj_max - OBS_CORE_TEMP)
        der_therm_headroom = max(0.0, tj_max - core_tmp)
        derived["DER_THERMAL_HEADROOM"] = DerivedMetric(
            name="DER_THERMAL_HEADROOM",
            domain=HardwareDomain.POWER_THERMAL,
            value=round(der_therm_headroom, 2),
            formula="max(0.0, tj_max - OBS_CORE_TEMP)",
            inputs_used=["OBS_CORE_TEMP"],
            timestamp_ns=now_ns,
            confidence=0.95
        )

        # DER_IO_PRESSURE = min(1.0, OBS_STORAGE_LATENCY / 50.0)
        der_io_pressure = min(1.0, max(0.0, st_lat / 50.0))
        derived["DER_IO_PRESSURE"] = DerivedMetric(
            name="DER_IO_PRESSURE",
            domain=HardwareDomain.STORAGE_IO,
            value=round(der_io_pressure, 4),
            formula="min(1.0, OBS_STORAGE_LATENCY / 50.0)",
            inputs_used=["OBS_STORAGE_LATENCY"],
            timestamp_ns=now_ns,
            confidence=0.90
        )

        # DER_WORKER_CONTENTION = min(1.0, OBS_CONTEXT_SWITCH_RATE / (1000.0 * logical_cores))
        der_worker_contention = min(1.0, max(0.0, c_ctx / (1000.0 * c_cores)))
        derived["DER_WORKER_CONTENTION"] = DerivedMetric(
            name="DER_WORKER_CONTENTION",
            domain=HardwareDomain.CPU_SCHEDULER,
            value=round(der_worker_contention, 4),
            formula="min(1.0, OBS_CONTEXT_SWITCH_RATE / (1000.0 * logical_cores))",
            inputs_used=["OBS_CONTEXT_SWITCH_RATE"],
            timestamp_ns=now_ns,
            confidence=0.92
        )

        hw_context = {
            "logical_cores": c_cores,
            "p_cores": int(cpu.p_cores_count),
            "e_cores": int(cpu.e_cores_count),
            "ram_total_mb": r_total,
            "tj_max_c": tj_max,
            "governor": str(cpu.active_governor),
            "vital_max_hp": VITAL_MAX_HP
        }

        return ObservationFabricSnapshot(
            snapshot_id=f"snap_{now_ns}",
            timestamp_ns=now_ns,
            observations=observations,
            derived=derived,
            hardware_context=hw_context,
            vital_max_hp=VITAL_MAX_HP
        )


# Global singleton observation plane
GLOBAL_OBSERVATION_FABRIC = HardwareObservationFabric()
