"""Dedicated regression suite for the 5 Double-Time mode bugfixes."""

from __future__ import annotations

import json

import pytest

from krystal_kernel.constants import ContractViolation
from krystal_kernel.double_time_coefficient_emulator import DoubleTimeCoefficientEmulator
from krystal_kernel.epistemic_cognition_compositor import MultiDataHybridCompositor


def test_double_time_compositor_clamping_and_uma_guard() -> None:
    comp = MultiDataHybridCompositor()

    # 1. Nominal 60 Hz (16.667 ms, k=4) -> verified lock
    dt60 = comp.run_double_time_compositor_cycle(frame_budget_ms=16.667, speculative_speedup_factor=4, uma_utilization=0.35)
    assert dt60["vital_max_hp"] == 6
    assert dt60["status"] == "DOUBLE_TIME_LOCK_VERIFIED"
    assert dt60["temporal_domains"]["tau_spec_lookahead_axis"]["lookahead_horizon_steps"] == 4

    # 2. 120 Hz Double-Time (8.333 ms, requested k=4) -> clamped to k=2 (4.90 ms <= 6.713 ms), zero overrun
    dt120 = comp.run_double_time_compositor_cycle(frame_budget_ms=8.333, speculative_speedup_factor=4, uma_utilization=0.35)
    spec120 = dt120["temporal_domains"]["tau_spec_lookahead_axis"]
    assert dt120["status"] == "SHADOW_PACED"
    assert spec120["requested_lookahead_steps"] == 4
    assert spec120["lookahead_horizon_steps"] == 2
    assert spec120["total_shadow_compute_ms"] <= spec120["available_phase_shadow_ms"]
    assert "t+1..t+2" in spec120["role_sk"]

    # 3. UMA >= 0.78 -> immediate suspension of tau_spec (Sovereign P3)
    dt_uma = comp.run_double_time_compositor_cycle(frame_budget_ms=16.667, speculative_speedup_factor=4, uma_utilization=0.84)
    assert dt_uma["status"] == "UMA_SHADOW_SUSPENDED_DELTA_S_ZERO"
    assert dt_uma["temporal_domains"]["tau_spec_lookahead_axis"]["lookahead_horizon_steps"] == 0
    assert dt_uma["synchronization_bridge"]["commit_cost_cpu_cycles"] == 0

    with pytest.raises(ContractViolation):
        comp.run_double_time_compositor_cycle(frame_budget_ms=1.0)
    with pytest.raises(ContractViolation):
        comp.run_double_time_compositor_cycle(uma_utilization=1.5)


def test_double_time_non_overlapping_128b_buffers_and_eta_threshold() -> None:
    emu = DoubleTimeCoefficientEmulator()
    sweeps = emu.emulate_coefficient_sweeps()
    assert sweeps["vital_max_hp"] == 6
    assert sweeps["critical_zero_stall_compression_threshold_eta"] == 1.72

    mb = sweeps["multi_buffer_hierarchy"]
    assert mb["buffers"][0]["size_bytes"] == 128
    assert mb["buffers"][1]["size_bytes"] == 128
    assert "0x000..0x07F" in mb["buffers"][0]["address_range_hex"]
    assert "0x080..0x0FF" in mb["buffers"][1]["address_range_hex"]
    assert "0x100..0x2FF" in mb["buffers"][2]["address_range_hex"]
    assert "за" not in json.dumps(mb, ensure_ascii=False)
