"""Tests for the OpenVINO Convolutional Operations-to-Hypervisor Step Bridge."""

from __future__ import annotations

import pytest

from krystal_kernel.constants import ContractViolation
from projects.trigonometric_hybrid_platform_framework.convolutional_hypervisor_step_bridge import (
    ConvolutionalHypervisorStepBridge,
)


def test_convolutional_hypervisor_bridge_nominal_spike_smoothing_and_uma_halt() -> None:
    bridge = ConvolutionalHypervisorStepBridge()

    # 1. Single transient 0.76 spike surrounded by low 0.30 UMA -> smoothed below 0.78, commits via StepExecutionGateProxy
    nominal_ops = [[0.30, 0.72, 0.65, 0.0, 0.0, 0.52] for _ in range(16)]
    nominal_ops[-2][0] = 0.76
    r_nom = bridge.evaluate_operations_to_hypervisor_step(nominal_ops, frame_budget_ms=8.333)
    assert r_nom["vital_max_hp"] == 6
    assert r_nom["conv_interposer"]["rel_error_vs_numpy"] < 1e-4
    assert r_nom["conv_interposer"]["conv_smoothed_uma"] < 0.76
    assert r_nom["hypervisor_step"]["mode"] == "COMMITTED_VIA_STEP_PROXY"
    assert r_nom["hypervisor_step"]["wntc_64b_crc_valid"] is True
    assert r_nom["hypervisor_step"]["double_time_lookahead_steps"] == 2  # 120 Hz clamped

    # 2. Sustained UMA >= 0.78 saturation -> StepExecutionGateProxy preserves Delta_S == 0 & triggers OP_01 fallback
    sat_ops = [[0.84, 0.80, 0.75, 0.0, 1.0, 0.52] for _ in range(16)]
    r_sat = bridge.evaluate_operations_to_hypervisor_step(sat_ops, frame_budget_ms=8.333)
    assert r_sat["hypervisor_step"]["mode"] == "OP_01_SHADOW_FALLBACK_DELTA_S_ZERO"
    assert r_sat["hypervisor_step"]["step_proxy_receipt"]["step_count_unchanged"] is True
    assert r_sat["hypervisor_step"]["wsl2_ram_gb"] == 8
    assert r_sat["hypervisor_step"]["double_time_lookahead_steps"] == 0

    with pytest.raises(ContractViolation):
        bridge.evaluate_operations_to_hypervisor_step([[0.3] * 6])
