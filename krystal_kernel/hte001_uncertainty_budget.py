"""
==============================================================================
KSL HTE-001 — MEASUREMENT UNCERTAINTY BUDGET (GUM / ISO 98-3)
==============================================================================
Formal Metrological Model for INA226 Telemetry on Xilinx KCU116 FPGA Board.
Features:
    - Rigorous Separation of Type-A (Random / Statistical) and Type-B (Systematic) Uncertainties
    - Common-Mode Covariance Cancellation in Paired Differential Active-Idle Measurements
    - Conservative (Uncorrelated) vs Correlated (Differential) Uncertainty Bounds
    - Expanded Measurement Uncertainty U_95% for Dynamic Energy per Validated Operation
==============================================================================
"""

import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass(frozen=True)
class Ina226HardwareSpecs:
    """TI INA226 Calibrated Hardware Specifications."""
    shunt_resistance_ohms: float = 0.010          # 10 mOhm precision shunt
    shunt_tolerance_rel: float = 0.001           # 0.1% tolerance
    shunt_tcr_ppm_per_c: float = 25.0            # 25 ppm / °C
    temp_delta_c: float = 20.0                   # Max ambient-to-active thermal rise
    temp_drift_active_to_idle_c: float = 0.5     # Residual thermal drift between paired idle and active
    vbus_gain_error_rel: float = 0.001           # 0.1% max gain error
    vbus_lsb_volts: float = 0.00125              # 1.25 mV LSB
    vshunt_gain_error_rel: float = 0.001         # 0.1% max gain error
    vshunt_offset_volts: float = 10.0e-6         # 10 uV max offset voltage
    vshunt_lsb_volts: float = 2.5e-6             # 2.5 uV LSB
    vshunt_input_noise_rms_volts: float = 2.5e-6 # Random input-referred noise floor
    time_base_jitter_rel: float = 5.0e-6         # Clock oscillator stability (5 ppm)


@dataclass
class MetrologicalUncertaintyBudget:
    """Calculates Type-A and Type-B combined and expanded uncertainties according to JCGM 100:2008 (GUM)."""
    specs: Ina226HardwareSpecs = Ina226HardwareSpecs()

    def calculate_single_sample_uncertainty(
        self,
        v_bus: float = 12.0,
        p_nominal: float = 4.18
    ) -> Dict[str, float]:
        """
        Decomposes uncertainty for a single instantaneous power reading into:
            u_A: Random noise (quantization, ADC thermal noise) - reduces with sqrt(N)
            u_B: Systematic error (gain errors, offset, resistor tolerance, TCR) - does NOT reduce with sqrt(N)
        """
        i_nominal = p_nominal / v_bus
        v_shunt = i_nominal * self.specs.shunt_resistance_ohms  # e.g. 0.34833 A * 0.010 Ohm = 3.4833 mV

        # ---------------------------------------------------------------------
        # 1. Type-A Components (Random per-sample fluctuations)
        # ---------------------------------------------------------------------
        u_vbus_quant = self.specs.vbus_lsb_volts / math.sqrt(12.0)
        u_vshunt_quant = self.specs.vshunt_lsb_volts / math.sqrt(12.0)
        u_vshunt_noise = self.specs.vshunt_input_noise_rms_volts

        c_vbus = i_nominal
        c_vshunt = v_bus / self.specs.shunt_resistance_ohms

        var_type_a = (c_vbus * u_vbus_quant)**2 + (c_vshunt * math.sqrt(u_vshunt_quant**2 + u_vshunt_noise**2))**2
        u_type_a = math.sqrt(var_type_a)

        # ---------------------------------------------------------------------
        # 2. Type-B Components (Systematic hardware bounds)
        # ---------------------------------------------------------------------
        u_vbus_gain = (self.specs.vbus_gain_error_rel * v_bus) / math.sqrt(3.0)
        u_vshunt_gain = (self.specs.vshunt_gain_error_rel * v_shunt) / math.sqrt(3.0)
        u_vshunt_offset = self.specs.vshunt_offset_volts / math.sqrt(3.0)

        u_r_tol = (self.specs.shunt_tolerance_rel * self.specs.shunt_resistance_ohms) / math.sqrt(3.0)
        tcr_drift_rel = (self.specs.shunt_tcr_ppm_per_c * 1e-6) * self.specs.temp_delta_c
        u_r_tcr = (tcr_drift_rel * self.specs.shunt_resistance_ohms) / math.sqrt(3.0)
        u_r = math.sqrt(u_r_tol**2 + u_r_tcr**2)

        c_r = - p_nominal / self.specs.shunt_resistance_ohms

        var_type_b = (
            (c_vbus * u_vbus_gain)**2 +
            (c_vshunt * math.sqrt(u_vshunt_gain**2 + u_vshunt_offset**2))**2 +
            (c_r * u_r)**2
        )
        u_type_b = math.sqrt(var_type_b)

        # Combined per-reading uncertainty
        u_c = math.sqrt(u_type_a**2 + u_type_b**2)

        return {
            "v_bus": v_bus,
            "i_shunt": i_nominal,
            "v_shunt": v_shunt,
            "p_nominal_w": p_nominal,
            "u_type_a_w": u_type_a,
            "u_type_b_w": u_type_b,
            "u_c_w": u_c,
            "u_r_ohms": u_r,
            "u_vshunt_offset_v": u_vshunt_offset,
            "u_vbus_gain_v": u_vbus_gain,
            "u_vshunt_gain_v": u_vshunt_gain
        }

    def calculate_energy_uncertainty_budget(
        self,
        p_active: float = 4.18,
        p_idle: float = 3.25,
        v_bus_active: float = 11.95,
        v_bus_idle: float = 12.00,
        t_burst_s: float = 0.208,
        n_validated_ops: int = 1_000_000,
        n_independent_active_samples: int = 742,
        n_independent_idle_samples: int = 358
    ) -> Dict[str, Any]:
        """
        Calculates the GUM uncertainty budget for differential energy measurement:
            Delta P = P_active - P_idle
            E_dynamic = Delta P * t_burst
            E_op = E_dynamic / N_validated_ops

        Evaluates two rigorous bounds:
          1. Conservative Bound (u_B does not cancel, worst-case independent errors)
          2. Correlated Differential Bound (Common-mode covariance cancellation of shunt R and offset V_OS)
        """
        active_p = self.calculate_single_sample_uncertainty(v_bus=v_bus_active, p_nominal=p_active)
        idle_p = self.calculate_single_sample_uncertainty(v_bus=v_bus_idle, p_nominal=p_idle)

        delta_p = p_active - p_idle
        e_dynamic_j = delta_p * t_burst_s
        e_op_j = e_dynamic_j / n_validated_ops
        e_op_uj = e_op_j * 1e6

        # ---------------------------------------------------------------------
        # A. Type-A Uncertainty (Averages down with sqrt(N))
        # ---------------------------------------------------------------------
        u_mean_active_a = active_p["u_type_a_w"] / math.sqrt(max(1, n_independent_active_samples))
        u_mean_idle_a = idle_p["u_type_a_w"] / math.sqrt(max(1, n_independent_idle_samples))
        u_delta_p_type_a = math.sqrt(u_mean_active_a**2 + u_mean_idle_a**2)

        # ---------------------------------------------------------------------
        # B. Type-B Model 1: Conservative Uncorrelated Bound
        # ---------------------------------------------------------------------
        u_delta_p_type_b_conservative = math.sqrt(active_p["u_type_b_w"]**2 + idle_p["u_type_b_w"]**2)
        u_delta_p_conservative = math.sqrt(u_delta_p_type_a**2 + u_delta_p_type_b_conservative**2)
        u_e_dynamic_conservative_j = u_delta_p_conservative * t_burst_s
        u_e_op_conservative_uj = (u_e_dynamic_conservative_j / n_validated_ops) * 1e6
        u95_conservative_uj = 2.0 * u_e_op_conservative_uj

        # ---------------------------------------------------------------------
        # C. Type-B Model 2: Correlated Differential Common-Mode Cancellation
        # ---------------------------------------------------------------------
        # 1. Shunt resistance error depends strictly on Delta P, not total P:
        #    d(Delta P) / d(R_shunt) = - Delta P / R_shunt
        c_delta_r = - delta_p / self.specs.shunt_resistance_ohms
        # Residual TCR drift between active burst and idle baseline (Delta T <= 0.5 °C in paired baseline)
        u_r_diff_tcr = (self.specs.shunt_tcr_ppm_per_c * 1e-6 * self.specs.temp_drift_active_to_idle_c * self.specs.shunt_resistance_ohms) / math.sqrt(3.0)
        u_r_diff = math.sqrt(active_p["u_r_ohms"]**2 + u_r_diff_tcr**2)
        var_diff_r = (c_delta_r * u_r_diff)**2

        # 2. Offset voltage cancellation:
        #    V_OS affects both Active and Idle through the same analog input.
        #    Delta P_offset = (V_bus_active - V_bus_idle) * V_OS / R_shunt
        v_bus_droop = abs(v_bus_active - v_bus_idle)
        var_diff_vos = ((v_bus_droop / self.specs.shunt_resistance_ohms) * active_p["u_vshunt_offset_v"])**2

        # 3. Differential gain errors:
        var_diff_gain = (delta_p * math.sqrt(2.0) * (self.specs.vshunt_gain_error_rel / math.sqrt(3.0)))**2

        u_delta_p_type_b_correlated = math.sqrt(var_diff_r + var_diff_vos + var_diff_gain)
        u_delta_p_correlated = math.sqrt(u_delta_p_type_a**2 + u_delta_p_type_b_correlated**2)
        u_e_dynamic_correlated_j = u_delta_p_correlated * t_burst_s
        u_e_op_correlated_uj = (u_e_dynamic_correlated_j / n_validated_ops) * 1e6
        u95_correlated_uj = 2.0 * u_e_op_correlated_uj

        # Timing uncertainty
        u_time = (self.specs.time_base_jitter_rel * t_burst_s) / math.sqrt(3.0)

        epsilon_energy_uj = 0.050  # Preregistered HTE-001 tolerance (50 nJ)

        return {
            "p_active_w": p_active,
            "p_idle_w": p_idle,
            "delta_p_w": delta_p,
            "t_burst_s": t_burst_s,
            "e_dynamic_j": e_dynamic_j,
            "e_op_uj": e_op_uj,
            "u_type_a_delta_p_w": u_delta_p_type_a,
            "u_type_b_conservative_w": u_delta_p_type_b_conservative,
            "u_type_b_correlated_w": u_delta_p_type_b_correlated,
            # Conservative bound (Worst-Case Uncorrelated Type-B)
            "u95_conservative_uj": u95_conservative_uj,
            "discrimination_ratio_conservative": epsilon_energy_uj / u95_conservative_uj,
            # Correlated bound (Common-Mode Differential Cancellation)
            "u95_correlated_uj": u95_correlated_uj,
            "discrimination_ratio_correlated": epsilon_energy_uj / u95_correlated_uj,
            # Official active benchmark (Correlated)
            "u95_e_op_uj": u95_correlated_uj,
            "epsilon_energy_uj": epsilon_energy_uj,
            "metrologically_resolvable": u95_correlated_uj < epsilon_energy_uj,
            "metrological_status": "TYPE_A_TYPE_B_DECOMPOSED"
        }

    def evaluate_residual_metrology(
        self,
        r_energy_uj: float,
        epsilon_energy_uj: float = 0.050,
        budget: Dict[str, Any] = None
    ) -> Tuple[str, str]:
        """
        Classifies energy residual against the metrological uncertainty threshold:
          1. Sub-Noise Region: |R_energy| <= U_95% -> Indistinguishable from noise.
          2. Resolvable Compliant: U_95% < |R_energy| <= epsilon -> Statistically valid discrepancy.
          3. Model Falsification: |R_energy| > epsilon -> Model envelope breached.
        """
        if budget is None:
            budget = self.calculate_energy_uncertainty_budget()

        u95 = budget["u95_correlated_uj"]
        abs_r = abs(r_energy_uj)

        if abs_r <= u95:
            classification = "SUB_NOISE_FLOOR"
            verdict = "Residual is indistinguishable from INA226 metrological uncertainty band."
        elif abs_r <= epsilon_energy_uj:
            classification = "RESOLVABLE_COMPLIANT"
            verdict = "Residual is physically resolved beyond measurement noise and within preregistered tolerance envelope."
        else:
            classification = "ENVELOPE_BREACH_FALSIFIED"
            verdict = "Residual exceeds preregistered tolerance envelope (model hypothesis falsified)."

        return classification, verdict
