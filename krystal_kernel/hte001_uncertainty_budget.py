"""
==============================================================================
KSL HTE-001 — MEASUREMENT UNCERTAINTY BUDGET (GUM / ISO 98-3)
==============================================================================
Formal Metrological Model for INA226 Telemetry on Xilinx KCU116 FPGA Board.
Calculates expanded measurement uncertainty U_95% for:
    - Bus Voltage (V_bus)
    - Shunt Current (I_shunt)
    - Instantaneous Board Power (P_board)
    - Dynamic Energy (E_dynamic)
    - Dynamic Energy per Validated Operation (E_op)
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
    vbus_gain_error_rel: float = 0.001           # 0.1% max gain error
    vbus_lsb_volts: float = 0.00125              # 1.25 mV LSB
    vshunt_gain_error_rel: float = 0.001         # 0.1% max gain error
    vshunt_offset_volts: float = 10.0e-6         # 10 uV max offset voltage
    vshunt_lsb_volts: float = 2.5e-6             # 2.5 uV LSB
    time_base_jitter_rel: float = 5.0e-6         # Clock oscillator stability (5 ppm)


@dataclass
class MetrologicalUncertaintyBudget:
    """Calculates combined and expanded uncertainties according to JCGM 100:2008 (GUM)."""
    specs: Ina226HardwareSpecs = Ina226HardwareSpecs()

    def calculate_power_uncertainty(
        self,
        v_bus: float = 12.0,
        p_nominal: float = 4.18
    ) -> Dict[str, float]:
        """
        Calculates power measurement uncertainty components for a given operating point.
        P = (V_bus * V_shunt) / R_shunt.
        """
        i_nominal = p_nominal / v_bus
        v_shunt = i_nominal * self.specs.shunt_resistance_ohms

        # 1. Bus Voltage Uncertainty Components (Rectangular distributions: / sqrt(3) or / sqrt(12))
        u_vbus_gain = (self.specs.vbus_gain_error_rel * v_bus) / math.sqrt(3.0)
        u_vbus_quant = self.specs.vbus_lsb_volts / math.sqrt(12.0)
        u_vbus = math.sqrt(u_vbus_gain**2 + u_vbus_quant**2)

        # 2. Shunt Voltage Uncertainty Components
        u_vshunt_gain = (self.specs.vshunt_gain_error_rel * v_shunt) / math.sqrt(3.0)
        u_vshunt_offset = self.specs.vshunt_offset_volts / math.sqrt(3.0)
        u_vshunt_quant = self.specs.vshunt_lsb_volts / math.sqrt(12.0)
        u_vshunt = math.sqrt(u_vshunt_gain**2 + u_vshunt_offset**2 + u_vshunt_quant**2)

        # 3. Shunt Resistor Uncertainty Components (tolerance + TCR drift)
        u_r_tol = (self.specs.shunt_tolerance_rel * self.specs.shunt_resistance_ohms) / math.sqrt(3.0)
        tcr_drift_rel = (self.specs.shunt_tcr_ppm_per_c * 1e-6) * self.specs.temp_delta_c
        u_r_tcr = (tcr_drift_rel * self.specs.shunt_resistance_ohms) / math.sqrt(3.0)
        u_r = math.sqrt(u_r_tol**2 + u_r_tcr**2)

        # 4. Partial Derivatives (Sensitivity Coefficients)
        c_vbus = v_shunt / self.specs.shunt_resistance_ohms  # = I
        c_vshunt = v_bus / self.specs.shunt_resistance_ohms
        c_r = - (v_bus * v_shunt) / (self.specs.shunt_resistance_ohms**2)  # = - P / R

        # 5. Combined Standard Uncertainty for Power: u_c(P)
        var_p = (c_vbus * u_vbus)**2 + (c_vshunt * u_vshunt)**2 + (c_r * u_r)**2
        u_c_power = math.sqrt(var_p)
        rel_u_power = u_c_power / p_nominal

        return {
            "v_bus": v_bus,
            "i_shunt": i_nominal,
            "p_nominal_w": p_nominal,
            "u_vbus_v": u_vbus,
            "u_vshunt_v": u_vshunt,
            "u_r_ohms": u_r,
            "u_c_power_w": u_c_power,
            "rel_u_power_percent": rel_u_power * 100.0,
            "u95_power_w": 2.0 * u_c_power
        }

    def calculate_energy_uncertainty_budget(
        self,
        p_active: float = 4.18,
        p_idle: float = 3.25,
        t_burst_s: float = 0.208,
        n_validated_ops: int = 1_000_000,
        n_independent_active_samples: int = 742,
        n_independent_idle_samples: int = 358
    ) -> Dict[str, Any]:
        """
        Propagates uncertainty to dynamic energy and per-operation energy:
            E_dynamic = (P_active - P_idle) * t_burst
            E_op = E_dynamic / N_validated_ops
        """
        active_p_res = self.calculate_power_uncertainty(v_bus=12.0, p_nominal=p_active)
        idle_p_res = self.calculate_power_uncertainty(v_bus=12.0, p_nominal=p_idle)

        u_p_active = active_p_res["u_c_power_w"]
        u_p_idle = idle_p_res["u_c_power_w"]

        # Mean power estimation uncertainty reduces with independent sample count:
        # u(mean_P) = u_P / sqrt(N_independent)
        u_mean_active = u_p_active / math.sqrt(max(1, n_independent_active_samples))
        u_mean_idle = u_p_idle / math.sqrt(max(1, n_independent_idle_samples))

        # Time duration uncertainty
        u_time = (self.specs.time_base_jitter_rel * t_burst_s) / math.sqrt(3.0)

        # Delta P = P_active - P_idle
        delta_p = p_active - p_idle
        u_delta_p = math.sqrt(u_mean_active**2 + u_mean_idle**2)

        # Dynamic Energy: E_dynamic = Delta_P * t_burst
        e_dynamic_j = delta_p * t_burst_s
        var_e_dynamic = (t_burst_s * u_delta_p)**2 + (delta_p * u_time)**2
        u_e_dynamic_j = math.sqrt(var_e_dynamic)

        # Per-operation energy: E_op = E_dynamic / N_ops
        e_op_j = e_dynamic_j / n_validated_ops
        e_op_uj = e_op_j * 1e6
        u_e_op_uj = (u_e_dynamic_j / n_validated_ops) * 1e6

        # Expanded uncertainty (coverage factor k = 2 for ~95% confidence interval)
        u95_e_op_uj = 2.0 * u_e_op_uj

        return {
            "p_active_w": p_active,
            "p_idle_w": p_idle,
            "delta_p_w": delta_p,
            "t_burst_s": t_burst_s,
            "e_dynamic_j": e_dynamic_j,
            "e_op_uj": e_op_uj,
            "u_c_e_op_uj": u_e_op_uj,
            "u95_e_op_uj": u95_e_op_uj,
            "rel_u95_percent": (u95_e_op_uj / e_op_uj) * 100.0 if e_op_uj > 0 else 0.0,
            "discrimination_noise_floor_uj": u95_e_op_uj,
            "epsilon_energy_uj": 0.050,  # Preregistered tolerance
            "metrologically_resolvable": u95_e_op_uj < 0.050
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

        u95 = budget["u95_e_op_uj"]
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
