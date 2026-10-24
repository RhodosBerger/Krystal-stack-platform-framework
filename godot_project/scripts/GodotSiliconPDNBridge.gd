# GodotSiliconPDNBridge.gd
# ==============================================================================
# KRYSTAL STACK // GODOT KRYSTAL EDITION
# Layer 0: Silicon PDN & Hardware Telemetry Bridge
# ==============================================================================
# Monitors physical CPU/GPU voltage (Vdd), junction temperature (Tj),
# transient inductive overshoot, and RAPL package power flux.
# Triggers automatic workload shedding when silicon safety envelopes are strained.
#
# Constitutional Invariant: VITAL_MAX_HP = 6
# ==============================================================================

class_name GodotSiliconPDNBridge
extends RefCounted

const VITAL_MAX_HP: int = 6
const NOMINAL_VDD_V: float = 0.85
const ABSOLUTE_MAX_VDD_V: float = 1.00
const MAX_VOLTAGE_OVERSHOOT_V: float = 0.15
const THERMAL_THROTTLE_CELSIUS: float = 85.0
const PL1_POWER_BUDGET_WATTS: float = 28.0

var current_vdd_v: float = NOMINAL_VDD_V
var current_overshoot_v: float = 0.08
var current_tj_celsius: float = 58.0
var current_power_watts: float = 18.5
var pdn_impedance_z0: float = 50.0
var is_throttled: bool = false
var stress_event_count: int = 0

func _init():
	assert(VITAL_MAX_HP == 6, "Constitutional invariant breach: VITAL_MAX_HP != 6")

func update_telemetry(raw_data: Dictionary) -> Dictionary:
	"""Updates silicon telemetry from kernel and determines throttle actions."""
	if raw_data.has("vdd_volts"):
		current_vdd_v = float(raw_data["vdd_volts"])
	if raw_data.has("overshoot_volts"):
		current_overshoot_v = float(raw_data["overshoot_volts"])
	if raw_data.has("tj_celsius"):
		current_tj_celsius = float(raw_data["tj_celsius"])
	if raw_data.has("power_watts"):
		current_power_watts = float(raw_data["power_watts"])
	if raw_data.has("pdn_z0"):
		pdn_impedance_z0 = float(raw_data["pdn_z0"])

	var voltage_strained: bool = (current_vdd_v + current_overshoot_v) > ABSOLUTE_MAX_VDD_V
	var thermal_strained: bool = current_tj_celsius >= THERMAL_THROTTLE_CELSIUS
	var power_strained: bool = current_power_watts > PL1_POWER_BUDGET_WATTS

	is_throttled = voltage_strained or thermal_strained or power_strained
	if is_throttled:
		stress_event_count += 1

	return get_status()

func get_status() -> Dictionary:
	return {
		"layer": "L0_SILICON_PDN",
		"vdd_volts": current_vdd_v,
		"overshoot_volts": current_overshoot_v,
		"peak_die_voltage_v": current_vdd_v + current_overshoot_v,
		"tj_celsius": current_tj_celsius,
		"power_watts": current_power_watts,
		"pdn_z0_ohms": pdn_impedance_z0,
		"is_throttled": is_throttled,
		"stress_events": stress_event_count,
		"safe_envelope_adhered": (current_vdd_v + current_overshoot_v <= ABSOLUTE_MAX_VDD_V),
		"vital_max_hp": VITAL_MAX_HP
	}
