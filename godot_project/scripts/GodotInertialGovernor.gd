# GodotInertialGovernor.gd
# ==============================================================================
# KRYSTAL STACK // GODOT KRYSTAL EDITION
# Layer 3: Cognitive Cadence & 12-Frame Dwell Hysteresis Governor
# ==============================================================================
# Enforces KSPP Principle P8 (Inertial Stability & Anti-Oscillation):
#   - Fast downscale: Immediate on deadline pressure (0-frame latency).
#   - Hysteretic upscale: Requires N_dwell >= 12 consecutive frames with
#     Merkle-Momentum smoothing: M_k = gamma * M_{k-1} + (1 - gamma) * T_k.
#   - 10 Hz Alpha cadence pulse generator for visual pacing and shader modulation.
#
# Constitutional Invariant: VITAL_MAX_HP = 6
# ==============================================================================

class_name GodotInertialGovernor
extends RefCounted

const VITAL_MAX_HP: int = 6
const MIN_DWELL_FRAMES: int = 12
const MERKLE_GAMMA_DEFAULT: float = 0.85
const ALPHA_CADENCE_HZ: float = 10.0

# Discrete scale steps for viewport scaling (e.g. 0.50x, 0.67x, 0.75x, 0.85x, 1.0x)
const SCALE_STEPS: Array[float] = [0.50, 0.67, 0.75, 0.85, 1.00]

var current_scale_index: int = 4  # Start at 1.00x
var target_scale_index: int = 4
var dwell_counter: int = 0
var merkle_momentum: float = 1.0
var gamma: float = MERKLE_GAMMA_DEFAULT
var alpha_phase: float = 0.0
var total_scale_adjustments: int = 0
var chattering_events_prevented: int = 0

func _init():
	assert(VITAL_MAX_HP == 6, "Constitutional invariant breach: VITAL_MAX_HP != 6")

func update_cadence(delta: float, frame_time_ms: float, target_deadline_ms: float = 16.67) -> float:
	"""
	Evaluates frame deadline margin and updates dynamic resolution scale
	via 12-frame dwell hysteresis and Merkle-Momentum filter.
	"""
	# Update 10 Hz Alpha cadence harmonic phase [0.0, 1.0)
	alpha_phase = fmod(alpha_phase + delta * ALPHA_CADENCE_HZ, 1.0)

	# Pressure indicator: ratio of actual frame time to deadline
	var pressure: float = frame_time_ms / max(1.0, target_deadline_ms)
	merkle_momentum = gamma * merkle_momentum + (1.0 - gamma) * pressure

	if pressure > 1.05:
		# DEADLINE PRESSURE: Fast downscale immediately (P8 fast path)
		if current_scale_index > 0:
			current_scale_index -= 1
			target_scale_index = current_scale_index
			dwell_counter = 0
			total_scale_adjustments += 1
	elif pressure < 0.80 and current_scale_index < (SCALE_STEPS.size() - 1):
		# HEADROOM SURPLUS: Candidate for upscale, requires 12-frame dwell
		if target_scale_index <= current_scale_index:
			target_scale_index = current_scale_index + 1
			dwell_counter = 1
		else:
			dwell_counter += 1
			if dwell_counter >= MIN_DWELL_FRAMES and merkle_momentum < 0.85:
				current_scale_index = target_scale_index
				dwell_counter = 0
				total_scale_adjustments += 1
	else:
		# In stable or fluctuating band: suppress upscale chattering
		if dwell_counter > 0 and dwell_counter < MIN_DWELL_FRAMES:
			chattering_events_prevented += 1
		dwell_counter = 0
		target_scale_index = current_scale_index

	return get_current_viewport_scale()

func get_current_viewport_scale() -> float:
	return SCALE_STEPS[current_scale_index]

func get_alpha_pulse_intensity() -> float:
	"""Returns smooth sine pulsation [0.0, 1.0] for holographic glow at 10 Hz."""
	return 0.5 + 0.5 * sin(alpha_phase * 2.0 * PI)

func get_status() -> Dictionary:
	return {
		"layer": "L3_COGNITIVE_CADENCE",
		"viewport_scale": get_current_viewport_scale(),
		"scale_step_index": current_scale_index,
		"target_scale_index": target_scale_index,
		"dwell_counter": dwell_counter,
		"min_dwell_required": MIN_DWELL_FRAMES,
		"merkle_momentum": merkle_momentum,
		"gamma": gamma,
		"alpha_cadence_hz": ALPHA_CADENCE_HZ,
		"alpha_pulse": get_alpha_pulse_intensity(),
		"chattering_prevented": chattering_events_prevented,
		"total_adjustments": total_scale_adjustments,
		"vital_max_hp": VITAL_MAX_HP
	}
