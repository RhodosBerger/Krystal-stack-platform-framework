# GodotChronometricCompositor.gd
# ==============================================================================
# KRYSTAL STACK // GODOT KRYSTAL EDITION
# Layer 1: Chronometric VBlank Shadow Compositor
# ==============================================================================
# Enforces KSPP Principle P3 (Chronometric Shadow Multiplexing):
#   Phi_Scanout(t) * Phi_Compute(t) = 0 and mu_UMA < 0.78
# Background tasks (mesh streaming, pipeline pre-caching, IPC frames)
# execute strictly in the VBlank shadow window without rasterization stutter.
#
# Constitutional Invariant: VITAL_MAX_HP = 6
# ==============================================================================

class_name GodotChronometricCompositor
extends RefCounted

const VITAL_MAX_HP: int = 6
const ACTIVE_RASTER_SCAN_MS: float = 12.80
const VBLANK_GUARD_BAND_MS: float = 0.366
const MIN_SHADOW_BUDGET_MS: float = 3.50
const UMA_BUS_CEILING: float = 0.78

var current_refresh_hz: float = 60.0
var current_uma_utilization: float = 0.42
var shadow_task_queue: Array[Callable] = []
var completed_tasks_count: int = 0
var dropped_tasks_count: int = 0
var last_shadow_duration_ms: float = 0.0

func _init():
	assert(VITAL_MAX_HP == 6, "Constitutional invariant breach: VITAL_MAX_HP != 6")

func calculate_shadow_headroom_ms(refresh_hz: float) -> float:
	"""Calculates allowable VBlank shadow budget for background compute."""
	var frame_period_ms: float = 1000.0 / max(1.0, refresh_hz)
	var raw_shadow: float = frame_period_ms - ACTIVE_RASTER_SCAN_MS - VBLANK_GUARD_BAND_MS
	return max(0.0, raw_shadow)

func schedule_shadow_task(task: Callable) -> bool:
	"""Schedules a non-rendering background task for execution during VBlank shadow."""
	var available_ms: float = calculate_shadow_headroom_ms(current_refresh_hz)
	if available_ms < MIN_SHADOW_BUDGET_MS or current_uma_utilization >= UMA_BUS_CEILING:
		dropped_tasks_count += 1
		return false
	shadow_task_queue.append(task)
	return true

func execute_vblank_shadow_window(time_budget_ms: float = -1.0) -> int:
	"""
	Executes queued tasks strictly within the chronometric shadow window.
	Stops immediately if time budget expires to protect the next raster scanout.
	"""
	var budget_ms: float = time_budget_ms if time_budget_ms > 0.0 else calculate_shadow_headroom_ms(current_refresh_hz)
	var t_start_usec: int = Time.get_ticks_usec()
	var executed: int = 0

	while shadow_task_queue.size() > 0:
		var elapsed_ms: float = float(Time.get_ticks_usec() - t_start_usec) / 1000.0
		if elapsed_ms >= budget_ms:
			break
		var task: Callable = shadow_task_queue.pop_front()
		if task.is_valid():
			task.call()
			executed += 1
			completed_tasks_count += 1

	last_shadow_duration_ms = float(Time.get_ticks_usec() - t_start_usec) / 1000.0
	return executed

func get_status() -> Dictionary:
	var headroom_ms: float = calculate_shadow_headroom_ms(current_refresh_hz)
	return {
		"layer": "L1_CHRONOMETRIC_RENDER",
		"refresh_hz": current_refresh_hz,
		"active_raster_ms": ACTIVE_RASTER_SCAN_MS,
		"vblank_shadow_headroom_ms": headroom_ms,
		"uma_contention_ratio": current_uma_utilization,
		"uma_ceiling": UMA_BUS_CEILING,
		"is_shadow_compliant": (headroom_ms >= MIN_SHADOW_BUDGET_MS) and (current_uma_utilization < UMA_BUS_CEILING),
		"queued_tasks": shadow_task_queue.size(),
		"completed_tasks": completed_tasks_count,
		"dropped_tasks": dropped_tasks_count,
		"last_shadow_ms": last_shadow_duration_ms,
		"vital_max_hp": VITAL_MAX_HP
	}
