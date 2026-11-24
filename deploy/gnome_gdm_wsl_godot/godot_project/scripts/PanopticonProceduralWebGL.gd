extends Node2D
# Godot 4.x GDScript procedurálny WebGL2 / GLES3 vykresľovač Panoptika a 60s Cyberpunk čankov
# Ústavné invarianty: VITAL_MAX_HP == 6 | GOLDEN_RATIO == 1.61803398875

const VITAL_MAX_HP: int = 6
const GOLDEN_RATIO: float = 1.61803398875
const UMA_LIMIT: float = 0.78

var phase_time: float = 0.0
var active_window_mode: String = "MODE_02_NIRI_SCROLLABLE_STRIP"

func _process(delta: float) -> void:
	phase_time += delta
	queue_redraw()

func compute_harmonic_radius(theta: float, r0: float = 140.0) -> Dictionary:
	var r: float = r0
	var dr_dtheta: float = 0.0
	for k in range(1, 5):
		var amp: float = 24.0 / pow(GOLDEN_RATIO, float(k))
		var omega: float = float(k * 3)
		r += amp * cos(omega * theta + phase_time * 0.8)
		dr_dtheta += -amp * omega * sin(omega * theta + phase_time * 0.8)
	return {"r": r, "dr_dtheta": dr_dtheta, "steepness_xi": abs(dr_dtheta) / max(1.0, r)}

func _draw() -> void:
	var center := Vector2(640.0, 360.0)
	var prev_pt := Vector2.ZERO
	for i in range(0, 129):
		var th: float = (float(i) / 128.0) * TAU
		var h := compute_harmonic_radius(th, 150.0)
		var pt := center + Vector2(cos(th), sin(th)) * float(h["r"])
		if i > 0:
			draw_line(prev_pt, pt, Color(0.0, 0.898, 1.0, 0.92), 2.2, true)
		prev_pt = pt
	draw_arc(center, 64.0, 0.0, TAU, 64, Color(1.0, 0.62, 0.17, 0.85), 2.0)
