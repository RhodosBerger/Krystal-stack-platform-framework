# ==============================================================================
# KRYSTAL-STACK // GODOT 4.X MULTI-AXIAL RADAR ASPECT HUD
# ==============================================================================
# File: examples/godot_radar_aspect_hud.gd
# Description:
#   Plug-and-play Godot 4.2+ (Forward+ / Vulkan) 2D Control node that renders
#   a dynamic 4-axis radar aspect diamond (OBT, KON, POŠ, DOS) and 6-pip vital HP.
#
# Usage:
#   Attach this script to a Control node in your Godot scene.
#   Call `set_aspect_values(4.0, 5.0, 3.0, 4.0)` or `fetch_from_krystal_hub("PK-CRYSTAL")`.
# ==============================================================================
class_name GodotRadarAspectHUD
extends Control

# Invariants
const VITAL_MAX_HP: int = 6

# Aspect Scores [0.0 - 5.0]
@export_group("Aspect Scores (0.0 - 5.0)")
@export var difficulty_obt: float = 4.0:
	set(val):
		difficulty_obt = clampf(val, 0.0, 5.0)
		queue_redraw()

@export var control_kon: float = 5.0:
	set(val):
		control_kon = clampf(val, 0.0, 5.0)
		queue_redraw()

@export var damage_pos: float = 3.0:
	set(val):
		damage_pos = clampf(val, 0.0, 5.0)
		queue_redraw()

@export var range_dos: float = 4.0:
	set(val):
		range_dos = clampf(val, 0.0, 5.0)
		queue_redraw()

# Vital HP
@export_group("Vital Invariant")
@export var current_hp: int = 6:
	set(val):
		current_hp = clampi(val, 0, VITAL_MAX_HP)
		queue_redraw()

# Visual Styling
@export_group("Visual Theme")
@export var radar_radius: float = 80.0
@export var polygon_fill_color: Color = Color(0.4, 0.98, 0.94, 0.25) # Cyan glow
@export var polygon_stroke_color: Color = Color(0.4, 0.98, 0.94, 0.9)
@export var grid_line_color: Color = Color(1.0, 1.0, 1.0, 0.12)
@export var axis_label_color: Color = Color(0.83, 0.68, 0.21, 1.0) # Gold
@export var heart_full_color: Color = Color(0.83, 0.68, 0.21, 1.0)  # Gold/Green
@export var heart_empty_color: Color = Color(0.2, 0.2, 0.25, 0.4)

# HTTP Request for dynamic sync with Krystal Hub
var _http: HTTPRequest

func _ready() -> void:
	custom_minimum_size = Vector2(240, 260)
	_http = HTTPRequest.new()
	add_child(_http)
	_http.request_completed.connect(_on_http_request_completed)

func _draw() -> void:
	var center = size * 0.5
	center.y -= 15.0 # Leave room for hearts at bottom

	# 1. Draw Grid Rings (Diamond levels: 100%, 66%, 33%)
	var levels = [1.0, 0.66, 0.33]
	for lvl in levels:
		var r = radar_radius * lvl
		var diamond = PackedVector2Array([
			center + Vector2(0, -r),
			center + Vector2(r, 0),
			center + Vector2(0, r),
			center + Vector2(-r, 0)
		])
		draw_polyline(diamond, grid_line_color, 1.0, true)

	# 2. Draw Crosshair Axes
	draw_line(center + Vector2(0, -radar_radius), center + Vector2(0, radar_radius), grid_line_color, 1.0)
	draw_line(center + Vector2(-radar_radius, 0), center + Vector2(radar_radius, 0), grid_line_color, 1.0)

	# 3. Compute Data Polygon Coordinates
	var p_top = center + Vector2(0, -(difficulty_obt / 5.0) * radar_radius)
	var p_right = center + Vector2((control_kon / 5.0) * radar_radius, 0)
	var p_bottom = center + Vector2(0, (damage_pos / 5.0) * radar_radius)
	var p_left = center + Vector2(-(range_dos / 5.0) * radar_radius, 0)

	var data_polygon = PackedVector2Array([p_top, p_right, p_bottom, p_left])
	draw_colored_polygon(data_polygon, polygon_fill_color)
	draw_polyline(data_polygon, polygon_stroke_color, 2.0, true)

	# Draw Points
	for p in data_polygon:
		draw_circle(p, 3.5, axis_label_color)

	# 4. Draw Axis Text Labels
	var font = ThemeDB.fallback_font
	var font_size = 11
	draw_string(font, center + Vector2(-12, -radar_radius - 6), "OBT", HORIZONTAL_ALIGNMENT_CENTER, -1, font_size, axis_label_color)
	draw_string(font, center + Vector2(radar_radius + 6, 4), "KON", HORIZONTAL_ALIGNMENT_LEFT, -1, font_size, axis_label_color)
	draw_string(font, center + Vector2(-12, radar_radius + 16), "POŠ", HORIZONTAL_ALIGNMENT_CENTER, -1, font_size, axis_label_color)
	draw_string(font, center + Vector2(-radar_radius - 32, 4), "DOS", HORIZONTAL_ALIGNMENT_RIGHT, -1, font_size, axis_label_color)

	# 5. Draw 6 Vital HP Hearts / Pips
	var heart_y = size.y - 18.0
	var heart_spacing = 16.0
	var start_x = center.x - ((VITAL_MAX_HP - 1) * heart_spacing * 0.5)

	for i in range(VITAL_MAX_HP):
		var hx = start_x + (i * heart_spacing)
		var col = heart_full_color if i < current_hp else heart_empty_color
		# Draw diamond pip
		var pip = PackedVector2Array([
			Vector2(hx, heart_y - 5),
			Vector2(hx + 5, heart_y),
			Vector2(hx, heart_y + 5),
			Vector2(hx - 5, heart_y)
		])
		draw_colored_polygon(pip, col)

# Public API for Game Systems
func set_aspect_values(obt: float, kon: float, pos: float, dos: float) -> void:
	difficulty_obt = obt
	control_kon = kon
	damage_pos = pos
	range_dos = dos

func fetch_from_krystal_hub(subject_id: String = "PK-CRYSTAL", host: String = "http://127.0.0.1:8080") -> void:
	var url = host + "/api/measurement-disciplines/radar/" + subject_id
	_http.request(url)

func _on_http_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	if response_code != 200:
		push_warning("Failed to fetch radar data from Krystal Hub. Code: " + str(response_code))
		return
	var json = JSON.new()
	if json.parse(body.get_string_from_utf8()) == OK:
		var profile = json.data.get("profile", {})
		var aspects = profile.get("aspects", [])
		for asp in aspects:
			var code = asp.get("code", "")
			var score = float(asp.get("normalized_score", 3.0))
			if code == "OBT": difficulty_obt = score
			elif code == "KON": control_kon = score
			elif code == "POŠ": damage_pos = score
			elif code == "DOS": range_dos = score
		queue_redraw()
