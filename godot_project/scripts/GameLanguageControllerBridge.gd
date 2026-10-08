# ==============================================================================
# KRYSTAL-STACK: GODOT GAME LANGUAGE CONTROLLER & API FETCHER BRIDGE
# Compatible with: Godot 4.2+ (Forward+ / Vulkan)
# Translates Natural Language Prompts and Remote Language API Calls into
# Real-Time In-Engine Game Actions (Camera Rigs, Weather, Cards, and Spawning).
# Invariant: VITAL_MAX_HP = 6
# ==============================================================================
class_name GameLanguageControllerBridge
extends Node

const VITAL_MAX_HP: int = 6
const GOLDEN_RATIO: float = 1.61803398875

@export_group("Network & Hub Settings")
@export var hub_api_url: String = "http://127.0.0.1:8089"
@export var auto_poll_daemon: bool = false
@export var poll_interval_s: float = 2.5

@export_group("Scene Node Bindings")
@export var atmosphere_controller: MinecraftAtmosphereController
@export var dual_camera_rig: DualCameraRig3D
@export var hex_arena_root: Node3D
@export var hero_avatar_node: Node3D

# Internal HTTP Clients
var _http_control: HTTPRequest
var _http_poll: HTTPRequest
var _poll_timer: Timer

# Signals for UI and Gameplay Systems
signal language_command_processed(result_data: Dictionary)
signal game_state_queried(query_response: Dictionary)
signal weather_shifted(weather_name: String, time_of_day: float)
signal camera_preset_changed(preset_name: String)
signal unit_relocated(destination: Vector2i)

func _ready() -> void:
	print("[GameLanguageBridge] Initializing Language API Controller Bridge Node...")
	
	# HTTP Client for outgoing language commands
	_http_control = HTTPRequest.new()
	add_child(_http_control)
	_http_control.request_completed.connect(_on_control_request_completed)

	# HTTP Client for daemon polling
	_http_poll = HTTPRequest.new()
	add_child(_http_poll)
	_http_poll.request_completed.connect(_on_poll_request_completed)

	if auto_poll_daemon:
		_start_polling_timer()

func _start_polling_timer() -> void:
	if not _poll_timer:
		_poll_timer = Timer.new()
		_poll_timer.wait_time = poll_interval_s
		_poll_timer.autostart = true
		_poll_timer.timeout.connect(_on_poll_tick)
		add_child(_poll_timer)

func _on_poll_tick() -> void:
	if _http_poll.get_http_client_status() == HTTPClient.STATUS_DISCONNECTED:
		var url = hub_api_url + "/api/game/language/history?limit=1"
		_http_poll.request(url)

func _on_poll_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	if response_code == 200:
		var json_str = body.get_string_from_utf8()
		var parse_res = JSON.parse_string(json_str)
		if parse_res and parse_res.has("history"):
			var hist = parse_res["history"]
			if hist.size() > 0:
				var latest = hist[0]
				# Apply in-engine changes if successful
				if latest.get("status") == "SUCCESS":
					_apply_engine_side_effects(latest)

# ==============================================================================
# PUBLIC API: EXECUTE NATURAL LANGUAGE COMMAND
# ==============================================================================
func send_language_command(command_text: String) -> void:
	"""
	Sends a raw natural language prompt (e.g., 'Nastav počasie na búrku',
	'Prepnúť kameru na izometrický pohľad', 'Zahraj Kryštálový Štít')
	to the Krystal Language Controller API.
	"""
	if command_text.strip_edges().is_empty():
		return
		
	var payload = {
		"prompt": command_text,
		"execute_api": true,
		"vital_max_hp_rule": VITAL_MAX_HP
	}
	var json_str = JSON.stringify(payload)
	var headers = ["Content-Type: application/json"]
	var url = hub_api_url + "/api/game/language/control"

	_http_control.request(url, headers, HTTPClient.METHOD_POST, json_str)

func query_battlefield_state(question: String = "Aký je stav zápasu?") -> void:
	"""Queries battlefield state via natural language."""
	var payload = {"query": question}
	var json_str = JSON.stringify(payload)
	var headers = ["Content-Type: application/json"]
	var url = hub_api_url + "/api/game/language/query"

	_http_control.request(url, headers, HTTPClient.METHOD_POST, json_str)

func _on_control_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	if response_code == 200:
		var json_str = body.get_string_from_utf8()
		var res = JSON.parse_string(json_str)
		if res:
			print("[GameLanguageBridge] Command response: ", res.get("game_feedback", "OK"))
			_apply_engine_side_effects(res)
			language_command_processed.emit(res)
	else:
		print("[GameLanguageBridge] Warning: Hub returned HTTP ", response_code)

# ==============================================================================
# ENGINE SIDE EFFECTS DISPATCHER
# ==============================================================================
func _apply_engine_side_effects(res_data: Dictionary) -> void:
	var intent = res_data.get("intent", {})
	var cmd_type = intent.get("command_type", "")
	var params = intent.get("parameters", {})
	
	match cmd_type:
		"WEATHER_ATMOSPHERE":
			var w_state = params.get("weather_state", "Clear")
			var tod = float(params.get("time_of_day_hours", 12.0))
			_set_engine_atmosphere(w_state, tod)
			weather_shifted.emit(w_state, tod)

		"CAMERA_CONTROL":
			var preset = params.get("camera_preset", "perspective_action_follow")
			_set_camera_rig(preset)
			camera_preset_changed.emit(preset)

		"TACTICAL_MOVE":
			var dest_array = params.get("destination_hex", [0, 0])
			if dest_array.size() >= 2:
				var dest = Vector2i(dest_array[0], dest_array[1])
				_move_hero_avatar(dest)
				unit_relocated.emit(dest)

		"SPAWN_ENTITY":
			var model_id = params.get("model_id", "CesiumMan")
			var pos_array = params.get("position", [0, 0])
			_spawn_3d_model(model_id, pos_array)

func _set_engine_atmosphere(weather_name: String, time_of_day: float) -> void:
	if atmosphere_controller:
		atmosphere_controller.time_of_day_hours = time_of_day
		match weather_name:
			"Clear": atmosphere_controller.weather_state = 0
			"Overcast": atmosphere_controller.weather_state = 1
			"Rain": 
				atmosphere_controller.weather_state = 2
				atmosphere_controller.rain_intensity = 0.8
			"Thunderstorm": 
				atmosphere_controller.weather_state = 3
				atmosphere_controller.rain_intensity = 1.0

func _set_camera_rig(preset_id: String) -> void:
	if dual_camera_rig:
		if preset_id.begins_with("orthographic"):
			dual_camera_rig.switch_mode(DualCameraRig3D.CameraMode.ORTHOGRAPHIC, 0.4)
		elif preset_id.begins_with("perspective"):
			dual_camera_rig.switch_mode(DualCameraRig3D.CameraMode.PERSPECTIVE, 0.4)
		elif preset_id == "dual_cinematic_hybrid":
			dual_camera_rig.switch_mode(DualCameraRig3D.CameraMode.DUAL_HYBRID, 0.6)

func _move_hero_avatar(dest_hex: Vector2i) -> void:
	if hero_avatar_node:
		# Convert pointy-topped axial hex [q, r] to 3D world coordinates
		var hex_size = 2.0
		var world_x = hex_size * sqrt(3.0) * (dest_hex.x + dest_hex.y / 2.0)
		var world_z = hex_size * 1.5 * dest_hex.y
		var target_pos = Vector3(world_x, hero_avatar_node.position.y, world_z)
		
		# Smooth Tween Interpolation
		var tween = create_tween()
		tween.tween_property(hero_avatar_node, "position", target_pos, 0.45).set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_OUT)

func _spawn_3d_model(model_id: String, pos_array: Array) -> void:
	print("[GameLanguageBridge] Spawning model in Godot: ", model_id, " at ", pos_array)
	# Target model instantiation logic or visibility toggles
