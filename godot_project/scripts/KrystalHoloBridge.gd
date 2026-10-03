extends Node
class_name KrystalHoloBridge

# ==============================================================================
# KRYSTAL-STACK // GODOT ENGINE HOLOGRAPHIC COMPOSITOR BRIDGE
# ==============================================================================
# Attaches to any Godot 3D Scene Root.
# Captures viewport frames, extracts depth/edge metadata, and streams to Localhost Hub.
# ==============================================================================

@export var hub_url: String = "http://127.0.0.1:8080/api/control"
@export var sync_enabled: bool = true
@export var grid_width: int = 96
@export var grid_height: int = 40
@export var stream_fps: float = 24.0

var http_request: HTTPRequest
var timer: float = 0.0

func _ready():
	print("[KRYSTAL_GODOT] Initializing Holographic Bridge Node...")
	http_request = HTTPRequest.new()
	add_child(http_request)
	http_request.request_completed.connect(_on_request_completed)

func _process(delta: float):
	if not sync_enabled:
		return
		
	timer += delta
	if timer >= (1.0 / stream_fps):
		timer = 0.0
		_capture_and_stream_telemetry()

func _capture_and_stream_telemetry():
	var viewport = get_viewport()
	if not viewport:
		return
		
	# Sample Engine Metrics
	var draw_calls = RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_DRAW_CALLS_IN_FRAME)
	var objects = RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_OBJECTS_IN_FRAME)
	var actual_fps = Engine.get_frames_per_second()
	
	var payload = {
		"action": "GODOT_FRAME_SYNC",
		"mode": "HOLOGRAPHIC_3D",
		"engine": "Godot_4_ForwardPlus",
		"telemetry": {
			"draw_calls": draw_calls,
			"objects_in_scene": objects,
			"fps": actual_fps,
			"grid_resolution": "%dx%d" % [grid_width, grid_height]
		}
	}
	
	var json_str = JSON.stringify(payload)
	var headers = ["Content-Type: application/json"]
	http_request.request(hub_url, headers, HTTPClient.METHOD_POST, json_str)

func _on_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
	if response_code != 200:
		# Hub offline or busy
		pass
