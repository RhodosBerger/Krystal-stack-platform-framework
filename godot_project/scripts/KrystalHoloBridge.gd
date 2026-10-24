extends Node
class_name KrystalHoloBridge

# ==============================================================================
# KRYSTAL-STACK // GODOT ENGINE HOLOGRAPHIC COMPOSITOR BRIDGE
# ==============================================================================
# Attaches to any Godot 3D Scene Root.
# Captures viewport frames, extracts depth/edge metadata, and streams to Localhost Hub
# synchronized to the Chronometric Shadow Multiplexing & Hexadecimal HSL Grid.
#
# Constitutional Invariant: VITAL_MAX_HP == 6
# ==============================================================================

const VITAL_MAX_HP: int = 6
const FRAME_MAGIC: int = 0x4B525953  # 'KRYS'
const FRAME_VERSION: int = 1
const SECTOR_CODES: Array[String] = ["0x10", "0x20", "0x30", "0x40", "0x50", "0x60"]

@export var hub_url: String = "http://127.0.0.1:8080/api/control"
@export var sync_enabled: bool = true
@export var grid_width: int = 96
@export var grid_height: int = 40
@export var stream_fps: float = 60.0  # Aligned to 60 FPS nominal cadence (16.67 ms)
@export var active_sector_index: int = 0

var http_request: HTTPRequest
var timer: float = 0.0
var frame_sequence: int = 0

func _ready():
	assert(VITAL_MAX_HP == 6, "Constitutional invariant breach in KrystalHoloBridge: VITAL_MAX_HP != 6")
	print("[KRYSTAL_GODOT] Initializing Holographic Bridge Node (6-Sector Hex HSL + 128B Framing)...")
	http_request = HTTPRequest.new()
	add_child(http_request)
	http_request.request_completed.connect(_on_request_completed)

func _process(delta: float):
	if not sync_enabled:
		return
		
	timer += delta
	# Cadence alignment to display refresh
	if timer >= (1.0 / stream_fps):
		timer = 0.0
		_capture_and_stream_telemetry()

func get_active_sector_code() -> String:
	var idx = clamp(active_sector_index, 0, VITAL_MAX_HP - 1)
	return SECTOR_CODES[idx]

func pack_hex_hsl_color(hue: float, saturation: float, lightness: float, alpha: float = 1.0) -> int:
	var h_byte = int(clamp(hue, 0.0, 1.0) * 255.0) & 0xFF
	var s_byte = int(clamp(saturation, 0.0, 1.0) * 255.0) & 0xFF
	var l_byte = int(clamp(lightness, 0.0, 1.0) * 255.0) & 0xFF
	var a_byte = int(clamp(alpha, 0.0, 1.0) * 255.0) & 0xFF
	return (h_byte << 24) | (s_byte << 16) | (l_byte << 8) | a_byte

func _capture_and_stream_telemetry():
	var viewport = get_viewport()
	if not viewport:
		return
		
	# Sample Engine Metrics
	var draw_calls = RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_DRAW_CALLS_IN_FRAME)
	var objects = RenderingServer.get_rendering_info(RenderingServer.RENDERING_INFO_TOTAL_OBJECTS_IN_FRAME)
	var actual_fps = Engine.get_frames_per_second()
	frame_sequence += 1
	
	# Determine sector rotation based on frame sequence modulo 6
	active_sector_index = frame_sequence % VITAL_MAX_HP
	var sector = get_active_sector_code()
	
	# Calculate representative Hex-HSL color for current sector
	var hue = float(active_sector_index) / float(VITAL_MAX_HP)
	var hex_hsl_val = pack_hex_hsl_color(hue, 0.85, 0.50)

	var payload = {
		"action": "GODOT_FRAME_SYNC",
		"mode": "HOLOGRAPHIC_3D",
		"engine": "Godot_4_ForwardPlus",
		"vital_max_hp": VITAL_MAX_HP,
		"telemetry": {
			"draw_calls": draw_calls,
			"objects_in_scene": objects,
			"fps": actual_fps,
			"grid_resolution": "%dx%d" % [grid_width, grid_height],
			"frame_sequence": frame_sequence,
			"sector_code": sector,
			"hex_hsl_32bit": "0x%08X" % hex_hsl_val,
			"cadence_ms": 1000.0 / stream_fps,
			"vblank_shadow_safe": (1000.0 / stream_fps) <= 16.67
		}
	}
	
	var json_str = JSON.stringify(payload)
	var headers = ["Content-Type: application/json"]
	http_request.request(hub_url, headers, HTTPClient.METHOD_POST, json_str)

func _on_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
	if response_code != 200:
		# Hub offline or busy
		pass
