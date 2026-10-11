# ==============================================================================
# KRYSTAL-STACK: GODOT 4.X FORWARD+ VULKAN NATIVE PROCEDURAL BRIDGE
# ==============================================================================
# Script: res://scripts/NativeProceduralVulkanBridge.gd
# Description: High-speed native Godot 4.x bridge connecting directly to
#              Python's multi-threaded procedural renderer over binary IPC socket.
#              Updates 8,192+ MultiMesh instances in a single draw call without
#              single-threaded JavaScript or browser DOM bottlenecks.
#
# System Invariant: VITAL_MAX_HP == 6 strictly enforced.
# Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
# ==============================================================================

class_name NativeProceduralVulkanBridge
extends Node3D

signal frame_synchronized(instance_count: int, latency_ms: float, fps: float)
signal connection_state_changed(is_connected: bool)

@export var server_host: String = "127.0.0.1"
@export var server_port: int = 8092
@export var target_instances: int = 8192
@export var multimesh_node_path: NodePath = "MultiMeshInstance3D"
@export var auto_reconnect: bool = true

const VITAL_MAX_HP: int = 6
const MAGIC_HEADER: int = 0x4B525953 # "KRYS" in hex
const INSTANCE_STRIDE_BYTES: int = 96
const INSTANCE_STRIDE_FLOATS: int = 24

var _tcp: StreamPeerTCP = StreamPeerTCP.new()
var _connected: bool = false
var _multimesh_instance: MultiMeshInstance3D
var _multimesh: MultiMesh
var _time_accumulator: float = 0.0
var _reconnect_timer: float = 0.0

func _ready() -> void:
	print("[KRYSTAL VULKAN BRIDGE] Initializing Godot 4.x Forward+ Vulkan Bridge...")
	_setup_multimesh()
	_connect_to_server()

func _setup_multimesh() -> void:
	if has_node(multimesh_node_path):
		_multimesh_instance = get_node(multimesh_node_path) as MultiMeshInstance3D
	else:
		_multimesh_instance = MultiMeshInstance3D.new()
		_multimesh_instance.name = "MultiMeshInstance3D"
		add_child(_multimesh_instance)

	_multimesh = MultiMesh.new()
	_multimesh.transform_format = MultiMesh.TRANSFORM_3D
	_multimesh.use_colors = true
	_multimesh.use_custom_data = true
	_multimesh.instance_count = target_instances

	# Set default procedural box mesh if none present
	if _multimesh_instance.multimesh == null or _multimesh_instance.multimesh.mesh == null:
		var box = BoxMesh.new()
		box.size = Vector3(0.5, 1.2, 0.5)
		_multimesh.mesh = box

	_multimesh_instance.multimesh = _multimesh
	print("[KRYSTAL VULKAN BRIDGE] MultiMesh configured for %d instances (1 draw call)." % target_instances)

func _connect_to_server() -> void:
	print("[KRYSTAL VULKAN BRIDGE] Connecting to Python IPC at %s:%d..." % [server_host, server_port])
	var err = _tcp.connect_to_host(server_host, server_port)
	if err != OK:
		push_warning("[KRYSTAL VULKAN BRIDGE] Connection attempt failed: %d" % err)

func _physics_process(delta: float) -> void:
	_time_accumulator += delta
	_tcp.poll()
	var status = _tcp.get_status()

	if status == StreamPeerTCP.STATUS_CONNECTED:
		if not _connected:
			_connected = true
			emit_signal("connection_state_changed", true)
			print("[KRYSTAL VULKAN BRIDGE] Connected to Native Python Procedural Engine!")

		# Send request packet every frame: 8 bytes [uint32 target_count, uint32 seed]
		var req = PackedByteArray()
		req.resize(8)
		req.encode_u32(0, target_instances)
		req.encode_u32(4, int(_time_accumulator * 100))
		_tcp.put_data(req)

		# Read response header: 16 bytes [magic, count, total_bytes, vital_max_hp]
		if _tcp.get_available_bytes() >= 16:
			var header_data = _tcp.get_data(16)
			if header_data[0] == OK:
				var hdr: PackedByteArray = header_data[1]
				var magic = hdr.decode_u32(0)
				var count = hdr.decode_u32(4)
				var total_bytes = hdr.decode_u32(8)
				var vital_hp = hdr.decode_u32(12)

				# Strict Invariant Verification
				if vital_hp != VITAL_MAX_HP:
					push_error("[KRYSTAL VULKAN BRIDGE] CRITICAL: Platform invariant breach: VITAL_MAX_HP is %d (must be 6)!" % vital_hp)

				if magic == MAGIC_HEADER and total_bytes > 0:
					_read_and_apply_buffer(total_bytes, count)

	elif status == StreamPeerTCP.STATUS_ERROR or status == StreamPeerTCP.STATUS_NONE:
		if _connected:
			_connected = false
			emit_signal("connection_state_changed", false)
			push_warning("[KRYSTAL VULKAN BRIDGE] Connection lost to Python engine.")

		if auto_reconnect:
			_reconnect_timer += delta
			if _reconnect_timer >= 2.0:
				_reconnect_timer = 0.0
				_connect_to_server()

func _read_and_apply_buffer(total_bytes: int, count: int) -> void:
	# Await remaining bytes in TCP socket
	var t0 = Time.get_ticks_usec()
	while _tcp.get_available_bytes() < total_bytes:
		_tcp.poll()
		if _tcp.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			return

	var body_data = _tcp.get_data(total_bytes)
	if body_data[0] != OK:
		return

	var raw_bytes: PackedByteArray = body_data[1]
	var float_count = total_bytes / 4
	var float_buffer = raw_bytes.to_float32_array()

	if _multimesh != null and float_buffer.size() >= count * INSTANCE_STRIDE_FLOATS:
		# Direct Vulkan buffer update: 1 API call for all 8K instances
		_multimesh.buffer = float_buffer
		var elapsed_ms = (Time.get_ticks_usec() - t0) / 1000.0
		var current_fps = Engine.get_frames_per_second()
		emit_signal("frame_synchronized", count, elapsed_ms, current_fps)
