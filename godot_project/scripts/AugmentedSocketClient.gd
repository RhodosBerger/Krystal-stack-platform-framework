# ==============================================================================
# KRYSTAL-STACK: GODOT 4.X AUGMENTED REALITY SOCKET CLIENT
# ==============================================================================
# Script: res://scripts/AugmentedSocketClient.gd
# Description: Real-time Godot 4.x client connecting via StreamPeerTCP or
#              WebSocketPeer to the Krystal Augmented Spatial Socket Engine.
#              Synchronizes multi-scale physical environments, continuous
#              spatial fields, and emergent stochastic items.
#
# Platform Invariant: VITAL_MAX_HP == 6
# Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
# ==============================================================================

class_name AugmentedSocketClient
extends Node3D

signal state_updated(state_data: Dictionary)
signal item_transmuted(item_id: String, transmutation_type: String)
signal connection_status_changed(connected: bool)

@export var server_host: String = "127.0.0.1"
@export var server_port: int = 8089
@export var auto_reconnect: bool = true
@export var reconnect_interval_s: float = 2.0
@export var enable_field_debug_gizmos: bool = true

const VITAL_MAX_HP: int = 6

var _tcp_client: StreamPeerTCP = StreamPeerTCP.new()
var _is_connected: bool = false
var _reconnect_timer: float = 0.0
var _buffer: String = ""

# Cached state
var current_anchor: Dictionary = {}
var active_items: Dictionary = {}
var spatial_fields: Dictionary = {}
var item_nodes: Dictionary = {}

func _ready() -> void:
	print("[KASA GODOT] Initializing Augmented Socket Client for Godot 4.x...")
	_connect_to_server()

func _connect_to_server() -> void:
	print("[KASA GODOT] Connecting to %s:%d..." % [server_host, server_port])
	var err = _tcp_client.connect_to_host(server_host, server_port)
	if err != OK:
		push_warning("[KASA GODOT] Initial TCP connection failed: %d" % err)

func _physics_process(delta: float) -> void:
	_tcp_client.poll()
	var status = _tcp_client.get_status()

	if status == StreamPeerTCP.STATUS_CONNECTED:
		if not _is_connected:
			_is_connected = true
			emit_signal("connection_status_changed", true)
			print("[KASA GODOT] Socket connected successfully.")

		# Read available bytes
		var bytes_avail = _tcp_client.get_available_bytes()
		if bytes_avail > 0:
			var data = _tcp_client.get_utf8_string(bytes_avail)
			_buffer += data
			_process_buffer()

	elif status == StreamPeerTCP.STATUS_ERROR or status == StreamPeerTCP.STATUS_NONE:
		if _is_connected:
			_is_connected = false
			emit_signal("connection_status_changed", false)
			push_warning("[KASA GODOT] Connection lost.")

		if auto_reconnect:
			_reconnect_timer += delta
			if _reconnect_timer >= reconnect_interval_s:
				_reconnect_timer = 0.0
				_connect_to_server()

func _process_buffer() -> void:
	var lines = _buffer.split("\n")
	# Keep the last unterminated fragment in the buffer
	_buffer = lines[-1]
	for i in range(lines.size() - 1):
		var line = lines[i].strip_edges()
		if line.length() > 0:
			_parse_packet(line)

func _parse_packet(json_str: String) -> void:
	var json = JSON.new()
	var parse_result = json.parse(json_str)
	if parse_result != OK:
		return

	var data = json.get_data()
	if not (data is Dictionary):
		return

	# Invariant verification
	if data.has("vital_max_hp") and int(data["vital_max_hp"]) != VITAL_MAX_HP:
		push_error("[KASA GODOT] Platform invariant breach: VITAL_MAX_HP must equal 6!")

	if data.has("environment_anchor"):
		current_anchor = data["environment_anchor"]

	if data.has("spatial_fields"):
		spatial_fields = data["spatial_fields"]

	if data.has("items"):
		_sync_items(data["items"])

	emit_signal("state_updated", data)

func _sync_items(items_array: Array) -> void:
	var seen_ids = {}
	for item_data in items_array:
		var iid = str(item_data.get("item_id", ""))
		seen_ids[iid] = true
		active_items[iid] = item_data

		var pos = item_data.get("position", [0.0, 0.0, 0.0])
		var target_pos = Vector3(float(pos[0]), float(pos[1]), float(pos[2]))
		var entropy = float(item_data.get("entropy", 0.0))
		var phase = str(item_data.get("phase_state", "STABLE"))
		var transmuted = bool(item_data.get("transmuted", false))

		if not item_nodes.has(iid):
			var node = _create_item_visual_node(item_data)
			add_child(node)
			item_nodes[iid] = node

		var node = item_nodes[iid]
		node.position = target_pos

		# Update visual modulation based on entropy and phase
		if node.has_node("MeshInstance3D"):
			var mesh_inst: MeshInstance3D = node.get_node("MeshInstance3D")
			var mat = mesh_inst.get_surface_override_material(0)
			if mat is StandardMaterial3D:
				if phase == "SUPERCRITICAL":
					mat.albedo_color = Color(1.0, 0.2, 0.1, 1.0)
					mat.emission_enabled = true
					mat.emission = Color(1.0, 0.5, 0.0)
				elif phase == "TRANSMUTING":
					mat.albedo_color = Color(0.4, 0.9, 1.0, 1.0)
					mat.emission_enabled = true
					mat.emission = Color(0.2, 0.8, 1.0)
				elif phase == "VOLATILE":
					mat.albedo_color = Color(0.9, 0.8, 0.1, 1.0)
				else:
					mat.albedo_color = Color(0.2, 0.6, 0.9, 1.0)

		if transmuted and not item_data.get("_transmute_emitted", false):
			item_data["_transmute_emitted"] = true
			var t_type = str(item_data.get("transmutation_type", "UNKNOWN"))
			emit_signal("item_transmuted", iid, t_type)

	# Clean up despawned items
	for iid in item_nodes.keys():
		if not seen_ids.has(iid):
			var dead_node = item_nodes[iid]
			dead_node.queue_free()
			item_nodes.erase(iid)

func _create_item_visual_node(item_data: Dictionary) -> Node3D:
	var root = Node3D.new()
	root.name = str(item_data.get("item_id", "ItemNode"))

	var mesh_inst = MeshInstance3D.new()
	mesh_inst.name = "MeshInstance3D"

	var itype = str(item_data.get("item_type", "GENERIC"))
	if itype == "VOLATILE_PHASE_SHELL":
		var sphere = SphereMesh.new()
		sphere.radius = 0.35
		sphere.height = 0.70
		mesh_inst.mesh = sphere
	elif itype == "ALCHEMICAL_RESONATOR":
		var prism = BoxMesh.new()
		prism.size = Vector3(0.6, 0.9, 0.6)
		mesh_inst.mesh = prism
	elif itype == "GRAVITATIONAL_MIRROR_LENS":
		var cylinder = CylinderMesh.new()
		cylinder.top_radius = 0.5
		cylinder.bottom_radius = 0.5
		cylinder.height = 0.1
		mesh_inst.mesh = cylinder
	else:
		var poly = BoxMesh.new()
		poly.size = Vector3(0.5, 0.5, 0.5)
		mesh_inst.mesh = poly

	var mat = StandardMaterial3D.new()
	mat.albedo_color = Color(0.2, 0.6, 0.9, 1.0)
	mat.metallic = 0.85
	mat.roughness = 0.25
	mesh_inst.set_surface_override_material(0, mat)

	root.add_child(mesh_inst)
	return root

func send_deploy_item(item_type: String, position: Vector3, velocity: Vector3) -> void:
	if not _is_connected:
		return
	var payload = {
		"action": "DEPLOY_ITEM",
		"item_type": item_type,
		"position": [position.x, position.y, position.z],
		"velocity": [velocity.x, velocity.y, velocity.z],
		"vital_max_hp": VITAL_MAX_HP
	}
	var msg = JSON.stringify(payload) + "\n"
	_tcp_client.put_utf8_string(msg)
