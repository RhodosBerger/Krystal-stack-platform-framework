# GodotKrystalEditionStage.gd
# ==============================================================================
# KRYSTAL STACK // GODOT KRYSTAL EDITION SHOWCASE CONTROLLER
# ==============================================================================
# Controller for the live 3D demonstration stage of Godot Krystal Edition:
# - Spawns 6 D6 symplectic orbital bodies in phase space.
# - Renders procedural Hex-HSL materials based on 3D sector mapping.
# - Displays live telemetry HUD across layers L0..L5.
# - Triggers test real-time interventions (safe vs quarantined).
#
# Constitutional Invariant: VITAL_MAX_HP = 6
# ==============================================================================

extends Node3D

const VITAL_MAX_HP: int = 6

@onready var hud_label: Label = $HUD/MarginContainer/VBoxContainer/TelemetryLabel
@onready var status_label: Label = $HUD/MarginContainer/VBoxContainer/StatusLabel
@onready var central_prism: MeshInstance3D = $CentralKrystalPrism

var orbital_nodes: Array[MeshInstance3D] = []
var simulation_time: float = 0.0

func _ready() -> void:
	assert(VITAL_MAX_HP == 6, "Constitutional invariant breach: VITAL_MAX_HP != 6")
	print("[SHOWCASE] Initializing Godot Krystal Edition Showcase Stage...")

	# Ensure GodotKrystalEdition singleton or local instance
	var edition = _get_edition()
	if edition:
		# Spawn 6 orbital bodies for D6 hexagonal sectors (0x10..0x60)
		for i in range(VITAL_MAX_HP):
			var angle: float = float(i) * (2.0 * PI / float(VITAL_MAX_HP))
			var radius: float = 3.0
			var pos: Vector3 = Vector3(cos(angle) * radius, 0.5 * sin(angle * 2.0), sin(angle) * radius)
			var vel: Vector3 = Vector3(-sin(angle) * 1.5, cos(angle) * 0.5, cos(angle) * 1.5)

			var body = edition.add_symplectic_orbital_body(pos, vel, 1.0)

			# Create corresponding visual 3D sphere node
			var mesh_inst: MeshInstance3D = MeshInstance3D.new()
			var sphere_mesh: SphereMesh = SphereMesh.new()
			sphere_mesh.radius = 0.25
			sphere_mesh.height = 0.5
			mesh_inst.mesh = sphere_mesh

			var mat: StandardMaterial3D = StandardMaterial3D.new()
			mat.albedo_color = body.hex_hsl_color
			mat.emission_enabled = true
			mat.emission = body.hex_hsl_color
			mat.emission_energy_multiplier = 2.0
			mesh_inst.material_override = mat

			add_child(mesh_inst)
			orbital_nodes.append(mesh_inst)

		status_label.text = "Godot Krystal Edition active. 6 D6 symplectic bodies initialized."

func _process(delta: float) -> void:
	simulation_time += delta
	var edition = _get_edition()
	if not edition:
		return

	# Update visual positions of orbital nodes from symplectic bodies
	for i in range(min(orbital_nodes.size(), edition.symplectic.bodies.size())):
		var body = edition.symplectic.bodies[i]
		var node = orbital_nodes[i]
		node.position = body.position

		var mat: StandardMaterial3D = node.material_override as StandardMaterial3D
		if mat:
			mat.albedo_color = body.hex_hsl_color
			mat.emission = body.hex_hsl_color

	# Rotate central prism
	if central_prism:
		central_prism.rotate_y(delta * 0.5)

	# Update HUD Text
	if hud_label:
		var report: Dictionary = edition.get_full_telemetry_report()
		var l0 = report["layers"]["L0_silicon"]
		var l1 = report["layers"]["L1_compositor"]
		var l2 = report["layers"]["L2_symplectic"]
		var l3 = report["layers"]["L3_governor"]
		var l4 = report["layers"]["L4_contract"]
		var l5 = report["layers"]["L5_gatekeeper"]

		var text: String = ""
		text += "=== GODOT KRYSTAL EDITION // 6-LAYER LIVE TELEMETRY ===\n"
		text += "FPS: %d (Frame: %.2f ms) | Invariant: VITAL_MAX_HP == %d\n\n" % [report["fps"], report["frame_time_ms"], VITAL_MAX_HP]
		text += "[L0 SILICON PDN]   Vdd: %.3f V | Tj: %.1f C | Power: %.1f W | Throttled: %s\n" % [l0["vdd_volts"], l0["tj_celsius"], l0["power_watts"], str(l0["is_throttled"])]
		text += "[L1 CHRONOMETRIC]  Refresh: %.1f Hz | VBlank Shadow: %.2f ms (>= 3.5ms: %s) | UMA: %.2f\n" % [l1["refresh_hz"], l1["vblank_shadow_headroom_ms"], str(l1["is_shadow_compliant"]), l1["uma_contention_ratio"]]
		text += "[L2 SYMPLECTIC]    Bodies: %d | det(J): %.12f | Energy Drift: %.6f\n" % [l2["active_bodies"], l2["det_j_preservation"], l2["max_energy_drift"]]
		text += "[L3 COGNITIVE]     Scale: %.2fx | Dwell: %d/%d frames | Anti-Chatter: %d | Alpha: %.2f\n" % [l3["viewport_scale"], l3["dwell_counter"], l3["min_dwell_required"], l3["chattering_prevented"], l3["alpha_pulse"]]
		text += "[L4 SPSC ZERO-COPY]Frame: %dB | Sent: %d | Recv: %d | Corrupted: %d (Healthy: %s)\n" % [l4["frame_size_bytes"], l4["frames_sent"], l4["frames_received"], l4["corrupted_frames"], str(l4["is_healthy"])]
		text += "[L5 GATEKEEPER]    Authorized: %d | Quarantined/Rejected: %d | Delta S==0 Verified" % [l5["authorized_count"], l5["rejected_count"]]
		hud_label.text = text

func test_safe_intervention() -> void:
	"""Proposes a safe change (e.g. slight PDN impedance tuning)."""
	var edition = _get_edition()
	if edition:
		var receipt = edition.request_safe_intervention("ELEM_PDN_IMPEDANCE", 2.0, "Showcase safe impedance tune")
		status_label.text = "Intervention: %s -> %s" % [receipt["status"], receipt["reason"]]

func test_unsafe_intervention() -> void:
	"""Proposes an unsafe change (e.g. 144Hz refresh truncating VBlank shadow) -> Delta S == 0."""
	var edition = _get_edition()
	if edition:
		var receipt = edition.request_safe_intervention("ELEM_DISPLAY_REFRESH_HZ", 84.0, "Showcase 144Hz breach")
		status_label.text = "Unsafe Intervention: %s -> %s" % [receipt["status"], receipt["reason"]]

var _local_edition_instance: GodotKrystalEdition = null

func _get_edition() -> GodotKrystalEdition:
	if Engine.has_singleton("GodotKrystalEdition"):
		return Engine.get_singleton("GodotKrystalEdition") as GodotKrystalEdition
	# Fallback if running directly without autoload configured
	var existing = get_node_or_null("/root/GodotKrystalEdition")
	if existing:
		return existing as GodotKrystalEdition
	if _local_edition_instance == null:
		_local_edition_instance = GodotKrystalEdition.new()
		add_child(_local_edition_instance)
	return _local_edition_instance

