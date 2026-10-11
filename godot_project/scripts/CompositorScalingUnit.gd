# ==============================================================================
# KRYSTAL-STACK: AUTONOMOUS COMPOSITOR SCALING UNIT (GODOT 4.X GDSCRIPT)
# ==============================================================================
# File: godot_project/scripts/CompositorScalingUnit.gd
# Description: Autonomous module operating as a scaling unit in the 3D compositor.
#              Directly synthesizes and drives a MultiMeshInstance3D with up to
#              several thousand instances using closed-form mathematical equations
#              (Fermat Phyllotaxis, Gielis Superformula, Harmonic Color Tensors).
#
# Non-negotiable Platform Invariant: VITAL_MAX_HP = 6
# Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
# ==============================================================================

class_name CompositorScalingUnit
extends MultiMeshInstance3D

const VITAL_MAX_HP: int = 6
const GOLDEN_ANGLE_RAD: float = 2.399963229728653  # ~137.507764 degrees
const GOLDEN_RATIO: float = 1.618033988749895

# ── EXPORTED PARAMETERS (HUMAN EDITABLE IN INSPECTOR WITHOUT AI) ─────────────
@export_group("Compositor Scaling Unit")
@export var unit_id: String = "scaling_unit_hex_01"
@export var archetype: String = "HEX_TILE"
@export var instance_count: int = 1024
@export var auto_generate_on_ready: bool = true

@export_group("Mathematical Placement Formula")
@export var c_scale: float = 0.85
@export var elevation_amplitude: float = 1.2
@export var elevation_frequency: float = 0.15

@export_group("Gielis Morphometry")
@export var gielis_m: float = 6.0       # 6=hexagon, 8=crystal, 5=star
@export var gielis_n1: float = 1.0
@export var gielis_n2: float = 1.0
@export var gielis_n3: float = 1.0

@export_group("Harmonic Color Tensor")
@export_range(0.0, 1.0) var base_hue: float = 0.12
@export_range(0.0, 1.0) var secondary_hue: float = 0.58
@export_range(0.0, 1.0) var roughness_base: float = 0.35
@export_range(0.0, 1.0) var metallic_base: float = 0.20
@export var emission_energy: float = 0.0


func _ready() -> void:
	# Read metadata overrides if present
	if has_meta("vital_max_hp"):
		assert(int(get_meta("vital_max_hp")) == VITAL_MAX_HP, "VITAL_MAX_HP invariant breach!")
	if has_meta("instance_count"):
		instance_count = int(get_meta("instance_count"))
	if has_meta("c_scale"):
		c_scale = float(get_meta("c_scale"))
	if has_meta("gielis_m"):
		gielis_m = float(get_meta("gielis_m"))
	if has_meta("base_hue"):
		base_hue = float(get_meta("base_hue"))
	if has_meta("emission"):
		emission_energy = float(get_meta("emission"))

	if auto_generate_on_ready:
		build_multimesh_instances()


# ── AUTONOMOUS MATHEMATICAL GENERATION ────────────────────────────────────────

func build_multimesh_instances() -> void:
	var mm: MultiMesh = multimesh
	if mm == null:
		mm = MultiMesh.new()
		multimesh = mm

	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	mm.use_custom_data = true
	mm.instance_count = instance_count

	# Mesh primitive fallback if not set
	if mm.mesh == null:
		var box := BoxMesh.new()
		box.size = Vector3(1.2, 1.2, 1.2)
		mm.mesh = box

	for k in range(instance_count):
		# 1. Fermat Phyllotaxis & Golden Spiral Placement
		var r: float = c_scale * sqrt(float(k))
		var theta: float = float(k) * GOLDEN_ANGLE_RAD

		var px: float = r * cos(theta)
		var pz: float = r * sin(theta)

		# 2. Gielis Superformula Morphometry Elevation
		var sf_val: float = evaluate_gielis(theta, gielis_m, gielis_n1, gielis_n2, gielis_n3)
		var py: float = sin(r * elevation_frequency) * elevation_amplitude + (sf_val * 0.4)

		# 3. Scale and Orientation
		var scale_mag: float = 0.7 + (sf_val * 0.3)
		var basis_rot := Basis(Vector3.UP, theta)
		var basis_scaled := basis_rot.scaled(Vector3(scale_mag, scale_mag, scale_mag))
		var xform := Transform3D(basis_scaled, Vector3(px, py, pz))

		mm.set_instance_transform(k, xform)

		# 4. Harmonic Color Tensor
		var hue_shift: float = fposmod(base_hue + (r * 0.005), 1.0)
		var col: Color = Color.from_hsv(hue_shift, 0.75, 0.90, 1.0)
		mm.set_instance_color(k, col)

		# 5. Custom Data: [Metallic, Roughness, Emission, Gielis Parameter]
		var custom_data := Color(metallic_base, roughness_base, emission_energy, sf_val)
		mm.set_instance_custom_data(k, custom_data)


func evaluate_gielis(theta: float, m: float, n1: float, n2: float, n3: float) -> float:
	var t: float = m * theta / 4.0
	var part1: float = pow(abs(cos(t)), n2)
	var part2: float = pow(abs(sin(t)), n3)
	var denom: float = part1 + part2
	if denom <= 0.0001:
		return 0.0
	return pow(denom, -1.0 / max(0.001, n1))


# ── LIVE INTERACTION WITHOUT AI ──────────────────────────────────────────────

func set_shape_symmetry(new_m: float) -> void:
	gielis_m = new_m
	build_multimesh_instances()

func set_spatial_spacing(new_c: float) -> void:
	c_scale = new_c
	build_multimesh_instances()

func set_theme_hue(new_hue: float) -> void:
	base_hue = new_hue
	build_multimesh_instances()
