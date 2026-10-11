# ==============================================================================
# KRYSTAL-STACK: FLOATING CITADEL & WELL OF SOULS CONTROLLER
# Compatible with: Godot 4.2+ (Forward+ / Vulkan)
# Manages:
#   1. 3D Floating Island with 3 tribal biomes (Crystal, Toxic, Druid).
#   2. Studna Duší (Well of Souls) aetheric vortex swirl and particles.
#   3. Smooth cinematic camera orbit with zoom and focus modes.
#   4. Live telemetry bridge to Krystal Localhost Hub (8080).
#   5. Invariant: VITAL_MAX_HP = 6.
# ==============================================================================
class_name FloatingCitadelController
extends Node3D

const VITAL_MAX_HP: int = 6

enum TribalBiome {
	OVERVIEW,
	CRYSTAL,
	TOXIC,
	DRUID
}

@export_group("Orbit & Camera Settings")
@export var auto_rotate_island: bool = true
@export var rotation_speed: float = 0.25
@export var current_biome: TribalBiome = TribalBiome.OVERVIEW

@export_group("Node References")
@export var island_root: Node3D
@export var well_vortex: MeshInstance3D
@export var camera_node: Camera3D
@export var crystal_spire_node: Node3D
@export var toxic_pool_node: Node3D
@export var druid_grove_node: Node3D

var _target_cam_pos: Vector3 = Vector3(0, 14, 22)
var _target_cam_look: Vector3 = Vector3(0, 0, 0)

# Signals
signal biome_selected(biome_name: String)

func _ready() -> void:
	print("[FloatingCitadel] Initializing Studna Duší & Tribal Biomes...")
	_setup_camera_presets()

func _process(delta: float) -> void:
	# Subtle floating bobbing and rotation of the central island
	if island_root and auto_rotate_island:
		island_root.rotation.y += delta * rotation_speed
		island_root.position.y = sin(Time.get_ticks_msec() * 0.0012) * 0.45

	# Smooth camera interpolation
	if camera_node:
		camera_node.position = camera_node.position.lerp(_target_cam_pos, delta * 3.5)
		var current_look = -camera_node.transform.basis.z + camera_node.position
		var next_look = current_look.lerp(_target_cam_look, delta * 3.5)
		camera_node.look_at(next_look, Vector3.UP)

func switch_biome(biome: TribalBiome) -> void:
	current_biome = biome
	match biome:
		TribalBiome.OVERVIEW:
			_target_cam_pos = Vector3(0, 14, 22)
			_target_cam_look = Vector3(0, 0, 0)
			biome_selected.emit("OVERVIEW")
		TribalBiome.CRYSTAL:
			# Focus on northern frozen spires
			_target_cam_pos = Vector3(0, 9, 12)
			_target_cam_look = Vector3(0, 3, -6)
			biome_selected.emit("CRYSTAL")
		TribalBiome.TOXIC:
			# Focus on southwest bubbling mire
			_target_cam_pos = Vector3(-9, 7, 10)
			_target_cam_look = Vector3(-6, 0.5, 4)
			biome_selected.emit("TOXIC")
		TribalBiome.DRUID:
			# Focus on southeast ancient tree grove
			_target_cam_pos = Vector3(9, 7, 10)
			_target_cam_look = Vector3(6, 1.2, 4)
			biome_selected.emit("DRUID")

func _setup_camera_presets() -> void:
	switch_biome(TribalBiome.OVERVIEW)
