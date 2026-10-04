# ==============================================================================
# KRYSTAL-STACK: MINECRAFT ATMOSPHERE CONTROLLER & SHADER BRIDGE
# Compatible with: Godot 4.2+ (Forward+ / Vulkan)
# Manages: Day/Night Cycle, Sun & Moon Direction, Volumetric Fog, Weather,
#          and Uniform Propagation to LabPBR Terrain and Ocean Shaders.
# Invariant: VITAL_MAX_HP = 6
# ==============================================================================
class_name MinecraftAtmosphereController
extends Node3D

const VITAL_MAX_HP: int = 6
const GOLDEN_RATIO: float = 1.61803398875

@export_group("Time of Day")
@export_range(0.0, 24.0, 0.1) var time_of_day_hours: float = 10.5
@export var day_cycle_speed: float = 0.05 # Hours per real-time second
@export var pause_day_cycle: bool = false

@export_group("Celestial Nodes")
@export var sun_light: DirectionalLight3D
@export var moon_light: DirectionalLight3D
@export var world_environment: WorldEnvironment

@export_group("Shader Materials")
@export var terrain_material: ShaderMaterial
@export var water_material: ShaderMaterial
@export var sky_material: ShaderMaterial

@export_group("Weather Simulation")
@export_enum("Clear", "Overcast", "Rain", "Thunderstorm") var weather_state: int = 0
@export_range(0.0, 1.0) var rain_intensity: float = 0.0

var _current_sun_dir: Vector3 = Vector3(0.5, 0.707, 0.5)

func _ready() -> void:
	_update_celestial_bodies()
	_propagate_shader_uniforms()

func _process(delta: float) -> void:
	if not pause_day_cycle:
		time_of_day_hours = fposmod(time_of_day_hours + day_cycle_speed * delta, 24.0)
		_update_celestial_bodies()
		_propagate_shader_uniforms()

func _update_celestial_bodies() -> void:
	# Convert 24-hour time to celestial angle
	var day_fraction: float = time_of_day_hours / 24.0
	var sun_angle: float = (day_fraction * TAU) - (PI * 0.5)
	
	# Calculate Sun & Moon directional vectors
	var sun_x: float = cos(sun_angle)
	var sun_y: float = sin(sun_angle)
	_current_sun_dir = Vector3(sun_x, sun_y, 0.35).normalized()
	var moon_dir: Vector3 = -_current_sun_dir
	
	# Rotate Sun DirectionalLight3D
	if sun_light:
		sun_light.look_at_from_position(Vector3.ZERO, -_current_sun_dir, Vector3.UP)
		var sun_height: float = clamp(_current_sun_dir.y, 0.0, 1.0)
		sun_light.light_energy = lerp(0.0, 1.8, sun_height)
		sun_light.visible = _current_sun_dir.y > -0.05
		
		# Golden hour / Sunset color transition
		if sun_height < 0.25 and sun_height > 0.0:
			var sunset_t: float = sun_height / 0.25
			sun_light.light_color = Color(1.0, 0.55, 0.25).lerp(Color(1.0, 0.98, 0.92), sunset_t)
		else:
			sun_light.light_color = Color(1.0, 0.98, 0.92)

	# Rotate Moon DirectionalLight3D
	if moon_light:
		moon_light.look_at_from_position(Vector3.ZERO, -moon_dir, Vector3.UP)
		var moon_height: float = clamp(moon_dir.y, 0.0, 1.0)
		moon_light.light_energy = lerp(0.0, 0.28, moon_height)
		moon_light.visible = moon_dir.y > -0.05

func _propagate_shader_uniforms() -> void:
	# 1. Update Sky Material
	if sky_material:
		sky_material.set_shader_parameter("sun_direction", _current_sun_dir)
		sky_material.set_shader_parameter("moon_direction", -_current_sun_dir)
		
		# Cloud coverage based on weather
		var target_coverage: float = 0.45
		match weather_state:
			1: target_coverage = 0.75 # Overcast
			2: target_coverage = 0.90 # Rain
			3: target_coverage = 0.98 # Thunderstorm
		sky_material.set_shader_parameter("cloud_coverage", target_coverage)
	
	# 2. Update Terrain Material (LabPBR wetness & puddles)
	if terrain_material:
		var wetness: float = 0.0
		if weather_state >= 2:
			wetness = clamp(rain_intensity, 0.2, 1.0)
		terrain_material.set_shader_parameter("wetness_intensity", wetness)
	
	# 3. Update Volumetric Fog in WorldEnvironment
	if world_environment and world_environment.environment:
		var env: Environment = world_environment.environment
		if env.volumetric_fog_enabled:
			var fog_density: float = 0.012
			if weather_state == 2:
				fog_density = 0.035
			elif weather_state == 3:
				fog_density = 0.065
			env.volumetric_fog_density = fog_density
