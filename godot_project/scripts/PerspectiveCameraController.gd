# PerspectiveCameraController.gd
# Advanced 3D Perspective Camera Controller for Godot 4.x
# Part of Krystal-Stack Platform Framework // Holographic & Tactical Engine
# Invariant: VITAL_MAX_HP = 6

class_name PerspectiveCameraController
extends Node3D

# --- Signals ---
signal target_acquired(target: Node3D)
signal fov_changed(new_fov: float)
signal trauma_updated(trauma_level: float)

# --- Configuration Exports ---
@export_group("Target Tracking")
@export var follow_target: Node3D = null
@export var follow_smooth_speed: float = 8.0
@export var target_offset: Vector3 = Vector3(0.0, 1.5, 0.0)

@export_group("Orbit & Rotation")
@export var mouse_sensitivity: float = 0.003
@export var min_pitch_deg: float = -80.0
@export var max_pitch_deg: float = 75.0
@export var smooth_orbit_damping: float = 12.0

@export_group("Distance & Zoom")
@export var default_distance: float = 6.0
@export var min_distance: float = 1.5
@export var max_distance: float = 24.0
@export var zoom_speed: float = 1.2
@export var smooth_zoom_speed: float = 10.0

@export_group("FOV & Perspective")
@export var default_fov: float = 75.0
@export var min_fov: float = 35.0
@export var max_fov: float = 95.0

@export_group("Camera Trauma & Shake")
@export var trauma_decay_rate: float = 1.2
@export var max_shake_pitch: float = 4.0
@export var max_shake_yaw: float = 4.0
@export var max_shake_roll: float = 2.5

# --- Internal Nodes ---
@onready var yaw_pivot: Node3D = self
@onready var pitch_pivot: Node3D = $PitchPivot if has_node("PitchPivot") else self
@onready var spring_arm: SpringArm3D = $PitchPivot/SpringArm3D if has_node("PitchPivot/SpringArm3D") else null
@onready var camera: Camera3D = $PitchPivot/SpringArm3D/Camera3D if has_node("PitchPivot/SpringArm3D/Camera3D") else ($Camera3D if has_node("Camera3D") else null)

# --- State Variables ---
var _target_yaw: float = 0.0
var _target_pitch: float = -15.0
var _current_distance: float = 6.0
var _target_distance: float = 6.0
var _target_fov: float = 75.0
var _trauma: float = 0.0
var _shake_time: float = 0.0
var _noise: FastNoiseLite = null

const VITAL_MAX_HP: int = 6

func _ready() -> void:
    _target_distance = default_distance
    _current_distance = default_distance
    _target_fov = default_fov
    
    # Initialize procedural noise for camera shake
    _noise = FastNoiseLite.new()
    _noise.seed = 1337
    _noise.frequency = 18.0
    _noise.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
    
    # Configure Camera3D if present
    if camera:
        camera.projection = Camera3D.PROJECTION_PERSPECTIVE
        camera.fov = default_fov
        
    if spring_arm:
        spring_arm.spring_length = _target_distance

func _unhandled_input(event: InputEvent) -> void:
    # Right-click drag orbit
    if event is InputEventMouseMotion and Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT):
        var mouse_delta: Vector2 = event.relative
        _target_yaw -= mouse_delta.x * mouse_sensitivity
        _target_pitch -= mouse_delta.y * mouse_sensitivity
        _target_pitch = clamp(_target_pitch, deg_to_rad(min_pitch_deg), deg_to_rad(max_pitch_deg))
        
    # Scroll wheel smooth zoom
    if event is InputEventMouseButton:
        if event.button_index == MOUSE_BUTTON_WHEEL_UP and event.pressed:
            _target_distance = max(_target_distance - zoom_speed, min_distance)
            _target_fov = max(_target_fov - 2.0, min_fov)
        elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN and event.pressed:
            _target_distance = min(_target_distance + zoom_speed, max_distance)
            _target_fov = min(_target_fov + 2.0, max_fov)

func _process(delta: float) -> void:
    # 1. Smoothly follow target position
    if follow_target and is_instance_valid(follow_target):
        var dest_pos: Vector3 = follow_target.global_position + target_offset
        global_position = global_position.lerp(dest_pos, follow_smooth_speed * delta)
        
    # 2. Smoothly interpolate yaw & pitch rotations
    rotation.y = lerp_angle(rotation.y, _target_yaw, smooth_orbit_damping * delta)
    if pitch_pivot and pitch_pivot != self:
        pitch_pivot.rotation.x = lerp_angle(pitch_pivot.rotation.x, _target_pitch, smooth_orbit_damping * delta)
    else:
        rotation.x = lerp_angle(rotation.x, _target_pitch, smooth_orbit_damping * delta)
        
    # 3. Smooth distance & SpringArm update
    _current_distance = lerp(_current_distance, _target_distance, smooth_zoom_speed * delta)
    if spring_arm:
        spring_arm.spring_length = _current_distance
    elif camera:
        camera.position.z = _current_distance
        
    # 4. Smooth FOV
    if camera:
        camera.fov = lerp(camera.fov, _target_fov, smooth_zoom_speed * delta)
        
    # 5. Decay and apply camera trauma / shake
    _apply_shake(delta)

func add_trauma(amount: float) -> void:
    _trauma = clamp(_trauma + amount, 0.0, 1.0)
    trauma_updated.emit(_trauma)

func _apply_shake(delta: float) -> void:
    if _trauma <= 0.0:
        return
        
    _shake_time += delta * 30.0
    var shake_intensity: float = _trauma * _trauma # Quadratic decay
    
    var yaw_shake: float = deg_to_rad(max_shake_yaw) * shake_intensity * _noise.get_noise_2d(_shake_time, 0.0)
    var pitch_shake: float = deg_to_rad(max_shake_pitch) * shake_intensity * _noise.get_noise_2d(0.0, _shake_time)
    var roll_shake: float = deg_to_rad(max_shake_roll) * shake_intensity * _noise.get_noise_2d(_shake_time, _shake_time)
    
    if camera:
        camera.rotation.y = yaw_shake
        camera.rotation.x = pitch_shake
        camera.rotation.z = roll_shake
        
    _trauma = max(_trauma - trauma_decay_rate * delta, 0.0)

func set_target(new_target: Node3D) -> void:
    follow_target = new_target
    target_acquired.emit(new_target)
