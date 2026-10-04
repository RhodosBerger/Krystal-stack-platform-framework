# OrthographicCameraController.gd
# Advanced Orthographic / Tactical Isometric Camera Controller for Godot 4.x
# Part of Krystal-Stack Platform Framework // Holographic & Tactical Engine
# Invariant: VITAL_MAX_HP = 6

class_name OrthographicCameraController
extends Node3D

# --- Signals ---
signal ortho_size_changed(new_size: float)
signal arena_bounds_clamped(current_position: Vector3)
signal projection_preset_applied(preset_name: String)

# --- Projection Presets ---
enum ProjectionPreset {
    ISOMETRIC_TRUE,       # -35.264° pitch (arctan(1/sqrt(2))), 45° yaw
    ISOMETRIC_DIMETRIC,   # -30.0° pitch, 45° yaw
    TACTICAL_TOPDOWN,     # -90.0° pitch, 0° yaw
    MILITARY_OBLIQUE,     # -45.0° pitch, 45° yaw
    SIDE_ELEVATION        # 0.0° pitch, 90° yaw
}

# --- Configuration Exports ---
@export_group("Projection & Presets")
@export var default_preset: ProjectionPreset = ProjectionPreset.ISOMETRIC_TRUE
@export var default_ortho_size: float = 12.0
@export var min_ortho_size: float = 3.0
@export var max_ortho_size: float = 45.0
@export var zoom_step: float = 1.5
@export var smooth_zoom_speed: float = 8.0

@export_group("Pan & Navigation")
@export var pan_speed: float = 18.0
@export var smooth_pan_speed: float = 10.0
@export var edge_pan_margin_px: int = 15
@export var edge_pan_enabled: bool = true
@export var middle_click_drag_enabled: bool = true
@export var drag_sensitivity: float = 0.025

@export_group("Tactical Arena Bounds")
@export var use_arena_bounds: bool = true
@export var arena_min_bounds: Vector2 = Vector2(-50.0, -50.0)
@export var arena_max_bounds: Vector2 = Vector2(50.0, 50.0)

@export_group("Grid Snapping")
@export var snap_to_tactical_grid: bool = false
@export var grid_step_size: float = 1.0

# --- Internal Nodes ---
@onready var yaw_pivot: Node3D = self
@onready var pitch_pivot: Node3D = $PitchPivot if has_node("PitchPivot") else self
@onready var camera: Camera3D = $PitchPivot/Camera3D if has_node("PitchPivot/Camera3D") else ($Camera3D if has_node("Camera3D") else null)

# --- State Variables ---
var _target_ortho_size: float = 12.0
var _current_ortho_size: float = 12.0
var _target_focal_point: Vector3 = Vector3.ZERO
var _current_velocity: Vector3 = Vector3.ZERO
var _is_dragging: bool = false
var _last_mouse_pos: Vector2 = Vector2.ZERO

const VITAL_MAX_HP: int = 6

func _ready() -> void:
    _target_ortho_size = default_ortho_size
    _current_ortho_size = default_ortho_size
    _target_focal_point = global_position
    
    # Configure Camera3D for Orthographic projection
    if camera:
        camera.projection = Camera3D.PROJECTION_ORTHOGONAL
        camera.size = default_ortho_size
        
    apply_preset(default_preset)

func apply_preset(preset: ProjectionPreset) -> void:
    match preset:
        ProjectionPreset.ISOMETRIC_TRUE:
            # Mathematical true isometric angle: asin(tan(30°)) = 35.264°
            rotation_degrees.y = 45.0
            if pitch_pivot and pitch_pivot != self:
                pitch_pivot.rotation_degrees.x = -35.264
            else:
                rotation_degrees.x = -35.264
            projection_preset_applied.emit("ISOMETRIC_TRUE")
            
        ProjectionPreset.ISOMETRIC_DIMETRIC:
            rotation_degrees.y = 45.0
            if pitch_pivot and pitch_pivot != self:
                pitch_pivot.rotation_degrees.x = -30.0
            else:
                rotation_degrees.x = -30.0
            projection_preset_applied.emit("ISOMETRIC_DIMETRIC")
            
        ProjectionPreset.TACTICAL_TOPDOWN:
            rotation_degrees.y = 0.0
            if pitch_pivot and pitch_pivot != self:
                pitch_pivot.rotation_degrees.x = -89.9
            else:
                rotation_degrees.x = -89.9
            projection_preset_applied.emit("TACTICAL_TOPDOWN")
            
        ProjectionPreset.MILITARY_OBLIQUE:
            rotation_degrees.y = 45.0
            if pitch_pivot and pitch_pivot != self:
                pitch_pivot.rotation_degrees.x = -45.0
            else:
                rotation_degrees.x = -45.0
            projection_preset_applied.emit("MILITARY_OBLIQUE")
            
        ProjectionPreset.SIDE_ELEVATION:
            rotation_degrees.y = 90.0
            if pitch_pivot and pitch_pivot != self:
                pitch_pivot.rotation_degrees.x = 0.0
            else:
                rotation_degrees.x = 0.0
            projection_preset_applied.emit("SIDE_ELEVATION")

func _unhandled_input(event: InputEvent) -> void:
    # Mouse drag panning (Middle Click or Alt+Left Click)
    if middle_click_drag_enabled:
        if event is InputEventMouseButton:
            if event.button_index == MOUSE_BUTTON_MIDDLE:
                _is_dragging = event.pressed
                _last_mouse_pos = event.position
        elif event is InputEventMouseMotion and _is_dragging:
            var delta_mouse: Vector2 = event.position - _last_mouse_pos
            _last_mouse_pos = event.position
            
            # Transform screen motion into world ground plane vector relative to camera yaw
            var right_vec: Vector3 = transform.basis.x.normalized()
            var forward_vec: Vector3 = -transform.basis.z.normalized()
            forward_vec.y = 0.0
            forward_vec = forward_vec.normalized()
            
            var drag_factor: float = (_current_ortho_size / default_ortho_size) * drag_sensitivity
            _target_focal_point -= (right_vec * delta_mouse.x - forward_vec * delta_mouse.y) * drag_factor
            
    # Scroll wheel smooth orthographic zoom
    if event is InputEventMouseButton:
        if event.button_index == MOUSE_BUTTON_WHEEL_UP and event.pressed:
            _target_ortho_size = max(_target_ortho_size - zoom_step, min_ortho_size)
            ortho_size_changed.emit(_target_ortho_size)
        elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN and event.pressed:
            _target_ortho_size = min(_target_ortho_size + zoom_step, max_ortho_size)
            ortho_size_changed.emit(_target_ortho_size)

func _process(delta: float) -> void:
    # 1. Keyboard / Edge Panning
    var move_dir: Vector3 = Vector3.ZERO
    
    if Input.is_key_pressed(KEY_W) or Input.is_key_pressed(KEY_UP):
        move_dir.z -= 1.0
    if Input.is_key_pressed(KEY_S) or Input.is_key_pressed(KEY_DOWN):
        move_dir.z += 1.0
    if Input.is_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT):
        move_dir.x -= 1.0
    if Input.is_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT):
        move_dir.x += 1.0
        
    # Optional screen edge pan
    if edge_pan_enabled and not _is_dragging:
        var viewport: Viewport = get_viewport()
        if viewport:
            var m_pos: Vector2 = viewport.get_mouse_position()
            var v_size: Vector2 = viewport.get_visible_rect().size
            if m_pos.x <= edge_pan_margin_px:
                move_dir.x -= 1.0
            elif m_pos.x >= v_size.x - edge_pan_margin_px:
                move_dir.x += 1.0
            if m_pos.y <= edge_pan_margin_px:
                move_dir.z -= 1.0
            elif m_pos.y >= v_size.y - edge_pan_margin_px:
                move_dir.z += 1.0
                
    if move_dir != Vector3.ZERO:
        move_dir = move_dir.normalized()
        # Rotate movement vector to match camera yaw
        var rot_rad: float = rotation.y
        var world_dir: Vector3 = Vector3(
            move_dir.x * cos(rot_rad) + move_dir.z * sin(rot_rad),
            0.0,
            -move_dir.x * sin(rot_rad) + move_dir.z * cos(rot_rad)
        )
        var zoom_scale: float = _current_ortho_size / default_ortho_size
        _target_focal_point += world_dir * pan_speed * zoom_scale * delta
        
    # 2. Arena Bounds Clamping
    if use_arena_bounds:
        _target_focal_point.x = clamp(_target_focal_point.x, arena_min_bounds.x, arena_max_bounds.x)
        _target_focal_point.z = clamp(_target_focal_point.z, arena_min_bounds.y, arena_max_bounds.y)
        arena_bounds_clamped.emit(_target_focal_point)
        
    # 3. Optional Grid Snapping
    var desired_pos: Vector3 = _target_focal_point
    if snap_to_tactical_grid and grid_step_size > 0.0:
        desired_pos.x = round(desired_pos.x / grid_step_size) * grid_step_size
        desired_pos.z = round(desired_pos.z / grid_step_size) * grid_step_size
        
    # 4. Smooth Position Lerp
    global_position = global_position.lerp(desired_pos, smooth_pan_speed * delta)
    
    # 5. Smooth Orthographic Size Lerp
    _current_ortho_size = lerp(_current_ortho_size, _target_ortho_size, smooth_zoom_speed * delta)
    if camera:
        camera.size = _current_ortho_size

func focus_on_coordinate(target_coords: Vector3, instant: bool = false) -> void:
    _target_focal_point = target_coords
    if instant:
        global_position = target_coords
