# DualCameraRig3D.gd
# Unified Perspective & Orthographic Dual-Mode Camera Rig for Godot 4.x
# Enables seamless runtime morphing between 3D Action Perspective and Isometric Tactical Orthographic.
# Part of Krystal-Stack Platform Framework // Holographic & Tactical Engine
# Invariant: VITAL_MAX_HP = 6

class_name DualCameraRig3D
extends Node3D

# --- Signals ---
signal mode_transition_started(target_is_perspective: bool, duration: float)
signal mode_transition_completed(is_perspective: bool)
signal camera_zoomed(current_param: float)

# --- Camera Modes ---
enum CameraMode {
    PERSPECTIVE_ACTION,     # 3rd person action follow / orbital perspective
    ORTHOGRAPHIC_ISOMETRIC, # True isometric tactical grid (arctan(1/sqrt(2)) = 35.264°)
    ORTHOGRAPHIC_TOPDOWN,   # Top-down RTS / hex grid view (90° pitch)
    PERSPECTIVE_CINEMATIC   # Cinematic wide-angle perspective (FOV 85°)
}

# --- Configuration Exports ---
@export_group("Rig Mode")
@export var initial_mode: CameraMode = CameraMode.PERSPECTIVE_ACTION
@export var transition_duration: float = 0.65
@export var transition_ease: Tween.EaseType = Tween.EASE_OUT
@export var transition_trans: Tween.TransitionType = Tween.TRANS_CUBIC

@export_group("Perspective Parameters")
@export var perspective_default_fov: float = 75.0
@export var perspective_distance: float = 8.0
@export var perspective_pitch_deg: float = -20.0
@export var perspective_yaw_deg: float = 0.0

@export_group("Orthographic Parameters")
@export var orthographic_default_size: float = 14.0
@export var isometric_pitch_deg: float = -35.264
@export var isometric_yaw_deg: float = 45.0
@export var topdown_pitch_deg: float = -89.9
@export var topdown_yaw_deg: float = 0.0

@export_group("Input & Hotkeys")
@export var enable_hotkey_switching: bool = true
@export var key_perspective: Key = KEY_1
@export var key_isometric: Key = KEY_2
@export var key_topdown: Key = KEY_3
@export var key_toggle: Key = KEY_TAB

# --- Sub-nodes ---
@onready var pitch_pivot: Node3D = $PitchPivot if has_node("PitchPivot") else self
@onready var camera: Camera3D = $PitchPivot/Camera3D if has_node("PitchPivot/Camera3D") else ($Camera3D if has_node("Camera3D") else null)

# --- State ---
var current_mode: CameraMode = CameraMode.PERSPECTIVE_ACTION
var is_transitioning: bool = false
var _active_tween: Tween = null
const VITAL_MAX_HP: int = 6

func _ready() -> void:
    _setup_camera()
    set_camera_mode(initial_mode, true)

func _setup_camera() -> void:
    if not camera:
        push_warning("DualCameraRig3D: No Camera3D node found! Creating dynamic Camera3D.")
        var pivot: Node3D = Node3D.new()
        pivot.name = "PitchPivot"
        add_child(pivot)
        pitch_pivot = pivot
        
        var cam: Camera3D = Camera3D.new()
        cam.name = "Camera3D"
        cam.current = true
        pivot.add_child(cam)
        camera = cam

func _unhandled_input(event: InputEvent) -> void:
    if not enable_hotkey_switching:
        return
        
    if event is InputEventKey and event.pressed and not event.echo:
        match event.keycode:
            KEY_1:
                switch_to_perspective()
            KEY_2:
                switch_to_isometric()
            KEY_3:
                switch_to_topdown()
            KEY_TAB:
                toggle_projection_mode()

func toggle_projection_mode() -> void:
    if current_mode == CameraMode.PERSPECTIVE_ACTION or current_mode == CameraMode.PERSPECTIVE_CINEMATIC:
        switch_to_isometric()
    else:
        switch_to_perspective()

func switch_to_perspective(instant: bool = false) -> void:
    set_camera_mode(CameraMode.PERSPECTIVE_ACTION, instant)

func switch_to_isometric(instant: bool = false) -> void:
    set_camera_mode(CameraMode.ORTHOGRAPHIC_ISOMETRIC, instant)

func switch_to_topdown(instant: bool = false) -> void:
    set_camera_mode(CameraMode.ORTHOGRAPHIC_TOPDOWN, instant)

func set_camera_mode(new_mode: CameraMode, instant: bool = false) -> void:
    if not camera:
        return
        
    current_mode = new_mode
    var target_is_persp: bool = (new_mode == CameraMode.PERSPECTIVE_ACTION or new_mode == CameraMode.PERSPECTIVE_CINEMATIC)
    
    var dest_yaw: float = 0.0
    var dest_pitch: float = 0.0
    var dest_dist: float = perspective_distance
    var target_fov: float = perspective_default_fov
    var target_size: float = orthographic_default_size
    
    match new_mode:
        CameraMode.PERSPECTIVE_ACTION:
            dest_yaw = deg_to_rad(perspective_yaw_deg)
            dest_pitch = deg_to_rad(perspective_pitch_deg)
            dest_dist = perspective_distance
            target_fov = perspective_default_fov
        CameraMode.PERSPECTIVE_CINEMATIC:
            dest_yaw = deg_to_rad(15.0)
            dest_pitch = deg_to_rad(-12.0)
            dest_dist = perspective_distance * 1.3
            target_fov = 85.0
        CameraMode.ORTHOGRAPHIC_ISOMETRIC:
            dest_yaw = deg_to_rad(isometric_yaw_deg)
            dest_pitch = deg_to_rad(isometric_pitch_deg)
            dest_dist = 20.0 # Distance in ortho does not affect perspective scale
            target_size = orthographic_default_size
        CameraMode.ORTHOGRAPHIC_TOPDOWN:
            dest_yaw = deg_to_rad(topdown_yaw_deg)
            dest_pitch = deg_to_rad(topdown_pitch_deg)
            dest_dist = 25.0
            target_size = orthographic_default_size * 1.2
            
    if instant or transition_duration <= 0.0:
        rotation.y = dest_yaw
        if pitch_pivot and pitch_pivot != self:
            pitch_pivot.rotation.x = dest_pitch
        camera.position.z = dest_dist
        if target_is_persp:
            camera.projection = Camera3D.PROJECTION_PERSPECTIVE
            camera.fov = target_fov
        else:
            camera.projection = Camera3D.PROJECTION_ORTHOGONAL
            camera.size = target_size
        mode_transition_completed.emit(target_is_persp)
        return
        
    # Smooth animated tween between camera configurations
    if _active_tween and _active_tween.is_valid():
        _active_tween.kill()
        
    _active_tween = create_tween().set_parallel(true)
    _active_tween.set_ease(transition_ease).set_trans(transition_trans)
    
    is_transitioning = true
    mode_transition_started.emit(target_is_persp, transition_duration)
    
    _active_tween.tween_property(self, "rotation:y", dest_yaw, transition_duration)
    if pitch_pivot and pitch_pivot != self:
        _active_tween.tween_property(pitch_pivot, "rotation:x", dest_pitch, transition_duration)
        
    _active_tween.tween_property(camera, "position:z", dest_dist, transition_duration)
    
    # Switch projection mid-way to blend sizes seamlessly
    var halfway: float = transition_duration * 0.5
    _active_tween.tween_callback(func():
        if target_is_persp:
            camera.projection = Camera3D.PROJECTION_PERSPECTIVE
            camera.fov = target_fov
        else:
            camera.projection = Camera3D.PROJECTION_ORTHOGONAL
            camera.size = target_size
    ).set_delay(halfway)
    
    _active_tween.finished.connect(func():
        is_transitioning = false
        mode_transition_completed.emit(target_is_persp)
    )

func zoom(delta_amount: float) -> void:
    if not camera:
        return
    if camera.projection == Camera3D.PROJECTION_PERSPECTIVE:
        camera.fov = clamp(camera.fov + delta_amount * 2.0, 30.0, 95.0)
        camera_zoomed.emit(camera.fov)
    else:
        camera.size = clamp(camera.size + delta_amount, 2.0, 45.0)
        camera_zoomed.emit(camera.size)
