# GodotPackageManager.gd
# In-Engine Godot 4.x Package & 3D Asset Manager
# Facilitates loading animated glTF/GLB models, scene templates, and camera presets.
# Part of Krystal-Stack Platform Framework // Holographic & Tactical Engine
# Invariant: VITAL_MAX_HP = 6

class_name GodotPackageManager
extends RefCounted

const VITAL_MAX_HP: int = 6
const MANIFEST_PATH: String = "res://assets/models/models_manifest.json"

static func get_installed_models() -> Array[Dictionary]:
    var models: Array[Dictionary] = []
    if not FileAccess.file_exists(MANIFEST_PATH):
        push_warning("GodotPackageManager: Manifest file not found at " + MANIFEST_PATH)
        return models
        
    var file: FileAccess = FileAccess.open(MANIFEST_PATH, FileAccess.READ)
    if not file:
        return models
        
    var json_text: String = file.get_as_text()
    var parsed: Variant = JSON.parse_string(json_text)
    if parsed is Dictionary and parsed.has("models"):
        for m in parsed["models"]:
            models.append(m)
    return models

static func instantiate_animated_model(model_id: String) -> Node3D:
    var installed: Array[Dictionary] = get_installed_models()
    var target_model: Dictionary = {}
    for m in installed:
        if m.get("id") == model_id:
            target_model = m
            break
            
    if target_model.is_empty():
        push_error("GodotPackageManager: Model ID not found: " + model_id)
        return null
        
    var path: String = target_model.get("local_path", "")
    if not ResourceLoader.exists(path):
        push_error("GodotPackageManager: Resource does not exist at path: " + path)
        return null
        
    var packed: PackedScene = load(path)
    if not packed:
        push_error("GodotPackageManager: Failed to load PackedScene from: " + path)
        return null
        
    var instance: Node3D = packed.instantiate() as Node3D
    return instance

static func list_camera_templates() -> Array[String]:
    return [
        "res://scenes/templates/CameraPerspectiveRig.tscn",
        "res://scenes/templates/CameraOrthographicRig.tscn",
        "res://scenes/templates/DualCameraRig3D.tscn"
    ]
