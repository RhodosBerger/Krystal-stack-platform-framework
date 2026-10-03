extends Node3D

class_name SymposiaGenerator

# ====================================================================
# KRYSTAL-STACK // Godot Symposia Machine Learning Generator
# ====================================================================
# Procedurally generates landscapes and "zatišie" (still life) using 
# highly efficient MultiMeshInstance3D techniques.
# It receives structural patterns from the ML Bot (via the Thermodynamic Engine)
# ====================================================================

@export var base_mesh: Mesh
@export var accent_mesh: Mesh

var multi_mesh_instance: MultiMeshInstance3D
var accent_multi_mesh: MultiMeshInstance3D

func _ready():
    print("[SymposiaGenerator] Initializing sophisticated ML-based procedural renderer...")
    
    # 1. Prepare highly optimized MultiMesh for thousands of instances in a single draw call
    multi_mesh_instance = MultiMeshInstance3D.new()
    var multi_mesh = MultiMesh.new()
    multi_mesh.transform_format = MultiMesh.TRANSFORM_3D
    multi_mesh.mesh = base_mesh if base_mesh else BoxMesh.new()
    multi_mesh_instance.multimesh = multi_mesh
    add_child(multi_mesh_instance)
    
    accent_multi_mesh = MultiMeshInstance3D.new()
    var a_mesh = MultiMesh.new()
    a_mesh.transform_format = MultiMesh.TRANSFORM_3D
    a_mesh.mesh = accent_mesh if accent_mesh else SphereMesh.new()
    accent_multi_mesh.multimesh = a_mesh
    add_child(accent_multi_mesh)

    # 2. Hook into the Antigravity Thermodynamic Engine UUID network
    var engine = get_node_or_null("/root/AntigravityThermodynamicEngine")
    if engine:
        engine.instruction_received.connect(_on_bot_instruction)
    else:
        push_warning("[SymposiaGenerator] AntigravityThermodynamicEngine not found. Awaiting manual trigger.")

func _on_bot_instruction(instruction_type: String, payload: Dictionary):
    if instruction_type == "SPAWN_GEOMETRY" or instruction_type == "SYMPOSIA_PATTERN":
        print("[SymposiaGenerator] Machine Learning Bot pattern received. Generating Symposia...")
        _generate_procedural_symposia(payload)

func _generate_procedural_symposia(pattern_data: Dictionary):
    var instance_count = pattern_data.get("complexity", 1000)
    var symmetry_order = pattern_data.get("symmetry", 6) # e.g. D6 symmetry
    var dispersion = pattern_data.get("dispersion", 50.0)
    
    multi_mesh_instance.multimesh.instance_count = instance_count
    accent_multi_mesh.multimesh.instance_count = int(instance_count * 0.1) # 10% accents
    
    var time_seed = Time.get_ticks_msec()
    
    for i in range(instance_count):
        # Generate algorithmic positions mimicking nature (Fibonacci spiral / Vogel's model)
        var t = float(i) / float(instance_count)
        var golden_angle = PI * (3.0 - sqrt(5.0))
        var theta = i * golden_angle
        var radius = sqrt(t) * dispersion
        
        var x = radius * cos(theta)
        var z = radius * sin(theta)
        var y = sin(radius * 0.5 + time_seed) * 2.0 # Procedural ML noise curve estimation
        
        # Apply symmetry folding if requested by the bot
        if symmetry_order > 0:
            var angle = atan2(z, x)
            var sector = PI / symmetry_order
            angle = abs(fmod(angle, 2.0 * sector)) - sector
            x = radius * cos(angle)
            z = radius * sin(angle)

        var tform = Transform3D()
        tform = tform.translated(Vector3(x, y, z))
        
        # Look at center and scale down towards edges
        tform = tform.looking_at(Vector3.ZERO, Vector3.UP)
        var s = (1.0 - t) * 2.0
        tform = tform.scaled_local(Vector3(s, s, s))
        
        multi_mesh_instance.multimesh.set_instance_transform(i, tform)
        
        # Populate accents
        if i < accent_multi_mesh.multimesh.instance_count:
            var a_tform = tform.translated(Vector3(0, 2.0, 0)) # Float above
            a_tform = a_tform.scaled_local(Vector3(0.5, 0.5, 0.5))
            accent_multi_mesh.multimesh.set_instance_transform(i, a_tform)

    print("[SymposiaGenerator] Symposia landscape fully generated with ", instance_count, " instances.")
