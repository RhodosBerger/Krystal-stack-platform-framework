extends Node

class_name AntigravityThermodynamicEngine

# ====================================================================
# KRYSTAL-STACK // Godot Stack Extension
# Antigravity Thermodynamic Engine & UUID Mesh Node
# ====================================================================
# This script acts as an AutoLoad (Singleton) in Godot.
# It integrates Godot into the Symplectic Hamiltonian phase space.
# It receives UUID mesh instructions from `antigravity_synthesizer.py`
# and responds to the "Brainwave" states (ALPHA, BETA, GAMMA, OMEGA).
# ====================================================================

const MISSION_CONTROL_URL = "http://127.0.0.1:8080"
var http_request_telemetry: HTTPRequest
var http_request_mesh: HTTPRequest

var current_brainwave: String = "ALPHA"
var local_mesh_congestion: float = 0.0
var processed_uuids: Array = []

signal instruction_received(instruction_type, payload)
signal brainwave_shifted(new_phase)

func _ready():
    print("[AntigravityEngine] Initializing Thermodynamic UUID Node...")
    
    # Setup Telemetry HTTP Node
    http_request_telemetry = HTTPRequest.new()
    add_child(http_request_telemetry)
    http_request_telemetry.request_completed.connect(_on_telemetry_completed)
    
    # Setup Mesh HTTP Node
    http_request_mesh = HTTPRequest.new()
    add_child(http_request_mesh)
    http_request_mesh.request_completed.connect(_on_mesh_sync_completed)
    
    # Start the Thermodynamic Cycle
    var timer = Timer.new()
    timer.wait_time = 0.5 # 2 Hz Sync
    timer.autostart = true
    timer.timeout.connect(_thermodynamic_cycle)
    add_child(timer)
    
    # Register this node into the Synthesizer Mesh
    _register_to_mesh()

func _thermodynamic_cycle():
    # 1. Measure Godot's visual entropy and framerate (Momentum & Drag)
    var current_fps = Engine.get_frames_per_second()
    var target_fps = Engine.max_fps if Engine.max_fps > 0 else 60.0
    var lag_factor = clamp(1.0 - (current_fps / target_fps), 0.0, 1.0)
    
    # 2. Local congestion calculation (if packets queue up)
    local_mesh_congestion = clamp(local_mesh_congestion - 0.05, 0.0, 1.0)
    
    # 3. Formulate the state vector to send back to Python (SymplecticCyclicEngine)
    var payload = {
        "visual_entropy": lag_factor,
        "mesh_congestion": local_mesh_congestion,
        "gpu_temp_c": 50.0, # Placeholder, can be read via OS extensions
        "node_id": "Godot-AR-Node"
    }
    
    var headers = ["Content-Type: application/json"]
    var json_payload = JSON.stringify(payload)
    
    if http_request_telemetry.get_http_client_status() == HTTPClient.STATUS_DISCONNECTED:
        http_request_telemetry.request(MISSION_CONTROL_URL + "/api/cyclic/godot_sync", headers, HTTPClient.METHOD_POST, json_payload)

func _on_telemetry_completed(result, response_code, headers, body):
    if response_code == 200:
        var response = JSON.parse_string(body.get_string_from_utf8())
        if response and response.has("brainwave_phase"):
            var new_phase = response["brainwave_phase"]
            if new_phase != current_brainwave:
                current_brainwave = new_phase
                brainwave_shifted.emit(current_brainwave)
                _apply_brainwave_state(current_brainwave)

func _register_to_mesh():
    # We poll the synthesizer to see if there are new instructions (Easter Egg UUID mesh)
    var headers = ["Content-Type: application/json"]
    http_request_mesh.request(MISSION_CONTROL_URL + "/api/synthesizer_sync/poll", headers, HTTPClient.METHOD_GET)

func _on_mesh_sync_completed(result, response_code, headers, body):
    if response_code == 200:
        var data = JSON.parse_string(body.get_string_from_utf8())
        if data and data.has("packet_id"):
            var uuid = data["packet_id"]
            if not processed_uuids.has(uuid):
                processed_uuids.append(uuid)
                _execute_instruction(data.get("instruction", {}))
                
                if processed_uuids.size() > 50:
                    processed_uuids.pop_front() # Keep memory clean
    
    # Re-poll the mesh after a delay
    await get_tree().create_timer(1.0).timeout
    if http_request_mesh.get_http_client_status() == HTTPClient.STATUS_DISCONNECTED:
        _register_to_mesh()
    else:
        # Network congestion
        local_mesh_congestion += 0.2 

func _execute_instruction(instruction: Dictionary):
    print("[AntigravityEngine] Executing Mesh Instruction: ", instruction)
    if instruction.has("pattern"):
        if instruction["pattern"] == "landscape":
            # Procedural generation trigger
            instruction_received.emit("SPAWN_GEOMETRY", instruction)
    
    if instruction.has("script_path"):
        # Simulated Janet plugin execution
        instruction_received.emit("LOAD_PLUGIN", instruction)

func _apply_brainwave_state(phase: String):
    match phase:
        "ALPHA":
            print("[Brainwave] ALPHA: Relaxing raymarching steps, increasing visuals.")
            RenderingServer.global_shader_parameter_set("u_recursion_limit", 32)
        "BETA":
            print("[Brainwave] BETA: Standard interactive flow.")
            RenderingServer.global_shader_parameter_set("u_recursion_limit", 16)
        "GAMMA":
            print("[Brainwave] GAMMA: Hyper focus. Pinning VRAM, reducing fluff.")
            RenderingServer.global_shader_parameter_set("u_recursion_limit", 8)
        "OMEGA":
            print("[Brainwave] OMEGA: Thermodynamic Backpressure. Extreme throttling.")
            RenderingServer.global_shader_parameter_set("u_recursion_limit", 2)
            # Potentially lower 3D resolution scale here to save GPU
            get_viewport().scaling_3d_scale = 0.5
