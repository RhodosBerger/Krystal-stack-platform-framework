extends Node
class_name GodotHardwareBridge

# Connects Godot 4.x to Krystal Localhost Hub to push hardware budget uniforms.
@export var hub_url: String = "http://127.0.0.1:8080/api/whisperer/calculator"
@export var update_interval: float = 2.0

var http_req: HTTPRequest
var timer: float = 0.0

func _ready():
	http_req = HTTPRequest.new()
	add_child(http_req)
	http_req.request_completed.connect(_on_budget_received)
	_fetch_budget()

func _process(delta: float):
	timer += delta
	if timer >= update_interval:
		timer = 0.0
		_fetch_budget()

func _fetch_budget():
	http_req.request(hub_url)

func _on_budget_received(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray):
	if response_code != 200:
		return
	var json = JSON.new()
	if json.parse(body.get_string_from_utf8()) == OK:
		var data = json.get_data()
		if data.has("budget_plan"):
			var plan = data["budget_plan"]
			var parent = get_parent()
			if parent and parent.has_node("RaymarchCanvas"):
				var rect: ColorRect = parent.get_node("RaymarchCanvas")
				var mat: ShaderMaterial = rect.material
				if mat:
					mat.set_shader_parameter("u_max_steps", plan.get("max_raymarching_steps", 64))
					mat.set_shader_parameter("u_coxeter_folds", plan.get("coxeter_folds", 6))
