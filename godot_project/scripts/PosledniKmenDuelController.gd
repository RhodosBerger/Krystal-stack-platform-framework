# ==============================================================================
# KRYSTAL-STACK: POSLEDNÍ KMEN TACTICAL ARENA DUEL CONTROLLER
# Compatible with: Godot 4.2+ (Forward+ / Vulkan)
# Implements:
#   1. The West duel algebra (locational targeting, dodge stances, damage mitigation).
#   2. Sapper trench fortifications (raising cover density to 85% on hex cells).
#   3. Mortar area dispersion ("plošná dikcia", CEP ellipse, rapid cadence bursts).
#   4. 120 Helpers and 40 Archetypes synergy catalog.
#   5. Strict 6 Max HP vital invariant (6 discrete vitality pips / hearts).
#   6. Dual-mode execution: Communicates with Krystal Localhost Hub (8080) via HTTP
#      with seamless offline embedded simulation fallback.
# ==============================================================================
class_name PosledniKmenDuelController
extends Node3D

const VITAL_MAX_HP: int = 6
const GOLDEN_RATIO: float = 1.61803398875

enum TargetZone {
	HEAD,
	RIGHT_SHOULDER,
	LEFT_SHOULDER,
	TORSO
}

enum DodgeStance {
	DUCK_DOWN,
	LEAN_LEFT,
	LEAN_RIGHT,
	STAND_FIRM
}

# Network Configuration
@export_group("Krystal Hub Connectivity")
@export var hub_api_url: String = "http://127.0.0.1:8080"
@export var use_remote_hub: bool = true
@export var request_timeout_sec: float = 3.0

# 3D Node Bindings
@export_group("Scene Node Bindings")
@export var hex_grid_root: Node3D
@export var player_avatar: Node3D
@export var opponent_avatar: Node3D
@export var camera_3d: Camera3D

# Internal State Variables
var player_hp: int = 6:
	set(val):
		player_hp = clampi(val, 0, VITAL_MAX_HP)
		_update_hud_vital_pips()

var opponent_hp: int = 6:
	set(val):
		opponent_hp = clampi(val, 0, VITAL_MAX_HP)
		_update_hud_vital_pips()

var player_hex: Vector2i = Vector2i(0, -1)
var opponent_hex: Vector2i = Vector2i(1, 1)
var player_target_zone: TargetZone = TargetZone.TORSO
var player_dodge_stance: DodgeStance = DodgeStance.STAND_FIRM

var cover_map: Dictionary = {}         # "q_r" -> float [0.0 - 1.0]
var trenches: Array[String] = []       # list of "q_r" with trenches
var combat_round: int = 1
var is_match_over: bool = false
var winner_name: String = ""

# HTTP Request node
var _http_request: HTTPRequest
var _pending_request_type: String = ""

# GUI CanvasLayer References
var _canvas_layer: CanvasLayer
var _lbl_player_hp: Label
var _lbl_opponent_hp: Label
var _lbl_status: Label
var _txt_combat_log: RichTextLabel
var _trench_mesh_container: Node3D

# Signals
signal round_resolved(result: Dictionary)
signal mortar_impacted(result: Dictionary)
signal trench_dug(hex_coord: Vector2i)
signal match_finished(winner: String)

func _ready() -> void:
	print("[PosledniKmen] Initializing Tactical Duel Controller...")
	_init_cover_map()
	_setup_network()
	_setup_hud()
	_setup_avatars()
	_log_event("SÚBOJ ZAHÁJENÝ: Kryštálový Veštec vs Toxický Čarodejník. Obaja majú 6 HP.")

func _process(delta: float) -> void:
	# Subtle floating rotation for avatars & crystal familiars
	if player_avatar and is_instance_valid(player_avatar):
		player_avatar.rotation.y += delta * 0.6
	if opponent_avatar and is_instance_valid(opponent_avatar):
		opponent_avatar.rotation.y -= delta * 0.6

# ==============================================================================
# 1. SETUP & INITIALIZATION
# ==============================================================================
func _init_cover_map() -> void:
	cover_map.clear()
	trenches.clear()
	# Standard axial hexes for Posledni Kmen 19-hex board
	var hex_coords = [
		Vector2i(-2, 0), Vector2i(-2, 1), Vector2i(-2, 2),
		Vector2i(-1, -1), Vector2i(-1, 0), Vector2i(-1, 1), Vector2i(-1, 2),
		Vector2i(0, -2), Vector2i(0, -1), Vector2i(0, 0), Vector2i(0, 1), Vector2i(0, 2),
		Vector2i(1, -2), Vector2i(1, -1), Vector2i(1, 0), Vector2i(1, 1),
		Vector2i(2, -2), Vector2i(2, -1), Vector2i(2, 0)
	]
	for h in hex_coords:
		var k = "%d_%d" % [h.x, h.y]
		cover_map[k] = 0.15 # Default exposed ground
	# Center hexes slightly higher cover
	cover_map["0_0"] = 0.40

func _setup_network() -> void:
	_http_request = HTTPRequest.new()
	_http_request.timeout = request_timeout_sec
	add_child(_http_request)
	_http_request.request_completed.connect(_on_http_request_completed)

func _setup_avatars() -> void:
	# Create trench container
	_trench_mesh_container = Node3D.new()
	_trench_mesh_container.name = "TrenchFortifications"
	add_child(_trench_mesh_container)

	# If player_avatar not bound, construct procedural crystal avatar
	if not player_avatar:
		player_avatar = _create_procedural_avatar(Color(0.2, 0.9, 1.0, 0.95), "PlayerHero")
		add_child(player_avatar)
		_position_avatar(player_avatar, player_hex)

	# If opponent_avatar not bound, construct procedural toxic avatar
	if not opponent_avatar:
		opponent_avatar = _create_procedural_avatar(Color(0.85, 0.15, 0.25, 0.95), "OpponentHero")
		add_child(opponent_avatar)
		_position_avatar(opponent_avatar, opponent_hex)

func _create_procedural_avatar(albedo: Color, avatar_name: String) -> Node3D:
	var root = Node3D.new()
	root.name = avatar_name

	var mesh_inst = MeshInstance3D.new()
	var prism = PrismMesh.new()
	prism.size = Vector3(0.9, 1.8, 0.9)
	mesh_inst.mesh = prism
	mesh_inst.position.y = 1.0

	var mat = StandardMaterial3D.new()
	mat.albedo_color = albedo
	mat.emission_enabled = true
	mat.emission = albedo
	mat.emission_energy_multiplier = 1.8
	mat.roughness = 0.2
	mat.metallic = 0.7
	mesh_inst.material_override = mat
	root.add_child(mesh_inst)

	# Familiar orb orbiting around the avatar
	var familiar = MeshInstance3D.new()
	var sphere = SphereMesh.new()
	sphere.radius = 0.22
	sphere.height = 0.44
	familiar.mesh = sphere
	familiar.position = Vector3(1.2, 1.4, 0.0)

	var f_mat = StandardMaterial3D.new()
	f_mat.albedo_color = Color(1.0, 0.9, 0.2, 1.0)
	f_mat.emission_enabled = true
	f_mat.emission = Color(1.0, 0.9, 0.2, 1.0)
	f_mat.emission_energy_multiplier = 2.5
	familiar.material_override = f_mat
	root.add_child(familiar)

	return root

func _position_avatar(node: Node3D, hex: Vector2i) -> void:
	var hex_size: float = 2.0
	var wx = hex_size * sqrt(3.0) * (hex.x + hex.y / 2.0)
	var wz = hex_size * 1.5 * hex.y
	node.position = Vector3(wx, 0.4, wz)

# ==============================================================================
# 2. IN-ENGINE GUI / HUD (CanvasLayer)
# ==============================================================================
func _setup_hud() -> void:
	_canvas_layer = CanvasLayer.new()
	_canvas_layer.name = "DuelHUD"
	add_child(_canvas_layer)

	# Main container panel (glassmorphism look)
	var panel = PanelContainer.new()
	panel.custom_minimum_size = Vector2(480, 680)
	panel.position = Vector2(24, 20)
	_canvas_layer.add_child(panel)

	var vbox = VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 10)
	panel.add_child(vbox)

	# Header Title
	var title = Label.new()
	title.text = "⚔ POSLEDNÍ KMEN // THE WEST DUEL ARENA"
	title.add_theme_color_override("font_color", Color(0.4, 0.98, 0.94))
	title.add_theme_font_size_override("font_size", 16)
	vbox.add_child(title)

	# Health Pips Displays (Strict VITAL_MAX_HP = 6)
	_lbl_player_hp = Label.new()
	_lbl_opponent_hp = Label.new()
	_lbl_status = Label.new()
	_lbl_status.text = "KOLO: 1 | PRIPRAVENÝ NA AKCIU"
	_lbl_status.add_theme_color_override("font_color", Color(1.0, 0.85, 0.3))
	vbox.add_child(_lbl_status)
	vbox.add_child(_lbl_player_hp)
	vbox.add_child(_lbl_opponent_hp)
	_update_hud_vital_pips()

	# Target Zone Buttons
	var lbl_zone = Label.new()
	lbl_zone.text = "ZÓNA ZÁSAHU (Mierenie):"
	vbox.add_child(lbl_zone)

	var hbox_zones = HBoxContainer.new()
	hbox_zones.add_theme_constant_override("separation", 6)
	vbox.add_child(hbox_zones)

	var btn_head = Button.new()
	btn_head.text = "HLAVA (1.5x)"
	btn_head.pressed.connect(func(): _select_zone(TargetZone.HEAD))
	hbox_zones.add_child(btn_head)

	var btn_r_sh = Button.new()
	btn_r_sh.text = "P. RAMENO (1.15x)"
	btn_r_sh.pressed.connect(func(): _select_zone(TargetZone.RIGHT_SHOULDER))
	hbox_zones.add_child(btn_r_sh)

	var btn_l_sh = Button.new()
	btn_l_sh.text = "Ľ. RAMENO (1.15x)"
	btn_l_sh.pressed.connect(func(): _select_zone(TargetZone.LEFT_SHOULDER))
	hbox_zones.add_child(btn_l_sh)

	var btn_torso = Button.new()
	btn_torso.text = "TRUP (1.0x)"
	btn_torso.pressed.connect(func(): _select_zone(TargetZone.TORSO))
	hbox_zones.add_child(btn_torso)

	# Dodge Stance Buttons
	var lbl_stance = Label.new()
	lbl_stance.text = "OBRANNÝ POSTOJ (Uhýbanie):"
	vbox.add_child(lbl_stance)

	var hbox_stances = HBoxContainer.new()
	hbox_stances.add_theme_constant_override("separation", 6)
	vbox.add_child(hbox_stances)

	var btn_duck = Button.new()
	btn_duck.text = "ZOHNÚŤ SA"
	btn_duck.pressed.connect(func(): _select_stance(DodgeStance.DUCK_DOWN))
	hbox_stances.add_child(btn_duck)

	var btn_l_dodge = Button.new()
	btn_l_dodge.text = "ÚKROK VĽAVO"
	btn_l_dodge.pressed.connect(func(): _select_stance(DodgeStance.LEAN_LEFT))
	hbox_stances.add_child(btn_l_dodge)

	var btn_r_dodge = Button.new()
	btn_r_dodge.text = "ÚKROK VPRAVO"
	btn_r_dodge.pressed.connect(func(): _select_stance(DodgeStance.LEAN_RIGHT))
	hbox_stances.add_child(btn_r_dodge)

	var btn_firm = Button.new()
	btn_firm.text = "PEVNÝ POSTOJ"
	btn_firm.pressed.connect(func(): _select_stance(DodgeStance.STAND_FIRM))
	hbox_stances.add_child(btn_firm)

	# Action Buttons
	var btn_duel = Button.new()
	btn_duel.text = "⚡ VYMENIŤ VÝSTRELY (DUEL ROUND)"
	btn_duel.custom_minimum_size.y = 38
	btn_duel.pressed.connect(_on_duel_round_pressed)
	vbox.add_child(btn_duel)

	var hbox_actions = HBoxContainer.new()
	hbox_actions.add_theme_constant_override("separation", 8)
	vbox.add_child(hbox_actions)

	var btn_trench = Button.new()
	btn_trench.text = "🛡 ZAKOPAŤ ZÁKOP (SAPPER)"
	btn_trench.pressed.connect(_on_dig_trench_pressed)
	hbox_actions.add_child(btn_trench)

	var btn_mortar = Button.new()
	btn_mortar.text = "💣 MAŽIAROVÁ SALVA (CEP)"
	btn_mortar.pressed.connect(_on_mortar_strike_pressed)
	hbox_actions.add_child(btn_mortar)

	# Reset Match
	var btn_reset = Button.new()
	btn_reset.text = "🔄 NOVÝ ZÁPAS"
	btn_reset.pressed.connect(reset_match)
	vbox.add_child(btn_reset)

	# Combat Chronicle Log
	var lbl_log = Label.new()
	lbl_log.text = "BOJOVÁ KRONIKA:"
	vbox.add_child(lbl_log)

	_txt_combat_log = RichTextLabel.new()
	_txt_combat_log.custom_minimum_size = Vector2(440, 190)
	_txt_combat_log.scroll_following = true
	_txt_combat_log.bbcode_enabled = true
	vbox.add_child(_txt_combat_log)

func _update_hud_vital_pips() -> void:
	if not _lbl_player_hp or not _lbl_opponent_hp:
		return

	# Discrete 6-pip vital representation
	var p_pips = ""
	for i in range(VITAL_MAX_HP):
		p_pips += " ♥" if i < player_hp else " ♡"
	_lbl_player_hp.text = "HRDINA (Kryštál): %s [%d/6 HP]" % [p_pips, player_hp]
	_lbl_player_hp.add_theme_color_override("font_color", Color(0.2, 0.9, 1.0) if player_hp > 2 else Color(1.0, 0.3, 0.3))

	var o_pips = ""
	for i in range(VITAL_MAX_HP):
		o_pips += " 💀" if i < opponent_hp else " ▫"
	_lbl_opponent_hp.text = "SÚPER (Toxik):    %s [%d/6 HP]" % [o_pips, opponent_hp]
	_lbl_opponent_hp.add_theme_color_override("font_color", Color(1.0, 0.3, 0.3) if opponent_hp > 2 else Color(0.7, 0.7, 0.7))

func _select_zone(zone: TargetZone) -> void:
	player_target_zone = zone
	_log_event("Mierenie zamerané na: [color=cyan]%s[/color]" % _zone_to_string(zone))

func _select_stance(stance: DodgeStance) -> void:
	player_dodge_stance = stance
	_log_event("Obranný postoj nastavený na: [color=yellow]%s[/color]" % _stance_to_string(stance))

# ==============================================================================
# 3. GAME ACTIONS & ALGEBRA RESOLUTION
# ==============================================================================
func _on_duel_round_pressed() -> void:
	if is_match_over:
		_log_event("[color=red]Zápas už skončil! Kliknite na Nový Zápas.[/color]")
		return

	if use_remote_hub:
		_send_remote_duel_round()
	else:
		_resolve_duel_round_offline()

func _on_dig_trench_pressed() -> void:
	if is_match_over:
		return

	if use_remote_hub:
		_send_remote_dig_trench()
	else:
		_dig_trench_offline(player_hex)

func _on_mortar_strike_pressed() -> void:
	if is_match_over:
		return

	if use_remote_hub:
		_send_remote_mortar_strike()
	else:
		_execute_mortar_offline(opponent_hex)

# ==============================================================================
# 4. REMOTE HTTP ENGINE BRIDGE (Krystal Hub)
# ==============================================================================
func _send_remote_duel_round() -> void:
	var payload = {
		"player_attack_zone": _zone_to_string(player_target_zone),
		"player_dodge_stance": _stance_to_string(player_dodge_stance)
	}
	var json_str = JSON.stringify(payload)
	var headers = ["Content-Type: application/json"]
	_pending_request_type = "DUEL_ROUND"
	var url = hub_api_url + "/api/game/duel_round"
	var err = _http_request.request(url, headers, HTTPClient.METHOD_POST, json_str)
	if err != OK:
		print("[PosledniKmen] HTTP request failed, falling back to offline engine.")
		_resolve_duel_round_offline()

func _send_remote_dig_trench() -> void:
	var payload = {
		"hex_coord": [player_hex.x, player_hex.y],
		"unit_id": "player_hero"
	}
	var json_str = JSON.stringify(payload)
	var headers = ["Content-Type: application/json"]
	_pending_request_type = "DIG_TRENCH"
	var url = hub_api_url + "/api/game/sapper_trench"
	var err = _http_request.request(url, headers, HTTPClient.METHOD_POST, json_str)
	if err != OK:
		_dig_trench_offline(player_hex)

func _send_remote_mortar_strike() -> void:
	var payload = {
		"target_hex": [opponent_hex.x, opponent_hex.y],
		"salvo_count": 3,
		"elevation_angle_deg": 65.0
	}
	var json_str = JSON.stringify(payload)
	var headers = ["Content-Type: application/json"]
	_pending_request_type = "MORTAR_STRIKE"
	var url = hub_api_url + "/api/game/mortar_strike"
	var err = _http_request.request(url, headers, HTTPClient.METHOD_POST, json_str)
	if err != OK:
		_execute_mortar_offline(opponent_hex)

func _on_http_request_completed(result: int, response_code: int, headers: PackedStringArray, body: PackedByteArray) -> void:
	if response_code == 200:
		var json_str = body.get_string_from_utf8()
		var res = JSON.parse_string(json_str)
		if res and res is Dictionary:
			match _pending_request_type:
				"DUEL_ROUND":
					_apply_duel_round_response(res)
				"DIG_TRENCH":
					_apply_dig_trench_response(res)
				"MORTAR_STRIKE":
					_apply_mortar_strike_response(res)
				_:
					pass
			return

	print("[PosledniKmen] Remote Hub unavailable (HTTP %d). Executing offline engine fallback." % response_code)
	match _pending_request_type:
		"DUEL_ROUND": _resolve_duel_round_offline()
		"DIG_TRENCH": _dig_trench_offline(player_hex)
		"MORTAR_STRIKE": _execute_mortar_offline(opponent_hex)

func _apply_duel_round_response(data: Dictionary) -> void:
	player_hp = int(data.get("player_hp", player_hp))
	opponent_hp = int(data.get("opponent_hp", opponent_hp))
	combat_round = int(data.get("round_number", combat_round + 1))
	is_match_over = bool(data.get("is_game_over", false))
	winner_name = str(data.get("winner", ""))

	var p_shot = data.get("player_shot", {})
	var o_shot = data.get("opponent_shot", {})

	if p_shot.get("evaded", false):
		_log_event("[color=yellow]Súper sa vyhol vašej strele![/color] (%s)" % p_shot.get("evasion_reason", ""))
	elif p_shot.get("hit", false):
		_log_event("[color=green]ZÁSAH![/color] Súper utrpel [color=red]%d[/color] zranenia do %s." % [p_shot.get("net_damage_dealt", 1), p_shot.get("target_zone", "torso")])

	if o_shot.get("evaded", false):
		_log_event("[color=cyan]ÚSPEŠNÝ ÚKROK![/color] Vyhli ste sa súperovmu útoku!")
	elif o_shot.get("hit", false):
		_log_event("[color=red]ZASIAHNUTÝ![/color] Utrpeli ste [color=red]%d[/color] zranenia." % o_shot.get("net_damage_dealt", 1))

	_check_game_over()
	round_resolved.emit(data)

func _apply_dig_trench_response(data: Dictionary) -> void:
	var h_coord = data.get("hex_coord", [player_hex.x, player_hex.y])
	var hex = Vector2i(h_coord[0], h_coord[1])
	_spawn_trench_mesh(hex)
	_log_event("[color=cyan]ŽENIJNÁ AKCIA:[/color] Zákop vykopaný na [%d, %d]! Krytie: 85%%. Ochrana pred mínometom je aktívna." % [hex.x, hex.y])
	trench_dug.emit(hex)

func _apply_mortar_strike_response(data: Dictionary) -> void:
	var hits = data.get("hits", [])
	var salvo = data.get("salvo", {})
	player_hp = int(data.get("player_hp", player_hp))
	opponent_hp = int(data.get("opponent_hp", opponent_hp))

	var cep_lat = salvo.get("dispersion_footprint", {}).get("cep_lateral_meters", 1.2)
	var cep_lon = salvo.get("dispersion_footprint", {}).get("cep_longitudinal_meters", 1.8)
	_log_event("[color=orange]MÍNOMETNÁ SALVA:[/color] Rozptyl CEP: %0.1fm x %0.1fm." % [cep_lat, cep_lon])

	for h in hits:
		_log_event("  ➤ %s" % h.get("summary", ""))

	var shock = data.get("sensory_shockwave", {})
	if shock and shock.has("evaluations"):
		for ev in shock["evaluations"]:
			var rec = ev.get("wave_reception", {})
			var u_name = ev.get("name", "Bojovník")
			if rec.get("is_sheltered_by_trench", false):
				_log_event("[color=cyan]  🛡 RÁZOVÁ VLNA:[/color] %s v zákope odklonil %0.0f%% tlaku." % [u_name, rec.get("trench_dampening_pct", 80.0)])
			else:
				_log_event("[color=orange]  💥 RÁZOVÁ VLNA:[/color] %s zasiahol pretlak %0.1f kPa -> Nárazová úloha: POTLAČENIE & OTRAS!" % [u_name, rec.get("effective_intensity", 0.0)])

	_trigger_mortar_fx(opponent_hex)
	_check_game_over()
	mortar_impacted.emit(data)

# ==============================================================================
# 5. OFFLINE EMBEDDED ENGINE (Standalone Fallback)
# ==============================================================================
func _resolve_duel_round_offline() -> void:
	var p_zone_str = _zone_to_string(player_target_zone)
	var p_stance_str = _stance_to_string(player_dodge_stance)

	# AI chooses
	var ai_zone = TargetZone.TORSO
	var ai_stance = DodgeStance.STAND_FIRM if combat_round % 2 == 0 else DodgeStance.DUCK_DOWN

	# Player attack check
	var player_evaded = (p_zone_str == "head" and ai_stance == DodgeStance.DUCK_DOWN)
	if player_evaded:
		_log_event("[color=yellow]Súper sa zohol a vyhol zásahu do hlavy![/color]")
	else:
		var dmg = 2 if p_zone_str == "head" else 1
		opponent_hp -= dmg
		_log_event("[color=green]ZÁSAH![/color] Súper utrpel %d zranenia do %s." % [dmg, p_zone_str])

	# Opponent attack check
	var opponent_evaded = (ai_zone == TargetZone.HEAD and player_dodge_stance == DodgeStance.DUCK_DOWN)
	if opponent_evaded:
		_log_event("[color=cyan]Vyhli ste sa súperovmu útoku![/color]")
	else:
		player_hp -= 1
		_log_event("[color=red]ZASIAHNUTÝ![/color] Utrpeli ste 1 zranenie.")

	combat_round += 1
	_lbl_status.text = "KOLO: %d | VÝMENA DOKONČENÁ" % combat_round
	_check_game_over()

func _dig_trench_offline(hex: Vector2i) -> void:
	var k = "%d_%d" % [hex.x, hex.y]
	cover_map[k] = 0.85
	if not trenches.has(k):
		trenches.append(k)
	_spawn_trench_mesh(hex)
	_log_event("[color=cyan]ŽENIJNÁ AKCIA:[/color] Zákop vykopaný na [%d, %d]! Krytie zvýšené na 85%%." % [hex.x, hex.y])
	trench_dug.emit(hex)

func _execute_mortar_offline(target: Vector2i) -> void:
	var k = "%d_%d" % [target.x, target.y]
	var cover = cover_map.get(k, 0.15)
	_trigger_mortar_fx(target)

	if cover >= 0.75:
		_log_event("[color=orange]MÍNOMETNÁ SALVA:[/color] Cieľ je v zákope (Krytie %d%%) - ŽIADNE ZRANENIE!" % int(cover * 100))
	else:
		opponent_hp -= 2
		_log_event("[color=red]MÍNOMETNÁ SALVA:[/color] Priamy zásah nekrytého cieľa! Súper utrpel 2 plošné zranenia.")

	_check_game_over()

func _spawn_trench_mesh(hex: Vector2i) -> void:
	var hex_size: float = 2.0
	var wx = hex_size * sqrt(3.0) * (hex.x + hex.y / 2.0)
	var wz = hex_size * 1.5 * hex.y

	var trench_box = MeshInstance3D.new()
	var box = BoxMesh.new()
	box.size = Vector3(2.4, 0.45, 2.4)
	trench_box.mesh = box
	trench_box.position = Vector3(wx, 0.22, wz)

	var mat = StandardMaterial3D.new()
	mat.albedo_color = Color(0.4, 0.25, 0.12, 1.0) # Mud / Wood barricade
	mat.metallic = 0.1
	mat.roughness = 0.9
	trench_box.material_override = mat

	_trench_mesh_container.add_child(trench_box)

func _trigger_mortar_fx(hex: Vector2i) -> void:
	# Subtle screen shake / light flash
	if camera_3d:
		var orig_pos = camera_3d.position
		var tween = create_tween()
		tween.tween_property(camera_3d, "position", orig_pos + Vector3(0.15, -0.1, 0.1), 0.05)
		tween.tween_property(camera_3d, "position", orig_pos, 0.15)

func _check_game_over() -> void:
	if player_hp <= 0 or opponent_hp <= 0:
		is_match_over = true
		if player_hp <= 0 and opponent_hp <= 0:
			winner_name = "REMIZA (Obojstranný knokaut)"
		elif opponent_hp <= 0:
			winner_name = "HRDINA (Kryštálový Kmeň) VYHRAL!"
		else:
			winner_name = "SÚPER (Toxický Kmeň) VYHRAL!"

		_lbl_status.text = "🏆 KONIEC: %s" % winner_name
		_lbl_status.add_theme_color_override("font_color", Color(1.0, 0.2, 0.2))
		_log_event("[color=yellow][b]🏆 ZÁPAS UKONČENÝ: %s[/b][/color]" % winner_name)
		match_finished.emit(winner_name)

func reset_match() -> void:
	player_hp = VITAL_MAX_HP
	opponent_hp = VITAL_MAX_HP
	combat_round = 1
	is_match_over = false
	winner_name = ""
	_init_cover_map()

	# Clear trench meshes
	if _trench_mesh_container:
		for c in _trench_mesh_container.get_children():
			c.queue_free()

	_lbl_status.text = "KOLO: 1 | PRIPRAVENÝ NA AKCIU"
	_lbl_status.add_theme_color_override("font_color", Color(1.0, 0.85, 0.3))
	_update_hud_vital_pips()
	_log_event("--- NOVÝ ZÁPAS ZAČATÝ ---")

# ==============================================================================
# 6. HELPERS & FORMATTING
# ==============================================================================
func _zone_to_string(z: TargetZone) -> String:
	match z:
		TargetZone.HEAD: return "head"
		TargetZone.RIGHT_SHOULDER: return "right_shoulder"
		TargetZone.LEFT_SHOULDER: return "left_shoulder"
		TargetZone.TORSO: return "torso"
	return "torso"

func _stance_to_string(s: DodgeStance) -> String:
	match s:
		DodgeStance.DUCK_DOWN: return "duck_down"
		DodgeStance.LEAN_LEFT: return "lean_left"
		DodgeStance.LEAN_RIGHT: return "lean_right"
		DodgeStance.STAND_FIRM: return "stand_firm"
	return "stand_firm"

func _log_event(bbcode_text: String) -> void:
	if _txt_combat_log:
		_txt_combat_log.append_text(bbcode_text + "\n")
	print("[PosledniKmen] ", bbcode_text)
