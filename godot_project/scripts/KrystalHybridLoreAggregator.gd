# ==============================================================================
# KRYSTAL-STACK: GODOT 4.X HYBRID CMS, LORE-LLM & ALGEBRAIC V-RAY AGGREGATOR
# ==============================================================================
# File: godot_project/scripts/KrystalHybridLoreAggregator.gd
# Description: Bridges Godot 4.x Engine <-> Local Validated Micro-Server & Strapi CMS:
#   1. Algebraically samples viewport render pipeline terms (V-Ray style signals).
#   2. Emulates 8x8 mechanical grid movements -> 24-byte KMB1 binary CMS packets.
#   3. Fetches dynamically mutating Lore Objectives from Strapi / Medusa / WooCommerce.
#   4. Stages Mariáš card turns & triggers semantic Bezier/shader animations on click.
#
# System Invariant: VITAL_MAX_HP = 6
# ==============================================================================

class_name KrystalHybridLoreAggregator
extends Node

signal objectives_fetched(objectives: Array, active_lens: String, cms_payload: Dictionary)
signal vray_signal_computed(signal_data: Dictionary)
signal grid_binary_emitted(binary_hex: String, bitboard_hex: String)
signal marias_semantic_animation_triggered(anim_descriptor: Dictionary)

@export var micro_server_url: String = "http://127.0.0.1:8091"
@export var strapi_cms_url: String = "http://127.0.0.1:1337"
@export var quest_slug: String = "citadel-well-of-souls"
@export var direct_irradiance: float = 1.35
@export var gi_bounce_luminance: float = 0.62
@export var specular_roughness: float = 0.24
@export var caustic_photon_flux: float = 0.48
@export var mortar_elevation_deg: float = 42.5

const VITAL_MAX_HP: int = 6

var http_client: HTTPRequest
var bitboard_state: int = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)

func _ready() -> void:
	http_client = HTTPRequest.new()
	add_child(http_client)
	http_client.request_completed.connect(_on_http_completed)
	fetch_cms_objectives("init_harmonic")

func pack_grid_move_binary_hex(from_cell: Vector2i, to_cell: Vector2i) -> String:
	var src_bit: int = clampi(from_cell.y, 0, 7) * 8 + clampi(from_cell.x, 0, 7)
	var dst_bit: int = clampi(to_cell.y, 0, 7) * 8 + clampi(to_cell.x, 0, 7)
	bitboard_state &= ~(1 << src_bit)
	bitboard_state |= (1 << dst_bit)
	var buf := StreamPeerBuffer.new()
	buf.big_endian = true
	buf.put_data("KMB1".to_ascii_buffer())
	buf.put_64(bitboard_state)
	buf.put_u8(src_bit)
	buf.put_u8(dst_bit)
	buf.put_u16(4850)
	buf.put_u32(VITAL_MAX_HP)
	var hex_str := buf.data_array.hex_encode()
	grid_binary_emitted.emit(hex_str, "0x%016X" % bitboard_state)
	return hex_str

func fetch_cms_objectives(choice_id: String = "init_harmonic", grid_from := Vector2i(0, 0), grid_to := Vector2i(2, 3)) -> void:
	if not http_client or http_client.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		return
	var url := "%s/api/hybrid/fetch_objectives" % micro_server_url
	var headers := PackedStringArray(["Content-Type: application/json"])
	var payload := JSON.stringify({
		"quest_slug": quest_slug,
		"choice_id": choice_id,
		"strapi_url": strapi_cms_url,
		"grid_move": "%d,%d:%d,%d" % [grid_from.x, grid_from.y, grid_to.x, grid_to.y],
		"render_terms": {
			"direct_irradiance": direct_irradiance,
			"gi_bounce_luminance": gi_bounce_luminance,
			"specular_roughness": specular_roughness,
			"caustic_photon_flux": caustic_photon_flux,
			"mortar_elevation_deg": mortar_elevation_deg
		}
	})
	http_client.request(url, headers, HTTPClient.METHOD_POST, payload)

func click_pending_marias_card(suit: String = "HEARTS_SRDCIA", rank: String = "KING_KRAL") -> void:
	if not http_client or http_client.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		return
	var url := "%s/api/hybrid/marias_click" % micro_server_url
	var headers := PackedStringArray(["Content-Type: application/json"])
	var payload := JSON.stringify({"suit": suit, "rank": rank, "quest_slug": quest_slug})
	http_client.request(url, headers, HTTPClient.METHOD_POST, payload)

func _on_http_completed(_result: int, response_code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	if response_code != 200:
		return
	var json := JSON.new()
	if json.parse(body.get_string_from_utf8()) != OK or not (json.data is Dictionary):
		return
	var data: Dictionary = json.data
	if data.has("lore_step"):
		var step_data: Dictionary = data["lore_step"]
		vray_signal_computed.emit(step_data.get("vray_signal", {}))
		objectives_fetched.emit(
			step_data.get("choice_hierarchy", []),
			step_data.get("active_perception_lens", "SOLAR_CLARITY"),
			data.get("cms_commerce_sync", {})
		)
	if data.has("marias_click_animation"):
		marias_semantic_animation_triggered.emit(data["marias_click_animation"].get("semantic_animation", {}))
