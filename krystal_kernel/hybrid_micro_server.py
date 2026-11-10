#!/usr/bin/env python3
"""
KRYSTAL-STACK: LOCAL VALIDATED HYBRID MICRO-SERVER
==================================================
Lightweight zero-dependency local micro-server (default http://127.0.0.1:8091)
providing schema-validated operations between Godot 4.x, Frontend Web UI,
Strapi v5 CMS, MedusaJS / WooCommerce, Fantasy Lore-LLM, and Mariáš turns.

Enforces System Invariant: VITAL_MAX_HP = 6
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Tuple

from krystal_kernel.hybrid_lore_aggregator_runtime import (
    GLOBAL_HYBRID_PLATFORM,
    GridMechanicalBitboardBridge,
    MariasSuit,
    RenderPipelineTerms,
    VITAL_MAX_HP,
)


def dispatch_validated_micro_operation(path: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    facade = GLOBAL_HYBRID_PLATFORM
    if VITAL_MAX_HP != 6:
        return 500, {"status": "INVARIANT_VIOLATION", "vital_max_hp": VITAL_MAX_HP}

    if path in ("/api/hybrid/status", "/health"):
        val = facade.runtime_builder.validate_all(facade.signal_extractor)
        return 200, {
            "status": "VALIDATED_OK",
            "vital_max_hp": VITAL_MAX_HP,
            "capabilities": facade.get_platform_capabilities(),
            "security_and_taxonomy_validation": val,
        }

    if path == "/api/hybrid/fetch_objectives":
        rt = payload.get("render_terms", {})
        terms = RenderPipelineTerms(
            direct_irradiance=float(rt.get("direct_irradiance", 1.35)),
            gi_bounce_luminance=float(rt.get("gi_bounce_luminance", 0.62)),
            specular_roughness=float(rt.get("specular_roughness", 0.24)),
            caustic_photon_flux=float(rt.get("caustic_photon_flux", 0.48)),
            ao_occlusion=float(rt.get("ao_occlusion", 0.18)),
            mortar_elevation_deg=float(rt.get("mortar_elevation_deg", 42.5)),
        )
        sig = facade.signal_extractor.evaluate(terms)

        grid_move_str = str(payload.get("grid_move", "0,0:2,3"))
        left, right = grid_move_str.split(":")
        fx, fy = (int(v) for v in left.split(","))
        tx, ty = (int(v) for v in right.split(","))
        grid_packet = facade.grid_bridge.emulate_mechanical_move((fx, fy), (tx, ty))
        verified_decode = GridMechanicalBitboardBridge.decode_binary_frame(grid_packet["binary_frame_hex"])

        cms_sync = facade.cms_connector.fetch_unified_catalog(
            probe_network=bool(payload.get("probe_network", False)),
            latest_binary_packet_hex=grid_packet["binary_frame_hex"],
        )
        quest_slug = str(payload.get("quest_slug", "citadel-well-of-souls"))
        choice_id = str(payload.get("choice_id", "init_harmonic"))
        seed = str(payload.get("new_game_seed", ""))
        genesis = facade.cognition_genesis.begin_new_storyline(seed, sig) if seed else None

        lore_step = facade.lore_engine.step_lore_quest(
            quest_slug=quest_slug,
            selected_choice_id=choice_id,
            signal=sig,
            cms_artifacts=cms_sync["commerce_artifacts"],
            attempt_live_ollama=bool(payload.get("probe_network", False)),
        )
        return 200, {
            "status": "VALIDATED_OBJECTIVES_READY",
            "vital_max_hp": VITAL_MAX_HP,
            "storyline_genesis": genesis,
            "grid_mechanical_binary": grid_packet,
            "grid_binary_crc_verified": verified_decode["crc_verified"],
            "cms_commerce_sync": cms_sync,
            "lore_step": lore_step,
        }

    if path == "/api/hybrid/marias_click":
        suit_str = str(payload.get("suit", "HEARTS_SRDCIA"))
        rank_str = str(payload.get("rank", "KING_KRAL"))
        suit_enum = next((s for s in MariasSuit if s.value == suit_str or s.name == suit_str), MariasSuit.SRDCIA_HEARTS)
        staged = facade.marias_engine.stage_card_play(
            suit=suit_enum,
            rank=rank_str,
            hand_ranks_in_suit=["KING_KRAL", "OBER_HORNIK", rank_str],
            target_objective=str(payload.get("quest_slug", "Citadel Ley-Grid Sanctum")),
        )
        resolved = facade.marias_engine.resolve_click_semantic_animation(staged["action_id"])
        return 200, {
            "status": "VALIDATED_MARIAS_RESOLVED",
            "vital_max_hp": VITAL_MAX_HP,
            "marias_staged": staged,
            "marias_click_animation": resolved,
        }

    return 404, {"status": "UNKNOWN_OPERATION", "path": path, "vital_max_hp": VITAL_MAX_HP}


class HybridMicroServerHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, code: int, body: Dict[str, Any]) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        code, resp = dispatch_validated_micro_operation(self.path.split("?")[0], {})
        self._send(code, resp)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        payload = json.loads(raw) if raw.strip() else {}
        code, resp = dispatch_validated_micro_operation(self.path.split("?")[0], payload)
        self._send(code, resp)


def run_micro_server(host: str = "127.0.0.1", port: int = 8091) -> None:
    srv = HTTPServer((host, port), HybridMicroServerHandler)
    print(f"[KRYSTAL MICRO-SERVER] Listening on http://{host}:{port} (VITAL_MAX_HP={VITAL_MAX_HP})")
    srv.serve_forever()


if __name__ == "__main__":
    run_micro_server()
