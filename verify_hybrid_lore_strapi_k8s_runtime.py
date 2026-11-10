#!/usr/bin/env python3
"""
verify_hybrid_lore_strapi_k8s_runtime.py
========================================
End-to-end verification suite for:
  1. V-Ray Algebraic Render Signal Extraction & SPIR-V Taxonomy Kernel Weighting.
  2. Fantasy LLM Beginning-of-Game Storyline Genesis & Learned Cognition Weights.
  3. Mariáš Algebraic Card/Turn Strategy (40-pt Trump Hláška) & Semantic Click Animation.
  4. 8x8 Grid Mechanical Movement -> 24-Byte KMB1 Binary Bitboard CMS Bridge (CRC32 verified).
  5. Strapi v5 + MedusaJS + WooCommerce Unified Objective Fetcher & Validated Micro-Server.
  6. Portable Taxonomy Runtime Bundle + Kubernetes Default-Deny NetworkPolicy + Fail2Ban/nftables.

Enforces System Invariant: VITAL_MAX_HP == 6
"""

import json
import subprocess
import sys
import threading
import urllib.request
from http.server import HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from krystal_kernel.hybrid_lore_aggregator_runtime import (
    GLOBAL_HYBRID_PLATFORM,
    GridMechanicalBitboardBridge,
    MariasSuit,
    RenderPipelineTerms,
    VITAL_MAX_HP,
)
from krystal_kernel.hybrid_micro_server import HybridMicroServerHandler


def test_1_runtime_compile_and_security_validation() -> None:
    print("[TEST 1] Portable SPIR-V Taxonomy Runtime & K8s/Fail2Ban/nftables Firewall...")
    assert VITAL_MAX_HP == 6
    manifest = GLOBAL_HYBRID_PLATFORM.runtime_builder.compile_portable_runtime(
        GLOBAL_HYBRID_PLATFORM.signal_extractor
    )
    assert len(manifest["copied_taxonomy_spv_kernels"]) == 4
    val = GLOBAL_HYBRID_PLATFORM.runtime_builder.validate_all(
        GLOBAL_HYBRID_PLATFORM.signal_extractor
    )
    assert val["valid"] is True, f"Validation failed: {val}"
    print("  -> PASS: 4 SPIR-V kernels bundled; K8s Default-Deny, Fail2Ban & nftables verified.")


def test_2_vray_signal_and_fantasy_genesis() -> None:
    print("[TEST 2] V-Ray Algebraic Signal & Per-Run Fantasy Cognition Genesis...")
    sig = GLOBAL_HYBRID_PLATFORM.signal_extractor.evaluate(
        RenderPipelineTerms(direct_irradiance=1.6, caustic_photon_flux=0.72, specular_roughness=0.18)
    )
    assert sig.vital_max_hp == 6
    assert sig.total_radiance_term > 1.5
    gen = GLOBAL_HYBRID_PLATFORM.cognition_genesis.begin_new_storyline("citadel-dawn-alpha", sig)
    assert gen["cognition_archetype"] in {a[0] for a in GLOBAL_HYBRID_PLATFORM.cognition_genesis.ARCHETYPES}
    print(f"  -> PASS: Lens={sig.active_lens}, Archetype={gen['cognition_archetype']}, Radiance={sig.total_radiance_term}")


def test_3_marias_marriage_and_semantic_click() -> None:
    print("[TEST 3] Mariáš Card Turn (40-pt Trump Hláška) & Semantic Click Animation...")
    staged = GLOBAL_HYBRID_PLATFORM.marias_engine.stage_card_play(
        suit=MariasSuit.SRDCIA_HEARTS,
        rank="KING_KRAL",
        hand_ranks_in_suit=["KING_KRAL", "OBER_HORNIK", "ACE_ESO"],
        target_objective="Citadel Ley-Grid Sanctum",
    )
    assert staged["marias_marriage_declared"] is True
    assert staged["marriage_bonus_points"] == 40
    resolved = GLOBAL_HYBRID_PLATFORM.marias_engine.resolve_click_semantic_animation(staged["action_id"])
    assert resolved["status"] == "CLICK_RESOLVED_ANIMATING"
    assert resolved["semantic_animation"]["gesture_archetype"] == "ROYAL_MARIAS_CONJUNCTION_ARC"
    print(f"  -> PASS: Mariáš score={resolved['total_algebraic_score']} -> {resolved['semantic_animation']['visual_aura']}")


def test_4_grid_mechanical_binary_cms_bridge() -> None:
    print("[TEST 4] 8x8 Mechanical Grid Movement -> 24B KMB1 Binary CMS Packet...")
    pkt = GLOBAL_HYBRID_PLATFORM.grid_bridge.emulate_mechanical_move((0, 0), (3, 4), mechanical_torque_nm=52.4)
    assert pkt["binary_frame_bytes_len"] == 24
    decoded = GridMechanicalBitboardBridge.decode_binary_frame(pkt["binary_frame_hex"])
    assert decoded["crc_verified"] is True
    assert decoded["vital_max_hp"] == 6
    assert decoded["manhattan_displacement"] == 7
    print(f"  -> PASS: KMB1 frame={pkt['binary_frame_hex'][:24]}... CRC={pkt['crc32']} verified.")


def test_5_live_micro_server_and_cli() -> None:
    print("[TEST 5] Live Local Validated Micro-Server & CLI Objective Fetch...")
    srv = HTTPServer(("127.0.0.1", 0), HybridMicroServerHandler)
    port = srv.server_port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/hybrid/fetch_objectives",
            data=json.dumps({
                "quest_slug": "citadel-well-of-souls",
                "choice_id": "step_harmonic",
                "new_game_seed": "omega-perception",
                "grid_move": "7,0:4,5",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            assert body["status"] == "VALIDATED_OBJECTIVES_READY"
            assert body["grid_binary_crc_verified"] is True
            assert len(body["lore_step"]["choice_hierarchy"]) == 3
    finally:
        srv.shutdown()
        srv.server_close()

    cli_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "krystal_kernel.hybrid_lore_aggregator_runtime",
            "--validate-all",
            "--step-quest",
            "citadel-well-of-souls",
            "--json",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    cli_json = json.loads(cli_proc.stdout)
    assert cli_json["validation"]["valid"] is True
    print("  -> PASS: Micro-server HTTP POST & CLI --validate-all passed 100%.")


if __name__ == "__main__":
    print("=" * 70)
    print("KRYSTAL-STACK: VERIFYING HYBRID LORE, MARIÁŠ, CMS & K8S RUNTIME")
    print("=" * 70)
    test_1_runtime_compile_and_security_validation()
    test_2_vray_signal_and_fantasy_genesis()
    test_3_marias_marriage_and_semantic_click()
    test_4_grid_mechanical_binary_cms_bridge()
    test_5_live_micro_server_and_cli()
    print("=" * 70)
    print("ALL 5 HYBRID PLATFORM TESTS PASSED WITH 100% INTEGRITY (VITAL_MAX_HP=6)!")
    print("=" * 70)
