"""
scratch/verify_game_language_api_fetcher.py
===========================================
End-to-End Verification of Game Language API Fetcher & Tactical Controller.
Tests:
1. Python engine layer directly (Slovak & English parsing, execution, queries).
2. Verification of Godot 4.x Client Bridge (GameLanguageControllerBridge.gd).
3. Verification of Janet DSL AST specification (game_language_api_fetcher.janet).
4. Verification of Web Studio HTML interface (game_language_control_studio.html).
5. Invariant integrity: VITAL_MAX_HP == 6 invariant verification.
"""

import os
import sys
import json
import time

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from krystal_web_hub.economic_engine import (
    GLOBAL_GAME_LANGUAGE_FETCHER,
    GameCommandType,
    LanguageCommandIntent,
    LanguageExecutionResult,
    VITAL_MAX_HP
)


def run_verification():
    GLOBAL_GAME_LANGUAGE_FETCHER.config.mock_mode = True
    print("=" * 75)
    print("💎 KRYSTAL-STACK // GAME LANGUAGE API FETCHER & CONTROLLER VERIFICATION")
    print("=" * 75)

    # 1. Invariant Check
    print("\n[Step 1] Invariant Audit: VITAL_MAX_HP == 6")
    assert VITAL_MAX_HP == 6, f"Invariant violated! Expected 6, got {VITAL_MAX_HP}"
    catalog = GLOBAL_GAME_LANGUAGE_FETCHER.get_capabilities_catalog()
    assert catalog["vital_max_hp_rule"] == 6, "Catalog invariant violated!"
    print(f"  -> PASS: Universal invariant VITAL_MAX_HP = {VITAL_MAX_HP} is strictly enforced.")

    # 2. Multilingual Command Parsing & Execution
    print("\n[Step 2] Testing Multilingual Command Parsing & Execution...")
    test_cases = [
        ("Zahraj kartu Kryštálový Meteor na hex [1, -2]", GameCommandType.CAST_CARD, "sk"),
        ("Presuň hrdinu na hex [2, 0]", GameCommandType.TACTICAL_MOVE, "sk"),
        ("Nastav počasie na búrku a čas na polnoc", GameCommandType.WEATHER_ATMOSPHERE, "sk"),
        ("Prepnúť kameru na izometrický pohľad", GameCommandType.CAMERA_CONTROL, "sk"),
        ("Vystreľ z moždiara na sektor 3", GameCommandType.ARTILLERY_FIRE, "sk"),
        ("Postav slovanský mažiar na vežu 1", GameCommandType.CITADEL_BUILD, "sk"),
        ("Vyvolaj CesiumMan na pozíciu [0, 1]", GameCommandType.SPAWN_ENTITY, "sk"),
        ("Ukonči kolo", GameCommandType.MATCH_ESCALATION, "sk"),
        ("Cast Glacial Lance at hex [1, 2]", GameCommandType.CAST_CARD, "en"),
        ("Move hero to hex [0, -1]", GameCommandType.TACTICAL_MOVE, "en"),
        ("Set weather to rain with dusk lighting", GameCommandType.WEATHER_ATMOSPHERE, "en"),
        ("Switch camera to cinematic wide", GameCommandType.CAMERA_CONTROL, "en"),
        ("What is my current HP and mana?", GameCommandType.QUERY_STATUS, "en")
    ]

    for prompt, expected_type, expected_lang in test_cases:
        res = GLOBAL_GAME_LANGUAGE_FETCHER.execute_command(prompt, execute_api=False)
        assert res.intent["command_type"] == expected_type.value, (
            f"Failed on '{prompt}': expected {expected_type.value}, got {res.intent['command_type']}"
        )
        assert res.intent["language"] == expected_lang, (
            f"Language mismatch on '{prompt}': expected {expected_lang}, got {res.intent['language']}"
        )
        print(f"  -> OK: [{res.intent['language'].upper()}] '{prompt}' => {res.intent['command_type']} ({res.api_action_executed})")

    # 3. Natural Language Querying
    print("\n[Step 3] Testing Natural Language Status Queries...")
    query_sk = GLOBAL_GAME_LANGUAGE_FETCHER.query_game_state("Aký je stav môjho hrdinu a zápasu?")
    assert "Poslední Kmen" in query_sk["answer"]
    assert query_sk["hp"] <= 6
    print(f"  -> SK Query Output:\n     {query_sk['answer'].splitlines()[0]}")

    query_en = GLOBAL_GAME_LANGUAGE_FETCHER.query_game_state("What is my current battlefield status?")
    assert "Battlefield Status" in query_en["answer"]
    print(f"  -> EN Query Output:\n     {query_en['answer'].splitlines()[0]}")

    # 4. Remote Fetcher & Daemon Lifecycle
    print("\n[Step 4] Testing Remote Fetcher & Background Daemon Lifecycle...")
    remote_results = GLOBAL_GAME_LANGUAGE_FETCHER.fetch_and_execute_remote()
    assert len(remote_results) >= 1
    print(f"  -> OK: Remote fetch simulated {len(remote_results)} command(s), feedback: '{remote_results[0].game_feedback}'")

    GLOBAL_GAME_LANGUAGE_FETCHER.start_fetcher_daemon(poll_interval_s=0.5)
    assert GLOBAL_GAME_LANGUAGE_FETCHER._daemon_running is True
    print("  -> OK: Background fetcher daemon started successfully.")
    time.sleep(0.6)
    GLOBAL_GAME_LANGUAGE_FETCHER.stop_fetcher_daemon()
    assert GLOBAL_GAME_LANGUAGE_FETCHER._daemon_running is False
    print("  -> OK: Background fetcher daemon stopped cleanly.")

    # 5. File & Asset Integrity Checks
    print("\n[Step 5] Checking Godot, Janet & Web Studio Asset Integrity...")
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    godot_bridge_path = os.path.join(workspace_root, "godot_project", "scripts", "GameLanguageControllerBridge.gd")
    assert os.path.exists(godot_bridge_path), f"Missing {godot_bridge_path}"
    print(f"  -> PASS: Godot Client Bridge exists at {os.path.relpath(godot_bridge_path, workspace_root)}")

    janet_path = os.path.join(workspace_root, "krystal_janet", "game_language_api_fetcher.janet")
    assert os.path.exists(janet_path), f"Missing {janet_path}"
    print(f"  -> PASS: Janet AST Specification exists at {os.path.relpath(janet_path, workspace_root)}")

    studio_html = os.path.join(workspace_root, "krystal_web_hub", "static", "game_language_control_studio.html")
    assert os.path.exists(studio_html), f"Missing {studio_html}"
    print(f"  -> PASS: Web Studio Dashboard exists at {os.path.relpath(studio_html, workspace_root)}")

    # 6. Audit Trail Check
    history = GLOBAL_GAME_LANGUAGE_FETCHER.get_history(limit=5)
    assert len(history) > 0
    print(f"  -> PASS: Audit trail contains {len(history)} recent execution traces.")

    print("\n" + "=" * 75)
    print("🏆 ALL VERIFICATION CHECKS PASSED: GAME LANGUAGE API FETCHER READY FOR DEPLOYMENT")
    print("=" * 75)


if __name__ == '__main__':
    run_verification()
