#!/usr/bin/env python3
"""
==============================================================================
VERIFICATION SUITE: JANET BYTECODE PROFILER & BINARY ALERT DECODER
==============================================================================
Tests:
  1. KSYN binary stream generation and 64-bit word packing.
  2. Dual-representation hex trace log and opcode mapping.
  3. Structured alert synthesis ("hlásenie") and VITAL_MAX_HP = 6 invariant assertion.
  4. SVG vector profile blueprint generation.
  5. Terminal ASCII dashboard formatting.
  6. Live REST API endpoints:
     - GET  /api/janet/status
     - GET  /api/janet/render_profile_svg?format=raw
     - POST /api/janet/decode_binary
==============================================================================
"""

import sys
import time
import json
import urllib.request
import urllib.error

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from krystal_kernel.janet_bytecode_decoder import (
    JanetBytecodeDecoder,
    generate_canonical_demo_stream,
    VITAL_MAX_HP
)


def test_1_binary_encoding_and_decoding():
    print("[TEST 1] Testing binary encoding, header verification, and 64-bit word unpacking...")
    demo_bytes = generate_canonical_demo_stream()
    assert len(demo_bytes) >= 8, "Demo stream must have at least 8 bytes"
    assert demo_bytes[:4] == b"KSYN", "Magic header must be KSYN"

    parsed = JanetBytecodeDecoder.parse_ksyn_binary_stream(demo_bytes)
    assert parsed["status"] == "ok", f"Expected ok status, got: {parsed}"
    assert parsed["magic"] == "KSYN"
    assert parsed["version"] == "1.0"
    assert parsed["seed"] == 1337
    assert parsed["total_words"] == 13
    assert len(parsed["instructions"]) == 13
    assert parsed["vital_max_hp"] == 6

    # Verify first instruction is OP_VITAL_ASSERT_HP (0x01)
    inst0 = parsed["instructions"][0]
    assert inst0["opcode"] == 0x01
    assert inst0["opcode_hex"] == "0x01"
    assert inst0["name"] == "OP_VITAL_ASSERT_HP"
    assert inst0["domain"] == "integrity"

    print("  -> PASSED: Binary stream parsed correctly with 13 words, magic KSYN, and valid opcodes.")


def test_2_dual_representation_hex_log():
    print("[TEST 2] Testing dual-representation hex trace log and opcode mapping...")
    demo_bytes = generate_canonical_demo_stream()
    parsed = JanetBytecodeDecoder.parse_ksyn_binary_stream(demo_bytes)

    hex_log = parsed.get("hex_trace_log", "")
    assert "4B 53 59 4E" in hex_log, "Hex log must contain KSYN hex bytes (4B 53 59 4E)"
    assert "|" in hex_log, "Hex log must contain ASCII representation divider"

    ascii_dashboard = JanetBytecodeDecoder.render_terminal_ascii_profile(parsed)
    assert "KRYSTAL-STACK: JANET BINÁRNE MAPOVANIE" in ascii_dashboard
    assert "OP_VITAL_ASSERT_HP" in ascii_dashboard
    assert "K_SPEC_INTERPOLATE_FRAME" in ascii_dashboard

    print("  -> PASSED: Hexadecimal trace log and ASCII dashboard generated with dual representation.")


def test_3_alert_synthesis_and_invariants():
    print("[TEST 3] Testing alert report synthesis ('hlásenie') and invariant validation...")
    demo_bytes = generate_canonical_demo_stream()
    parsed = JanetBytecodeDecoder.parse_ksyn_binary_stream(demo_bytes)
    report = JanetBytecodeDecoder.synthesize_alert_report(parsed)

    assert "HLÁSENIE KERNELU" in report["title"]
    assert report["vital_max_hp"] == 6
    assert len(report["alerts"]) >= 3

    # Check invariant alert
    hp_alert = next((a for a in report["alerts"] if a["code"] == "INVARIANT_VERIFIED"), None)
    assert hp_alert is not None, "Must contain INVARIANT_VERIFIED alert"
    assert "VITAL_MAX_HP = 6" in hp_alert["message"]

    # Check power alert
    pwr_alert = next((a for a in report["alerts"] if "POWER" in a["code"]), None)
    assert pwr_alert is not None, "Must contain power envelope alert"

    # Check K-ISA alert
    kisa_alert = next((a for a in report["alerts"] if a["code"] == "KISA_SPECULATIVE_ACTIVE"), None)
    assert kisa_alert is not None, "Must contain KISA_SPECULATIVE_ACTIVE alert"

    # Test failure condition when invariant assert is missing
    unsafe_stream = JanetBytecodeDecoder.encode_ksyn_stream([
        {"opcode": 0x02, "flags": 0, "param1": 1, "param2": 2}
    ])
    unsafe_parsed = JanetBytecodeDecoder.parse_ksyn_binary_stream(unsafe_stream)
    unsafe_report = JanetBytecodeDecoder.synthesize_alert_report(unsafe_parsed)
    missing_hp = next((a for a in unsafe_report["alerts"] if a["code"] == "MISSING_INVARIANT_ASSERT"), None)
    assert missing_hp is not None, "Missing invariant must trigger WARNING alert"
    assert unsafe_report["severity"] == "WARNING"

    print("  -> PASSED: Kernel alert synthesis properly enforces VITAL_MAX_HP = 6, power caps, and K-ISA tags.")


def test_4_svg_vector_rendering():
    print("[TEST 4] Testing SVG vector execution profile rendering...")
    demo_bytes = generate_canonical_demo_stream()
    parsed = JanetBytecodeDecoder.parse_ksyn_binary_stream(demo_bytes)
    svg = JanetBytecodeDecoder.render_execution_profile_svg(parsed)

    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "JANET BINARY EXECUTION PROFILE" in svg
    assert "VITAL_MAX_HP = 6" in svg
    assert "<rect" in svg
    assert "<line" in svg

    print(f"  -> PASSED: SVG blueprint rendered ({len(svg)} chars) with all domain color tracks.")


def test_5_live_rest_api():
    print("[TEST 5] Testing Live REST API Endpoints on http://127.0.0.1:8080...")
    base_url = "http://127.0.0.1:8080"

    # 1. GET /api/janet/status
    try:
        req = urllib.request.Request(f"{base_url}/api/janet/status", headers={"User-Agent": "KrystalTest/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "OK"
            assert data["vital_max_hp"] == 6
            assert data["parsed"]["magic"] == "KSYN"
            print("  -> GET /api/janet/status: 200 OK")
    except Exception as e:
        print(f"  -> FAIL on GET /api/janet/status: {e}")
        raise

    # 2. GET /api/janet/render_profile_svg?format=raw
    try:
        req = urllib.request.Request(f"{base_url}/api/janet/render_profile_svg?format=raw")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            content_type = resp.headers.get("Content-Type", "")
            assert "image/svg+xml" in content_type
            svg_body = resp.read().decode("utf-8")
            assert "<svg" in svg_body
            print("  -> GET /api/janet/render_profile_svg?format=raw: 200 OK (image/svg+xml)")
    except Exception as e:
        print(f"  -> FAIL on GET /api/janet/render_profile_svg: {e}")
        raise

    # 3. POST /api/janet/decode_binary (Custom Hex Payload)
    try:
        custom_instructions = [
            {"opcode": 0x01, "flags": 0x00, "param1": 6, "param2": 0},
            {"opcode": 0xA2, "flags": 0x01, "param1": 120, "param2": 1},
            {"opcode": 0x00, "flags": 0x00, "param1": 0, "param2": 0},
        ]
        raw_custom = JanetBytecodeDecoder.encode_ksyn_stream(custom_instructions, version=0x0102, seed=999)
        payload = json.dumps({"binary_hex": raw_custom.hex()}).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url}/api/janet/decode_binary",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "OK"
            assert data["parsed"]["version"] == "1.2"
            assert data["parsed"]["seed"] == 999
            assert data["parsed"]["total_words"] == 3
            assert data["report"]["vital_max_hp"] == 6
            assert data["report"]["severity"] == "NOMINAL"
            print("  -> POST /api/janet/decode_binary: 200 OK (Custom Hex decoded into structured alerts)")
    except Exception as e:
        print(f"  -> FAIL on POST /api/janet/decode_binary: {e}")
        raise

    print("  -> PASSED: All REST API endpoints validated successfully.")


if __name__ == "__main__":
    print("================================================================================")
    print("  RUNNING VERIFICATION SUITE: JANET BYTECODE PROFILER & BINARY ALERT DECODER")
    print("================================================================================")
    test_1_binary_encoding_and_decoding()
    test_2_dual_representation_hex_log()
    test_3_alert_synthesis_and_invariants()
    test_4_svg_vector_rendering()
    test_5_live_rest_api()
    print("================================================================================")
    print("  ALL 5 VERIFICATION TESTS PASSED SUCCESSFULLY! (VITAL_MAX_HP = 6)")
    print("================================================================================")
