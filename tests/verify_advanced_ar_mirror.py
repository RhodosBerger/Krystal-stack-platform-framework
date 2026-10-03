"""
Krystal-Stack Automated Verification: Advanced AR Mirror & Antigravity Engine
=============================================================================
Tests all components:
1. Instance Catalog Loading & Composition Rules
2. Dihedral Symmetry Folds & Golden Harmonic Scaling
3. Antigravity Natural Language Prompt Parsing (EN & SK)
4. Vulkan GLSL Uniform Code Generation
5. Godot 4.x Recursive Mirror Shader Syntax Integrity
6. Live HTTP Endpoints on Localhost Hub:
   - GET  /api/health
   - GET  /api/instances
   - POST /api/antigravity/prompt
   - POST /api/templates/compose
   - GET  /api/active-template
   - POST /api/control (RECURSIVE_MIRROR mode switch)
"""

import sys
import os
import json
import urllib.request
import urllib.parse

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def run_tests():
    print("=" * 70)
    print(" [TEST SUITE] ADVANCED RECURSIVE AR MIRROR & ANTIGRAVITY ENGINE")
    print("=" * 70)

    # ── Test 1: InstanceManager ──────────────────────────────────────────
    print("\n[TEST 1] Verifying InstanceManager & Catalog Integrity...")
    from instances.instance_manager import InstanceManager
    mgr = InstanceManager()
    geoms = mgr.list_geometric_instances()
    arts = mgr.list_artistic_instances()
    assert len(geoms) >= 7, f"Expected >= 7 geometric instances, got {len(geoms)}"
    assert len(arts) >= 5, f"Expected >= 5 artistic instances, got {len(arts)}"
    print(f"  -> Found {len(geoms)} Geometric instances.")
    print(f"  -> Found {len(arts)} Artistic presets.")

    # ── Test 2: Composition & Dihedral Folds ─────────────────────────────
    print("\n[TEST 2] Verifying Composition Rules & Dihedral Symmetries...")
    tpl = mgr.compose_template("PLATONIC_OCTAHEDRON", "CYBERPUNK_NEON_AR", mirror_folds=8)
    assert tpl["composition_rules"]["mirror_folds"] == 8
    assert "D8" in tpl["composition_rules"]["symmetry_group"]
    assert "u_mirror_folds" in tpl["vulkan_shader_uniforms"]
    ascii_out = mgr.render_ascii_frame(tpl, t=0.5, width=48, height=12)
    assert len(ascii_out.split('\n')) == 12
    print(f"  -> Template '{tpl['name']}' synthesized with D8 symmetry.")
    print(f"  -> ASCII frame rendered successfully (12 lines).")

    # ── Test 3: Antigravity Prompt Engine ────────────────────────────────
    print("\n[TEST 3] Verifying Antigravity Prompt Engine (SK/EN semantics)...")
    from antigravity_prompt_engine import AntigravityPromptEngine
    engine = AntigravityPromptEngine()
    
    sk_prompt = "zrkadli 6-uholníkovú posvätnú geometriu v štýle alchýmie"
    parsed_sk = engine.parse_prompt(sk_prompt)
    assert parsed_sk["composition_rules"]["mirror_folds"] == 6
    assert parsed_sk["artistic_instance"]["id"] == "MONASTIC_ALCHEMICAL"
    print(f"  -> SK prompt: '{sk_prompt}' => {parsed_sk['name']} (Folds: 6)")

    en_prompt = "recursive Menger sponge with 8-fold dihedral mirror in cyberpunk neon"
    parsed_en = engine.parse_prompt(en_prompt)
    assert parsed_en["geometric_instance"]["id"] == "MENGER_SPONGE_RECURSIVE"
    assert parsed_en["artistic_instance"]["id"] == "CYBERPUNK_NEON_AR"
    assert parsed_en["composition_rules"]["mirror_folds"] == 8
    print(f"  -> EN prompt: '{en_prompt}' => {parsed_en['name']} (Folds: 8)")

    glsl = engine.export_vulkan_glsl_constants(parsed_en)
    assert "AntigravityARBlock" in glsl
    assert "u_mirror_folds" in glsl
    print("  -> Vulkan GLSL uniform block generated correctly.")

    # ── Test 4: Godot 4.x Shader Integrity ──────────────────────────────
    print("\n[TEST 4] Verifying Godot 4.x Recursive Mirror Shader...")
    shader_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "godot_project", "shaders", "recursive_ar_mirror.gdshader")
    assert os.path.exists(shader_path), f"Shader not found at {shader_path}"
    with open(shader_path, "r", encoding="utf-8") as f:
        shader_content = f.read()
    assert "shader_type canvas_item;" in shader_content
    assert "u_mirror_folds" in shader_content
    assert "fold_dihedral" in shader_content
    assert "u_fresnel_factor" in shader_content
    print(f"  -> Shader verified: {len(shader_content)} bytes, all uniforms and folding functions present.")

    # ── Test 5: Live Localhost Hub HTTP Endpoints ────────────────────────
    print("\n[TEST 5] Verifying Live Localhost Hub HTTP Endpoints (http://127.0.0.1:8080)...")
    base_url = "http://127.0.0.1:8080"

    # 5a. Health
    req = urllib.request.Request(f"{base_url}/api/health")
    with urllib.request.urlopen(req, timeout=3) as resp:
        health_data = json.loads(resp.read().decode("utf-8"))
        assert health_data.get("status") == "HEALTHY"
        print("  -> GET /api/health: OK (HEALTHY)")

    # 5b. Instances Catalog
    req = urllib.request.Request(f"{base_url}/api/instances")
    with urllib.request.urlopen(req, timeout=3) as resp:
        inst_data = json.loads(resp.read().decode("utf-8"))
        assert inst_data.get("status") == "OK"
        assert len(inst_data.get("geometric", [])) >= 7
        assert len(inst_data.get("artistic", [])) >= 5
        print(f"  -> GET /api/instances: OK ({len(inst_data['geometric'])} geoms, {len(inst_data['artistic'])} arts)")

    # 5c. Antigravity Prompt POST
    prompt_payload = json.dumps({"prompt": "Calabi-Yau 6D cross section with quantum holographic laser interference"}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/api/antigravity/prompt", data=prompt_payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as resp:
        ag_resp = json.loads(resp.read().decode("utf-8"))
        assert ag_resp.get("status") == "SUCCESS"
        assert "Calabi-Yau" in ag_resp["template"]["name"]
        print(f"  -> POST /api/antigravity/prompt: OK ({ag_resp['template']['name']})")

    # 5d. Template Manual Compose POST
    compose_payload = json.dumps({
        "geometric_id": "TORUS_KNOT_P3_Q5",
        "artistic_id": "BLUEPRINT_SCHEMATIC",
        "mirror_folds": 6,
        "recursion_depth": 16
    }).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/api/templates/compose", data=compose_payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as resp:
        comp_resp = json.loads(resp.read().decode("utf-8"))
        assert comp_resp.get("status") == "SUCCESS"
        print(f"  -> POST /api/templates/compose: OK ({comp_resp['template']['name']})")

    # 5e. Active Template Verification
    req = urllib.request.Request(f"{base_url}/api/active-template")
    with urllib.request.urlopen(req, timeout=3) as resp:
        act_resp = json.loads(resp.read().decode("utf-8"))
        assert act_resp.get("status") == "OK"
        assert act_resp["active_template"]["composition_rules"]["mirror_folds"] == 6
        print("  -> GET /api/active-template: OK (Active template synced in EngineState)")

    # 5f. Control Mode Switch to RECURSIVE_MIRROR
    ctrl_payload = json.dumps({"mode": "RECURSIVE_MIRROR", "mirror_folds": 8}).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/api/control", data=ctrl_payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3) as resp:
        ctrl_resp = json.loads(resp.read().decode("utf-8"))
        assert ctrl_resp.get("mode") == "RECURSIVE_MIRROR"
        print("  -> POST /api/control: OK (Engine mode set to RECURSIVE_MIRROR)")

    print("\n" + "=" * 70)
    print(" [SUCCESS] ALL 5 ADVANCED VERIFICATION TEST SUITES PASSED (100%)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
