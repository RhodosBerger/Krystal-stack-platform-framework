"""
Krystal-Stack Automated Verification: Advanced AR Mirror & Antigravity Engine
=============================================================================
Converted to standard unittest.TestCase for full discoverability and CI reliability.
Verifies:
1. Instance Catalog Loading & Composition Rules
2. Dihedral Symmetry Folds (D8) & Golden Harmonic Scaling
3. Antigravity Natural Language Prompt Parsing (EN & SK semantics)
4. Vulkan GLSL Uniform Code Generation
5. Godot 4.x Recursive Mirror Shader Syntax Integrity
6. Live HTTP Endpoints on Localhost Hub (when available)
"""

import sys
import os
import json
import urllib.request
import urllib.parse
import urllib.error
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class TestAdvancedARMirror(unittest.TestCase):
    """
    Standard discoverable test suite for Advanced AR Mirror and Antigravity Engine.
    """

    def setUp(self):
        self.root_dir = ROOT_DIR

    def test_instance_database(self):
        """Verifies InstanceManager catalog integrity and minimum instance counts."""
        from instances.instance_manager import InstanceManager
        mgr = InstanceManager()
        geoms = mgr.list_geometric_instances()
        arts = mgr.list_artistic_instances()
        self.assertGreaterEqual(
            len(geoms), 7,
            f"Expected >= 7 geometric instances, got {len(geoms)}"
        )
        self.assertGreaterEqual(
            len(arts), 5,
            f"Expected >= 5 artistic instances, got {len(arts)}"
        )

    def test_dihedral_mirror_composition(self):
        """Verifies D8 dihedral mirror composition and 12-line ASCII projection."""
        from instances.instance_manager import InstanceManager
        mgr = InstanceManager()
        tpl = mgr.compose_template("PLATONIC_OCTAHEDRON", "CYBERPUNK_NEON_AR", mirror_folds=8)
        self.assertEqual(tpl["composition_rules"]["mirror_folds"], 8)
        self.assertIn("D8", tpl["composition_rules"]["symmetry_group"])
        self.assertIn("u_mirror_folds", tpl["vulkan_shader_uniforms"])

        ascii_out = mgr.render_ascii_frame(tpl, t=0.5, width=48, height=12)
        lines = ascii_out.split('\n')
        self.assertEqual(
            len(lines), 12,
            f"Expected 12 lines for ASCII frame, got {len(lines)}"
        )

    def test_antigravity_prompt_engine(self):
        """Verifies Antigravity natural language prompt parsing in Slovak and English, and GLSL block."""
        from antigravity_prompt_engine import AntigravityPromptEngine
        engine = AntigravityPromptEngine()

        # Slovak prompt verification
        sk_prompt = "zrkadli 6-uholníkovú posvätnú geometriu v štýle alchýmie"
        parsed_sk = engine.parse_prompt(sk_prompt)
        self.assertEqual(parsed_sk["composition_rules"]["mirror_folds"], 6)
        self.assertEqual(parsed_sk["artistic_instance"]["id"], "MONASTIC_ALCHEMICAL")

        # English prompt verification
        en_prompt = "recursive Menger sponge with 8-fold dihedral mirror in cyberpunk neon"
        parsed_en = engine.parse_prompt(en_prompt)
        self.assertEqual(parsed_en["geometric_instance"]["id"], "MENGER_SPONGE_RECURSIVE")
        self.assertEqual(parsed_en["artistic_instance"]["id"], "CYBERPUNK_NEON_AR")
        self.assertEqual(parsed_en["composition_rules"]["mirror_folds"], 8)

        # Vulkan GLSL uniform block generation
        glsl = engine.export_vulkan_glsl_constants(parsed_en)
        self.assertIn("AntigravityARBlock", glsl)
        self.assertIn("u_mirror_folds", glsl)

    def test_godot_shader_syntax_integrity(self):
        """Verifies Godot 4.x Recursive Mirror Shader syntax integrity."""
        shader_path = os.path.join(self.root_dir, "godot_project", "shaders", "recursive_ar_mirror.gdshader")
        self.assertTrue(os.path.exists(shader_path), f"Shader not found at {shader_path}")
        with open(shader_path, "r", encoding="utf-8") as f:
            shader_content = f.read()
        self.assertIn("shader_type canvas_item;", shader_content)
        self.assertIn("u_mirror_folds", shader_content)
        self.assertIn("fold_dihedral", shader_content)
        self.assertIn("u_fresnel_factor", shader_content)

    def test_live_hub_endpoints(self):
        """Verifies Live Localhost Hub HTTP Endpoints for AR Mirror (skipped if hub is offline)."""
        base_url = "http://127.0.0.1:8080"
        try:
            req = urllib.request.Request(f"{base_url}/api/health")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                health_data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(health_data.get("status"), "HEALTHY")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self.skipTest(f"Localhost hub at {base_url} is offline or unreachable: {e}")

        # If hub is reachable, test remaining endpoints
        # 1. Instances Catalog
        req = urllib.request.Request(f"{base_url}/api/instances")
        with urllib.request.urlopen(req, timeout=3) as resp:
            inst_data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(inst_data.get("status"), "OK")
            self.assertGreaterEqual(len(inst_data.get("geometric", [])), 7)
            self.assertGreaterEqual(len(inst_data.get("artistic", [])), 5)

        # 2. Antigravity Prompt POST
        prompt_payload = json.dumps({"prompt": "Calabi-Yau 6D cross section with quantum holographic laser interference fringes"}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/antigravity/prompt", data=prompt_payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            ag_resp = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(ag_resp.get("status"), "SUCCESS")
            self.assertIn("Calabi-Yau", ag_resp["template"]["name"])

        # 3. Template Manual Compose POST
        compose_payload = json.dumps({
            "geometric_id": "TORUS_KNOT_P3_Q5",
            "artistic_id": "BLUEPRINT_SCHEMATIC",
            "mirror_folds": 6,
            "recursion_depth": 16
        }).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/templates/compose", data=compose_payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            comp_resp = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(comp_resp.get("status"), "SUCCESS")

        # 4. Active Template Verification
        req = urllib.request.Request(f"{base_url}/api/active-template")
        with urllib.request.urlopen(req, timeout=3) as resp:
            act_resp = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(act_resp.get("status"), "OK")
            self.assertEqual(act_resp["active_template"]["composition_rules"]["mirror_folds"], 6)

        # 5. Control Mode Switch to RECURSIVE_MIRROR
        ctrl_payload = json.dumps({"mode": "RECURSIVE_MIRROR", "mirror_folds": 8}).encode("utf-8")
        req = urllib.request.Request(f"{base_url}/api/control", data=ctrl_payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            ctrl_resp = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(ctrl_resp.get("mode"), "RECURSIVE_MIRROR")


if __name__ == "__main__":
    unittest.main()
