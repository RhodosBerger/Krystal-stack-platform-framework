# ==============================================================================
# KRYSTAL-STACK: UNIT & INTEGRATION TESTS FOR HYBRID MCP ANTIGRAVITY CONNECTOR
# ==============================================================================
import unittest
import json
import math
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from krystal_web_hub.hybrid_mcp_antigravity_connector import (
    HybridMcpAntigravityConnector,
    calculate_dihedral_manifold,
    compute_state_blake2b,
    VITAL_MAX_HP
)


class TestHybridMcpAntigravityConnector(unittest.TestCase):
    def setUp(self):
        self.connector = HybridMcpAntigravityConnector()

    # ─── 1. CORE INVARIANT & MATHEMATICAL CALCULATIONS ───────────────────────

    def test_01_vital_max_hp_invariant(self):
        """Strictly asserts that platform-wide invariant VITAL_MAX_HP is 6."""
        self.assertEqual(VITAL_MAX_HP, 6, "Inviolable system invariant: VITAL_MAX_HP must equal 6")

    def test_02_invariant_gatekeeper(self):
        """Verifies dual-gate rejection of out-of-bound state modifications."""
        # Valid state
        valid, err = self.connector.verify_invariants({"hero_hp": 6, "mirror_folds": 8})
        self.assertTrue(valid)
        self.assertIsNone(err)

        # Illegal HP > 6
        bad_hp, err_hp = self.connector.verify_invariants({"hero_hp": 7})
        self.assertFalse(bad_hp)
        self.assertIn("exceeds VITAL_MAX_HP", err_hp)

        # Negative HP
        bad_neg, err_neg = self.connector.verify_invariants({"hero_hp": -1})
        self.assertFalse(bad_neg)
        self.assertIn("cannot be negative", err_neg)

        # Out-of-bound folds (< 2 or > 32)
        bad_f1, _ = self.connector.verify_invariants({"mirror_folds": 1})
        bad_f33, _ = self.connector.verify_invariants({"mirror_folds": 33})
        self.assertFalse(bad_f1)
        self.assertFalse(bad_f33)

    def test_03_dihedral_manifold_solver(self):
        """Verifies mathematical properties of dihedral symmetry group D_n."""
        m6 = calculate_dihedral_manifold(folds=6, recursion_depth=16)
        self.assertEqual(m6["symmetry_group"], "D6")
        self.assertEqual(m6["group_order"], 12)  # 2 * 6 = 12
        self.assertAlmostEqual(m6["fundamental_wedge_deg"], 30.0, places=2)
        self.assertAlmostEqual(m6["fundamental_wedge_rad"], math.pi / 6, places=5)
        self.assertEqual(m6["lipschitz_constant"], 1.0)
        self.assertEqual(m6["lipschitz_status"], "CONSERVED_ISOMETRY")
        self.assertEqual(len(m6["reflection_planes"]), 6)

        # Test point folding
        m8 = calculate_dihedral_manifold(folds=8, test_point=[1.0, 0.5, 0.0])
        self.assertIsNotNone(m8["folded_test_point"])
        self.assertEqual(len(m8["folded_test_point"]), 3)

    def test_04_blake2b_state_determinism(self):
        """Verifies that state hashing is deterministic and collision-resistant."""
        s1 = {"hero_hp": 6, "mana": 100, "active_template": "OCTAHEDRON"}
        s2 = {"mana": 100, "active_template": "OCTAHEDRON", "hero_hp": 6}
        s3 = {"hero_hp": 5, "mana": 100, "active_template": "OCTAHEDRON"}

        h1 = compute_state_blake2b(s1)
        h2 = compute_state_blake2b(s2)
        h3 = compute_state_blake2b(s3)

        self.assertEqual(h1, h2, "Canonical JSON sorting must produce identical BLAKE2b hashes")
        self.assertNotEqual(h1, h3, "Different states must yield distinct hashes")
        self.assertEqual(len(h1), 64, "BLAKE2b hex string must be 64 characters")

    # ─── 2. JSON-RPC 2.0 PROTOCOL VERIFICATION ───────────────────────────────

    def test_05_jsonrpc_initialize(self):
        """Verifies JSON-RPC 2.0 handshake initialization."""
        req = {
            "jsonrpc": "2.0",
            "id": 101,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {"name": "antigravity-ide", "version": "2.0.0"}
            }
        }
        res = self.connector.handle_jsonrpc_request(req)
        self.assertEqual(res["jsonrpc"], "2.0")
        self.assertEqual(res["id"], 101)
        self.assertIn("result", res)
        self.assertEqual(res["result"]["serverInfo"]["name"], "krystal-antigravity-hybrid-mcp")
        self.assertEqual(res["result"]["protocolVersion"], "2024-11-05")
        self.assertTrue(res["result"]["capabilities"]["tools"]["listChanged"])

    def test_06_jsonrpc_notifications_and_ping(self):
        """Verifies notification swallowing and ping responses."""
        # Notification should return None (no response on wire)
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        res_notif = self.connector.handle_jsonrpc_request(notif)
        self.assertIsNone(res_notif)

        # Ping should return empty result
        ping = {"jsonrpc": "2.0", "id": 102, "method": "ping"}
        res_ping = self.connector.handle_jsonrpc_request(ping)
        self.assertEqual(res_ping["result"], {})

    def test_07_jsonrpc_tools_list_enumeration(self):
        """Verifies tools/list returns all 12 tools with valid schemas."""
        req = {"jsonrpc": "2.0", "id": 103, "method": "tools/list", "params": {}}
        res = self.connector.handle_jsonrpc_request(req)
        tools = res["result"]["tools"]
        tool_names = [t["name"] for t in tools]

        # Antigravity specific
        self.assertIn("antigravity-compose-template", tool_names)
        self.assertIn("antigravity-sync-state", tool_names)
        self.assertIn("antigravity-evaluate-manifold", tool_names)
        self.assertIn("antigravity-export-vulkan-glsl", tool_names)
        self.assertIn("antigravity-bridge-status", tool_names)

        # Krystal core & extensions
        self.assertIn("krystal-get-instructions", tool_names)
        self.assertIn("krystal-discover-primitives", tool_names)
        self.assertIn("krystal-declarative-to-scene", tool_names)
        self.assertIn("krystal-insert-theme-tokens", tool_names)
        self.assertIn("krystal-bind-ledger-data", tool_names)
        self.assertIn("krystal-preview-scene", tool_names)
        self.assertIn("krystal-bridge-to-oxygen", tool_names)
        self.assertIn("krystal-read-grams", tool_names)
        self.assertIn("krystal-evaluate-impedance", tool_names)
        self.assertEqual(len(tools), 14)

        for t in tools:
            self.assertIn("inputSchema", t)
            self.assertEqual(t["inputSchema"]["type"], "object")

    # ─── 3. ANTIGRAVITY TOOL EXECUTION ───────────────────────────────────────

    def test_08_antigravity_compose_template(self):
        """Tests natural language prompt composition into dihedral D6 geometry."""
        req = {
            "jsonrpc": "2.0",
            "id": 104,
            "method": "tools/call",
            "params": {
                "name": "antigravity-compose-template",
                "arguments": {
                    "prompt": "zrkadli 6-uholníkovú posvätnú geometriu v štýle alchýmie",
                    "export_vulkan": True,
                    "ascii_preview": True
                }
            }
        }
        res = self.connector.handle_jsonrpc_request(req)
        self.assertFalse(res["result"]["isError"])
        content_text = res["result"]["content"][0]["text"]
        payload = json.loads(content_text)

        self.assertEqual(payload["status"], "TEMPLATE_COMPOSED")
        self.assertEqual(payload["mirror_folds"], 6)
        self.assertIn("D6", payload["symmetry_group"])
        self.assertIn("vulkan_glsl_code", payload)
        self.assertIn("layout(push_constant)", payload["vulkan_glsl_code"])
        self.assertIn("ascii_frame", payload)

    def test_09_antigravity_sync_state_bidirectional(self):
        """Verifies state synchronization, clock increments, and invariant gate."""
        # 1. Valid Sync
        sync_req = {
            "jsonrpc": "2.0",
            "id": 105,
            "method": "tools/call",
            "params": {
                "name": "antigravity-sync-state",
                "arguments": {
                    "client_state": {
                        "hero_hp": 6,
                        "mana": 75,
                        "mirror_folds": 8,
                        "theme_tokens": {"--accent-mcp": "#00ffcc"}
                    }
                }
            }
        }
        res = self.connector.handle_jsonrpc_request(sync_req)
        payload = json.loads(res["result"]["content"][0]["text"])
        self.assertEqual(payload["status"], "SYNCHRONIZED")
        self.assertGreater(payload["vector_clock"], 1)
        self.assertEqual(payload["state"]["hero_hp"], 6)
        self.assertEqual(payload["state"]["mana"], 75)
        self.assertEqual(payload["state"]["mirror_folds"], 8)
        self.assertIn("--accent-mcp", payload["state"]["theme_tokens"])

        # 2. Invariant Fault (HP = 8 > 6)
        bad_sync = {
            "jsonrpc": "2.0",
            "id": 106,
            "method": "tools/call",
            "params": {
                "name": "antigravity-sync-state",
                "arguments": {
                    "client_state": {"hero_hp": 8}
                }
            }
        }
        bad_res = self.connector.handle_jsonrpc_request(bad_sync)
        bad_payload = json.loads(bad_res["result"]["content"][0]["text"])
        self.assertEqual(bad_payload["status"], "SYNC_FAILED")
        self.assertEqual(bad_payload["code"], -32000)
        self.assertIn("exceeds VITAL_MAX_HP", bad_payload["error"])

    def test_10_antigravity_bridge_status_telemetry(self):
        """Verifies operational telemetry and transport registration."""
        res = self.connector.execute_tool("antigravity-bridge-status", {})
        self.assertEqual(res["status"], "ONLINE")
        self.assertEqual(res["invariants"]["VITAL_MAX_HP"], 6)
        self.assertEqual(res["invariants"]["status"], "PASSING")
        self.assertIn("stdio_jsonrpc_2.0", res["transports"])
        self.assertIn("http_sse", res["transports"])
        self.assertGreaterEqual(res["uptime_seconds"], 0.0)

    def test_11_krystal_core_tool_delegation(self):
        """Verifies delegation to Krystal MCP protocol server tools."""
        # get-instructions
        inst = self.connector.execute_tool("krystal-get-instructions", {})
        self.assertIn("instructions", inst)
        self.assertEqual(inst["game_rules"]["max_hp"], 6)

        # declarative-to-scene
        scene = self.connector.execute_tool("krystal-declarative-to-scene", {
            "prompt": "Aréna hybridného MCP testu",
            "inject_buildings": ["aether_conduit"]
        })
        self.assertEqual(scene["status"], "SCENE_COMPILED")
        self.assertIn("Building_aether_conduit", scene["created_injections"])

    def test_12_stdio_transport_loop(self):
        """Verifies end-to-end stdio transport loop using stream redirection."""
        import io
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        try:
            req_json = json.dumps({
                "jsonrpc": "2.0",
                "id": 999,
                "method": "tools/call",
                "params": {
                    "name": "antigravity-evaluate-manifold",
                    "arguments": {"folds": 6}
                }
            }) + "\n"
            sys.stdin = io.StringIO(req_json)
            sys.stdout = io.StringIO()

            self.connector.run_stdio_loop()

            output = sys.stdout.getvalue()
            self.assertTrue(bool(output), "Expected response line from stdio MCP server")
            resp = json.loads(output.strip())
            self.assertEqual(resp["id"], 999)
            content = json.loads(resp["result"]["content"][0]["text"])
            self.assertEqual(content["status"], "MANIFOLD_EVALUATED")
            self.assertEqual(content["manifold"]["symmetry_group"], "D6")
            self.assertEqual(content["manifold"]["group_order"], 12)
        finally:
            sys.stdin = old_stdin
            sys.stdout = old_stdout


if __name__ == "__main__":
    unittest.main()

