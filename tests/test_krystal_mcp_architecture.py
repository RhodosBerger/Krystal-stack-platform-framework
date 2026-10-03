# ==============================================================================
# KRYSTAL-STACK: UNIT TESTS FOR NATIVE MCP SERVER & OXYGEN CONNECTOR ARCHITECTURE
# ==============================================================================
import unittest
import json
from krystal_web_hub.krystal_mcp_server import KrystalMcpProtocolServer

class TestKrystalMcpArchitecture(unittest.TestCase):
    def setUp(self):
        self.server = KrystalMcpProtocolServer()

    def test_01_tool_registration(self):
        registered = list(self.server.tool_registry.keys())
        expected_tools = [
            "krystal-get-instructions",
            "krystal-discover-primitives",
            "krystal-declarative-to-scene",
            "krystal-insert-theme-tokens",
            "krystal-bind-ledger-data",
            "krystal-set-spatial-conditions",
            "krystal-preview-scene",
            "krystal-edit-node",
            "krystal-bridge-to-oxygen"
        ]
        for tool in expected_tools:
            self.assertIn(tool, registered, f"Tool '{tool}' was not registered in MCP server.")

    def test_02_get_instructions_contract(self):
        res = self.server.dispatch_call("krystal-get-instructions", {})
        self.assertIn("instructions", res)
        self.assertIn("game_rules", res)
        self.assertEqual(res["game_rules"]["max_hp"], 6)
        self.assertIn("escalation_stages", res["game_rules"])

    def test_03_discover_primitives(self):
        res = self.server.dispatch_call("krystal-discover-primitives", {})
        prims = res.get("primitives", {})
        self.assertIn("Spatial", prims)
        self.assertIn("MeshInstance", prims)
        self.assertIn("OmniLight", prims)
        self.assertIn("Particles", prims)
        
        self.assertIn("crystal_meteor", res["card_ids"])
        self.assertIn("druid_strike", res["card_ids"])
        self.assertIn("decay_strike", res["card_ids"])

    def test_04_declarative_to_scene_single_pass(self):
        res = self.server.dispatch_call("krystal-declarative-to-scene", {
            "prompt": "Taktická aréna s konduitom a meteorom",
            "inject_buildings": ["aether_conduit"],
            "inject_spells": ["crystal_meteor"]
        })
        self.assertEqual(res["status"], "SCENE_COMPILED")
        self.assertGreater(res["node_count"], 15)
        self.assertIn("Building_aether_conduit", res["created_injections"])
        self.assertIn("Spell_CrystalMeteor", res["created_injections"])
        self.assertIn("gd_scene", res["tscn_preview"])

    def test_05_insert_theme_tokens(self):
        res = self.server.dispatch_call("krystal-insert-theme-tokens", {
            "tokens": {
                "--custom-accent": "#ff007f",
                "--custom-blur": "blur(32px)"
            }
        })
        self.assertEqual(res["status"], "TOKENS_UPDATED")
        self.assertEqual(res["tokens"]["--custom-accent"], "#ff007f")
        self.assertEqual(res["tokens"]["--custom-blur"], "blur(32px)")

    def test_06_bind_ledger_data(self):
        res = self.server.dispatch_call("krystal-bind-ledger-data", {
            "node_name": "HealthIndicator",
            "data_field": "player_hp"
        })
        self.assertEqual(res["status"], "BINDING_ACTIVE")
        self.assertEqual(res["node_name"], "HealthIndicator")
        self.assertEqual(res["data_field"], "player_hp")
        self.assertIn("dynamic_meta", res)
        self.assertEqual(res["dynamic_meta"]["source"], "KrystalEconomicLedger")

    def test_07_set_spatial_conditions_dnf(self):
        # Match starts in 'skirmish'
        res = self.server.dispatch_call("krystal-set-spatial-conditions", {
            "node_name": "ApexPortal",
            "rule_groups": [
                [{"field": "escalation", "op": "==", "value": "total_war_apex"}],
                [{"field": "player_mana", "op": ">=", "value": 20}]
            ]
        })
        self.assertEqual(res["status"], "CONDITIONS_CONFIGURED")
        # In default skirmish state, portal should not be visible yet
        self.assertFalse(res["evaluated_visibility"])

    def test_08_preview_scene(self):
        # First ensure a scene is compiled
        self.server.dispatch_call("krystal-declarative-to-scene", {"prompt": "Mini Arena"})
        res = self.server.dispatch_call("krystal-preview-scene", {})
        self.assertEqual(res["status"], "PREVIEW_READY")
        self.assertGreater(res["node_count"], 0)
        self.assertGreater(res["godot_tscn_length"], 100)

    def test_09_edit_node_escape_hatch(self):
        # Compile scene with building
        self.server.dispatch_call("krystal-declarative-to-scene", {
            "prompt": "Aréna s vežou",
            "inject_buildings": ["capacitor_tower"]
        })
        # Modify the injected building scale
        res = self.server.dispatch_call("krystal-edit-node", {
            "node_name": "Building_capacitor_tower",
            "properties": {"scale": [2.5, 2.5, 2.5]}
        })
        self.assertEqual(res["status"], "NODE_UPDATED")
        self.assertEqual(res["updated_properties"]["scale"], [2.5, 2.5, 2.5])

    def test_10_bridge_to_oxygen_nabytok47(self):
        # 1. Invoice payload bridge
        inv_res = self.server.dispatch_call("krystal-bridge-to-oxygen", {
            "payload_type": "invoice",
            "context_data": {
                "uuid": "INV-2026-X100",
                "amount": "890.00 EUR",
                "supplier": "Bratislava IT Solutions s.r.o."
            }
        })
        self.assertEqual(inv_res["status"], "BRIDGE_PAYLOAD_PREPARED")
        self.assertEqual(inv_res["target_mcp_server"], "nabytok47")
        self.assertEqual(inv_res["target_tool"], "oxygen-create-post")
        self.assertEqual(inv_res["payload"]["meta"]["invoice_amount"], "890.00 EUR")

        # 2. Stylesheet payload bridge
        css_res = self.server.dispatch_call("krystal-bridge-to-oxygen", {
            "payload_type": "economic_summary"
        })
        self.assertEqual(css_res["target_tool"], "oxygen-insert-stylesheet")
        self.assertIn("--krystal-cyan", css_res["payload"]["css"])

if __name__ == '__main__':
    unittest.main()
