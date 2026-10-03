# ==============================================================================
# KRYSTAL-STACK: NATIVE MCP SERVER & OXYGEN CONNECTOR ARCHITECTURE
# ==============================================================================
# Inspired by the architectural patterns of the Oxygen Builder MCP Connector:
# 1. Discovery-First Pre-flight (get-instructions, discover-primitives)
# 2. Declarative Monolithic Assembly (declarative-to-scene with 3D markers)
# 3. Design Token Separation (insert-theme-tokens, HyperOS 4 variables)
# 4. Dynamic Data Binding (bind-ledger-data with typed contracts)
# 5. Component Loops & Grid Instantiation (3D loops in raw mode)
# 6. Conditional Logic & Phase Unlocks (Disjunctive Normal Form rule groups)
# 7. Verification & Bidirectional Preview (preview-scene, Godot .tscn export)
# 8. Surgical Escape Hatches (edit-node)
# ==============================================================================

import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple

from .krystal_engine_core import (
    generate_scene_ast_internal, export_scene_to_tscn,
    KMEN_CARDS, GAME_STATE, ASSET_DIR
)
from .economic_engine import (
    Tribe, AttackType, AbilityType, RoundPhase, EscalationStage,
    BUILDING_REGISTRY, ABILITY_REGISTRY, RoundController,
    EconomicLedger, MatchState, calculate_hex_distance, validate_target_range
)

class KrystalMcpProtocolServer:
    """
    Model Context Protocol (MCP) Server for the Krystal-Stack Platform.
    Exposes high-level tools to AI agents for procedural 3D scene generation,
    economic combat balancing, Godot 4 export, and WordPress/Oxygen mirroring.
    """
    def __init__(self):
        self.server_name = "krystal-builder-mcp"
        self.version = "1.0.0"
        self.active_scene_ast: Optional[Dict[str, Any]] = None
        self.theme_tokens: Dict[str, Any] = {
            "--hyper-glass": "rgba(18, 22, 34, 0.72)",
            "--hyper-blur": "blur(28px) saturate(190%) contrast(105%)",
            "--accent-cyan": "#66fcf1",
            "--neon-green": "#39ff14",
            "--druid-gold": "#ffd700",
            "--alert-red": "#ff453a",
            "--font-title": "Cinzel",
            "--font-code": "Fira Code"
        }
        self.tool_registry = {
            "krystal-get-instructions": self.handle_get_instructions,
            "krystal-discover-primitives": self.handle_discover_primitives,
            "krystal-declarative-to-scene": self.handle_declarative_to_scene,
            "krystal-insert-theme-tokens": self.handle_insert_theme_tokens,
            "krystal-bind-ledger-data": self.handle_bind_ledger_data,
            "krystal-set-spatial-conditions": self.handle_set_spatial_conditions,
            "krystal-preview-scene": self.handle_preview_scene,
            "krystal-edit-node": self.handle_edit_node,
            "krystal-bridge-to-oxygen": self.handle_bridge_to_oxygen
        }

    # --------------------------------------------------------------------------
    # 1. DISCOVERY FIRST: PRE-FLIGHT INSTRUCTIONS & SCHEMAS
    # --------------------------------------------------------------------------
    def handle_get_instructions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Returns mandatory pre-flight instructions and architectural guidelines for AI agents."""
        return {
            "instructions": (
                "Krystal-Stack 3D & Economic Builder MCP Server.\n\n"
                "## CORE ARCHITECTURAL PATTERNS (Patterned after Oxygen Builder MCP):\n"
                "1. PREFERRED BUILD PATH: Author new 3D scenes declaratively via 'krystal-declarative-to-scene' "
                "with high-level spatial markers (k-loop, k-mesh, k-light, k-anim). Do NOT assemble complex 3D worlds "
                "node-by-node across multiple calls.\n"
                "2. DESIGN TOKENS: Ingest visual materials, lighting palettes, and HyperOS 4 styling tokens via "
                "'krystal-insert-theme-tokens' before building instances.\n"
                "3. DYNAMIC DATA BINDINGS: Never hardcode economy balances or hero health. Use 'krystal-bind-ledger-data' "
                "to link 3D billboard HUDs and status cards directly to the double-entry accounting ledger.\n"
                "4. POSLEDNÍ KMEN RULE CONTRACTS: Hero HP is strictly clamped in [0, 6]. Axial hex distance is governed by "
                "D(H1, H2) = (|dq| + |dq+dr| + |dr|) / 2. Melee attacks require range 1; ranged shots enforce min/max bounds.\n"
                "5. VERIFICATION: Verify scene builds with 'krystal-preview-scene' before concluding. It returns the compiled "
                "Godot 4 .tscn text, SceneTree hierarchy, and node counts.\n"
                "6. ESCAPE HATCH: Use 'krystal-edit-node' only for fine-grained property tweaks and surgical node repositioning."
            ),
            "version": self.version,
            "game_rules": {
                "max_hp": 6,
                "tribes": ["crystal", "toxic", "druid"],
                "accounting": "Double-entry dual-earn ledger",
                "escalation_stages": ["skirmish", "industrial_surge", "total_war_apex", "cataclysm"]
            }
        }

    def handle_discover_primitives(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Discovers available 3D spatial node primitives, attributes, and baked Godot assets."""
        available_assets = []
        if os.path.exists(ASSET_DIR):
            available_assets = os.listdir(ASSET_DIR)

        return {
            "primitives": {
                "Spatial": {
                    "description": "Base 3D transform node in Godot. Roots subtrees and positions groups in R^3.",
                    "properties": ["position", "rotation", "scale"]
                },
                "MeshInstance": {
                    "description": "Renders a 3D geometry mesh with PBR material shaders.",
                    "properties": ["mesh", "material", "color", "scale", "position"]
                },
                "OmniLight": {
                    "description": "Point light source radiating energy in 3D space.",
                    "properties": ["color", "energy", "range", "position"]
                },
                "Particles": {
                    "description": "GPU particle emitter for magic spells, vapor, and impact bursts.",
                    "properties": ["amount", "color", "spread", "velocity"]
                },
                "Area": {
                    "description": "Spatial collision and interaction detection zone.",
                    "properties": ["radius", "position"]
                },
                "AnimationPlayer": {
                    "description": "Animates node properties over time (dropAnim, floatAnim, bubbleAnim).",
                    "properties": ["anim", "duration", "speed"]
                }
            },
            "assets": available_assets,
            "card_ids": list(KMEN_CARDS.keys()),
            "building_ids": list(BUILDING_REGISTRY.keys()),
            "ability_ids": list(ABILITY_REGISTRY.keys())
        }

    # --------------------------------------------------------------------------
    # 2. DECLARATIVE MONOLITHIC ASSEMBLY (Equivalent to html-to-page)
    # --------------------------------------------------------------------------
    def handle_declarative_to_scene(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compiles a high-level declarative 3D scene description into a full Godot AST in one single pass.
        Understands markers:
          - prompt: Natural language scene prompt (e.g. 'Taktická aréna 3 kmeňov')
          - layout_type: 'hex_arena' | 'conduit_complex' | 'sacred_grove' | 'toxic_refinery'
          - inject_buildings: List of building IDs to instantiate on the grid
          - inject_spells: List of spell FX to position on coordinates
        """
        prompt = args.get("prompt", "Taktická Aréna Poslední Kmen")
        layout_type = args.get("layout_type", "hex_arena")
        inject_buildings = args.get("inject_buildings", [])
        inject_spells = args.get("inject_spells", [])

        # Compile base scene AST using embedded Janet AST compiler
        scene_data = generate_scene_ast_internal(prompt)
        tree = scene_data.get("tree", {})

        # Process Declarative Building Injections (Equivalent to bd-loop / bd-woo)
        created_nodes = []
        for b_id in inject_buildings:
            if b_id in BUILDING_REGISTRY:
                b_spec = BUILDING_REGISTRY[b_id]
                b_node = {
                    "type": "Spatial",
                    "name": f"Building_{b_spec.id}",
                    "properties": {"position": [0.0, 0.2, 0.0], "scale": [1.0, 1.0, 1.0]},
                    "children": [
                        {
                            "type": "MeshInstance",
                            "name": f"Mesh_{b_spec.id}",
                            "properties": {
                                "mesh": f"res://godot_assets/{b_spec.mesh_asset}",
                                "color": b_spec.color,
                                "scale": [1.2, 1.4, 1.2]
                            },
                            "children": []
                        },
                        {
                            "type": "OmniLight",
                            "name": f"Light_{b_spec.id}",
                            "properties": {"color": b_spec.color, "energy": 2.2, "range": 6.0},
                            "children": []
                        }
                    ]
                }
                tree.setdefault("children", []).append(b_node)
                created_nodes.append(b_node["name"])

        # Process Declarative Spell Injections
        for s_id in inject_spells:
            if s_id in KMEN_CARDS:
                card = KMEN_CARDS[s_id]
                s_node = json.loads(json.dumps(card["mesh_node"]))
                tree.setdefault("children", []).append(s_node)
                created_nodes.append(s_node["name"])

        # Re-export Godot 4 .tscn code
        scene_data["tree"] = tree
        scene_data["tscn"] = export_scene_to_tscn(scene_data)
        self.active_scene_ast = scene_data

        return {
            "status": "SCENE_COMPILED",
            "prompt": prompt,
            "root_name": tree.get("name", "WorldRoot"),
            "node_count": self._count_nodes(tree),
            "created_injections": created_nodes,
            "tscn_preview": scene_data["tscn"][:300] + "..."
        }

    # --------------------------------------------------------------------------
    # 3. DESIGN TOKEN INGESTION (Equivalent to insert-stylesheet & CSS variables)
    # --------------------------------------------------------------------------
    def handle_insert_theme_tokens(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Ingests visual style tokens, PBR shader parameters, and HyperOS 4 styling variables."""
        tokens = args.get("tokens", {})
        for k, v in tokens.items():
            self.theme_tokens[k] = v

        return {
            "status": "TOKENS_UPDATED",
            "active_token_count": len(self.theme_tokens),
            "tokens": self.theme_tokens
        }

    # --------------------------------------------------------------------------
    # 4. DYNAMIC DATA BINDINGS (Equivalent to bd-bind & companion dynamic meta)
    # --------------------------------------------------------------------------
    def handle_bind_ledger_data(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Binds 3D HUD texts, health bars, and indicators directly to the live dual-earn ledger."""
        node_name = args.get("node_name", "PlayerHUD")
        data_field = args.get("data_field", "player_hp")
        fallback_value = args.get("fallback", "6 HP")

        # Map to live values
        value = fallback_value
        if data_field in GAME_STATE:
            value = str(GAME_STATE[data_field])

        dynamic_meta = {
            "field": data_field,
            "return_type": "string" if "hp" in data_field or "mana" in data_field else "number",
            "source": "KrystalEconomicLedger",
            "current_value": value
        }

        return {
            "status": "BINDING_ACTIVE",
            "node_name": node_name,
            "data_field": data_field,
            "bound_value": value,
            "dynamic_meta": dynamic_meta
        }

    # --------------------------------------------------------------------------
    # 5. CONDITIONAL LOGIC & PHASE UNLOCKS (Equivalent to set-element-conditions)
    # --------------------------------------------------------------------------
    def handle_set_spatial_conditions(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sets conditional visibility or activation locks using Disjunctive Normal Form rule groups:
        rule_groups: [[ { "field": "escalation", "op": "==", "value": "total_war_apex" } ]]
        """
        node_name = args.get("node_name")
        rule_groups = args.get("rule_groups", [])

        # Evaluate conditions against live match state
        match = RoundController.initialize_match("cond_eval", Tribe.CRYSTAL, Tribe.TOXIC)
        current_escalation = match.escalation.value

        is_visible = False
        for group in rule_groups:
            group_match = True
            for rule in group:
                f = rule.get("field")
                op = rule.get("op")
                val = rule.get("value")

                if f == "escalation":
                    if op == "==" and current_escalation != val: group_match = False
                    elif op == "!=" and current_escalation == val: group_match = False
                elif f == "player_mana":
                    if op == ">=" and match.player.mana < int(val): group_match = False
            if group_match:
                is_visible = True
                break

        return {
            "status": "CONDITIONS_CONFIGURED",
            "node_name": node_name,
            "rule_groups": rule_groups,
            "evaluated_visibility": is_visible,
            "reason": f"Evaluated against escalation '{current_escalation}' and mana {match.player.mana}"
        }

    # --------------------------------------------------------------------------
    # 6. VERIFICATION & PREVIEW (Equivalent to preview-post & get-post-tree)
    # --------------------------------------------------------------------------
    def handle_preview_scene(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Returns verification preview of the active scene AST, node counts, and Godot 4 text format."""
        if not self.active_scene_ast:
            # Generate default tactical arena if none active
            self.active_scene_ast = generate_scene_ast_internal("Taktická Aréna Poslední Kmen")
            self.active_scene_ast["tscn"] = export_scene_to_tscn(self.active_scene_ast)

        tree = self.active_scene_ast.get("tree", {})
        return {
            "status": "PREVIEW_READY",
            "node_count": self._count_nodes(tree),
            "hierarchy_summary": self._summarize_tree(tree),
            "godot_tscn_length": len(self.active_scene_ast.get("tscn", "")),
            "godot_tscn_snippet": self.active_scene_ast.get("tscn", "")[:500]
        }

    # --------------------------------------------------------------------------
    # 7. SURGICAL ESCAPE HATCH (Equivalent to edit-post / set-element-*)
    # --------------------------------------------------------------------------
    def handle_edit_node(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Surgically modifies properties of a specific node in the active scene tree."""
        if not self.active_scene_ast:
            return {"error": "No active scene loaded. Call 'krystal-declarative-to-scene' first."}

        target_name = args.get("node_name")
        properties_to_update = args.get("properties", {})

        found = self._update_node_recursive(self.active_scene_ast.get("tree", {}), target_name, properties_to_update)
        if found:
            self.active_scene_ast["tscn"] = export_scene_to_tscn(self.active_scene_ast)
            return {
                "status": "NODE_UPDATED",
                "node_name": target_name,
                "updated_properties": properties_to_update
            }
        return {"error": f"Node '{target_name}' not found in active SceneTree."}

    # --------------------------------------------------------------------------
    # 8. BIDIRECTIONAL CONNECTOR BRIDGE TO OXYGEN BUILDER (nabytok47)
    # --------------------------------------------------------------------------
    def handle_bridge_to_oxygen(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats Krystal game telemetry, dual-earn ledger transactions, and invoice MRP records
        into exact Oxygen Builder MCP payloads (matching oxygen-create-post, oxygen-create-template).
        """
        payload_type = args.get("payload_type", "economic_summary")
        context_data = args.get("context_data", {})

        if payload_type == "invoice":
            inv_uuid = context_data.get("uuid", "inv-mock-001")
            amount = context_data.get("amount", "125.50 EUR")
            supplier = context_data.get("supplier", "Krystal Crystalline s.r.o.")
            oxygen_payload = {
                "tool_to_call": "oxygen-create-post",
                "arguments": {
                    "title": f"Faktúra // {inv_uuid}",
                    "post_type": "accounting_invoice",
                    "status": "publish",
                    "meta": {
                        "invoice_amount": amount,
                        "invoice_supplier": supplier,
                        "scan_timestamp": int(time.time()),
                        "thermodynamic_status": "VERIFIED_LEDGER"
                    }
                }
            }
        else: # economic_summary
            oxygen_payload = {
                "tool_to_call": "oxygen-insert-stylesheet",
                "arguments": {
                    "stylesheet_name": "krystal-hyperos-theme",
                    "css": (
                        ":root {\n"
                        f"  --krystal-cyan: {self.theme_tokens.get('--accent-cyan')};\n"
                        f"  --krystal-neon: {self.theme_tokens.get('--neon-green')};\n"
                        f"  --krystal-gold: {self.theme_tokens.get('--druid-gold')};\n"
                        f"  --krystal-red: {self.theme_tokens.get('--alert-red')};\n"
                        f"  --krystal-glass: {self.theme_tokens.get('--hyper-glass')};\n"
                        "}\n"
                        ".krystal-card { backdrop-filter: blur(28px); border-radius: 20px; border: 1px solid rgba(255,255,255,0.12); }"
                    )
                }
            }

        return {
            "status": "BRIDGE_PAYLOAD_PREPARED",
            "target_mcp_server": "nabytok47",
            "target_tool": oxygen_payload["tool_to_call"],
            "payload": oxygen_payload["arguments"]
        }

    # --------------------------------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------------------------------
    def dispatch_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an MCP tool call to the registered handler."""
        if tool_name not in self.tool_registry:
            return {"error": f"Tool '{tool_name}' not registered in {self.server_name}."}
        try:
            return self.tool_registry[tool_name](arguments)
        except Exception as e:
            return {"error": f"Internal execution error in {tool_name}: {str(e)}"}

    def _count_nodes(self, node: Dict[str, Any]) -> int:
        count = 1
        for child in node.get("children", []):
            count += self._count_nodes(child)
        return count

    def _summarize_tree(self, node: Dict[str, Any], depth: int = 0) -> List[str]:
        lines = [f"{'  ' * depth}[{node.get('type')}] {node.get('name')}"]
        for child in node.get("children", [])[:4]: # summarize top children
            lines.extend(self._summarize_tree(child, depth + 1))
        return lines

    def _update_node_recursive(self, node: Dict[str, Any], target_name: str, new_props: Dict[str, Any]) -> bool:
        if node.get("name") == target_name:
            node.setdefault("properties", {}).update(new_props)
            return True
        for child in node.get("children", []):
            if self._update_node_recursive(child, target_name, new_props):
                return True
        return False
