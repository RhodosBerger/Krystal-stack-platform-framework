# KRYSTAL-STACK: MCP OXYGEN BUILDER PATTERNS & SPATIAL MCP ARCHITECTURE

**Document ID:** `KRYSTAL-RESEARCH-MCP-02`  
**Classification:** Systems Architecture & MCP Protocol Engineering  
**Author:** Dušan Kopecký & Krystal-Stack Architecture Team  
**Date:** 2026-10-03  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Executive Summary

This architecture specification synthesizes the design patterns of the **Oxygen Builder MCP Connector (`nabytok47`)** and establishes a new, unified **Model Context Protocol (MCP) Architecture** for the Krystal-Stack Platform Framework.

Instead of brittle, imperative micro-calls or unconstrained script generation, we formulate a **Dual-Plane MCP Architecture**:
1. **Plane 1: Krystal 3D Spatial & Economic Builder MCP Server (`krystal-builder-mcp`)**: Exposes our Godot 4 scene tree, Janet AST compiler, dual-earn ledger, and combat targeting system as native MCP tools to any AI model (Gemini, Claude, local models).
2. **Plane 2: Bidirectional Web & ERP Connector Bridge (`krystal-oxygen-bridge`)**: Connects Krystal's OpenCV/OCR invoice pipeline and game telemetry directly to external CMS environments (WordPress, Oxygen Builder, WooCommerce) via standard MCP payloads.

---

## 2. Learning Study: The 8 Core Patterns of the Oxygen Builder MCP Connector

Analysis of the `nabytok47` MCP tools (`oxygen-get-instructions`, `oxygen-html-to-page`, `oxygen-insert-stylesheet`, `oxygen-create-template`, `oxygen-set-element-conditions`, etc.) reveals 8 foundational patterns:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       OXYGEN BUILDER MCP PATTERN MATRIX                     │
├───────────────────────────────┬─────────────────────────────────────────────┤
│ 1. Pre-flight Discovery       │ Mandatory call to get-instructions, schemas │
│ 2. Declarative Assembly       │ html-to-page single round-trip with markers │
│ 3. Design Token Ingestion     │ :root CSS variables & insert-stylesheet     │
│ 4. Typed Dynamic Data Binding │ bd-bind with typed returns & dynamic_meta   │
│ 5. Collection Loops in Raw Mode│ bd-loop with Component extraction           │
│ 6. Conditional Logic (DNF)    │ OR-lists of AND-groups (rule_groups)        │
│ 7. Bidirectional Preview      │ preview-post & get-post-tree verification   │
│ 8. Surgical Escape Hatches    │ edit-post for fine-grained property tweaks  │
└───────────────────────────────┴─────────────────────────────────────────────┘
```

### Pattern 1: Discovery-First Contract (Pre-Flight Grounding)
- **Problem:** AI agents hallucinate element properties, field slugs, and breakpoint names when guessing APIs.
- **Oxygen Solution:** The agent *must* call `get-instructions` and discovery tools (`get-element-slugs`, `get-element-schemas`, `get-dynamic-fields`, `get-breakpoints`) before mutating state.
- **Krystal Adaptation:** `krystal-get-instructions` and `krystal-discover-primitives` enforce ground truth on available Godot nodes (`Spatial`, `MeshInstance`, `OmniLight`, `Particles`), baked assets, and 6 HP / range rules.

### Pattern 2: Declarative Monolithic Assembly (`html-to-page`)
- **Problem:** Assembling trees node-by-node across dozens of sequential tool calls causes massive latency, network overhead, and frequent context truncation.
- **Oxygen Solution:** The agent passes high-level semantic HTML + CSS with special attributes (`bd-loop`, `bd-bind`, `bd-src`, `bd-animate`). The backend parses this into the builder's internal AST in **one single pass**.
- **Krystal Adaptation:** `krystal-declarative-to-scene` receives natural language scene descriptions or Janet S-expressions with spatial markers (`k-loop="hex_grid"`, `k-mesh="crystal_shard.obj"`, `k-light="#00ffff"`) and compiles the complete Godot AST and Three.js tree in one round-trip.

### Pattern 3: Global Design Token Separation (`insert-stylesheet`)
- **Problem:** Hardcoding colors and fonts on individual components causes style fragmentation and breaks global re-theming.
- **Oxygen Solution:** Ingests design tokens into `:root` CSS variables and utility classes first; components only consume variables.
- **Krystal Adaptation:** `krystal-insert-theme-tokens` injects PBR material shaders, lighting ambiance presets, and Xiaomi HyperOS 4 design tokens (`--hyper-glass`, `--hyper-blur`, `--accent-cyan`).

### Pattern 4: Typed Dynamic Data Binding (`bd-bind` + Companion Meta)
- **Problem:** Hardcoding dynamic values breaks templates, and stripped metadata breaks the visual editor.
- **Oxygen Solution:** Binds fields with typed return contracts (`string`, `url`, `image_url`) and writes a companion `<prop>_dynamic_meta` object so the builder UI can inspect the binding.
- **Krystal Adaptation:** `krystal-bind-ledger-data` binds 3D floating texts, health bars, and HUD pills to live `EconomicLedger` fields with typed metadata (`current_value`, `source`).

### Pattern 5: Collection Loops in Raw Mode (`bd-loop`)
- **Problem:** Repeating collections by copy-pasting cards bloats DOM and destroys maintainability.
- **Oxygen Solution:** The author marks the container `bd-loop`, authors ONE card, and the converter extracts a reusable Component in raw mode (zero wrapper bloat).
- **Krystal Adaptation:** In 3D space, repetitive hex grids, building clusters, and sector garrisons are stamped from a single archetype node definition.

### Pattern 6: Disjunctive Normal Form Conditions (`rule_groups`)
- **Problem:** Hardcoding conditional visibility logic in scripts creates spaghetti code.
- **Oxygen Solution:** Uses DNF boolean expressions: OR-list of AND-groups (`[[A, B], [C]]`).
- **Krystal Adaptation:** `krystal-set-spatial-conditions` unlocks Tier 3 ultimate spells and Cataclysm sectors strictly when `match.escalation == 'total_war_apex'` and mana thresholds are satisfied.

### Pattern 7: Bidirectional Preview & Verification (`preview-post`)
- **Problem:** Blind execution leaves AI agents unaware of visual or syntax errors.
- **Oxygen Solution:** `preview-post` and `preview-element` return the rendered frontend HTML and generated CSS.
- **Krystal Adaptation:** `krystal-preview-scene` returns Godot 4 `.tscn` text, SceneTree hierarchy, and node counts for verification.

### Pattern 8: Surgical Escape Hatches (`edit-post`)
- **Problem:** If a single property needs tweaking, re-running monolithic generation is wasteful.
- **Oxygen Solution:** `edit-post` allows surgical property modification on specific element IDs.
- **Krystal Adaptation:** `krystal-edit-node` modifies individual node properties (`position`, `color`, `energy`) in the active tree.

---

## 3. New Architecture: The Krystal-Stack MCP Connection

```
                               THE DUAL-PLANE MCP ARCHITECTURE
                                              
                       +───────────────────────────────────────────────+
                       │               AI AGENT / LLM                  │
                       │     (Gemini 2.5 / Claude / Local Model)       │
                       +───────────────────────────────────────────────+
                                       │               │
                     MCP JSON-RPC 2.0  │               │  MCP JSON-RPC 2.0
                     (Spatial Tools)   │               │  (CMS / Web Tools)
                                       ▼               ▼
        ┌─────────────────────────────────────────┐  ┌─────────────────────────────────┐
        │  PLANE 1: KRYSTAL BUILDER MCP SERVER   │  │  PLANE 2: NABYTOK47 OXYGEN MCP  │
        │  (krystal_web_hub/krystal_mcp_server.py)│  │  (WordPress / Oxygen / WooCommerce)│
        ├─────────────────────────────────────────┤  ├─────────────────────────────────┤
        │ • krystal-get-instructions              │  │ • oxygen-get-instructions       │
        │ • krystal-discover-primitives           │  │ • oxygen-html-to-page           │
        │ • krystal-declarative-to-scene          │  │ • oxygen-insert-stylesheet      │
        │ • krystal-insert-theme-tokens           │  │ • oxygen-create-post            │
        │ • krystal-bind-ledger-data              │  │ • oxygen-create-template        │
        │ • krystal-set-spatial-conditions        │  │ • oxygen-set-element-conditions │
        │ • krystal-preview-scene                 │  │ • oxygen-preview-post           │
        │ • krystal-edit-node                     │  │ • oxygen-edit-post              │
        │ • krystal-bridge-to-oxygen              │  └─────────────────────────────────┘
        └─────────────────────────────────────────┘                   ▲
                                       │                              │
                                       ▼                              │
        ┌─────────────────────────────────────────────────────────────┴─┐
        │                 KRYSTAL-STACK PLATFORM ENGINE                 │
        │  • Godot 4 AST Compiler & .TSCN Exporter                      │
        │  • Three.js WebGL Spatial Studio (HyperOS 4 UI)               │
        │  • Double-Entry Dual-Earn Ledger (Poslední Kmen)              │
        │  • 3D Animated Targeting Arrow & Range Engine                 │
        │  • OpenCV / OCR Invoice Scanner & MRP Accounting              │
        └───────────────────────────────────────────────────────────────┘
```

---

## 4. MCP Tool Specifications: `krystal-builder-mcp`

### Tool 1: `krystal-get-instructions`
- **Description:** Returns pre-flight guidelines, combat range rules, dual-earn accounting constraints, and AST assembly standards.
- **Parameters:** `{}`
- **Response:**
  ```json
  {
    "instructions": "...",
    "game_rules": {
      "max_hp": 6,
      "tribes": ["crystal", "toxic", "druid"],
      "accounting": "Double-entry dual-earn ledger"
    }
  }
  ```

### Tool 2: `krystal-discover-primitives`
- **Description:** Discovers spatial node types, PBR properties, available `.obj` meshes, cards, and abilities.
- **Parameters:** `{}`
- **Response:**
  ```json
  {
    "primitives": { "Spatial": {...}, "MeshInstance": {...}, "OmniLight": {...}, "Particles": {...} },
    "assets": ["crystal_shard.obj", "hex_tile.obj", "acid_slime.obj", ...],
    "card_ids": ["crystal_meteor", "druid_strike", "decay_strike", ...]
  }
  ```

### Tool 3: `krystal-declarative-to-scene`
- **Description:** Single-pass 3D scene compiler from high-level prompts and declarative injection lists.
- **Parameters:**
  ```json
  {
    "prompt": "Taktická aréna s kryštálovým meteorom a aéterovým konduitom",
    "layout_type": "hex_arena",
    "inject_buildings": ["aether_conduit", "capacitor_tower"],
    "inject_spells": ["crystal_meteor"]
  }
  ```
- **Response:**
  ```json
  {
    "status": "SCENE_COMPILED",
    "root_name": "WorldRoot",
    "node_count": 24,
    "tscn_preview": "[gd_scene format=2]..."
  }
  ```

### Tool 4: `krystal-insert-theme-tokens`
- **Description:** Ingests HyperOS 4 visual variables and PBR material shaders.
- **Parameters:**
  ```json
  {
    "tokens": {
      "--hyper-glass": "rgba(18, 22, 34, 0.72)",
      "--hyper-blur": "blur(28px) saturate(190%) contrast(105%)",
      "--accent-cyan": "#66fcf1"
    }
  }
  ```

### Tool 5: `krystal-bind-ledger-data`
- **Description:** Binds 3D HUD billboard texts and health bars to live economic ledger data.
- **Parameters:**
  ```json
  {
    "node_name": "PlayerHPBar",
    "data_field": "player_hp",
    "fallback": "6 HP"
  }
  ```

### Tool 6: `krystal-set-spatial-conditions`
- **Description:** Sets activation locks using DNF rule groups.
- **Parameters:**
  ```json
  {
    "node_name": "UltimateSpire",
    "rule_groups": [
      [{"field": "escalation", "op": "==", "value": "total_war_apex"}]
    ]
  }
  ```

### Tool 7: `krystal-preview-scene`
- **Description:** Returns SceneTree hierarchy, node counts, and Godot 4 `.tscn` text for verification.
- **Parameters:** `{}`

### Tool 8: `krystal-edit-node`
- **Description:** Surgical property editor for existing nodes in the active SceneTree.
- **Parameters:**
  ```json
  {
    "node_name": "Building_aether_conduit",
    "properties": { "scale": [1.5, 1.5, 1.5] }
  }
  ```

### Tool 9: `krystal-bridge-to-oxygen`
- **Description:** Formats Krystal data into exact payloads expected by `nabytok47` tools (`oxygen-create-post`, `oxygen-insert-stylesheet`).
- **Parameters:**
  ```json
  {
    "payload_type": "invoice",
    "context_data": {
      "uuid": "INV-2026-0042",
      "amount": "450.00 EUR",
      "supplier": "Crystalline Hardware s.r.o."
    }
  }
  ```
- **Response:**
  ```json
  {
    "status": "BRIDGE_PAYLOAD_PREPARED",
    "target_mcp_server": "nabytok47",
    "target_tool": "oxygen-create-post",
    "payload": {
      "title": "Faktúra // INV-2026-0042",
      "post_type": "accounting_invoice",
      "status": "publish",
      "meta": { "invoice_amount": "450.00 EUR", "invoice_supplier": "Crystalline Hardware s.r.o." }
    }
  }
  ```

---

## 5. Implementation & Verification Plan

1. **Server Module:** [`krystal_web_hub/krystal_mcp_server.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/krystal_web_hub/krystal_mcp_server.py) implements the complete tool dispatch pipeline and JSON-RPC structure.
2. **Automated Test Suite:** [`tests/test_krystal_mcp_architecture.py`](file:///c:/Users/dusan/Documents/GitHub/Krystal-stack-platform-framework/tests/test_krystal_mcp_architecture.py) verifies:
   - Tool registration and schema completeness.
   - Discovery pre-flight contracts.
   - Declarative single-pass scene generation with injected buildings and spells.
   - Theme token insertion and dynamic ledger data binding.
   - DNF condition evaluation against escalation stages.
   - Surgical node property updating.
   - Bidirectional payload bridging to `nabytok47`.
