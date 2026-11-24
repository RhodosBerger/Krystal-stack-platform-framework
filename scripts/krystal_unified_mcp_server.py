#!/usr/bin/env python3
r"""Zjednotený MCP (Model Context Protocol, JSON-RPC 2.0 `2024-11-05`) server pre všetku komunikáciu
medzi:
  - Antigravity IDE (`krystal-unified-mcp`),
  - LM Studio (`C:\Users\dusan\.lmstudio` / `gemma-4-E2B-it-Q4_K_M.gguf`),
  - Bionic (`C:\Users\dusan\AppData\Local\Programs\Bionic\Bionic.exe`),
  - Krystal `#![no_std]` Rust firmvérom, Kisak/Fedora-Gallium ovládačmi, OpenVINO mostom a čipsetovým swapom.

Podporuje:
  1. Štandardný `stdio` JSON-RPC 2.0 prenos (pre priame pripojenie do LM Studio `mcp.json`, Bionic a IDE),
  2. Programové volanie `handle_mcp_jsonrpc_request(req_dict)` pre HTTP koncový bod `POST /mcp/v1` na porte `8090`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation


MCP_TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "name": "krystal_relationship_and_fedora_dlss_sheets",
        "description": "Vráti 5 mapovacích listov vzťahov komponentov, 6 Fedora NVK/NAK/Nouveau patternov, 6 Krystal-DLSS knižníc a 6 platformových pluginov.",
        "inputSchema": {
            "type": "object",
            "properties": {"uma_utilization": {"type": "number", "default": 0.34}},
        },
    },
    {
        "name": "krystal_2b_explain_and_train_codebase",
        "description": "Analyzuje AST/logické konštrukcie kódu cez 2B neurosymbolický model (Gemma 4 E2B + AST), vytvorí nové definície a voliteľne dotrénuje model na používateľských dátach.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target_file": {"type": "string", "default": "native/krystal_sovereign_firmware_rs/src/lib.rs"},
                "user_training_samples": {"type": "array", "items": {"type": "object"}},
            },
        },
    },
    {
        "name": "krystal_chipset_cooling_freq_and_unified_swap",
        "description": "Profiluje bezpečné frekvencie čipsetu Tiger Lake, chladenie, prepínanie vlákien, 14 GB UMA VRAM a unifikovaný pamäťový swap.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cooling_mode": {"type": "string", "default": "ADRENALIN_BALANCED_VAPOR"},
                "requested_cpu_mhz": {"type": "integer", "default": 3800},
                "requested_vram_mb": {"type": "integer", "default": 14336},
            },
        },
    },
    {
        "name": "krystal_replay_pending_processes",
        "description": "Zobrazí a bezpečne zopakuje (vráti do behu) čakajúce pending procesy z trvalého WAL žurnálu.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "krystal_openvino_gallium_bridge_status",
        "description": "Spustí bezkópiový most OpenVINO <-> Kisak-Mesa Gallium (CPU + Iris Xe GPU) a prediktor NIR shaderov.",
        "inputSchema": {
            "type": "object",
            "properties": {"uma_utilization": {"type": "number", "default": 0.34}},
        },
    },
]


def handle_mcp_jsonrpc_request(req: Dict[str, Any]) -> Dict[str, Any]:
    req_id = req.get("id", 1)
    method = str(req.get("method", ""))
    params = req.get("params") or {}

    if VITAL_MAX_HP != 6:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32006, "message": "Constitutional VITAL_MAX_HP != 6"},
        }

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False}},
                "serverInfo": {
                    "name": "krystal-sovereign-adrenalin-mcp",
                    "version": "2.0.0",
                    "vital_max_hp": VITAL_MAX_HP,
                },
            },
        }

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": MCP_TOOLS_SCHEMA}}

    if method == "resources/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "resources": [
                    {
                        "uri": "krystal://design/adrenalin-manual",
                        "name": "Krystal Adrenalin Design Manual",
                        "mimeType": "text/markdown",
                    },
                    {
                        "uri": "krystal://vram/lmstudio-bionic-config",
                        "name": "LM Studio & Bionic 14GB UMA VRAM Acceleration Config",
                        "mimeType": "application/json",
                    },
                ]
            },
        }

    if method == "resources/read":
        uri = str(params.get("uri", ""))
        if uri == "krystal://design/adrenalin-manual":
            txt = (REPO_ROOT / "docs" / "design" / "KRYSTAL_ADRENALIN_DESIGN_MANUAL.md").read_text(encoding="utf-8")
            return {"jsonrpc": "2.0", "id": req_id, "result": {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": txt}]}}
        cfg_p = REPO_ROOT / "configs" / "lmstudio_bionic_vram_mcp_bundle.json"
        txt = cfg_p.read_text(encoding="utf-8") if cfg_p.exists() else "{}"
        return {"jsonrpc": "2.0", "id": req_id, "result": {"contents": [{"uri": uri, "mimeType": "application/json", "text": txt}]}}

    if method == "tools/call":
        name = str(params.get("name", ""))
        args = params.get("arguments") or {}
        try:
            from projects.trigonometric_hybrid_platform_framework.fedora_dlss_vram_bionic_mcp_and_2b_ast import (
                dispatch_krystal_mcp_tool,
            )

            payload = dispatch_krystal_mcp_tool(name, args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                    "structuredContent": payload,
                    "isError": False,
                },
            }
        except ContractViolation as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"error": str(exc), "code": exc.violation_code, "delta_s": 0})}],
                    "isError": True,
                },
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Unknown MCP method: {method}"},
    }


def main_stdio() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        if "id" not in req:
            # JSON-RPC notification (napr. notifications/initialized)
            continue
        resp = handle_mcp_jsonrpc_request(req)
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main_stdio()
