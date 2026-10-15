#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: SSOT AUTHORITY PROFILE BITS CODE GENERATOR
==============================================================================
Module: scripts/generate_authority_bits.py
Description:
    Reads specs/authority_profile_bits_v1.yaml and compiles canonical bindings for:
      - Python IntFlag: krystal_kernel/verification/authority_bits.py
      - C ABI Header:   include/kr_authority_bits.h
      - JSON Schema:    schemas/authority_profile_bits.schema.json
==============================================================================
"""

import os
import sys
import json
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(ROOT_DIR, "specs", "authority_profile_bits_v1.yaml")
PY_OUT_PATH = os.path.join(ROOT_DIR, "krystal_kernel", "verification", "authority_bits.py")
C_OUT_PATH = os.path.join(ROOT_DIR, "include", "kr_authority_bits.h")
JSON_OUT_PATH = os.path.join(ROOT_DIR, "schemas", "authority_profile_bits.schema.json")


def parse_yaml_minimal(text: str) -> dict:
    """Minimal YAML parser without third-party dependencies."""
    profiles = []
    lines = text.splitlines()
    cur_prof = None
    for line in lines:
        line_s = line.strip()
        if not line_s or line_s.startswith("#"):
            continue
        if line_s.startswith("- profile_id:"):
            if cur_prof:
                profiles.append(cur_prof)
            cur_prof = {"profile_id": line_s.split(":", 1)[1].strip().strip('"').strip("'")}
        elif cur_prof:
            if line_s.startswith("enum_name:"):
                cur_prof["enum_name"] = line_s.split(":", 1)[1].strip().strip('"').strip("'")
            elif line_s.startswith("bit:"):
                cur_prof["bit"] = int(line_s.split(":", 1)[1].strip())
            elif line_s.startswith("hex:"):
                cur_prof["hex"] = line_s.split(":", 1)[1].strip().strip('"').strip("'")
            elif line_s.startswith("description:"):
                cur_prof["description"] = line_s.split(":", 1)[1].strip().strip('"').strip("'")
    if cur_prof:
        profiles.append(cur_prof)
    return {"schema_version": "1.0", "profiles": profiles}


def generate():
    if not os.path.exists(SPEC_PATH):
        raise FileNotFoundError(f"Spec file not found: {SPEC_PATH}")

    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        spec_text = f.read()

    data = parse_yaml_minimal(spec_text)
    profiles = data["profiles"]

    # 1. Generate Python IntFlag
    py_lines = [
        "#!/usr/bin/env python3",
        '"""',
        "AUTOMATICALLY GENERATED FROM specs/authority_profile_bits_v1.yaml. DO NOT EDIT DIRECTLY.",
        '"""',
        "from __future__ import annotations",
        "from enum import IntFlag, unique",
        "from typing import Dict, Set, Optional, Tuple",
        "",
        "@unique",
        "class AuthorityProfileBit(IntFlag):",
        '    """Canonical 64-bit authority mask for governor profile eligibility."""',
        "    NONE = 0",
    ]
    for p in profiles:
        py_lines.append(f"    {p['enum_name']} = 1 << {p['bit']}  # {p['hex']} - {p['description']}")
    
    all_mask_expr = " | ".join(f"{p['enum_name']}" for p in profiles)
    py_lines.extend([
        f"    ALL_PROFILES = {all_mask_expr}",
        "",
        "PROFILE_ID_TO_BIT: Dict[str, AuthorityProfileBit] = {",
    ])
    for p in profiles:
        py_lines.append(f'    "{p["profile_id"]}": AuthorityProfileBit.{p["enum_name"]},')
    py_lines.extend([
        "}",
        "",
        "BIT_TO_PROFILE_ID: Dict[AuthorityProfileBit, str] = {",
    ])
    for p in profiles:
        py_lines.append(f'    AuthorityProfileBit.{p["enum_name"]}: "{p["profile_id"]}",')
    py_lines.extend([
        "}",
        "",
        "def set_to_mask(profile_names: Set[str]) -> int:",
        '    """Converts a set of profile name strings into a uint64 bitmask."""',
        "    mask = 0",
        "    for name in profile_names:",
        "        bit = PROFILE_ID_TO_BIT.get(name)",
        "        if bit is not None:",
        "            mask |= bit.value",
        "    return mask",
        "",
        "def mask_to_set(mask: int) -> Set[str]:",
        '    """Converts a uint64 bitmask into a set of profile name strings."""',
        "    names = set()",
        "    for bit, name in BIT_TO_PROFILE_ID.items():",
        "        if mask & bit.value:",
        "            names.add(name)",
        "    return names",
        "",
        "def mask_to_highest_profile(mask: int) -> Optional[str]:",
        '    """Extracts the highest priority profile name present in the mask."""',
        "    for p in reversed(list(PROFILE_ID_TO_BIT.keys())):",
        "        bit_val = PROFILE_ID_TO_BIT[p].value",
        "        if mask & bit_val:",
        "            return p",
        "    return None",
    ])

    os.makedirs(os.path.dirname(PY_OUT_PATH), exist_ok=True)
    with open(PY_OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(py_lines) + "\n")
    print(f"[OK] Generated: {PY_OUT_PATH}")

    # 2. Generate C ABI Header
    c_lines = [
        "/* AUTOMATICALLY GENERATED FROM specs/authority_profile_bits_v1.yaml. DO NOT EDIT DIRECTLY. */",
        "#ifndef KR_AUTHORITY_BITS_H",
        "#define KR_AUTHORITY_BITS_H",
        "",
        "#include <stdint.h>",
        "",
        "typedef uint64_t kr_authority_mask;",
        "",
        "enum kr_profile_bit {",
        "    KR_PROFILE_NONE = 0ULL,",
    ]
    for p in profiles:
        c_lines.append(f"    {p['enum_name']} = (1ULL << {p['bit']}), /* {p['hex']} */")
    c_lines.extend([
        "};",
        "",
        "typedef struct {",
        "    kr_authority_mask requested;",
        "    kr_authority_mask thermal_safe;",
        "    kr_authority_mask battery_safe;",
        "    kr_authority_mask capability_supported;",
        "    kr_authority_mask evidence_supported;",
        "    kr_authority_mask policy_allowed;",
        "} kr_authority_input;",
        "",
        "static inline kr_authority_mask kr_reduce_authority(const kr_authority_input *in) {",
        "    return in->requested",
        "         & in->thermal_safe",
        "         & in->battery_safe",
        "         & in->capability_supported",
        "         & in->evidence_supported",
        "         & in->policy_allowed;",
        "}",
        "",
        "#endif /* KR_AUTHORITY_BITS_H */",
    ])

    os.makedirs(os.path.dirname(C_OUT_PATH), exist_ok=True)
    with open(C_OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(c_lines) + "\n")
    print(f"[OK] Generated: {C_OUT_PATH}")

    # 3. Generate JSON Schema
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "AuthorityProfileBitsSchema",
        "description": "Schema for authority profile eligibility bitmasks and profiles.",
        "type": "object",
        "required": ["schema_version", "mask_semantics", "bitmask_width", "profiles"],
        "properties": {
            "schema_version": {"type": "string", "const": "1.0"},
            "mask_semantics": {"type": "string", "const": "PROFILE_ELIGIBILITY"},
            "bitmask_width": {"type": "integer", "const": 64},
            "profiles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["profile_id", "enum_name", "bit", "hex"],
                    "properties": {
                        "profile_id": {"type": "string"},
                        "enum_name": {"type": "string"},
                        "bit": {"type": "integer", "minimum": 0, "maximum": 63},
                        "hex": {"type": "string", "pattern": "^0x[0-9a-fA-F]{16}$"},
                        "description": {"type": "string"}
                    }
                }
            }
        }
    }
    os.makedirs(os.path.dirname(JSON_OUT_PATH), exist_ok=True)
    with open(JSON_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"[OK] Generated: {JSON_OUT_PATH}")


if __name__ == "__main__":
    generate()
