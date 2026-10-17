#!/usr/bin/env python3
"""
==============================================================================
KARAF: ALGORITHM TEMPLATE REGISTRY & PARSER
==============================================================================
Module: krystal_kernel/karaf/template_registry.py
Description:
    Discovers, parses, validates, and manages canonical algorithm templates
    stored in `krystal_kernel/templates/*.yaml`.

    Includes a robust zero-dependency YAML subset parser for environments
    where PyYAML is not installed.
==============================================================================
"""

from __future__ import annotations
import os
import re
import json
from typing import Dict, Any, List, Optional, Tuple

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.karaf.algorithm_template import AlgorithmTemplate


def _parse_yaml_value(val_str: str) -> Any:
    """Parses a YAML leaf value into int, float, bool, inline list, or string."""
    v = val_str.strip()
    if not v:
        return ""
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.lower() in ("null", "none", "~"):
        return None
    # Inline list: e.g. [-2, -1, 0, 1, 2] or [12.0, 35.0]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        items = [x.strip() for x in inner.split(",")]
        return [_parse_yaml_value(item) for item in items]
    # Inline dict: e.g. {a: 1, b: 2}
    if v.startswith("{") and v.endswith("}"):
        inner = v[1:-1].strip()
        if not inner:
            return {}
        res = {}
        for pair in inner.split(","):
            if ":" in pair:
                pk, pv = pair.split(":", 1)
                res[pk.strip()] = _parse_yaml_value(pv)
        return res
    # Quoted string
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    # Numbers
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v


def parse_yaml_subset(text: str) -> Dict[str, Any]:
    """
    Zero-dependency, clean YAML-subset parser supporting indented dicts,
    block lists ('- '), inline lists, and typed scalars.
    """
    try:
        import yaml
        return yaml.safe_load(text)
    except Exception:
        pass

    lines = text.splitlines()
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Any, Optional[str]]] = [(-1, root, None)]

    for raw_line in lines:
        # Strip comments
        if "#" in raw_line:
            # simple check avoiding quotes
            comment_pos = raw_line.find("#")
            line = raw_line[:comment_pos]
        else:
            line = raw_line

        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        # Pop stack until parent has smaller indent
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()

        parent_indent, parent_obj, parent_key = stack[-1]

        # Case 1: List item '- '
        if stripped.startswith("- "):
            item_text = stripped[2:].strip()
            if isinstance(parent_obj, dict) and parent_key is not None:
                # If current parent_key is not yet a list, make it a list
                if not isinstance(parent_obj[parent_key], list):
                    parent_obj[parent_key] = []
                cur_list = parent_obj[parent_key]
            elif isinstance(parent_obj, list):
                cur_list = parent_obj
            elif (isinstance(parent_obj, dict) and not parent_obj and len(stack) > 1
                  and stack[-2][2] is not None and isinstance(stack[-2][1], dict)
                  and stack[-2][1].get(stack[-2][2]) is parent_obj):
                # Empty placeholder dict created by 'key:' -> this key holds a block list
                cur_list = []
                stack[-2][1][stack[-2][2]] = cur_list
                stack[-1] = (stack[-1][0], cur_list, None)
            else:
                cur_list = []

            # Check if list item is key-value dict: '- key: val'
            if ":" in item_text and not (item_text.startswith("{") or item_text.startswith("[")):
                sub_k, sub_v = item_text.split(":", 1)
                sub_k = sub_k.strip()
                sub_v = sub_v.strip()
                item_dict = {sub_k: _parse_yaml_value(sub_v)} if sub_v else {sub_k: {}}
                cur_list.append(item_dict)
                if not sub_v:
                    stack.append((indent, item_dict, sub_k))
            else:
                cur_list.append(_parse_yaml_value(item_text))

        # Case 2: Key-Value pair 'key: value' or 'key:'
        elif ":" in stripped:
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()

            if val:
                parsed_val = _parse_yaml_value(val)
                if isinstance(parent_obj, dict):
                    parent_obj[key] = parsed_val
                stack.append((indent, parent_obj, key))
            else:
                # Empty key begins a nested object or list
                new_dict = {}
                if isinstance(parent_obj, dict):
                    parent_obj[key] = new_dict
                stack.append((indent, parent_obj, key))
                stack.append((indent, new_dict, None))

    return root


class TemplateRegistry:
    """Discovers and caches algorithm templates."""

    def __init__(self, templates_dir: Optional[str] = None):
        assert VITAL_MAX_HP == 6, "VITAL_MAX_HP must equal 6"
        self.vital_max_hp = VITAL_MAX_HP
        if templates_dir is None:
            # Resolve relative to repo layout
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(os.path.dirname(curr_dir), "templates")
            if os.path.isdir(candidate):
                templates_dir = candidate
            else:
                templates_dir = os.path.join(curr_dir, "templates")

        self.templates_dir = templates_dir
        self._templates: Dict[str, AlgorithmTemplate] = {}
        self.reload_templates()

    def reload_templates(self):
        """Scans templates directory and loads all .yaml / .yml files."""
        self._templates.clear()
        if not os.path.isdir(self.templates_dir):
            return

        for fname in os.listdir(self.templates_dir):
            if fname.endswith(".yaml") or fname.endswith(".yml"):
                fpath = os.path.join(self.templates_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
                    data = parse_yaml_subset(text)
                    if data:
                        tpl = AlgorithmTemplate.from_dict(data)
                        self._templates[tpl.identity.id] = tpl
                        # Also register file basename without extension
                        base = os.path.splitext(fname)[0]
                        if base != tpl.identity.id:
                            self._templates[base] = tpl
                except Exception as ex:
                    pass

    def get_template(self, template_id: str) -> Optional[AlgorithmTemplate]:
        return self._templates.get(template_id)

    def list_templates(self) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for tpl in self._templates.values():
            if tpl.identity.id in seen:
                continue
            seen.add(tpl.identity.id)
            out.append(tpl.to_dict())
        return out

    def register_template(self, tpl: AlgorithmTemplate):
        self._templates[tpl.identity.id] = tpl


# Global singleton registry
GLOBAL_TEMPLATE_REGISTRY = TemplateRegistry()
