"""Minimal, stdlib-only JSON Schema validator (documented subset of draft 2020-12).

Why this exists
---------------
``krystal_kernel`` must stay stdlib-only and the project venv has no ``jsonschema``
package. Instead of silently ignoring keywords we do not implement, this module
validates a *fixed* subset and exposes :func:`unsupported_keywords`, which tests use to
assert that every schema in the repository only uses keywords this validator enforces.

Supported keywords
------------------
Validation: ``type`` (string or list), ``enum``, ``const``, ``required``, ``properties``,
``additionalProperties`` (bool or schema), ``items`` (single schema), ``minItems``,
``maxItems``, ``uniqueItems``, ``minimum``, ``maximum``, ``exclusiveMinimum``,
``exclusiveMaximum``, ``minLength``, ``maxLength``, ``pattern``, ``allOf``, ``anyOf``,
``oneOf``, ``not``, ``$ref``.

``$ref`` forms: ``#/$defs/name`` (local JSON pointer), ``other.schema.json`` and
``other.schema.json#/$defs/name`` (sibling file resolved against ``base_dir``).

Annotations (accepted, not enforced): ``$schema``, ``$id``, ``$defs``, ``$comment``,
``title``, ``description``, ``default``, ``examples``, ``format``, ``readOnly``.

Semantics notes
---------------
* ``integer`` and ``number`` never match ``bool`` (Python ``True`` is an ``int``).
* ``integer`` accepts floats with an integral value only if they are finite (``2.0``),
  matching the JSON Schema data model.
* ``pattern`` uses :func:`re.search` (unanchored), as the specification requires.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

VALIDATION_KEYWORDS = frozenset({
    "type", "enum", "const", "required", "properties", "additionalProperties",
    "items", "minItems", "maxItems", "uniqueItems", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "pattern",
    "allOf", "anyOf", "oneOf", "not", "$ref",
})
ANNOTATION_KEYWORDS = frozenset({
    "$schema", "$id", "$defs", "$comment", "title", "description", "default",
    "examples", "format", "readOnly",
})
SUPPORTED_KEYWORDS = VALIDATION_KEYWORDS | ANNOTATION_KEYWORDS

# Keywords whose value is a schema / list of schemas / map of schemas.
_SCHEMA_MAP_KEYWORDS = ("properties", "$defs")
_SCHEMA_LIST_KEYWORDS = ("allOf", "anyOf", "oneOf")
_SCHEMA_VALUE_KEYWORDS = ("items", "not", "additionalProperties")

_MAX_REF_DEPTH = 64


class SchemaError(ValueError):
    """The schema itself is malformed or uses an unresolvable ``$ref``."""


def _type_matches(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    if isinstance(value, bool):
        return False
    if type_name == "integer":
        if isinstance(value, int):
            return True
        return isinstance(value, float) and math.isfinite(value) and value.is_integer()
    if type_name == "number":
        return isinstance(value, (int, float))
    raise SchemaError(f"unknown type name {type_name!r}")


def _json_equal(a: Any, b: Any) -> bool:
    """Equality that does not treat ``True == 1`` (JSON Schema data model)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_json_equal(x, y) for x, y in zip(a, b))
    return a == b


def _pointer_get(doc: Any, pointer: str) -> Any:
    if pointer in ("", "#"):
        return doc
    if not pointer.startswith("#/"):
        raise SchemaError(f"only '#/...' JSON pointers are supported, got {pointer!r}")
    node = doc
    for raw in pointer[2:].split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise SchemaError(f"unresolvable JSON pointer {pointer!r}")
    return node


class _Resolver:
    def __init__(self, root: Dict[str, Any], base_dir: Optional[Path]):
        self.base_dir = base_dir
        self._docs: Dict[str, Any] = {"": root}

    def _load(self, name: str) -> Any:
        if name not in self._docs:
            if self.base_dir is None:
                raise SchemaError(f"external $ref {name!r} needs base_dir")
            if "/" in name or "\\" in name or name.startswith("."):
                raise SchemaError(f"external $ref must be a sibling file name, got {name!r}")
            path = self.base_dir / name
            try:
                self._docs[name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise SchemaError(f"cannot load $ref {name!r}: {exc}") from exc
        return self._docs[name]

    def resolve(self, ref: str, current_doc: str) -> "tuple[Any, str]":
        doc_name, _, frag = ref.partition("#")
        doc_name = doc_name or current_doc
        doc = self._docs[""] if doc_name == "" else self._load(doc_name)
        return _pointer_get(doc, "#" + frag if frag else "#"), doc_name


def validate(instance: Any, schema: Dict[str, Any], *, base_dir: Optional[Path] = None) -> List[str]:
    """Validate ``instance`` and return a list of human-readable errors (empty = valid).

    Each error is prefixed with a JSON-pointer-like path to the offending value, e.g.
    ``/nodes/2/params/n: 0 is less than minimum 1``.
    """
    resolver = _Resolver(schema, Path(base_dir) if base_dir is not None else None)
    errors: List[str] = []
    _validate(instance, schema, "", errors, resolver, "", 0)
    return errors


def _validate(inst: Any, sch: Any, path: str, errors: List[str], resolver: _Resolver,
              doc: str, depth: int) -> None:
    if sch is True:
        return
    if sch is False:
        errors.append(f"{path or '/'}: no value is allowed here")
        return
    if not isinstance(sch, dict):
        raise SchemaError(f"schema at {path or '/'} must be an object or boolean")
    if depth > _MAX_REF_DEPTH:
        raise SchemaError("$ref nesting too deep (cycle?)")

    where = path or "/"
    add = errors.append

    if "$ref" in sch:
        target, target_doc = resolver.resolve(sch["$ref"], doc)
        _validate(inst, target, path, errors, resolver, target_doc, depth + 1)

    if "type" in sch:
        types = sch["type"] if isinstance(sch["type"], list) else [sch["type"]]
        if not any(_type_matches(inst, t) for t in types):
            add(f"{where}: expected type {'/'.join(types)}, got {type(inst).__name__}")
            return  # further keyword checks would only produce noise

    if "const" in sch and not _json_equal(inst, sch["const"]):
        add(f"{where}: must equal {sch['const']!r}")
    if "enum" in sch and not any(_json_equal(inst, e) for e in sch["enum"]):
        add(f"{where}: {inst!r} is not one of {sch['enum']!r}")

    is_num = isinstance(inst, (int, float)) and not isinstance(inst, bool)
    if is_num:
        if "minimum" in sch and inst < sch["minimum"]:
            add(f"{where}: {inst!r} is less than minimum {sch['minimum']!r}")
        if "maximum" in sch and inst > sch["maximum"]:
            add(f"{where}: {inst!r} is greater than maximum {sch['maximum']!r}")
        if "exclusiveMinimum" in sch and inst <= sch["exclusiveMinimum"]:
            add(f"{where}: {inst!r} must be > {sch['exclusiveMinimum']!r}")
        if "exclusiveMaximum" in sch and inst >= sch["exclusiveMaximum"]:
            add(f"{where}: {inst!r} must be < {sch['exclusiveMaximum']!r}")

    if isinstance(inst, str):
        if "minLength" in sch and len(inst) < sch["minLength"]:
            add(f"{where}: string shorter than {sch['minLength']}")
        if "maxLength" in sch and len(inst) > sch["maxLength"]:
            add(f"{where}: string longer than {sch['maxLength']}")
        if "pattern" in sch and re.search(sch["pattern"], inst) is None:
            add(f"{where}: {inst!r} does not match pattern {sch['pattern']!r}")

    if isinstance(inst, list):
        if "minItems" in sch and len(inst) < sch["minItems"]:
            add(f"{where}: needs at least {sch['minItems']} items, has {len(inst)}")
        if "maxItems" in sch and len(inst) > sch["maxItems"]:
            add(f"{where}: allows at most {sch['maxItems']} items, has {len(inst)}")
        if sch.get("uniqueItems"):
            for i in range(len(inst)):
                if any(_json_equal(inst[i], inst[j]) for j in range(i)):
                    add(f"{where}: items are not unique (index {i})")
                    break
        if "items" in sch:
            for i, item in enumerate(inst):
                _validate(item, sch["items"], f"{path}/{i}", errors, resolver, doc, depth)

    if isinstance(inst, dict):
        for key in sch.get("required", ()):
            if key not in inst:
                add(f"{where}: missing required property {key!r}")
        props = sch.get("properties", {})
        for key, value in inst.items():
            child = f"{path}/{key}"
            if key in props:
                _validate(value, props[key], child, errors, resolver, doc, depth)
            elif "additionalProperties" in sch:
                extra = sch["additionalProperties"]
                if extra is False:
                    add(f"{where}: unexpected property {key!r}")
                else:
                    _validate(value, extra, child, errors, resolver, doc, depth)

    def sub_errors(s: Any) -> List[str]:
        out: List[str] = []
        _validate(inst, s, path, out, resolver, doc, depth + 1)
        return out

    for s in sch.get("allOf", ()):
        errors.extend(sub_errors(s))
    if "anyOf" in sch:
        results = [sub_errors(s) for s in sch["anyOf"]]
        if all(results):
            add(f"{where}: does not match any allowed alternative "
                f"(first mismatch: {results[0][0]})")
    if "oneOf" in sch:
        ok = sum(1 for s in sch["oneOf"] if not sub_errors(s))
        if ok != 1:
            add(f"{where}: must match exactly one alternative, matched {ok}")
    if "not" in sch and not sub_errors(sch["not"]):
        add(f"{where}: must not match the 'not' schema")


def unsupported_keywords(schema: Any, path: str = "") -> List[str]:
    """Return ``path: keyword`` for every keyword this validator would silently ignore."""
    found: List[str] = []
    if not isinstance(schema, dict):
        return found
    for key, value in schema.items():
        here = f"{path}/{key}"
        if key not in SUPPORTED_KEYWORDS:
            found.append(f"{path or '/'}: {key}")
            continue
        if key in _SCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            for name, sub in value.items():
                found.extend(unsupported_keywords(sub, f"{here}/{name}"))
        elif key in _SCHEMA_LIST_KEYWORDS and isinstance(value, list):
            for i, sub in enumerate(value):
                found.extend(unsupported_keywords(sub, f"{here}/{i}"))
        elif key in _SCHEMA_VALUE_KEYWORDS and isinstance(value, dict):
            found.extend(unsupported_keywords(value, here))
    return found


def load_json(path: "str | Path") -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


Validator = Callable[[Any], List[str]]
