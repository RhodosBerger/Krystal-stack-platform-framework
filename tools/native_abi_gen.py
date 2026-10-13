"""
Generate C17 header, Rust repr(C) struct, Python ctypes module and a golden
binary fixture from native/spec/binary_layouts.yaml (single source of truth).

    python -m tools.native_abi_gen
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "native" / "spec" / "binary_layouts.yaml"
OUT = ROOT / "native" / "generated"

# type -> (size, align, C, Rust, ctypes, struct-fmt)
SCALARS = {
    "u8": (1, 1, "uint8_t", "u8", "c_uint8", "B"),
    "u16": (2, 2, "uint16_t", "u16", "c_uint16", "H"),
    "u32": (4, 4, "uint32_t", "u32", "c_uint32", "I"),
    "u64": (8, 8, "uint64_t", "u64", "c_uint64", "Q"),
    "f64": (8, 8, "double", "f64", "c_double", "d"),
}


def load_spec() -> Dict[str, Any]:
    text = SPEC.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)  # spec files are JSON-subset YAML


def parse_type(t: str) -> Tuple[str, int]:
    if t.endswith("]") or "[" in t:
        base, n = t[:-1].split("[")
        return base, int(n)
    return t, 0


def field_geometry(f: Dict[str, Any]) -> Tuple[int, int]:
    base, n = parse_type(f["type"])
    size, align = SCALARS[base][0], SCALARS[base][1]
    return size * (n or 1), align


def validate_struct(s: Dict[str, Any]) -> None:
    """Reject implicit padding, misaligned or overlapping fields."""
    cursor = 0
    for f in s["fields"]:
        size, align = field_geometry(f)
        if f["offset"] != cursor:
            raise ValueError(f"{s['name']}.{f['name']}: offset {f['offset']} != packed cursor {cursor} (implicit padding forbidden)")
        if f["offset"] % align:
            raise ValueError(f"{s['name']}.{f['name']}: misaligned")
        cursor += size
    if cursor != s["size"]:
        raise ValueError(f"{s['name']}: fields sum to {cursor}, declared {s['size']}")


def gen_c(s: Dict[str, Any], enums: Dict[str, Dict[str, int]]) -> str:
    n = s["name"]
    guard = n.upper() + "_H"
    L: List[str] = [
        "/* GENERATED from native/spec/binary_layouts.yaml - DO NOT EDIT */",
        f"#ifndef {guard}", f"#define {guard}",
        "#include <stdint.h>", "#include <stddef.h>", "#include <assert.h>", "",
    ]
    for ename, members in enums.items():
        L.append(f"typedef enum {{")
        for k, v in members.items():
            L.append(f"    KR_{ename.upper()}_{k} = {v},")
        L.append(f"}} kr_{ename}_t;\n")
    L.append(f"typedef struct _Alignas({s['alignment']}) {n} {{")
    for f in s["fields"]:
        base, cnt = parse_type(f["type"])
        c = SCALARS[base][2]
        L.append(f"    {c} {f['name']}" + (f"[{cnt}]" if cnt else "") + ";")
    L.append(f"}} {n}_t;\n")
    L.append(f"static_assert(sizeof({n}_t) == {s['size']}, \"{n} size\");")
    L.append(f"static_assert(_Alignof({n}_t) == {s['alignment']}, \"{n} align\");")
    for f in s["fields"]:
        L.append(f"static_assert(offsetof({n}_t, {f['name']}) == {f['offset']}, \"{n}.{f['name']} offset\");")
    L.append(f"\n#endif /* {guard} */\n")
    return "\n".join(L)


def gen_rust(s: Dict[str, Any]) -> str:
    n = s["name"]
    tname = "".join(p.capitalize() for p in n.split("_"))
    L = ["// GENERATED from native/spec/binary_layouts.yaml - DO NOT EDIT",
         "#[repr(C, align(%d))]" % s["alignment"], "#[derive(Clone, Copy, Debug, PartialEq)]",
         f"pub struct {tname} {{"]
    for f in s["fields"]:
        base, cnt = parse_type(f["type"])
        r = SCALARS[base][3]
        L.append(f"    pub {f['name']}: " + (f"[{r}; {cnt}]" if cnt else r) + ",")
    L.append("}\n")
    L.append(f"const _: () = assert!(core::mem::size_of::<{tname}>() == {s['size']});")
    L.append(f"const _: () = assert!(core::mem::align_of::<{tname}>() == {s['alignment']});")
    for f in s["fields"]:
        L.append(f"const _: () = assert!(core::mem::offset_of!({tname}, {f['name']}) == {f['offset']});")
    return "\n".join(L) + "\n"


def gen_ctypes(s: Dict[str, Any]) -> str:
    n = s["name"]
    L = ['"""GENERATED from native/spec/binary_layouts.yaml - DO NOT EDIT."""',
         "import ctypes", "", f"class {n}(ctypes.LittleEndianStructure):",
         "    _pack_ = 1", "    _fields_ = ["]
    for f in s["fields"]:
        base, cnt = parse_type(f["type"])
        ct = "ctypes." + SCALARS[base][4]
        L.append(f"        ({f['name']!r}, {ct}" + (f" * {cnt}" if cnt else "") + "),")
    L.append("    ]\n")
    L.append(f"SIZE = {s['size']}")
    L.append(f"OFFSETS = {{{', '.join(repr(f['name']) + ': ' + str(f['offset']) for f in s['fields'])}}}")
    return "\n".join(L) + "\n"


def encode(s: Dict[str, Any], values: Dict[str, Any]) -> bytes:
    """Reference encoder (little endian). Computes crc32 over declared range."""
    fmt = "<"
    vals: List[Any] = []
    for f in s["fields"]:
        base, cnt = parse_type(f["type"])
        code = SCALARS[base][5]
        v = values.get(f["name"], [0] * cnt if cnt else 0)
        if cnt:
            fmt += f"{cnt}{code}"
            vals.extend(v)
        else:
            fmt += code
            vals.append(v)
    buf = bytearray(struct.pack(fmt, *vals))
    crc = s.get("crc")
    if crc:
        lo, hi = crc["covers"]
        off = next(f["offset"] for f in s["fields"] if f["name"] == crc["field"])
        buf[off:off + 4] = struct.pack("<I", zlib.crc32(bytes(buf[lo:hi])) & 0xFFFFFFFF)
    assert len(buf) == s["size"]
    return bytes(buf)


def decode(s: Dict[str, Any], data: bytes) -> Dict[str, Any]:
    if len(data) != s["size"]:
        raise ValueError("bad size")
    out: Dict[str, Any] = {}
    for f in s["fields"]:
        base, cnt = parse_type(f["type"])
        code = SCALARS[base][5]
        o = f["offset"]
        if cnt:
            out[f["name"]] = list(struct.unpack_from(f"<{cnt}{code}", data, o))
        else:
            out[f["name"]] = struct.unpack_from("<" + code, data, o)[0]
    crc = s.get("crc")
    if crc:
        lo, hi = crc["covers"]
        if zlib.crc32(data[lo:hi]) & 0xFFFFFFFF != out[crc["field"]]:
            raise ValueError("bad crc")
    return out


def main() -> None:
    spec = load_spec()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in spec["structs"]:
        validate_struct(s)
        n = s["name"]
        (OUT / f"{n}.h").write_text(gen_c(s, spec["enums"]), encoding="utf-8", newline="\n")
        (OUT / f"{n}.rs").write_text(gen_rust(s), encoding="utf-8", newline="\n")
        (OUT / f"{n}_ctypes.py").write_text(gen_ctypes(s), encoding="utf-8", newline="\n")
        (OUT / f"{n}.golden.bin").write_bytes(encode(s, s["golden"]))
        print(f"generated {n}: size={s['size']}")


if __name__ == "__main__":
    main()
