"""A tiny, replicable drawing bytecode (stdlib only).

A *program* is a flat list of `(opcode, args)`; args are bytes. Programs serialise to a checksummed
byte string (`KB1` container), execute on a bounded VM that paints a `Canvas`, and can be
*replicated* with point mutations or crossed over, which is what makes them a searchable genome.

Safety: the VM has no jumps, no loops and no I/O. Every op has a fixed arity and per-argument valid
range, so any byte string either parses to a valid program or is rejected; execution cost is bounded by
`MAX_OPS` and the canvas size caps.
"""
from __future__ import annotations

import copy
import random
import struct
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import composer as C

MAGIC, VERSION, MAX_OPS = b"KB1", 1, 512
NROLE = len(C.ROLES)
B = (0, 255)
ROLE = (0, NROLE - 1)

# opcode -> (name, morse letter, [(lo, hi) per arg])
OPS: Dict[int, Tuple[str, str, List[Tuple[int, int]]]] = {
    0x01: ("CANVAS", "C", [(8, 255), (8, 255)]),
    0x02: ("PALETTE", "P", [(0, len(C.PALETTE_IDS) - 1)]),
    0x03: ("SEED", "S", [B, B, B, B]),
    0x04: ("HORIZON", "H", [B]),
    0x10: ("SKY", "K", [(0, 1)]),
    0x11: ("DISC", "D", [B, B, B, ROLE]),
    0x12: ("RIDGE", "R", [B, (10, 90), ROLE, (2, 80)]),
    0x13: ("TREE", "T", [B, B, B, ROLE]),
    0x14: ("CLOUD", "U", [B, B, B, ROLE]),
    0x15: ("LAKE", "L", [B, ROLE]),
    0x20: ("TABLE", "B", [B]),
    0x21: ("VASE", "V", [B, B, B, ROLE]),
    0x22: ("FRUIT", "F", [B, B, B, ROLE]),
    0x23: ("BOTTLE", "O", [B, B, B, ROLE]),
    0x24: ("LIGHT", "G", [B, B]),
    0x30: ("DITHER", "I", [(0, 1)]),
    0x31: ("OUTLINE", "N", []),
    0x32: ("POSTERIZE", "Z", [(2, 8)]),
    0x40: ("GRID", "Q", [(4, 64)]),
    0x41: ("BARS", "A", [B, B, B, B]),
    0xFF: ("END", "E", []),
}
BY_NAME = {v[0]: k for k, v in OPS.items()}
BY_LETTER = {v[1]: k for k, v in OPS.items()}
HEADER_OPS = {0x01, 0x02, 0x03, 0xFF}


class BytecodeError(ValueError):
    pass


class Program:
    def __init__(self, ops: Optional[List[Tuple[int, Tuple[int, ...]]]] = None):
        self.ops: List[Tuple[int, Tuple[int, ...]]] = ops or []

    def emit(self, name: str, *args: int) -> "Program":
        self.ops.append((BY_NAME[name], tuple(int(a) for a in args)))
        self.validate()
        return self

    # -- validation / serialisation -------------------------------------------------
    def validate(self) -> None:
        if len(self.ops) > MAX_OPS:
            raise BytecodeError(f"program has {len(self.ops)} ops; limit is {MAX_OPS}")
        for i, (op, args) in enumerate(self.ops):
            if op not in OPS:
                raise BytecodeError(f"op {i}: unknown opcode 0x{op:02x}")
            name, _, spec = OPS[op]
            if len(args) != len(spec):
                raise BytecodeError(f"op {i} {name}: expected {len(spec)} args, got {len(args)}")
            for a, (lo, hi) in zip(args, spec):
                if not lo <= a <= hi:
                    raise BytecodeError(f"op {i} {name}: arg {a} outside {lo}..{hi}")

    def to_bytes(self) -> bytes:
        self.validate()
        body = MAGIC + bytes([VERSION]) + struct.pack(">H", len(self.ops))
        for op, args in self.ops:
            body += bytes([op]) + bytes(args)
        return body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    @classmethod
    def from_bytes(cls, data: bytes) -> "Program":
        if len(data) < 10 or data[:3] != MAGIC:
            raise BytecodeError("bad magic")
        if data[3] != VERSION:
            raise BytecodeError(f"unsupported version {data[3]}")
        if struct.unpack(">I", data[-4:])[0] != zlib.crc32(data[:-4]) & 0xFFFFFFFF:
            raise BytecodeError("checksum mismatch")
        (n,) = struct.unpack(">H", data[4:6])
        if n > MAX_OPS:
            raise BytecodeError("too many ops")
        pos, ops = 6, []
        for _ in range(n):
            if pos >= len(data) - 4:
                raise BytecodeError("truncated")
            op = data[pos]
            if op not in OPS:
                raise BytecodeError(f"unknown opcode 0x{op:02x}")
            k = len(OPS[op][2])
            if pos + 1 + k > len(data) - 4:
                raise BytecodeError("truncated args")
            ops.append((op, tuple(data[pos + 1:pos + 1 + k])))
            pos += 1 + k
        if pos != len(data) - 4:
            raise BytecodeError("trailing bytes")
        p = cls(ops)
        p.validate()
        return p

    def digest(self) -> str:
        return format(zlib.crc32(self.to_bytes()) & 0xFFFFFFFF, "08x")

    def disasm(self) -> str:
        return "\n".join(f"{i:03d} {OPS[op][0]:<9} {' '.join(map(str, args))}".rstrip() for i, (op, args) in enumerate(self.ops))

    def __len__(self) -> int:
        return len(self.ops)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Program) and self.ops == other.ops


# ---------------------------------------------------------------------- VM
def run(prog: Program) -> C.Canvas:
    prog.validate()
    cv: Optional[C.Canvas] = None
    w = h = 64
    pal = C.PALETTE_IDS[0]
    seed, dither, light = 0, False, (-1, -1)
    for idx, (op, a) in enumerate(prog.ops):
        name = OPS[op][0]
        if name == "CANVAS":
            w, h = a
        elif name == "PALETTE":
            pal = C.PALETTE_IDS[a[0]]
        elif name == "SEED":
            seed = int.from_bytes(bytes(a), "big")
        elif name == "END":
            break
        else:
            if cv is None:
                cv = C.Canvas(w, h, pal)
            rng = random.Random((seed * 1000003 + idx) & 0xFFFFFFFF)
            if name == "HORIZON":
                cv.horizon = min(cv.h - 1, a[0])
            elif name == "SKY":
                C.paint_sky(cv, a[0], dither)
            elif name == "DISC":
                C.paint_disc(cv, a[0], a[1], a[2], a[3])
            elif name == "RIDGE":
                C.paint_ridge(cv, rng, a[0], a[1] / 100.0, a[2], a[3])
            elif name == "TREE":
                C.paint_tree(cv, a[0], a[1], a[2], a[3])
            elif name == "CLOUD":
                C.paint_cloud(cv, rng, a[0], a[1], a[2], a[3])
            elif name == "LAKE":
                C.paint_lake(cv, a[0], a[1])
            elif name == "TABLE":
                C.paint_table(cv, a[0], C.R["wall"], C.R["table"])
            elif name == "VASE":
                C.paint_vase(cv, a[0], a[1], a[2], a[3], light[0])
            elif name == "FRUIT":
                C.paint_fruit(cv, a[0], a[1], a[2], a[3], light[0], light[1])
            elif name == "BOTTLE":
                C.paint_bottle(cv, a[0], a[1], a[2], a[3])
            elif name == "LIGHT":
                light = (a[0] - 128, a[1] - 128)
            elif name == "DITHER":
                dither = bool(a[0])
            elif name == "OUTLINE":
                cv.outline(C.R["outline"])
            elif name == "POSTERIZE":
                cv.posterize(a[0])
            elif name == "GRID":
                cv.grid_overlay(a[0], C.R["hud"])
            elif name == "BARS":
                cv.bars_overlay([v / 255.0 for v in a], C.R["hud"])
    return cv if cv is not None else C.Canvas(w, h, pal)


# ---------------------------------------------------------------------- config -> program
TECHNIQUES = {"dither", "outline", "posterize", "grid", "hud"}
SCENES = {"landscape", "still_life"}
COMPUTE_TRIGGERS = {"now", "idle", "scheduled", "on_pattern"}

DEFAULT_CONFIG: Dict[str, Any] = {
    "scene": "landscape", "width": 64, "height": 48, "seed": 1, "palette": "dusk",
    "techniques": ["dither", "outline"], "ridges": 3, "trees": 6, "clouds": 3, "lake": True, "sun": True,
    "objects": 3, "light": [-1, -1], "scale": 6, "hud": {"metrics": [0.5, 0.3, 0.8, 0.2], "grid_step": 16},
    "compute": {"trigger": "now", "budget_ms": 250, "defer_if_health": "critical", "pattern": None},
}


def normalize_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    raw = raw or {}
    if not isinstance(raw, dict):
        raise BytecodeError("config must be an object")
    unknown = set(raw) - set(DEFAULT_CONFIG)
    if unknown:
        raise BytecodeError(f"unknown config keys: {sorted(unknown)}")
    for k, v in raw.items():
        if isinstance(DEFAULT_CONFIG[k], dict) and isinstance(v, dict):
            cfg[k].update(v)
        else:
            cfg[k] = v

    def intr(key: str, lo: int, hi: int) -> int:
        v = cfg[key]
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise BytecodeError(f"{key} must be an integer in {lo}..{hi}")
        return v

    if cfg["scene"] not in SCENES:
        raise BytecodeError(f"scene must be one of {sorted(SCENES)}")
    if cfg["palette"] not in C.PALETTES:
        raise BytecodeError(f"palette must be one of {C.PALETTE_IDS}")
    intr("width", 16, 255), intr("height", 16, 255), intr("seed", 0, 2 ** 32 - 1)
    intr("ridges", 1, 4), intr("trees", 0, 30), intr("clouds", 0, 8), intr("objects", 1, 6), intr("scale", 1, 16)
    if not isinstance(cfg["techniques"], list) or not set(cfg["techniques"]) <= TECHNIQUES:
        raise BytecodeError(f"techniques must be a list drawn from {sorted(TECHNIQUES)}")
    lt = cfg["light"]
    if not (isinstance(lt, list) and len(lt) == 2 and all(isinstance(x, int) and -100 <= x <= 100 for x in lt)):
        raise BytecodeError("light must be [dx, dy] integers in -100..100")
    hud = cfg["hud"]
    m = hud.get("metrics", [])
    if not (isinstance(m, list) and len(m) <= 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in m)):
        raise BytecodeError("hud.metrics must be up to 4 numbers")
    if not (isinstance(hud.get("grid_step"), int) and 4 <= hud["grid_step"] <= 64):
        raise BytecodeError("hud.grid_step must be an integer in 4..64")
    comp = cfg["compute"]
    if comp.get("trigger") not in COMPUTE_TRIGGERS:
        raise BytecodeError(f"compute.trigger must be one of {sorted(COMPUTE_TRIGGERS)}")
    if not (isinstance(comp.get("budget_ms"), (int, float)) and 1 <= comp["budget_ms"] <= 60000):
        raise BytecodeError("compute.budget_ms must be 1..60000")
    if comp.get("defer_if_health") not in ("ok", "degraded", "critical", None):
        raise BytecodeError("compute.defer_if_health must be ok|degraded|critical")
    return cfg


def compile_config(raw: Optional[Dict[str, Any]]) -> Program:
    """Turn declarative parameters into an explicit program. All randomness is resolved *here*,
    so the resulting bytes fully describe the picture."""
    cfg = normalize_config(raw)
    W, H = cfg["width"], cfg["height"]
    rng = random.Random(cfg["seed"])
    tech = set(cfg["techniques"])
    p = Program()
    p.emit("CANVAS", W, H).emit("PALETTE", C.PALETTE_IDS.index(cfg["palette"]))
    p.emit("SEED", *cfg["seed"].to_bytes(4, "big"))
    p.emit("LIGHT", cfg["light"][0] + 128, cfg["light"][1] + 128)
    p.emit("DITHER", 1 if "dither" in tech else 0)
    if cfg["scene"] == "landscape":
        horizon = int(H * rng.uniform(0.50, 0.66))
        p.emit("HORIZON", horizon).emit("SKY", 1 if "dither" in tech else 0)
        if cfg["sun"]:
            p.emit("DISC", rng.randint(W // 5, W * 4 // 5), rng.randint(H // 8, max(H // 8 + 1, horizon - 6)), max(2, W // 14), C.R["sun"])
        for _ in range(cfg["clouds"]):
            p.emit("CLOUD", rng.randint(0, W - 8), rng.randint(2, max(3, horizon - 8)), rng.randint(W // 8, W // 4), C.R["cloud"])
        for i in range(cfg["ridges"]):
            frac = i / max(1, cfg["ridges"])
            p.emit("RIDGE", int(horizon - (H // 6) * (1 - frac) + (H // 10) * frac), rng.randint(35, 70), C.R["ridge0"] + i, max(4, H // 5 - i * 3))
        if cfg["lake"]:
            p.emit("LAKE", min(H - 4, horizon + max(4, H // 8)), C.R["water"])
        for _ in range(cfg["trees"]):
            x = rng.randint(2, W - 3)
            p.emit("TREE", x, rng.randint(horizon + 2, H - 2), rng.randint(max(6, H // 8), max(7, H // 4)), C.R["tree"])
    else:
        base = int(H * 0.62)
        p.emit("TABLE", base)
        p.emit("VASE", W // 4, base + 2, int(H * 0.5), C.R["obj2"])
        p.emit("BOTTLE", W * 3 // 4, base + 3, int(H * 0.55), C.R["obj1"])
        for i in range(cfg["objects"]):
            p.emit("FRUIT", rng.randint(W // 3, W * 2 // 3), base + rng.randint(2, max(3, H // 3)), rng.randint(max(3, H // 14), max(4, H // 8)),
                   C.R["obj0"] + (i % 2))
    if "posterize" in tech:
        p.emit("POSTERIZE", 4)
    if "outline" in tech:
        p.emit("OUTLINE")
    if "grid" in tech:
        p.emit("GRID", cfg["hud"]["grid_step"])
    if "hud" in tech:
        vals = [max(0.0, min(1.0, float(v))) for v in cfg["hud"]["metrics"]] + [0.0] * 4
        p.emit("BARS", *[int(round(v * 255)) for v in vals[:4]])
    p.emit("END")
    return p


# ---------------------------------------------------------------------- replication
def _perturb(rng: random.Random, value: int, lo: int, hi: int) -> int:
    span = max(1, (hi - lo) // 8)
    return max(lo, min(hi, value + rng.randint(-span, span)))


def mutate(prog: Program, rng: random.Random, rate: float = 0.08) -> Program:
    """Point-mutate args inside their valid ranges; rarely delete or duplicate a *drawing* op."""
    ops = []
    for op, args in prog.ops:
        spec = OPS[op][2]
        if op in HEADER_OPS or not spec:
            ops.append((op, args))
            continue
        r = rng.random()
        if r < rate * 0.15:
            continue                                   # deletion
        new = tuple(_perturb(rng, a, lo, hi) if rng.random() < rate * 3 else a for a, (lo, hi) in zip(args, spec))
        ops.append((op, new))
        if rate * 0.15 <= r < rate * 0.30 and len(ops) < MAX_OPS - 2:
            ops.append((op, tuple(_perturb(rng, a, lo, hi) for a, (lo, hi) in zip(new, spec))))   # duplication
    child = Program(ops)
    child.validate()
    return child


def replicate(prog: Program, rng: random.Random, rate: float = 0.08) -> Dict[str, Any]:
    """Produce a child genome and report parentage, so lineages can be logged and replayed."""
    child = mutate(prog, rng, rate)
    return {"child": child, "parent_digest": prog.digest(), "child_digest": child.digest(), "identical": child == prog}


def crossover(a: Program, b: Program, rng: random.Random) -> Program:
    head_a = [o for o in a.ops if o[0] in HEADER_OPS and o[0] != 0xFF]
    body_a = [o for o in a.ops if o[0] not in HEADER_OPS]
    body_b = [o for o in b.ops if o[0] not in HEADER_OPS]
    ca, cb = rng.randint(0, len(body_a)), rng.randint(0, len(body_b))
    ops = head_a + body_a[:ca] + body_b[cb:] + [(0xFF, ())]
    child = Program(ops[:MAX_OPS - 1] + ([(0xFF, ())] if len(ops) > MAX_OPS - 1 else []))
    child.validate()
    return child
