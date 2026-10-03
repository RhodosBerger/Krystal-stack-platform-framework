"""Pixel-art canvas, geometry primitives, palettes, scene painters, PNG/SVG/ASCII writers (stdlib only).

Everything here is deterministic given (config, seed). Scenes are *painted by the bytecode VM*
(`krystal_bot.bytecode`), so the same program bytes always reproduce the same picture.
"""
from __future__ import annotations

import math
import random
import struct
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROLES = ["sky0", "sky1", "sky2", "sky3", "sun", "ridge0", "ridge1", "ridge2", "ridge3", "water", "tree", "cloud",
         "outline", "table", "wall", "obj0", "obj1", "obj2", "shadow", "hud"]
R = {name: i for i, name in enumerate(ROLES)}

PALETTES: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "dusk": {"sky0": (24, 20, 56), "sky1": (74, 44, 110), "sky2": (176, 82, 122), "sky3": (250, 160, 110), "sun": (255, 224, 150),
             "ridge0": (96, 70, 130), "ridge1": (70, 52, 106), "ridge2": (46, 36, 82), "ridge3": (26, 22, 52), "water": (60, 56, 112),
             "tree": (16, 30, 40), "cloud": (200, 130, 150), "outline": (10, 8, 24), "table": (80, 50, 44), "wall": (52, 40, 70),
             "obj0": (230, 90, 70), "obj1": (240, 190, 70), "obj2": (110, 190, 120), "shadow": (30, 20, 40), "hud": (120, 255, 220)},
    "noon": {"sky0": (60, 130, 220), "sky1": (100, 168, 238), "sky2": (150, 200, 248), "sky3": (208, 232, 252), "sun": (255, 250, 200),
             "ridge0": (140, 168, 190), "ridge1": (96, 140, 130), "ridge2": (60, 112, 80), "ridge3": (36, 82, 50), "water": (70, 140, 200),
             "tree": (22, 70, 40), "cloud": (255, 255, 255), "outline": (20, 40, 30), "table": (150, 100, 60), "wall": (214, 200, 170),
             "obj0": (214, 60, 52), "obj1": (244, 200, 60), "obj2": (80, 160, 80), "shadow": (90, 70, 60), "hud": (255, 60, 120)},
    "gameboy": {"sky0": (155, 188, 15), "sky1": (155, 188, 15), "sky2": (139, 172, 15), "sky3": (139, 172, 15), "sun": (15, 56, 15),
                "ridge0": (139, 172, 15), "ridge1": (48, 98, 48), "ridge2": (48, 98, 48), "ridge3": (15, 56, 15), "water": (139, 172, 15),
                "tree": (15, 56, 15), "cloud": (155, 188, 15), "outline": (15, 56, 15), "table": (48, 98, 48), "wall": (139, 172, 15),
                "obj0": (15, 56, 15), "obj1": (48, 98, 48), "obj2": (15, 56, 15), "shadow": (15, 56, 15), "hud": (15, 56, 15)},
    "vapor": {"sky0": (40, 10, 90), "sky1": (120, 30, 160), "sky2": (240, 80, 180), "sky3": (255, 170, 160), "sun": (255, 230, 120),
              "ridge0": (60, 200, 220), "ridge1": (40, 150, 210), "ridge2": (30, 90, 180), "ridge3": (20, 40, 110), "water": (30, 60, 150),
              "tree": (10, 20, 60), "cloud": (255, 140, 200), "outline": (8, 4, 30), "table": (90, 30, 120), "wall": (60, 20, 100),
              "obj0": (255, 90, 160), "obj1": (255, 230, 100), "obj2": (80, 240, 200), "shadow": (30, 10, 60), "hud": (255, 255, 255)},
    "mono": {"sky0": (20, 20, 20), "sky1": (60, 60, 60), "sky2": (110, 110, 110), "sky3": (170, 170, 170), "sun": (255, 255, 255),
             "ridge0": (140, 140, 140), "ridge1": (100, 100, 100), "ridge2": (64, 64, 64), "ridge3": (32, 32, 32), "water": (80, 80, 80),
             "tree": (20, 20, 20), "cloud": (200, 200, 200), "outline": (0, 0, 0), "table": (70, 70, 70), "wall": (120, 120, 120),
             "obj0": (230, 230, 230), "obj1": (190, 190, 190), "obj2": (150, 150, 150), "shadow": (10, 10, 10), "hud": (255, 255, 255)},
}
PALETTES["aurora"] = {"sky0": (4, 12, 32), "sky1": (8, 40, 70), "sky2": (20, 120, 110), "sky3": (110, 255, 170), "sun": (230, 255, 240),
                      "ridge0": (30, 70, 90), "ridge1": (20, 50, 70), "ridge2": (12, 32, 52), "ridge3": (6, 18, 34), "water": (14, 60, 90),
                      "tree": (4, 14, 24), "cloud": (120, 255, 200), "outline": (2, 6, 14), "table": (30, 40, 60), "wall": (16, 28, 48),
                      "obj0": (255, 120, 200), "obj1": (255, 240, 130), "obj2": (120, 255, 220), "shadow": (6, 14, 28), "hud": (200, 255, 255)}
DONOR_PALETTES = {"aurora"}   # appended last so existing palette indices stay stable in saved bytecode
PALETTE_IDS = list(PALETTES)

BAYER4 = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]
MAX_DIM = 256
MAX_OUT_PIXELS = 1_500_000


class Canvas:
    def __init__(self, w: int, h: int, palette: str = "dusk"):
        if not (8 <= w <= MAX_DIM and 8 <= h <= MAX_DIM):
            raise ValueError(f"canvas must be 8..{MAX_DIM} px per side")
        if palette not in PALETTES:
            raise ValueError(f"unknown palette {palette!r}; choose one of {PALETTE_IDS}")
        self.w, self.h, self.palette_name = w, h, palette
        self.px = [bytearray([R["sky0"]]) * w for _ in range(h)]
        self.mask = [bytearray(w) for _ in range(h)]   # 1 = object pixel (gets outlined)
        self.overlay = [bytearray([255]) * w for _ in range(h)]  # 255 = transparent (AR layer)
        self.horizon = h // 2

    # -- primitives ---------------------------------------------------------------
    def put(self, x: int, y: int, c: int, obj: bool = False) -> None:
        if 0 <= x < self.w and 0 <= y < self.h:
            self.px[y][x] = c
            if obj:
                self.mask[y][x] = 1

    def hline(self, x0: int, x1: int, y: int, c: int, obj: bool = False) -> None:
        if not 0 <= y < self.h:
            return
        for x in range(max(0, min(x0, x1)), min(self.w - 1, max(x0, x1)) + 1):
            self.px[y][x] = c
            if obj:
                self.mask[y][x] = 1

    def rect(self, x: int, y: int, w: int, h: int, c: int, obj: bool = False) -> None:
        for yy in range(y, y + h):
            self.hline(x, x + w - 1, yy, c, obj)

    def line(self, x0: int, y0: int, x1: int, y1: int, c: int, obj: bool = False) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        for _ in range(self.w + self.h + 4):
            self.put(x0, y0, c, obj)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def ellipse(self, cx: int, cy: int, rx: int, ry: int, c: int, obj: bool = False) -> None:
        if rx <= 0 or ry <= 0:
            self.put(cx, cy, c, obj)
            return
        for y in range(-ry, ry + 1):
            span = int(rx * math.sqrt(max(0.0, 1.0 - (y / ry) ** 2)) + 0.5)
            self.hline(cx - span, cx + span, cy + y, c, obj)

    def polygon(self, pts: Sequence[Tuple[int, int]], c: int, obj: bool = False) -> None:
        """Scanline even-odd fill."""
        ys = [p[1] for p in pts]
        for y in range(max(0, min(ys)), min(self.h - 1, max(ys)) + 1):
            xs = []
            for i in range(len(pts)):
                (x0, y0), (x1, y1) = pts[i], pts[(i + 1) % len(pts)]
                if y0 == y1:
                    continue
                if min(y0, y1) <= y < max(y0, y1):
                    xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
            xs.sort()
            for a, b in zip(xs[::2], xs[1::2]):
                self.hline(int(math.ceil(a)), int(math.floor(b)), y, c, obj)

    # -- post passes --------------------------------------------------------------
    def outline(self, c: int) -> None:
        marks = []
        for y in range(self.h):
            for x in range(self.w):
                if self.mask[y][x]:
                    continue
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.w and 0 <= ny < self.h and self.mask[ny][nx]:
                        marks.append((x, y))
                        break
        for x, y in marks:
            self.px[y][x] = c

    def posterize(self, levels: int) -> None:
        levels = max(2, min(levels, 8))
        pal = PALETTES[self.palette_name]
        order = [pal[n] for n in ROLES]
        lum = [0.3 * r + 0.59 * g + 0.11 * b for r, g, b in order]
        step = 256.0 / levels
        keep: Dict[int, int] = {}
        for i, l in enumerate(lum):
            q = int(l // step)
            keep.setdefault(q, i)
        for row in self.px:
            for x, v in enumerate(row):
                row[x] = keep[int(lum[v] // step)]

    def grid_overlay(self, step: int, c: int) -> None:
        step = max(4, step)
        for y in range(0, self.h, step):
            for x in range(self.w):
                if (x // 2) % 2 == 0:
                    self.overlay[y][x] = c
        for x in range(0, self.w, step):
            for y in range(self.h):
                if (y // 2) % 2 == 0:
                    self.overlay[y][x] = c

    def bars_overlay(self, values: Sequence[float], c: int) -> None:
        n = max(1, min(len(values), 8))
        bw = max(2, self.w // (n * 2 + 1))
        for i, v in enumerate(values[:n]):
            v = 0.0 if v != v else min(1.0, max(0.0, float(v)))
            hgt = max(1, int(v * (self.h // 4)))
            x0 = bw + i * 2 * bw
            for y in range(self.h - 2 - hgt, self.h - 2):
                for x in range(x0, min(self.w, x0 + bw)):
                    self.overlay[y][x] = c

    # -- export ---------------------------------------------------------------------
    def composite(self, with_overlay: bool = True) -> List[bytes]:
        rows = []
        for y in range(self.h):
            r = bytearray(self.px[y])
            if with_overlay:
                o = self.overlay[y]
                for x in range(self.w):
                    if o[x] != 255:
                        r[x] = o[x]
            rows.append(bytes(r))
        return rows

    def fingerprint(self) -> str:
        import hashlib
        h = hashlib.blake2b(digest_size=8)
        for r in self.composite(True):
            h.update(r)
        return h.hexdigest()

    def to_png(self, scale: int = 4, with_overlay: bool = True) -> bytes:
        scale = max(1, min(int(scale), 16))
        if self.w * self.h * scale * scale > MAX_OUT_PIXELS:
            scale = max(1, int(math.sqrt(MAX_OUT_PIXELS / (self.w * self.h))))
        pal = PALETTES[self.palette_name]
        plte = b"".join(bytes(pal[n]) for n in ROLES)
        raw = bytearray()
        for row in self.composite(with_overlay):
            line = bytearray()
            for v in row:
                line.extend(bytes([v]) * scale)
            for _ in range(scale):
                raw.append(0)
                raw.extend(line)

        def chunk(tag: bytes, data: bytes) -> bytes:
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        ihdr = struct.pack(">IIBBBBB", self.w * scale, self.h * scale, 8, 3, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"PLTE", plte) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b"")

    def to_svg(self, scale: int = 4, with_overlay: bool = True) -> str:
        pal = PALETTES[self.palette_name]
        scale = max(1, min(int(scale), 16))
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w * scale}" height="{self.h * scale}" '
                 f'viewBox="0 0 {self.w} {self.h}" shape-rendering="crispEdges">']
        for y, row in enumerate(self.composite(with_overlay)):
            x = 0
            while x < self.w:
                x2 = x
                while x2 + 1 < self.w and row[x2 + 1] == row[x]:
                    x2 += 1
                r, g, b = pal[ROLES[row[x]]]
                parts.append(f'<rect x="{x}" y="{y}" width="{x2 - x + 1}" height="1" fill="#{r:02x}{g:02x}{b:02x}"/>')
                x = x2 + 1
        parts.append("</svg>")
        return "".join(parts)

    def to_ascii(self, with_overlay: bool = True) -> str:
        ramp = " .:-=+*#%@"
        pal = PALETTES[self.palette_name]
        lum = [0.3 * pal[n][0] + 0.59 * pal[n][1] + 0.11 * pal[n][2] for n in ROLES]
        return "\n".join("".join(ramp[min(9, int(lum[v] / 25.6))] for v in row) for row in self.composite(with_overlay))


# ---------------------------------------------------------------------- scene painters
def paint_sky(cv: Canvas, style: int, dither: bool) -> None:
    """style 0 = 4 hard bands, 1 = Bayer-dithered gradient (the usual pixel-art technique)."""
    h = max(2, cv.horizon)
    for y in range(cv.h):
        t = min(1.0, y / h) * 3.0
        base = int(min(2, t))
        frac = t - base
        for x in range(cv.w):
            if style == 1 or dither:
                use_next = frac * 16 > BAYER4[y % 4][x % 4]
                cv.px[y][x] = R["sky0"] + min(3, base + (1 if use_next else 0))
            else:
                cv.px[y][x] = R["sky0"] + min(3, int(t + 0.5))


def paint_disc(cv: Canvas, x: int, y: int, r: int, c: int) -> None:
    cv.ellipse(x, y, r, r, c)


def _midpoint(rng: random.Random, n: int, rough: float, lo: float, hi: float) -> List[float]:
    size = 1
    while size < n - 1:
        size *= 2
    pts = [0.0] * (size + 1)
    pts[0], pts[size] = rng.uniform(lo, hi), rng.uniform(lo, hi)
    step, amp = size, (hi - lo) * 0.5
    while step > 1:
        half = step // 2
        for i in range(half, size, step):
            pts[i] = (pts[i - half] + pts[i + half]) / 2 + rng.uniform(-amp, amp)
        step, amp = half, amp * rough
    return [min(hi, max(lo, v)) for v in pts[:n]]


def paint_ridge(cv: Canvas, rng: random.Random, base_y: int, rough: float, c: int, amp: int) -> None:
    hs = _midpoint(rng, cv.w, max(0.2, min(0.9, rough)), base_y - amp, base_y + amp * 0.3)
    for x in range(cv.w):
        top = int(hs[x])
        for y in range(max(0, top), cv.h):
            cv.px[y][x] = c


def paint_tree(cv: Canvas, x: int, y: int, h: int, c: int) -> None:
    h = max(4, h)
    cv.rect(x, y - h // 4, max(1, h // 8), h // 4 + 1, c, True)
    for i in range(3):
        top, base = y - h + i * (h // 4), y - h // 4 - (2 - i) * (h // 8)
        half = h // 4 + i * max(1, h // 8)
        cv.polygon([(x, top), (x - half, base), (x + half, base)], c, True)


def paint_cloud(cv: Canvas, rng: random.Random, x: int, y: int, w: int, c: int) -> None:
    for _ in range(4 + w // 8):
        cv.ellipse(x + rng.randint(0, w), y + rng.randint(-1, 1), max(3, w // 5), max(2, w // 9), c, True)


def paint_lake(cv: Canvas, y: int, water: int) -> None:
    for yy in range(max(0, y), cv.h):
        src = 2 * y - yy - 1
        for x in range(cv.w):
            if 0 <= src < cv.h and (x + yy) % 2 == 0:
                cv.px[yy][x] = cv.px[src][x]
            else:
                cv.px[yy][x] = water


def paint_table(cv: Canvas, y: int, wall: int, table: int) -> None:
    for yy in range(cv.h):
        for x in range(cv.w):
            cv.px[yy][x] = wall if yy < y else table
    cv.line(0, y, cv.w - 1, y, R["outline"])


def paint_vase(cv: Canvas, x: int, y: int, h: int, c: int, light_dx: int) -> None:
    h = max(8, h)
    for i in range(h):
        t = i / (h - 1)
        r = max(1, int(h * (0.16 + 0.18 * math.sin(math.pi * (0.15 + 0.9 * t)) - 0.06 * t)))
        yy = y - i
        cv.hline(x - r, x + r, yy, c, True)
        shade = x + (r if light_dx < 0 else -r)
        cv.hline(min(shade, shade + (-r // 2 if light_dx < 0 else r // 2)), shade, yy, R["shadow"] if (yy + i) % 2 == 0 else c, True)


def paint_fruit(cv: Canvas, x: int, y: int, r: int, c: int, light_dx: int, light_dy: int) -> None:
    r = max(2, r)
    cv.ellipse(x + r // 2, y + r - 1, r + r // 2, max(1, r // 3), R["shadow"])   # cast shadow
    for yy in range(-r, r + 1):
        for xx in range(-r, r + 1):
            d2 = xx * xx + yy * yy
            if d2 <= r * r:
                lit = (xx * light_dx + yy * light_dy) / (r * (abs(light_dx) + abs(light_dy) or 1))
                col = c if lit > -0.25 else R["shadow"] if (xx + yy) % 2 == 0 else c
                cv.put(x + xx, y + yy, col, True)


def paint_bottle(cv: Canvas, x: int, y: int, h: int, c: int) -> None:
    h = max(8, h)
    body_w = max(2, h // 5)
    cv.rect(x - body_w, y - h * 2 // 3, body_w * 2 + 1, h * 2 // 3, c, True)
    cv.rect(x - body_w // 3, y - h, max(1, body_w // 3 * 2 + 1), h // 3 + 1, c, True)
    cv.rect(x - body_w // 3, y - h, max(1, body_w // 3 * 2 + 1), 2, R["outline"], True)
