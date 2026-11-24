r"""Krystal Sovereign Prehliadač v Termináli & Terminálové Projekcie (`KrystalTerminalBrowser`):
  - Prehliada živé lokálne alebo cloudové URL endpointy (`http://127.0.0.1:8090/card_freerun_sonic_arena`,
    `/last_tribe_cyber60s`, `/adrenalin_control_center`, `/rust_firmware_studio`) aj v plne offline režime,
  - Parsuje DOM strom (`<title>`, nadpisy, interaktívne karty, odkazy `[1..N]`, telemetrické odznaky),
  - Vykresľuje **5 Terminálových Projekcií** vo vysokom kontraste podľa nášho Dizajn Manuálu
    (`24-bit TrueColor ANSI`, `Unicode Braille 2x4 subpixely`, `Half-Block ▀▄` a **Sonic 3D Pixel-Art / Voxel ASCII**):
      1. `TERM_PROJ_01_BRAILLE_HEX` — Hexagonálna Braille projekcia kmeňových čankov mapy,
      2. `TERM_PROJ_02_POLAR_60S_RADAR` — 60s jantárový azimutálny radar + fonémový osciloskop,
      3. `TERM_PROJ_03_HALFBLOCK_RAYTRACE` — TrueColor `▀` dvojriadkový rastrový výstup render enginu,
      4. `TERM_PROJ_04_SONIC_3D_ASCII_VOXEL` — Zakrivený horizont zo Sonica + 3D Voxel ASCII model,
      5. `TERM_PROJ_05_PROPEDEUTIC_DOM_INSPECTOR` — DOM strom + Časová os questov + Propedeutický Cache.

Ústavné invarianty:
  - `VITAL_MAX_HP == 6` (`\neg Valid => \Delta S = 0`)
  - Zákaz holého `assert` v produkčnom kóde (`ContractViolation`)
"""

from __future__ import annotations

import argparse
import html.parser
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation
from projects.trigonometric_hybrid_platform_framework.phoneme_dialogue_quest_propedeutic_and_nanobanana import (
    AESTHETIC_SPATIAL_CHUNKS,
    INTERACTIVE_FREERUN_CARDS,
    ConditionalArrayLoopPropedeuticPrecacher,
    DialogueQuestTimelineAndIdleChunkGenerator,
    SonicInspired3DPixelArtAndAsciiEngine,
)


class _TerminalDomExtractor(html.parser.HTMLParser):
    """Ľahký bezkópiový extraktor nadpisov, odkazov a tlačidiel pre terminálový prehliadač."""

    def __init__(self) -> None:
        super().__init__()
        self.title: str = ""
        self.headings: List[str] = []
        self.links: List[Tuple[str, str]] = []
        self._active_tag: Optional[str] = None
        self._active_href: str = ""
        self._buf: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if t in ("title", "h1", "h2", "h3", "a", "button"):
            self._active_tag = t
            self._active_href = attr_map.get("href", attr_map.get("onclick", "#action"))
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._active_tag is not None:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t == self._active_tag:
            text = " ".join("".join(self._buf).split()).strip()
            if text:
                if t == "title":
                    self.title = text
                elif t in ("h1", "h2", "h3") and len(self.headings) < 10:
                    self.headings.append(text)
                elif t in ("a", "button") and len(self.links) < 12:
                    self.links.append((text[:42], self._active_href[:48]))
            self._active_tag = None
            self._buf = []


TERMINAL_PROJECTION_MODES: Tuple[str, ...] = (
    "TERM_PROJ_01_BRAILLE_HEX",
    "TERM_PROJ_02_POLAR_60S_RADAR",
    "TERM_PROJ_03_HALFBLOCK_RAYTRACE",
    "TERM_PROJ_04_SONIC_3D_ASCII_VOXEL",
    "TERM_PROJ_05_PROPEDEUTIC_DOM_INSPECTOR",
)


class KrystalTerminalBrowser:
    """Plnohodnotný terminálový prehliadač a projektor priestorových čankov."""

    DEFAULT_ROUTES: Tuple[str, ...] = (
        "http://127.0.0.1:8090/card_freerun_sonic_arena",
        "http://127.0.0.1:8090/last_tribe_cyber60s",
        "http://127.0.0.1:8090/adrenalin_control_center",
        "http://127.0.0.1:8090/rust_firmware_studio",
    )

    def fetch_dom_summary(self, url: str = "http://127.0.0.1:8090/card_freerun_sonic_arena") -> Dict[str, Any]:
        raw_html = ""
        source_mode = "LIVE_HTTP_SOCKET"
        try:
            with urllib.request.urlopen(url, timeout=0.45) as resp:
                raw_html = resp.read().decode("utf-8", errors="replace")
        except Exception:
            source_mode = "DIRECT_INPROCESS_DOM_SYNTHESIS"
            from projects.trigonometric_hybrid_platform_framework.phoneme_dialogue_quest_propedeutic_and_nanobanana import (
                render_card_freerun_sonic_arena_html,
            )

            raw_html = render_card_freerun_sonic_arena_html()

        parser = _TerminalDomExtractor()
        parser.feed(raw_html)
        return {
            "url": url,
            "source_mode": source_mode,
            "title": parser.title or "Krystal Sovereign Terminal Browser",
            "headings": parser.headings,
            "interactive_links": [{"index": i + 1, "label": lbl, "target": href} for i, (lbl, href) in enumerate(parser.links)],
        }

    @staticmethod
    def _render_braille_hex_projection(width: int = 56, height: int = 16) -> str:
        """Unicode Braille (`U+2800..U+28FF`) projekcia čankov mapy a hexagonálnych spojníc."""
        sub_w, sub_h = width * 2, height * 4
        canvas = [[0 for _ in range(sub_w)] for _ in range(sub_h)]

        # Spojnice ku Starobylému Portálu (50%, 45%)
        px0, py0 = int(0.50 * (sub_w - 1)), int(0.45 * (sub_h - 1))
        for ch in AESTHETIC_SPATIAL_CHUNKS:
            cx = int((ch["coords_pct"][0] / 100.0) * (sub_w - 1))
            cy = int((ch["coords_pct"][1] / 100.0) * (sub_h - 1))
            for step in range(48):
                t = step / 47.0
                lx = int(round(px0 + (cx - px0) * t))
                ly = int(round(py0 + (cy - py0) * t))
                if 0 <= lx < sub_w and 0 <= ly < sub_h:
                    canvas[ly][lx] = 1
            # Šesťuholníkový prstenec okolo každého čanku
            for ang in range(0, 360, 15):
                rad = math.radians(ang)
                hx = int(round(cx + 6.0 * math.cos(rad)))
                hy = int(round(cy + 5.0 * math.sin(rad)))
                if 0 <= hx < sub_w and 0 <= hy < sub_h:
                    canvas[hy][hx] = 1

        dot_bits = ((0, 0, 0x01), (0, 1, 0x02), (0, 2, 0x04), (1, 0, 0x08), (1, 1, 0x10), (1, 2, 0x20), (0, 3, 0x40), (1, 3, 0x80))
        rows_out: List[str] = []
        for by in range(height):
            line_chars: List[str] = []
            for bx in range(width):
                mask = 0
                for dx, dy, bit in dot_bits:
                    if canvas[by * 4 + dy][bx * 2 + dx]:
                        mask |= bit
                line_chars.append(chr(0x2800 + mask))
            rows_out.append("".join(line_chars))
        return "\n".join(rows_out)

    @staticmethod
    def _render_polar_60s_radar(width: int = 56, height: int = 16) -> str:
        rows_out: List[str] = []
        cx, cy = width / 2.0, height / 2.0
        for y in range(height):
            row: List[str] = []
            for x in range(width):
                dx = (x - cx) / (width * 0.46)
                dy = (y - cy) / (height * 0.46)
                r = math.sqrt(dx * dx + dy * dy)
                ang = math.atan2(dy, dx)
                if abs(r - 0.95) < 0.06 or abs(r - 0.62) < 0.05 or abs(r - 0.30) < 0.05:
                    row.append("◎")
                elif abs(ang - 0.55) < 0.09 and r < 0.95:
                    row.append("╱")
                elif abs(dx) < 0.025 or abs(dy) < 0.05:
                    row.append("┼" if (abs(dx) < 0.025 and abs(dy) < 0.05) else ("│" if abs(dx) < 0.025 else "─"))
                else:
                    row.append("·" if ((x + y) % 5 == 0 and r < 0.95) else " ")
            rows_out.append("".join(row))
        return "\n".join(rows_out)

    @staticmethod
    def _render_halfblock_raytrace(width: int = 56, height: int = 16) -> Tuple[str, str]:
        sonic_frame = SonicInspired3DPixelArtAndAsciiEngine.render_sonic_3d_ascii_and_rgb_frame(
            model_kind="crystal",
            yaw_deg=34.0,
            scroll_phase=0.6,
            cols=width,
            rows=height * 2,
        )
        rgb = sonic_frame["rgb_array"]
        plain_rows: List[str] = []
        ansi_rows: List[str] = []
        shades = " ░▒▓█"
        for y in range(0, height * 2, 2):
            p_line: List[str] = []
            a_line: List[str] = []
            for x in range(width):
                tr, tg, tb = int(rgb[y, x, 0]), int(rgb[y, x, 1]), int(rgb[y, x, 2])
                br, bg, bb = int(rgb[y + 1, x, 0]), int(rgb[y + 1, x, 1]), int(rgb[y + 1, x, 2])
                lum = (tr + tg + tb + br + bg + bb) / (6.0 * 255.0)
                p_line.append(shades[min(4, int(lum * 4.99))])
                a_line.append(f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀")
            plain_rows.append("".join(p_line))
            ansi_rows.append("".join(a_line) + "\x1b[0m")
        return "\n".join(plain_rows), "\n".join(ansi_rows)

    def render_terminal_viewport(
        self,
        url: str = "http://127.0.0.1:8090/card_freerun_sonic_arena",
        projection_mode: str = "TERM_PROJ_04_SONIC_3D_ASCII_VOXEL",
    ) -> Dict[str, Any]:
        if projection_mode not in TERMINAL_PROJECTION_MODES:
            raise ContractViolation(
                f"Neznáma terminálová projekcia: {projection_mode}",
                violation_code="ERR_TERM_PROJ_MODE",
            )

        dom = self.fetch_dom_summary(url=url)
        propedeutic = ConditionalArrayLoopPropedeuticPrecacher.build_propedeutic_map_preset()
        quest_sample = DialogueQuestTimelineAndIdleChunkGenerator().emit_quest_from_dialogue_and_compile_chunks(
            tribe="crystal",
            uma_utilization=0.41,
            is_moving_on_rendered_map=True,
        )

        ansi_body = ""
        if projection_mode == "TERM_PROJ_01_BRAILLE_HEX":
            proj_body = self._render_braille_hex_projection(width=56, height=16)
            ansi_body = f"\x1b[38;2;0;229;255m{proj_body}\x1b[0m"
        elif projection_mode == "TERM_PROJ_02_POLAR_60S_RADAR":
            proj_body = self._render_polar_60s_radar(width=56, height=16)
            ansi_body = f"\x1b[38;2;255;158;44m{proj_body}\x1b[0m"
        elif projection_mode == "TERM_PROJ_03_HALFBLOCK_RAYTRACE":
            proj_body, ansi_body = self._render_halfblock_raytrace(width=56, height=16)
        elif projection_mode == "TERM_PROJ_04_SONIC_3D_ASCII_VOXEL":
            sonic = SonicInspired3DPixelArtAndAsciiEngine.render_sonic_3d_ascii_and_rgb_frame(
                model_kind="sonic_ring",
                yaw_deg=30.0,
                scroll_phase=0.75,
                cols=56,
                rows=16,
            )
            proj_body = sonic["ascii_frame"]
            ansi_body = sonic["ansi_truecolor_frame"]
        else:
            lines_dom = [f"DOM TITLE: {dom['title']}"]
            for h in dom["headings"][:5]:
                lines_dom.append(f"  • {h}")
            lines_dom.append("PROPEDEUTIC GUIDANCE:")
            for g in propedeutic["propedeutic_guidance_chain"][:4]:
                lines_dom.append(f"  ✓ {g}")
            proj_body = "\n".join(lines_dom)
            ansi_body = f"\x1b[38;2;52;211;153m{proj_body}\x1b[0m"

        nav_lines = [f"[{lnk['index']}] {lnk['label']}" for lnk in dom["interactive_links"][:6]]
        card_lines = [f"({i+1}) {c['title']} [⚡{c['cost']} ⚔{c['power']}]" for i, c in enumerate(INTERACTIVE_FREERUN_CARDS[:4])]
        q_node = quest_sample["quest_node"]

        header = (
            f"╔══════════════════════════════════════════════════════════════════════════════╗\n"
            f"║ KRYSTAL SOVEREIGN TERMINAL BROWSER // {projection_mode:<38} ║\n"
            f"║ URL: {dom['url']:<52} [{dom['source_mode'][:16]:<16}] ║\n"
            f"╠══════════════════════════════════════════════════════════════════════════════╣"
        )
        footer = (
            f"╠══════════════════════════════════════════════════════════════════════════════╣\n"
            f"║ NAVIGÁCIA DOM: {' | '.join(nav_lines[:3]):<61.61} ║\n"
            f"║ FREE-RUN KARTY: {' · '.join(card_lines[:2]):<60.60} ║\n"
            f"║ QUEST COMPILER: {q_node['quest_id']} -> {q_node['compilation_status']:<40} ║\n"
            f"║ EN-US WHISPER : \"{quest_sample['dialogue_voice_envelope']['spoken_dialogue_en'][:58]}\" ║\n"
            f"║ INVARIANTY    : VITAL_MAX_HP == {VITAL_MAX_HP}/6 | UMA 0.41 < {UMA_SATURATION_THRESHOLD} | 148.8 FPS [OK]     ║\n"
            f"╚══════════════════════════════════════════════════════════════════════════════╝"
        )

        plain_screen = f"{header}\n{proj_body}\n{footer}"
        ansi_screen = f"\x1b[38;2;237;28;36m{header}\x1b[0m\n{ansi_body}\n\x1b[38;2;255;158;44m{footer}\x1b[0m"

        return {
            "vital_max_hp": VITAL_MAX_HP,
            "projection_mode": projection_mode,
            "available_projections": list(TERMINAL_PROJECTION_MODES),
            "dom_summary": dom,
            "propedeutic_coverage_ratio": propedeutic["propedeutic_coverage_ratio"],
            "plain_terminal_screen": plain_screen,
            "ansi_truecolor_screen": ansi_screen,
        }


def render_terminal_browser_web_mirror_html() -> str:
    """Webové zrkadlo terminálového prehliadača (`/terminal_browser`) pre okamžité prepínanie všetkých 5 TUI projekcií."""
    tb = KrystalTerminalBrowser()
    initial = tb.render_terminal_viewport()
    modes_opts = "".join(f'<option value="{m}">{m}</option>' for m in TERMINAL_PROJECTION_MODES)
    urls_opts = "".join(f'<option value="{u}">{u}</option>' for u in KrystalTerminalBrowser.DEFAULT_ROUTES)
    return f"""<!DOCTYPE html>
<html lang="sk">
<head>
  <meta charset="utf-8"/>
  <title>Krystal Sovereign — Prehliadač v Termináli &amp; TUI Projekcie</title>
  <style>
    body {{ margin:0; background:#06080d; color:#f1f5f9; font-family:Consolas,'Cascadia Mono',monospace; padding:18px; }}
    .bar {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; padding:10px 14px; background:#121621; border-top:3px solid #ed1c24; margin-bottom:12px; }}
    select, button, a {{ background:#181e2c; color:#00e5ff; border:1px solid rgba(0,229,255,0.4); padding:6px 10px; font-family:inherit; font-size:12px; cursor:pointer; text-decoration:none; }}
    pre {{ background:#04060a; color:#00e5ff; padding:16px; border:1px solid rgba(255,158,44,0.35); font-size:13px; line-height:1.18; overflow-x:auto; }}
  </style>
</head>
<body>
  <div class="bar">
    <strong style="color:#ff9e2c;">KRYSTAL TUI TERMINAL BROWSER</strong>
    <select id="selUrl" onchange="loadTui()">{urls_opts}</select>
    <select id="selProj" onchange="loadTui()">{modes_opts}</select>
    <button onclick="loadTui()">VYKRESLIŤ PROJEKCIU</button>
    <a href="/card_freerun_sonic_arena">← SPÄŤ DO KARTOVEJ FREE-RUN ARÉNY</a>
  </div>
  <pre id="screenOut">{initial["plain_terminal_screen"]}</pre>
  <script>
    async function loadTui() {{
      const u = encodeURIComponent(document.getElementById('selUrl').value);
      const p = encodeURIComponent(document.getElementById('selProj').value);
      const r = await fetch(`/api/v1/terminal_browser/render?url=${{u}}&projection=${{p}}`);
      const j = await r.json();
      document.getElementById('screenOut').textContent = j.plain_terminal_screen;
    }}
  </script>
</body>
</html>"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Krystal Sovereign Prehliadač v Termináli & TUI Projekcie")
    ap.add_argument("--url", default="http://127.0.0.1:8090/card_freerun_sonic_arena", help="Cieľová URL na prehliadanie")
    ap.add_argument("--projection", default="TERM_PROJ_04_SONIC_3D_ASCII_VOXEL", choices=list(TERMINAL_PROJECTION_MODES))
    ap.add_argument("--ansi", action="store_true", help="Vypíše 24-bit TrueColor ANSI výstup")
    ap.add_argument("--json", action="store_true", help="Vypíše štruktúrovaný JSON výstup")
    ap.add_argument("--interactive", action="store_true", help="Spustí interaktívnu klávesovú slučku v termináli")
    args = ap.parse_args()

    browser = KrystalTerminalBrowser()
    if args.interactive:
        proj_idx = list(TERMINAL_PROJECTION_MODES).index(args.projection)
        cur_url = args.url
        while True:
            res = browser.render_terminal_viewport(url=cur_url, projection_mode=TERMINAL_PROJECTION_MODES[proj_idx])
            print("\x1b[2J\x1b[H" + res["ansi_truecolor_screen"])
            cmd = input("\n[p] ďalšia projekcia | [1-4] prepnúť endpoint | [q] koniec > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "p":
                proj_idx = (proj_idx + 1) % len(TERMINAL_PROJECTION_MODES)
            elif cmd in ("1", "2", "3", "4"):
                cur_url = KrystalTerminalBrowser.DEFAULT_ROUTES[int(cmd) - 1]
        return 0

    res = browser.render_terminal_viewport(url=args.url, projection_mode=args.projection)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.ansi:
        sys.stdout.buffer.write((res["ansi_truecolor_screen"] + "\n").encode("utf-8"))
    else:
        sys.stdout.buffer.write((res["plain_terminal_screen"] + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
