#!/usr/bin/env python3
# ==============================================================================
# KRYSTAL-STACK: PROCEDURAL TEXT ANIMATION & TYPOGRAPHIC ASCII COMPOSITOR
# ==============================================================================
# Module: krystal_kernel/text_animation_engine.py
# Description:
#     Real-time text animation and typography rasterization engine for Krystal Stack.
#     Supports drawing animated text onto ASCII streams, terminal consoles, and
#     3D procedural canvas buffers.
#
#     Features:
#       1. Vectorized 5x7 and 8x8 Bitmap Glyph Rasterizer (A-Z, 0-9, Invariants).
#       2. 5 Distinct Procedural Animation Modes:
#          - KINETIC_MATRIX: Matrix glyph rain condensing into target text.
#          - SINE_WAVE_FLOAT: Undulating sinusoidal wave text with depth shadow.
#          - GLITCH_CYBERPUNK: High-entropy chromatic slicing and neon block glyphs.
#          - TYPEWRITER_DECRYPT: Hexadecimal scramble resolving into plaintext.
#          - VOXEL_3D_EXTRUDE: 3D extruded voxel text with yaw/pitch rotation.
#       3. Non-Destructive Compositing:
#          - Blends animated text directly over 3D raymarching or procedural scenes.
#       4. Platform Invariant: VITAL_MAX_HP == 6 strictly enforced.
#
# Author: Dušan Kopecký & Krystal-Stack Architecture Team (2026)
# ==============================================================================

import sys
import os
import time
import math
import random
from enum import Enum, unique
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"


# ==============================================================================
# 1. 5x7 BITMAP FONT GLYPH TABLE
# ==============================================================================

# Each character is defined as 7 rows of 5 binary pixels (1 = filled, 0 = empty)
FONT_5X7: Dict[str, List[str]] = {
    'A': [" █ ", "█ █", "███", "█ █", "█ █"],
    'B': ["██ ", "█ █", "██ ", "█ █", "██ "],
    'C': [" ██", "█  ", "█  ", "█  ", " ██"],
    'D': ["██ ", "█ █", "█ █", "█ █", "██ "],
    'E': ["███", "█  ", "██ ", "█  ", "███"],
    'F': ["███", "█  ", "██ ", "█  ", "█  "],
    'G': [" ██", "█  ", "█ █", "█ █", " ██"],
    'H': ["█ █", "█ █", "███", "█ █", "█ █"],
    'I': ["███", " █ ", " █ ", " █ ", "███"],
    'J': ["  █", "  █", "  █", "█ █", " █ "],
    'K': ["█ █", "██ ", "█  ", "██ ", "█ █"],
    'L': ["█  ", "█  ", "█  ", "█  ", "███"],
    'M': ["█ █", "███", "█ █", "█ █", "█ █"],
    'N': ["██ ", "█ █", "█ █", "█ █", "█ █"],
    'O': [" █ ", "█ █", "█ █", "█ █", " █ "],
    'P': ["██ ", "█ █", "██ ", "█  ", "█  "],
    'Q': [" █ ", "█ █", "█ █", "██ ", "  █"],
    'R': ["██ ", "█ █", "██ ", "█ █", "█ █"],
    'S': [" ██", "█  ", " █ ", "  █", "██ "],
    'T': ["███", " █ ", " █ ", " █ ", " █ "],
    'U': ["█ █", "█ █", "█ █", "█ █", " █ "],
    'V': ["█ █", "█ █", "█ █", "█ █", " █ "],
    'W': ["█ █", "█ █", "█ █", "███", "█ █"],
    'X': ["█ █", "█ █", " █ ", "█ █", "█ █"],
    'Y': ["█ █", "█ █", " █ ", " █ ", " █ "],
    'Z': ["███", "  █", " █ ", "█  ", "███"],
    '0': ["███", "█ █", "█ █", "█ █", "███"],
    '1': [" █ ", "██ ", " █ ", " █ ", "███"],
    '2': ["███", "  █", "███", "█  ", "███"],
    '3': ["███", "  █", "███", "  █", "███"],
    '4': ["█ █", "█ █", "███", "  █", "  █"],
    '5': ["███", "█  ", "███", "  █", "███"],
    '6': ["███", "█  ", "███", "█ █", "███"],
    '7': ["███", "  █", "  █", "  █", "  █"],
    '8': ["███", "█ █", "███", "█ █", "███"],
    '9': ["███", "█ █", "███", "  █", "███"],
    ':': ["   ", " █ ", "   ", " █ ", "   "],
    '=': ["   ", "███", "   ", "███", "   "],
    '-': ["   ", "   ", "███", "   ", "   "],
    '_': ["   ", "   ", "   ", "   ", "███"],
    '!': [" █ ", " █ ", " █ ", "   ", " █ "],
    '?': ["██ ", "  █", " █ ", "   ", " █ "],
    '.': ["   ", "   ", "   ", "   ", " █ "],
    ',': ["   ", "   ", "   ", " █ ", "█  "],
    '#': ["█ █", "███", "█ █", "███", "█ █"],
    '/': ["  █", "  █", " █ ", "█  ", "█  "],
    '\\': ["█  ", "█  ", " █ ", "  █", "  █"],
    ' ': ["   ", "   ", "   ", "   ", "   "]
}

BITMAP_FONT_5X7 = FONT_5X7


@unique
class TextAnimationMode(str, Enum):
    KINETIC_MATRIX       = "KINETIC_MATRIX"       # Matrix code rain resolving into text
    SINE_WAVE_FLOAT      = "SINE_WAVE_FLOAT"      # Undulating sine wave text
    GLITCH_CYBERPUNK     = "GLITCH_CYBERPUNK"     # Slicing neon cyber text with random jitter
    TYPEWRITER_DECRYPT   = "TYPEWRITER_DECRYPT"   # Encrypted hex noise resolving left to right
    VOXEL_3D_EXTRUDE     = "VOXEL_3D_EXTRUDE"     # 3D block extruded voxel letters


@dataclass
class TextAnimationConfig:
    """Configuration parameters for the text animation compositor."""
    text: str = "KRYSTAL STACK"
    mode: TextAnimationMode = TextAnimationMode.SINE_WAVE_FLOAT
    x_offset: int = 4
    y_offset: int = 4
    speed: float = 1.0
    scale: int = 1                     # 1 = standard 5x5, 2 = 10x10 block
    wave_frequency: float = 0.4
    wave_amplitude: float = 3.0
    primary_glyph: str = "█"
    secondary_glyph: str = "▓"
    background_glyph: str = " "
    vital_max_hp: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "mode": self.mode.value,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "speed": self.speed,
            "scale": self.scale,
            "wave_frequency": self.wave_frequency,
            "wave_amplitude": self.wave_amplitude,
            "primary_glyph": self.primary_glyph,
            "secondary_glyph": self.secondary_glyph,
            "vital_max_hp": self.vital_max_hp
        }


# ==============================================================================
# 2. TEXT RASTERIZER & ANIMATION COMPOSITOR
# ==============================================================================

class TextAnimationEngine:
    """
    Renders procedural animated text and composites typography onto ASCII canvas buffers.
    """

    def __init__(self, default_config: Optional[TextAnimationConfig] = None):
        self.config = default_config or TextAnimationConfig()

    @classmethod
    def rasterize_glyph(cls, char: str) -> List[str]:
        """Returns the 5-row pattern for a character."""
        c = char.upper()
        return FONT_5X7.get(c, FONT_5X7['?'])

    def render_text_matrix(
        self,
        text: str,
        t: float,
        width: int = 80,
        height: int = 24,
        config: Optional[TextAnimationConfig] = None
    ) -> List[str]:
        """
        Renders standalone animated text frame at specified width/height.
        """
        cfg = config or self.config
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6"

        grid = [[cfg.background_glyph for _ in range(width)] for _ in range(height)]
        mode = cfg.mode
        time_scaled = t * cfg.speed

        if mode == TextAnimationMode.SINE_WAVE_FLOAT:
            self._render_sine_wave(grid, text, time_scaled, width, height, cfg)
        elif mode == TextAnimationMode.KINETIC_MATRIX:
            self._render_kinetic_matrix(grid, text, time_scaled, width, height, cfg)
        elif mode == TextAnimationMode.GLITCH_CYBERPUNK:
            self._render_glitch_cyberpunk(grid, text, time_scaled, width, height, cfg)
        elif mode == TextAnimationMode.TYPEWRITER_DECRYPT:
            self._render_typewriter_decrypt(grid, text, time_scaled, width, height, cfg)
        elif mode == TextAnimationMode.VOXEL_3D_EXTRUDE:
            self._render_voxel_3d_extrude(grid, text, time_scaled, width, height, cfg)
        else:
            self._render_sine_wave(grid, text, time_scaled, width, height, cfg)

        return ["".join(row) for row in grid]

    # ─── ANIMATION MODE IMPLEMENTATIONS ──────────────────────────────────────

    def _render_sine_wave(
        self,
        grid: List[List[str]],
        text: str,
        t: float,
        width: int,
        height: int,
        cfg: TextAnimationConfig
    ) -> None:
        """Renders text undulating in sinusoidal wave with 3D drop-shadow."""
        chars = text.upper()
        glyph_width = 4  # 3 chars + 1 space
        total_text_width = len(chars) * glyph_width
        start_x = max(2, (width - total_text_width) // 2)

        for char_idx, char in enumerate(chars):
            cur_x = start_x + (char_idx * glyph_width)
            # Sine wave vertical displacement per character
            y_disp = math.sin(t * 2.5 + (char_idx * cfg.wave_frequency)) * cfg.wave_amplitude
            base_y = int((height // 2 - 2) + y_disp)

            pattern = self.rasterize_glyph(char)

            # Draw Shadow First (offset +1, +1)
            for r_idx, row in enumerate(pattern):
                py = base_y + r_idx + 1
                for c_idx, pixel in enumerate(row):
                    px = cur_x + c_idx + 1
                    if 0 <= py < height and 0 <= px < width:
                        if pixel != ' ' and grid[py][px] == cfg.background_glyph:
                            grid[py][px] = "░"

            # Draw Main Glyph
            for r_idx, row in enumerate(pattern):
                py = base_y + r_idx
                for c_idx, pixel in enumerate(row):
                    px = cur_x + c_idx
                    if 0 <= py < height and 0 <= px < width:
                        if pixel != ' ':
                            grid[py][px] = cfg.primary_glyph

    def _render_kinetic_matrix(
        self,
        grid: List[List[str]],
        text: str,
        t: float,
        width: int,
        height: int,
        cfg: TextAnimationConfig
    ) -> None:
        """Matrix rain stream that condenses into target text."""
        matrix_glyphs = "01アイウエオカキクケコサシスセソタチツテト0123456789ABCDEF"
        chars = text.upper()
        glyph_width = 4
        start_x = max(2, (width - len(chars) * glyph_width) // 2)
        base_y = height // 2 - 2

        # 1. Fill background with dynamic matrix rain
        for x in range(0, width, 2):
            stream_speed = 1.0 + (x % 5) * 0.4
            head_y = int((t * 8.0 * stream_speed + (x * 7)) % (height + 8)) - 4
            for trail in range(6):
                py = head_y - trail
                if 0 <= py < height:
                    g = matrix_glyphs[(py * 7 + x + int(t * 5)) % len(matrix_glyphs)]
                    grid[py][x] = g if trail == 0 else ("|" if trail < 3 else ":")

        # 2. Condense target text with brightness mask
        for char_idx, char in enumerate(chars):
            cur_x = start_x + (char_idx * glyph_width)
            pattern = self.rasterize_glyph(char)
            # Pulse opacity based on time
            freeze_factor = min(1.0, max(0.0, (math.sin(t * 1.5) + 1.0) * 0.5 + 0.2))

            for r_idx, row in enumerate(pattern):
                py = base_y + r_idx
                for c_idx, pixel in enumerate(row):
                    px = cur_x + c_idx
                    if 0 <= py < height and 0 <= px < width:
                        if pixel != ' ':
                            if random.random() < freeze_factor:
                                grid[py][px] = cfg.primary_glyph
                            else:
                                grid[py][px] = random.choice(matrix_glyphs)

    def _render_glitch_cyberpunk(
        self,
        grid: List[List[str]],
        text: str,
        t: float,
        width: int,
        height: int,
        cfg: TextAnimationConfig
    ) -> None:
        """High-energy glitch text with horizontal slice shifts and scanline noise."""
        chars = text.upper()
        glyph_width = 4
        start_x = max(4, (width - len(chars) * glyph_width) // 2)
        base_y = height // 2 - 2

        # Glitch triggers periodically
        is_glitch = (int(t * 8) % 7 == 0) or (int(t * 14) % 11 == 0)
        slice_y = int((t * 11) % height)
        slice_offset = random.randint(-4, 4) if is_glitch else 0

        # Scanlines background
        for y in range(height):
            if y % 2 == 0:
                for x in range(0, width, 4):
                    grid[y][x] = "-"

        for char_idx, char in enumerate(chars):
            cur_x = start_x + (char_idx * glyph_width)
            pattern = self.rasterize_glyph(char)

            for r_idx, row in enumerate(pattern):
                py = base_y + r_idx
                # Apply slice displacement
                px_shift = slice_offset if (is_glitch and abs(py - slice_y) <= 1) else 0

                for c_idx, pixel in enumerate(row):
                    px = cur_x + c_idx + px_shift
                    if 0 <= py < height and 0 <= px < width:
                        if pixel != ' ':
                            if is_glitch and random.random() < 0.25:
                                grid[py][px] = random.choice(["#", "/", "\\", "%", "X"])
                            else:
                                grid[py][px] = cfg.primary_glyph

    def _render_typewriter_decrypt(
        self,
        grid: List[List[str]],
        text: str,
        t: float,
        width: int,
        height: int,
        cfg: TextAnimationConfig
    ) -> None:
        """Cryptographic typewriter resolving hex scrambles from left to right."""
        hex_chars = "0123456789ABCDEF"
        chars = text.upper()
        glyph_width = 4
        start_x = max(2, (width - len(chars) * glyph_width) // 2)
        base_y = height // 2 - 2

        # Progress cycles through the text every 4 seconds
        cycle_t = t % 4.0
        revealed_chars = int((cycle_t / 3.0) * len(chars))

        for char_idx, char in enumerate(chars):
            cur_x = start_x + (char_idx * glyph_width)

            if char_idx < revealed_chars:
                # Fully decrypted plaintext
                display_pattern = self.rasterize_glyph(char)
                char_glyph = cfg.primary_glyph
            elif char_idx == revealed_chars:
                # Active decrypting character (scrambling)
                rand_char = random.choice(hex_chars)
                display_pattern = self.rasterize_glyph(rand_char)
                char_glyph = cfg.secondary_glyph
            else:
                # Encrypted locked placeholder
                display_pattern = self.rasterize_glyph(".")
                char_glyph = "·"

            for r_idx, row in enumerate(display_pattern):
                py = base_y + r_idx
                for c_idx, pixel in enumerate(row):
                    px = cur_x + c_idx
                    if 0 <= py < height and 0 <= px < width:
                        if pixel != ' ':
                            grid[py][px] = char_glyph

        # Blinking cursor right after the decrypting character
        if revealed_chars <= len(chars):
            cursor_x = start_x + (revealed_chars * glyph_width)
            if int(t * 4) % 2 == 0:
                for r_idx in range(5):
                    py = base_y + r_idx
                    if 0 <= py < height and 0 <= cursor_x < width:
                        grid[py][cursor_x] = "▌"

    def _render_voxel_3d_extrude(
        self,
        grid: List[List[str]],
        text: str,
        t: float,
        width: int,
        height: int,
        cfg: TextAnimationConfig
    ) -> None:
        """Pseudo-3D extruded voxel text with yaw rotation and depth shading."""
        chars = text.upper()
        glyph_width = 4
        start_x = max(4, (width - len(chars) * glyph_width) // 2)
        base_y = height // 2 - 2

        # 3D Depth layers (front to back)
        yaw = math.sin(t * 1.5) * 0.4
        depth_steps = 3

        for d in range(depth_steps, -1, -1):
            offset_x = int(d * yaw * 3.0)
            offset_y = d
            shade_glyph = cfg.primary_glyph if d == 0 else ("▒" if d == 1 else "░")

            for char_idx, char in enumerate(chars):
                cur_x = start_x + (char_idx * glyph_width) + offset_x
                pattern = self.rasterize_glyph(char)

                for r_idx, row in enumerate(pattern):
                    py = base_y + r_idx - offset_y
                    for c_idx, pixel in enumerate(row):
                        px = cur_x + c_idx
                        if 0 <= py < height and 0 <= px < width:
                            if pixel != ' ':
                                # Only overwrite if empty or front layer
                                if d == 0 or grid[py][px] == cfg.background_glyph:
                                    grid[py][px] = shade_glyph

    # ─── NON-DESTRUCTIVE COMPOSITING ─────────────────────────────────────────

    def composite_text_overlay(
        self,
        background_lines: List[str],
        text: str,
        t: float,
        mode: Optional[TextAnimationMode] = None,
        x_pos: Optional[int] = None,
        y_pos: Optional[int] = None
    ) -> List[str]:
        """
        Overlays the animated text directly onto an existing ASCII frame buffer.
        Leaves transparent pixels (' ') intact so the 3D raymarching or procedural
        background shines through.
        """
        assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6"
        if not background_lines:
            return background_lines

        height = len(background_lines)
        width = len(background_lines[0]) if height > 0 else 80

        # Clone background into 2D char grid
        grid = [list(line) for line in background_lines]

        # Temporary overlay config
        cfg = TextAnimationConfig(
            text=text,
            mode=mode or self.config.mode,
            speed=self.config.speed,
            wave_frequency=self.config.wave_frequency,
            wave_amplitude=self.config.wave_amplitude,
            background_glyph=" "
        )

        # Generate animated text frame
        text_frame = self.render_text_matrix(text, t, width=width, height=height, config=cfg)

        # Composite non-space characters
        for y in range(height):
            t_row = text_frame[y] if y < len(text_frame) else ""
            for x in range(min(width, len(t_row))):
                ch = t_row[x]
                if ch != ' ':
                    grid[y][x] = ch

        return ["".join(row) for row in grid]


# Global Singleton Text Animation Engine
GLOBAL_TEXT_ANIMATION_ENGINE = TextAnimationEngine()


if __name__ == "__main__":
    print("=== Krystal Text Animation Engine Diagnostic ===")
    assert VITAL_MAX_HP == 6, "Invariant VITAL_MAX_HP == 6"

    engine = GLOBAL_TEXT_ANIMATION_ENGINE
    sample_text = "KRYSTAL"

    print("\n--- Sine Wave Float Mode (t=0.5s) ---")
    cfg_wave = TextAnimationConfig(text=sample_text, mode=TextAnimationMode.SINE_WAVE_FLOAT)
    lines_wave = engine.render_text_matrix(sample_text, t=0.5, width=48, height=12, config=cfg_wave)
    print("\n".join(lines_wave))

    print("\n--- Voxel 3D Extrude Mode (t=1.0s) ---")
    cfg_3d = TextAnimationConfig(text=sample_text, mode=TextAnimationMode.VOXEL_3D_EXTRUDE)
    lines_3d = engine.render_text_matrix(sample_text, t=1.0, width=48, height=12, config=cfg_3d)
    print("\n".join(lines_3d))

    print("\n✓ Text Animation Engine operational with VITAL_MAX_HP == 6.")
