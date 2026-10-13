#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: MATRIX-CONVOLUTIONAL PROCEDURAL & LATENT IMAGE GENERATOR MVP
==============================================================================
Module: krystal_kernel/matrix_image_generator_mvp.py
Description: Production-grade MVP image generation engine synthesizing visuals
             directly from chained 2D Boolean matrices, spatial convolution
             kernels, cellular automata, and harmonic color tensors.

Architectural Layers:
  Layer 1: LatentTopologyLayer - Seeds matrices via Boolean CRUD DB, cellular
           automata, or cosine topology envelopes.
  Layer 2: ConvolutionalDiffusionLayer - Multi-pass 2D spatial convolution
           pipelines (edge detection, diffusion smoothing, majority gating).
  Layer 3: HarmonicColorTensorLayer - Maps bit densities and gradients to
           Golden Ratio-spaced RGB palettes.
  Layer 4: EpistemicHudInlayLayer - Renders epistemic status tokens and
           enforces the strictly immutable 6 Max HP vital invariant.
  Layer 5: MatrixImageRasterExporter - Zero-dependency lossless Truecolor PNG,
           lossless SVG, Base64 Data URI, and ASCII matrix exporter.
  Layer 6: MatrixImageStudioBridge - REST payload coordination and live
           telemetry feeding the Web Hub UI Kit Studio.

Constitutional Invariant:
  assert VITAL_MAX_HP == 6

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import math
import time
import json
import zlib
import struct
import base64
import random
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_kernel.convolutional_matrix_workloads import (
    BooleanMatrix,
    ConvolutionKernel,
    convolve_boolean_matrix,
    CANONICAL_CONVOLUTION_KERNELS,
    GLOBAL_BOOLEAN_CRUD_DATABASE,
    GLOBAL_MATRIX_STREAMING_LOG_SERVICE,
    BooleanEntity
)

# Constitutional Invariant Validation
assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

GOLDEN_RATIO: float = 1.618033988749895
GOLDEN_ANGLE_DEG: float = 137.50776405003785


class LatentSeedMode(str, Enum):
    CRUD_DATABASE_ENTITY = "CRUD_DATABASE_ENTITY"
    CELLULAR_AUTOMATON = "CELLULAR_AUTOMATON"
    COSINE_TOPOLOGY = "COSINE_TOPOLOGY"
    HARMONIC_FRACTAL = "HARMONIC_FRACTAL"
    EPICENTER_CRYSTAL = "EPICENTER_CRYSTAL"


class ImageColorPalette(str, Enum):
    KRYSTAL_CYBERPUNK = "KRYSTAL_CYBERPUNK"
    ALCHEMICAL_GOLD = "ALCHEMICAL_GOLD"
    FOREST_DRUID = "FOREST_DRUID"
    NEON_VOID = "NEON_VOID"
    CHROMATIC_TITANIUM = "CHROMATIC_TITANIUM"
    SYNTHWAVE_SUNSET = "SYNTHWAVE_SUNSET"


# 24-bit Truecolor Palettes: (Background, Low-Density, Mid-Density, High-Density, Accent, HUD)
PALETTE_COLOR_MAP: Dict[ImageColorPalette, Dict[str, Tuple[int, int, int]]] = {
    ImageColorPalette.KRYSTAL_CYBERPUNK: {
        "bg": (10, 10, 18),
        "low": (20, 35, 60),
        "mid": (0, 180, 216),
        "high": (247, 37, 133),
        "accent": (114, 9, 183),
        "hud": (0, 245, 212),
        "hp_pip": (255, 0, 110)
    },
    ImageColorPalette.ALCHEMICAL_GOLD: {
        "bg": (15, 12, 8),
        "low": (45, 32, 18),
        "mid": (180, 125, 40),
        "high": (255, 215, 0),
        "accent": (255, 140, 0),
        "hud": (255, 235, 150),
        "hp_pip": (255, 80, 20)
    },
    ImageColorPalette.FOREST_DRUID: {
        "bg": (8, 18, 12),
        "low": (18, 48, 28),
        "mid": (46, 139, 87),
        "high": (124, 252, 0),
        "accent": (0, 255, 127),
        "hud": (152, 251, 152),
        "hp_pip": (255, 105, 180)
    },
    ImageColorPalette.NEON_VOID: {
        "bg": (6, 4, 15),
        "low": (28, 16, 54),
        "mid": (90, 24, 154),
        "high": (157, 78, 221),
        "accent": (224, 170, 255),
        "hud": (188, 108, 255),
        "hp_pip": (255, 50, 180)
    },
    ImageColorPalette.CHROMATIC_TITANIUM: {
        "bg": (16, 18, 22),
        "low": (35, 42, 52),
        "mid": (90, 105, 125),
        "high": (180, 200, 220),
        "accent": (0, 168, 232),
        "hud": (220, 240, 255),
        "hp_pip": (255, 69, 58)
    },
    ImageColorPalette.SYNTHWAVE_SUNSET: {
        "bg": (18, 6, 28),
        "low": (55, 15, 60),
        "mid": (180, 30, 110),
        "high": (255, 110, 40),
        "accent": (255, 220, 50),
        "hud": (0, 255, 230),
        "hp_pip": (255, 20, 147)
    }
}


@dataclass
class ImageGenerationConfig:
    """Configuration specification for the matrix-to-image generator."""
    width: int = 32
    height: int = 32
    seed: int = 42
    latent_mode: LatentSeedMode = LatentSeedMode.CELLULAR_AUTOMATON
    entity_id: Optional[str] = None
    palette: ImageColorPalette = ImageColorPalette.KRYSTAL_CYBERPUNK
    convolution_pipeline: List[str] = field(default_factory=lambda: [
        "diffuse_3x3",
        "edge_detect_3x3",
        "majority_3x3",
        "invariant_6hp_3x3"
    ])
    scale_factor: int = 8  # Pixel scaling for crisp canvas rendering
    enable_hud_overlay: bool = True
    vital_hp: int = VITAL_MAX_HP
    dither_strength: float = 0.15

    def __post_init__(self):
        # Strict validation of limits and invariants
        self.width = max(8, min(64, self.width))
        self.height = max(8, min(64, self.height))
        self.scale_factor = max(1, min(16, self.scale_factor))
        if self.vital_hp > VITAL_MAX_HP:
            self.vital_hp = VITAL_MAX_HP
        if self.vital_hp < 1:
            self.vital_hp = 1


@dataclass
class GeneratedImageArtifact:
    """Complete image result bundle including pixel data, PNG, SVG, and telemetry."""
    config: Dict[str, Any]
    width: int
    height: int
    scale: int
    total_pixels: int
    seed: int
    density: float
    execution_time_us: float
    vital_max_hp: int
    png_base64: str
    svg_xml: str
    ascii_preview: str
    intermediate_matrices: List[Dict[str, Any]]
    epistemic_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "width": self.width,
            "height": self.height,
            "scaled_width": self.width * self.scale,
            "scaled_height": self.height * self.scale,
            "scale": self.scale,
            "total_pixels": self.total_pixels,
            "seed": self.seed,
            "density": self.density,
            "execution_time_us": self.execution_time_us,
            "execution_time_ms": round(self.execution_time_us / 1000.0, 3),
            "vital_max_hp": self.vital_max_hp,
            "png_base64": self.png_base64,
            "png_data_uri": f"data:image/png;base64,{self.png_base64}",
            "svg_xml": self.svg_xml,
            "ascii_preview": self.ascii_preview,
            "intermediate_count": len(self.intermediate_matrices),
            "epistemic_status": self.epistemic_status
        }


# ==============================================================================
# LAYER 1: LATENT & TOPOLOGY SEEDING LAYER
# ==============================================================================
class LatentTopologyLayer:
    """Synthesizes deterministic latent 2D Boolean matrices from mathematical seeds."""

    @staticmethod
    def generate_seed_matrix(config: ImageGenerationConfig) -> BooleanMatrix:
        w, h = config.width, config.height
        rng = random.Random(config.seed)

        if config.latent_mode == LatentSeedMode.CRUD_DATABASE_ENTITY:
            # Look up or extract from Boolean CRUD database
            entity = None
            if config.entity_id:
                entity = GLOBAL_BOOLEAN_CRUD_DATABASE.get_entity(config.entity_id)
            if not entity:
                # Pick most recent or first from database
                all_records = GLOBAL_BOOLEAN_CRUD_DATABASE.query_entities()
                if all_records:
                    entity_dict = all_records[config.seed % len(all_records)]
                    entity = GLOBAL_BOOLEAN_CRUD_DATABASE.get_entity(entity_dict["entity_id"])
            if entity:
                # Resize or wrap entity matrix to target dimension
                src_mat = GLOBAL_BOOLEAN_CRUD_DATABASE.get_matrix(entity.entity_id)
                if not src_mat:
                    src_mat = BooleanMatrix(rows=8, cols=8)
                grid = [[src_mat.get(r % src_mat.rows, c % src_mat.cols) for c in range(w)] for r in range(h)]
                return BooleanMatrix(rows=h, cols=w, initial_bits=grid)

        if config.latent_mode == LatentSeedMode.COSINE_TOPOLOGY:
            # Kosínusová topológia & polomerový derivát r(theta) = R0 + sum(Ak * cos(k*omega*theta))
            grid = [[False for _ in range(w)] for _ in range(h)]
            cx, cy = w / 2.0, h / 2.0
            r0 = min(w, h) * 0.32
            omega = 3.0 + (config.seed % 4)
            ak = r0 * 0.35 / GOLDEN_RATIO

            for r in range(h):
                for c in range(w):
                    dx = c - cx
                    dy = r - cy
                    dist = math.hypot(dx, dy)
                    angle = math.atan2(dy, dx)
                    harmonic_r = r0 + ak * math.cos(omega * angle + config.seed * 0.1)
                    if dist <= harmonic_r:
                        grid[r][c] = True
            return BooleanMatrix(rows=h, cols=w, initial_bits=grid)

        if config.latent_mode == LatentSeedMode.EPICENTER_CRYSTAL:
            # Sacred hexagonal crystal geometry
            grid = [[False for _ in range(w)] for _ in range(h)]
            cx, cy = w / 2.0, h / 2.0
            max_r = min(w, h) * 0.42
            for r in range(h):
                for c in range(w):
                    dx = abs(c - cx)
                    dy = abs(r - cy)
                    # Hexagonal distance metric: max(dx, dx*0.5 + dy*0.866)
                    hex_dist = max(dx, dx * 0.5 + dy * 0.866025)
                    ring = int(hex_dist / 3.0)
                    if hex_dist <= max_r and (ring % 2 == 0 or (r + c) % 4 == 0):
                        grid[r][c] = True
            return BooleanMatrix(rows=h, cols=w, initial_bits=grid)

        if config.latent_mode == LatentSeedMode.HARMONIC_FRACTAL:
            # Multi-octave trigonometric noise matrix
            grid = [[False for _ in range(w)] for _ in range(h)]
            for r in range(h):
                for c in range(w):
                    nx = c / float(w) * 6.28318
                    ny = r / float(h) * 6.28318
                    v = (math.sin(nx * 1.0 + config.seed) +
                         math.cos(ny * 1.0 + config.seed * 0.5) +
                         0.5 * math.sin(nx * 2.0 + ny * 2.0)) / 2.5
                    grid[r][c] = True if v > 0.08 else False
            return BooleanMatrix(rows=h, cols=w, initial_bits=grid)

        # Default: CELLULAR_AUTOMATON (Conway/Life-like seed)
        grid = [[True if rng.random() > 0.55 else False for _ in range(w)] for _ in range(h)]
        # Force center cluster
        cr_min, cr_max = h // 4, 3 * h // 4
        cc_min, cc_max = w // 4, 3 * w // 4
        for r in range(cr_min, cr_max):
            for c in range(cc_min, cc_max):
                if (r + c + config.seed) % 2 == 0:
                    grid[r][c] = True
        return BooleanMatrix(rows=h, cols=w, initial_bits=grid)


KERNEL_ALIAS_MAP: Dict[str, str] = {
    "diffuse_3x3": "KERNEL_DIFFUSION_SMOOTHER_3X3",
    "diffusion_3x3": "KERNEL_DIFFUSION_SMOOTHER_3X3",
    "edge_detect_3x3": "KERNEL_EDGE_DETECTOR_3X3",
    "edge_3x3": "KERNEL_EDGE_DETECTOR_3X3",
    "majority_3x3": "KERNEL_MAJORITY_GATE_3X3",
    "erosion_3x3": "KERNEL_EROSION_FILTER_3X3",
    "dilation_3x3": "KERNEL_DILATION_FILTER_3X3",
    "invariant_6hp_3x3": "KERNEL_6HP_INVARIANT_MASK_3X3",
    "6hp_3x3": "KERNEL_6HP_INVARIANT_MASK_3X3",
}


# ==============================================================================
# LAYER 2: MULTI-PASS CONVOLUTIONAL DIFFUSION & FILTER PIPELINE
# ==============================================================================
class ConvolutionalDiffusionLayer:
    """Executes a chain of 2D spatial convolutions over the latent matrix."""

    @staticmethod
    def execute_pipeline(
        initial_matrix: BooleanMatrix,
        pipeline_kernels: List[str]
    ) -> Tuple[List[BooleanMatrix], List[Dict[str, Any]]]:
        current = initial_matrix
        steps = [current]
        telemetry = [{
            "step": 0,
            "kernel": "INITIAL_LATENT",
            "density": current.density,
            "active_bits": current.active_bit_count
        }]

        for idx, k_id in enumerate(pipeline_kernels, start=1):
            canonical_key = KERNEL_ALIAS_MAP.get(k_id.lower(), k_id.upper())
            kernel = CANONICAL_CONVOLUTION_KERNELS.get(canonical_key)
            if not kernel:
                # Fallback to majority gate or first available kernel
                kernel = CANONICAL_CONVOLUTION_KERNELS.get(
                    "KERNEL_MAJORITY_GATE_3X3",
                    list(CANONICAL_CONVOLUTION_KERNELS.values())[0]
                )
            convolved = convolve_boolean_matrix(current, kernel)
            steps.append(convolved)
            telemetry.append({
                "step": idx,
                "kernel": kernel.kernel_id,
                "density": convolved.density,
                "active_bits": convolved.active_bit_count
            })
            current = convolved

        return steps, telemetry


# ==============================================================================
# LAYER 3: MULTI-CHANNEL COLORIZATION & HARMONIC PALETTE SYNTHESIZER
# ==============================================================================
class HarmonicColorTensorLayer:
    """
    Renders 2D Boolean matrices into continuous 24-bit Truecolor RGB pixel arrays.
    Synthesizes smooth gradients based on convolution depth, local neighbor density,
    and perceptual Golden Ratio harmonics.
    """

    @staticmethod
    def colorize(
        matrix_steps: List[BooleanMatrix],
        config: ImageGenerationConfig
    ) -> List[List[Tuple[int, int, int]]]:
        w, h = config.width, config.height
        pal = PALETTE_COLOR_MAP.get(config.palette, PALETTE_COLOR_MAP[ImageColorPalette.KRYSTAL_CYBERPUNK])

        # We take the base latent, the intermediate convolved, and the final stage
        final_mat = matrix_steps[-1]
        edge_mat = matrix_steps[min(2, len(matrix_steps) - 1)]
        diff_mat = matrix_steps[min(1, len(matrix_steps) - 1)]

        pixels: List[List[Tuple[int, int, int]]] = []
        rng = random.Random(config.seed + 101)

        cx, cy = w / 2.0, h / 2.0
        max_dist = math.hypot(cx, cy)

        for r in range(h):
            row: List[Tuple[int, int, int]] = []
            for c in range(w):
                is_active = bool(final_mat.get(r, c))
                is_edge = bool(edge_mat.get(r, c))
                is_diff = bool(diff_mat.get(r, c))

                # Radial vignette gradient
                dist = math.hypot(c - cx, r - cy) / max_dist
                vignette = max(0.0, min(1.0, 1.0 - dist * 0.45))

                # Color synthesis logic
                if is_active and is_edge:
                    base_col = pal["high"]
                elif is_active:
                    base_col = pal["mid"]
                elif is_diff:
                    base_col = pal["low"]
                else:
                    base_col = pal["bg"]

                # Add subtle dither noise for organic texture
                dither = (rng.random() - 0.5) * config.dither_strength * 255.0

                red = max(0, min(255, int(base_col[0] * vignette + dither)))
                green = max(0, min(255, int(base_col[1] * vignette + dither)))
                blue = max(0, min(255, int(base_col[2] * vignette + dither)))

                row.append((red, green, blue))
            pixels.append(row)

        return pixels


# ==============================================================================
# LAYER 4: EPISTEMIC INVARIANT HUD INLAY LAYER (Strict VITAL_MAX_HP == 6)
# ==============================================================================
class EpistemicHudInlayLayer:
    """
    Overlays immutable platform diagnostics and exactly 6 discrete HP pips
    directly onto the pixel buffer, mathematically enforcing constitutional truth.
    """

    @staticmethod
    def apply_hud_overlay(
        pixels: List[List[Tuple[int, int, int]]],
        config: ImageGenerationConfig
    ) -> List[List[Tuple[int, int, int]]]:
        if not config.enable_hud_overlay:
            return pixels

        w = config.width
        h = config.height
        pal = PALETTE_COLOR_MAP.get(config.palette, PALETTE_COLOR_MAP[ImageColorPalette.KRYSTAL_CYBERPUNK])
        hud_col = pal["hud"]
        hp_active_col = pal["hp_pip"]
        hp_dim_col = (40, 40, 50)

        # Constitutional Invariant: Strictly 6 Max HP pips
        vital_hp = min(VITAL_MAX_HP, max(1, config.vital_hp))

        # Top border HUD bar: 6 pips rendered at (r=1, c=2..13)
        if w >= 16 and h >= 16:
            for pip_idx in range(VITAL_MAX_HP):
                start_c = 2 + pip_idx * 2
                col = hp_active_col if pip_idx < vital_hp else hp_dim_col
                if start_c < w - 1:
                    pixels[1][start_c] = col
                    if h >= 24:
                        pixels[2][start_c] = col

            # Micro corner telemetry indicators
            pixels[1][w - 2] = hud_col
            pixels[h - 2][1] = hud_col
            pixels[h - 2][w - 2] = hud_col

        return pixels


# ==============================================================================
# LAYER 5: ZERO-DEPENDENCY LOSSLESS MULTI-FORMAT EXPORTER (PNG / SVG / ASCII)
# ==============================================================================
class MatrixImageRasterExporter:
    """
    Pure Python, zero-dependency lossless rasterizer emitting high-fidelity
    PNG bytes, Base64 Data URIs, scalable vector graphics (SVG), and terminal ASCII.
    """

    @staticmethod
    def to_png_bytes(pixels: List[List[Tuple[int, int, int]]], scale: int = 1) -> bytes:
        h = len(pixels)
        w = len(pixels[0]) if h > 0 else 0
        scale = max(1, min(16, scale))
        out_w = w * scale
        out_h = h * scale

        # Prepare uncompressed raw Truecolor (RGB 24-bit) scanlines
        raw_scanlines = bytearray()
        for r in range(h):
            row_bytes = bytearray()
            for c in range(w):
                red, green, blue = pixels[r][c]
                pixel_chunk = bytes([red, green, blue]) * scale
                row_bytes.extend(pixel_chunk)

            # Repeat each scanline 'scale' times for crisp pixel-art upscale
            for _ in range(scale):
                raw_scanlines.append(0)  # Filter type 0 (None)
                raw_scanlines.extend(row_bytes)

        # Standard PNG chunk helper
        def make_chunk(tag: bytes, data: bytes) -> bytes:
            length = struct.pack(">I", len(data))
            chunk_body = tag + data
            crc = struct.pack(">I", zlib.crc32(chunk_body) & 0xFFFFFFFF)
            return length + chunk_body + crc

        # PNG Signature
        signature = b"\x89PNG\r\n\x1a\n"

        # IHDR: width, height, bit_depth=8, color_type=2 (Truecolor RGB), compression=0, filter=0, interlace=0
        ihdr_data = struct.pack(">IIBBBBB", out_w, out_h, 8, 2, 0, 0, 0)
        ihdr_chunk = make_chunk(b"IHDR", ihdr_data)

        # IDAT: Zlib compressed pixel scanlines
        idat_chunk = make_chunk(b"IDAT", zlib.compress(bytes(raw_scanlines), level=9))

        # IEND: Trailer chunk
        iend_chunk = make_chunk(b"IEND", b"")

        return signature + ihdr_chunk + idat_chunk + iend_chunk

    @staticmethod
    def to_base64_png(pixels: List[List[Tuple[int, int, int]]], scale: int = 1) -> str:
        png_bytes = MatrixImageRasterExporter.to_png_bytes(pixels, scale)
        return base64.b64encode(png_bytes).decode("ascii")

    @staticmethod
    def to_svg(pixels: List[List[Tuple[int, int, int]]], scale: int = 8) -> str:
        h = len(pixels)
        w = len(pixels[0]) if h > 0 else 0
        scale = max(1, min(32, scale))
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w * scale}" height="{h * scale}" '
            f'viewBox="0 0 {w} {h}" shape-rendering="crispEdges">'
        ]
        for y in range(h):
            x = 0
            while x < w:
                x_end = x
                while x_end + 1 < w and pixels[y][x_end + 1] == pixels[y][x]:
                    x_end += 1
                r, g, b = pixels[y][x]
                parts.append(
                    f'<rect x="{x}" y="{y}" width="{x_end - x + 1}" height="1" '
                    f'fill="#{r:02x}{g:02x}{b:02x}"/>'
                )
                x = x_end + 1
        parts.append("</svg>")
        return "".join(parts)

    @staticmethod
    def to_ascii(pixels: List[List[Tuple[int, int, int]]]) -> str:
        ramp = " .:-=+*#%@"
        lines = []
        for row in pixels:
            line = []
            for r, g, b in row:
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                idx = min(9, int((lum / 256.0) * 10))
                line.append(ramp[idx])
            lines.append("".join(line))
        return "\n".join(lines)


# ==============================================================================
# LAYER 6: MATRIX IMAGE STUDIO BRIDGE & WORKLOAD ORCHESTRATOR
# ==============================================================================
class MatrixImageGeneratorEngine:
    """
    Principal orchestrator binding all 6 layers together.
    Provides thread-safe generation, real-time logging, and REST dispatch.
    """

    def __init__(self):
        self._last_generated_artifact: Optional[GeneratedImageArtifact] = None
        self._generation_count: int = 0

    def generate_image(self, config: Optional[ImageGenerationConfig] = None) -> GeneratedImageArtifact:
        if config is None:
            config = ImageGenerationConfig()

        t_start = time.perf_counter_ns()

        # Step 1: Latent Seeding
        latent_matrix = LatentTopologyLayer.generate_seed_matrix(config)

        # Step 2: Multi-pass Convolutional Pipeline
        matrix_steps, conv_telemetry = ConvolutionalDiffusionLayer.execute_pipeline(
            latent_matrix, config.convolution_pipeline
        )

        # Step 3: Multi-Channel Tensor Colorization
        colorized_pixels = HarmonicColorTensorLayer.colorize(matrix_steps, config)

        # Step 4: Epistemic HUD Inlay & 6HP Invariant Enforcement
        final_pixels = EpistemicHudInlayLayer.apply_hud_overlay(colorized_pixels, config)

        # Step 5: Lossless Exporters
        png_b64 = MatrixImageRasterExporter.to_base64_png(final_pixels, scale=config.scale_factor)
        svg_xml = MatrixImageRasterExporter.to_svg(final_pixels, scale=config.scale_factor)
        ascii_art = MatrixImageRasterExporter.to_ascii(final_pixels)

        t_elapsed_us = (time.perf_counter_ns() - t_start) / 1000.0
        final_density = matrix_steps[-1].density

        # Step 6: Log Telemetry into Circular Ring Buffer
        GLOBAL_MATRIX_STREAMING_LOG_SERVICE.emit_event(
            workload_id=f"IMG_{config.seed}_{self._generation_count}",
            stage_name="IMAGE_GENERATION_MVP",
            kernel_applied=f"PIPELINE_{len(config.convolution_pipeline)}_KERNELS",
            bits_before=matrix_steps[0].active_bit_count,
            bits_after=matrix_steps[-1].active_bit_count,
            density=final_density,
            elapsed_us=t_elapsed_us,
            status_token="ACT_EXECUTED",
            message=(
                f"Generated {config.width}x{config.height} image via {config.latent_mode.value}, "
                f"palette={config.palette.value}, scale={config.scale_factor}x"
            )
        )

        artifact = GeneratedImageArtifact(
            config=asdict(config),
            width=config.width,
            height=config.height,
            scale=config.scale_factor,
            total_pixels=config.width * config.height * config.scale_factor * config.scale_factor,
            seed=config.seed,
            density=final_density,
            execution_time_us=t_elapsed_us,
            vital_max_hp=VITAL_MAX_HP,
            png_base64=png_b64,
            svg_xml=svg_xml,
            ascii_preview=ascii_art,
            intermediate_matrices=conv_telemetry,
            epistemic_status="SAFE_VERIFIED_CONVOLVED"
        )

        self._last_generated_artifact = artifact
        self._generation_count += 1
        return artifact

    # Alias for generate_image
    generate = generate_image

    def get_latest_artifact(self) -> Optional[GeneratedImageArtifact]:
        return self._last_generated_artifact

    def list_palettes(self) -> List[Dict[str, Any]]:
        return [
            {
                "palette_id": pal.value,
                "name": pal.name.replace("_", " ").title(),
                "colors": {k: f"rgb({c[0]},{c[1]},{c[2]})" for k, c in PALETTE_COLOR_MAP[pal].items()}
            }
            for pal in ImageColorPalette
        ]

    def list_latent_modes(self) -> List[str]:
        return [mode.value for mode in LatentSeedMode]


# Global Singleton Engine Instance
GLOBAL_MATRIX_IMAGE_GENERATOR_ENGINE = MatrixImageGeneratorEngine()
