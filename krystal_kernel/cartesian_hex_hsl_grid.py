#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: 3D CARTESIAN GRID COEFFICIENTS & HEXADECIMAL HSL SPECTRUM
==============================================================================
Module: krystal_kernel/cartesian_hex_hsl_grid.py
Milestone: V-CARTESIAN-HEX-HSL-01
Constitutional Invariant: VITAL_MAX_HP == 6
Normative Reference:
  - artifacts/benchmarks/H07_SILICON_TELEMETRY_AND_OPENVINO_LEXICON_SPEC.md
  - legacy_docs/HEXADECIMAL_SYSTEM_WITH_ASCII_COMPOSITION.md
  - krystal_kernel/native_procedural_renderer.py (PBR Harmonic Color Tensors)

WHAT THIS MODULE PROVIDES:
  A rigorous mathematical framework defining spatial transform coefficients
  mapping 3D Cartesian coordinates (x, y, z) into a continuous and quantized
  Hexadecimal HSL (Hue, Saturation, Lightness) color spectrum:

  1. Parametric 3D Grid Spatial Coefficients (GridCoefficients3D):
     - Hue Spatial Gradients:
         * alpha_x, alpha_y, alpha_z: Directional Cartesian linear slopes.
         * omega_rot: Azimuthal vortex coefficient (atan2(z, x)).
         * lambda_wave: Radial spatial wavelength for harmonic ripples.
         * kappa_harm: Radial ripple harmonic amplitude.
         * phi_phase: Golden ratio base phase offset (Phi = 1.6180339887).
     - Saturation Spatial Envelope:
         * beta_base: Core saturation magnitude.
         * beta_radial_decay: Radial desaturation gradient.
         * beta_min / beta_max: Boundary saturation clamps.
     - Lightness / Luminance Plasticity Envelope:
         * gamma_base: Baseline luminance midpoint.
         * gamma_y_gradient: Vertical altitude lighting gradient.
         * gamma_fresnel: Depth plasticity and view-distance modulation.
         * gamma_min / gamma_max: Luminance guardrails.

  2. Hexadecimal Spectrum Quantization & Lattice:
     - 6-Sector Constitutional Partition (VITAL_MAX_HP == 6):
         * Sector 0 (0x10): RED_EMBER      [0° .. 60°)
         * Sector 1 (0x20): AMBER_GOLD     [60° .. 120°)
         * Sector 2 (0x30): EMERALD_BIO    [120° .. 180°)
         * Sector 3 (0x40): CYAN_CRYSTAL   [180° .. 240°)
         * Sector 4 (0x50): SAPPHIRE_VOID  [240° .. 300°)
         * Sector 5 (0x60): AMETHYST_AETHER[300° .. 360°)
     - 32-bit Packed Hexadecimal HSLA Descriptor: 0xHHSSLLAA.
     - 24-bit sRGB Hexadecimal String Representation: "#RRGGBB".
     - 96-byte MultiMesh GPU Buffer Packaging:
         [4x4 Transform Matrix (16 floats)] + [RGBA (4 floats)] + [Custom Data (4 floats)]
     - 128-byte Zero-Copy Contract Framing.

System Invariant: VITAL_MAX_HP == 6.
Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

from __future__ import annotations

import os
import sys
import math
import struct
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Final

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.constants import (
    VITAL_MAX_HP,
    FRAME_SIZE_BYTES,
    ContractViolation,
)
from krystal_kernel.zero_copy_contracts import (
    ZeroCopyContractFrame,
    canonicalize_float32,
)

assert VITAL_MAX_HP == 6, f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}"

GOLDEN_RATIO_PHI: Final[float] = 1.618033988749895


@dataclass(frozen=True)
class GridCoefficients3D:
    """
    Formal coefficient pack parameterizing the 3D Cartesian -> HSL mapping manifold:
      H = f(x, y, z; alpha, omega, lambda, kappa, phi)
      S = g(x, y, z; beta)
      L = h(x, y, z; gamma)
    """
    # 1. Hue Directional & Harmonic Coefficients
    alpha_x: float = 0.035           # Linear X gradient (cycles/meter)
    alpha_y: float = 0.050           # Linear Y gradient (cycles/meter)
    alpha_z: float = 0.025           # Linear Z gradient (cycles/meter)
    omega_rot: float = 0.750         # Azimuthal rotational vortex weight
    lambda_wave: float = 12.0        # Radial ripple wavelength (meters)
    kappa_harm: float = 0.125        # Radial harmonic modulation depth
    phi_phase: float = GOLDEN_RATIO_PHI  # Golden ratio phase anchor

    # 2. Saturation Envelope Coefficients
    beta_base: float = 0.85          # Central baseline saturation
    beta_radial_decay: float = 0.015 # Radial desaturation slope
    beta_min: float = 0.20           # Saturation clamp floor
    beta_max: float = 1.00           # Saturation clamp ceiling

    # 3. Lightness / Luminance Plasticity Coefficients
    gamma_base: float = 0.50         # Midpoint baseline luminance
    gamma_y_gradient: float = 0.035  # Elevation / altitude lighting slope
    gamma_fresnel: float = 0.120     # Depth plasticity modulation
    gamma_min: float = 0.08          # Luminance floor (prevents absolute black crush)
    gamma_max: float = 0.92          # Luminance ceiling (prevents blown-out clipping)

    # 4. Constitutional & Quantization Parameters
    vital_sectors_count: int = VITAL_MAX_HP  # Fixed to 6 sectors
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP or self.vital_sectors_count != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional invariant breach in GridCoefficients3D: {self.vital_max_hp} != {VITAL_MAX_HP}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CartesianGridPoint:
    """
    Comprehensive representation of a single evaluated 3D point in the
    Cartesian-to-Hexadecimal HSL manifold.
    """
    index: int
    x: float
    y: float
    z: float
    radius: float
    azimuth_rad: float
    elevation_rad: float
    hue: float                       # [0.0, 1.0)
    saturation: float                # [0.0, 1.0]
    lightness: float                 # [0.0, 1.0]
    rgb: Tuple[float, float, float]  # (r, g, b) normalized in [0.0, 1.0]
    hex_color: str                   # "#RRGGBB"
    hex_hsla_u32: int                # 0xHHSSLLAA
    hex_sector_code: int             # 0x10 .. 0x60
    sector_name: str                 # RED_EMBER .. AMETHYST_AETHER
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional invariant breach in CartesianGridPoint: {self.vital_max_hp} != {VITAL_MAX_HP}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CartesianHexHslGrid:
    """
    Engine evaluating 3D Cartesian coordinates against parametric grid coefficients
    and projecting them into the hexadecimal HSL spectrum.
    """

    SECTOR_NAMES: Final[Tuple[str, ...]] = (
        "RED_EMBER",       # Sector 0 (0° - 60°)
        "AMBER_GOLD",      # Sector 1 (60° - 120°)
        "EMERALD_BIO",     # Sector 2 (120° - 180°)
        "CYAN_CRYSTAL",    # Sector 3 (180° - 240°)
        "SAPPHIRE_VOID",   # Sector 4 (240° - 300°)
        "AMETHYST_AETHER", # Sector 5 (300° - 360°)
    )

    SECTOR_HEX_CODES: Final[Tuple[int, ...]] = (
        0x10, 0x20, 0x30, 0x40, 0x50, 0x60
    )

    def __init__(
        self,
        coefficients: Optional[GridCoefficients3D] = None,
        vital_max_hp: int = VITAL_MAX_HP
    ) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional invariant breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp = vital_max_hp
        self.coeff = coefficients or GridCoefficients3D()

    @staticmethod
    def hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
        """Converts normalized HSL (h in [0, 1), s in [0, 1], l in [0, 1]) to sRGB floats."""
        c = (1.0 - abs(2.0 * l - 1.0)) * s
        h_prime = (h * 6.0) % 6.0
        x = c * (1.0 - abs((h_prime % 2.0) - 1.0))
        m = l - (c / 2.0)

        sector = int(h_prime)
        if sector == 0:
            r_temp, g_temp, b_temp = c, x, 0.0
        elif sector == 1:
            r_temp, g_temp, b_temp = x, c, 0.0
        elif sector == 2:
            r_temp, g_temp, b_temp = 0.0, c, x
        elif sector == 3:
            r_temp, g_temp, b_temp = 0.0, x, c
        elif sector == 4:
            r_temp, g_temp, b_temp = x, 0.0, c
        else:
            r_temp, g_temp, b_temp = c, 0.0, x

        return (
            max(0.0, min(1.0, r_temp + m)),
            max(0.0, min(1.0, g_temp + m)),
            max(0.0, min(1.0, b_temp + m))
        )

    def evaluate_point(self, x: float, y: float, z: float, index: int = 0) -> CartesianGridPoint:
        """
        Evaluates a single Cartesian point (x, y, z) into the full HSL and hexadecimal manifold.
        """
        c = self.coeff

        # 1. Polar / Spherical Decomposition
        r_xz = math.sqrt(x * x + z * z)
        radius = math.sqrt(x * x + y * y + z * z)
        azimuth = math.atan2(z, x) if r_xz > 1e-6 else 0.0
        elevation = math.asin(max(-1.0, min(1.0, y / max(1e-6, radius))))

        # 2. Continuous Hue Evaluation H(x, y, z)
        azimuth_term = (c.omega_rot * (azimuth / (2.0 * math.pi)))
        harmonic_term = c.kappa_harm * math.sin((2.0 * math.pi * r_xz) / max(0.1, c.lambda_wave))
        linear_term = (c.alpha_x * x) + (c.alpha_y * y) + (c.alpha_z * z)
        raw_hue = c.phi_phase + linear_term + azimuth_term + harmonic_term
        hue = raw_hue % 1.0
        if hue < 0.0:
            hue += 1.0

        # 3. Saturation Evaluation S(x, y, z)
        raw_sat = c.beta_base - (c.beta_radial_decay * r_xz)
        saturation = max(c.beta_min, min(c.beta_max, raw_sat))

        # 4. Lightness Evaluation L(x, y, z)
        y_term = c.gamma_y_gradient * y
        fresnel_term = c.gamma_fresnel * (1.0 - math.exp(-0.1 * r_xz))
        raw_lightness = c.gamma_base + y_term + fresnel_term
        lightness = max(c.gamma_min, min(c.gamma_max, raw_lightness))

        # 5. RGB & Hex Color Conversion
        rgb = self.hsl_to_rgb(hue, saturation, lightness)
        r_255 = max(0, min(255, int(round(rgb[0] * 255.0))))
        g_255 = max(0, min(255, int(round(rgb[1] * 255.0))))
        b_255 = max(0, min(255, int(round(rgb[2] * 255.0))))
        hex_color = f"#{r_255:02X}{g_255:02X}{b_255:02X}"

        # 6. Packed 32-bit HSLA integer: 0xHHSSLLAA
        h_255 = max(0, min(255, int(round(hue * 255.0))))
        s_255 = max(0, min(255, int(round(saturation * 255.0))))
        l_255 = max(0, min(255, int(round(lightness * 255.0))))
        a_255 = 0xFF
        hex_hsla_u32 = (h_255 << 24) | (s_255 << 16) | (l_255 << 8) | a_255

        # 7. Constitutional 6-Sector Partition
        sector_idx = int(math.floor(hue * c.vital_sectors_count)) % c.vital_sectors_count
        hex_sector_code = self.SECTOR_HEX_CODES[sector_idx]
        sector_name = self.SECTOR_NAMES[sector_idx]

        return CartesianGridPoint(
            index=index,
            x=round(x, 4),
            y=round(y, 4),
            z=round(z, 4),
            radius=round(radius, 4),
            azimuth_rad=round(azimuth, 4),
            elevation_rad=round(elevation, 4),
            hue=round(hue, 5),
            saturation=round(saturation, 4),
            lightness=round(lightness, 4),
            rgb=(round(rgb[0], 4), round(rgb[1], 4), round(rgb[2], 4)),
            hex_color=hex_color,
            hex_hsla_u32=hex_hsla_u32,
            hex_sector_code=hex_sector_code,
            sector_name=sector_name,
            vital_max_hp=self.vital_max_hp
        )

    def generate_cubic_lattice(
        self,
        dim_x: int = 4,
        dim_y: int = 4,
        dim_z: int = 4,
        step: float = 2.0,
        center_origin: bool = True
    ) -> List[CartesianGridPoint]:
        """
        Generates a 3D regular Cartesian cubic grid of evaluated points.
        """
        points: List[CartesianGridPoint] = []
        offset_x = ((dim_x - 1) * step * 0.5) if center_origin else 0.0
        offset_y = ((dim_y - 1) * step * 0.5) if center_origin else 0.0
        offset_z = ((dim_z - 1) * step * 0.5) if center_origin else 0.0

        idx = 0
        for ix in range(dim_x):
            for iy in range(dim_y):
                for iz in range(dim_z):
                    px = (ix * step) - offset_x
                    py = (iy * step) - offset_y
                    pz = (iz * step) - offset_z
                    points.append(self.evaluate_point(px, py, pz, index=idx))
                    idx += 1

        return points

    def generate_fibonacci_phyllotaxis_lattice(
        self,
        point_count: int = 36,
        radius_scale: float = 8.0,
        height_span: float = 6.0
    ) -> List[CartesianGridPoint]:
        """
        Generates a 3D cylindrical Fibonacci spiral grid conserving the golden ratio:
          theta_k = k * 2*pi * (1 - 1/Phi)
          r_k = radius_scale * sqrt(k / N)
          y_k = (k / N - 0.5) * height_span
        """
        points: List[CartesianGridPoint] = []
        golden_angle = 2.0 * math.pi * (1.0 - (1.0 / GOLDEN_RATIO_PHI))

        for k in range(point_count):
            fraction = float(k) / max(1.0, float(point_count - 1))
            theta = k * golden_angle
            r = radius_scale * math.sqrt(fraction)
            px = r * math.cos(theta)
            pz = r * math.sin(theta)
            py = (fraction - 0.5) * height_span

            points.append(self.evaluate_point(px, py, pz, index=k))

        return points

    def generate_hex_axial_lattice(
        self,
        radius_steps: int = 2,
        hex_radius: float = 2.0
    ) -> List[CartesianGridPoint]:
        """
        Generates a hexagonal axial grid mapped to 3D Cartesian coordinates.
        q, r in [-radius_steps .. radius_steps] with |q + r| <= radius_steps.
        """
        points: List[CartesianGridPoint] = []
        idx = 0

        for q in range(-radius_steps, radius_steps + 1):
            r1 = max(-radius_steps, -q - radius_steps)
            r2 = min(radius_steps, -q + radius_steps)
            for r in range(r1, r2 + 1):
                px = hex_radius * math.sqrt(3.0) * (q + (r / 2.0))
                pz = hex_radius * 1.5 * r
                py = 0.4 * math.sin(q * 1.2) * math.cos(r * 1.2)
                points.append(self.evaluate_point(px, py, pz, index=idx))
                idx += 1

        return points

    def calculate_sector_distribution(
        self,
        points: List[CartesianGridPoint]
    ) -> Dict[str, int]:
        """
        Computes the histogram distribution across the 6 constitutional sectors.
        """
        dist: Dict[str, int] = {name: 0 for name in self.SECTOR_NAMES}
        for pt in points:
            dist[pt.sector_name] = dist.get(pt.sector_name, 0) + 1
        return dist

    def pack_to_multimesh_96b(self, point: CartesianGridPoint, scale: float = 1.0) -> bytes:
        """
        Packs a point into the native 96-byte MultiMesh instance layout:
        - 16 floats (4x4 Transform row-major)
        - 4 floats (Color RGBA)
        - 4 floats (Custom data: [hue, saturation, lightness, sector_code])
        """
        # Transform matrix (translation + uniform scale)
        m00, m01, m02, m03 = scale, 0.0, 0.0, point.x
        m10, m11, m12, m13 = 0.0, scale, 0.0, point.y
        m20, m21, m22, m23 = 0.0, 0.0, scale, point.z
        m30, m31, m32, m33 = 0.0, 0.0, 0.0, 1.0

        r, g, b = point.rgb
        a = 1.0

        custom0 = point.hue
        custom1 = point.saturation
        custom2 = point.lightness
        custom3 = float(point.hex_sector_code)

        return struct.pack(
            "<24f",
            m00, m01, m02, m03,
            m10, m11, m12, m13,
            m20, m21, m22, m23,
            m30, m31, m32, m33,
            r, g, b, a,
            custom0, custom1, custom2, custom3
        )

    def create_contract_frame(
        self,
        points: List[CartesianGridPoint],
        sequence: int = 1
    ) -> ZeroCopyContractFrame:
        """
        Synthesizes a 128-byte cache-line aligned ZeroCopyContractFrame
        encoding the batch centroid, primary hue and sector code.
        """
        assert self.vital_max_hp == 6, "Invariant VITAL_MAX_HP == 6 must hold"
        count = len(points)
        if count == 0:
            avg_x, avg_y, avg_z, avg_h = 0.0, 0.0, 0.0, 0.0
        else:
            avg_x = sum(p.x for p in points) / count
            avg_y = sum(p.y for p in points) / count
            avg_z = sum(p.z for p in points) / count
            avg_h = sum(p.hue for p in points) / count

        hash_payload = f"GRID-HEX-HSL:{count}:{avg_x:.3f}:{avg_y:.3f}:{avg_z:.3f}:{avg_h:.3f}:{VITAL_MAX_HP}"
        digest = hashlib.sha256(hash_payload.encode("utf-8")).digest()

        return ZeroCopyContractFrame(
            vital_hp=VITAL_MAX_HP,
            contract_type=0x04,  # COMPUTATIONAL_GEOMETRY
            sequence=sequence,
            predecessor_hash=digest,
            gpu_render_ms=canonicalize_float32(avg_h),
            drs_scale_factor=canonicalize_float32(float(count))
        )


# Global Singleton instance
GLOBAL_CARTESIAN_HEX_HSL_GRID: Final[CartesianHexHslGrid] = CartesianHexHslGrid()
