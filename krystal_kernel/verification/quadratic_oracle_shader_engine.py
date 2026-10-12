#!/usr/bin/env python3
"""
KRYSTAL-STACK // QUADRATIC ORACLE SHADER ENGINE
=============================================================================
Module: krystal_kernel/verification/quadratic_oracle_shader_engine.py
Role:
Analytical quadratic oracle engine acting as the CPU/GPU shadow verifier.
Solves quadratic manifold equations:
  A * t^2 + B * t + C = 0
  Delta = B^2 - 4 * A * C

Verifies:
  1. Real root solutions (Delta >= 0): State lies on feasible manifold -> VERIFIED_SAFE.
  2. Complex root solutions (Delta < 0): State lies outside feasible manifold -> COMPLEX_HAZARD_VETO.
  3. Parabolic apex projection: t_apex = -B / (2A) for minimal distance estimation.
  4. Platform Invariant: VITAL_MAX_HP == 6 strictly enforced.
=============================================================================
"""

import os
import sys
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP


@dataclass
class QuadraticManifoldCoefficients:
    """Quadratic 3D manifold parameters: a * p^2 + b * p + c = 0."""
    a: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    b: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    c: float = -1.0


@dataclass
class OracleShaderVerificationResult:
    """Verification outcome of the analytical quadratic solver."""
    is_verified: bool
    discriminant: float
    roots: Tuple[Optional[float], Optional[float]]
    apex_t: float
    analytical_normal: Tuple[float, float, float]
    oracle_verdict: str  # "VERIFIED_REAL_ROOT", "COMPLEX_HAZARD_VETO", "DEGENERATE_LINEAR"
    vital_hp: int = VITAL_MAX_HP
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_verified": self.is_verified,
            "discriminant": round(self.discriminant, 6),
            "roots": [round(r, 6) if r is not None else None for r in self.roots],
            "apex_t": round(self.apex_t, 6),
            "analytical_normal": [round(n, 6) for n in self.analytical_normal],
            "oracle_verdict": self.oracle_verdict,
            "vital_hp": self.vital_hp,
            "timestamp": self.timestamp
        }


class QuadraticOracleShaderEngine:
    """
    Evaluates analytical quadratic oracle solutions matching GPU shader logic
    without requiring iterative step-loops or graphics context initialization.
    """
    def __init__(self, vital_hp: int = VITAL_MAX_HP):
        if vital_hp != VITAL_MAX_HP:
            raise ValueError(f"Universal invariant breach: vital_hp must be {VITAL_MAX_HP}")
        self.vital_hp = vital_hp

    def solve_oracle(
        self,
        ray_origin: Tuple[float, float, float],
        ray_dir: Tuple[float, float, float],
        manifold: QuadraticManifoldCoefficients
    ) -> OracleShaderVerificationResult:
        """
        Solves A * t^2 + B * t + C = 0 along ray P(t) = ro + t * rd.
        Matches solve_quadratic_oracle in quadratic_oracle_shader.gdshader.
        """
        ro_x, ro_y, ro_z = ray_origin
        rd_x, rd_y, rd_z = ray_dir
        a_x, a_y, a_z = manifold.a
        b_x, b_y, b_z = manifold.b
        c = manifold.c

        # Normalize ray direction
        rd_len = math.sqrt(rd_x**2 + rd_y**2 + rd_z**2)
        if rd_len > 1e-12:
            rd_x /= rd_len
            rd_y /= rd_len
            rd_z /= rd_len

        # A = sum(a_i * rd_i^2)
        A = a_x * (rd_x ** 2) + a_y * (rd_y ** 2) + a_z * (rd_z ** 2)

        # B = 2.0 * sum(a_i * ro_i * rd_i) + sum(b_i * rd_i)
        B = 2.0 * (a_x * ro_x * rd_x + a_y * ro_y * rd_y + a_z * ro_z * rd_z) + (b_x * rd_x + b_y * rd_y + b_z * rd_z)

        # C = sum(a_i * ro_i^2) + sum(b_i * ro_i) + c
        C = (a_x * (ro_x ** 2) + a_y * (ro_y ** 2) + a_z * (ro_z ** 2)) + (b_x * ro_x + b_y * ro_y + b_z * ro_z) + c

        # Degenerate linear case (A == 0)
        if abs(A) < 1e-8:
            if abs(B) < 1e-8:
                return OracleShaderVerificationResult(
                    is_verified=False,
                    discriminant=-1.0,
                    roots=(None, None),
                    apex_t=0.0,
                    analytical_normal=(0.0, 1.0, 0.0),
                    oracle_verdict="COMPLEX_HAZARD_VETO",
                    vital_hp=self.vital_hp
                )
            t_lin = -C / B
            p_lin = (ro_x + t_lin * rd_x, ro_y + t_lin * rd_y, ro_z + t_lin * rd_z)
            norm = self._calculate_normal(p_lin, manifold)
            return OracleShaderVerificationResult(
                is_verified=True,
                discriminant=0.0,
                roots=(t_lin, t_lin),
                apex_t=t_lin,
                analytical_normal=norm,
                oracle_verdict="DEGENERATE_LINEAR",
                vital_hp=self.vital_hp
            )

        delta = (B ** 2) - 4.0 * A * C
        apex_t = -B / (2.0 * A)

        if delta < 0.0:
            # Complex roots: Outside manifold boundary (Safety Veto)
            p_apex = (ro_x + apex_t * rd_x, ro_y + apex_t * rd_y, ro_z + apex_t * rd_z)
            norm = self._calculate_normal(p_apex, manifold)
            return OracleShaderVerificationResult(
                is_verified=False,
                discriminant=delta,
                roots=(None, None),
                apex_t=apex_t,
                analytical_normal=norm,
                oracle_verdict="COMPLEX_HAZARD_VETO",
                vital_hp=self.vital_hp
            )

        sqrt_delta = math.sqrt(delta)
        t1 = (-B - sqrt_delta) / (2.0 * A)
        t2 = (-B + sqrt_delta) / (2.0 * A)

        # If both roots are in the past (ray aiming away), treat as complex hazard veto
        if t1 < 0.0 and t2 < 0.0:
            p_apex = (ro_x + apex_t * rd_x, ro_y + apex_t * rd_y, ro_z + apex_t * rd_z)
            norm = self._calculate_normal(p_apex, manifold)
            return OracleShaderVerificationResult(
                is_verified=False,
                discriminant=-abs(delta) if delta >= 0.0 else delta,
                roots=(None, None),
                apex_t=apex_t,
                analytical_normal=norm,
                oracle_verdict="COMPLEX_HAZARD_VETO",
                vital_hp=self.vital_hp
            )

        # Nearest forward root
        t_near = t1 if t1 >= 0.0 else t2
        p_hit = (ro_x + t_near * rd_x, ro_y + t_near * rd_y, ro_z + t_near * rd_z)
        norm = self._calculate_normal(p_hit, manifold)

        return OracleShaderVerificationResult(
            is_verified=True,
            discriminant=delta,
            roots=(t1, t2),
            apex_t=apex_t,
            analytical_normal=norm,
            oracle_verdict="VERIFIED_REAL_ROOT",
            vital_hp=self.vital_hp
        )

    def _calculate_normal(
        self,
        point: Tuple[float, float, float],
        manifold: QuadraticManifoldCoefficients
    ) -> Tuple[float, float, float]:
        """Calculates analytical gradient vector: N = normalize(2 * a * p + b)."""
        px, py, pz = point
        ax, ay, az = manifold.a
        bx, by, bz = manifold.b

        nx = 2.0 * ax * px + bx
        ny = 2.0 * ay * py + by
        nz = 2.0 * az * pz + bz

        length = math.sqrt(nx**2 + ny**2 + nz**2)
        if length > 1e-12:
            return (nx / length, ny / length, nz / length)
        return (0.0, 1.0, 0.0)

    def export_uniform_buffer_bytes(self, manifold: QuadraticManifoldCoefficients) -> bytes:
        """
        Serializes uniform parameters into a 64-byte std140 / C-struct memory layout
        compatible with Vulkan UBO and Direct3D 12 Constant Buffers.
        """
        import struct
        # layout(std140):
        # vec3 quad_curvature (12 bytes) + 4 pad = 16
        # vec3 quad_linear    (12 bytes) + 4 pad = 16
        # float quad_constant (4 bytes) + int vital_hp (4 bytes) + 24 pad = 32
        # total = 16 + 16 + 32 = 64 bytes
        return struct.pack(
            "fff4xfff4xfi24x",
            manifold.a[0], manifold.a[1], manifold.a[2],
            manifold.b[0], manifold.b[1], manifold.b[2],
            manifold.c,
            self.vital_hp
        )
