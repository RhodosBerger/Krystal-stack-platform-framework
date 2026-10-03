"""
Krystal-Stack: Fast Math & Vectorized Procedural Acceleration Kernels
=====================================================================
Micro-optimized mathematical functions for SDF evaluation, ray-sphere intersection,
analytical normals, and high-frequency visual entropy calculation.
"""

import math
import array
from typing import Tuple, List

Vec3 = Tuple[float, float, float]

def fast_inv_sqrt(x: float) -> float:
    """Fast inverse square root approximation (1 / sqrt(x))."""
    if x <= 1e-12:
        return 0.0
    return 1.0 / math.sqrt(x)

def fast_norm3(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Normalizes a 3D vector with a single square root."""
    sq = x * x + y * y + z * z
    if sq < 1e-12:
        return (0.0, 1.0, 0.0)
    inv = 1.0 / math.sqrt(sq)
    return (x * inv, y * inv, z * inv)

def compute_spatial_entropy_contiguous(luma_buffer: array.array) -> float:
    """
    Computes spatial visual entropy over a contiguous C-level float array.
    Avoids Python list pointer chasing and memory fragmentation.
    """
    n = len(luma_buffer)
    if n == 0:
        return 0.0
    total = sum(luma_buffer)
    mean = total / n
    # Variance
    var = sum((x - mean) * (x - mean) for x in luma_buffer) / n
    return min(1.0, var * 4.0)

def compute_temporal_entropy_contiguous(curr: array.array, prev: array.array) -> float:
    """
    Computes temporal frame-to-frame delta entropy between contiguous float buffers.
    """
    n = min(len(curr), len(prev))
    if n == 0:
        return 0.3
    diff_sum = sum(abs(c - p) for c, p in zip(curr[:n], prev[:n]))
    return min(1.0, (diff_sum / n) * 8.0)

def batch_sdf_sphere(points: List[Vec3], radius: float = 1.0) -> List[float]:
    """
    Batch evaluation of sphere SDF for an array of 3D query coordinates.
    Pre-allocated flat float list for CPU cache efficiency.
    """
    r = radius
    return [math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - r for p in points]
