"""
Krystal-Stack Platform Framework: Mimicry Primitives
====================================================
Fundamental 3D Signed Distance Functions (SDFs) and Constructive Solid
Geometry (CSG) operators for mimicking real-world complex objects.
"""

import math
from typing import Tuple, List, Callable

Vec3 = Tuple[float, float, float]

# ─── Vector Utilities ─────────────────────────────────────────────────────────

def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def v_scale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)

def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def v_len(a: Vec3) -> float:
    return math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def v_norm(a: Vec3) -> Vec3:
    l = v_len(a)
    return (a[0] / l, a[1] / l, a[2] / l) if l > 1e-7 else (0.0, 1.0, 0.0)

def v_rot_y(p: Vec3, theta: float) -> Vec3:
    c, s = math.cos(theta), math.sin(theta)
    return (p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c)

def v_rot_x(p: Vec3, theta: float) -> Vec3:
    c, s = math.cos(theta), math.sin(theta)
    return (p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c)

def v_rot_z(p: Vec3, theta: float) -> Vec3:
    c, s = math.cos(theta), math.sin(theta)
    return (p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2])


# ─── Fundamental 3D SDF Primitives ───────────────────────────────────────────

def sdf_sphere(p: Vec3, r: float = 1.0) -> float:
    """Signed distance to a sphere of radius r at origin."""
    return v_len(p) - r

def sdf_box(p: Vec3, b: Vec3 = (1.0, 1.0, 1.0)) -> float:
    """Signed distance to a box of half-extents b at origin."""
    qx = abs(p[0]) - b[0]
    qy = abs(p[1]) - b[1]
    qz = abs(p[2]) - b[2]
    outside = math.sqrt(max(qx, 0.0)**2 + max(qy, 0.0)**2 + max(qz, 0.0)**2)
    inside = min(max(qx, max(qy, qz)), 0.0)
    return outside + inside

def sdf_round_box(p: Vec3, b: Vec3 = (1.0, 1.0, 1.0), r: float = 0.1) -> float:
    """Box with rounded edges."""
    return sdf_box(p, (b[0] - r, b[1] - r, b[2] - r)) - r

def sdf_cylinder(p: Vec3, r: float = 0.5, h: float = 1.0) -> float:
    """Vertical cylinder along Y axis with radius r and half-height h."""
    dx = math.sqrt(p[0]**2 + p[2]**2) - r
    dy = abs(p[1]) - h
    outside = math.sqrt(max(dx, 0.0)**2 + max(dy, 0.0)**2)
    inside = min(max(dx, dy), 0.0)
    return outside + inside

def sdf_capped_cone(p: Vec3, h: float = 1.0, r1: float = 0.6, r2: float = 0.2) -> float:
    """Capped cone along Y with height h, bottom radius r1, top radius r2."""
    q_x = math.sqrt(p[0]**2 + p[2]**2)
    q_y = p[1]
    k1_x = r2
    k1_y = h
    k2_x = r2 - r1
    k2_y = 2.0 * h
    ca_x = max(0.0, min(k1_x, q_x - ((q_y < 0.0) * r1)))
    ca_y = abs(q_y) - h
    cb_x = q_x - k1_x + k2_x * max(0.0, min(1.0, (q_x * k2_x + (q_y - h) * k2_y) / (k2_x**2 + k2_y**2)))
    cb_y = q_y - h + k2_y * max(0.0, min(1.0, (q_x * k2_x + (q_y - h) * k2_y) / (k2_x**2 + k2_y**2)))
    s = -1.0 if (cb_x < 0.0 and ca_y < 0.0) else 1.0
    return s * math.sqrt(min(ca_x**2 + ca_y**2, cb_x**2 + cb_y**2))

def sdf_torus(p: Vec3, r1: float = 1.0, r2: float = 0.25) -> float:
    """Torus lying in XZ plane with major radius r1 and minor radius r2."""
    qx = math.sqrt(p[0]**2 + p[2]**2) - r1
    qy = p[1]
    return math.sqrt(qx**2 + qy**2) - r2

def sdf_capsule(p: Vec3, a: Vec3, b: Vec3, r: float) -> float:
    """Capsule from point a to point b with radius r."""
    pa = v_sub(p, a)
    ba = v_sub(b, a)
    ba_len_sq = v_dot(ba, ba)
    h = max(0.0, min(1.0, v_dot(pa, ba) / max(1e-7, ba_len_sq)))
    return v_len(v_sub(pa, v_scale(ba, h))) - r

def sdf_hex_prism(p: Vec3, h: float = 0.8, r: float = 0.8) -> float:
    """Hexagonal prism along Y axis with radius r and half-height h."""
    k = (0.86602540378, -0.5, 0.57735026919) # (sqrt(3)/2, -0.5, tan(pi/6))
    px = abs(p[0])
    pz = abs(p[2])
    d_dot = 2.0 * min(k[0] * px + k[1] * pz, 0.0)
    px -= d_dot * k[0]
    pz -= d_dot * k[1]
    dx = px - min(max(px, -r), r)
    dz = pz - r
    outside_xz = math.sqrt(dx**2 + max(dz, 0.0)**2)
    dy = abs(p[1]) - h
    outside = math.sqrt(max(outside_xz, 0.0)**2 + max(dy, 0.0)**2)
    inside = min(max(outside_xz, dy), 0.0)
    return outside + inside

def sdf_octahedron(p: Vec3, s: float = 1.0) -> float:
    """Exact octahedron centered at origin."""
    px, py, pz = abs(p[0]), abs(p[1]), abs(p[2])
    m = px + py + pz - s
    q = (0.0, 0.0, 0.0)
    if 3.0 * px < m:
        q = (px, py, pz)
    elif 3.0 * py < m:
        q = (py, pz, px)
    elif 3.0 * pz < m:
        q = (pz, px, py)
    else:
        return m * 0.57735027
    k = max(0.0, min(s, (q[2] - q[1] + s) * 0.5))
    return v_len((q[0], q[1] - s + k, q[2] - k))


# ─── Blender CSG Smooth Operators (Smooth Minimum / Maximum) ────────────────

def smin(a: float, b: float, k: float = 0.2) -> float:
    """
    Polynomial smooth minimum (Blender geometry nodes / Quilez smin).
    Smoothly unions two shapes with radius k.
    """
    h = max(k - abs(a - b), 0.0) / max(1e-7, k)
    return min(a, b) - h * h * k * 0.25

def smax(a: float, b: float, k: float = 0.2) -> float:
    """Polynomial smooth maximum (smooth intersection)."""
    h = max(k - abs(a - b), 0.0) / max(1e-7, k)
    return max(a, b) + h * h * k * 0.25

def smooth_difference(a: float, b: float, k: float = 0.2) -> float:
    """Smooth subtraction of shape b from shape a."""
    return smax(a, -b, k)
