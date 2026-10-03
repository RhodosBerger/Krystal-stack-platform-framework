"""
Krystal-Lang: Bytecode-to-Shape Transpiler
==========================================
Translates topological bytecode instructions into continuous 3D Signed
Distance Fields (SDFs), allowing programs to be visually inspected and
reasoned about by LLMs and human engineers as geometric manifolds.
"""

import math
from typing import Dict, Any, List, Tuple, Optional

from mimicry_engine.primitives import (
    Vec3, v_add, v_sub, v_scale, smin, smax,
    sdf_sphere, sdf_box, sdf_cylinder, sdf_torus, sdf_octahedron
)

class BytecodeShapeTranspiler:
    """
    Transforms topological bytecode and queue graphs into continuous 3D SDFs.
    Code instructions materialize as physical geometric structures in metric space.
    """
    def __init__(self, compilation_result: Dict[str, Any]):
        self.comp = compilation_result
        self.bytecode = self.comp.get("bytecode", [])
        self.queues = []
        self.shapes = []
        self.pipelines = []
        self._decode_bytecode()

    def _decode_bytecode(self):
        for instr in self.bytecode:
            op = instr.get("op")
            if op == "OP_ALLOC_QUEUE":
                self.queues.append({
                    "name": instr["name"],
                    "pos": tuple(instr["spatial_coord"]),
                    "radius": 0.4 + (instr["priority"] * 0.15)
                })
            elif op == "OP_EMIT_SHAPE":
                self.shapes.append({
                    "type": instr["shape_type"],
                    "params": instr["params"],
                    "csg": instr["csg_op"],
                    "smoothness": instr.get("smoothness", 0.2)
                })
            elif op == "OP_QUEUE_DISPATCH":
                self.pipelines.append({
                    "from": instr["from_queue"],
                    "to": instr["to_queue"]
                })

    def evaluate_program_sdf(self, p: Vec3, t: float = 0.0) -> float:
        """
        Evaluates the global signed distance field of the compiled program.
        Negative inside code geometry, positive outside.
        """
        d_total = float('inf')

        # 1. Evaluate Queue Energy Nodes (Spheres with pulsing activity)
        for q in self.queues:
            qx, qy, qz = q["pos"]
            pulse = 0.08 * math.sin(t * 3.0 + qx * 2.0)
            d_q = sdf_sphere((p[0] - qx, p[1] - qy, p[2] - qz), q["radius"] + pulse)
            d_total = smin(d_total, d_q, 0.25)

        # 2. Evaluate Pipeline Data Streams (Streamline tubes connecting queues)
        for pipe in self.pipelines:
            q_from = next((q for q in self.queues if q["name"] == pipe["from"]), None)
            q_to = next((q for q in self.queues if q["name"] == pipe["to"]), None)
            if q_from and q_to:
                # Segment distance from p to line(q_from, q_to)
                p1 = q_from["pos"]
                p2 = q_to["pos"]
                v = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
                v_len_sq = v[0]**2 + v[1]**2 + v[2]**2
                if v_len_sq > 1e-4:
                    w = (p[0] - p1[0], p[1] - p1[1], p[2] - p1[2])
                    c = max(0.0, min(1.0, (w[0]*v[0] + w[1]*v[1] + w[2]*v[2]) / v_len_sq))
                    closest = (p1[0] + v[0]*c, p1[1] + v[1]*c, p1[2] + v[2]*c)
                    d_tube = math.hypot(p[0] - closest[0], math.hypot(p[1] - closest[1], p[2] - closest[2])) - 0.12
                    d_total = smin(d_total, d_tube, 0.15)

        # 3. Evaluate Central Processing Kernel (Algorithmic Manifold)
        for sh in self.shapes:
            st = sh["type"]
            d_kernel = float('inf')
            scale = sh["params"].get("scale", 1.0)

            # Central position with slow analytical rotation
            theta = t * 0.8
            rx = p[0] * math.cos(theta) - p[2] * math.sin(theta)
            rz = p[0] * math.sin(theta) + p[2] * math.cos(theta)
            p_rot = (rx, p[1] - 0.5, rz)

            if st in ("SPHERE", "CORE"):
                d_kernel = sdf_sphere(p_rot, 0.8 * scale)
            elif st in ("BOX", "MEMORY"):
                d_kernel = sdf_box(p_rot, (0.7 * scale, 0.7 * scale, 0.7 * scale))
            elif st in ("GYROID", "TPMS"):
                # Triply Periodic Minimal Surface gyroid
                freq = sh["params"].get("frequency", 2.0)
                thick = sh["params"].get("thickness", 0.12)
                d_bound = sdf_sphere(p_rot, 1.2 * scale)
                val = abs(math.sin(p[0]*freq) * math.cos(p[1]*freq) +
                          math.sin(p[1]*freq) * math.cos(p[2]*freq) +
                          math.sin(p[2]*freq) * math.cos(p[0]*freq)) - thick
                d_kernel = max(d_bound, val * 0.5)
            elif st in ("IFS_FRACTAL", "OCTAHEDRON"):
                d_kernel = sdf_octahedron(p_rot, 1.0 * scale)
            else:
                d_kernel = sdf_torus(p_rot, 0.9 * scale, 0.3 * scale)

            if sh["csg"] == "UNION":
                d_total = min(d_total, d_kernel)
            elif sh["csg"] == "INTERSECTION":
                d_total = max(d_total, d_kernel)
            else: # SMOOTH_MIN
                d_total = smin(d_total, d_kernel, sh["smoothness"])

        return d_total

    def render_shape_ascii(
        self,
        width: int = 64,
        height: int = 24,
        t: float = 0.0,
        cam_pos: Vec3 = (0.0, 3.5, -8.0),
        cam_target: Vec3 = (0.0, 0.0, 0.0)
    ) -> str:
        """Renders an ASCII camera view of the compiled program shape."""
        # Camera basis
        fwd = (cam_target[0] - cam_pos[0], cam_target[1] - cam_pos[1], cam_target[2] - cam_pos[2])
        f_len = math.hypot(fwd[0], math.hypot(fwd[1], fwd[2]))
        fwd = (fwd[0] / f_len, fwd[1] / f_len, fwd[2] / f_len) if f_len > 1e-5 else (0, 0, 1)

        up_guess = (0.0, 1.0, 0.0)
        rx = fwd[1]*up_guess[2] - fwd[2]*up_guess[1]
        ry = fwd[2]*up_guess[0] - fwd[0]*up_guess[2]
        rz = fwd[0]*up_guess[1] - fwd[1]*up_guess[0]
        r_len = math.hypot(rx, math.hypot(ry, rz))
        right = (rx / r_len, ry / r_len, rz / r_len) if r_len > 1e-5 else (1, 0, 0)

        ux = right[1]*fwd[2] - right[2]*fwd[1]
        uy = right[2]*fwd[0] - right[0]*fwd[2]
        uz = right[0]*fwd[1] - right[1]*fwd[0]
        up = (ux, uy, uz)

        aspect = (width / height) * 0.5
        fov = math.tan(math.radians(30.0))
        ramp = " .:;+=*#%@"
        lines = []

        sun = (0.577, 0.707, 0.408)

        for y in range(height):
            row = []
            sy = (1.0 - (2.0 * y / (height - 1))) * fov
            for x in range(width):
                sx = ((2.0 * x / (width - 1)) - 1.0) * aspect * fov
                rd = (fwd[0] + right[0]*sx + up[0]*sy,
                      fwd[1] + right[1]*sx + up[1]*sy,
                      fwd[2] + right[2]*sx + up[2]*sy)
                rd_len = math.hypot(rd[0], math.hypot(rd[1], rd[2]))
                rd = (rd[0] / rd_len, rd[1] / rd_len, rd[2] / rd_len)

                # Raymarch
                dist = 0.5
                hit = False
                for _ in range(32):
                    p = (cam_pos[0] + rd[0]*dist, cam_pos[1] + rd[1]*dist, cam_pos[2] + rd[2]*dist)
                    d = self.evaluate_program_sdf(p, t)
                    if d < 0.04:
                        hit = True
                        break
                    dist += max(0.12, d * 0.8)
                    if dist > 25.0:
                        break

                if hit:
                    # Normal approximation
                    eps = 0.05
                    p_hit = (cam_pos[0] + rd[0]*dist, cam_pos[1] + rd[1]*dist, cam_pos[2] + rd[2]*dist)
                    d_base = self.evaluate_program_sdf(p_hit, t)
                    nx = self.evaluate_program_sdf((p_hit[0]+eps, p_hit[1], p_hit[2]), t) - d_base
                    ny = self.evaluate_program_sdf((p_hit[0], p_hit[1]+eps, p_hit[2]), t) - d_base
                    nz = self.evaluate_program_sdf((p_hit[0], p_hit[1], p_hit[2]+eps), t) - d_base
                    n_len = math.hypot(nx, math.hypot(ny, nz))
                    diff = max(0.0, (nx*sun[0] + ny*sun[1] + nz*sun[2]) / (n_len + 1e-5))
                    idx = int((diff * 0.75 + 0.25) * (len(ramp) - 1))
                    idx = max(0, min(len(ramp) - 1, idx))
                    row.append(ramp[idx])
                else:
                    row.append(" ")
            lines.append("".join(row))

        return "\n".join(lines)
