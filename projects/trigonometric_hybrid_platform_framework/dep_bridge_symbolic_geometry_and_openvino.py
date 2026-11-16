"""Dependency Bridge Module 2: Symbolic Geometry (SymPy / NetworkX / Trimesh)
<-> Shared OpenVINO Hardware Runtime (`CPU` / `GPU` Iris Xe).

Prevents duplicate OpenCL driver initialization and bridges algebraic hex topology,
forward-kinematics rig poses, and watertight 3D terrain meshes into a unified descriptor.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import networkx as nx
import sympy as sp
import trimesh

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation

from .combat_turn_planner import DEFAULT_CACHE_DIR
from .intel_accelerator_and_power_thread_governor import get_shared_ov_core
from .dep_bridge_numeric_and_packet_abi import sanitize_vector_f32


def build_bridged_geometry_and_rig_state(
    arena_radius: int = 3,
    theta_deg: float = 58.3,
    hip_deg: float = 22.0,
    knee_deg: float = -34.0,
) -> Dict[str, Any]:
    """Bridge SymPy hex combinatorics, NetworkX 2D limb FK, Trimesh patch, and shared OpenVINO core."""
    if VITAL_MAX_HP != 6:
        raise ContractViolation("VITAL_MAX_HP != 6 in geometry bridge", violation_code="ERR_GEO_HP")
    if arena_radius < 1:
        raise ContractViolation("arena_radius must be >= 1", violation_code="ERR_GEO_RADIUS")

    # 1. Symbolic hex cell & directed arc count via SymPy
    r_sym = sp.Symbol("R", integer=True, positive=True)
    n_cells_expr = 1 + 3 * r_sym * (r_sym + 1)
    n_cells = int(n_cells_expr.subs(r_sym, arena_radius))

    # 2. Forward kinematics over a directed kinematic chain via NetworkX
    rig = nx.DiGraph()
    rig.add_edges_from([("pelvis", "hip"), ("hip", "knee"), ("knee", "foot")])
    if not nx.is_tree(rig):
        raise ContractViolation("Kinematic rig graph must be a tree", violation_code="ERR_GEO_RIG_TREE")

    a1 = math.radians(hip_deg)
    a2 = math.radians(hip_deg + knee_deg)
    knee_xy = sanitize_vector_f32([0.45 * math.sin(a1), -0.45 * math.cos(a1)])
    foot_xy = sanitize_vector_f32([knee_xy[0] + 0.42 * math.sin(a2), knee_xy[1] - 0.42 * math.cos(a2)])

    # 3. Compact procedural hex prism via Trimesh
    rad = math.radians(theta_deg)
    gonio_dir = sanitize_vector_f32([math.cos(rad), math.sin(rad)])
    cyl = trimesh.creation.cylinder(radius=1.0, height=0.35, sections=6)

    # 4. Shared memoized OpenVINO runtime check (zero extra OpenCL cold init)
    _, _, avail_devices = get_shared_ov_core(DEFAULT_CACHE_DIR)

    return {
        "bridge_id": "DEP_BRIDGE_02_SYMBOLIC_GEOMETRY_AND_OPENVINO",
        "vital_max_hp": VITAL_MAX_HP,
        "symbolic_hex_cells": n_cells,
        "goniometric_heading_xy": gonio_dir,
        "rig_joints_xy": {"pelvis": [0.0, 0.0], "knee": knee_xy, "foot": foot_xy},
        "hex_prism_mesh": {
            "vertices": int(len(cyl.vertices)),
            "faces": int(len(cyl.faces)),
            "is_watertight": bool(cyl.is_watertight),
        },
        "openvino_devices": list(avail_devices),
    }
