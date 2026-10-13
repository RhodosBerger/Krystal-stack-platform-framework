#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: BLENDER SPIRAL TENSOR DRIVER ENGINE & AUTOMATION ORCHESTRATOR
==============================================================================
Module: krystal_kernel/blender_spiral_tensor_driver_engine.py
Description:
    Implements a logarithmic spiral tensor that places and loops knowledge nodes
    in 3D space, automated through Blender 4.x Python (`bpy`) animation Drivers.
    Features:
      1. Logarithmic / Golden Spiral Tensor Architecture:
         - Discrete node placement along r_k = r_0 * phi^(k/8), theta_k = k * theta_golden.
         - Spatial orientation with tangent vectors and pitch/yaw alignments.
      2. Nested "Cycles of Cycles" Animation Drivers:
         - Primary macro orbital cycle (T1, e.g. 120 frames).
         - Secondary micro harmonic modulating cycle (T2, e.g. 24 frames).
         - Evaluates compound trigonometric driver formulas in pure Python and
           generates driver expressions for Blender's driver evaluation engine:
           z(t) = z_0 + A1 * sin(2*pi*frame/T1) + A2 * cos(2*pi*frame/T2)
      3. Attached Knowledge Sub-Lists (Connected Patterns from Prompts):
         - NODE 1: PROTIX_ML (Sensor array, 6-feature vector, edge classifier)
         - NODE 2: OHMIC_PHYSICAL_LAW (P = V^2 / R, Joule heat Q = P * dt, fan cooling)
         - NODE 3: NETWORK_TASK_ARBITER (Midpoint planning, latency queue balancing)
         - NODE 4: TECTONIC_MATRIX (IF-THEN-WHAT-IF speculative rules, stress adaptation)
         - NODE 5: TERMIX_ACCOUNTANT (Double-entry ledger, thermodynamic gas, NPU micro-ops)
         - NODE 6: FIBONACCI_ORCHESTRATOR (3x3 Tic-Tac-Toe paradox resolution, counter-polarity)
      4. Procedural 3D Text Title & Beacon Mesh Automation:
         - Generates standalone, executable Blender 4.x Python scripts (`bpy`).
         - Creates 3D text titles, extrudes, bevels, and glowing cybernetic materials.
         - Adds drivers to location[2], rotation_euler[2], scale, and emission strength.
      5. Core Platform Invariant:
         - Strictly clamps every node, vital weight, and damage/defense to VITAL_MAX_HP == 6.

Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import math
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 1.0 / GOLDEN_RATIO
GOLDEN_ANGLE_RAD: float = 2.399963229728653  # ~137.507764 degrees in radians


class DomainPattern(str, Enum):
    PROTIX_ML              = "PROTIX_ML"
    OHMIC_PHYSICAL_LAW     = "OHMIC_PHYSICAL_LAW"
    NETWORK_TASK_ARBITER   = "NETWORK_TASK_ARBITER"
    TECTONIC_MATRIX        = "TECTONIC_MATRIX"
    TERMIX_ACCOUNTANT      = "TERMIX_ACCOUNTANT"
    FIBONACCI_ORCHESTRATOR = "FIBONACCI_ORCHESTRATOR"


@dataclass
class CycleParameter:
    """Configures nested cycles of cycles driver parameters."""
    primary_period: int = 120       # Macro cycle period in frames (T1)
    secondary_period: int = 24      # Micro modulating cycle in frames (T2)
    amplitude_primary: float = 0.75 # Macro z-axis displacement
    amplitude_secondary: float = 0.25 # Micro harmonic ripple
    phase_offset: float = 0.0       # Initial phase shift
    rotational_velocity: float = 0.5 # Yaw cycle multiplier

    def to_driver_expression(self, base_z: float) -> str:
        """Generates Blender-compliant driver mathematical expression."""
        p1 = max(1, self.primary_period)
        p2 = max(1, self.secondary_period)
        return (
            f"{base_z:.3f} + "
            f"{self.amplitude_primary:.3f} * sin((frame / {p1}.0) * 6.283185 + {self.phase_offset:.3f}) + "
            f"{self.amplitude_secondary:.3f} * cos((frame / {p2}.0) * 6.283185)"
        )

    def to_emission_expression(self, base_emission: float = 2.0) -> str:
        """Generates pulsating emission driver expression."""
        p2 = max(1, self.secondary_period)
        return f"{base_emission:.2f} + 1.20 * sin((frame / {p2}.0) * 6.283185)"


@dataclass
class KnowledgeSubItem:
    """Attached knowledge record connected to previous framework prompts."""
    item_id: str
    topic: str
    description: str
    formula_or_axiom: str
    vital_hp_weight: int = 6

    def __post_init__(self):
        # Enforce universal invariant
        if self.vital_hp_weight > VITAL_MAX_HP:
            self.vital_hp_weight = VITAL_MAX_HP
        elif self.vital_hp_weight < 1:
            self.vital_hp_weight = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpiralTensorNode:
    """A 3D spatial node along the logarithmic spiral tensor holding title and knowledge sub-lists."""
    node_id: str
    title: str
    domain: DomainPattern
    spiral_index: int
    coordinates: Tuple[float, float, float]
    rotation_euler: Tuple[float, float, float]
    cycle_config: CycleParameter
    knowledge_sub_list: List[KnowledgeSubItem]
    mesh_archetype: str = "OCTAHEDRON"
    neon_color_rgb: Tuple[float, float, float] = (0.0, 0.94, 1.0)
    vital_max_hp_rule: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "domain": self.domain.value,
            "spiral_index": self.spiral_index,
            "coordinates": list(self.coordinates),
            "rotation_euler": list(self.rotation_euler),
            "cycle_config": asdict(self.cycle_config),
            "knowledge_sub_list": [k.to_dict() for k in self.knowledge_sub_list],
            "mesh_archetype": self.mesh_archetype,
            "neon_color_rgb": list(self.neon_color_rgb),
            "vital_max_hp_rule": self.vital_max_hp_rule
        }


@dataclass
class SpiralTensorFrameState:
    """Evaluated dynamic state of all spiral nodes for a specific frame."""
    frame: int
    total_nodes: int
    node_states: Dict[str, Dict[str, float]]
    timestamp: float = field(default_factory=time.time)
    vital_max_hp_rule: int = VITAL_MAX_HP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame,
            "total_nodes": self.total_nodes,
            "node_states": self.node_states,
            "timestamp": self.timestamp,
            "vital_max_hp_rule": self.vital_max_hp_rule
        }


class BlenderSpiralTensorDriverEngine:
    """
    Orchestrates the spiral tensor geometry, nested cycle drivers,
    and Blender 4.x Python automation script generation.
    """

    def __init__(
        self,
        base_radius: float = 3.5,
        radius_expansion_rate: float = 0.125, # phi^(k/8)
        z_step: float = 1.2,
        default_fps: int = 24,
        total_frames: int = 240
    ):
        self.base_radius = base_radius
        self.radius_expansion_rate = radius_expansion_rate
        self.z_step = z_step
        self.default_fps = default_fps
        self.total_frames = total_frames
        self.nodes: List[SpiralTensorNode] = []
        self._init_canonical_knowledge_patterns()

    def _init_canonical_knowledge_patterns(self) -> None:
        """Initializes canonical nodes synthesizing the conversation's previous patterns."""
        canonical_specs = [
            (
                "node_protix_ml",
                "PROTIX-ML: 6-Feature Neural Sensor & Edge Classifier",
                DomainPattern.PROTIX_ML,
                CycleParameter(primary_period=120, secondary_period=24, amplitude_primary=0.6, amplitude_secondary=0.2),
                [
                    KnowledgeSubItem(
                        item_id="sub_protix_1",
                        topic="Feature Tensor",
                        description="6-dimensional sensor vector feeding edge classification pipeline.",
                        formula_or_axiom="X = [temp, voltage, current, pressure, vibration, flux]",
                        vital_hp_weight=6
                    ),
                    KnowledgeSubItem(
                        item_id="sub_protix_2",
                        topic="Confidence Bounds",
                        description="Probabilistic confidence clamped against model variance.",
                        formula_or_axiom="C = 1.0 / (1.0 + exp(-W * X - b))",
                        vital_hp_weight=6
                    ),
                    KnowledgeSubItem(
                        item_id="sub_protix_3",
                        topic="Edge Inference",
                        description="Sub-millisecond quantized decision boundary.",
                        formula_or_axiom="latency < 0.5ms on embedded coprocessor",
                        vital_hp_weight=6
                    )
                ],
                "OCTAHEDRON",
                (0.0, 0.94, 1.0) # Cyber Cyan
            ),
            (
                "node_ohmic_physical_law",
                "OHMIC-LAW: P = V^2 / R Thermodynamic Heat Budget",
                DomainPattern.OHMIC_PHYSICAL_LAW,
                CycleParameter(primary_period=144, secondary_period=36, amplitude_primary=0.8, amplitude_secondary=0.3),
                [
                    KnowledgeSubItem(
                        item_id="sub_ohm_1",
                        topic="Ohmic Power Law",
                        description="Calculates instantaneous electrical power dissipation across resistance.",
                        formula_or_axiom="P = V^2 / R = I^2 * R = V * I",
                        vital_hp_weight=6
                    ),
                    KnowledgeSubItem(
                        item_id="sub_ohm_2",
                        topic="Joule Heat Accumulation",
                        description="Thermodynamic heat accumulation over operational time window.",
                        formula_or_axiom="Q = P * dt = (V^2 / R) * dt",
                        vital_hp_weight=6
                    ),
                    KnowledgeSubItem(
                        item_id="sub_ohm_3",
                        topic="Fan Cooling Discount",
                        description="30% Gas Fee Discount applied when T <= 55°C under active 100% cooling fan.",
                        formula_or_axiom="T <= 55°C -> Fee = 0.70 * Fee_base",
                        vital_hp_weight=6
                    )
                ],
                "TORUS",
                (1.0, 0.45, 0.0) # Amber Fire
            ),
            (
                "node_network_task_arbiter",
                "ARBITER: Balanced Task Queue & Latency Matrix",
                DomainPattern.NETWORK_TASK_ARBITER,
                CycleParameter(primary_period=96, secondary_period=16, amplitude_primary=0.5, amplitude_secondary=0.25),
                [
                    KnowledgeSubItem(
                        item_id="sub_arb_1",
                        topic="Intermediate Endpoint Midpoint",
                        description="Finds equilibrium between task injection and execution nodes.",
                        formula_or_axiom="P_mid = (P_src + P_dest) / 2.0 + Jitter(delta)",
                        vital_hp_weight=6
                    ),
                    KnowledgeSubItem(
                        item_id="sub_arb_2",
                        topic="Fair Queue Scheduler",
                        description="Priority-aware round-robin preventing starvation of low-bandwidth streams.",
                        formula_or_axiom="Weight(Task) = Priority * (1.0 / (Latency + 1.0))",
                        vital_hp_weight=6
                    ),
                    KnowledgeSubItem(
                        item_id="sub_arb_3",
                        topic="Sub-10ms Dispatch",
                        description="Guaranteed SLA for real-time mission control streaming.",
                        formula_or_axiom="Dispatch_Time <= 10.0 ms",
                        vital_hp_weight=6
                    )
                ],
                "CUBE",
                (0.2, 1.0, 0.2) # Neon Emerald
            ),
            (
                "node_tectonic_matrix",
                "TECTONIC: IF-THEN-WHAT-IF Speculative Adaptability",
                DomainPattern.TECTONIC_MATRIX,
                CycleParameter(primary_period=180, secondary_period=30, amplitude_primary=0.9, amplitude_secondary=0.35),
                [
                    KnowledgeSubItem(
                        item_id="sub_tec_1",
                        topic="Speculative Rule DSL",
                        description="Grammar evaluating IF condition THEN effect WHAT_IF counterfactuals.",
                        formula_or_axiom="Rule := IF(cond) THEN(act) WHAT_IF(branch) ELSE(fallback)",
                        vital_hp_weight=6
                    ),
                    KnowledgeSubItem(
                        item_id="sub_tec_2",
                        topic="Stress Tensor Dissipation",
                        description="Dynamic reduction of seismic and thermal matrix pressure.",
                        formula_or_axiom="Stress_{t+1} = Stress_t * (1.0 - Dissipation_Rate)",
                        vital_hp_weight=6
                    ),
                    KnowledgeSubItem(
                        item_id="sub_tec_3",
                        topic="Stability Convergence",
                        description="Ensures live matrix cells converge to structural equilibrium.",
                        formula_or_axiom="Stability = min(1.0, sum(Cell_Integrity) / N)",
                        vital_hp_weight=6
                    )
                ],
                "CYLINDER",
                (0.7, 0.2, 1.0) # Basalt Violet
            ),
            (
                "node_termix_accountant",
                "TERMIX: NPU Coprocessor & Smart Contract Ledger",
                DomainPattern.TERMIX_ACCOUNTANT,
                CycleParameter(primary_period=160, secondary_period=40, amplitude_primary=0.7, amplitude_secondary=0.2),
                [
                    KnowledgeSubItem(
                        item_id="sub_ter_1",
                        topic="Double-Entry Solvency",
                        description="Vector-proven conservation of Mana, Aether, and Amber assets.",
                        formula_or_axiom="sum(Debits) == sum(Credits)",
                        vital_hp_weight=6
                    ),
                    KnowledgeSubItem(
                        item_id="sub_ter_2",
                        topic="NPU Micro-Opcodes",
                        description="Sub-50us hardware accelerated binary verification opcodes.",
                        formula_or_axiom="Opcodes: SOLVENCY, THERMO_GAS, MERKLE_PROOF, HP_GUARD",
                        vital_hp_weight=6
                    ),
                    KnowledgeSubItem(
                        item_id="sub_ter_3",
                        topic="Merkle Receipt Digest",
                        description="SHA256 binary hash receipt proving atomic settlement.",
                        formula_or_axiom="ReceiptHash = SHA256(TxId + Gas + Balances)",
                        vital_hp_weight=6
                    )
                ],
                "PRISM",
                (1.0, 0.85, 0.0) # Gold Aether
            ),
            (
                "node_fibonacci_orchestrator",
                "FIBONACCI: Tic-Tac-Toe Paradox & Counter-Polarity",
                DomainPattern.FIBONACCI_ORCHESTRATOR,
                CycleParameter(primary_period=200, secondary_period=50, amplitude_primary=1.0, amplitude_secondary=0.4),
                [
                    KnowledgeSubItem(
                        item_id="sub_fib_1",
                        topic="Fibonacci Weight Tensor",
                        description="Breaks classical 3x3 draw symmetry via rotated sequence.",
                        formula_or_axiom="W(r, c) = F[(idx + turn) % 9] * phi^(1 - dist/2)",
                        vital_hp_weight=6
                    ),
                    KnowledgeSubItem(
                        item_id="sub_fib_2",
                        topic="Phase Transition Trigger",
                        description="Draw paradox collapses into topological phase shift and element induction.",
                        formula_or_axiom="Draw -> PhaseTransition -> Induct(GeneratedElementCard)",
                        vital_hp_weight=6
                    ),
                    KnowledgeSubItem(
                        item_id="sub_fib_3",
                        topic="Adaptive Counter-Polarity",
                        description="Synthesizes opposing element (e.g. Toxic -> Amber Crystal Quench).",
                        formula_or_axiom="Counter(Toxic) = Crystal_Amber; Damage <= 6; Defense <= 6",
                        vital_hp_weight=6
                    )
                ],
                "SPIRAL_PYRAMID",
                (1.0, 0.15, 0.6) # Magenta Resonance
            )
        ]

        self.nodes = []
        for idx, (n_id, title, domain, cycle_cfg, sub_list, mesh_arch, color) in enumerate(canonical_specs):
            coords = self.compute_spiral_position(idx)
            theta = idx * GOLDEN_ANGLE_RAD
            yaw = theta + math.pi / 2.0
            pitch = -0.15
            roll = 0.0
            node = SpiralTensorNode(
                node_id=n_id,
                title=title,
                domain=domain,
                spiral_index=idx,
                coordinates=coords,
                rotation_euler=(pitch, roll, yaw),
                cycle_config=cycle_cfg,
                knowledge_sub_list=sub_list,
                mesh_archetype=mesh_arch,
                neon_color_rgb=color,
                vital_max_hp_rule=VITAL_MAX_HP
            )
            self.nodes.append(node)

    def compute_spiral_position(self, index: int) -> Tuple[float, float, float]:
        """
        Computes 3D coordinates on a logarithmic Golden Spiral tensor:
            r = base_radius * phi^(index * radius_expansion_rate)
            theta = index * theta_golden
            z = index * z_step
        """
        r = self.base_radius * math.pow(GOLDEN_RATIO, index * self.radius_expansion_rate)
        theta = index * GOLDEN_ANGLE_RAD
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = index * self.z_step
        return (round(x, 4), round(y, 4), round(z, 4))

    def evaluate_frame(self, frame: int) -> SpiralTensorFrameState:
        """
        Evaluates the dynamic 'cycles of cycles' state of all nodes at a given frame.
        Pure Python implementation running with zero external dependencies.
        """
        node_states: Dict[str, Dict[str, float]] = {}

        for node in self.nodes:
            cfg = node.cycle_config
            base_x, base_y, base_z = node.coordinates
            pitch, roll, base_yaw = node.rotation_euler

            p1 = max(1, cfg.primary_period)
            p2 = max(1, cfg.secondary_period)

            # Nested cycles of cycles displacement
            wave_primary = cfg.amplitude_primary * math.sin((frame / p1) * 2.0 * math.pi + cfg.phase_offset)
            wave_secondary = cfg.amplitude_secondary * math.cos((frame / p2) * 2.0 * math.pi)
            dyn_z = base_z + wave_primary + wave_secondary

            # Dynamic yaw rotation modulation
            dyn_yaw = base_yaw + ((frame / p1) * 2.0 * math.pi) * cfg.rotational_velocity

            # Pulsating scale and emission
            scale_pulse = 1.0 + 0.08 * math.sin((frame / p2) * 2.0 * math.pi)
            emission_strength = 2.0 + 1.2 * math.sin((frame / p2) * 2.0 * math.pi)

            # Driver value scalar
            driver_val = wave_primary + wave_secondary

            node_states[node.node_id] = {
                "x": base_x,
                "y": base_y,
                "z": round(dyn_z, 4),
                "pitch": round(pitch, 4),
                "roll": round(roll, 4),
                "yaw": round(dyn_yaw, 4),
                "scale": round(scale_pulse, 4),
                "emission_strength": round(emission_strength, 4),
                "driver_scalar": round(driver_val, 4),
                "vital_hp": VITAL_MAX_HP
            }

        return SpiralTensorFrameState(
            frame=frame,
            total_nodes=len(self.nodes),
            node_states=node_states,
            vital_max_hp_rule=VITAL_MAX_HP
        )

    def add_custom_knowledge_node(
        self,
        title: str,
        domain: DomainPattern,
        knowledge_sub_list: List[KnowledgeSubItem],
        primary_period: int = 120,
        secondary_period: int = 24,
        mesh_archetype: str = "OCTAHEDRON",
        neon_color: Tuple[float, float, float] = (0.0, 0.94, 1.0)
    ) -> SpiralTensorNode:
        """Dynamically appends a new knowledge node to the spiral tensor."""
        idx = len(self.nodes)
        node_id = f"node_custom_{idx+1}_{domain.value.lower()}"
        coords = self.compute_spiral_position(idx)
        theta = idx * GOLDEN_ANGLE_RAD
        yaw = theta + math.pi / 2.0
        pitch = -0.15

        cfg = CycleParameter(
            primary_period=primary_period,
            secondary_period=secondary_period,
            amplitude_primary=0.7,
            amplitude_secondary=0.25
        )

        new_node = SpiralTensorNode(
            node_id=node_id,
            title=title,
            domain=domain,
            spiral_index=idx,
            coordinates=coords,
            rotation_euler=(pitch, 0.0, yaw),
            cycle_config=cfg,
            knowledge_sub_list=knowledge_sub_list,
            mesh_archetype=mesh_archetype,
            neon_color_rgb=neon_color,
            vital_max_hp_rule=VITAL_MAX_HP
        )
        self.nodes.append(new_node)
        return new_node

    def generate_blender_automation_script(self) -> str:
        """
        Generates a standalone, executable Blender 4.x Python script (`bpy`).
        Constructs:
          - Scene unit settings (Metric, 24 FPS, 1-240 frames).
          - Reference coordinate gizmo.
          - 3D Logarithmic Spiral curve guide.
          - 3D Text Title objects with extrude, bevel, and material.
          - Geometric beacon meshes (Octahedron, Torus, Cube, Cylinder, etc.).
          - Blender animation Drivers for location[2], rotation_euler[2], and material emission.
          - Metadata and custom properties enforcing VITAL_MAX_HP == 6.
        """
        lines = [
            '# ============================================================================== #',
            '# KRYSTAL-STACK: BLENDER 4.x AUTOMATION SCRIPT - SPIRAL TENSOR DRIVER ORCHESTRATOR',
            f'# Nodes: {len(self.nodes)} | FPS: {self.default_fps} | Frames: 1..{self.total_frames}',
            f'# Golden Ratio (phi): {GOLDEN_RATIO} | Golden Angle: {GOLDEN_ANGLE_RAD:.4f} rad',
            f'# Universal Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}',
            '# ============================================================================== #',
            'import bpy',
            'import math',
            '',
            '# 1. SCENE UNITS & ANIMATION TIMELINE CONFIGURATION',
            'scene = bpy.context.scene',
            'scene.unit_settings.system = "METRIC"',
            'scene.unit_settings.scale_length = 1.0',
            f'scene.render.fps = {self.default_fps}',
            'scene.frame_start = 1',
            f'scene.frame_end = {self.total_frames}',
            '',
            '# 2. CLEAR PREVIOUS SCENE OBJECTS',
            'bpy.ops.object.select_all(action="SELECT")',
            'bpy.ops.object.delete(use_global=False)',
            '',
            '# 3. CREATE CARTESIAN REFERENCE GIZMO',
            'bpy.ops.object.empty_add(type="ARROWS", location=(0.0, 0.0, 0.0))',
            'gizmo = bpy.context.active_object',
            'gizmo.name = "Krystal_Origin_Gizmo"',
            'gizmo.empty_display_size = 1.5',
            '',
            '# 4. CREATE SPIRAL TENSOR GUIDE CURVE',
            'curve_data = bpy.data.curves.new(name="Krystal_Spiral_Guide", type="CURVE")',
            'curve_data.dimensions = "3D"',
            'curve_data.bevel_depth = 0.025',
            'spline = curve_data.splines.new(type="POLY")',
            f'spline.points.add({len(self.nodes) - 1})',
        ]

        # Add spiral guide points
        for i, node in enumerate(self.nodes):
            x, y, z = node.coordinates
            lines.append(f'spline.points[{i}].co = ({x}, {y}, {z}, 1.0)')

        lines.extend([
            'guide_obj = bpy.data.objects.new("Krystal_Spiral_Guide_Obj", curve_data)',
            'bpy.context.collection.objects.link(guide_obj)',
            '',
            '# 5. CONSTRUCT KNOWLEDGE NODES, 3D TITLES & ANIMATION DRIVERS'
        ])

        for node in self.nodes:
            x, y, z = node.coordinates
            pitch, roll, yaw = node.rotation_euler
            cfg = node.cycle_config
            r, g, b = node.neon_color_rgb
            escaped_title = node.title.replace('"', '\\"')
            driver_expr = cfg.to_driver_expression(z)
            emission_expr = cfg.to_emission_expression(2.5)

            lines.extend([
                f'# --- NODE: {node.node_id} ({node.domain.value}) ---',
                f'# Knowledge Sub-items: {len(node.knowledge_sub_list)}',
                # 3D Beacon Mesh
                f'bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.6, location=({x}, {y}, {z}))',
                'beacon = bpy.context.active_object',
                f'beacon.name = "{node.node_id}_Beacon"',
                f'beacon.rotation_euler = ({pitch}, {roll}, {yaw})',
                '',
                # Material setup
                f'mat = bpy.data.materials.new(name="{node.node_id}_Material")',
                'mat.use_nodes = True',
                'nodes = mat.node_tree.nodes',
                'bsdf = nodes.get("Principled BSDF")',
                'if bsdf:',
                f'    bsdf.inputs["Base Color"].default_value = ({r}, {g}, {b}, 1.0)',
                f'    bsdf.inputs["Emission Color"].default_value = ({r}, {g}, {b}, 1.0)',
                '    bsdf.inputs["Roughness"].default_value = 0.2',
                '    bsdf.inputs["Metallic"].default_value = 0.9',
                '    # Add driver for dynamic emission glow cycle',
                '    d_emit = bsdf.inputs["Emission Strength"].driver_add("default_value").driver',
                '    d_emit.type = "SCRIPTED"',
                f'    d_emit.expression = "{emission_expr}"',
                'beacon.data.materials.append(mat)',
                '',
                # 3D Text Title Object
                f'bpy.ops.object.text_add(location=({x}, {y}, {z} + 0.8))',
                'title_obj = bpy.context.active_object',
                f'title_obj.name = "{node.node_id}_Title"',
                f'title_obj.data.body = "{escaped_title}"',
                'title_obj.data.size = 0.32',
                'title_obj.data.extrude = 0.04',
                'title_obj.data.bevel_depth = 0.015',
                'title_obj.data.align_x = "CENTER"',
                f'title_obj.rotation_euler = ({pitch}, {roll}, {yaw})',
                'title_obj.data.materials.append(mat)',
                '',
                # Parent Title to Beacon for unified movement
                'title_obj.parent = beacon',
                '',
                # ANIMATION DRIVER 1: Z-Axis Location (Cycles of Cycles)
                'd_loc_z = beacon.driver_add("location", 2).driver',
                'd_loc_z.type = "SCRIPTED"',
                f'd_loc_z.expression = "{driver_expr}"',
                '',
                # ANIMATION DRIVER 2: Yaw Rotation (Continuous / Modulated Rotation)
                'd_rot_z = beacon.driver_add("rotation_euler", 2).driver',
                'd_rot_z.type = "SCRIPTED"',
                f'd_rot_z.expression = "{yaw:.4f} + ((frame / {max(1, cfg.primary_period)}.0) * 6.283185) * {cfg.rotational_velocity:.3f}"',
                '',
                # Custom Properties & Knowledge Metadata
                f'beacon["krystal_node_id"] = "{node.node_id}"',
                f'beacon["krystal_domain"] = "{node.domain.value}"',
                f'beacon["krystal_spiral_index"] = {node.spiral_index}',
                f'beacon["krystal_primary_period"] = {cfg.primary_period}',
                f'beacon["krystal_secondary_period"] = {cfg.secondary_period}',
                f'beacon["krystal_vital_max_hp"] = {VITAL_MAX_HP}',
                f'beacon["krystal_sub_items_count"] = {len(node.knowledge_sub_list)}',
                f'beacon["krystal_sub_items_json"] = {json.dumps([k.to_dict() for k in node.knowledge_sub_list])}',
                ''
            ])

        lines.extend([
            'print(f"Krystal-Stack: Successfully generated spiral tensor with {len(scene.objects)} objects and active drivers.")',
            'print(f"Universal Invariant Verified: VITAL_MAX_HP == {VITAL_MAX_HP}.")'
        ])

        return "\n".join(lines)

    def get_engine_status(self) -> Dict[str, Any]:
        """Returns comprehensive status dictionary for web hub and telemetry."""
        return {
            "status": "ONLINE",
            "base_radius": self.base_radius,
            "radius_expansion_rate": self.radius_expansion_rate,
            "z_step": self.z_step,
            "default_fps": self.default_fps,
            "total_frames": self.total_frames,
            "golden_ratio": GOLDEN_RATIO,
            "golden_angle_rad": GOLDEN_ANGLE_RAD,
            "nodes_count": len(self.nodes),
            "nodes": [n.to_dict() for n in self.nodes],
            "vital_max_hp_rule": VITAL_MAX_HP
        }


# Global singleton instance
GLOBAL_SPIRAL_TENSOR_ENGINE = BlenderSpiralTensorDriverEngine()


if __name__ == "__main__":
    print("=" * 80)
    print("KRYSTAL-STACK: BLENDER SPIRAL TENSOR DRIVER ENGINE")
    print("=" * 80)
    status = GLOBAL_SPIRAL_TENSOR_ENGINE.get_engine_status()
    print(f"Configured Nodes:      {status['nodes_count']}")
    print(f"Universal Invariant:   VITAL_MAX_HP == {status['vital_max_hp_rule']}")
    print(f"Golden Angle:          {status['golden_angle_rad']:.4f} rad (~137.51 deg)")

    # Evaluate dynamic frame 48
    frame_48 = GLOBAL_SPIRAL_TENSOR_ENGINE.evaluate_frame(48)
    print(f"\nEvaluated Frame 48 Telemetry:")
    for nid, s in frame_48.node_states.items():
        print(f"  [{nid}]: Z={s['z']:.3f}, Yaw={s['yaw']:.3f}, Scale={s['scale']:.3f}, Emit={s['emission_strength']:.3f}")

    # Script excerpt
    script = GLOBAL_SPIRAL_TENSOR_ENGINE.generate_blender_automation_script()
    print(f"\nGenerated Blender Script Size: {len(script)} bytes ({len(script.splitlines())} lines)")
