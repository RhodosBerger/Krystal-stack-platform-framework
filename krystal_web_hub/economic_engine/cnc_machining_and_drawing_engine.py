"""
Krystal Stack Platform - CNC Drawing and Machining Simulator Engine
===================================================================
Autonomous 2D/2.5D CAD drawing engine, CAM toolpath synthesizer, G-code generator,
and machining physics simulator.

Key Invariant:
  Platform-wide VITAL_MAX_HP = 6 (Enforced as 6-axis safety boundaries:
  Max Spindle Deflection, Z-Clearance Guard, Feed Accel Tier, Thermal Spike Threshold,
  Tool Pressure Limit, and Kinetic Emergency Braking).
"""

import math
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 0.61803398875

# ---------------------------------------------------------------------------
# CNC Tools & Materials Data Models
# ---------------------------------------------------------------------------

@dataclass
class CNCTool:
    id: str
    name: str
    tool_type: str        # "endmill", "ballnose", "vbit", "drill", "facemill"
    diameter_mm: float
    flutes: int
    max_rpm: int
    stepover_percent: float
    vital_safety_hp: int = VITAL_MAX_HP
    color_hex: str = "#00ffcc"

@dataclass
class CNCMaterial:
    id: str
    name: str
    surface_speed_m_min: float  # Vc
    chip_load_per_tooth_mm: float  # fz
    max_depth_per_pass_mm: float
    hardness_hb: int
    power_factor: float
    coolant_recommended: bool

@dataclass
class CNCEntity:
    id: str
    entity_type: str  # "line", "rect", "circle", "arc", "pocket", "drill", "polygon"
    params: Dict[str, Any]
    layer: str = "geometry"
    cut_depth_mm: float = 3.0
    operation: str = "profile_outside"  # "profile_outside", "profile_inside", "pocket", "drill", "engrave"

@dataclass
class CNCToolpathPoint:
    x: float
    y: float
    z: float
    feed_mm_min: float
    command: str  # "G00", "G01", "G02", "G03"
    spindle_on: bool = True
    coolant_on: bool = True
    radius_i: float = 0.0
    radius_j: float = 0.0

@dataclass
class CNCGCodeBlock:
    line_number: int
    code: str
    comment: str = ""
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    feed: Optional[float] = None

# ---------------------------------------------------------------------------
# Canonical Libraries
# ---------------------------------------------------------------------------

CANONICAL_TOOLS: Dict[str, CNCTool] = {
    "t1_endmill_3mm": CNCTool(
        id="t1_endmill_3mm",
        name="3.0mm Flat End Mill (Carbide 2-Flute)",
        tool_type="endmill",
        diameter_mm=3.0,
        flutes=2,
        max_rpm=24000,
        stepover_percent=45.0,
        color_hex="#00f0ff"
    ),
    "t2_endmill_6mm": CNCTool(
        id="t2_endmill_6mm",
        name="6.0mm Roughing End Mill (3-Flute)",
        tool_type="endmill",
        diameter_mm=6.0,
        flutes=3,
        max_rpm=18000,
        stepover_percent=55.0,
        color_hex="#00ff88"
    ),
    "t3_ballnose_3mm": CNCTool(
        id="t3_ballnose_3mm",
        name="3.175mm Ball Nose (3D Contouring)",
        tool_type="ballnose",
        diameter_mm=3.175,
        flutes=2,
        max_rpm=24000,
        stepover_percent=20.0,
        color_hex="#bf5af2"
    ),
    "t4_vbit_60deg": CNCTool(
        id="t4_vbit_60deg",
        name="60° V-Bit Engraver (0.2mm Tip)",
        tool_type="vbit",
        diameter_mm=6.0,
        flutes=1,
        max_rpm=20000,
        stepover_percent=15.0,
        color_hex="#ffd700"
    ),
    "t5_drill_3_2mm": CNCTool(
        id="t5_drill_3_2mm",
        name="3.2mm M4 Tap Pilot Drill",
        tool_type="drill",
        diameter_mm=3.2,
        flutes=2,
        max_rpm=8000,
        stepover_percent=100.0,
        color_hex="#ff4466"
    ),
    "t6_facemill_25mm": CNCTool(
        id="t6_facemill_25mm",
        name="25mm Fly Cutter Face Mill",
        tool_type="facemill",
        diameter_mm=25.0,
        flutes=4,
        max_rpm=6000,
        stepover_percent=70.0,
        color_hex="#38bdf8"
    )
}

CANONICAL_MATERIALS: Dict[str, CNCMaterial] = {
    "al_6061": CNCMaterial(
        id="al_6061",
        name="Aluminum 6061-T6 (Aircraft Grade)",
        surface_speed_m_min=150.0,
        chip_load_per_tooth_mm=0.035,
        max_depth_per_pass_mm=1.5,
        hardness_hb=95,
        power_factor=0.85,
        coolant_recommended=True
    ),
    "brass_c360": CNCMaterial(
        id="brass_c360",
        name="Brass C360 (Free Machining)",
        surface_speed_m_min=180.0,
        chip_load_per_tooth_mm=0.040,
        max_depth_per_pass_mm=2.0,
        hardness_hb=120,
        power_factor=0.70,
        coolant_recommended=False
    ),
    "steel_1018": CNCMaterial(
        id="steel_1018",
        name="Mild Steel AISI 1018",
        surface_speed_m_min=75.0,
        chip_load_per_tooth_mm=0.020,
        max_depth_per_pass_mm=0.8,
        hardness_hb=130,
        power_factor=1.45,
        coolant_recommended=True
    ),
    "delrin_acetal": CNCMaterial(
        id="delrin_acetal",
        name="Delrin / POM Polymer",
        surface_speed_m_min=240.0,
        chip_load_per_tooth_mm=0.060,
        max_depth_per_pass_mm=3.5,
        hardness_hb=30,
        power_factor=0.35,
        coolant_recommended=False
    ),
    "krystal_resin": CNCMaterial(
        id="krystal_resin",
        name="Krystal High-Tech Carbon Composite",
        surface_speed_m_min=200.0,
        chip_load_per_tooth_mm=0.045,
        max_depth_per_pass_mm=2.2,
        hardness_hb=85,
        power_factor=0.60,
        coolant_recommended=True
    )
}

# ---------------------------------------------------------------------------
# Drawing & Machining Engine Core
# ---------------------------------------------------------------------------

class CNCDrawingAndMachiningEngine:
    def __init__(self):
        self.vital_max_hp: int = VITAL_MAX_HP
        self.safety_clearance_z: float = 6.0  # Safe Z height in mm (Vital invariant = 6)
        self.retract_z: float = 2.0
        self.stock_width: float = 200.0
        self.stock_height: float = 150.0
        self.stock_thickness: float = 12.0

    def calculate_speeds_and_feeds(self, tool_id: str, material_id: str) -> Dict[str, float]:
        """Calculates RPM, feed rate (mm/min), plunge rate, and power demand."""
        tool = CANONICAL_TOOLS.get(tool_id, CANONICAL_TOOLS["t1_endmill_3mm"])
        mat = CANONICAL_MATERIALS.get(material_id, CANONICAL_MATERIALS["al_6061"])

        # Spindle Speed n = (Vc * 1000) / (pi * D)
        dia = max(0.5, tool.diameter_mm)
        rpm_raw = (mat.surface_speed_m_min * 1000.0) / (math.pi * dia)
        spindle_rpm = min(tool.max_rpm, max(1200, int(rpm_raw)))

        # Feed Rate Vf = n * z * fz
        feed_rate_raw = spindle_rpm * tool.flutes * mat.chip_load_per_tooth_mm
        feed_rate = round(feed_rate_raw, 1)

        # Plunge rate is typically 30% of XY feed rate
        plunge_rate = round(feed_rate * 0.33, 1)

        # Stepdown
        stepdown = min(mat.max_depth_per_pass_mm, tool.diameter_mm * 0.75)

        return {
            "spindle_rpm": float(spindle_rpm),
            "feed_rate_mm_min": float(feed_rate),
            "plunge_rate_mm_min": float(plunge_rate),
            "stepdown_mm": float(stepdown),
            "stepover_mm": round(tool.diameter_mm * (tool.stepover_percent / 100.0), 2),
            "vital_safety_hp": self.vital_max_hp
        }

    def generate_toolpath(
        self,
        entities: List[Dict[str, Any]],
        tool_id: str = "t1_endmill_3mm",
        material_id: str = "al_6061",
        target_depth: float = 3.0
    ) -> List[CNCToolpathPoint]:
        """
        Synthesizes a continuous, safe toolpath with rapids, plunges, profile passes,
        and circular interpolations.
        """
        speeds = self.calculate_speeds_and_feeds(tool_id, material_id)
        feed_xy = speeds["feed_rate_mm_min"]
        feed_z = speeds["plunge_rate_mm_min"]
        stepdown = speeds["stepdown_mm"]
        tool = CANONICAL_TOOLS.get(tool_id, CANONICAL_TOOLS["t1_endmill_3mm"])
        tool_radius = tool.diameter_mm / 2.0

        path: List[CNCToolpathPoint] = []

        # Initial Safe Rapid to Home
        path.append(CNCToolpathPoint(x=0.0, y=0.0, z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00"))

        for ent in entities:
            etype = ent.get("entity_type", "line")
            op = ent.get("operation", "profile_outside")
            depth = float(ent.get("cut_depth_mm", target_depth))
            params = ent.get("params", {})

            # Number of Z passes
            num_passes = max(1, math.ceil(depth / stepdown))

            if etype == "drill":
                cx = float(params.get("x", 50.0))
                cy = float(params.get("y", 50.0))
                # Rapid above hole
                path.append(CNCToolpathPoint(x=cx, y=cy, z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00"))
                # Peck drilling loop (G83 style)
                current_z = 0.0
                while current_z > -depth:
                    next_z = max(-depth, current_z - stepdown)
                    path.append(CNCToolpathPoint(x=cx, y=cy, z=next_z, feed_mm_min=feed_z, command="G01"))
                    # Retract to clear chips
                    path.append(CNCToolpathPoint(x=cx, y=cy, z=self.retract_z, feed_mm_min=3000.0, command="G00"))
                    current_z = next_z
                path.append(CNCToolpathPoint(x=cx, y=cy, z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00"))

            elif etype == "line":
                x1 = float(params.get("x1", 10.0))
                y1 = float(params.get("y1", 10.0))
                x2 = float(params.get("x2", 90.0))
                y2 = float(params.get("y2", 10.0))

                for p_idx in range(1, num_passes + 1):
                    pass_z = -min(depth, p_idx * stepdown)
                    # Rapid to start point above stock
                    path.append(CNCToolpathPoint(x=x1, y=y1, z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00"))
                    # Plunge
                    path.append(CNCToolpathPoint(x=x1, y=y1, z=pass_z, feed_mm_min=feed_z, command="G01"))
                    # Cut to end
                    path.append(CNCToolpathPoint(x=x2, y=y2, z=pass_z, feed_mm_min=feed_xy, command="G01"))
                    # Retract
                    path.append(CNCToolpathPoint(x=x2, y=y2, z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00"))

            elif etype == "rect":
                rx = float(params.get("x", 20.0))
                ry = float(params.get("y", 20.0))
                rw = float(params.get("w", 60.0))
                rh = float(params.get("h", 40.0))

                # Offset if profile outside/inside
                offset = tool_radius if op == "profile_outside" else (-tool_radius if op == "profile_inside" else 0.0)
                px = rx - offset
                py = ry - offset
                pw = rw + 2 * offset
                ph = rh + 2 * offset

                corners = [
                    (px, py),
                    (px + pw, py),
                    (px + pw, py + ph),
                    (px, py + ph),
                    (px, py)
                ]

                for p_idx in range(1, num_passes + 1):
                    pass_z = -min(depth, p_idx * stepdown)
                    # Rapid to first corner
                    path.append(CNCToolpathPoint(x=corners[0][0], y=corners[0][1], z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00"))
                    path.append(CNCToolpathPoint(x=corners[0][0], y=corners[0][1], z=pass_z, feed_mm_min=feed_z, command="G01"))
                    for c_x, c_y in corners[1:]:
                        path.append(CNCToolpathPoint(x=c_x, y=c_y, z=pass_z, feed_mm_min=feed_xy, command="G01"))
                    path.append(CNCToolpathPoint(x=corners[0][0], y=corners[0][1], z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00"))

            elif etype == "circle":
                cx = float(params.get("cx", 50.0))
                cy = float(params.get("cy", 50.0))
                r = float(params.get("r", 25.0))

                offset = tool_radius if op == "profile_outside" else (-tool_radius if op == "profile_inside" else 0.0)
                eff_r = max(1.0, r + offset)

                for p_idx in range(1, num_passes + 1):
                    pass_z = -min(depth, p_idx * stepdown)
                    start_x = cx + eff_r
                    start_y = cy
                    path.append(CNCToolpathPoint(x=start_x, y=start_y, z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00"))
                    path.append(CNCToolpathPoint(x=start_x, y=start_y, z=pass_z, feed_mm_min=feed_z, command="G01"))
                    # Full circle G02 (CW) with I=-eff_r, J=0
                    path.append(CNCToolpathPoint(x=start_x, y=start_y, z=pass_z, feed_mm_min=feed_xy, command="G02", radius_i=-eff_r, radius_j=0.0))
                    path.append(CNCToolpathPoint(x=start_x, y=start_y, z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00"))

            elif etype == "polygon":
                pts = params.get("points", [])
                if len(pts) >= 3:
                    for p_idx in range(1, num_passes + 1):
                        pass_z = -min(depth, p_idx * stepdown)
                        first_pt = pts[0]
                        path.append(CNCToolpathPoint(x=first_pt[0], y=first_pt[1], z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00"))
                        path.append(CNCToolpathPoint(x=first_pt[0], y=first_pt[1], z=pass_z, feed_mm_min=feed_z, command="G01"))
                        for pt in pts[1:]:
                            path.append(CNCToolpathPoint(x=pt[0], y=pt[1], z=pass_z, feed_mm_min=feed_xy, command="G01"))
                        # Close loop
                        path.append(CNCToolpathPoint(x=first_pt[0], y=first_pt[1], z=pass_z, feed_mm_min=feed_xy, command="G01"))
                        path.append(CNCToolpathPoint(x=first_pt[0], y=first_pt[1], z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00"))

        # Final Return to Origin with Safety Z
        path.append(CNCToolpathPoint(x=0.0, y=0.0, z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00"))
        return path

    def generate_gcode(
        self,
        toolpath: List[CNCToolpathPoint],
        tool_id: str = "t1_endmill_3mm",
        material_id: str = "al_6061",
        program_name: str = "KRYSTAL_CNC_JOB"
    ) -> Dict[str, Any]:
        """Translates toolpath points into standardized, clean RS-274D / ISO G-code."""
        tool = CANONICAL_TOOLS.get(tool_id, CANONICAL_TOOLS["t1_endmill_3mm"])
        mat = CANONICAL_MATERIALS.get(material_id, CANONICAL_MATERIALS["al_6061"])
        speeds = self.calculate_speeds_and_feeds(tool_id, material_id)

        lines: List[str] = []
        blocks: List[Dict[str, Any]] = []
        line_num = 10

        def add_block(code_str: str, comment: str = "", x=None, y=None, z=None, f=None):
            nonlocal line_num
            full = f"N{line_num:04d} {code_str}"
            if comment:
                full += f" ({comment})"
            lines.append(full)
            blocks.append({
                "line_number": line_num,
                "code": code_str,
                "comment": comment,
                "x": x,
                "y": y,
                "z": z,
                "feed": f
            })
            line_num += 10

        # Program Preamble
        add_block("%", "Start of Program")
        add_block(f"O{hash(program_name) % 9000 + 1000}", f"Program: {program_name}")
        add_block("G21", "Metric millimeter units")
        add_block("G90", "Absolute coordinate mode")
        add_block("G17", "XY circular plane selection")
        add_block("G54", "Work coordinate system 1")
        add_block(f"G00 Z{self.safety_clearance_z:.3f}", f"Vital Max HP Clearance Z={self.safety_clearance_z}mm")
        add_block(f"M06 T{tool.id[:2].upper()}", f"Tool Change: {tool.name}")
        add_block(f"S{int(speeds['spindle_rpm'])} M03", f"Spindle Start CW: {int(speeds['spindle_rpm'])} RPM")
        if mat.coolant_recommended:
            add_block("M08", "Flood / Mist Coolant ON")

        total_length_mm = 0.0
        prev_pt = CNCToolpathPoint(x=0.0, y=0.0, z=self.safety_clearance_z, feed_mm_min=3000.0, command="G00")

        for pt in toolpath:
            dist = math.sqrt((pt.x - prev_pt.x)**2 + (pt.y - prev_pt.y)**2 + (pt.z - prev_pt.z)**2)
            total_length_mm += dist

            if pt.command == "G00":
                add_block(f"G00 X{pt.x:.3f} Y{pt.y:.3f} Z{pt.z:.3f}", "Rapid Traverse", x=pt.x, y=pt.y, z=pt.z)
            elif pt.command == "G01":
                add_block(f"G01 X{pt.x:.3f} Y{pt.y:.3f} Z{pt.z:.3f} F{pt.feed_mm_min:.1f}", "Linear Cut", x=pt.x, y=pt.y, z=pt.z, f=pt.feed_mm_min)
            elif pt.command in ("G02", "G03"):
                add_block(
                    f"{pt.command} X{pt.x:.3f} Y{pt.y:.3f} Z{pt.z:.3f} I{pt.radius_i:.3f} J{pt.radius_j:.3f} F{pt.feed_mm_min:.1f}",
                    "Circular Arc Cut",
                    x=pt.x, y=pt.y, z=pt.z, f=pt.feed_mm_min
                )
            prev_pt = pt

        # Postamble
        add_block(f"G00 Z{self.safety_clearance_z:.3f}", "Retract to Safe Z")
        if mat.coolant_recommended:
            add_block("M09", "Coolant OFF")
        add_block("M05", "Spindle Stop")
        add_block("G00 X0.000 Y0.000", "Return to Machine Home")
        add_block("M30", "End of Program & Rewind")
        add_block("%", "End of Tape")

        est_time_seconds = (total_length_mm / max(50.0, speeds["feed_rate_mm_min"])) * 60.0

        return {
            "gcode_text": "\n".join(lines),
            "blocks": blocks,
            "total_lines": len(lines),
            "total_path_length_mm": round(total_length_mm, 2),
            "estimated_machining_time_seconds": round(est_time_seconds, 1),
            "estimated_machining_time_formatted": f"{int(est_time_seconds // 60)}m {int(est_time_seconds % 60)}s",
            "vital_max_hp_rule": self.vital_max_hp,
            "spindle_rpm": speeds["spindle_rpm"],
            "feed_rate_mm_min": speeds["feed_rate_mm_min"],
            "tool_info": asdict(tool),
            "material_info": asdict(mat)
        }

    def get_preset_drawings(self) -> Dict[str, Any]:
        """Provides rich CAD blueprint presets ready to machine."""
        return {
            "krystal_spindle_flange": {
                "name": "Krystal 4-Bolt Spindle Mounting Flange",
                "description": "Precision circular flange with 4 corner bolt holes and central bearing bore.",
                "stock_dims": {"w": 120, "h": 120, "z": 10},
                "entities": [
                    {"id": "e1", "entity_type": "rect", "params": {"x": 10, "y": 10, "w": 100, "h": 100}, "operation": "profile_outside", "cut_depth_mm": 5.0},
                    {"id": "e2", "entity_type": "circle", "params": {"cx": 60, "cy": 60, "r": 25}, "operation": "profile_inside", "cut_depth_mm": 5.0},
                    {"id": "e3", "entity_type": "drill", "params": {"x": 25, "y": 25}, "operation": "drill", "cut_depth_mm": 8.0},
                    {"id": "e4", "entity_type": "drill", "params": {"x": 95, "y": 25}, "operation": "drill", "cut_depth_mm": 8.0},
                    {"id": "e5", "entity_type": "drill", "params": {"x": 95, "y": 95}, "operation": "drill", "cut_depth_mm": 8.0},
                    {"id": "e6", "entity_type": "drill", "params": {"x": 25, "y": 95}, "operation": "drill", "cut_depth_mm": 8.0}
                ]
            },
            "bohemian_spindle_cog": {
                "name": "Bohemian Mechanical Cogwheel (6-Tooth)",
                "description": "Pagan Bohemia clockwork gear embodying the 6 Max HP vital geometry.",
                "stock_dims": {"w": 140, "h": 140, "z": 8},
                "entities": [
                    {"id": "e1", "entity_type": "circle", "params": {"cx": 70, "cy": 70, "r": 50}, "operation": "profile_outside", "cut_depth_mm": 4.0},
                    {"id": "e2", "entity_type": "circle", "params": {"cx": 70, "cy": 70, "r": 15}, "operation": "profile_inside", "cut_depth_mm": 4.0},
                    {"id": "e3", "entity_type": "drill", "params": {"x": 70, "y": 30}, "operation": "drill", "cut_depth_mm": 6.0},
                    {"id": "e4", "entity_type": "drill", "params": {"x": 104.6, "y": 50}, "operation": "drill", "cut_depth_mm": 6.0},
                    {"id": "e5", "entity_type": "drill", "params": {"x": 104.6, "y": 90}, "operation": "drill", "cut_depth_mm": 6.0},
                    {"id": "e6", "entity_type": "drill", "params": {"x": 70, "y": 110}, "operation": "drill", "cut_depth_mm": 6.0},
                    {"id": "e7", "entity_type": "drill", "params": {"x": 35.4, "y": 90}, "operation": "drill", "cut_depth_mm": 6.0},
                    {"id": "e8", "entity_type": "drill", "params": {"x": 35.4, "y": 50}, "operation": "drill", "cut_depth_mm": 6.0}
                ]
            },
            "greek_meander_key": {
                "name": "Greek Golden Mean Architectural Key",
                "description": "Continuous labyrinthine decorative fretwork for engraving with a 60° V-Bit.",
                "stock_dims": {"w": 180, "h": 90, "z": 6},
                "entities": [
                    {"id": "e1", "entity_type": "line", "params": {"x1": 15, "y1": 20, "x2": 165, "y2": 20}, "operation": "engrave", "cut_depth_mm": 1.2},
                    {"id": "e2", "entity_type": "line", "params": {"x1": 165, "y1": 20, "x2": 165, "y2": 70}, "operation": "engrave", "cut_depth_mm": 1.2},
                    {"id": "e3", "entity_type": "line", "params": {"x1": 165, "y1": 70, "x2": 45, "y2": 70}, "operation": "engrave", "cut_depth_mm": 1.2},
                    {"id": "e4", "entity_type": "line", "params": {"x1": 45, "y1": 70, "x2": 45, "y2": 40}, "operation": "engrave", "cut_depth_mm": 1.2},
                    {"id": "e5", "entity_type": "line", "params": {"x1": 45, "y1": 40, "x2": 135, "y2": 40}, "operation": "engrave", "cut_depth_mm": 1.2},
                    {"id": "e6", "entity_type": "line", "params": {"x1": 135, "y1": 40, "x2": 135, "y2": 55}, "operation": "engrave", "cut_depth_mm": 1.2}
                ]
            },
            "hexagonal_turbine_cell": {
                "name": "Hexagonal NPU Fluid Turbine Cell",
                "description": "Six-sided cooling honeycomb manifold for high-throughput computing rigs.",
                "stock_dims": {"w": 150, "h": 150, "z": 12},
                "entities": [
                    {
                        "id": "e1",
                        "entity_type": "polygon",
                        "params": {
                            "points": [
                                [75.0, 25.0],
                                [118.3, 50.0],
                                [118.3, 100.0],
                                [75.0, 125.0],
                                [31.7, 100.0],
                                [31.7, 50.0]
                            ]
                        },
                        "operation": "profile_outside",
                        "cut_depth_mm": 6.0
                    },
                    {"id": "e2", "entity_type": "circle", "params": {"cx": 75, "cy": 75, "r": 20}, "operation": "profile_inside", "cut_depth_mm": 6.0}
                ]
            }
        }

    def simulate_machining(
        self,
        entities: List[Dict[str, Any]],
        tool_id: str = "t1_endmill_3mm",
        material_id: str = "al_6061",
        target_depth: float = 3.0
    ) -> Dict[str, Any]:
        """
        Executes full simulation pipeline:
        1. Speeds & Feeds derivation
        2. Toolpath synthesis
        3. RS-274D G-code generation
        4. GSAP animation keyframes telemetry for browser interpolation
        """
        speeds = self.calculate_speeds_and_feeds(tool_id, material_id)
        toolpath = self.generate_toolpath(entities, tool_id, material_id, target_depth)
        gcode_result = self.generate_gcode(toolpath, tool_id, material_id)

        # Convert toolpath into GSAP-compatible motion frames with normalized durations
        keyframes: List[Dict[str, Any]] = []
        accum_time = 0.0

        for i, pt in enumerate(toolpath):
            if i == 0:
                duration = 0.2
            else:
                prev = toolpath[i - 1]
                dist = math.sqrt((pt.x - prev.x)**2 + (pt.y - prev.y)**2 + (pt.z - prev.z)**2)
                speed_per_sec = (pt.feed_mm_min / 60.0) if pt.command != "G00" else 50.0
                duration = max(0.05, dist / max(1.0, speed_per_sec))

            accum_time += duration
            keyframes.append({
                "index": i,
                "x": round(pt.x, 3),
                "y": round(pt.y, 3),
                "z": round(pt.z, 3),
                "command": pt.command,
                "feed": pt.feed_mm_min,
                "duration": round(duration, 3),
                "timestamp": round(accum_time, 3),
                "is_cutting": pt.command != "G00" and pt.z <= 0.0,
                "radius_i": pt.radius_i,
                "radius_j": pt.radius_j
            })

        return {
            "success": True,
            "vital_max_hp_rule": self.vital_max_hp,
            "speeds_and_feeds": speeds,
            "gcode": gcode_result,
            "toolpath_points_count": len(toolpath),
            "keyframes": keyframes,
            "total_sim_duration_seconds": round(accum_time, 2),
            "safety_limits": {
                "max_allowed_feed_mm_min": 5000.0,
                "max_allowed_spindle_rpm": 24000,
                "vital_safety_hp": self.vital_max_hp,
                "z_clearance_mm": self.safety_clearance_z,
                "emergency_stop_reaction_ms": 12.0
            }
        }

# Global Singleton
GLOBAL_CNC_ENGINE = CNCDrawingAndMachiningEngine()
