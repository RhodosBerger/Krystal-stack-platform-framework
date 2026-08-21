"""
Manufacturing Lexicon (The DSL) 📚⚙️
Provides high-level semantic primitives for CNC program generation.
Supports multiple versions: BASIC (Safety-First) and PREMIUM (High-Speed/Expert).
"""
import math
from typing import List, Dict, Any

class LexiconPrimitive:
    def __init__(self, name: str, params: Dict[str, Any], version: str = "BASIC"):
        self.name = name
        self.params = params
        self.version = version

class Lexicon:
    def __init__(self, version: str = "BASIC"):
        self.version = version # "BASIC" or "PREMIUM"
        self.feed_limit = 2000 if version == "PREMIUM" else 800
        self.safety_height = 5.0
        
    def rapid_to(self, x: float = 0, y: float = 0, z: float = None) -> str:
        z_val = z if z is not None else self.safety_height
        return f"G00 X{x:.3f} Y{y:.3f} Z{z_val:.3f}"

    def mill_face(self, x_start: float, y_start: float, x_end: float, y_end: float, z_depth: float, stepover: float = 2.0) -> List[str]:
        """Generates a zig-zag facing operation."""
        commands = [
            "(--- LEXICON: MILL_FACE ---)",
            self.rapid_to(x_start, y_start),
            f"G01 Z{z_depth:.3f} F{self.feed_limit / 2:.0f}"
        ]
        
        current_y = y_start
        while current_y <= y_end:
            commands.append(f"G01 X{x_end:.3f} Y{current_y:.3f} F{self.feed_limit:.0f}")
            current_y += stepover
            if current_y <= y_end:
                commands.append(f"G01 X{x_start:.3f} Y{current_y:.3f} F{self.feed_limit:.0f}")
                current_y += stepover
                
        commands.append(f"G00 Z{self.safety_height:.3f}")
        return commands

    def drill_circle(self, x: float, y: float, z_depth: float, peck: float = 2.0) -> List[str]:
        """Generates a pecking drill cycle."""
        commands = [
            "(--- LEXICON: DRILL_CIRCLE ---)",
            self.rapid_to(x, y)
        ]
        
        current_z = 0.0
        while current_z > z_depth:
            current_z = max(z_depth, current_z - peck)
            commands.append(f"G01 Z{current_z:.3f} F{self.feed_limit / 4:.0f}")
            commands.append(f"G00 Z{self.safety_height / 2:.3f}")
            commands.append(f"G00 Z{current_z + 0.5:.3f}")
            
        commands.append(f"G00 Z{self.safety_height:.3f}")
        return commands

    def get_lexicon_definitions(self) -> Dict[str, str]:
        """Returns metadata for LLM context injection."""
        return {
            "mill_face": "Z-depth facing (zig-zag). Parameters: x_start, y_start, x_end, y_end, z_depth, stepover",
            "drill_circle": "Pecking drill cycle. Parameters: x, y, z_depth, peck",
            "rapid_to": "Safe move to coordinate. Parameters: x, y, z",
            "active_version": self.version
        }

class GCodeCompiler:
    """Compiles Lexicon Primitives into Absolute (G90) G-Code programs."""
    @staticmethod
    def compile(lexicon_calls: List[List[str]], origin_uid: str = None) -> str:
        header = [
            "%",
            f"(PROGRAM: LEXICON_GEN_V1)",
            f"(ORIGIN_ID: {origin_uid})" if origin_uid else "(ORIGIN_ID: UNTRACKED)",
            "G21 (Metric)",
            "G90 (ABSOLUTE SAFETY PROTOCOL - LOCKED)",
            "M03 S1500 (Spindle ON)",
            ""
        ]
        
        footer = [
            "",
            f"(END_OF_PROGRAM - TRACE_ID: {origin_uid[:8] if origin_uid else 'N/A'})",
            "M05 (Spindle OFF)",
            "M30 (End Program)",
            "%"
        ]
        
        body = []
        for call in lexicon_calls:
            body.extend(call)
            
        return "\n".join(header + body + footer)
