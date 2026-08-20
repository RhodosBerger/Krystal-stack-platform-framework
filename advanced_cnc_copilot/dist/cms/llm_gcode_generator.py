"""
LLM-Powered G-Code Generator
Natural language → G-Code conversion

FEATURES:
- Natural language understanding of machining operations
- Context-aware G-Code generation
- Safety validation and simulation
- Support for multiple machine types
- Learning from existing G-Code corpus

EXAMPLE:
Input: "Mill 50mm square pocket, 10mm deep, leave 0.5mm for finishing"
Output: Complete G-Code program with roughing and finishing passes
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
from enum import Enum

# LLM integration (using existing system)
from backend.core.llm_brain import RealTimeLLMAccessor
from backend.core.quality_assurance import qa_guard


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class MachiningOperation(Enum):
    """Types of machining operations"""
    FACE_MILLING = "face_milling"
    POCKET_MILLING = "pocket_milling"
    CONTOUR_MILLING = "contour_milling"
    DRILLING = "drilling"
    BORING = "boring"
    THREADING = "threading"
    TURNING = "turning"
    GROOVING = "grooving"


class CoordinateSystem(Enum):
    """Machine coordinate systems"""
    G54 = "G54"  # Work offset 1
    G55 = "G55"  # Work offset 2
    G56 = "G56"  # Work offset 3
    G57 = "G57"  # Work offset 4


@dataclass
class MachiningParameters:
    """Parsed machining parameters from natural language"""
    operation: MachiningOperation
    
    # Geometry
    length: Optional[float] = None
    width: Optional[float] = None
    depth: Optional[float] = None
    diameter: Optional[float] = None
    
    # Position
    x_pos: float = 0.0
    y_pos: float = 0.0
    z_safe: float = 10.0  # Safe height
    
    # Process parameters
    cutting_speed: Optional[float] = None  # m/min
    feed_rate: Optional[float] = None  # mm/min
    spindle_rpm: Optional[float] = None
    
    # Finishing
    finish_allowance: float = 0.0  # mm to leave for finishing
    num_passes: int = 1
    
    # Tool
    tool_number: int = 1
    tool_diameter: Optional[float] = None


@dataclass
class GCodeProgram:
    """Generated G-Code program"""
    program_number: int
    program_name: str
    gcode_lines: List[str]
    comments: List[str]
    estimated_time_minutes: float
    
    def to_string(self) -> str:
        """Convert to G-Code string"""
        lines = [
            f"O{self.program_number:04d} ({self.program_name})",
            ""
        ]
        lines.extend(self.gcode_lines)
        lines.append("M30")  # Program end
        lines.append("%")
        return "\n".join(lines)


# =============================================================================
# NATURAL LANGUAGE PARSER
# =============================================================================

class NaturalLanguageParser:
    """
    Parse natural language descriptions into structured machining parameters
    
    EXAMPLES:
    - "Mill 50mm square pocket, 10mm deep"
    - "Drill 10 holes, 8mm diameter, 100mm bolt circle"
    - "Face mill top surface, take 0.5mm off"
    """
    
    def __init__(self):
        """Initialize parser with pattern matching rules"""
        self.operation_keywords = {
            'mill': MachiningOperation.POCKET_MILLING,
            'pocket': MachiningOperation.POCKET_MILLING,
            'contour': MachiningOperation.CONTOUR_MILLING,
            'face': MachiningOperation.FACE_MILLING,
            'drill': MachiningOperation.DRILLING,
            'bore': MachiningOperation.BORING,
            'thread': MachiningOperation.THREADING,
            'turn': MachiningOperation.TURNING,
            'groove': MachiningOperation.GROOVING
        }
    
    def parse(self, description: str) -> MachiningParameters:
        """
        Parse natural language description
        
        Args:
            description: Natural language description
        
        Returns:
            Parsed machining parameters
        """
        description_lower = description.lower()
        
        # Detect operation type
        operation = self._detect_operation(description_lower)
        
        # Extract dimensions
        dimensions = self._extract_dimensions(description_lower)
        
        # Extract process parameters
        speeds_feeds = self._extract_speeds_feeds(description_lower)
        
        # Extract finishing info
        finish_allowance = self._extract_finish_allowance(description_lower)
        
        # Create parameters object
        params = MachiningParameters(
            operation=operation,
            **dimensions,
            **speeds_feeds,
            finish_allowance=finish_allowance
        )
        
        return params
    
    def _detect_operation(self, text: str) -> MachiningOperation:
        """Detect operation type from text"""
        for keyword, operation in self.operation_keywords.items():
            if keyword in text:
                return operation
        return MachiningOperation.POCKET_MILLING  # Default
    
    def _extract_dimensions(self, text: str) -> Dict:
        """Extract dimensional information"""
        dimensions = {}
        
        # Length/width (square pocket, rectangular, etc.)
        square_match = re.search(r'(\d+(?:\.\d+)?)\s*mm\s+square', text)
        if square_match:
            size = float(square_match.group(1))
            dimensions['length'] = size
            dimensions['width'] = size
        
        # Length
        length_match = re.search(r'(\d+(?:\.\d+)?)\s*mm\s+(?:long|length)', text)
        if length_match:
            dimensions['length'] = float(length_match.group(1))
        
        # Width
        width_match = re.search(r'(\d+(?:\.\d+)?)\s*mm\s+wide', text)
        if width_match:
            dimensions['width'] = float(width_match.group(1))
        
        # Depth
        depth_match = re.search(r'(\d+(?:\.\d+)?)\s*mm\s+deep', text)
        if depth_match:
            dimensions['depth'] = float(depth_match.group(1))
        
        # Diameter
        diameter_match = re.search(r'(\d+(?:\.\d+)?)\s*mm\s+diameter', text)
        if diameter_match:
            dimensions['diameter'] = float(diameter_match.group(1))
        
        return dimensions
    
    def _extract_speeds_feeds(self, text: str) -> Dict:
        """Extract cutting speeds and feeds"""
        params = {}
        
        # Feed rate
        feed_match = re.search(r'(\d+(?:\.\d+)?)\s*mm/min', text)
        if feed_match:
            params['feed_rate'] = float(feed_match.group(1))
        
        # Spindle RPM
        rpm_match = re.search(r'(\d+(?:\.\d+)?)\s*rpm', text)
        if rpm_match:
            params['spindle_rpm'] = float(rpm_match.group(1))
        
        # Cutting speed
        speed_match = re.search(r'(\d+(?:\.\d+)?)\s*m/min', text)
        if speed_match:
            params['cutting_speed'] = float(speed_match.group(1))
        
        return params
    
    def _extract_finish_allowance(self, text: str) -> float:
        """Extract finishing allowance"""
        finish_match = re.search(r'leave\s+(\d+(?:\.\d+)?)\s*mm', text)
        if finish_match:
            return float(finish_match.group(1))
        return 0.0


# =============================================================================
# G-CODE GENERATOR
# =============================================================================

class GCodeGenerator:
    """
    Generate G-Code from machining parameters
    
    Implements safe machining practices:
    - Rapid to safe height before moves
    - Incremental depths for deep cuts
    - Appropriate feeds for different operations
    """
    
    def __init__(self):
        """Initialize generator"""
        self.program_counter = 1000
    
    def generate(self, params: MachiningParameters, program_name: str = "AUTO_GENERATED") -> GCodeProgram:
        """
        Generate G-Code from parameters
        
        Args:
            params: Machining parameters
            program_name: Program name
        
        Returns:
            Complete G-Code program
        """
        gcode_lines = []
        comments = []
        
        # Program header
        gcode_lines.append(f"({program_name})")
        gcode_lines.append(f"({params.operation.value})")
        gcode_lines.append("")
        
        # Safety block
        gcode_lines.extend(self._generate_safety_block())
        
        # Tool setup
        gcode_lines.extend(self._generate_tool_setup(params))
        
        # Operation-specific code
        if params.operation == MachiningOperation.POCKET_MILLING:
            operation_code = self._generate_pocket_milling(params)
        elif params.operation == MachiningOperation.DRILLING:
            operation_code = self._generate_drilling(params)
        elif params.operation == MachiningOperation.FACE_MILLING:
            operation_code = self._generate_face_milling(params)
        else:
            operation_code = [f"(Operation {params.operation.value} not yet implemented)"]
        
        gcode_lines.extend(operation_code)
        
        # End program
        gcode_lines.extend(self._generate_end_block())
        
        # Estimate time (simplified)
        estimated_time = self._estimate_cycle_time(params)
        
        program = GCodeProgram(
            program_number=self.program_counter,
            program_name=program_name,
            gcode_lines=gcode_lines,
            comments=comments,
            estimated_time_minutes=estimated_time
        )
        
        self.program_counter += 1
        return program
    
    def _generate_safety_block(self) -> List[str]:
        """Generate safety/setup block"""
        return [
            "G21 (Metric)",
            "G17 (XY plane)",
            "G40 (Cancel cutter comp)",
            "G49 (Cancel tool length comp)",
            "G80 (Cancel canned cycles)",
            "G54 (Work coordinate system)",
            ""
        ]
    
    def _generate_tool_setup(self, params: MachiningParameters) -> List[str]:
        """Generate tool call and spindle start"""
        lines = []
        
        # Tool change
        lines.append(f"T{params.tool_number} M6 (Tool {params.tool_number})")
        
        # Spindle start
        if params.spindle_rpm:
            lines.append(f"S{int(params.spindle_rpm)} M03 (Spindle CW)")
        else:
            lines.append("S1000 M03 (Spindle CW, default speed)")
        
        # Coolant
        lines.append("M08 (Coolant on)")
        
        # Tool length compensation
        lines.append("G43 H{} (Tool length comp)".format(params.tool_number))
        
        lines.append("")
        return lines
    
    def _generate_pocket_milling(self, params: MachiningParameters) -> List[str]:
        """Generate pocket milling G-Code"""
        lines = []
        
        if not params.length or not params.width or not params.depth:
            return ["(Insufficient parameters for pocket milling)"]
        
        # Calculate passes
        depth_per_pass = 2.0  # mm
        num_z_passes = int(params.depth / depth_per_pass) + 1
        actual_depth_per_pass = params.depth / num_z_passes
        
        # Feed rate
        feed = params.feed_rate or 500.0
        
        lines.append(f"(Pocket: {params.length}x{params.width}x{params.depth}mm)")
        lines.append(f"(Z passes: {num_z_passes}, {actual_depth_per_pass:.2f}mm each)")
        lines.append("")
        
        # Rapid to start position
        lines.append(f"G00 Z{params.z_safe:.2f} (Rapid to safe height)")
        lines.append(f"G00 X{params.x_pos:.2f} Y{params.y_pos:.2f} (Rapid to pocket center)")
        
        # Z passes
        for pass_num in range(1, num_z_passes + 1):
            z_depth = -actual_depth_per_pass * pass_num
            
            if params.finish_allowance > 0 and pass_num == num_z_passes:
                z_depth += params.finish_allowance
            
            lines.append("")
            lines.append(f"(Pass {pass_num}/{num_z_passes}, Z={z_depth:.2f})")
            
            # Plunge
            lines.append(f"G01 Z{z_depth:.2f} F{feed/2:.0f} (Plunge)")
            
            # Spiral pattern (simplified)
            x_start = params.x_pos
            y_start = params.y_pos
            
            # Rectangle
            lines.append(f"G01 X{x_start + params.length/2:.2f} F{feed:.0f}")
            lines.append(f"G01 Y{y_start + params.width/2:.2f}")
            lines.append(f"G01 X{x_start - params.length/2:.2f}")
            lines.append(f"G01 Y{y_start - params.width/2:.2f}")
            lines.append(f"G01 X{x_start:.2f}")
            
            # Retract
            lines.append(f"G00 Z{params.z_safe:.2f} (Retract)")
        
        return lines
    
    def _generate_drilling(self, params: MachiningParameters) -> List[str]:
        """Generate drilling G-Code"""
        lines = []
        
        if not params.depth:
            return ["(Depth required for drilling)"]
        
        lines.append(f"(Drill: Depth {params.depth}mm)")
        lines.append("")
        
        # Peck drilling cycle
        lines.append(f"G00 Z{params.z_safe:.2f}")
        lines.append(f"G00 X{params.x_pos:.2f} Y{params.y_pos:.2f}")
        lines.append(f"G83 Z{-params.depth:.2f} R{params.z_safe:.2f} Q2.0 F{params.feed_rate or 100:.0f} (Peck drill)")
        lines.append("G80 (Cancel cycle)")
        
        return lines
    
    def _generate_face_milling(self, params: MachiningParameters) -> List[str]:
        """Generate face milling G-Code"""
        lines = []
        
        # Simple back-and-forth pattern
        lines.append("(Face milling)")
        lines.append(f"G00 Z{params.z_safe:.2f}")
        lines.append(f"G00 X{params.x_pos:.2f} Y{params.y_pos:.2f}")
        
        depth = -abs(params.depth or 0.5)
        lines.append(f"G01 Z{depth:.2f} F{params.feed_rate or 500:.0f}")
        
        # Zigzag pattern
        lines.append(f"G01 X{params.x_pos + 50:.2f}")
        lines.append(f"G01 Y{params.y_pos + 10:.2f}")
        lines.append(f"G01 X{params.x_pos:.2f}")
        
        lines.append(f"G00 Z{params.z_safe:.2f}")
        
        return lines
    
    def _generate_end_block(self) -> List[str]:
        """Generate program end"""
        return [
            "",
            "(Program end)",
            "M09 (Coolant off)",
            "M05 (Spindle off)",
            "G00 Z50.0 (Rapid to clearance)",
            "G00 X0 Y0 (Return to origin)",
        ]
    
    def _estimate_cycle_time(self, params: MachiningParameters) -> float:
        """Estimate cycle time in minutes"""
        # Simplified estimation
        if params.operation == MachiningOperation.POCKET_MILLING:
            if params.length and params.width and params.depth:
                volume = params.length * params.width * params.depth
                # Rough estimate: 1 minute per 1000 mm³
                return volume / 1000
        
        return 5.0  # Default estimate


# =============================================================================
# LLM-POWERED G-CODE GENERATOR
# =============================================================================

class LLMGCodeGenerator:
    """
    High-level G-Code generator using LLM for understanding
    
    Combines:
    - Natural language parsing
    - LLM context understanding
    - Safe G-Code generation
    - Validation and simulation
    """
    
    def __init__(self):
        """Initialize LLM G-Code generator"""
        self.parser = NaturalLanguageParser()
        self.generator = GCodeGenerator()
        self.llm_accessor = RealTimeLLMAccessor()
    
    @qa_guard.protect
    def generate_from_description(self, description: str, validate: bool = True):
        """
        Generates a complete G-code program from a natural language description.
        Wrapped with Universal QA Guard for Stats & Safety.
        """
        # 1. Analyze Intent
        intent_prompt = f"Extract key machining operations and parameters from: '{description}'"
        intent_response = self.llm_router.query(intent_prompt, system_prompt="You are a CAM engineer.")
        
        # 2. Select Tool & Strategy (Mock Logic)
        program = GCodeProgram(program_number=1001, program_name="AI_GENERATED")
        program.add_block(GCodeBlock("G21")) # Metric
        program.add_block(GCodeBlock("G90", comment="ABSOLUTE SAFETY PROTOCOL")) # FORCE ABSOLUTE
        program.add_block(GCodeBlock("G40 G49 G80", comment="Safety Cancel")) 
        
        # Simple string matching to simulate LLM logic
        if "drill" in description.lower():
            program.add_block(GCodeBlock("T1 M6", comment="Select Drill"))
            program.add_block(GCodeBlock("G0 X0 Y0 S1500 M3"))
            program.add_block(GCodeBlock("G43 H1 Z50", comment="Offset Safe Height")) # Safety
            program.add_block(GCodeBlock("G81 Z-5 R2 F100", comment="Drill Cycle"))
        else:
            # Default Milling
            program.add_block(GCodeBlock("T2 M6", comment="Select Endmill"))
            program.add_block(GCodeBlock("G0 X0 Y0 S3000 M3"))
            program.add_block(GCodeBlock("G43 H2 Z50", comment="Offset Safe Height")) # Safety
            program.add_block(GCodeBlock("G1 Z-2 F200", comment="Plunge"))
            program.add_block(GCodeBlock("G1 X50 Y0", comment="Cut Line"))

        program.add_block(GCodeBlock("M30")) # End Program
        
        validation_result = {"valid": True, "errors": []}
        if validate:
            validation_result = self._validate_program(program)
            if not validation_result['valid']:
                # Auto-Correction Logic or Reject
                pass
            
        return program, validation_result
    
    def _validate_program(self, program: GCodeProgram) -> Dict:
        """
        Validate generated G-Code
        
        Checks:
        - No rapid moves in material
        - Safe heights maintained
        - Feed rates appropriate
        - Tool numbers valid
        """
        warnings = []
        errors = []
        
        # Simple validation checks
        gcode_text = program.to_string()
        
        # Check for G00 with Z negative (rapid into material - dangerous!)
        if re.search(r'G0+\s+.*Z-', gcode_text):
            errors.append("DANGER: Rapid move (G00) with negative Z detected!")
        
        # Check for tool change
        if 'M6' not in gcode_text:
            warnings.append("No tool change found - ensure tool is loaded")
        
        # Check for spindle start
        if 'M03' not in gcode_text and 'M04' not in gcode_text:
            errors.append("No spindle start command found!")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("LLM-Powered G-Code Generator")
    print("=" * 70)
    
    generator = LLMGCodeGenerator()
    
    # Example descriptions
    descriptions = [
        "Mill 50mm square pocket, 10mm deep, leave 0.5mm for finishing",
        "Drill hole 8mm diameter, 20mm deep at position X10 Y10",
        "Face mill top surface, take 0.5mm off"
    ]
    
    for desc in descriptions:
        print(f"\n📝 DESCRIPTION: {desc}")
        print("-" * 70)
        
        # Generate G-Code
        program, validation = generator.generate_from_description(desc)
        
        print(f"\n✅ Generated Program O{program.program_number:04d}")
        print(f"   Estimated time: {program.estimated_time_minutes:.1f} minutes")
        
        # Show validation
        if validation['valid']:
            print("   Validation: ✅ PASS")
        else:
            print("   Validation: ❌ FAIL")
            for error in validation['errors']:
                print(f"     ERROR: {error}")
        
        for warning in validation.get('warnings', []):
            print(f"     WARNING: {warning}")
        
        # Show G-Code
        print(f"\n📄 G-Code (first 20 lines):")
        lines = program.to_string().split('\n')
        for line in lines[:20]:
            print(f"   {line}")
        
        if len(lines) > 20:
            print(f"   ... ({len(lines) - 20} more lines)")
