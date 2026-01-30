# XYZ Dimensional Scaling API
# Backend support for coordinate transformations

"""
Paradigm: Coordinate Systems as Map Projections

Think of different coordinate systems like map projections:
- Mercator (distorts size near poles) = non-uniform scaling
- Equirectangular (preserves angles) = preserves proportions
- Robinson (balanced distortion) = balanced scaling

Types of Coordinates in CNC:
1. Machine Coordinates (absolute world position)
   - Like GPS coordinates (fixed reference)
   - Never changes regardless of workpiece
   
2. Work Coordinates (relative to workpiece)
   - Like street address (relative to city)
   - Changes when you set work offset (G54-G59)
   
3. Tool Coordinates (relative to tool tip)
   - Like measuring from your outstretched hand
   - Changes with tool length offset

Analogy: Nested Coordinate Systems
- Universe = Machine Coordinates
- Country = Work Coordinate System
- City = Specific G-code program
- Building = Tool offset
- Room = Current position
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import math
import numpy as np

@dataclass
class Vector3D:
    """
    3D Point in space
    Analogy: GPS coordinates (latitude, longitude, altitude)
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def __add__(self, other):
        """Vector addition (displacement)"""
        return Vector3D(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z
        )
    
    def __sub__(self, other):
        """Vector subtraction (distance between points)"""
        return Vector3D(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z
        )
    
    def __mul__(self, scalar):
        """Scalar multiplication (uniform scaling)"""
        return Vector3D(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar
        )
    
    def magnitude(self):
        """
        Vector magnitude (distance from origin)
        Analogy: Odometer reading
        """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        """
        Unit vector (direction only, length = 1)
        Analogy: Compass heading (direction without distance)
        """
        mag = self.magnitude()
        if mag == 0:
            return Vector3D()
        return Vector3D(
            self.x / mag,
            self.y / mag,
            self.z / mag
        )

@dataclass
class Transform3D:
    """
    3D Transformation (scale, rotate, translate)
    Analogy: Photo editing (resize, rotate, move)
    """
    
    # Scale factors (like zoom)
    scale: Vector3D = Vector3D(1, 1, 1)
    
    # Translation (like moving picture)
    translation: Vector3D = Vector3D(0, 0, 0)
    
    # Rotation angles in degrees (like turning picture)
    rotation: Vector3D = Vector3D(0, 0, 0)
    
    def get_scale_matrix(self):
        """
        Scale transformation matrix
        Analogy: Zoom lens setting
        
        [Sx  0   0 ]
        [0   Sy  0 ]
        [0   0   Sz]
        """
        return np.array([
            [self.scale.x, 0, 0],
            [0, self.scale.y, 0],
            [0, 0, self.scale.z]
        ])
    
    def get_rotation_matrix_x(self):
        """
        Rotation around X-axis
        Analogy: Rotating door on vertical hinge
        """
        angle = math.radians(self.rotation.x)
        return np.array([
            [1, 0, 0],
            [0, math.cos(angle), -math.sin(angle)],
            [0, math.sin(angle), math.cos(angle)]
        ])
    
    def get_rotation_matrix_y(self):
        """
        Rotation around Y-axis
        Analogy: Spinning top
        """
        angle = math.radians(self.rotation.y)
        return np.array([
            [math.cos(angle), 0, math.sin(angle)],
            [0, 1, 0],
            [-math.sin(angle), 0, math.cos(angle)]
        ])
    
    def get_rotation_matrix_z(self):
        """
        Rotation around Z-axis
        Analogy: Clock hands rotating
        """
        angle = math.radians(self.rotation.z)
        return np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1]
        ])
    
    def get_combined_matrix(self):
        """
        Combined transformation matrix
        Analogy: Recipe - order matters!
        1. Scale (resize)
        2. Rotate (turn)
        3. Translate (move)
        """
        # Combine rotations
        rotation = (
            self.get_rotation_matrix_z() @
            self.get_rotation_matrix_y() @
            self.get_rotation_matrix_x()
        )
        
        # Apply scale then rotation
        return rotation @ self.get_scale_matrix()
    
    def apply(self, point: Vector3D) -> Vector3D:
        """
        Apply transformation to a point
        Analogy: Photo editing pipeline
        """
        # Convert to numpy array
        p = np.array([point.x, point.y, point.z])
        
        # Apply transformation
        transformed = self.get_combined_matrix() @ p
        
        # Add translation
        result = transformed + np.array([
            self.translation.x,
            self.translation.y,
            self.translation.z
        ])
        
        return Vector3D(result[0], result[1], result[2])

class CoordinateSystem:
    """
    Coordinate system manager
    Analogy: Multi-layered map system
    """
    
    def __init__(self):
        # Machine zero (absolute origin - like prime meridian)
        self.machine_zero = Vector3D(0, 0, 0)
        
        # Work offset (like city center as reference)
        self.work_offset = Vector3D(0, 0, 0)
        
        # Tool length offset (like measuring stick length)
        self.tool_offset = Vector3D(0, 0, 0)
        
        # Current transformation
        self.transform = Transform3D()
    
    def machine_to_work(self, machine_pos: Vector3D) -> Vector3D:
        """
        Convert machine coordinates to work coordinates
        Analogy: Convert GPS to street address
        """
        return machine_pos - self.work_offset
    
    def work_to_machine(self, work_pos: Vector3D) -> Vector3D:
        """
        Convert work coordinates to machine coordinates
        Analogy: Convert street address to GPS
        """
        return work_pos + self.work_offset
    
    def apply_scaling(self, point: Vector3D, scale: Vector3D) -> Vector3D:
        """
        Apply non-uniform scaling
        Analogy: Stretching taffy in different directions
        """
        return Vector3D(
            point.x * scale.x,
            point.y * scale.y,
            point.z * scale.z
        )
    
    def apply_mirror(self, point: Vector3D, axis: str) -> Vector3D:
        """
        Mirror across axis
        Analogy: Reflection in mirror
        """
        mirrored = Vector3D(point.x, point.y, point.z)
        
        if 'x' in axis.lower():
            mirrored.x = -mirrored.x
        if 'y' in axis.lower():
            mirrored.y = -mirrored.y
        if 'z' in axis.lower():
            mirrored.z = -mirrored.z
        
        return mirrored
    
    def interpolate_linear(self, start: Vector3D, end: Vector3D, t: float) -> Vector3D:
        """
        Linear interpolation between two points
        Analogy: Walking in straight line from A to B
        t = 0.0 → start position
        t = 0.5 → halfway
        t = 1.0 → end position
        """
        return Vector3D(
            start.x + (end.x - start.x) * t,
            start.y + (end.y - start.y) * t,
            start.z + (end.z - start.z) * t
        )
    
    def interpolate_circular(self, center: Vector3D, radius: float, 
                            start_angle: float, end_angle: float, 
                            t: float, plane: str = 'XY') -> Vector3D:
        """
        Circular interpolation (arc movement)
        Analogy: Driving around roundabout
        """
        angle = start_angle + (end_angle - start_angle) * t
        angle_rad = math.radians(angle)
        
        if plane == 'XY':
            return Vector3D(
                center.x + radius * math.cos(angle_rad),
                center.y + radius * math.sin(angle_rad),
                center.z
            )
        elif plane == 'XZ':
            return Vector3D(
                center.x + radius * math.cos(angle_rad),
                center.y,
                center.z + radius * math.sin(angle_rad)
            )
        elif plane == 'YZ':
            return Vector3D(
                center.x,
                center.y + radius * math.cos(angle_rad),
                center.z + radius * math.sin(angle_rad)
            )

"""
Scaling Strategies in Manufacturing

1. Uniform Scaling (proportional)
   - All axes scaled equally
   - Maintains shape (circle stays circle)
   - Like photocopier zoom
   - Example: Scale entire part by 110% for shrinkage compensation

2. Non-Uniform Scaling (anamorphic)
   - Different scale per axis
   - Changes proportions (circle becomes ellipse)
   - Like funhouse mirror
   - Example: Compensate for tool deflection in one axis

3. Adaptive Scaling
   - Scale varies by position
   - Like curved space
   - Example: Thermal expansion compensation (hotter areas expand more)

Real-World Use Cases:
- Material shrinkage (casting, injection molding)
- Thermal expansion/contraction
- Tool deflection compensation
- Mirror part creation (left/right hand parts)
- Metric/Imperial conversion
- Tolerance compensation
"""

# Example usage
if __name__ == "__main__":
    # Create coordinate system (like setting up workshop)
    cs = CoordinateSystem()
    
    # Define work offset (set workpiece location)
    cs.work_offset = Vector3D(100, 100, 50)
    
    # Original point in work coordinates (where we want to cut)
    work_point = Vector3D(25, 30, 5)
    
    # Convert to machine coordinates (absolute position)
    machine_point = cs.work_to_machine(work_point)
    print(f"Machine position: {machine_point}")
    
    # Apply scaling (compensate for shrinkage)
    scale_factor = Vector3D(1.05, 1.05, 1.0)  # 5% larger in XY
    scaled_point = cs.apply_scaling(work_point, scale_factor)
    print(f"Scaled point: {scaled_point}")
    
    # Create transformation (combo of operations)
    transform = Transform3D(
        scale=Vector3D(2, 2, 2),  # Double size
        rotation=Vector3D(0, 0, 45),  # Rotate 45° around Z
        translation=Vector3D(10, 10, 0)  # Move 10mm in X and Y
    )
    
    # Apply transformation
    origin = Vector3D(0, 0, 0)
    transformed = transform.apply(origin)
    print(f"Transformed origin: {transformed}")
    
    # Linear interpolation (straight line movement)
    start = Vector3D(0, 0, 0)
    end = Vector3D(100, 100, 50)
    
    print("\nLinear interpolation (G01):")
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        point = cs.interpolate_linear(start, end, t)
        print(f"  t={t}: {point}")
    
    # Circular interpolation (arc movement)
    center = Vector3D(50, 50, 0)
    radius = 25
    
    print("\nCircular interpolation (G02/G03):")
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        point = cs.interpolate_circular(center, radius, 0, 90, t, 'XY')
        print(f"  t={t}: {point}")
