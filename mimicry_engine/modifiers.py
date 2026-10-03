"""
Krystal-Stack Platform Framework: Blender Modifier Core Engine
==============================================================
Inherits Blender's modifier stack paradigm for procedural object mimicry,
including Array, Mirror, Boolean, Bevel, Displace, Deform, and Solidify.
"""

import math
from typing import Dict, Any, List, Optional, Callable, Tuple
from mimicry_engine.primitives import (
    Vec3, v_add, v_sub, v_scale, v_len, v_rot_x, v_rot_y, v_rot_z,
    smin, smax, smooth_difference
)

class Modifier:
    """Abstract base class for all procedural modifiers."""
    def __init__(self, name: str = "Modifier", enabled: bool = True):
        self.name = name
        self.enabled = enabled

    def apply_transform(self, p: Vec3) -> Vec3:
        """Transforms spatial coordinates before SDF evaluation."""
        return p

    def evaluate(self, p: Vec3, base_sdf: Callable[[Vec3], float]) -> float:
        """Evaluates the modified signed distance value at point p."""
        if not self.enabled:
            return base_sdf(p)
        return base_sdf(self.apply_transform(p))

    def to_dict(self) -> Dict[str, Any]:
        """Serializes modifier parameters for templates and Vulkan uniforms."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "enabled": self.enabled
        }


# ─── 1. Array Modifier (Linear & Radial Repetition) ───────────────────────────

class ArrayModifier(Modifier):
    """
    Duplicates an object along linear axes or around a radial center,
    mirroring Blender's Array Modifier with constant/relative offset.
    """
    def __init__(
        self,
        name: str = "Array",
        count: int = 3,
        offset: Vec3 = (1.5, 0.0, 0.0),
        radial: bool = False,
        radial_axis: str = "y",
        enabled: bool = True
    ):
        super().__init__(name, enabled)
        self.count = max(1, count)
        self.offset = offset
        self.radial = radial
        self.radial_axis = radial_axis.lower()

    def evaluate(self, p: Vec3, base_sdf: Callable[[Vec3], float]) -> float:
        if not self.enabled or self.count <= 1:
            return base_sdf(p)

        min_d = float('inf')

        if self.radial:
            # Radial array around axis
            angle_step = (2.0 * math.pi) / self.count
            for i in range(self.count):
                theta = i * angle_step
                if self.radial_axis == "y":
                    pi = v_rot_y(p, -theta)
                elif self.radial_axis == "x":
                    pi = v_rot_x(p, -theta)
                else:
                    pi = v_rot_z(p, -theta)
                pi = v_sub(pi, self.offset)
                d = base_sdf(pi)
                min_d = min(min_d, d)
        else:
            # Linear array centered or forward
            for i in range(self.count):
                shift = v_scale(self.offset, float(i) - (self.count - 1) * 0.5)
                pi = v_sub(p, shift)
                d = base_sdf(pi)
                min_d = min(min_d, d)

        return min_d

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "count": self.count,
            "offset": list(self.offset),
            "radial": self.radial,
            "radial_axis": self.radial_axis
        })
        return d


# ─── 2. Mirror Modifier (Bilateral / Quadrant Symmetry) ──────────────────────

class MirrorModifier(Modifier):
    """
    Mirrors object across X, Y, or Z coordinate planes with bisect offset,
    mirroring Blender's Mirror Modifier.
    """
    def __init__(
        self,
        name: str = "Mirror",
        use_x: bool = True,
        use_y: bool = False,
        use_z: bool = False,
        offset: Vec3 = (0.0, 0.0, 0.0),
        enabled: bool = True
    ):
        super().__init__(name, enabled)
        self.use_x = use_x
        self.use_y = use_y
        self.use_z = use_z
        self.offset = offset

    def apply_transform(self, p: Vec3) -> Vec3:
        px = abs(p[0] - self.offset[0]) + self.offset[0] if self.use_x else p[0]
        py = abs(p[1] - self.offset[1]) + self.offset[1] if self.use_y else p[1]
        pz = abs(p[2] - self.offset[2]) + self.offset[2] if self.use_z else p[2]
        return (px, py, pz)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "use_x": self.use_x,
            "use_y": self.use_y,
            "use_z": self.use_z,
            "offset": list(self.offset)
        })
        return d


# ─── 3. Boolean Modifier (CSG with Smooth Blending) ──────────────────────────

class BooleanModifier(Modifier):
    """
    Constructive Solid Geometry (CSG) modifier combining base shape with an
    operand shape via Union, Difference, or Intersect with adjustable smooth blend (smin).
    """
    UNION = "UNION"
    DIFFERENCE = "DIFFERENCE"
    INTERSECT = "INTERSECT"

    def __init__(
        self,
        name: str = "Boolean",
        operation: str = UNION,
        operand_sdf: Optional[Callable[[Vec3], float]] = None,
        smooth_k: float = 0.15,
        enabled: bool = True
    ):
        super().__init__(name, enabled)
        self.operation = operation.upper()
        self.operand_sdf = operand_sdf
        self.smooth_k = smooth_k

    def evaluate(self, p: Vec3, base_sdf: Callable[[Vec3], float]) -> float:
        d_base = base_sdf(p)
        if not self.enabled or self.operand_sdf is None:
            return d_base

        d_op = self.operand_sdf(p)

        if self.operation == self.UNION:
            return smin(d_base, d_op, self.smooth_k)
        elif self.operation == self.DIFFERENCE:
            return smooth_difference(d_base, d_op, self.smooth_k)
        elif self.operation == self.INTERSECT:
            return smax(d_base, d_op, self.smooth_k)
        else:
            return d_base

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "operation": self.operation,
            "smooth_k": self.smooth_k
        })
        return d


# ─── 4. Bevel Modifier (Edge Chamfer / Radius) ───────────────────────────────

class BevelModifier(Modifier):
    """Bevels outer edges of any 3D SDF shape by an offset radius."""
    def __init__(self, name: str = "Bevel", radius: float = 0.08, enabled: bool = True):
        super().__init__(name, enabled)
        self.radius = max(0.0, radius)

    def evaluate(self, p: Vec3, base_sdf: Callable[[Vec3], float]) -> float:
        if not self.enabled or self.radius <= 0.0:
            return base_sdf(p)
        return base_sdf(p) - self.radius

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["radius"] = self.radius
        return d


# ─── 5. Displace Modifier (Procedural Harmonic Noise) ────────────────────────

class DisplaceModifier(Modifier):
    """
    Displaces surface along normals using procedural harmonic frequency waves,
    mimicking Blender's Displace modifier with procedural clouds/wood/voronoi.
    """
    def __init__(
        self,
        name: str = "Displace",
        strength: float = 0.08,
        frequency: float = 4.0,
        harmonic_octaves: int = 2,
        enabled: bool = True
    ):
        super().__init__(name, enabled)
        self.strength = strength
        self.frequency = frequency
        self.octaves = harmonic_octaves

    def evaluate(self, p: Vec3, base_sdf: Callable[[Vec3], float]) -> float:
        d = base_sdf(p)
        if not self.enabled or abs(self.strength) < 1e-6:
            return d

        # Evaluate procedural 3D harmonic displacement
        disp = 0.0
        amp = 1.0
        freq = self.frequency
        for _ in range(self.octaves):
            disp += amp * math.sin(p[0] * freq) * math.cos(p[1] * freq) * math.sin(p[2] * freq)
            amp *= 0.5
            freq *= 2.0

        return d + self.strength * disp

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "strength": self.strength,
            "frequency": self.frequency,
            "octaves": self.octaves
        })
        return d


# ─── 6. Simple Deform Modifier (Twist, Taper, Bend) ──────────────────────────

class DeformModifier(Modifier):
    """
    Non-linear spatial deformation: Twist, Taper, or Bend along an axis,
    matching Blender's SimpleDeform modifier.
    """
    TWIST = "TWIST"
    TAPER = "TAPER"
    BEND = "BEND"

    def __init__(
        self,
        name: str = "Deform",
        deform_type: str = TWIST,
        factor: float = 0.5,
        axis: str = "y",
        enabled: bool = True
    ):
        super().__init__(name, enabled)
        self.deform_type = deform_type.upper()
        self.factor = factor
        self.axis = axis.lower()

    def apply_transform(self, p: Vec3) -> Vec3:
        if not self.enabled or abs(self.factor) < 1e-6:
            return p

        x, y, z = p[0], p[1], p[2]

        if self.deform_type == self.TWIST:
            # Twist angle increases with height along Y
            theta = y * self.factor
            c, s = math.cos(theta), math.sin(theta)
            return (x * c - z * s, y, x * s + z * c)

        elif self.deform_type == self.TAPER:
            # Scale x and z as a function of y
            scale = max(0.05, 1.0 + y * self.factor)
            return (x / scale, y, z / scale)

        elif self.deform_type == self.BEND:
            # Bend around X axis into a curve
            theta = z * self.factor
            c, s = math.cos(theta), math.sin(theta)
            return (x, y * c - z * s, y * s + z * c)

        return p

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "deform_type": self.deform_type,
            "factor": self.factor,
            "axis": self.axis
        })
        return d


# ─── 7. Solidify Modifier (Shell Thickness) ──────────────────────────────────

class SolidifyModifier(Modifier):
    """Hollows an SDF volume into a solid shell with wall thickness."""
    def __init__(self, name: str = "Solidify", thickness: float = 0.08, enabled: bool = True):
        super().__init__(name, enabled)
        self.thickness = max(0.001, thickness)

    def evaluate(self, p: Vec3, base_sdf: Callable[[Vec3], float]) -> float:
        if not self.enabled:
            return base_sdf(p)
        return abs(base_sdf(p)) - self.thickness

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["thickness"] = self.thickness
        return d


# ─── Modifier Stack Container (Directed Acyclic Evaluation) ──────────────────

class ModifierStack:
    """Chains multiple modifiers and evaluates them in linear DAG order."""
    def __init__(self, base_sdf: Callable[[Vec3], float]):
        self.base_sdf = base_sdf
        self.modifiers: List[Modifier] = []

    def add_modifier(self, mod: Modifier) -> "ModifierStack":
        self.modifiers.append(mod)
        return self

    def evaluate(self, p: Vec3) -> float:
        """Evaluates the stack from first to last modifier."""
        current_sdf = self.base_sdf
        for mod in self.modifiers:
            if not mod.enabled:
                continue
            # Wrap current_sdf into the modifier's evaluation
            prev_sdf = current_sdf
            current_sdf = (lambda m, s: (lambda pt: m.evaluate(pt, s)))(mod, prev_sdf)
        return current_sdf(p)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self.modifiers]
