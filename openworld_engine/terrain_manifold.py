"""
Krystal-Stack Platform Framework: Open-World Terrain Manifold
=============================================================
Continuous procedural heightfield generation, multi-fractal noise
harmonics, coupled hydraulic/thermal erosion, and biome phase space
coordinates (Temperature, Moisture, Elevation).
"""

import math
import sys
from typing import Dict, Any, Tuple, List, Optional, Callable

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

Vec2 = Tuple[float, float]
Vec3 = Tuple[float, float, float]

def hash21(x: float, y: float, seed: int = 1337) -> float:
    """Deterministic 2D -> 1D pseudo-random hash in [0, 1)."""
    # Bitwise/sine-free deterministic integer hash
    xi = int(math.floor(x)) + seed
    yi = int(math.floor(y)) + seed * 31
    n = (xi * 374761393 + yi * 668265263) ^ 0x5bf03635
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return (n & 0x7fffffff) / 2147483648.0

def smoothstep(t: float) -> float:
    """Hermite quintic interpolation for zero 2nd-derivative discontinuities."""
    t_clamped = max(0.0, min(1.0, t))
    return t_clamped * t_clamped * t_clamped * (t_clamped * (t_clamped * 6.0 - 15.0) + 10.0)

def value_noise_2d(x: float, y: float, seed: int = 42) -> float:
    """Smooth 2D Value Noise with quintic Hermite filtering."""
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = x0 + 1.0
    y1 = y0 + 1.0

    tx = smoothstep(x - x0)
    ty = smoothstep(y - y0)

    v00 = hash21(x0, y0, seed)
    v10 = hash21(x1, y0, seed)
    v01 = hash21(x0, y1, seed)
    v11 = hash21(x1, y1, seed)

    vx0 = v00 + tx * (v10 - v00)
    vx1 = v01 + tx * (v11 - v01)
    return vx0 + ty * (vx1 - vx0)

def fbm_2d(
    x: float,
    y: float,
    octaves: int = 6,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    seed: int = 101
) -> float:
    """Fractal Brownian Motion (fBm) multi-octave summation."""
    total = 0.0
    frequency = 1.0
    amplitude = 1.0
    max_value = 0.0

    for i in range(octaves):
        total += value_noise_2d(x * frequency, y * frequency, seed + i * 17) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / max_value if max_value > 0.0 else 0.0

def ridged_multifractal_2d(
    x: float,
    y: float,
    octaves: int = 5,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    seed: int = 202
) -> float:
    """
    Ridged Multifractal generator producing sharp tectonic mountain ridges
    and river trench canyons via inverted absolute noise.
    """
    total = 0.0
    frequency = 1.0
    amplitude = 1.0
    weight = 1.0
    max_value = 0.0

    for i in range(octaves):
        n = value_noise_2d(x * frequency, y * frequency, seed + i * 29)
        # Invert absolute noise to create sharp ridges: 1 - |2*n - 1|
        signal = 1.0 - abs(2.0 * n - 1.0)
        signal = signal * signal  # Square signal for sharper ridges
        signal *= weight

        total += signal * amplitude
        max_value += amplitude
        weight = max(0.0, min(1.0, signal * 2.0))
        amplitude *= gain
        frequency *= lacunarity

    return total / max_value if max_value > 0.0 else 0.0

def cellular_voronoi_2d(x: float, y: float, seed: int = 303) -> Tuple[float, float]:
    """
    Worley / Cellular Voronoi noise returning (F1, F2 - F1).
    F1 is distance to closest cellular site; F2 - F1 is cell edge boundary.
    """
    xi = math.floor(x)
    yi = math.floor(y)

    min_dist_1 = 999.0
    min_dist_2 = 999.0

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx = xi + dx
            cy = yi + dy
            px = cx + hash21(cx, cy, seed)
            py = cy + hash21(cx, cy, seed + 99)

            d = math.hypot(x - px, y - py)
            if d < min_dist_1:
                min_dist_2 = min_dist_1
                min_dist_1 = d
            elif d < min_dist_2:
                min_dist_2 = d

    return min_dist_1, (min_dist_2 - min_dist_1)

class BiomePhaseSpace:
    """
    Whittaker-inspired multi-dimensional biome classification based on:
    - Temperature T in [-1.0 (Cryo/Arctic), 1.0 (Volcanic/Infernal)]
    - Moisture M in [-1.0 (Arid/Dune Barren), 1.0 (Lush/Bioluminescent Swamp)]
    - Tech/Anomaly Level A in [0.0 (Primal Nature), 1.0 (Cyberpunk/Biomechanical)]
    """
    BIOME_CATALOG = {
        "CYBERPUNK_WASTELAND": {
            "name": "Cyberpunk Neon Wasteland",
            "centroid": (0.1, -0.7, 0.95),
            "primary_color": (0.05, 0.95, 0.85),
            "ambient_fog": (0.04, 0.03, 0.08),
            "roughness": 0.65,
            "flora_density": 0.05,
            "artifact_types": ["CYBER_TURRET_MK4", "CYBERPUNK_DATA_SPIRE"]
        },
        "CRYSTALLINE_HIGHLANDS": {
            "name": "Sacred Crystalline Highlands",
            "centroid": (-0.6, 0.4, 0.6),
            "primary_color": (0.7, 0.85, 1.0),
            "ambient_fog": (0.08, 0.12, 0.22),
            "roughness": 0.85,
            "flora_density": 0.15,
            "artifact_types": ["ANCIENT_OBELISK_MONOLITH", "RETRO_SOLAR_EXPLORER"]
        },
        "VOLCANIC_CRAGS": {
            "name": "Infernal Volcanic Crags & Basalt",
            "centroid": (0.9, -0.8, 0.2),
            "primary_color": (1.0, 0.25, 0.05),
            "ambient_fog": (0.15, 0.04, 0.02),
            "roughness": 0.92,
            "flora_density": 0.0,
            "artifact_types": ["MECH_WALKER_TITAN", "ANCIENT_OBELISK_MONOLITH"]
        },
        "BIOMECHANICAL_HIVE": {
            "name": "Xenobiotic Giger Swarm Chamber",
            "centroid": (0.4, 0.8, 0.9),
            "primary_color": (0.4, 0.6, 0.2),
            "ambient_fog": (0.03, 0.07, 0.04),
            "roughness": 0.78,
            "flora_density": 0.45,
            "artifact_types": ["BIOMECHANICAL_XENODRONE"]
        },
        "ALCHEMICAL_ETHER_PLAINS": {
            "name": "Alchemical Monastic Ether Plains",
            "centroid": (-0.2, 0.1, 0.4),
            "primary_color": (0.95, 0.75, 0.2),
            "ambient_fog": (0.09, 0.07, 0.03),
            "roughness": 0.35,
            "flora_density": 0.30,
            "artifact_types": ["ANCIENT_OBELISK_MONOLITH", "CYBERPUNK_DATA_SPIRE"]
        }
    }

    @classmethod
    def classify_biome(cls, temp: float, moisture: float, anomaly: float) -> Tuple[str, Dict[str, Any], float]:
        """
        Finds the closest biome in phase space and returns (biome_id, biome_data, weight).
        Uses inverse square distance weighting.
        """
        best_id = "CYBERPUNK_WASTELAND"
        best_dist = 999.0
        weights: Dict[str, float] = {}

        for b_id, b_data in cls.BIOME_CATALOG.items():
            ct, cm, ca = b_data["centroid"]
            dist = math.sqrt((temp - ct)**2 + (moisture - cm)**2 + (anomaly - ca)**2)
            weights[b_id] = 1.0 / (dist * dist + 1e-4)
            if dist < best_dist:
                best_dist = dist
                best_id = b_id

        total_weight = sum(weights.values())
        norm_weight = weights[best_id] / total_weight if total_weight > 0.0 else 1.0
        return best_id, cls.BIOME_CATALOG[best_id], norm_weight


class TerrainManifold:
    """
    Parametric Open-World Terrain Manifold with multi-frequency procedural synthesis,
    coupled thermal/hydraulic erosion approximation, and normal tensor evaluation.
    """
    def __init__(
        self,
        base_elevation: float = 0.0,
        height_scale: float = 2.5,
        frequency: float = 0.08,
        octaves: int = 5,
        lacunarity: float = 2.05,
        persistence: float = 0.48,
        ridge_weight: float = 0.5,
        erosion_strength: float = 0.4,
        temperature_bias: float = 0.0,
        moisture_bias: float = 0.0,
        anomaly_bias: float = 0.5,
        seed: int = 42
    ):
        self.base_elevation = base_elevation
        self.height_scale = height_scale
        self.frequency = frequency
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.persistence = persistence
        self.ridge_weight = ridge_weight
        self.erosion_strength = erosion_strength
        self.temperature_bias = temperature_bias
        self.moisture_bias = moisture_bias
        self.anomaly_bias = anomaly_bias
        self.seed = seed

    def sample_raw_height(self, x: float, z: float) -> float:
        """Evaluates raw un-eroded multi-fractal heightfield at world (x, z)."""
        nx = x * self.frequency
        nz = z * self.frequency

        # 1. Base rolling hills via fBm
        h_fbm = fbm_2d(
            nx, nz,
            octaves=self.octaves,
            lacunarity=self.lacunarity,
            persistence=self.persistence,
            seed=self.seed
        )

        # 2. Alpine ridges and tectonic trenches
        if self.ridge_weight > 0.01:
            h_ridge = ridged_multifractal_2d(
                nx * 1.5, nz * 1.5,
                octaves=max(2, self.octaves - 1),
                lacunarity=self.lacunarity,
                gain=self.persistence,
                seed=self.seed + 107
            )
            h = (1.0 - self.ridge_weight) * h_fbm + self.ridge_weight * h_ridge
        else:
            h = h_fbm

        # 3. Cellular fault line modulation
        f1, f_edge = cellular_voronoi_2d(nx * 0.4, nz * 0.4, seed=self.seed + 301)
        canyon_mask = smoothstep(f_edge * 1.8)
        h = h * (0.35 + 0.65 * canyon_mask)

        return self.base_elevation + (h * 2.0 - 1.0) * self.height_scale

    def sample_height(self, x: float, z: float) -> float:
        """Alias for sample_raw_height for standard heightfield evaluation."""
        return self.sample_raw_height(x, z)

    def sample_height_and_erosion(self, x: float, z: float) -> Tuple[float, float, float]:
        """
        Samples height with thermal weathering and hydraulic channel incision.
        Returns: (eroded_height, slope_steepness, sediment_factor)
        """
        raw_h = self.sample_raw_height(x, z)
        if self.erosion_strength <= 0.01:
            return raw_h, 0.0, 0.0

        # Differential numerical gradient for slope estimation
        eps = 0.15
        hx_p = self.sample_raw_height(x + eps, z)
        hx_m = self.sample_raw_height(x - eps, z)
        hz_p = self.sample_raw_height(x, z + eps)
        hz_m = self.sample_raw_height(x, z - eps)

        dx = (hx_p - hx_m) / (2.0 * eps)
        dz = (hz_p - hz_m) / (2.0 * eps)
        slope_sq = dx * dx + dz * dz
        slope = math.sqrt(slope_sq)

        # Thermal erosion: sediment falls off steep slopes (> critical angle tan(35 deg) ~= 0.7)
        critical_angle = 0.68
        talus = max(0.0, slope - critical_angle) * 0.4

        # Hydraulic incision: water flows downhill along gradient and cuts V-channels
        # High curvature + high slope = intense sediment carry capacity
        laplacian = ((hx_p + hx_m - 2.0 * raw_h) + (hz_p + hz_m - 2.0 * raw_h)) / (eps * eps)
        incision = max(-1.0, min(1.0, laplacian * 0.15)) * self.erosion_strength

        eroded_h = raw_h - talus * self.erosion_strength + incision * 0.25
        sediment = max(0.0, min(1.0, 1.0 - slope))
        return eroded_h, slope, sediment

    def sample_normal(self, x: float, z: float) -> Vec3:
        """Computes analytical unit normal vector pointing upward from terrain surface."""
        eps = 0.12
        h1, _, _ = self.sample_height_and_erosion(x - eps, z)
        h2, _, _ = self.sample_height_and_erosion(x + eps, z)
        h3, _, _ = self.sample_height_and_erosion(x, z - eps)
        h4, _, _ = self.sample_height_and_erosion(x, z + eps)

        dx = (h2 - h1) / (2.0 * eps)
        dz = (h4 - h3) / (2.0 * eps)

        # Normal is perpendicular to (-dx, 1, -dz)
        nx = -dx
        ny = 1.0
        nz = -dz
        mag = math.hypot(nx, math.hypot(ny, nz))
        if mag > 1e-6:
            return (nx / mag, ny / mag, nz / mag)
        return (0.0, 1.0, 0.0)

    def sample_biome(self, x: float, z: float) -> Tuple[str, Dict[str, Any], float]:
        """
        Computes dynamic temperature, moisture, and anomaly phase space values
        at world (x, z) to classify the local biome.
        """
        scale = self.frequency * 0.25
        temp = value_noise_2d(x * scale, z * scale, seed=self.seed + 505) * 2.0 - 1.0 + self.temperature_bias
        moist = value_noise_2d(x * scale + 100.0, z * scale - 50.0, seed=self.seed + 606) * 2.0 - 1.0 + self.moisture_bias
        anom = value_noise_2d(x * scale * 2.0, z * scale * 2.0, seed=self.seed + 707) * 0.5 + self.anomaly_bias

        temp = max(-1.0, min(1.0, temp))
        moist = max(-1.0, min(1.0, moist))
        anom = max(0.0, min(1.0, anom))

        return BiomePhaseSpace.classify_biome(temp, moist, anom)

    def evaluate_world_point_sdf(self, px: float, py: float, pz: float) -> float:
        """
        Signed Distance Field of the continuous terrain surface:
        Negative under ground, zero on surface, positive in air.
        """
        h = self.sample_raw_height(px, pz)
        dy = py - h
        return dy * 0.75  # 0.75 Lipschitz bound factor to prevent ray overshoot

    def to_dict(self) -> Dict[str, Any]:
        """Serializes terrain parameters into a dictionary."""
        return {
            "base_elevation": self.base_elevation,
            "height_scale": self.height_scale,
            "frequency": self.frequency,
            "octaves": self.octaves,
            "lacunarity": self.lacunarity,
            "persistence": self.persistence,
            "ridge_weight": self.ridge_weight,
            "erosion_strength": self.erosion_strength,
            "temperature_bias": self.temperature_bias,
            "moisture_bias": self.moisture_bias,
            "anomaly_bias": self.anomaly_bias,
            "seed": self.seed
        }
