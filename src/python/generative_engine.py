"""
Generative Engine - Autonomous Content Creation Orchestrator

Coordinates C-level generators with ML models via OpenVINO,
integrates with guardian for scheduling and reward feedback.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import ctypes
import threading
import queue
import time
from pathlib import Path

# ============================================================
# Types
# ============================================================

class GeneratorType(Enum):
    PERLIN = auto()
    SIMPLEX = auto()
    VORONOI = auto()
    CELLULAR = auto()
    WAVE_FUNCTION = auto()
    VAE = auto()
    GAN = auto()
    DIFFUSION = auto()
    TRANSFORMER = auto()

class OutputType(Enum):
    MESH = auto()
    TEXTURE = auto()
    VOXEL = auto()
    HEIGHTMAP = auto()
    TEXT = auto()
    BEHAVIOR = auto()
    PRESET = auto()  # Performance preset

class QualityLevel(Enum):
    DRAFT = 0
    PREVIEW = 1
    PRODUCTION = 2
    ULTRA = 3

@dataclass
class LatentVector:
    """Latent space representation for generative models."""
    data: np.ndarray
    temperature: float = 1.0
    guidance_scale: float = 7.5
    seed: Optional[int] = None

    @classmethod
    def random(cls, dim: int = 512, seed: Optional[int] = None) -> 'LatentVector':
        rng = np.random.default_rng(seed)
        return cls(data=rng.standard_normal(dim).astype(np.float32), seed=seed)

    @classmethod
    def interpolate(cls, a: 'LatentVector', b: 'LatentVector', t: float) -> 'LatentVector':
        # Spherical interpolation for better results
        omega = np.arccos(np.clip(np.dot(a.data, b.data) /
                         (np.linalg.norm(a.data) * np.linalg.norm(b.data)), -1, 1))
        if omega < 1e-6:
            return cls(data=a.data * (1 - t) + b.data * t)
        so = np.sin(omega)
        return cls(data=(np.sin((1-t)*omega)/so) * a.data + (np.sin(t*omega)/so) * b.data)

    def mutate(self, rate: float = 0.1) -> 'LatentVector':
        noise = np.random.standard_normal(self.data.shape) * rate
        return LatentVector(data=self.data + noise, temperature=self.temperature)

@dataclass
class GeneratedOutput:
    """Result from generation pipeline."""
    output_type: OutputType
    data: Any
    quality_score: float
    generation_id: int
    latent: Optional[LatentVector] = None
    generation_time_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)

@dataclass
class GenerationRequest:
    """Request for content generation."""
    output_type: OutputType
    generator_type: GeneratorType
    latent: Optional[LatentVector] = None
    quality: QualityLevel = QualityLevel.PREVIEW
    prompt: Optional[str] = None
    constraints: Dict = field(default_factory=dict)
    priority: int = 5
    callback: Optional[Callable[[GeneratedOutput], None]] = None

@dataclass
class GeneratorConfig:
    """Configuration for a generator."""
    name: str
    generator_type: GeneratorType
    model_path: Optional[str] = None
    device: str = "CPU"  # CPU, GPU, NPU
    precision: str = "FP16"
    latent_dim: int = 512
    batch_size: int = 1

# ============================================================
# OpenVINO Model Wrapper
# ============================================================

class OpenVINOGenerator:
    """OpenVINO-based generative model."""

    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.model = None
        self.compiled = None
        self._load_model()

    def _load_model(self):
        if not self.config.model_path:
            return
        try:
            from openvino import Core
            core = Core()
            self.model = core.read_model(self.config.model_path)
            self.compiled = core.compile_model(self.model, self.config.device)
        except Exception as e:
            print(f"Failed to load model: {e}")

    def generate(self, latent: LatentVector) -> np.ndarray:
        if not self.compiled:
            # Fallback to noise-based generation
            return self._noise_fallback(latent)

        # Run inference
        input_tensor = latent.data.reshape(1, -1)
        result = self.compiled([input_tensor])
        return result[0]

    def _noise_fallback(self, latent: LatentVector) -> np.ndarray:
        """Procedural fallback when no ML model available."""
        rng = np.random.default_rng(latent.seed)
        # Generate based on type
        if self.config.generator_type in (GeneratorType.VAE, GeneratorType.GAN):
            # Simulate decoded output
            return rng.random((64, 64, 3)).astype(np.float32)
        elif self.config.generator_type == GeneratorType.DIFFUSION:
            # Simulate diffusion output
            return rng.random((256, 256, 3)).astype(np.float32)
        return rng.random((32, 32)).astype(np.float32)

# ============================================================
# Procedural Generators
# ============================================================

class ProceduralGenerator:
    """Classic procedural content generators."""

    @staticmethod
    def perlin_2d(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
        """2D Perlin noise implementation."""
        # Simplified Perlin noise
        np.random.seed(seed)
        perm = np.random.permutation(256)
        perm = np.tile(perm, 2)

        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        def lerp(a, b, t):
            return a + t * (b - a)

        def grad(h, x, y):
            vectors = [(1,1), (-1,1), (1,-1), (-1,-1)]
            g = vectors[h % 4]
            return g[0]*x + g[1]*y

        xi, yi = x.astype(int) & 255, y.astype(int) & 255
        xf, yf = x - x.astype(int), y - y.astype(int)
        u, v = fade(xf), fade(yf)

        aa = perm[perm[xi] + yi]
        ab = perm[perm[xi] + yi + 1]
        ba = perm[perm[xi + 1] + yi]
        bb = perm[perm[xi + 1] + yi + 1]

        x1 = lerp(grad(aa, xf, yf), grad(ba, xf-1, yf), u)
        x2 = lerp(grad(ab, xf, yf-1), grad(bb, xf-1, yf-1), u)

        return lerp(x1, x2, v)

    @staticmethod
    def fbm(x: np.ndarray, y: np.ndarray, octaves: int = 6,
            persistence: float = 0.5, seed: int = 0) -> np.ndarray:
        """Fractional Brownian Motion."""
        result = np.zeros_like(x)
        amplitude = 1.0
        frequency = 1.0

        for i in range(octaves):
            result += amplitude * ProceduralGenerator.perlin_2d(
                x * frequency, y * frequency, seed + i
            )
            amplitude *= persistence
            frequency *= 2.0

        return result

    @staticmethod
    def voronoi(x: np.ndarray, y: np.ndarray, n_points: int = 16,
                seed: int = 0) -> np.ndarray:
        """Voronoi diagram generation."""
        rng = np.random.default_rng(seed)
        points = rng.random((n_points, 2))

        # Compute distance to nearest point
        result = np.full(x.shape, np.inf)
        for px, py in points:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            result = np.minimum(result, dist)

        return result

# ============================================================
# Recipe Generator (RPG Craft Integration)
# ============================================================

class RecipeGenerator:
    """Generate performance presets as 'recipes'."""

    def __init__(self):
        self.recipes: Dict[str, Dict] = {}
        self.evolution_history: List[Dict] = []

    def generate_preset(self, latent: LatentVector,
                        constraints: Dict = None) -> GeneratedOutput:
        """Generate a performance preset from latent space."""
        # Map latent dimensions to preset parameters
        params = self._decode_latent(latent)

        # Apply constraints
        if constraints:
            params = self._apply_constraints(params, constraints)

        preset = {
            "cpu_clock_offset": int(params[0] * 200 - 100),
            "gpu_clock_offset": int(params[1] * 300 - 100),
            "power_limit_pct": int(params[2] * 40 + 60),
            "fan_curve": self._generate_fan_curve(params[3:7]),
            "memory_clock_offset": int(params[7] * 400 - 200),
            "voltage_offset": int(params[8] * 50 - 25),
            "thermal_target": int(params[9] * 20 + 70),
        }

        quality = self._evaluate_preset(preset)

        return GeneratedOutput(
            output_type=OutputType.PRESET,
            data=preset,
            quality_score=quality,
            generation_id=int(time.time() * 1000),
            latent=latent,
            metadata={"constraints": constraints}
        )

    def _decode_latent(self, latent: LatentVector) -> np.ndarray:
        """Decode latent to normalized parameter space."""
        # Use sigmoid for bounded outputs
        return 1 / (1 + np.exp(-latent.data[:16]))

    def _apply_constraints(self, params: np.ndarray,
                           constraints: Dict) -> np.ndarray:
        """Apply user constraints to parameters."""
        if "max_power" in constraints:
            params[2] = min(params[2], constraints["max_power"] / 100)
        if "max_temp" in constraints:
            params[9] = min(params[9], (constraints["max_temp"] - 70) / 20)
        return params

    def _generate_fan_curve(self, params: np.ndarray) -> List[tuple]:
        """Generate fan curve points from parameters."""
        temps = [40, 55, 70, 85]
        speeds = [int(p * 70 + 30) for p in params[:4]]
        return list(zip(temps, speeds))

    def _evaluate_preset(self, preset: Dict) -> float:
        """Heuristic quality score for preset."""
        score = 0.5
        # Prefer balanced settings
        if 80 <= preset["power_limit_pct"] <= 100:
            score += 0.2
        if 75 <= preset["thermal_target"] <= 85:
            score += 0.2
        if -50 <= preset["voltage_offset"] <= 0:
            score += 0.1
        return min(1.0, score)

    def evolve_preset(self, parent: GeneratedOutput,
                      reward: float) -> GeneratedOutput:
        """Evolve preset based on reward feedback."""
        if not parent.latent:
            return parent

        # Mutation rate inversely proportional to reward
        mutation_rate = 0.2 * (1 - reward)
        new_latent = parent.latent.mutate(mutation_rate)

        return self.generate_preset(new_latent,
                                    parent.metadata.get("constraints"))

# ============================================================
# Main Generative Engine
# ============================================================

class GenerativeEngine:
    """Main orchestrator for all generative systems."""

    def __init__(self):
        self.generators: Dict[str, OpenVINOGenerator] = {}
        self.procedural = ProceduralGenerator()
        self.recipe_gen = RecipeGenerator()
        self.request_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.results: Dict[int, GeneratedOutput] = {}
        self.experiences: List[tuple] = []  # (generation_id, reward)
        self._generation_counter = 0
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self.boost_zone_id: Optional[int] = None
        self.stats = {
            "total_generations": 0,
            "avg_time_ms": 0.0,
            "best_quality": 0.0,
        }

    def register_generator(self, name: str, config: GeneratorConfig):
        """Register a new generator."""
        self.generators[name] = OpenVINOGenerator(config)

    def start(self):
        """Start async generation worker."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """Stop async generation."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

    def generate(self, request: GenerationRequest) -> GeneratedOutput:
        """Synchronous generation."""
        return self._process_request(request)

    def generate_async(self, request: GenerationRequest) -> int:
        """Queue async generation, returns generation_id."""
        self._generation_counter += 1
        gen_id = self._generation_counter
        self.request_queue.put((10 - request.priority, gen_id, request))
        return gen_id

    def get_result(self, generation_id: int) -> Optional[GeneratedOutput]:
        """Get result of async generation."""
        return self.results.get(generation_id)

    def log_reward(self, generation_id: int, reward: float):
        """Log reward feedback for generation."""
        self.experiences.append((generation_id, reward, time.time()))

        # Update best latents
        if generation_id in self.results:
            output = self.results[generation_id]
            if output.latent and reward > 0.8:
                self._store_good_latent(output)

    def get_best_latents(self, output_type: OutputType,
                         count: int = 5) -> List[LatentVector]:
        """Get highest-rewarded latents for a type."""
        # Filter experiences by type and sort by reward
        relevant = [(gid, r) for gid, r, _ in self.experiences
                    if gid in self.results and
                    self.results[gid].output_type == output_type]
        relevant.sort(key=lambda x: x[1], reverse=True)

        return [self.results[gid].latent for gid, _ in relevant[:count]
                if self.results[gid].latent]

    def set_boost_zone(self, zone_id: int):
        """Set thread boost zone for generation tasks."""
        self.boost_zone_id = zone_id

    def _worker_loop(self):
        """Async worker loop."""
        while self._running:
            try:
                priority, gen_id, request = self.request_queue.get(timeout=0.1)
                output = self._process_request(request)
                output.generation_id = gen_id
                self.results[gen_id] = output

                if request.callback:
                    request.callback(output)

            except queue.Empty:
                continue

    def _process_request(self, request: GenerationRequest) -> GeneratedOutput:
        """Process a generation request."""
        start = time.time()

        latent = request.latent or LatentVector.random()

        if request.output_type == OutputType.PRESET:
            output = self.recipe_gen.generate_preset(latent, request.constraints)

        elif request.output_type == OutputType.HEIGHTMAP:
            size = request.constraints.get("size", 256)
            x = np.linspace(0, 4, size)
            y = np.linspace(0, 4, size)
            xx, yy = np.meshgrid(x, y)
            data = self.procedural.fbm(xx, yy, seed=latent.seed or 0)
            output = GeneratedOutput(
                output_type=OutputType.HEIGHTMAP,
                data=data,
                quality_score=0.8,
                generation_id=0,
                latent=latent
            )

        elif request.output_type == OutputType.TEXTURE:
            # Use ML generator if available
            gen_name = request.constraints.get("generator", "default")
            if gen_name in self.generators:
                data = self.generators[gen_name].generate(latent)
            else:
                # Procedural fallback
                data = self._procedural_texture(latent, request.constraints)
            output = GeneratedOutput(
                output_type=OutputType.TEXTURE,
                data=data,
                quality_score=0.7,
                generation_id=0,
                latent=latent
            )

        else:
            # Generic procedural output
            output = GeneratedOutput(
                output_type=request.output_type,
                data=np.random.random((64, 64)),
                quality_score=0.5,
                generation_id=0,
                latent=latent
            )

        output.generation_time_ms = (time.time() - start) * 1000
        self._update_stats(output)
        return output

    def _procedural_texture(self, latent: LatentVector,
                            constraints: Dict) -> np.ndarray:
        """Generate procedural texture."""
        size = constraints.get("size", 256)
        x = np.linspace(0, 8, size)
        y = np.linspace(0, 8, size)
        xx, yy = np.meshgrid(x, y)

        # Multi-layer procedural texture
        r = self.procedural.fbm(xx, yy, octaves=4, seed=latent.seed or 0)
        g = self.procedural.fbm(xx + 100, yy + 100, octaves=4, seed=(latent.seed or 0) + 1)
        b = self.procedural.voronoi(xx/8, yy/8, seed=(latent.seed or 0) + 2)

        # Normalize and stack
        r = (r - r.min()) / (r.max() - r.min() + 1e-6)
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)
        b = (b - b.min()) / (b.max() - b.min() + 1e-6)

        return np.stack([r, g, b], axis=-1).astype(np.float32)

    def _store_good_latent(self, output: GeneratedOutput):
        """Store high-quality latent for future use."""
        pass  # Would persist to experience store

    def _update_stats(self, output: GeneratedOutput):
        """Update generation statistics."""
        self.stats["total_generations"] += 1
        n = self.stats["total_generations"]
        self.stats["avg_time_ms"] = (
            self.stats["avg_time_ms"] * (n-1) + output.generation_time_ms
        ) / n
        self.stats["best_quality"] = max(
            self.stats["best_quality"], output.quality_score
        )


def create_generative_engine() -> GenerativeEngine:
    """Create and configure generative engine."""
    engine = GenerativeEngine()

    # Register default generators
    engine.register_generator("vae_texture", GeneratorConfig(
        name="vae_texture",
        generator_type=GeneratorType.VAE,
        latent_dim=256
    ))

    engine.register_generator("diffusion", GeneratorConfig(
        name="diffusion",
        generator_type=GeneratorType.DIFFUSION,
        latent_dim=512,
        device="GPU"
    ))

    return engine
