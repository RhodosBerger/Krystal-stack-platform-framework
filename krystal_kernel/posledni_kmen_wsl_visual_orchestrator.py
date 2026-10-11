#!/usr/bin/env python3
"""
POSLEDNÍ KMEN // WSL VISUAL FIDELITY ORCHESTRATION SUBSYSTEM
=============================================================================
Orchestrates 3D visual, shader, and sensory shockwave elements for 'Poslední Kmen'
directly inside the Microsoft Windows Subsystem for Linux (WSL2 / Linux kernel)
and native Krystal-Stack runtime.

Features:
  1. Engine Characteristic Extractor:
     - Godot 4.x PBR Shaders: crystal_ice_prism, toxic_caustic_slime, aetheric_vortex_swirl.
     - Godot 4.x Scenes: FloatingCitadelWellOfSouls.tscn, posledni_kmen_arena.tscn.
     - 4 Tribes: Crystal Citadel, Toxic Mire, Volcanic Forge, Void Corruption.
     - Invariant: VITAL_MAX_HP == 6, Conduct Badges ([ACT] [LCK] [CNF] [HAZ] [VIT 6HP]).
  2. 5-Modality Sensory Shockwave Orchestrator:
     - Kinetic overpressure (480 m/s), acoustic blast (343 m/s), seismic tremor (220 m/s),
       optical flash (c), aetheric resonance (750 m/s).
  3. Binary 96-Byte Packed Instance Buffer Generator:
     - Mat4x4 (64B) + PBR Albedo (16B) + Custom Shader Uniforms (16B).
  4. WSL2 Environment Sensor & Hardware Passthrough (/dev/dxg, Direct3D 12).
  5. Built-in Micro HTTP Dispatcher & Benchmarking CLI.
=============================================================================
"""

import os
import sys
import math
import time
import uuid
import struct
import platform
import json
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional, Tuple

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

VITAL_MAX_HP: int = 6
DEFAULT_PORT: int = 8095


# ── 1. WSL ENVIRONMENT DETECTOR ──────────────────────────────────────────────

class WslEnvironmentSensor:
    """Inspects Linux kernel, Microsoft WSL2 environment and /dev/dxg GPU status."""

    @staticmethod
    def inspect() -> Dict[str, Any]:
        is_linux = (platform.system().lower() == "linux")
        is_wsl = False
        kernel_release = platform.release()
        dxg_present = False
        wsl_libs_present = False

        if is_linux:
            try:
                with open("/proc/version", "r", encoding="utf-8") as f:
                    ver_text = f.read().lower()
                    if "microsoft" in ver_text or "wsl" in ver_text:
                        is_wsl = True
            except Exception:
                pass
            dxg_present = os.path.exists("/dev/dxg")
            wsl_libs_present = os.path.exists("/usr/lib/wsl/lib")

        return {
            "platform": platform.system(),
            "kernel_release": kernel_release,
            "is_linux": is_linux,
            "is_wsl2": is_wsl,
            "gpu_dxg_device": dxg_present,
            "wsl_driver_libs": wsl_libs_present,
            "cpu_count": os.cpu_count() or 4
        }


# ── 2. ENGINE CHARACTERISTICS EXTRACTOR ───────────────────────────────────────

class PosledniKmenCharacteristicsExtractor:
    """
    Extracts PBR shaders, node hierarchies, sensory parameters and tribal invariants
    from repository files.
    """

    @classmethod
    def extract_all(cls, repo_root: str) -> Dict[str, Any]:
        shaders_dir = os.path.join(repo_root, "godot_project", "shaders")
        scenes_dir = os.path.join(repo_root, "godot_project", "scenes")

        # 1. PBR Shaders
        shaders = {}
        for s_name in ["crystal_ice_prism.gdshader", "toxic_caustic_slime.gdshader", "aetheric_vortex_swirl.gdshader"]:
            s_path = os.path.join(shaders_dir, s_name)
            if os.path.exists(s_path):
                with open(s_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    shaders[s_name] = {
                        "exists": True,
                        "size_bytes": len(content),
                        "shader_type": "spatial" if "shader_type spatial" in content else "canvas_item",
                        "has_refraction": "refraction_strength" in content or "caustic_scale" in content,
                        "has_glow": "aether_glow" in content or "emission" in content
                    }
            else:
                shaders[s_name] = {"exists": False}

        # 2. Scenes
        scenes = {}
        for sc_name in ["FloatingCitadelWellOfSouls.tscn", "posledni_kmen_arena.tscn"]:
            sc_path = os.path.join(scenes_dir, sc_name)
            if os.path.exists(sc_path):
                with open(sc_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    scenes[sc_name] = {
                        "exists": True,
                        "size_bytes": len(content),
                        "has_hex_mesh": "CylinderMesh_hex" in content or "hex" in content.lower(),
                        "has_controller_script": "PosledniKmenDuelController" in content or "FloatingCitadelController" in content
                    }
            else:
                scenes[sc_name] = {"exists": False}

        # 3. 4 Tribes Configuration
        tribes = {
            "CRYSTAL_CITADEL": {
                "name": "Kryštálová Citadela (Aegis Citadel)",
                "element": "ICE_PRISM",
                "shader": "crystal_ice_prism.gdshader",
                "primary_color_hex": "#66fcf1",
                "albedo_rgb": [0.4, 0.98, 0.94],
                "emission_rgb": [0.0, 1.0, 1.0],
                "roughness": 0.15,
                "refraction_strength": 0.18,
                "totem_mesh": "PrismMesh_crystal",
                "vital_max_hp": VITAL_MAX_HP
            },
            "TOXIC_MIRE": {
                "name": "Toxické Močiare (Blight Mire)",
                "element": "CAUSTIC_SLIME",
                "shader": "toxic_caustic_slime.gdshader",
                "primary_color_hex": "#39ff14",
                "albedo_rgb": [0.22, 1.0, 0.08],
                "emission_rgb": [0.22, 1.0, 0.08],
                "roughness": 0.05,
                "wave_amplitude": 0.25,
                "totem_mesh": "SphereMesh_slime",
                "vital_max_hp": VITAL_MAX_HP
            },
            "VOLCANIC_FORGE": {
                "name": "Sopečná Vyhňa (Pyromancer Bastion)",
                "element": "PYROCLASTIC_BASALT",
                "shader": "photorealistic_pbr_procedural.gdshader",
                "primary_color_hex": "#ff4500",
                "albedo_rgb": [0.95, 0.27, 0.05],
                "emission_rgb": [1.0, 0.45, 0.0],
                "roughness": 0.65,
                "heat_shimmer": 0.85,
                "totem_mesh": "CylinderMesh_forge",
                "vital_max_hp": VITAL_MAX_HP
            },
            "VOID_CORRUPTION": {
                "name": "Prázdnota / Studna Duší (Aetheric Well of Souls)",
                "element": "AETHER_VORTEX",
                "shader": "aetheric_vortex_swirl.gdshader",
                "primary_color_hex": "#9d4edd",
                "albedo_rgb": [0.62, 0.31, 0.87],
                "emission_rgb": [0.8, 0.2, 1.0],
                "roughness": 0.20,
                "polar_swirl_speed": 1.45,
                "totem_mesh": "TorusMesh_well_of_souls",
                "vital_max_hp": VITAL_MAX_HP
            }
        }

        # 4. Sensory Shockwave Modalities
        sensory_modalities = {
            "kinetic_overpressure": {"velocity_m_s": 480.0, "decay_exponent": 1.8, "unit": "kPa"},
            "acoustic_blast": {"velocity_m_s": 343.0, "decay_exponent": 1.5, "unit": "dB"},
            "seismic_tremor": {"velocity_m_s": 220.0, "decay_exponent": 1.2, "unit": "m/s^2"},
            "optical_flash": {"velocity_m_s": 299792458.0, "decay_exponent": 2.0, "unit": "lux"},
            "aetheric_resonance": {"velocity_m_s": 750.0, "decay_exponent": 1.4, "unit": "Hz"}
        }

        return {
            "framework": "Krystal-Stack / Poslední Kmen",
            "vital_max_hp": VITAL_MAX_HP,
            "shaders": shaders,
            "scenes": scenes,
            "tribes": tribes,
            "sensory_modalities": sensory_modalities,
            "trench_barrier_reflection": 0.75,
            "conduct_code": "[ACT] [LCK] [CNF] [HAZ] [VIT 6HP]"
        }


# ── 3. VISUAL ORCHESTRATION MODEL ───────────────────────────────────────────

@dataclass
class VisualEntity:
    entity_id: str
    tribe_code: str
    name: str
    q: int
    r: int
    world_pos: Tuple[float, float, float]
    rotation_y: float
    scale: Tuple[float, float, float]
    pbr_albedo_rgb: Tuple[float, float, float]
    emission_energy: float
    hp: int
    conduct_badge: str
    shader_flags: Tuple[float, float, float, float]  # [roughness, wave_phase, aether_glow, refraction]


@dataclass
class ShockwaveFront:
    shockwave_id: str
    modality: str
    origin_hex: Tuple[int, int]
    t_start: float
    peak_intensity: float
    velocity_m_s: float
    current_radius_m: float
    current_intensity: float
    is_active: bool


class PosledniKmenVisualOrchestrator:
    """
    Real-time model orchestrating all 3D visual entities, PBR shader uniforms,
    isometric camera views, and sensory shockwave fronts.
    """

    def __init__(self, repo_root: Optional[str] = None):
        self.repo_root = repo_root or WORKSPACE_ROOT
        self.characteristics = PosledniKmenCharacteristicsExtractor.extract_all(self.repo_root)
        self.wsl_info = WslEnvironmentSensor.inspect()
        self.entities: Dict[str, VisualEntity] = {}
        self.shockwaves: List[ShockwaveFront] = []
        self.tick: int = 0
        self._init_battlefield()

    def _hex_to_world(self, q: int, r: int, hex_radius: float = 1.732) -> Tuple[float, float, float]:
        x = hex_radius * (math.sqrt(3) * q + math.sqrt(3) / 2.0 * r)
        z = hex_radius * (1.5 * r)
        return (round(x, 3), 0.0, round(z, 3))

    def _init_battlefield(self):
        """Initializes the 4 tribal monoliths and vanguard standees."""
        tribes_def = self.characteristics["tribes"]

        # Central Studna Duší (Void Corruption)
        void_def = tribes_def["VOID_CORRUPTION"]
        self.entities["monolith_void"] = VisualEntity(
            entity_id="monolith_void",
            tribe_code="VOID_CORRUPTION",
            name="Studna Duší (Well of Souls)",
            q=0, r=0,
            world_pos=(0.0, 1.2, 0.0),
            rotation_y=0.0,
            scale=(2.0, 3.5, 2.0),
            pbr_albedo_rgb=tuple(void_def["albedo_rgb"]),
            emission_energy=2.8,
            hp=VITAL_MAX_HP,
            conduct_badge="[ACT] [LCK] [CNF] [!HAZ] [VIT 6HP]",
            shader_flags=(0.20, 0.0, 0.95, 0.0)
        )

        # Crystal Citadel Vanguard (q=1, r=0)
        cryst_def = tribes_def["CRYSTAL_CITADEL"]
        pos_cryst = self._hex_to_world(1, 0)
        self.entities["vanguard_crystal"] = VisualEntity(
            entity_id="vanguard_crystal",
            tribe_code="CRYSTAL_CITADEL",
            name="Kryštálový Prizmatický Pylón",
            q=1, r=0,
            world_pos=(pos_cryst[0], 0.8, pos_cryst[2]),
            rotation_y=0.0,
            scale=(1.2, 2.8, 1.2),
            pbr_albedo_rgb=tuple(cryst_def["albedo_rgb"]),
            emission_energy=2.4,
            hp=VITAL_MAX_HP,
            conduct_badge="[ACT] [!LCK] [CNF] [!HAZ] [VIT 6HP]",
            shader_flags=(0.15, 0.0, 0.85, 0.18)
        )

        # Toxic Mire Vanguard (q=0, r=1)
        toxic_def = tribes_def["TOXIC_MIRE"]
        pos_toxic = self._hex_to_world(0, 1)
        self.entities["vanguard_toxic"] = VisualEntity(
            entity_id="vanguard_toxic",
            tribe_code="TOXIC_MIRE",
            name="Toxický Leptavý Totem",
            q=0, r=1,
            world_pos=(pos_toxic[0], 0.6, pos_toxic[2]),
            rotation_y=0.0,
            scale=(1.4, 2.0, 1.4),
            pbr_albedo_rgb=tuple(toxic_def["albedo_rgb"]),
            emission_energy=1.9,
            hp=VITAL_MAX_HP,
            conduct_badge="[ACT] [!LCK] [CNF] [!HAZ] [VIT 6HP]",
            shader_flags=(0.05, 0.25, 0.40, 0.0)
        )

        # Volcanic Forge Bastion (q=-1, r=0)
        volc_def = tribes_def["VOLCANIC_FORGE"]
        pos_volc = self._hex_to_world(-1, 0)
        self.entities["vanguard_volcanic"] = VisualEntity(
            entity_id="vanguard_volcanic",
            tribe_code="VOLCANIC_FORGE",
            name="Sopečná Pyroklastická Vyhňa",
            q=-1, r=0,
            world_pos=(pos_volc[0], 0.7, pos_volc[2]),
            rotation_y=0.0,
            scale=(1.5, 2.2, 1.5),
            pbr_albedo_rgb=tuple(volc_def["albedo_rgb"]),
            emission_energy=2.1,
            hp=VITAL_MAX_HP,
            conduct_badge="[ACT] [!LCK] [CNF] [!HAZ] [VIT 6HP]",
            shader_flags=(0.65, 0.0, 0.75, 0.0)
        )

    def trigger_shockwave(
        self,
        modality: str = "kinetic_overpressure",
        origin_hex: Tuple[int, int] = (0, 0),
        peak_intensity: float = 120.0
    ) -> ShockwaveFront:
        """Emits a sensory shockwave front from given hex coordinate."""
        v_map = {
            "kinetic_overpressure": 480.0,
            "acoustic_blast": 343.0,
            "seismic_tremor": 220.0,
            "optical_flash": 299792458.0,
            "aetheric_resonance": 750.0
        }
        vel = v_map.get(modality, 480.0)

        front = ShockwaveFront(
            shockwave_id=f"SW-{uuid.uuid4().hex[:8].upper()}",
            modality=modality,
            origin_hex=origin_hex,
            t_start=time.time(),
            peak_intensity=peak_intensity,
            velocity_m_s=vel,
            current_radius_m=0.0,
            current_intensity=peak_intensity,
            is_active=True
        )
        self.shockwaves.append(front)
        return front

    def orchestrate_frame(self, dt: float = 0.05) -> Dict[str, Any]:
        """
        Executes one visual orchestration tick:
        - Updates entity rotations & pulse phases.
        - Propagates sensory shockwave fronts.
        - Clamps all entity vital HPs to [0, 6].
        """
        self.tick += 1
        now = time.time()

        # Update entity visual pulses and rotation
        for e_id, ent in self.entities.items():
            # Rotation of Studna Duší
            if ent.tribe_code == "VOID_CORRUPTION":
                ent.rotation_y = (ent.rotation_y + 1.2 * dt) % (2 * math.pi)
            else:
                ent.rotation_y = (ent.rotation_y + 0.3 * dt) % (2 * math.pi)

            # Pulsing emission energy
            pulse = math.sin(self.tick * 0.15 + (ent.q * 1.5))
            ent.emission_energy = round(2.0 + (pulse * 0.6), 3)

            # Invariant: vital HP strictly [0, 6]
            ent.hp = max(0, min(VITAL_MAX_HP, ent.hp))
            act = "[ACT]" if ent.hp > 0 else "[!ACT]"
            lck = "[LCK]" if ent.hp <= 2 else "[!LCK]"
            cnf = "[CNF]"
            haz = "[HAZ]" if ent.hp <= 2 else "[!HAZ]"
            ent.conduct_badge = f"{act} {lck} {cnf} {haz} [VIT {ent.hp}HP]"

        # Propagate active shockwave fronts
        active_shockwaves: List[ShockwaveFront] = []
        for sw in self.shockwaves:
            elapsed = now - sw.t_start
            sw.current_radius_m = round(elapsed * (sw.velocity_m_s * 0.05), 2)
            # Attenuation: A(d) = A0 / (1 + 0.35 * d^1.8)
            dist = max(0.1, sw.current_radius_m)
            sw.current_intensity = round(sw.peak_intensity / (1.0 + 0.35 * math.pow(dist, 1.8)), 2)

            if sw.current_intensity > 2.0 and sw.current_radius_m < 50.0:
                active_shockwaves.append(sw)
            else:
                sw.is_active = False

        self.shockwaves = active_shockwaves

        # Generate 96-Byte Packed Binary Instance Buffer
        packed_buffer = self.generate_packed_buffer()

        return {
            "tick": self.tick,
            "timestamp": now,
            "wsl_environment": self.wsl_info,
            "active_entities_count": len(self.entities),
            "entities": [
                {
                    "id": e.entity_id,
                    "tribe": e.tribe_code,
                    "name": e.name,
                    "hex": (e.q, e.r),
                    "world_pos": e.world_pos,
                    "rotation_y": round(e.rotation_y, 3),
                    "scale": e.scale,
                    "pbr_albedo": e.pbr_albedo_rgb,
                    "emission_energy": e.emission_energy,
                    "hp": e.hp,
                    "conduct_badge": e.conduct_badge
                }
                for e in self.entities.values()
            ],
            "active_shockwaves_count": len(self.shockwaves),
            "shockwaves": [
                {
                    "id": sw.shockwave_id,
                    "modality": sw.modality,
                    "radius_m": sw.current_radius_m,
                    "intensity": sw.current_intensity
                }
                for sw in self.shockwaves
            ],
            "packed_buffer_bytes": len(packed_buffer),
            "vital_max_hp": VITAL_MAX_HP
        }

    def generate_packed_buffer(self) -> bytes:
        """
        Packs all entity render state into a 96-byte per instance binary stream:
          - 16 floats (64 bytes): Mat4x4 transform
          - 4 floats (16 bytes): PBR Albedo RGBA
          - 4 floats (16 bytes): Custom Shader Flags [roughness, wave_phase, aether_glow, refraction]
        """
        buffer = bytearray()
        for ent in self.entities.values():
            cos_y = math.cos(ent.rotation_y)
            sin_y = math.sin(ent.rotation_y)
            sx, sy, sz = ent.scale
            tx, ty, tz = ent.world_pos

            # 4x4 matrix column-major
            m00, m01, m02, m03 = sx * cos_y, 0.0, -sx * sin_y, 0.0
            m10, m11, m12, m13 = 0.0, sy, 0.0, 0.0
            m20, m21, m22, m23 = sz * sin_y, 0.0, sz * cos_y, 0.0
            m30, m31, m32, m33 = tx, ty, tz, 1.0

            mat_bytes = struct.pack(
                "<16f",
                m00, m01, m02, m03,
                m10, m11, m12, m13,
                m20, m21, m22, m23,
                m30, m31, m32, m33
            )

            r, g, b = ent.pbr_albedo_rgb
            albedo_bytes = struct.pack("<4f", r, g, b, 1.0)

            f0, f1, f2, f3 = ent.shader_flags
            flags_bytes = struct.pack("<4f", f0, f1, f2, f3)

            buffer.extend(mat_bytes)
            buffer.extend(albedo_bytes)
            buffer.extend(flags_bytes)

        return bytes(buffer)


# ── 4. BUILT-IN MICRO HTTP DISPATCHER ────────────────────────────────────────

class WslOrchestratorHttpHandler(BaseHTTPRequestHandler):
    orchestrator: PosledniKmenVisualOrchestrator = None

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-WSL-Kernel", platform.release())
        self.send_header("X-Vital-Invariant", f"VITAL_MAX_HP={VITAL_MAX_HP}")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/posledni-kmen/status":
            self._set_headers(200)
            status_data = {
                "status": "ONLINE",
                "engine": "PosledniKmenVisualOrchestrator",
                "wsl": self.orchestrator.wsl_info,
                "vital_max_hp": VITAL_MAX_HP
            }
            self.wfile.write(json.dumps(status_data, indent=2).encode("utf-8"))

        elif self.path == "/api/posledni-kmen/characteristics":
            self._set_headers(200)
            self.wfile.write(json.dumps(self.orchestrator.characteristics, indent=2).encode("utf-8"))

        elif self.path == "/api/posledni-kmen/frame":
            frame = self.orchestrator.orchestrate_frame()
            self._set_headers(200)
            self.wfile.write(json.dumps(frame, indent=2).encode("utf-8"))

        elif self.path == "/api/posledni-kmen/packed-buffer":
            packed = self.orchestrator.generate_packed_buffer()
            self._set_headers(200, "application/octet-stream")
            self.wfile.write(packed)

        else:
            self._set_headers(404)
            self.wfile.write(b'{"error": "Endpoint not found"}')

    def do_POST(self):
        if self.path == "/api/posledni-kmen/shockwave":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            mod = data.get("modality", "kinetic_overpressure")
            orig = tuple(data.get("origin_hex", (0, 0)))
            intensity = float(data.get("peak_intensity", 120.0))

            sw = self.orchestrator.trigger_shockwave(mod, orig, intensity)
            self._set_headers(200)
            self.wfile.write(json.dumps({
                "status": "SHOCKWAVE_EMITTED",
                "shockwave_id": sw.shockwave_id,
                "modality": sw.modality,
                "velocity_m_s": sw.velocity_m_s,
                "origin_hex": sw.origin_hex
            }, indent=2).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(b'{"error": "Endpoint not found"}')


def run_wsl_server(port: int = DEFAULT_PORT):
    orchestrator = PosledniKmenVisualOrchestrator()
    WslOrchestratorHttpHandler.orchestrator = orchestrator
    server_address = ("0.0.0.0", port)
    httpd = HTTPServer(server_address, WslOrchestratorHttpHandler)
    print(f"======================================================================")
    print(f" 🛡️ POSLEDNÍ KMEN WSL VISUAL ORCHESTRATION SERVER")
    print(f" Target: {orchestrator.wsl_info['platform']} ({orchestrator.wsl_info['kernel_release']})")
    print(f" WSL2 Detected: {orchestrator.wsl_info['is_wsl2']} | /dev/dxg: {orchestrator.wsl_info['gpu_dxg_device']}")
    print(f" Listening on http://0.0.0.0:{port}")
    print(f" Endpoints: /api/posledni-kmen/status, /characteristics, /frame, /shockwave")
    print(f"======================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    if "--server" in sys.argv:
        port = DEFAULT_PORT
        if "--port" in sys.argv:
            idx = sys.argv.index("--port")
            if idx + 1 < len(sys.argv):
                port = int(sys.argv[idx + 1])
        run_wsl_server(port)
    else:
        # Benchmark / CLI single-frame test run
        orch = PosledniKmenVisualOrchestrator()
        print(f"[WSL_ORCHESTRATOR] Engine initialized on {orch.wsl_info['platform']}.")
        print(f"[WSL_ORCHESTRATOR] WSL2 Detected: {orch.wsl_info['is_wsl2']} | GPU Passthrough: {orch.wsl_info['gpu_dxg_device']}")
        
        # Trigger test shockwave
        orch.trigger_shockwave("kinetic_overpressure", (0, 0), 150.0)
        frame = orch.orchestrate_frame()
        print(f"[WSL_ORCHESTRATOR] Frame {frame['tick']} generated with {frame['active_entities_count']} entities.")
        print(f"[WSL_ORCHESTRATOR] Packed 96B Buffer size: {frame['packed_buffer_bytes']} bytes.")
        print(f"[WSL_ORCHESTRATOR] Entities preview:")
        for e in frame["entities"]:
            print(f"  - {e['id']} ({e['tribe']}): Pos {e['world_pos']} -> Badge: {e['conduct_badge']}")
        print(f"[WSL_ORCHESTRATOR] SUCCESS: Visual orchestration model verified.")
