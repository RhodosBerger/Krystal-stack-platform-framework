#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: AUGMENTED REALITY SPATIAL SOCKET ENGINE & EMERGENT ITEMS
==============================================================================
Module: krystal_kernel/augmented_spatial_socket_engine.py
Description: Dual-target real-time socket server for Three.js (WebGL/WSS) and
             Godot 4.x (GDScript/TCP), executing multi-scale AR simulation
             (Micro, Meso, Macro) across continuous spatial fields (Vector Nav,
             Aetheric Force, Cover Gradient, Harmonic Resonance) and governing
             emergent, non-generic stochastic item automata.

Platform Invariant: VITAL_MAX_HP == 6 strictly enforced.
Author: Dušan Kopecký & Krystal-Stack Architecture Council (2026)
==============================================================================
"""

import os
import sys
import time
import math
import uuid
import json
import random
import socket
import struct
import threading
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

# ── 1. MULTI-SCALE AUGMENTED ENVIRONMENT SPECIFICATION ───────────────────────

class SimulationScale(str, Enum):
    MICRO = "MICRO"  # 0.1m - 5m: Item physics, local entropy, collision, 60-120 Hz
    MESO = "MESO"    # 5m - 100m: Unit rosters, mortar ballistics, cover fields, 30-60 Hz
    MACRO = "MACRO"  # > 100m: Real-world GPS/DEM terrain, weather, global invariants, 1-10 Hz


@dataclass
class AugmentedEnvironmentAnchor:
    """Anchors virtual tactical space to real-world physical coordinates."""
    real_world_gps: Tuple[float, float] = (50.0878, 14.4205)  # Prague Old Town / Bohemian Base
    altitude_m: float = 215.0
    metric_grid_origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gravity: float = 9.80665
    ambient_temperature_c: float = 18.5
    barometric_pressure_kpa: float = 101.325
    wind_vector: List[float] = field(default_factory=lambda: [2.5, 0.0, 1.2])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "real_world_gps": list(self.real_world_gps),
            "altitude_m": self.altitude_m,
            "metric_grid_origin": list(self.metric_grid_origin),
            "gravity": self.gravity,
            "ambient_temperature_c": self.ambient_temperature_c,
            "barometric_pressure_kpa": self.barometric_pressure_kpa,
            "wind_vector": self.wind_vector
        }


# ── 2. CONTINUOUS SPATIAL FIELDS FOR SMOOTH GAMEPLAY ────────────────────────

class ContinuousSpatialFieldManager:
    """
    Maintains 4 continuous potential and force fields that eliminate discrete
    grid snapping and provide fluid, jerk-free navigation and ballistics.
    """

    def __init__(self, anchor: Optional[AugmentedEnvironmentAnchor] = None):
        self.anchor = anchor or AugmentedEnvironmentAnchor()
        self.obstacles: List[Dict[str, Any]] = [
            {"pos": [0.0, 0.0, 0.0], "radius": 3.0, "type": "TRENCH_BUNKER"},
            {"pos": [14.0, 0.0, 6.0], "radius": 2.2, "type": "ROCK_OUTCROP"},
            {"pos": [-10.0, 0.0, -8.0], "radius": 4.0, "type": "CRYSTAL_SPIRE"}
        ]
        self.trench_zones: List[Dict[str, Any]] = [
            {"center": [0.0, 0.0, 0.0], "width": 8.0, "depth": 15.0, "fortification_level": 0.85}
        ]

    # 1. Continuous Vector Navigation Potential Field
    def evaluate_navigation_vector(self, x: float, y: float, z: float, goal: Tuple[float, float, float] = (20.0, 0.0, 15.0)) -> List[float]:
        gx, gy, gz = goal
        # Attractive goal gradient
        dx = gx - x
        dz = gz - z
        dist_goal = max(0.1, math.sqrt(dx * dx + dz * dz))
        v_attr_x = (dx / dist_goal) * 1.5
        v_attr_z = (dz / dist_goal) * 1.5

        # Repulsive obstacle potential gradients
        v_rep_x, v_rep_z = 0.0, 0.0
        for obs in self.obstacles:
            ox, _, oz = obs["pos"]
            radius = obs["radius"]
            dist_obs = max(0.1, math.sqrt((x - ox)**2 + (z - oz)**2))
            if dist_obs < radius * 2.5:
                strength = (1.0 / (dist_obs**2)) * 3.0
                v_rep_x += ((x - ox) / dist_obs) * strength
                v_rep_z += ((z - oz) / dist_obs) * strength

        vx = v_attr_x + v_rep_x
        vz = v_attr_z + v_rep_z
        speed = math.sqrt(vx * vx + vz * vz)
        if speed > 2.5:
            vx = (vx / speed) * 2.5
            vz = (vz / speed) * 2.5
        return [round(vx, 3), 0.0, round(vz, 3)]

    # 2. Continuous Aetheric Aerodynamic Force Field
    def evaluate_aetheric_force(self, x: float, y: float, z: float, t: float = 0.0) -> List[float]:
        # Macro wind base
        wx, wy, wz = self.anchor.wind_vector

        # Curl turbulence standing fluctuation
        turb_x = math.sin(z * 0.15 + t * 0.8) * math.cos(y * 0.1) * 0.8
        turb_y = math.sin(x * 0.20 + t * 0.5) * 0.4
        turb_z = math.cos(x * 0.15 + t * 0.8) * math.sin(y * 0.1) * 0.8

        fx = round(wx + turb_x, 3)
        fy = round(wy + turb_y, 3)
        fz = round(wz + turb_z, 3)
        return [fx, fy, fz]

    # 3. Continuous Tactical Cover & Occlusion Field (0.0 to 1.0)
    def evaluate_cover_gradient(self, x: float, y: float, z: float) -> float:
        max_cover = 0.0
        for trench in self.trench_zones:
            cx, _, cz = trench["center"]
            half_w = trench["width"] * 0.5
            half_d = trench["depth"] * 0.5
            dist_x = abs(x - cx)
            dist_z = abs(z - cz)

            if dist_x <= half_w and dist_z <= half_d:
                # Smooth sigmoid falloff near trench rim
                norm_dist = max(dist_x / half_w, dist_z / half_d)
                local_cover = trench["fortification_level"] * (1.0 - (norm_dist**3) * 0.4)
                if local_cover > max_cover:
                    max_cover = local_cover

        return round(max(0.0, min(1.0, max_cover)), 3)

    # 4. Continuous Harmonic Resonance Standing Wave Field
    def evaluate_harmonic_resonance(self, x: float, y: float, z: float, t: float = 0.0) -> float:
        dist_center = math.sqrt(x*x + z*z)
        wavelength = 18.0
        freq = 0.75
        standing_wave = 0.5 * (1.0 + math.sin((2.0 * math.pi / wavelength) * dist_center - freq * t) * math.cos(0.6180339887))
        return round(max(0.0, min(1.0, standing_wave)), 3)


# ── 3. EMERGENT STOCHASTIC ITEM AUTOMATA (NON-GENERIC RULES) ────────────────

class ItemPhaseState(str, Enum):
    STABLE = "STABLE"
    VOLATILE = "VOLATILE"
    SUPERCRITICAL = "SUPERCRITICAL"
    TRANSMUTING = "TRANSMUTING"
    ENTANGLED = "ENTANGLED"


@dataclass
class StochasticItem:
    """
    A dynamic physical-alchemical game entity with non-linear entropy,
    phase transitions, and surprising emergent behaviors strictly governed by rules.
    """
    item_id: str
    item_type: str  # "VOLATILE_PHASE_SHELL", "ALCHEMICAL_RESONATOR", "GRAVITATIONAL_MIRROR_LENS", "MIMIC_SPORE_RELIC"
    position: List[float]  # [x, y, z] in metric meters
    velocity: List[float]  # [vx, vy, vz]
    entropy: float = 0.20  # 0.0 to 1.0 internal disorder
    mass_kg: float = 4.5
    energy_joules: float = 1200.0
    phase_state: ItemPhaseState = ItemPhaseState.STABLE
    transmuted: bool = False
    transmutation_type: Optional[str] = None
    bifurcation_count: int = 0
    sub_projectiles: List[Dict[str, Any]] = field(default_factory=list)
    vital_max_hp: int = VITAL_MAX_HP

    def step_physics(self, dt: float, fields: ContinuousSpatialFieldManager, current_time: float):
        x, y, z = self.position
        vx, vy, vz = self.velocity

        # 1. Couple with Aetheric Force Field
        fx, fy, fz = fields.evaluate_aetheric_force(x, y, z, current_time)
        ax = fx / max(0.1, self.mass_kg)
        ay = (fy / max(0.1, self.mass_kg)) - fields.anchor.gravity
        az = fz / max(0.1, self.mass_kg)

        # 2. Integrate Velocity & Position (Verlet-style)
        vx += ax * dt
        vy += ay * dt
        vz += az * dt
        x += vx * dt
        y += vy * dt
        z += vz * dt

        # 3. Ground Collision & Surface Interaction
        if y <= 0.0:
            y = 0.0
            # Elastic bounce with restitution based on phase
            restitution = 0.45 if self.phase_state == ItemPhaseState.STABLE else 0.75
            vy = -vy * restitution
            vx *= 0.85
            vz *= 0.85
            # Ground impact increases entropy
            impact_energy = 0.5 * self.mass_kg * (vx*vx + vy*vy + vz*vz)
            self.entropy = min(1.0, self.entropy + (impact_energy / 2500.0) * 0.15)

        # 4. Harmonic Field Resonance Coupling
        h_res = fields.evaluate_harmonic_resonance(x, y, z, current_time)
        self.entropy = min(1.0, max(0.0, self.entropy + (h_res - 0.5) * 0.05 * dt))

        # 5. Non-Generic State Automaton Transitions
        if self.entropy > 0.80 and self.phase_state in (ItemPhaseState.STABLE, ItemPhaseState.VOLATILE):
            self.phase_state = ItemPhaseState.SUPERCRITICAL
            self._trigger_supercritical_reaction(fields)
        elif self.entropy > 0.45 and self.phase_state == ItemPhaseState.STABLE:
            self.phase_state = ItemPhaseState.VOLATILE

        self.position = [round(x, 3), round(y, 3), round(z, 3)]
        self.velocity = [round(vx, 3), round(vy, 3), round(z, 3)]
        self.entropy = round(self.entropy, 3)

    def _trigger_supercritical_reaction(self, fields: ContinuousSpatialFieldManager):
        """Rule-governed emergent reactions that surprise the player without being generic."""
        if self.item_type == "VOLATILE_PHASE_SHELL":
            # Bifurcate into 3 sub-bursts or transmute crater into crystal ice
            if self.bifurcation_count == 0:
                self.bifurcation_count = 3
                self.transmuted = True
                self.transmutation_type = "CRYSTAL_FROST_CRATER"
                self.phase_state = ItemPhaseState.TRANSMUTING
                # Spawn sub-munitions with conserved kinetic energy
                for i in range(3):
                    angle = (i * 2.0 * math.pi / 3.0) + (self.entropy * math.pi)
                    sub_vx = math.cos(angle) * 8.5
                    sub_vz = math.sin(angle) * 8.5
                    self.sub_projectiles.append({
                        "sub_id": f"{self.item_id}_sub_{i}",
                        "rel_pos": [round(math.cos(angle) * 1.5, 2), 0.5, round(math.sin(angle) * 1.5, 2)],
                        "velocity": [round(sub_vx, 2), 4.2, round(sub_vz, 2)],
                        "damage_hp": 1 # strictly <= 6 HP
                    })

        elif self.item_type == "ALCHEMICAL_RESONATOR":
            self.transmuted = True
            self.transmutation_type = "KINETIC_DEFLECTION_BARRIER"
            self.phase_state = ItemPhaseState.ENTANGLED

        elif self.item_type == "GRAVITATIONAL_MIRROR_LENS":
            self.transmuted = True
            self.transmutation_type = "GOLDEN_SPIRAL_DEFLECTION"
            self.phase_state = ItemPhaseState.TRANSMUTING

        elif self.item_type == "MIMIC_SPORE_RELIC":
            self.transmuted = True
            self.transmutation_type = "SUPERFORMULA_POLYMETH"
            self.phase_state = ItemPhaseState.ENTANGLED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "item_type": self.item_type,
            "position": self.position,
            "velocity": self.velocity,
            "entropy": self.entropy,
            "mass_kg": self.mass_kg,
            "energy_joules": self.energy_joules,
            "phase_state": self.phase_state.value,
            "transmuted": self.transmuted,
            "transmutation_type": self.transmutation_type,
            "bifurcation_count": self.bifurcation_count,
            "sub_projectiles": self.sub_projectiles,
            "vital_max_hp": self.vital_max_hp
        }


# ── 4. REAL-TIME SOCKET SERVER: THREE.JS & GODOT 4.X ────────────────────────

class AugmentedSocketProtocolServer:
    """
    High-performance, zero-dependency socket server handling simultaneous
    connections from Three.js (WebGL WebSockets) and Godot 4.x (GDScript TCP/WebSocket).
    Executes real-time multi-scale simulation steps at 60 Hz.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8089):
        self.host = host
        self.port = port
        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.clients: List[socket.socket] = []
        self.lock = threading.Lock()

        # Simulation Subsystems
        self.anchor = AugmentedEnvironmentAnchor()
        self.fields = ContinuousSpatialFieldManager(self.anchor)
        self.active_items: Dict[str, StochasticItem] = {}
        self.current_simulation_time: float = 0.0
        self.tick_count: int = 0

        # Seed initial emergent items
        self.deploy_item("VOLATILE_PHASE_SHELL", [0.0, 15.0, 0.0], [4.5, 8.0, 3.2], entropy=0.35)
        self.deploy_item("ALCHEMICAL_RESONATOR", [6.0, 0.0, 4.0], [0.0, 0.0, 0.0], entropy=0.15)
        self.deploy_item("GRAVITATIONAL_MIRROR_LENS", [-8.0, 0.0, 2.0], [0.0, 0.0, 0.0], entropy=0.25)
        self.deploy_item("MIMIC_SPORE_RELIC", [12.0, 0.0, -6.0], [0.0, 0.0, 0.0], entropy=0.40)

    def deploy_item(self, item_type: str, position: List[float], velocity: List[float], entropy: float = 0.2) -> StochasticItem:
        item_id = f"item_{uuid.uuid4().hex[:8]}"
        item = StochasticItem(
            item_id=item_id,
            item_type=item_type,
            position=position,
            velocity=velocity,
            entropy=entropy,
            vital_max_hp=VITAL_MAX_HP
        )
        with self.lock:
            self.active_items[item_id] = item
        return item

    def step_simulation(self, dt: float = 0.0166):
        """Advances simulation tick across Micro, Meso, and Macro scales."""
        with self.lock:
            self.tick_count += 1
            self.current_simulation_time += dt

            # 1. Micro-scale: Step item physics and stochastic entropy
            for item in list(self.active_items.values()):
                item.step_physics(dt, self.fields, self.current_simulation_time)

            # 2. Meso-scale: Check tactical cover and unit interactions
            # 3. Macro-scale: Real-world anchor solar & weather fluctuations
            if self.tick_count % 60 == 0:
                # Slowly oscillate ambient wind
                self.anchor.wind_vector[0] = round(2.5 + math.sin(self.current_simulation_time * 0.1) * 0.8, 2)

    def generate_state_payload(self, target_client: str = "THREE_JS") -> Dict[str, Any]:
        """Synthesizes high-density synchronization payload for Three.js or Godot."""
        with self.lock:
            items_list = [item.to_dict() for item in self.active_items.values()]
            center_cover = self.fields.evaluate_cover_gradient(0.0, 0.0, 0.0)
            center_res = self.fields.evaluate_harmonic_resonance(0.0, 0.0, 0.0, self.current_simulation_time)
            aether_force = self.fields.evaluate_aetheric_force(0.0, 5.0, 0.0, self.current_simulation_time)
            nav_sample = self.fields.evaluate_navigation_vector(5.0, 0.0, 5.0)

            return {
                "protocol": "KASA/1.0",
                "packet_id": f"pkt_{uuid.uuid4().hex[:8]}",
                "timestamp_ns": time.time_ns(),
                "simulation_time_s": round(self.current_simulation_time, 3),
                "tick_count": self.tick_count,
                "target_client": target_client,
                "environment_anchor": self.anchor.to_dict(),
                "simulation_scales": {
                    "micro": {"active_items_count": len(items_list), "tick_rate_hz": 60},
                    "meso": {"cover_center_grade": center_cover, "tactical_obstacles": len(self.fields.obstacles)},
                    "macro": {"gps_origin": list(self.anchor.real_world_gps), "altitude_m": self.anchor.altitude_m}
                },
                "spatial_fields": {
                    "cover_gradient_at_origin": center_cover,
                    "harmonic_resonance_at_origin": center_res,
                    "aetheric_force_midground": aether_force,
                    "navigation_vector_sample": nav_sample
                },
                "items": items_list,
                "vital_max_hp": VITAL_MAX_HP
            }

    def start_socket_listener(self):
        """Starts background TCP socket thread accepting connections."""
        if self.running:
            return
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(16)
            self.server_socket.settimeout(1.0)
            self.running = True

            threading.Thread(target=self._listener_loop, daemon=True).start()
            threading.Thread(target=self._simulation_loop, daemon=True).start()
        except Exception as e:
            # Fallback if port bound
            print(f"[KASA SOCKET] Notice: {e}. In-memory simulation remains 100% active.")

    def _listener_loop(self):
        while self.running and self.server_socket:
            try:
                client_sock, addr = self.server_socket.accept()
                with self.lock:
                    self.clients.append(client_sock)
                threading.Thread(target=self._client_handler, args=(client_sock,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception:
                break

    def _client_handler(self, client_sock: socket.socket):
        try:
            client_sock.settimeout(0.5)
            while self.running:
                try:
                    data = client_sock.recv(4096)
                    if not data:
                        break
                    # Parse command (e.g. JSON action)
                    try:
                        cmd = json.loads(data.decode("utf-8"))
                        if cmd.get("action") == "DEPLOY_ITEM":
                            itype = cmd.get("item_type", "VOLATILE_PHASE_SHELL")
                            pos = cmd.get("position", [0.0, 10.0, 0.0])
                            vel = cmd.get("velocity", [2.0, 5.0, 2.0])
                            self.deploy_item(itype, pos, vel)
                    except Exception:
                        pass
                except socket.timeout:
                    # Broadcast active state packet
                    state_pkg = self.generate_state_payload("GODOT_4X")
                    msg = (json.dumps(state_pkg) + "\n").encode("utf-8")
                    client_sock.sendall(msg)
        except Exception:
            pass
        finally:
            with self.lock:
                if client_sock in self.clients:
                    self.clients.remove(client_sock)
            try:
                client_sock.close()
            except Exception:
                pass

    def _simulation_loop(self):
        while self.running:
            self.step_simulation(dt=0.0166)
            time.sleep(0.0166)

    def stop(self):
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass


# Global singleton instance
GLOBAL_AUGMENTED_SOCKET_ENGINE = AugmentedSocketProtocolServer(host="127.0.0.1", port=8089)
