# ==============================================================================
# KRYSTAL-STACK: AERIAL BALLOONS, ROGALLO WINGS & PLUNGING MORTAR ENGINE
# ==============================================================================
# Implements:
#   1. Aerostat & Balloon Dynamics (AerialBalloonIslandEngine):
#      - Multi-tier balloon aerostats (thermal sun-furnace, aether cells, ironclad siege island).
#      - Buoyancy lift equations: F_b = (rho_air - rho_gas) * V * g.
#      - Floating island networks connected by skybridge suspension paths and tether cables.
#   2. High-Altitude Plunging Mortar Artillery (PlungingMortarArtilleryEngine):
#      - Elevation advantage ballistics from 150m-450m floating islands.
#      - Trajectory arc generation, gravity plunge bonus, wind dispersion.
#      - Strict 6 Max HP vital invariant compliance for targets and crews.
#   3. Rogallo Hang Gliders & Steerable Parachutes (RogaloAndParachuteFlightEngine):
#      - Rogallo wing flexible delta flight dynamics (7.5:1 glide ratio, thermal updrafts).
#      - High-drag steerable parachute descent (terminal velocity ~5.2 m/s).
#      - Just Cause grapple hook attachment and mid-air island boarding.
#   4. Borderlands Cel-Shaded Comic Aesthetics & Just Cause Kinetic Integration.
# ==============================================================================

import math
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Strict vital invariant across the Krystal ecosystem
VITAL_MAX_HP: int = 6


# ── 1. DATA MODELS & CONFIGURATION ───────────────────────────────────────────

@dataclass
class AerostatProfile:
    id: str
    name: str
    envelope_volume_m3: float
    gas_density_kgm3: float
    empty_mass_kg: float
    max_payload_kg: float
    cruising_altitude_m: float
    ascent_rate_mps: float
    has_mortar_mount: bool
    description: str


@dataclass
class FloatingIslandNode:
    id: str
    name: str
    aerostat_type: str
    altitude_m: float
    coordinates: Tuple[float, float, float]  # (x, y, z)
    island_mass_kg: float
    mortar_equipped: bool
    mortar_caliber_mm: float
    connected_bridges: List[str]
    defense_turrets: int
    garrison_hp: int  # Bound by VITAL_MAX_HP


# ── 2. AEROSTAT & FLOATING ISLAND ENGINE ───────────────────────────────────────

class AerialBalloonIslandEngine:
    """
    Manages balloon buoyancy physics, altitude equilibrium, floating island chains,
    and skybridge suspension paths.
    """

    AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m^3
    GRAVITY = 9.80665              # m/s^2

    CANONICAL_AEROSTATS: Dict[str, AerostatProfile] = {
        "thermal_sun_furnace": AerostatProfile(
            id="thermal_sun_furnace",
            name="Thermal Sun-Furnace Aerostat",
            envelope_volume_m3=3500.0,
            gas_density_kgm3=0.92,
            empty_mass_kg=850.0,
            max_payload_kg=1400.0,
            cruising_altitude_m=220.0,
            ascent_rate_mps=3.8,
            has_mortar_mount=False,
            description="High-temperature solar-heated canvas envelope with quick-ascent burner coils."
        ),
        "aether_gas_cell": AerostatProfile(
            id="aether_gas_cell",
            name="Crystalline Aether Buoyancy Cell",
            envelope_volume_m3=7200.0,
            gas_density_kgm3=0.18,
            empty_mass_kg=1600.0,
            max_payload_kg=5200.0,
            cruising_altitude_m=380.0,
            ascent_rate_mps=6.2,
            has_mortar_mount=False,
            description="Infused with anti-gravity aether vapor giving extraordinary lift efficiency."
        ),
        "ironclad_siege_island": AerostatProfile(
            id="ironclad_siege_island",
            name="Ironclad Floating Siege Bastion",
            envelope_volume_m3=24000.0,
            gas_density_kgm3=0.09,
            empty_mass_kg=5500.0,
            max_payload_kg=18500.0,
            cruising_altitude_m=280.0,
            ascent_rate_mps=2.4,
            has_mortar_mount=True,
            description="Quadruple-tethered armored sky-island platform mounting a 240mm siege mortar."
        ),
        "rogallo_launch_hub": AerostatProfile(
            id="rogallo_launch_hub",
            name="Rogallo Skydock Aerodrome",
            envelope_volume_m3=12500.0,
            gas_density_kgm3=0.15,
            empty_mass_kg=2800.0,
            max_payload_kg=9500.0,
            cruising_altitude_m=340.0,
            ascent_rate_mps=4.5,
            has_mortar_mount=False,
            description="Floating runway cantilever island specialized for launching Rogallo hang gliders."
        )
    }

    def __init__(self):
        self.islands: Dict[str, FloatingIslandNode] = {}
        self.skybridges: List[Dict[str, Any]] = []
        self._initialize_default_sky_network()

    def _initialize_default_sky_network(self):
        """Constructs an initial chain of floating sky-islands connected by skybridges."""
        island_a = FloatingIslandNode(
            id="island_alpha",
            name="Alpha Bastion (Siege Mortar Citadel)",
            aerostat_type="ironclad_siege_island",
            altitude_m=280.0,
            coordinates=(0.0, 280.0, 0.0),
            island_mass_kg=16000.0,
            mortar_equipped=True,
            mortar_caliber_mm=240.0,
            connected_bridges=["bridge_alpha_beta", "bridge_alpha_gamma"],
            defense_turrets=4,
            garrison_hp=VITAL_MAX_HP
        )

        island_b = FloatingIslandNode(
            id="island_beta",
            name="Beta Roost (Rogallo Hang Glider Dock)",
            aerostat_type="rogallo_launch_hub",
            altitude_m=340.0,
            coordinates=(180.0, 340.0, 45.0),
            island_mass_kg=8500.0,
            mortar_equipped=False,
            mortar_caliber_mm=0.0,
            connected_bridges=["bridge_alpha_beta", "bridge_beta_delta"],
            defense_turrets=2,
            garrison_hp=VITAL_MAX_HP
        )

        island_c = FloatingIslandNode(
            id="island_gamma",
            name="Gamma Foundry (Thermal Furnace Balloon)",
            aerostat_type="thermal_sun_furnace",
            altitude_m=220.0,
            coordinates=(-140.0, 220.0, 60.0),
            island_mass_kg=2100.0,
            mortar_equipped=False,
            mortar_caliber_mm=0.0,
            connected_bridges=["bridge_alpha_gamma"],
            defense_turrets=1,
            garrison_hp=VITAL_MAX_HP
        )

        island_d = FloatingIslandNode(
            id="island_delta",
            name="Delta Aether Spire (Drop-Pod & Parachute Station)",
            aerostat_type="aether_gas_cell",
            altitude_m=380.0,
            coordinates=(320.0, 380.0, -20.0),
            island_mass_kg=4800.0,
            mortar_equipped=False,
            mortar_caliber_mm=0.0,
            connected_bridges=["bridge_beta_delta"],
            defense_turrets=3,
            garrison_hp=VITAL_MAX_HP
        )

        self.islands = {
            "island_alpha": island_a,
            "island_beta": island_b,
            "island_gamma": island_c,
            "island_delta": island_d
        }

        self.skybridges = [
            {
                "id": "bridge_alpha_beta",
                "source_island": "island_alpha",
                "target_island": "island_beta",
                "bridge_type": "steel_cables_and_planks",
                "length_m": 195.4,
                "sway_factor": 0.12,
                "max_load_kg": 3500.0,
                "has_zipline": True
            },
            {
                "id": "bridge_alpha_gamma",
                "source_island": "island_alpha",
                "target_island": "island_gamma",
                "bridge_type": "rope_and_bamboo_suspension",
                "length_m": 164.2,
                "sway_factor": 0.28,
                "max_load_kg": 1800.0,
                "has_zipline": True
            },
            {
                "id": "bridge_beta_delta",
                "source_island": "island_beta",
                "target_island": "island_delta",
                "bridge_type": "reinforced_carbon_conduit",
                "length_m": 158.0,
                "sway_factor": 0.05,
                "max_load_kg": 5000.0,
                "has_zipline": True
            }
        ]

    def compute_buoyancy(self, profile_id: str, current_payload_kg: float) -> Dict[str, Any]:
        """Calculates aerodynamic lift and net buoyant force in Newtons."""
        profile = self.CANONICAL_AEROSTATS.get(profile_id)
        if not profile:
            raise ValueError(f"Unknown aerostat profile: {profile_id}")

        air_density = self.AIR_DENSITY_SEA_LEVEL
        gross_lift_n = (air_density - profile.gas_density_kgm3) * profile.envelope_volume_m3 * self.GRAVITY
        total_mass_kg = profile.empty_mass_kg + current_payload_kg
        total_weight_n = total_mass_kg * self.GRAVITY
        net_force_n = gross_lift_n - total_weight_n
        acceleration_mps2 = net_force_n / total_mass_kg if total_mass_kg > 0 else 0.0

        return {
            "profile_id": profile_id,
            "profile_name": profile.name,
            "gross_lift_n": round(gross_lift_n, 2),
            "total_weight_n": round(total_weight_n, 2),
            "net_buoyant_force_n": round(net_force_n, 2),
            "vertical_accel_mps2": round(acceleration_mps2, 3),
            "is_buoyant": net_force_n >= 0,
            "max_payload_kg": profile.max_payload_kg,
            "payload_utilization_pct": round((current_payload_kg / profile.max_payload_kg) * 100.0, 1)
        }

    def get_island_network(self) -> Dict[str, Any]:
        """Returns the full floating sky-island graph and connected suspension pathways."""
        return {
            "islands": {k: asdict(v) for k, v in self.islands.items()},
            "skybridges": self.skybridges,
            "total_islands": len(self.islands),
            "total_skybridges": len(self.skybridges)
        }

    def get_aerostat_profiles(self) -> List[Dict[str, Any]]:
        """Returns all canonical aerostat profiles as serialized dictionaries."""
        return [asdict(p) for p in self.CANONICAL_AEROSTATS.values()]

    def traverse_skybridge(self, bridge_id: str, traveler_weight_kg: float, method: str = "walk") -> Dict[str, Any]:
        """Simulates traversal across an aerial skybridge linking two balloon islands."""
        bridge = next((b for b in self.skybridges if b["id"] == bridge_id), None)
        if not bridge:
            return {"status": "error", "message": f"Bridge {bridge_id} not found."}

        speed_mps = 1.4 if method == "walk" else (4.2 if method == "run" else 15.0)  # zipline
        traversal_time_sec = bridge["length_m"] / speed_mps
        wind_gust_mps = 7.5
        instability_index = min(1.0, bridge["sway_factor"] * (traveler_weight_kg / 100.0) * (wind_gust_mps / 5.0))

        return {
            "bridge_id": bridge_id,
            "source_island": bridge["source_island"],
            "target_island": bridge["target_island"],
            "method": method,
            "speed_mps": speed_mps,
            "traversal_time_sec": round(traversal_time_sec, 2),
            "instability_index": round(instability_index, 3),
            "safe_crossing": instability_index < 0.85
        }


# ── 3. HIGH-ALTITUDE PLUNGING MORTAR ARTILLERY ENGINE ─────────────────────────

class PlungingMortarArtilleryEngine:
    """
    Computes ballistic trajectories, gravity plunge acceleration bonus, and splash
    damage for heavy siege mortars mounted on floating balloon islands.
    Strictly observes the 6 Max HP vital invariant.
    """

    GRAVITY = 9.80665

    def fire_plunging_mortar(
        self,
        island_elevation_m: float,
        muzzle_velocity_mps: float,
        pitch_angle_deg: float,
        yaw_angle_deg: float = 0.0,
        shell_caliber_mm: float = 240.0,
        wind_speed_mps: float = 4.0,
        wind_direction_deg: float = 90.0,
        target_dist_m: float = 450.0,
        target_initial_hp: int = VITAL_MAX_HP
    ) -> Dict[str, Any]:
        """
        Executes a high-angle mortar discharge plunging down from an elevated sky island.
        Calculates trajectory points and impact blast.
        """
        # Enforce vital invariant input bounds
        target_initial_hp = max(1, min(VITAL_MAX_HP, target_initial_hp))

        pitch_rad = math.radians(pitch_angle_deg)
        yaw_rad = math.radians(yaw_angle_deg)
        wind_rad = math.radians(wind_direction_deg)

        vx0 = muzzle_velocity_mps * math.cos(pitch_rad) * math.cos(yaw_rad)
        vy0 = muzzle_velocity_mps * math.sin(pitch_rad)
        vz0 = muzzle_velocity_mps * math.cos(pitch_rad) * math.sin(yaw_rad)

        wx = wind_speed_mps * math.cos(wind_rad)
        wz = wind_speed_mps * math.sin(wind_rad)

        # High-altitude plunging flight time: y(t) = h0 + vy0*t - 0.5*g*t^2 = 0
        discriminant = (vy0 ** 2) + (2.0 * self.GRAVITY * island_elevation_m)
        t_flight = (vy0 + math.sqrt(max(0.0, discriminant))) / self.GRAVITY

        # Trajectory discretization (25 steps)
        steps = 25
        dt = t_flight / steps
        trajectory_points: List[Dict[str, float]] = []

        for i in range(steps + 1):
            t = i * dt
            # Air drag dampens lateral wind slightly
            x = (vx0 + 0.5 * wx) * t
            y = max(0.0, island_elevation_m + (vy0 * t) - (0.5 * self.GRAVITY * (t ** 2)))
            z = (vz0 + 0.5 * wz) * t
            trajectory_points.append({"time_sec": round(t, 2), "x": round(x, 1), "y": round(y, 1), "z": round(z, 1)})

        impact_x = trajectory_points[-1]["x"]
        impact_z = trajectory_points[-1]["z"]
        horizontal_range_m = math.sqrt(impact_x ** 2 + impact_z ** 2)

        # Impact terminal velocity includes gravitational plunge acceleration
        impact_vy = -(vy0 + self.GRAVITY * t_flight)
        impact_speed_mps = math.sqrt((vx0 + wx) ** 2 + (impact_vy ** 2) + (vz0 + wz) ** 2)
        kinetic_energy_kj = 0.5 * (shell_caliber_mm * 0.12) * ((impact_speed_mps / 10.0) ** 2)

        # Distance error to intended target
        miss_distance_m = abs(horizontal_range_m - target_dist_m)

        # Damage calculation bounded by 6 Max HP invariant
        # Direct hit (<15m) deals 3-5 damage; Near miss (<40m) deals 1-2 damage; Far miss deals 0 damage
        if miss_distance_m <= 15.0:
            damage = min(target_initial_hp, max(3, int(round((shell_caliber_mm / 240.0) * 4.5))))
            hit_quality = "DIRECT_HIT_OBLITERATION"
        elif miss_distance_m <= 45.0:
            damage = min(target_initial_hp, max(1, int(round((shell_caliber_mm / 240.0) * 2.0))))
            hit_quality = "SHRAPNEL_SPLASH"
        else:
            damage = 0
            hit_quality = "OUT_OF_RANGE_MISS"

        # Strictly clamp damage between 0 and (VITAL_MAX_HP - 1) unless it's a direct finishing blow
        damage = min(VITAL_MAX_HP, damage)
        target_remaining_hp = max(0, target_initial_hp - damage)

        return {
            "shot_id": f"mortar_{uuid.uuid4().hex[:8]}",
            "elevation_advantage_m": island_elevation_m,
            "flight_time_sec": round(t_flight, 2),
            "horizontal_range_m": round(horizontal_range_m, 2),
            "target_dist_m": target_dist_m,
            "miss_distance_m": round(miss_distance_m, 2),
            "impact_speed_mps": round(impact_speed_mps, 2),
            "plunging_kinetic_energy_kj": round(kinetic_energy_kj, 1),
            "hit_quality": hit_quality,
            "damage_inflicted": damage,
            "target_initial_hp": target_initial_hp,
            "target_remaining_hp": target_remaining_hp,
            "target_destroyed": target_remaining_hp == 0,
            "vital_max_hp_rule_observed": target_initial_hp <= VITAL_MAX_HP and target_remaining_hp <= VITAL_MAX_HP,
            "trajectory_sample_count": len(trajectory_points),
            "trajectory_points": trajectory_points
        }


# ── 4. ROGALLO HANG GLIDERS & STEERABLE PARACHUTES ENGINE ─────────────────────

class RogaloAndParachuteFlightEngine:
    """
    Simulates flight physics for Rogallo flexible hang gliders and steerable parachutes,
    incorporating thermal updrafts, glide ratios (7.5:1), and Just Cause aerial boarding.
    """

    AIR_DENSITY = 1.225
    GRAVITY = 9.80665

    def simulate_rogallo_glider(
        self,
        launch_altitude_m: float,
        initial_airspeed_mps: float = 18.0,
        glide_ratio: float = 7.5,
        thermal_updraft_mps: float = 3.5,
        flight_duration_sec: float = 40.0,
        wind_headwind_mps: float = 2.0
    ) -> Dict[str, Any]:
        """
        Simulates Rogallo flexible delta hang glider flight with thermal updraft lifts.
        Glide ratio determines horizontal reach per meter of descent.
        """
        sink_rate_mps = initial_airspeed_mps / glide_ratio
        effective_vertical_speed = thermal_updraft_mps - sink_rate_mps

        effective_ground_speed_mps = max(5.0, initial_airspeed_mps - wind_headwind_mps)
        flight_points: List[Dict[str, float]] = []

        altitude = launch_altitude_m
        dist_x = 0.0
        dt = 2.0
        steps = int(flight_duration_sec / dt)

        for step in range(steps + 1):
            t = step * dt
            flight_points.append({"time_sec": round(t, 1), "altitude_m": round(altitude, 1), "distance_m": round(dist_x, 1)})
            altitude = max(0.0, altitude + effective_vertical_speed * dt)
            dist_x += effective_ground_speed_mps * dt
            if altitude <= 0.0:
                break

        total_glide_distance_m = dist_x
        altitude_loss_or_gain_m = altitude - launch_altitude_m

        return {
            "vehicle_type": "rogallo_hang_glider",
            "glide_ratio": glide_ratio,
            "launch_altitude_m": launch_altitude_m,
            "final_altitude_m": round(altitude, 1),
            "altitude_delta_m": round(altitude_loss_or_gain_m, 1),
            "total_glide_distance_m": round(total_glide_distance_m, 1),
            "effective_sink_rate_mps": round(sink_rate_mps, 2),
            "thermal_lift_active": thermal_updraft_mps > sink_rate_mps,
            "touchdown": altitude <= 0.0,
            "flight_points": flight_points
        }

    def simulate_steerable_parachute(
        self,
        deployment_altitude_m: float,
        payload_mass_kg: float = 85.0,
        canopy_area_m2: float = 28.0,
        drag_coefficient: float = 1.45,
        steer_lateral_mps: float = 3.5,
        descent_duration_sec: float = 30.0
    ) -> Dict[str, Any]:
        """
        Simulates high-drag steerable parachute canopy descent and landing.
        Terminal velocity: v_t = sqrt(2 * m * g / (rho * A * C_D)).
        """
        denom = 0.5 * self.AIR_DENSITY * canopy_area_m2 * drag_coefficient
        terminal_velocity_mps = math.sqrt((payload_mass_kg * self.GRAVITY) / denom)

        altitude = deployment_altitude_m
        drift_x = 0.0
        dt = 2.0
        steps = int(descent_duration_sec / dt)
        points: List[Dict[str, float]] = []

        for step in range(steps + 1):
            t = step * dt
            points.append({"time_sec": round(t, 1), "altitude_m": round(altitude, 1), "lateral_drift_m": round(drift_x, 1)})
            altitude = max(0.0, altitude - (terminal_velocity_mps * dt))
            drift_x += steer_lateral_mps * dt
            if altitude <= 0.0:
                break

        return {
            "vehicle_type": "steerable_parachute",
            "deployment_altitude_m": deployment_altitude_m,
            "terminal_velocity_mps": round(terminal_velocity_mps, 2),
            "canopy_area_m2": canopy_area_m2,
            "soft_landing_guaranteed": terminal_velocity_mps < 6.0,
            "final_altitude_m": round(altitude, 1),
            "total_lateral_drift_m": round(drift_x, 1),
            "touchdown": altitude <= 0.0,
            "descent_points": points
        }

    def execute_just_cause_grapple_to_balloon_island(
        self,
        hero_initial_pos: Tuple[float, float, float],
        target_island_id: str,
        target_island_pos: Tuple[float, float, float],
        grapple_cable_length_max_m: float = 80.0
    ) -> Dict[str, Any]:
        """
        Executes Just Cause style grappling hook attachment to a floating island,
        reeling in the hero while deploying the parachute or Rogallo glider at the apex.
        """
        hx, hy, hz = hero_initial_pos
        tx, ty, tz = target_island_pos
        dx = tx - hx
        dy = ty - hy
        dz = tz - hz
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        if distance > grapple_cable_length_max_m:
            return {
                "status": "out_of_range",
                "distance_m": round(distance, 1),
                "max_cable_length_m": grapple_cable_length_max_m,
                "message": "Grapple cable cannot reach the floating island. Ascend closer on Rogallo wing!"
            }

        # Reel-in speed 24 m/s (Just Cause 3/4 tether speed)
        reel_speed_mps = 24.0
        pull_time_sec = distance / reel_speed_mps
        slingshot_boost_mps = 16.5  # Apex slingshot catapult momentum

        return {
            "status": "hook_attached_and_reeled",
            "target_island_id": target_island_id,
            "distance_m": round(distance, 1),
            "pull_time_sec": round(pull_time_sec, 2),
            "slingshot_boost_mps": slingshot_boost_mps,
            "recommended_next_action": "Apex release into Steerable Parachute or Rogallo Wing flight",
            "vital_max_hp": VITAL_MAX_HP
        }
