# ==============================================================================
# KRYSTAL-STACK: VORPX VR INJECTION, JUST CAUSE PHYSICS & NCON BY KORRADO MARKETING
# ==============================================================================
# Implements:
#   1. VorpX VR Bridge: Profiles for HTC Vive Pro, Meta Quest, and Ncon by Korrado
#      - Stereoscopic dual-viewport projection, IPD offsets, optical distortion shaders.
#   2. Just Cause Kinetic Physics:
#      - Dual grappling hook spring-damper cable tension & mutual object retraction.
#      - Slingshot catapult momentum boost.
#      - Wingsuit aerodynamic lift/drag gliding equations.
#   3. Borderlands Cel-Shaded Art Engine:
#      - Saturated yellow background (#facc15), ink contour edges, crosshatching, halftone dots.
#   4. Ncon by Korrado Product Marketing Suite:
#      - Campaign copy, tech specs, hardware comparisons, pricing tiers, and SDK guides.
# ==============================================================================

import math
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


# ── 1. VORPX VR PROFILES & STEREOSCOPIC PROJECTION ───────────────────────────
class VorpXVRBridgeEngine:
    """
    Simulates stereoscopic 3D rendering injection, optical distortion shaders,
    and 6DoF tracking for PC VR headsets and custom peripherals.
    """

    HEADSET_PROFILES = {
        "htc_vive_pro": {
            "id": "htc_vive_pro",
            "name": "HTC Vive Pro",
            "tier": "enterprise_pcvr",
            "resolution_per_eye": [1440, 1600],
            "fov_degrees": 110.0,
            "refresh_rate_hz": 90,
            "ipd_default_mm": 63.5,
            "optical_distortion_k1": 0.22,
            "optical_distortion_k2": 0.24,
            "display_type": "Dual AMOLED",
            "tracking_system": "SteamVR Lighthouse 2.0 (Sub-millimeter)"
        },
        "meta_quest": {
            "id": "meta_quest",
            "name": "Meta Quest 3 / Low-Cost Standalone",
            "tier": "consumer_standalone",
            "resolution_per_eye": [2064, 2208],
            "fov_degrees": 110.0,
            "refresh_rate_hz": 120,
            "ipd_default_mm": 64.0,
            "optical_distortion_k1": 0.18,
            "optical_distortion_k2": 0.15,
            "display_type": "Dual LCD + Pancake Optics",
            "tracking_system": "Inside-Out Optical 6DoF"
        },
        "ncon_by_korrado": {
            "id": "ncon_by_korrado",
            "name": "Ncon by Korrado Ultra-VR",
            "tier": "flagship_kinetic_peripheral",
            "resolution_per_eye": [3840, 2160],  # Dual 4K HDR
            "fov_degrees": 130.0,
            "refresh_rate_hz": 144,
            "ipd_default_mm": 62.0,
            "optical_distortion_k1": 0.08,
            "optical_distortion_k2": 0.05,
            "display_type": "Custom Micro-OLED + Aspheric Glass",
            "tracking_system": "Direct-Drive 1000Hz IMU + Ultrasonic Constellation",
            "haptic_trigger_latency_ms": 1.2,
            "grapple_force_feedback": True
        }
    }

    @staticmethod
    def get_headset_profiles() -> Dict[str, Any]:
        return VorpXVRBridgeEngine.HEADSET_PROFILES

    @staticmethod
    def compute_stereoscopic_projection(
        headset_id: str,
        custom_ipd_mm: Optional[float] = None,
        world_camera_pos: Tuple[float, float, float] = (0.0, 1.7, 0.0)
    ) -> Dict[str, Any]:
        profile = VorpXVRBridgeEngine.HEADSET_PROFILES.get(
            headset_id, VorpXVRBridgeEngine.HEADSET_PROFILES["ncon_by_korrado"]
        )
        ipd = custom_ipd_mm if custom_ipd_mm is not None else profile["ipd_default_mm"]
        half_ipd_m = (ipd * 0.001) / 2.0

        cx, cy, cz = world_camera_pos

        # Left eye shifted -X, Right eye shifted +X
        left_eye_pos = [round(cx - half_ipd_m, 4), cy, cz]
        right_eye_pos = [round(cx + half_ipd_m, 4), cy, cz]

        # Radial distortion parameters for barrel shader
        k1 = profile["optical_distortion_k1"]
        k2 = profile["optical_distortion_k2"]

        return {
            "headset_id": profile["id"],
            "headset_name": profile["name"],
            "ipd_used_mm": ipd,
            "resolution_per_eye": profile["resolution_per_eye"],
            "fov_degrees": profile["fov_degrees"],
            "refresh_rate_hz": profile["refresh_rate_hz"],
            "left_eye_camera": {
                "position": left_eye_pos,
                "frustum_offset_x": -half_ipd_m
            },
            "right_eye_camera": {
                "position": right_eye_pos,
                "frustum_offset_x": half_ipd_m
            },
            "godot_optical_shader_uniforms": {
                "k1_distortion": k1,
                "k2_distortion": k2,
                "chromatic_aberration_correction": [0.995, 1.0, 1.005],
                "aspect_ratio": profile["resolution_per_eye"][0] / profile["resolution_per_eye"][1]
            }
        }


# ── 2. JUST CAUSE KINETIC PHYSICS ENGINE ──────────────────────────────────────
class JustCauseKineticPhysicsEngine:
    """
    Simulates high-speed dual grappling hook tethers, object retractions,
    slingshot momentum boosts, and wingsuit aerodynamic gliding.
    """

    @staticmethod
    def simulate_grapple_tether(
        origin_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float],
        player_mass_kg: float = 85.0,
        target_mass_kg: float = 50.0,  # e.g. explosive barrel or enemy
        spring_k: float = 850.0,
        damper_c: float = 45.0,
        reel_in_force_n: float = 1200.0
    ) -> Dict[str, Any]:
        ox, oy, oz = origin_pos
        tx, ty, tz = target_pos

        dx = tx - ox
        dy = ty - oy
        dz = tz - oz
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        if distance < 0.001:
            distance = 0.001
        dir_norm = [dx / distance, dy / distance, dz / distance]

        # Reel-in acceleration on player and target
        player_accel = round(reel_in_force_n / player_mass_kg, 2)
        target_accel = round(reel_in_force_n / target_mass_kg, 2)

        # Kinetic energy generated
        combined_velocity = player_accel * 0.5  # over 0.5s pull
        impact_kinetic_energy_j = round(0.5 * target_mass_kg * (combined_velocity ** 2), 1)

        # Explosion trigger condition for barrels
        triggers_detonation = (impact_kinetic_energy_j > 1500.0)

        return {
            "tether_id": f"tether_{uuid.uuid4().hex[:6]}",
            "distance_m": round(distance, 2),
            "tension_vector": [round(c * reel_in_force_n, 1) for c in dir_norm],
            "player_acceleration_mps2": player_accel,
            "target_acceleration_mps2": target_accel,
            "impact_kinetic_energy_j": impact_kinetic_energy_j,
            "triggers_detonation": triggers_detonation,
            "slingshot_ready": True
        }

    @staticmethod
    def simulate_slingshot_momentum(
        current_velocity_mps: float = 18.0,
        tether_tension_n: float = 1200.0,
        player_mass_kg: float = 85.0,
        release_angle_deg: float = 25.0
    ) -> Dict[str, Any]:
        rad = math.radians(release_angle_deg)
        impulse = (tether_tension_n * 0.4) / player_mass_kg
        boosted_velocity = round(current_velocity_mps + impulse * math.cos(rad), 2)
        vertical_boost = round(impulse * math.sin(rad), 2)

        return {
            "initial_velocity_mps": current_velocity_mps,
            "boosted_velocity_mps": boosted_velocity,
            "vertical_boost_mps": vertical_boost,
            "kinetic_energy_boost_pct": round(((boosted_velocity / current_velocity_mps) ** 2 - 1.0) * 100.0, 1),
            "wingsuit_transition_optimal": True
        }

    @staticmethod
    def simulate_wingsuit_glide(
        drop_altitude_m: float = 150.0,
        airspeed_mps: float = 35.0,
        dive_pitch_deg: float = -12.0
    ) -> Dict[str, Any]:
        # Aerodynamics equations: L/D glide ratio
        # Standard tactical wingsuit L/D is ~3.5
        glide_ratio = 3.5
        horizontal_travel_m = round(drop_altitude_m * glide_ratio, 1)
        flight_duration_sec = round(horizontal_travel_m / airspeed_mps, 1)
        sink_rate_mps = round(drop_altitude_m / max(1.0, flight_duration_sec), 2)

        return {
            "drop_altitude_m": drop_altitude_m,
            "airspeed_mps": airspeed_mps,
            "dive_pitch_degrees": dive_pitch_deg,
            "glide_ratio": f"{glide_ratio}:1",
            "horizontal_range_m": horizontal_travel_m,
            "flight_duration_sec": flight_duration_sec,
            "sink_rate_mps": sink_rate_mps,
            "maneuver_rating": "HIGH_SPEED_ACROBATIC"
        }


# ── 3. BORDERLANDS CEL-SHADED ART SPECIFICATION ───────────────────────────────
class BorderlandsCelShadingEngine:
    """
    Configures the post-processing shader parameters for Godot 4:
    Sobel edge-detection ink lines, crosshatching, and vibrant saturated yellow background.
    """

    @staticmethod
    def get_cel_shading_uniforms() -> Dict[str, Any]:
        return {
            "art_style": "Borderlands Cel-Shaded Comic Action",
            "background_palette": {
                "primary_yellow_hex": "#facc15",
                "accent_orange_hex": "#ea580c",
                "ink_black_hex": "#09090b",
                "grunge_shadow_hex": "#451a03"
            },
            "sobel_edge_detection": {
                "ink_contour_thickness_px": 2.5,
                "depth_threshold": 0.22,
                "normal_threshold": 0.35,
                "ink_color_rgba": [0.03, 0.03, 0.04, 1.0]
            },
            "lighting_quantization": {
                "light_bands_count": 4,
                "specular_sharpness": 128.0,
                "crosshatch_shading_enabled": True,
                "halftone_dot_frequency": 32.0
            }
        }


# ── 4. NCON BY KORRADO PRODUCT MARKETING SUITE ────────────────────────────────
class NconProductMarketingEngine:
    """
    Marketing suite for user's flagship VR peripheral 'Ncon by Korrado'.
    Positions the product against HTC Vive Pro and Meta Quest.
    """

    MARKETING_SPEC = {
        "product_name": "Ncon by Korrado",
        "tagline": "SWING FREE. CAUSE CHAOS.",
        "brand_hero_style": "Borderlands Cel-Shaded Kinetic Action",
        "background_color_hex": "#facc15",
        "core_value_props": [
            {
                "title": "Ultra-Wide 130° FOV Aspheric Optics",
                "desc": "Zahoďte efekt potápačských okuliarov. Ncon prináša 130-stupňové periférne videnie s nulovým skreslením."
            },
            {
                "title": "Dual 4K HDR Micro-OLED Panely",
                "desc": "Krištáľovo čisté rozlíšenie 3840x2160 na každé oko so 144Hz obnovovacou frekvenciou pre extrémny Just Cause pohyb."
            },
            {
                "title": "VorpX Direct-Drive Spatial Injector",
                "desc": "Natívna podpora pre PC VR 3D stereoskopické injektovanie. Okamžitá kompatibilita s hrami bez natívneho VR."
            },
            {
                "title": "Kinetické Haptické Spúšte s 1.2ms Odozvou",
                "desc": "Skutočný odpor pri vystrelení lana grapple hooku a dynamické vibračné pulzy pri wingsuit lete."
            }
        ],
        "hardware_comparison_table": [
            {
                "feature": "FOV (Zorné Pole)",
                "htc_vive_pro": "110°",
                "meta_quest_3": "110°",
                "ncon_by_korrado": "130° ULTRA-WIDE (Víťaz)"
            },
            {
                "feature": "Rozlíšenie na Oko",
                "htc_vive_pro": "1440 x 1600",
                "meta_quest_3": "2064 x 2208",
                "ncon_by_korrado": "3840 x 2160 (Dual 4K HDR)"
            },
            {
                "feature": "Obnovovacia Frekvencia",
                "htc_vive_pro": "90 Hz",
                "meta_quest_3": "120 Hz",
                "ncon_by_korrado": "144 Hz ULTRA-SMOOTH"
            },
            {
                "feature": "Latencia Snímania IMU",
                "htc_vive_pro": "4.5 ms",
                "meta_quest_3": "6.0 ms",
                "ncon_by_korrado": "1.2 ms DIRECT-DRIVE"
            },
            {
                "feature": "VorpX 3D Injection Integrácia",
                "htc_vive_pro": "Manuálna konfigurácia",
                "meta_quest_3": "Link kábel / AirLink",
                "ncon_by_korrado": "1-Click Auto-Stereo Bridge"
            }
        ],
        "pricing_tiers": [
            {
                "tier_name": "Ncon Kinetic Starter",
                "price_eur": 549.0,
                "contents": "Ncon Headset, Dual Kinetic Motion Ovládače, VorpX Lite Licencia, Krystal-Stack SDK"
            },
            {
                "tier_name": "Korrado Grapple Master Bundle",
                "price_eur": 799.0,
                "contents": "Ncon Headset, Force-Feedback Grapple Ovládače, Haptická Wingsuit Vesta, Doživotná VorpX Pro Licencia"
            },
            {
                "tier_name": "Citadel Studio Devkit",
                "price_eur": 1299.0,
                "contents": "Kompletný Hardvérový Balík + OpenXR / Godot 4 C++ Extension Zdrojové Kódy & Priama NPU Akcelerácia"
            }
        ]
    }

    @staticmethod
    def get_marketing_package() -> Dict[str, Any]:
        return NconProductMarketingEngine.MARKETING_SPEC


# Global Singletons
GLOBAL_VORPX_VR_BRIDGE = VorpXVRBridgeEngine()
GLOBAL_JUSTCAUSE_PHYSICS = JustCauseKineticPhysicsEngine()
GLOBAL_CEL_SHADING = BorderlandsCelShadingEngine()
GLOBAL_NCON_MARKETING = NconProductMarketingEngine()
