# ==============================================================================
# KRYSTAL-STACK: VISUAL PHENOMENA, SPELL PROJECTIONS & TOTEM ANOMALY DETECTOR
# ==============================================================================
# Implements:
#   1. Volumetric Visual Phenomena (Chromatic Aberration, Aurora Streams, St. Elmo's Plasma).
#   2. Spell Projections (Conical AoE, Radial Decals, Ballistic Beams, Hex Line Pierce).
#   3. Spatial Anomaly Detector Array (Gradient Divergence, Ley-line Shifts, Void Rifts).
#   4. Sector Totem Manager (Crystal, Toxic, Druid, Sulfur, Nexus Totems in contested sectors).
#   5. Totem State Modulation (Attunement, Overcharge, Corruption, Nullification, Resonance).
# ==============================================================================

import math
import time
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple


# ── 1. ENUMS & CONSTANTS ──────────────────────────────────────────────────────
class VisualPhenomenonType(str, Enum):
    CHROMATIC_ABERRATION_BURST = "chromatic_aberration_burst"
    AETHERIC_AURORA_STREAM = "aetheric_aurora_stream"
    ST_ELMOS_PLASMA_DISCHARGE = "st_elmos_plasma_discharge"
    GRAVITATIONAL_LENS_WARP = "gravitational_lens_warp"
    VOLUMETRIC_VOID_FOG = "volumetric_void_fog"


class SpellProjectionType(str, Enum):
    CONICAL_AOE = "conical_aoe"
    RADIAL_BLAST_DECAL = "radial_blast_decal"
    BALLISTIC_ARC_BEAM = "ballistic_arc_beam"
    HEX_LINE_PIERCE = "hex_line_pierce"


class SectorAnomalyType(str, Enum):
    DIMENSIONAL_RIFT = "dimensional_rift"
    ENTROPY_STORM = "entropy_storm"
    LEY_LINE_SURGE = "ley_line_surge"
    VOID_CORRUPTION_ZONE = "void_corruption_zone"
    CHRONO_DILATION_WARP = "chrono_dilation_warp"


from dataclasses import dataclass, asdict

class TotemStatus(str, Enum):
    ATTUNED = "attuned"
    OVERCHARGED = "overcharged"
    CORRUPTED = "corrupted"
    NULLIFIED = "nullified"
    DORMANT = "dormant"


@dataclass
class SectorTotem:
    id: str
    name: str
    sector_id: str
    element: str
    hex_coords: List[int]
    world_pos: List[float]
    base_frequency_hz: float = 432.0
    current_frequency_hz: float = 432.0
    hp: int = 6
    max_hp: int = 6
    ward_shield: int = 10
    resonance_charge: float = 75.0
    aura_radius_hex: int = 2
    buff_effect: str = ""
    status: str = TotemStatus.ATTUNED.value

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── 2. VISUAL PHENOMENA & ATMOSPHERIC GENERATOR ───────────────────────────────
class VisualPhenomenaEngine:
    """
    Generates procedural shader parameters, volumetric uniforms, and particle emitter
    presets for atmospheric and magical visual phenomena in the sector.
    """

    @staticmethod
    def generate_phenomenon(
        phenomenon_type: str,
        coordinates: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        intensity: float = 0.75,
        resonance_hz: float = 432.0
    ) -> Dict[str, Any]:
        p_id = f"phenom_{uuid.uuid4().hex[:8]}"
        norm_intensity = max(0.1, min(1.0, intensity))

        # Uniform parameters for Godot 4 VisualShader / GPUParticles3D
        if phenomenon_type == VisualPhenomenonType.CHROMATIC_ABERRATION_BURST.value:
            shader_params = {
                "distortion_amplitude": round(0.04 * norm_intensity, 4),
                "chroma_shift_rgb": [round(0.015 * norm_intensity, 4), 0.0, round(-0.015 * norm_intensity, 4)],
                "lens_curvature": round(0.25 * norm_intensity, 3),
                "glow_color_rgba": [0.4, 0.98, 0.95, round(0.85 * norm_intensity, 2)],
                "pulse_frequency_hz": round(resonance_hz / 100.0, 2)
            }
            emitter_spec = {
                "particle_type": "optical_lens_dust",
                "particle_count": int(64 * norm_intensity),
                "emission_shape": "sphere_radius_3m",
                "lifetime_sec": 1.4
            }
        elif phenomenon_type == VisualPhenomenonType.AETHERIC_AURORA_STREAM.value:
            shader_params = {
                "aurora_speed": round(0.65 * norm_intensity, 3),
                "aurora_vertical_wave": round(2.8 * norm_intensity, 2),
                "plasma_intensity": round(0.85 * norm_intensity, 3),
                "glow_color_rgba": [0.61, 0.31, 0.87, round(0.9 * norm_intensity, 2)],
                "noise_texture_scale": 1.85
            }
            emitter_spec = {
                "particle_type": "aurora_spark_stream",
                "particle_count": int(120 * norm_intensity),
                "emission_shape": "ribbon_mesh_path",
                "lifetime_sec": 3.2
            }
        elif phenomenon_type == VisualPhenomenonType.ST_ELMOS_PLASMA_DISCHARGE.value:
            shader_params = {
                "arc_branch_count": int(5 * norm_intensity),
                "arc_jitter": round(0.35 * norm_intensity, 3),
                "plasma_lumens": round(1500.0 * norm_intensity, 1),
                "glow_color_rgba": [1.0, 0.84, 0.0, round(0.95 * norm_intensity, 2)],
                "corona_discharge_radius": round(2.5 * norm_intensity, 2)
            }
            emitter_spec = {
                "particle_type": "electric_spark_sparks",
                "particle_count": int(90 * norm_intensity),
                "emission_shape": "totem_crown_emitter",
                "lifetime_sec": 0.6
            }
        elif phenomenon_type == VisualPhenomenonType.GRAVITATIONAL_LENS_WARP.value:
            shader_params = {
                "gravitational_radius": round(4.5 * norm_intensity, 2),
                "vorticity_spin": round(1.8 * norm_intensity, 3),
                "event_horizon_fade": 0.12,
                "glow_color_rgba": [0.15, 0.15, 0.25, round(0.8 * norm_intensity, 2)],
                "spatial_compression": round(1.6 * norm_intensity, 2)
            }
            emitter_spec = {
                "particle_type": "singularity_accretion_dust",
                "particle_count": int(80 * norm_intensity),
                "emission_shape": "torus_ring_4m",
                "lifetime_sec": 2.0
            }
        else:  # VOLUMETRIC_VOID_FOG
            shader_params = {
                "fog_density": round(0.65 * norm_intensity, 3),
                "void_absorption_coef": round(0.82 * norm_intensity, 3),
                "light_scattering": 0.25,
                "glow_color_rgba": [0.1, 0.05, 0.18, round(0.9 * norm_intensity, 2)],
                "height_falloff": 0.4
            }
            emitter_spec = {
                "particle_type": "void_smoke_billow",
                "particle_count": int(45 * norm_intensity),
                "emission_shape": "hex_box_volume",
                "lifetime_sec": 4.5
            }

        return {
            "phenomenon_id": p_id,
            "type": phenomenon_type,
            "world_coordinates": list(coordinates),
            "intensity": norm_intensity,
            "resonance_frequency_hz": resonance_hz,
            "godot_shader_uniforms": shader_params,
            "gpu_particles_spec": emitter_spec,
            "created_at": time.time()
        }

    @staticmethod
    def get_all_phenomena_catalog() -> Dict[str, Any]:
        return {
            p.value: VisualPhenomenaEngine.generate_phenomenon(p.value, (0.0, 0.0, 0.0), 0.8)
            for p in VisualPhenomenonType
        }


# ── 3. SPELL PROJECTION ENGINE ────────────────────────────────────────────────
class SpellProjectionEngine:
    """
    Computes spatial projections of cast spells onto the hex grid,
    calculating geometric intersections with hexes and sector totems.
    """

    SPELL_CATALOG = {
        "frost_crystal_nova": {
            "name": "Mrazivá Kryštálová Nova",
            "type": SpellProjectionType.CONICAL_AOE.value,
            "element": "crystal",
            "half_angle_deg": 45.0,
            "max_range_m": 12.0,
            "damage_potency": 4,
            "totem_attunement_affinity": "crystal",
            "visual_phenomenon": VisualPhenomenonType.CHROMATIC_ABERRATION_BURST.value
        },
        "toxic_miasma_eruption": {
            "name": "Erupcia Toxickej Miazmy",
            "type": SpellProjectionType.RADIAL_BLAST_DECAL.value,
            "element": "toxic",
            "radius_m": 8.0,
            "damage_potency": 3,
            "totem_attunement_affinity": "toxic",
            "visual_phenomenon": VisualPhenomenonType.VOLUMETRIC_VOID_FOG.value
        },
        "druid_thorn_wrath": {
            "name": "Hnev Prastarých Tŕňov",
            "type": SpellProjectionType.RADIAL_BLAST_DECAL.value,
            "element": "druid",
            "radius_m": 7.5,
            "damage_potency": 3,
            "totem_attunement_affinity": "druid",
            "visual_phenomenon": VisualPhenomenonType.AETHERIC_AURORA_STREAM.value
        },
        "aether_cosmic_lance": {
            "name": "Kozmická Aéterová Kopija",
            "type": SpellProjectionType.BALLISTIC_ARC_BEAM.value,
            "element": "nexus",
            "max_range_m": 18.0,
            "apex_height_m": 6.5,
            "damage_potency": 5,
            "totem_attunement_affinity": "nexus",
            "visual_phenomenon": VisualPhenomenonType.ST_ELMOS_PLASMA_DISCHARGE.value
        }
    }

    @staticmethod
    def project_spell(
        spell_key: str,
        caster_origin: Tuple[float, float],
        target_heading_deg: float = 0.0,
        totems_in_sector: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        spec = SpellProjectionEngine.SPELL_CATALOG.get(spell_key, SpellProjectionEngine.SPELL_CATALOG["frost_crystal_nova"])
        proj_type = spec["type"]
        heading_rad = math.radians(target_heading_deg)
        ox, oy = caster_origin

        hit_totems = []
        totems_list = totems_in_sector or []

        for totem in totems_list:
            tx, ty = totem.get("world_pos", (0.0, 0.0))
            dx = tx - ox
            dy = ty - oy
            dist = math.sqrt(dx * dx + dy * dy)

            is_hit = False
            hit_factor = 0.0

            if proj_type == SpellProjectionType.CONICAL_AOE.value:
                half_angle_rad = math.radians(spec.get("half_angle_deg", 45.0))
                max_range = spec.get("max_range_m", 12.0)
                if dist <= max_range and dist > 0.001:
                    angle_to_target = math.atan2(dy, dx)
                    diff = abs(angle_to_target - heading_rad)
                    if diff > math.pi:
                        diff = 2.0 * math.pi - diff
                    if diff <= half_angle_rad:
                        is_hit = True
                        hit_factor = round(1.0 - (dist / max_range) * 0.4, 3)

            elif proj_type == SpellProjectionType.RADIAL_BLAST_DECAL.value:
                radius = spec.get("radius_m", 8.0)
                if dist <= radius:
                    is_hit = True
                    hit_factor = round(1.0 - (dist / radius) * 0.5, 3)

            elif proj_type == SpellProjectionType.BALLISTIC_ARC_BEAM.value:
                max_range = spec.get("max_range_m", 18.0)
                # Check beam trajectory alignment
                if dist <= max_range and dist > 0.001:
                    angle_to_target = math.atan2(dy, dx)
                    diff = abs(angle_to_target - heading_rad)
                    if diff > math.pi:
                        diff = 2.0 * math.pi - diff
                    # Tight beam width (within 15 degrees)
                    if diff <= math.radians(15.0):
                        is_hit = True
                        hit_factor = round(1.0 - (diff / math.radians(15.0)) * 0.3, 3)

            if is_hit:
                hit_totems.append({
                    "totem_id": totem.get("id"),
                    "totem_name": totem.get("name"),
                    "distance_m": round(dist, 2),
                    "hit_effectiveness": hit_factor,
                    "affinity_matched": (spec["totem_attunement_affinity"] == totem.get("element")),
                    "element": spec["element"]
                })

        # Generate accompanying visual phenomenon
        phenomenon = VisualPhenomenaEngine.generate_phenomenon(
            phenomenon_type=spec["visual_phenomenon"],
            coordinates=(ox, oy, 1.5),
            intensity=0.85
        )

        return {
            "projection_id": f"proj_{uuid.uuid4().hex[:8]}",
            "spell_key": spell_key,
            "spell_name": spec["name"],
            "projection_type": proj_type,
            "origin": list(caster_origin),
            "heading_degrees": target_heading_deg,
            "damage_potency": spec["damage_potency"],
            "totems_affected_count": len(hit_totems),
            "affected_totems": hit_totems,
            "visual_phenomenon": phenomenon
        }


# ── 4. SPATIAL ANOMALY DETECTION ENGINE ───────────────────────────────────────
class AnomalyDetectorSensorArray:
    """
    Sensor array detecting dimensional anomalies, entropy spikes, and ley-line shifts
    across sector quadrants through gradient divergence and resonance flux.
    """

    def __init__(self):
        self.active_anomalies: Dict[str, Dict[str, Any]] = {}
        self.sensor_calibration_qps: float = 120.0

    def trigger_anomaly(
        self,
        sector_id: str,
        anomaly_type: str,
        epicenter_coords: Tuple[float, float],
        magnitude: float = 0.8,
        duration_sec: float = 60.0
    ) -> Dict[str, Any]:
        anomaly_id = f"anom_{uuid.uuid4().hex[:8]}"
        now = time.time()
        mag = max(0.1, min(1.0, magnitude))

        anomaly = {
            "id": anomaly_id,
            "sector_id": sector_id,
            "type": anomaly_type,
            "epicenter": list(epicenter_coords),
            "magnitude": mag,
            "detected_at": now,
            "expires_at": now + duration_sec,
            "frequency_drift_hz": round(mag * 48.5, 2),
            "gradient_divergence": round(mag * 3.1415, 3),
            "active": True
        }
        self.active_anomalies[anomaly_id] = anomaly
        return anomaly

    def scan_sector(self, sector_id: str, sector_totems: List[Dict[str, Any]]) -> Dict[str, Any]:
        now = time.time()
        # Clean expired
        active = [a for a in self.active_anomalies.values() if a["sector_id"] == sector_id and a["expires_at"] > now]

        readings = []
        for anom in active:
            ax, ay = anom["epicenter"]
            severity = "MINOR_FLUCTUATION"
            if anom["magnitude"] >= 0.8:
                severity = "CRITICAL_CATACLYSM"
            elif anom["magnitude"] >= 0.5:
                severity = "MODERATE_SURGE"

            # Check totems within anomaly radius (10m)
            threatened_totems = []
            for t in sector_totems:
                tx, ty = t.get("world_pos", (0.0, 0.0))
                dist = math.sqrt((tx - ax) ** 2 + (ty - ay) ** 2)
                if dist <= 10.0:
                    threatened_totems.append({
                        "totem_id": t.get("id"),
                        "distance_m": round(dist, 2),
                        "resonance_interference_pct": round((1.0 - dist / 10.0) * anom["magnitude"] * 100.0, 1)
                    })

            readings.append({
                "anomaly_id": anom["id"],
                "type": anom["type"],
                "severity": severity,
                "magnitude": anom["magnitude"],
                "gradient_flux": anom["gradient_divergence"],
                "frequency_drift_hz": anom["frequency_drift_hz"],
                "threatened_totems": threatened_totems
            })

        return {
            "sector_id": sector_id,
            "scan_timestamp": now,
            "anomalies_detected_count": len(readings),
            "sensor_readings": readings,
            "sector_stability_index": round(max(0.0, 100.0 - sum(r["magnitude"] * 40.0 for r in readings)), 1)
        }

    def get_active_anomalies(self) -> List[Dict[str, Any]]:
        now = time.time()
        return [a for a in self.active_anomalies.values() if a.get("expires_at", 0) > now]


# ── 5. SECTOR TOTEM MANAGER & STATE MODULATION ────────────────────────────────
class SectorTotemManager:
    """
    Manages the physical presence, resonance attunement, vital 6 HP constraint,
    and spell/anomaly modifications of totems in each strategic sector.
    """

    def __init__(self):
        self.totems: Dict[str, Dict[str, Any]] = {}
        self._seed_default_sector_totems()

    def _seed_default_sector_totems(self):
        defaults = [
            {
                "id": "totem_north_crystal",
                "name": "Kryštálový Rezonančný Totem",
                "sector_id": "sector_north_crystal",
                "element": "crystal",
                "hex_coords": [0, -1],
                "world_pos": [0.0, -8.66],
                "base_frequency_hz": 432.0,
                "current_frequency_hz": 432.0,
                "hp": 6,
                "max_hp": 6,
                "ward_shield": 8,
                "resonance_charge": 75.0,
                "aura_radius_hex": 2,
                "buff_effect": "+15% Zisk Aéterových Kryštálov & +1 k Ward Regenerácii",
                "status": TotemStatus.ATTUNED.value
            },
            {
                "id": "totem_south_toxic",
                "name": "Toxický Miazmatický Totem",
                "sector_id": "sector_south_toxic",
                "element": "toxic",
                "hex_coords": [0, 1],
                "world_pos": [0.0, 8.66],
                "base_frequency_hz": 216.5,
                "current_frequency_hz": 216.5,
                "hp": 6,
                "max_hp": 6,
                "ward_shield": 6,
                "resonance_charge": 60.0,
                "aura_radius_hex": 2,
                "buff_effect": "Kyslé Miazma: -1 k Obrannému postoju nepriateľov v sektore",
                "status": TotemStatus.ATTUNED.value
            },
            {
                "id": "totem_east_druid",
                "name": "Druidský Živototvorný Totem",
                "sector_id": "sector_east_druid",
                "element": "druid",
                "hex_coords": [1, 0],
                "world_pos": [7.5, 0.0],
                "base_frequency_hz": 528.0,
                "current_frequency_hz": 528.0,
                "hp": 6,
                "max_hp": 6,
                "ward_shield": 10,
                "resonance_charge": 80.0,
                "aura_radius_hex": 3,
                "buff_effect": "Prastaré Korene: Pasívne liečenie +1 HP každé 2 kolá",
                "status": TotemStatus.ATTUNED.value
            },
            {
                "id": "totem_west_sulfur",
                "name": "Sírový Pyromantický Totem",
                "sector_id": "sector_west_sulfur",
                "element": "sulfur",
                "hex_coords": [-1, 0],
                "world_pos": [-7.5, 0.0],
                "base_frequency_hz": 360.0,
                "current_frequency_hz": 360.0,
                "hp": 6,
                "max_hp": 6,
                "ward_shield": 5,
                "resonance_charge": 50.0,
                "aura_radius_hex": 2,
                "buff_effect": "Termálny Pretlak: +2 k poškodeniu moždiarov a delostrelectva",
                "status": TotemStatus.ATTUNED.value
            },
            {
                "id": "totem_center_nexus",
                "name": "Centrálny Aéterový Nexus Totem",
                "sector_id": "sector_center_citadel",
                "element": "nexus",
                "hex_coords": [0, 0],
                "world_pos": [0.0, 0.0],
                "base_frequency_hz": 864.0,
                "current_frequency_hz": 864.0,
                "hp": 6,
                "max_hp": 6,
                "ward_shield": 15,
                "resonance_charge": 90.0,
                "aura_radius_hex": 4,
                "buff_effect": "Kozmické Spojenie: Globálna koordinácia 12 Apoštolov & NPU akcelerácia",
                "status": TotemStatus.OVERCHARGED.value
            }
        ]

        for t in defaults:
            self.totems[t["id"]] = t

    def get_totems(self, sector_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if sector_id:
            return [t for t in self.totems.values() if t["sector_id"] == sector_id]
        return list(self.totems.values())

    def apply_spell_impact_to_totem(
        self,
        totem_id: str,
        spell_element: str,
        potency: float,
        hit_factor: float
    ) -> Dict[str, Any]:
        totem = self.totems.get(totem_id)
        if not totem:
            return {"success": False, "error": f"Totem {totem_id} neexistuje."}

        is_affinity_aligned = (totem["element"] == spell_element)

        if is_affinity_aligned:
            # Overcharge resonance boost
            delta_charge = potency * hit_factor * 12.0
            totem["resonance_charge"] = min(100.0, round(totem["resonance_charge"] + delta_charge, 1))
            if totem["resonance_charge"] >= 85.0:
                totem["status"] = TotemStatus.OVERCHARGED.value
                totem["aura_radius_hex"] = max(totem["aura_radius_hex"], 3)
            event_type = "OVERCHARGE_ATTUNEMENT"
        else:
            # Opposition impact: damages ward shield or alters frequency
            shield_damage = int(potency * hit_factor * 2)
            totem["ward_shield"] = max(0, totem["ward_shield"] - shield_damage)
            if totem["ward_shield"] == 0:
                # Vital 6 HP intact unless shield broken
                totem["hp"] = max(1, totem["hp"] - 1)  # Strict minimum 1 to maintain 6 HP invariant
            totem["resonance_charge"] = max(0.0, round(totem["resonance_charge"] - (potency * 5.0), 1))
            if totem["resonance_charge"] <= 15.0:
                totem["status"] = TotemStatus.NULLIFIED.value
            event_type = "DISRUPTION_DAMAGE"

        return {
            "success": True,
            "event": event_type,
            "totem": totem,
            "is_aligned": is_affinity_aligned,
            "new_resonance": totem["resonance_charge"],
            "new_status": totem["status"]
        }

    def apply_anomaly_flux_to_totem(
        self,
        totem_id: str,
        anomaly_type: str,
        magnitude: float
    ) -> Dict[str, Any]:
        totem = self.totems.get(totem_id)
        if not totem:
            return {"success": False, "error": f"Totem {totem_id} neexistuje."}

        drift = round(magnitude * 25.0, 2)
        totem["current_frequency_hz"] = round(totem["base_frequency_hz"] + drift, 2)

        if anomaly_type == SectorAnomalyType.VOID_CORRUPTION_ZONE.value:
            totem["status"] = TotemStatus.CORRUPTED.value
            totem["buff_effect"] = "KORUPCIA: Negatívna aura vysáva -1 HP za kolo!"
        elif anomaly_type == SectorAnomalyType.LEY_LINE_SURGE.value:
            totem["resonance_charge"] = 100.0
            totem["status"] = TotemStatus.OVERCHARGED.value
            totem["buff_effect"] = "MASÍVNY PRETLAK: +50% k poškodeniu kúziel spojencov"
        elif anomaly_type == SectorAnomalyType.CHRONO_DILATION_WARP.value:
            totem["status"] = TotemStatus.NULLIFIED.value
            totem["buff_effect"] = "ČASOVÉ ZMRAZENIE: Aura dočasne znehybnená"

        return {
            "success": True,
            "totem_id": totem_id,
            "anomaly_type": anomaly_type,
            "frequency_drift_hz": drift,
            "totem_state": totem
        }

    def attune_totem(
        self,
        totem_id: str,
        caster_tribe: str = "crystal",
        channel_energy: float = 25.0
    ) -> Dict[str, Any]:
        totem = self.totems.get(totem_id)
        if not totem:
            return {"success": False, "error": f"Totem {totem_id} neexistuje."}

        totem["resonance_charge"] = min(100.0, round(totem["resonance_charge"] + channel_energy, 1))
        totem["current_frequency_hz"] = totem["base_frequency_hz"]
        totem["ward_shield"] = min(15, totem["ward_shield"] + int(channel_energy / 5.0))
        # Enforce vital 6 max HP invariant
        totem["hp"] = min(6, totem["hp"] + 1)

        if totem["resonance_charge"] >= 85.0:
            totem["status"] = TotemStatus.OVERCHARGED.value
            totem["buff_effect"] = "PRETLAKOVANÁ REZONANCIA: +50% dosah aury & zdvojené bonusy sektora"
        else:
            totem["status"] = TotemStatus.ATTUNED.value
            totem["buff_effect"] = f"HARMONICKÁ REZONANCIA: Naladené kmeňom {caster_tribe.upper()}"

        return {
            "success": True,
            "totem_id": totem_id,
            "resonance_charge": totem["resonance_charge"],
            "current_frequency_hz": totem["current_frequency_hz"],
            "ward_shield": totem["ward_shield"],
            "hp": totem["hp"],
            "status": totem["status"],
            "totem": totem
        }

    def reset_totems(self):
        self.totems.clear()
        self._seed_default_sector_totems()


# Global Singletons
GLOBAL_VISUAL_PHENOMENA = VisualPhenomenaEngine()
GLOBAL_SPELL_PROJECTION = SpellProjectionEngine()
GLOBAL_ANOMALY_DETECTOR = AnomalyDetectorSensorArray()
GLOBAL_TOTEM_MANAGER = SectorTotemManager()
