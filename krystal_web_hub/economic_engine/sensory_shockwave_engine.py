# ==============================================================================
# KRYSTAL-STACK: SENSORY SHOCKWAVE & IMPULSE TASK PROCESSING ENGINE
# ==============================================================================
# Implements:
#   1. SENSORY MODALITIES:
#      - Kinetic / Overpressure wave (Tlaková / nárazová vlna $\Delta P$).
#      - Acoustic blast (Akustický tresk / rázový hluk).
#      - Seismic ground tremor (Seizmický otras terénu).
#      - Optical flash (Svetelný záblesk / oslepenie).
#      - Aetheric resonance (Aéterová magická frekvencia).
#   2. WAVE PROPAGATION & TRENCH DEFLECTION:
#      - Radial propagation delay: t_arrival = distance / v_wave.
#      - Attenuation & dissipation: A(d) = A0 / (1 + alpha * d^1.8).
#      - Sapper trench barrier dampening: Parapets reflect 75% of overpressure
#        upward and away from entrenched combatants.
#   3. IMPULSE TASKS (NÁRAZOVÉ ÚLOHY):
#      - Concussion & Stagger check (Otras mozgu a dezorientácia).
#      - Tactical Suppression (Zníženie Presnosti a Uhýbania o -20% až -40%).
#      - Trench Structural Integrity Degradation (Erózia zákopov).
#      - Cognitive Stress Influx (Hypervigilance / Regression pod paľbou).
#      - Damage Resolution strictly clamped to VITAL_MAX_HP = 6.
# ==============================================================================

from __future__ import annotations

import math
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple

VITAL_MAX_HP: int = 6


class SensoryModality(str, Enum):
    KINETIC_OVERPRESSURE = "kinetic_overpressure" # Blast overpressure (kPa)
    ACOUSTIC_BLAST = "acoustic_blast"             # Sound pressure level (dB)
    SEISMIC_TREMOR = "seismic_tremor"             # Ground acceleration (m/s^2)
    OPTICAL_FLASH = "optical_flash"               # Lux / luminous intensity
    AETHERIC_RESONANCE = "aetheric_resonance"     # Mana harmonic frequency (Hz)


class CognitiveStressState(str, Enum):
    STABLE = "STABLE"
    HYPERVIGILANCE = "HYPERVIGILANCE" # Heightened alertness, fast twitch, panic aim
    SUPPRESSED = "SUPPRESSED"         # Ducked down, unable to take offensive action
    REGRESSION = "REGRESSION"         # Severe concussion / fetal position on trench floor


class SensoryWaveEmitter:
    """
    Computes emission parameters for detonations, artillery impacts,
    and heavy ballistic clashes in the tactical arena.
    """

    # Propagation velocities in standard atmospheric conditions (m/s)
    VELOCITIES = {
        SensoryModality.KINETIC_OVERPRESSURE.value: 480.0,  # Supersonic blast wave
        SensoryModality.ACOUSTIC_BLAST.value: 343.0,        # Speed of sound
        SensoryModality.SEISMIC_TREMOR.value: 220.0,        # Rayleigh ground waves in soil
        SensoryModality.OPTICAL_FLASH.value: 299792458.0,   # Speed of light
        SensoryModality.AETHERIC_RESONANCE.value: 750.0     # Sub-space mana waves
    }

    @staticmethod
    def emit_shockwave(
        epicenter_hex: List[int],
        modality: SensoryModality = SensoryModality.KINETIC_OVERPRESSURE,
        peak_intensity: float = 120.0, # kPa for overpressure
        blast_radius_m: float = 8.5
    ) -> Dict[str, Any]:
        return {
            "epicenter_hex": epicenter_hex,
            "modality": modality.value,
            "peak_intensity": round(peak_intensity, 2),
            "blast_radius_m": round(blast_radius_m, 2),
            "velocity_m_s": SensoryWaveEmitter.VELOCITIES.get(modality.value, 343.0),
            "timestamp": time.time()
        }


class ShockwavePropagationEngine:
    """
    Calculates how wave fronts propagate across the hexagonal arena,
    accounting for distance attenuation and sapper trench deflections.
    """

    HEX_DIAMETER_METERS: float = 3.46 # 2 * sqrt(3) meters

    @staticmethod
    def calculate_hex_distance_meters(hex_a: List[int], hex_b: List[int]) -> float:
        # Axial distance: max(|dq|, |dr|, |dq+dr|)
        dq = hex_a[0] - hex_b[0]
        dr = hex_a[1] - hex_b[1]
        axial_dist = max(abs(dq), abs(dr), abs(dq + dr))
        return axial_dist * ShockwavePropagationEngine.HEX_DIAMETER_METERS

    @staticmethod
    def propagate_wave_to_target(
        wave: Dict[str, Any],
        target_hex: List[int],
        cover_density: float = 0.15,
        is_trench: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates received wave parameters at target_hex:
          - Arrival delay (ms)
          - Dissipated wave intensity
          - Sapper trench barrier dampening factor
        """
        dist_m = ShockwavePropagationEngine.calculate_hex_distance_meters(
            wave["epicenter_hex"], target_hex
        )
        v = wave.get("velocity_m_s", 343.0)
        delay_ms = round((dist_m / max(1.0, v)) * 1000.0, 2)

        peak_i = wave.get("peak_intensity", 100.0)
        blast_r = wave.get("blast_radius_m", 8.0)

        # Attenuation law: A(d) = A0 / (1 + (d / r)^1.8)
        ratio = dist_m / max(0.5, blast_r)
        attenuation_factor = 1.0 / (1.0 + math.pow(ratio, 1.8))
        raw_intensity = peak_i * attenuation_factor

        # Sapper Trench Parapet Deflection:
        # A deep trench with a parapet (cover_density >= 0.75 or is_trench=True)
        # defuses kinetic overpressure and acoustic concussion by 75-85%.
        trench_dampening = 0.0
        effective_intensity = raw_intensity
        if is_trench or cover_density >= 0.75:
            trench_dampening = 0.80 # 80% deflected upwards over the trench lip
            effective_intensity = raw_intensity * (1.0 - trench_dampening)

        is_direct_hit = (dist_m < ShockwavePropagationEngine.HEX_DIAMETER_METERS * 0.8)

        return {
            "target_hex": target_hex,
            "distance_meters": round(dist_m, 2),
            "arrival_delay_ms": delay_ms,
            "raw_intensity": round(raw_intensity, 2),
            "trench_dampening_pct": round(trench_dampening * 100.0, 1),
            "effective_intensity": round(effective_intensity, 2),
            "is_direct_hit": is_direct_hit,
            "is_sheltered_by_trench": (trench_dampening > 0.0)
        }


class ImpulseTaskType(str, Enum):
    CONCUSSION_STAGGER = "concussion_stagger"         # Balance and orientation loss
    TACTICAL_SUPPRESSION = "tactical_suppression"     # Aim and mobility penalty
    TERRAIN_EROSION = "terrain_erosion"               # Soil displacement / trench wear
    COGNITIVE_OVERLOAD = "cognitive_overload"         # Panic / reflex degradation
    VITAL_DAMAGE_ABSORPTION = "vital_damage_absorption" # Net damage to 6 HP pool


class ImpulseTaskGenerator:
    """
    Transforms arriving shockwave fronts into structured 'Nárazové úlohy'
    that the game engine and cognitive processor must execute.
    """

    @staticmethod
    def generate_impulse_tasks(
        combatant: Dict[str, Any],
        wave_reception: Dict[str, Any],
        round_number: int
    ) -> List[Dict[str, Any]]:
        tasks = []
        eff_i = wave_reception["effective_intensity"]
        sheltered = wave_reception["is_sheltered_by_trench"]

        # 1. Concussion & Stagger Task
        stagger_probability = min(0.95, eff_i / 80.0)
        if sheltered:
            stagger_probability *= 0.25 # Much lower in trench

        tasks.append({
            "task_id": f"IMP-CONC-{round_number}-{combatant.get('unit_id', 'hero')}",
            "task_type": ImpulseTaskType.CONCUSSION_STAGGER.value,
            "priority_rank": 1, # Highest execution urgency
            "intensity": eff_i,
            "stagger_probability": round(stagger_probability, 3),
            "concussion_applied": (stagger_probability > 0.35),
            "description": f"Otras mozgu a dezorientácia jednotky {combatant.get('name', 'Bojovník')}."
        })

        # 2. Tactical Suppression Task
        suppression_pct = 0
        if eff_i > 25.0:
            suppression_pct = int(min(50, (eff_i - 25.0) * 0.8))
        if sheltered:
            suppression_pct = int(suppression_pct * 0.30)

        tasks.append({
            "task_id": f"IMP-SUPP-{round_number}-{combatant.get('unit_id', 'hero')}",
            "task_type": ImpulseTaskType.TACTICAL_SUPPRESSION.value,
            "priority_rank": 2,
            "aim_penalty_pct": suppression_pct,
            "dodge_penalty_pct": int(suppression_pct * 0.75),
            "duration_rounds": 1,
            "description": f"Taktické potlačenie paľbou: -{suppression_pct}% k Presnosti."
        })

        # 3. Cognitive Overload & Coping Mechanism Shift
        if eff_i > 70.0 and not sheltered:
            coping_state = CognitiveStressState.REGRESSION
        elif eff_i > 40.0:
            coping_state = CognitiveStressState.HYPERVIGILANCE
        elif suppression_pct > 20:
            coping_state = CognitiveStressState.SUPPRESSED
        else:
            coping_state = CognitiveStressState.STABLE

        tasks.append({
            "task_id": f"IMP-COGN-{round_number}-{combatant.get('unit_id', 'hero')}",
            "task_type": ImpulseTaskType.COGNITIVE_OVERLOAD.value,
            "priority_rank": 3,
            "stress_state": coping_state.value,
            "description": f"Kognitívna odozva: {coping_state.value}."
        })

        # 4. Vital Damage Absorption Task (Clamped to 6 Max HP)
        net_damage = 0
        if wave_reception["is_direct_hit"]:
            net_damage = 2 if not sheltered else 0
        elif eff_i > 50.0:
            net_damage = 1 if not sheltered else 0

        cur_hp = combatant.get("current_hp", VITAL_MAX_HP)
        hp_after = max(0, min(VITAL_MAX_HP, cur_hp - net_damage))

        tasks.append({
            "task_id": f"IMP-DMG-{round_number}-{combatant.get('unit_id', 'hero')}",
            "task_type": ImpulseTaskType.VITAL_DAMAGE_ABSORPTION.value,
            "priority_rank": 1,
            "raw_damage": net_damage,
            "current_hp_before": cur_hp,
            "current_hp_after": hp_after,
            "vital_max_hp": VITAL_MAX_HP,
            "description": f"Kinetické poškodenie: -{net_damage} HP (Zostáva: {hp_after}/6 HP)."
        })

        return tasks


class SensoryShockwaveEngine:
    """
    Master processor linking sensory wave physics with tactical combat
    and Krystal Kernel micro-slice burst task scheduling.
    """

    def __init__(self):
        self.active_waves: List[Dict[str, Any]] = []
        self.processed_burst_tasks: List[Dict[str, Any]] = []

    def trigger_tactical_detonation(
        self,
        epicenter_hex: List[int],
        combatants: List[Dict[str, Any]],
        cover_map: Dict[str, float],
        trenches: List[str],
        round_number: int,
        peak_intensity_kpa: float = 125.0,
        blast_radius_m: float = 9.0
    ) -> Dict[str, Any]:
        """
        Executes complete sensory wave cycle:
          1. Emits kinetic overpressure + acoustic shockwave.
          2. Propagates wavefronts to all combatants across the hex board.
          3. Generates and triages impulse tasks (Nárazové úlohy).
          4. Applies suppression, concussion, and 6 HP invariant damage.
        """
        # Step 1: Emit shockwave
        wave = SensoryWaveEmitter.emit_shockwave(
            epicenter_hex=epicenter_hex,
            modality=SensoryModality.KINETIC_OVERPRESSURE,
            peak_intensity=peak_intensity_kpa,
            blast_radius_m=blast_radius_m
        )
        self.active_waves.append(wave)

        # Step 2 & 3: Propagate to each combatant and generate impulse tasks
        evaluation_results = []
        all_tasks = []

        for unit in combatants:
            u_hex = unit.get("current_hex", [0, 0])
            hex_key = f"{u_hex[0]}_{u_hex[1]}"
            cov_density = cover_map.get(hex_key, 0.15)
            is_trench = (hex_key in trenches)

            reception = ShockwavePropagationEngine.propagate_wave_to_target(
                wave=wave,
                target_hex=u_hex,
                cover_density=cov_density,
                is_trench=is_trench
            )

            tasks = ImpulseTaskGenerator.generate_impulse_tasks(
                combatant=unit,
                wave_reception=reception,
                round_number=round_number
            )

            # Apply side-effects to combatant state
            dmg_task = next((t for t in tasks if t["task_type"] == ImpulseTaskType.VITAL_DAMAGE_ABSORPTION.value), None)
            supp_task = next((t for t in tasks if t["task_type"] == ImpulseTaskType.TACTICAL_SUPPRESSION.value), None)
            cogn_task = next((t for t in tasks if t["task_type"] == ImpulseTaskType.COGNITIVE_OVERLOAD.value), None)

            if dmg_task:
                unit["current_hp"] = dmg_task["current_hp_after"]
            if supp_task:
                unit["active_suppression_pct"] = supp_task["aim_penalty_pct"]
            if cogn_task:
                unit["cognitive_stress_state"] = cogn_task["stress_state"]

            evaluation_results.append({
                "unit_id": unit.get("unit_id"),
                "name": unit.get("name"),
                "hex": u_hex,
                "wave_reception": reception,
                "impulse_tasks": tasks,
                "resulting_hp": unit.get("current_hp"),
                "vital_max_hp": VITAL_MAX_HP
            })
            all_tasks.extend(tasks)

        self.processed_burst_tasks.extend(all_tasks)

        return {
            "status": "PROCESSED",
            "round_number": round_number,
            "shockwave_spec": wave,
            "evaluations": evaluation_results,
            "total_impulse_tasks_dispatched": len(all_tasks),
            "vital_max_hp": VITAL_MAX_HP
        }


GLOBAL_SENSORY_SHOCKWAVE_ENGINE = SensoryShockwaveEngine()
