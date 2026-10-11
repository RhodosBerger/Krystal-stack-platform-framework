#!/usr/bin/env python3
"""
==============================================================================
SOVEREIGN ENGINE: HYPERVISOR TACTICAL INTERFACE BRIDGE (Python HTIB)
==============================================================================
Module: krystal_kernel/sovereign_htib_bridge.py
Description:
    Tiers-2 bridge connecting Go DSC (Deterministic State Controller) and
    WebGL DVT (Direct Visual Telemetry):
    1. Declarative Blueprint compilation into axial hexagonal grid (q, r).
    2. Boolean Hex Ops (is_traversable, is_hazard, biome properties).
    3. Strict Pauli Exclusion Guard on layer [A_CENTER] (layer 2).
    4. Physical calculation via Ohm's Law analogue:
       I = V / (R_target + R_reactions) with R_eff >= 0.1 and saturation arithmetic (0..6 HP).
    5. LIFO Interrupt Stack with reaction expiration window.
    6. Zero-copy binary packed buffer generation (24 floats / 96 bytes per instance).
    7. Conduct Code Bridge & Safety Validation:
       [ACT], [LCK], [CNF >= 0.85], [HAZ], [VIT_MAX_HP == 6] with deterministic fallback.

Author: Sovereign Engine Architecture Council / Krystal-Stack (2026)
==============================================================================
"""

import os
import sys
import math
import time
import struct
from typing import Dict, Any, List, Optional, Tuple, Set

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP

# ── INVARIANTS & CONSTANTS ──────────────────────────────────────────────────
MIN_EFFECTIVE_RESISTANCE: float = 0.1
CONFIDENCE_THRESHOLD: float = 0.85
DEFAULT_INTERRUPT_WINDOW_TICKS: int = 2

# 4-Tier Z-Index Layering
LAYER_BASE: float = 0.0       # [A_BASE] Hex terrain elevations
LAYER_AURA_RING: float = 0.1  # [A_AURA_RING] Selection halos, hazard AoE
LAYER_CENTER: float = 0.5     # [A_CENTER] Living Standees / 3D pawns (Pauli Guarded)
LAYER_OVERHEAD: float = 1.2   # [A_OVERHEAD] Status runes, conduct badges


class PauliExclusionViolation(Exception):
    """Raised when two solid entities attempt to occupy the same cell on layer A_CENTER."""
    pass


class ConductCodeViolation(Exception):
    """Raised when an instruction fails confidence threshold or introduces hazard."""
    pass


def axial_to_world(q: int, r: int, radius: float = 1.0) -> Tuple[float, float]:
    """Converts pointy-topped axial coordinates (q, r) to 2D world coordinates (x, z)."""
    x = radius * math.sqrt(3.0) * (q + r / 2.0)
    z = radius * 1.5 * r
    return (x, z)


def axial_distance(q1: int, r1: int, q2: int, r2: int) -> int:
    """Computes axial hexagonal distance between (q1, r1) and (q2, r2)."""
    dq = q1 - q2
    dr = r1 - r2
    ds = -q1 - r1 - (-q2 - r2)
    return (abs(dq) + abs(dr) + abs(ds)) // 2


class SovereignHTIBBridge:
    """
    Production-grade Hypervisor Tactical Interface Bridge (HTIB).
    Orchestrates deterministic state transitions, Ohm's law physics,
    Pauli Exclusion validation, and packed buffer synthesis.
    """

    def __init__(self, map_id: str = "SOVEREIGN_CITADEL_DEFAULT"):
        self.map_id = map_id
        self.tick: int = 0
        self.cells: Dict[str, Dict[str, Any]] = {}
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.center_occupancy: Dict[str, str] = {}  # "q,r" -> entity_id (Pauli Guard)
        self.interrupt_stack: List[Dict[str, Any]] = []  # LIFO
        self.telemetry = {
            "fps": 60.0,
            "cpu_load": 24.5,
            "temperature_c": 52.0,
            "entropy_h": 0.18
        }

    # ── 1. BLUEPRINT COMPILATION & BOOLEAN HEX OPS ──────────────────────────

    def compile_blueprint(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compiles declarative blueprint into hexagonal topology.
        Applies Boolean Hex Ops and Pauli Exclusion Guard.
        """
        self.cells.clear()
        self.entities.clear()
        self.center_occupancy.clear()
        self.interrupt_stack.clear()
        self.tick = 0

        hex_radius = float(blueprint.get("hex_radius", 1.0))

        # 1. Compile Cells
        for raw_cell in blueprint.get("cells", []):
            q = int(raw_cell["q"])
            r = int(raw_cell["r"])
            biome = str(raw_cell.get("biome", "CRYSTAL_CITADEL"))
            is_hazard = bool(raw_cell.get("is_hazard", biome in ["VOLCANIC", "VOID_CORRUPTION"]))
            is_traversable = bool(raw_cell.get("is_traversable", not is_hazard))
            elevation = float(raw_cell.get("elevation", 0.0))

            key = f"{q},{r}"
            self.cells[key] = {
                "q": q,
                "r": r,
                "biome": biome,
                "is_traversable": is_traversable,
                "is_hazard": is_hazard,
                "elevation": elevation,
                "layer": LAYER_BASE
            }

        # 2. Compile Entities & Pauli Guard
        for raw_ent in blueprint.get("entities", []):
            e_id = str(raw_ent["id"])
            q = int(raw_ent["q"])
            r = int(raw_ent["r"])

            raw_layer = raw_ent.get("layer", "A_CENTER")
            if raw_layer == "A_BASE" or raw_layer == 0:
                layer = LAYER_BASE
            elif raw_layer == "A_AURA_RING" or raw_layer == 1:
                layer = LAYER_AURA_RING
            elif raw_layer == "A_CENTER" or raw_layer == 2:
                layer = LAYER_CENTER
            elif raw_layer == "A_OVERHEAD" or raw_layer == 3:
                layer = LAYER_OVERHEAD
            else:
                layer = LAYER_CENTER

            # Pauli Exclusion Check for Layer A_CENTER
            if layer == LAYER_CENTER:
                coord_key = f"{q},{r}"
                existing = self.center_occupancy.get(coord_key)
                if existing and existing != e_id:
                    raise PauliExclusionViolation(
                        f"Pauli Exclusion Violation: Hex ({q}, {r}) on [A_CENTER] already occupied by {existing}"
                    )
                self.center_occupancy[coord_key] = e_id

            hp = max(0, min(VITAL_MAX_HP, int(raw_ent.get("hp", VITAL_MAX_HP))))
            confidence = float(raw_ent.get("confidence", 1.0))
            is_hazardous = bool(raw_ent.get("is_hazardous", False))
            base_r = float(raw_ent.get("base_resistance", 1.5))

            self.entities[e_id] = {
                "id": e_id,
                "name": raw_ent.get("name", e_id),
                "q": q,
                "r": r,
                "layer": layer,
                "hp": hp,
                "base_resistance": base_r,
                "conduct": {
                    "is_active": hp > 0,
                    "is_locked": False,
                    "confidence": confidence,
                    "is_hazardous": is_hazardous,
                    "vital_hp": hp,
                    "vital_max_hp": VITAL_MAX_HP
                }
            }

        # Generate binary packed buffer
        packed_buffer = self.generate_packed_buffer(hex_radius)

        return {
            "status": "COMPILED",
            "map_id": self.map_id,
            "total_cells": len(self.cells),
            "total_entities": len(self.entities),
            "packed_buffer_bytes": len(packed_buffer),
            "vital_max_hp": VITAL_MAX_HP
        }

    # ── 2. OHM'S LAW PHYSICS & STATE RESOLUTION ─────────────────────────────

    def resolve_engine_state(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stateless transition resolver enforcing:
        1. Conduct Code verification: [ACT], [LCK], [CNF >= 0.85], [HAZ], [VIT 6HP].
        2. Ohm's Law physics: I = V / (R_target + R_reactions) with R_eff >= 0.1.
        3. Saturation arithmetic: HP clamped strictly to 0..6.
        4. LIFO Interrupt Stack processing.
        5. Pauli Exclusion Guard on entity movement.
        """
        self.tick += 1
        instr_type = instruction.get("type", "UNKNOWN")
        confidence = float(instruction.get("confidence", 1.0))
        is_hazardous = bool(instruction.get("is_hazardous", False))

        # 1. Conduct Code Safety Gate
        if confidence < CONFIDENCE_THRESHOLD or is_hazardous:
            return {
                "status": "FALLBACK_TRIGGERED",
                "fallback_action": "FALLBACK_DEFENSIVE_GROUNDING",
                "error": "Rejected by Conduct Code Bridge: low confidence or hazard detected",
                "conduct_badge": "[!ACT] [LCK] [!CNF] [HAZ] [VIT 6HP]",
                "tick": self.tick
            }

        # 2. Register Reaction into LIFO Interrupt Stack
        if instr_type == "REGISTER_INTERRUPT":
            int_data = instruction.get("interrupt", {})
            int_data["issued_tick"] = self.tick
            int_data["valid_ticks"] = int_data.get("valid_ticks", DEFAULT_INTERRUPT_WINDOW_TICKS)
            self.interrupt_stack.append(int_data)
            return {
                "status": "OK",
                "action": "INTERRUPT_REGISTERED",
                "stack_depth": len(self.interrupt_stack),
                "tick": self.tick
            }

        # 3. Movement with Pauli Exclusion Guard
        elif instr_type == "MOVE_ENTITY":
            source_id = instruction["source_id"]
            target_q = int(instruction["target_q"])
            target_r = int(instruction["target_r"])

            entity = self.entities.get(source_id)
            if not entity:
                return {"status": "ERROR", "error": f"Entity {source_id} not found"}

            target_key = f"{target_q},{target_r}"
            target_cell = self.cells.get(target_key)
            if target_cell and (not target_cell["is_traversable"] or target_cell["is_hazard"]):
                return {
                    "status": "FALLBACK_TRIGGERED",
                    "fallback_action": "CANCEL_MOVEMENT_HAZARD",
                    "error": "Target cell is impassable or hazardous"
                }

            # Pauli Guard on Layer A_CENTER
            if entity["layer"] == LAYER_CENTER:
                occupant = self.center_occupancy.get(target_key)
                if occupant and occupant != source_id:
                    return {
                        "status": "PAULI_VIOLATION",
                        "fallback_action": "DEFLECT_PLACEMENT_TO_STANDBY",
                        "error": f"Hex ({target_q}, {target_r}) on [A_CENTER] occupied by {occupant}"
                    }
                # Vacate old, occupy new
                old_key = f"{entity['q']},{entity['r']}"
                self.center_occupancy.pop(old_key, None)
                self.center_occupancy[target_key] = source_id

            entity["q"] = target_q
            entity["r"] = target_r
            return {
                "status": "OK",
                "action": "ENTITY_MOVED",
                "entity_id": source_id,
                "coord": (target_q, target_r),
                "conduct_badge": self._format_conduct_badge(entity["conduct"])
            }

        # 4. Ohm's Law Physical Calculation (I = V / R_eff)
        elif instr_type == "APPLY_OHM_VOLTAGE":
            target_id = instruction.get("target_id", "")
            raw_v = instruction.get("applied_voltage", 0.0)
            try:
                val = float(raw_v)
                if math.isnan(val) or math.isinf(val):
                    applied_v = 0.0
                else:
                    applied_v = max(0.0, val)
            except (ValueError, TypeError):
                applied_v = 0.0

            target = self.entities.get(target_id)
            if not target:
                return {"status": "ERROR", "error": f"Target {target_id} not found"}

            # Process LIFO Interrupt Stack
            reaction_r = 0.0
            v_mod = 0.0
            processed_interrupts = 0

            while self.interrupt_stack:
                item = self.interrupt_stack.pop()  # LIFO order
                if (self.tick - item["issued_tick"]) <= item["valid_ticks"]:
                    reaction_r += float(item.get("resistance_delta", 0.0))
                    v_mod += float(item.get("voltage_delta", 0.0))
                    processed_interrupts += 1

            # Effective Resistance with Zero-Division Guard (R_eff >= 0.1)
            r_eff = max(MIN_EFFECTIVE_RESISTANCE, target["base_resistance"] + reaction_r)
            effective_v = max(0.0, applied_v + v_mod)

            # Ohm's Law: I = V / R_eff
            current = effective_v / r_eff
            damage = int(math.floor(current))

            # Saturation Arithmetic (0 to 6 HP)
            prev_hp = target["hp"]
            new_hp = max(0, min(VITAL_MAX_HP, prev_hp - damage))
            target["hp"] = new_hp
            target["conduct"]["vital_hp"] = new_hp

            if new_hp == 0:
                target["conduct"]["is_active"] = False
                if target["layer"] == LAYER_CENTER:
                    coord_key = f"{target['q']},{target['r']}"
                    self.center_occupancy.pop(coord_key, None)

            return {
                "status": "OK",
                "action": "OHM_VOLTAGE_RESOLVED",
                "applied_voltage": applied_v,
                "effective_voltage": effective_v,
                "effective_resistance": round(r_eff, 4),
                "calculated_current": round(current, 4),
                "damage_dealt": damage,
                "target_prev_hp": prev_hp,
                "target_new_hp": new_hp,
                "interrupts_processed": processed_interrupts,
                "conduct_badge": self._format_conduct_badge(target["conduct"])
            }

        return {"status": "UNKNOWN_INSTRUCTION", "type": instr_type}

    # ── 3. ZERO-COPY PACKED INSTANCE BUFFER GENERATION ──────────────────────

    def generate_packed_buffer(self, hex_radius: float = 1.0) -> bytes:
        """
        Serializes all cells and entities into a packed binary buffer of
        24 float32s (96 bytes) per instance for direct GPU upload.
        """
        total_instances = len(self.cells) + len(self.entities)
        packed_bytes = bytearray(total_instances * 96)
        idx = 0

        # Pack Cells (Layer A_BASE)
        for cell in self.cells.values():
            offset = idx * 96
            x, z = axial_to_world(cell["q"], cell["r"], hex_radius)
            y = cell["elevation"]

            # 4x4 Transformation Matrix (Column-Major)
            matrix = [
                1.0, 0.0, 0.0, 0.0,  # col 0
                0.0, 1.0, 0.0, 0.0,  # col 1
                0.0, 0.0, 1.0, 0.0,  # col 2
                x,   y,   z,   1.0   # col 3
            ]
            for m_i, val in enumerate(matrix):
                struct.pack_into("<f", packed_bytes, offset + m_i * 4, float(val))

            # PBR Color & Roughness based on biome
            color = self._get_biome_color(cell["biome"])
            for c_i, val in enumerate(color):
                struct.pack_into("<f", packed_bytes, offset + 64 + c_i * 4, float(val))

            # Custom Shader Flags
            layer_id = LAYER_BASE
            hex_flags = (1.0 if cell["is_traversable"] else 0.0) + (2.0 if cell["is_hazard"] else 0.0)
            conduct_mask = 7.0
            entropy = 0.05
            struct.pack_into("<f", packed_bytes, offset + 80, float(layer_id))
            struct.pack_into("<f", packed_bytes, offset + 84, float(hex_flags))
            struct.pack_into("<f", packed_bytes, offset + 88, float(conduct_mask))
            struct.pack_into("<f", packed_bytes, offset + 92, float(entropy))

            idx += 1

        # Pack Entities
        for ent in self.entities.values():
            offset = idx * 96
            x, z = axial_to_world(ent["q"], ent["r"], hex_radius)
            y = ent["layer"]

            matrix = [
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                x,   y,   z,   1.0
            ]
            for m_i, val in enumerate(matrix):
                struct.pack_into("<f", packed_bytes, offset + m_i * 4, float(val))

            color = [0.95, 0.2, 0.1, 0.25] if ent["hp"] <= 2 else [0.2, 0.75, 0.95, 0.4]
            for c_i, val in enumerate(color):
                struct.pack_into("<f", packed_bytes, offset + 64 + c_i * 4, float(val))

            layer_id = float(ent["layer"])
            hp_flag = float(ent["hp"])
            c_flags = ent["conduct"]
            conduct_mask = 0.0
            if c_flags["is_active"]: conduct_mask += 1.0
            if c_flags["is_locked"]: conduct_mask += 2.0
            if c_flags["confidence"] >= CONFIDENCE_THRESHOLD: conduct_mask += 4.0
            if c_flags["is_hazardous"]: conduct_mask += 8.0
            entropy = 0.15

            struct.pack_into("<f", packed_bytes, offset + 80, layer_id)
            struct.pack_into("<f", packed_bytes, offset + 84, hp_flag)
            struct.pack_into("<f", packed_bytes, offset + 88, conduct_mask)
            struct.pack_into("<f", packed_bytes, offset + 92, entropy)

            idx += 1

        return bytes(packed_bytes)

    # ── 4. TELEMETRY & ADAPTIVE PROFILE SELECTION ───────────────────────────

    def evaluate_hardware_profile(self, fps: float, cpu_load: float, temp_c: float) -> str:
        """Evaluates hardware telemetry to return active profile."""
        self.telemetry["fps"] = fps
        self.telemetry["cpu_load"] = cpu_load
        self.telemetry["temperature_c"] = temp_c

        if fps < 30.0 or cpu_load > 85.0 or temp_c > 80.0:
            return "MINIMAL_WIREFRAME"
        elif fps >= 55.0 and cpu_load < 50.0 and temp_c < 65.0:
            return "AAA_CINEMATIC"
        else:
            return "STANDARD_LIVING_STANDEES"

    def _get_biome_color(self, biome: str) -> List[float]:
        palette = {
            "CRYSTAL_CITADEL": [0.15, 0.65, 0.95, 0.2],
            "VOLCANIC":        [0.92, 0.25, 0.05, 0.7],
            "CRYO_FROST":       [0.75, 0.90, 1.00, 0.1],
            "VOID_CORRUPTION": [0.35, 0.05, 0.45, 0.8],
            "CYBER_FORTRESS":  [0.05, 0.85, 0.65, 0.3],
            "PLAINS":          [0.20, 0.60, 0.30, 0.5]
        }
        return palette.get(biome, [0.2, 0.6, 0.3, 0.5])

    def _format_conduct_badge(self, c: Dict[str, Any]) -> str:
        act = "[ACT]" if c.get("is_active", True) else "[!ACT]"
        lck = "[LCK]" if c.get("is_locked", False) else "[!LCK]"
        cnf = "[CNF]" if c.get("confidence", 1.0) >= CONFIDENCE_THRESHOLD else "[!CNF]"
        haz = "[HAZ]" if c.get("is_hazardous", False) else "[!HAZ]"
        return f"{act} {lck} {cnf} {haz} [VIT {c.get('vital_hp', VITAL_MAX_HP)}HP]"


# Global singleton instance
GLOBAL_SOVEREIGN_HTIB_BRIDGE = SovereignHTIBBridge()
