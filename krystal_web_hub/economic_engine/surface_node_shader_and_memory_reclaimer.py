"""
KRYSTAL-STACK // SURFACE NODE SHADER COMPOSER & REVERSE MEMORY SLOT RECLAIMER
=============================================================================
Advanced node-based surface material simulator integrating:
  1. Multi-layer procedural surface adjustments (normals, triplanar, Fresnel, AO, self-shadows).
  2. Reverse memory slot scavenging (Memory Block Slab Hole Reclamation).
  3. Dynamic scenario expansion: Vacant memory slots are recycled in reverse order,
     injecting permutation seeds into shader parameters to create exponential
     combinatorial surface scenarios without increasing memory footprint.
  4. Platform Invariant: VITAL_MAX_HP = 6
"""

import math
import hashlib
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

VITAL_MAX_HP: int = 6
GOLDEN_RATIO: float = 1.61803398875
INV_GOLDEN_RATIO: float = 0.61803398875
DEFAULT_SLAB_SLOTS: int = 64
SLOT_SIZE_BYTES: int = 4096  # 4 KB per slot


@dataclass
class MemorySlot:
    """Represents an atomic allocation slot within a hardware/DDR memory slab."""
    slot_id: int
    offset_bytes: int
    size_bytes: int = SLOT_SIZE_BYTES
    is_occupied: bool = False
    owner_id: str = ""
    reclaimed: bool = False
    entropy_signature: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "offset_bytes": self.offset_bytes,
            "size_bytes": self.size_bytes,
            "is_occupied": self.is_occupied,
            "owner_id": self.owner_id,
            "reclaimed": self.reclaimed,
            "entropy_signature": round(self.entropy_signature, 4)
        }


class MemoryBlockSlab:
    """
    Manages a contiguous block slab of memory slots with reverse scavenger logic.
    Reclaims unoccupied/fragmented gaps in reverse sequence (LIFO) to maximize
    cache spatial locality and generate procedural variation seeds.
    """

    def __init__(self, num_slots: int = DEFAULT_SLAB_SLOTS):
        self.num_slots = num_slots
        self.slots: List[MemorySlot] = [
            MemorySlot(
                slot_id=i,
                offset_bytes=i * SLOT_SIZE_BYTES,
                is_occupied=False,
                owner_id="",
                reclaimed=False,
                entropy_signature=round(math.sin((i + 1) * INV_GOLDEN_RATIO * math.pi) * 0.5 + 0.5, 4)
            )
            for i in range(num_slots)
        ]

    def allocate_slot(self, slot_id: int, owner_id: str) -> bool:
        if 0 <= slot_id < self.num_slots:
            slot = self.slots[slot_id]
            slot.is_occupied = True
            slot.owner_id = owner_id
            slot.reclaimed = False
            return True
        return False

    def free_slot(self, slot_id: int) -> bool:
        if 0 <= slot_id < self.num_slots:
            slot = self.slots[slot_id]
            slot.is_occupied = False
            slot.owner_id = ""
            slot.reclaimed = False
            return True
        return False

    def reverse_scavenge_empty_slots(self) -> List[Dict[str, Any]]:
        """
        Reverse scan (from top slot num_slots - 1 down to 0) looking for empty holes.
        Flags them as reclaimed and computes entropy seeds that drive shader permutation scenarios.
        """
        scavenged: List[Dict[str, Any]] = []
        for i in range(self.num_slots - 1, -1, -1):
            slot = self.slots[i]
            if not slot.is_occupied:
                slot.reclaimed = True
                # Generate unique permutation seed based on reverse position and golden ratio
                seed_val = abs(math.sin((i * 13.37 + 1.618) * math.pi))
                feature_mod = i % 8  # 8 distinct surface functionalities mapped
                scavenged.append({
                    "slot_id": slot.slot_id,
                    "offset_bytes": slot.offset_bytes,
                    "permutation_seed": round(seed_val, 5),
                    "feature_channel": feature_mod,
                    "reclaim_order": len(scavenged) + 1
                })
        return scavenged

    def get_slab_telemetry(self) -> Dict[str, Any]:
        occupied_count = sum(1 for s in self.slots if s.is_occupied)
        reclaimed_count = sum(1 for s in self.slots if s.reclaimed and not s.is_occupied)
        empty_count = self.num_slots - occupied_count

        return {
            "total_slots": self.num_slots,
            "total_capacity_kb": (self.num_slots * SLOT_SIZE_BYTES) // 1024,
            "occupied_slots": occupied_count,
            "empty_slots": empty_count,
            "reclaimed_slots": reclaimed_count,
            "fragmentation_ratio": round(empty_count / max(1, self.num_slots), 3),
            "slots": [s.to_dict() for s in self.slots]
        }


# ── Standalone Surface Shader Functionality Mappings ─────────────────────────
SURFACE_FEATURE_CHANNELS = {
    0: {
        "name": "Micro-Faceted Specular Refraction",
        "description": "Kryštálové fazety s rezonančným indexom lomu (IOR 1.54).",
        "uniform": "u_specular_refraction",
        "category": "optics"
    },
    1: {
        "name": "Parallax Self-Shadowing (Occlusion)",
        "description": "Vrhá vnútorné kontaktné tiene podľa výškovej mapy reliéfu.",
        "uniform": "u_self_shadow_density",
        "category": "shadows"
    },
    2: {
        "name": "Subsurface Chromatic Dispersion",
        "description": "Podpovrchový rozptyl svetla simulujúci minerálnu translucenciu.",
        "uniform": "u_subsurface_scatter",
        "category": "subsurface"
    },
    3: {
        "name": "Toxic Slime Caustic Bubbling",
        "description": "Organická kaustická viskozita s pulzujúcim reliéfnym šumom.",
        "uniform": "u_caustic_flux",
        "category": "organic"
    },
    4: {
        "name": "Druidic Weathering & Ambient Crevice",
        "description": "Akumulácia patiny a mikro-machu v hlbokých dutinách povrchu.",
        "uniform": "u_crevice_occlusion",
        "category": "weathering"
    },
    5: {
        "name": "Harmonic Golden Mean Fresnel Rim",
        "description": "Svetelný okrajový obal riadený Aristotelovským pomerom phi^-1.",
        "uniform": "u_fresnel_rim",
        "category": "lighting"
    },
    6: {
        "name": "Anisotropic Metallic Weathering",
        "description": "Smerové mikro-škrabance a brúsené kovové odlesky.",
        "uniform": "u_anisotropy_flow",
        "category": "metallic"
    },
    7: {
        "name": "Volumetric Absorption Gradient",
        "description": "Absorpcia svetla v hrúbke materiálu podľa Beer-Lambertovho zákona.",
        "uniform": "u_volumetric_absorb",
        "category": "volumetric"
    }
}


# ── Node Graph Visual Schema ────────────────────────────────────────────────
@dataclass
class NodePin:
    pin_id: str
    name: str
    pin_type: str  # 'float', 'vector3', 'color', 'texture', 'memory_seed'
    direction: str  # 'in' or 'out'
    default_val: Any = None


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    title: str
    pos_x: float
    pos_y: float
    inputs: List[NodePin] = field(default_factory=list)
    outputs: List[NodePin] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "title": self.title,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
            "parameters": self.parameters,
            "inputs": [
                {"pin_id": p.pin_id, "name": p.name, "type": p.pin_type, "default": p.default_val}
                for p in self.inputs
            ],
            "outputs": [
                {"pin_id": p.pin_id, "name": p.name, "type": p.pin_type}
                for p in self.outputs
            ]
        }


class SurfaceNodeShaderEngine:
    """
    Evaluates multi-layer visual node graphs, calculates shader adjustments,
    and maps reverse-scavenged memory slots into combinatorial scenario expansions.
    """

    def __init__(self):
        self.slab = MemoryBlockSlab(num_slots=DEFAULT_SLAB_SLOTS)
        self._init_default_slab_state()

    def _init_default_slab_state(self):
        # Seed realistic initial fragmentation: 36 slots occupied, 28 vacant holes
        for i in range(DEFAULT_SLAB_SLOTS):
            if (i % 3 != 0 and i % 5 != 0) or (i < 8):
                self.slab.allocate_slot(i, owner_id=f"alloc_proc_{i % 4}")

    def get_node_catalog(self) -> Dict[str, Any]:
        """Returns catalog of all available node primitives and surface features."""
        return {
            "vital_max_hp_rule": VITAL_MAX_HP,
            "golden_ratio_phi": GOLDEN_RATIO,
            "feature_channels": SURFACE_FEATURE_CHANNELS,
            "node_types": [
                {
                    "type": "SURFACE_BASE_LAYER",
                    "category": "layer",
                    "title": "Základná Vrstva Povrchu (Albedo & Base Normal)",
                    "description": "Generuje základnú farbu, drsnosť (roughness) a primárnu normálovú mapu."
                },
                {
                    "type": "NOISE_PERTURBATION",
                    "category": "adjustment",
                    "title": "Nelineárna Šumová Perturbácia",
                    "description": "Simuluje mikro-detaily a drsnosť cez fraktálny Perlin / Voronoi šum."
                },
                {
                    "type": "SELF_SHADOW_OCCLUSION",
                    "category": "shadows",
                    "title": "Samotienenie & Kontaktná Oklúzia",
                    "description": "Počíta horizon-based kontaktné tiene v závislosti od výškového reliéfu."
                },
                {
                    "type": "FRESNEL_HARMONIC_RIM",
                    "category": "adjustment",
                    "title": "Fresnel Svetelný Lem (Phi^-1)",
                    "description": "Nastavuje okrajový svetelný gradient pod zlatým uhlom."
                },
                {
                    "type": "MEMORY_SLOT_SCAVENGER",
                    "category": "memory",
                    "title": "Spätný Zberač Prázdnych Slotov Pamäte",
                    "description": "Vychytáva prázdne bloky pamäte odzadu a generuje z nich permutačné jadrá."
                },
                {
                    "type": "SCENARIO_EXPANDER",
                    "category": "combinatorics",
                    "title": "Expanzia Scenárov a Variácií",
                    "description": "Kombinatoricky násobí vizuálne varianty povrchu bez alokácie ďalšej RAM."
                },
                {
                    "type": "MASTER_SURFACE_OUTPUT",
                    "category": "output",
                    "title": "Finálny PBR Shader & Materiálový Výstup",
                    "description": "Výstupný bod posielaný do WebGL / Vulkan shader potrubia."
                }
            ]
        }

    def scavenge_and_generate_scenarios(self, alloc_pattern: Optional[List[int]] = None, free_pattern: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Simulates dynamic memory allocation/deallocation churn, runs reverse scavenging
        of vacant slots, and computes scenario combinatorial growth.
        """
        # Apply optional custom alloc / free instructions
        if alloc_pattern:
            for s_id in alloc_pattern:
                self.slab.allocate_slot(s_id, "dynamic_worker")
        if free_pattern:
            for s_id in free_pattern:
                self.slab.free_slot(s_id)

        # Execute reverse hole-punch scavenging
        reclaimed_slots = self.slab.reverse_scavenge_empty_slots()
        k_reclaimed = len(reclaimed_slots)

        # Combinatorial scenario growth formula:
        # P(8, min(8, k)) * 2^(k / 4)
        n_features = 8
        k_sample = min(8, k_reclaimed)
        permutations = math.perm(n_features, k_sample) if k_sample <= n_features else math.factorial(8)
        scenario_multiplier = int(permutations * (2 ** (min(16, k_reclaimed) / 4.0)))

        # Formulate active shader uniforms derived from reclaimed slots
        active_shader_uniforms: Dict[str, float] = {}
        for r in reclaimed_slots:
            ch_idx = r["feature_channel"]
            feat = SURFACE_FEATURE_CHANNELS[ch_idx]
            u_name = feat["uniform"]
            current_u = active_shader_uniforms.get(u_name, 0.0)
            # Accumulate with golden-mean dampening
            active_shader_uniforms[u_name] = round(min(1.0, current_u + r["permutation_seed"] * 0.25), 4)

        return {
            "vital_max_hp_rule": VITAL_MAX_HP,
            "slab_telemetry": self.slab.get_slab_telemetry(),
            "reclaimed_count": k_reclaimed,
            "reclaimed_slots": reclaimed_slots[:16],  # Preview top 16 reclaimed slots
            "scenario_multiplication_factor": scenario_multiplier,
            "scenario_growth_level": "EXPONENTIAL" if scenario_multiplier > 100000 else "POLYNOMIAL",
            "active_shader_uniforms": active_shader_uniforms,
            "memory_saved_kb": (k_reclaimed * SLOT_SIZE_BYTES) // 1024,
            "notes": "Spätné vychytávanie prázdnych slotov pamäte vytvorilo bez dodatočnej alokácie RAM nárast unikátnych scenárov."
        }

    def evaluate_node_graph(self, graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a node graph with multi-layer surface adjustments, shadow parameters,
        and memory slot scavenging to produce the compiled PBR shader uniforms.
        """
        nodes = graph_data.get("nodes", [])
        params = graph_data.get("parameters", {})

        # Extract graph parameters or fallbacks
        base_color = params.get("base_color", [0.15, 0.75, 0.95])
        roughness = float(params.get("roughness", 0.35))
        metallic = float(params.get("metallic", 0.65))
        normal_strength = float(params.get("normal_strength", 1.25))
        fresnel_power = float(params.get("fresnel_power", 2.618))
        self_shadow_depth = float(params.get("self_shadow_depth", 0.8))
        layer_blend_factor = float(params.get("layer_blend_factor", 0.5))

        # Re-evaluate memory slot scavenger
        scavenge_result = self.scavenge_and_generate_scenarios()
        uniforms = scavenge_result["active_shader_uniforms"]

        # Combined multi-layer surface descriptor
        compiled_material = {
            "diffuse_albedo": base_color,
            "roughness": round(roughness * (1.0 + uniforms.get("u_anisotropy_flow", 0.0) * 0.2), 3),
            "metallic": metallic,
            "normal_scale": normal_strength,
            "fresnel_intensity": round(fresnel_power * (uniforms.get("u_fresnel_rim", 0.5)), 3),
            "contact_shadow_bias": round(self_shadow_depth * (uniforms.get("u_self_shadow_density", 0.7)), 3),
            "subsurface_tint": [
                round(base_color[0] * 1.2, 3),
                round(base_color[1] * 0.9, 3),
                round(base_color[2] * 0.7, 3)
            ],
            "scenario_count": scavenge_result["scenario_multiplication_factor"],
            "reclaimed_memory_slots": scavenge_result["reclaimed_count"]
        }

        return {
            "vital_max_hp_rule": VITAL_MAX_HP,
            "nodes_count": len(nodes),
            "compiled_material": compiled_material,
            "shader_uniforms": uniforms,
            "scenario_growth": scavenge_result["scenario_multiplication_factor"],
            "status": "COMPILED_OPTIMIZED"
        }

    def get_default_graph(self) -> Dict[str, Any]:
        """Provides a canonical rich multi-layer surface graph ready for the Node Editor."""
        nodes = [
            GraphNode(
                node_id="node_base_layer",
                node_type="SURFACE_BASE_LAYER",
                title="Základný Kryštálový Povrch",
                pos_x=60,
                pos_y=80,
                parameters={"base_color": "#00f0ff", "roughness": 0.25, "metallic": 0.85},
                outputs=[
                    NodePin("out_color", "Albedo", "color", "out"),
                    NodePin("out_normal", "Normal", "vector3", "out"),
                    NodePin("out_roughness", "Roughness", "float", "out")
                ]
            ),
            GraphNode(
                node_id="node_noise_perturb",
                node_type="NOISE_PERTURBATION",
                title="Voronoi Perturbácia",
                pos_x=60,
                pos_y=320,
                parameters={"frequency": 14.5, "amplitude": 0.45},
                outputs=[
                    NodePin("out_bump", "Bump Map", "float", "out"),
                    NodePin("out_height", "Height", "float", "out")
                ]
            ),
            GraphNode(
                node_id="node_self_shadow",
                node_type="SELF_SHADOW_OCCLUSION",
                title="Kontaktné Samotienenie",
                pos_x=360,
                pos_y=280,
                parameters={"shadow_bias": 0.75, "horizon_samples": 8},
                inputs=[
                    NodePin("in_height", "Height", "float", "in")
                ],
                outputs=[
                    NodePin("out_shadow_occlusion", "Shadow Mask", "float", "out")
                ]
            ),
            GraphNode(
                node_id="node_fresnel",
                node_type="FRESNEL_HARMONIC_RIM",
                title="Harmonický Fresnel Lem",
                pos_x=360,
                pos_y=100,
                parameters={"phi_rim_exponent": 1.618, "rim_color": "#ffd700"},
                outputs=[
                    NodePin("out_rim_glow", "Rim Glow", "color", "out")
                ]
            ),
            GraphNode(
                node_id="node_mem_scavenger",
                node_type="MEMORY_SLOT_SCAVENGER",
                title="Spätný Zberač Slotov Pamäte",
                pos_x=660,
                pos_y=320,
                parameters={"slab_slots": 64, "reverse_scan": True},
                outputs=[
                    NodePin("out_reclaimed_entropy", "Slot Entropy", "memory_seed", "out"),
                    NodePin("out_scenario_seed", "Scenario Seed", "float", "out")
                ]
            ),
            GraphNode(
                node_id="node_scenario_expander",
                node_type="SCENARIO_EXPANDER",
                title="Kombinatorický Expanzér Scenárov",
                pos_x=960,
                pos_y=320,
                parameters={"combinatorial_rule": "permutations_p8k"},
                inputs=[
                    NodePin("in_seed", "Scenario Seed", "float", "in")
                ],
                outputs=[
                    NodePin("out_scenario_multiplier", "Multiplikátor Scenárov", "float", "out")
                ]
            ),
            GraphNode(
                node_id="node_master_output",
                node_type="MASTER_SURFACE_OUTPUT",
                title="Master PBR Shader Výstup",
                pos_x=1280,
                pos_y=160,
                inputs=[
                    NodePin("in_diffuse", "Diffuse", "color", "in"),
                    NodePin("in_roughness", "Roughness", "float", "in"),
                    NodePin("in_normal", "Normal", "vector3", "in"),
                    NodePin("in_shadow", "Shadow AO", "float", "in"),
                    NodePin("in_rim", "Rim", "color", "in"),
                    NodePin("in_scenario_mult", "Scenario Multiplier", "float", "in")
                ]
            )
        ]

        connections = [
            {"from_node": "node_noise_perturb", "from_pin": "out_height", "to_node": "node_self_shadow", "to_pin": "in_height"},
            {"from_node": "node_base_layer", "from_pin": "out_color", "to_node": "node_master_output", "to_pin": "in_diffuse"},
            {"from_node": "node_base_layer", "from_pin": "out_roughness", "to_node": "node_master_output", "to_pin": "in_roughness"},
            {"from_node": "node_base_layer", "from_pin": "out_normal", "to_node": "node_master_output", "to_pin": "in_normal"},
            {"from_node": "node_self_shadow", "from_pin": "out_shadow_occlusion", "to_node": "node_master_output", "to_pin": "in_shadow"},
            {"from_node": "node_fresnel", "from_pin": "out_rim_glow", "to_node": "node_master_output", "to_pin": "in_rim"},
            {"from_node": "node_mem_scavenger", "from_pin": "out_scenario_seed", "to_node": "node_scenario_expander", "to_pin": "in_seed"},
            {"from_node": "node_scenario_expander", "from_pin": "out_scenario_multiplier", "to_node": "node_master_output", "to_pin": "in_scenario_mult"}
        ]

        return {
            "graph_id": "surface_alchemical_multi_layer_canonical",
            "title": "Alchymistický Viacvrstvový Povrch so Spätným Vychytávaním Pamäte",
            "vital_max_hp_rule": VITAL_MAX_HP,
            "nodes": [n.to_dict() for n in nodes],
            "connections": connections
        }


# Global singleton instance
GLOBAL_SURFACE_NODE_ENGINE = SurfaceNodeShaderEngine()
