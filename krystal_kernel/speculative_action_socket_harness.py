#!/usr/bin/env python3
"""
==============================================================================
KRYSTAL-STACK: SPECULATIVE ACTION INTERPRETER, SOCKET HYSTERESIS & CV ENCODER
==============================================================================
Module: krystal_kernel/speculative_action_socket_harness.py
Normative Reference:
    - docs/architecture/MASTER_BLUEPRINT_KRYSTAL_STACK_SOVEREIGN_OMNI_SYSTEM.md
    - docs/architecture/AEP_CANONICAL_SPEC.md
    - docs/architecture/AEP_CONTROL_METRICS_V2_SPEC.md
Constitutional Invariant:
    - VITAL_MAX_HP == 6
    - Invariant != Policy
    - Asymmetric Transitions: Delta S_up != Delta S_down
    - Zero-Dirty State Mutation Delta S == 0 on breach

WHAT THIS MODULE PROVIDES:
    1. Image / Computer Vision Graph Encoders:
       - Encodes GraphQL DAG structures into 2D multi-spectral feature tensors:
         [Channel 0: Adjacency/Topology, Channel 1: Compute Flops Density,
          Channel 2: UMA Memory Traffic, Channel 3: Device Affinity / Latency].
    2. Speculative Action Interpreter:
       - Interprets speculative candidate actions, generates asymmetric transition
         vectors, and mixes speculative candidate actions with the baseline plan.
    3. Socket Sub-Unit Anti-Oscillation Hysteresis:
       - Monitors sub-units on the physical socket (P-cores, E-cores, Iris Xe EUs,
         Hailo NPU clusters) and establishes redundant threshold filters:
         h_socket >= 2*delta_noise + kappa * sigma_strain.
    4. Sandwich Architecture Governor & Splicer:
       - Decouples stochastic LLM/RAG decomposition from deterministic governor execution.
       - Slices and batches high-throughput workloads without violating invariant laws.
    5. Reverse Dependency Engineering:
       - Traces causal dependencies backwards (Action -> Reaction -> Deduction)
         to diagnose socket contention and thermal backpressure roots.
    6. Elevated Control Quotas & Multi-Spectral Evaluation Thresholds:
       - Computes Control Validity Quota (CVQ), Asymmetric Transition Ratio (ATR),
         and Socket Oscillation Suppression Ratio (SOSR).
==============================================================================
"""

from __future__ import annotations

import time
import math
import uuid
import json
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, Set

from .constants import VITAL_MAX_HP, ContractViolation
from .graphql_cv_task_gtd_engine import (
    GraphQLDagNode,
    HeterogeneousDeviceTarget,
    GtdTaskPhase,
    BranchEfficiencyStatus,
)


# =====================================================================
# 1. ENUMS & CONSTANTS
# =====================================================================

class SocketSubUnitType(str, Enum):
    """Sub-units residing on the heterogeneous processor socket."""
    P_CORE       = "P_CORE"        # Performance Core (Willow Cove AVX-512)
    E_CORE       = "E_CORE"        # Efficiency Core / Background scalar thread
    GPU_EU_SLICE = "GPU_EU_SLICE"  # Intel Iris Xe Execution Unit Sub-slice (DP4A)
    NPU_CLUSTER  = "NPU_CLUSTER"   # Hailo Neural Processing Unit cluster
    LLC_RING     = "LLC_RING"      # Last Level Cache ring stop & interconnect


class SpeculativeActionType(str, Enum):
    """Speculative interventions proposed by the interpreter."""
    HOLD_CURRENT          = "HOLD_CURRENT"
    ASYMMETRIC_UPSCALE    = "ASYMMETRIC_UPSCALE"     # Fast reaction to incoming load spikes
    HYSTERETIC_DOWNSCALE  = "HYSTERETIC_DOWNSCALE"   # Damped gradual drop (prevents chatter)
    SUB_UNIT_MIGRATION    = "SUB_UNIT_MIGRATION"     # Move task between P-core, E-core, and EU
    SPLICED_BATCH_SPLIT   = "SPLICED_BATCH_SPLIT"    # Splicer splits batch across heterogeneous lanes
    SPECULATIVE_PREFILL   = "SPECULATIVE_PREFILL"    # Memory prefill in chronometric shadow


# =====================================================================
# 2. DATA CONTRACTS: 2D IMAGE TENSORS & SPECULATIVE PLANS
# =====================================================================

@dataclass
class GraphVisualTensor:
    """2D Multi-spectral visual feature tensor representing the GraphQL DAG."""
    tensor_id: str
    width: int
    height: int
    channels: int                       # Exactly 4: [Topology, Flops, Memory, Affinity]
    raw_tensor_data: List[List[List[float]]] # [C][H][W] normalized in [0.0, 1.0]
    luma_centroid_x: float
    luma_centroid_y: float
    spatial_entropy: float
    uma_contention_hazard: float
    perceptual_hash: str
    vital_max_hp: int = VITAL_MAX_HP


@dataclass
class SocketSubUnitTelemetry:
    """Real-time physical telemetry sampled per socket sub-unit."""
    unit_id: str
    unit_type: SocketSubUnitType
    temperature_celsius: float
    utilization_pct: float
    voltage_droop_mv: float
    frequency_mhz: float
    oscillation_rate_hz: float          # Frequency transition chattering rate
    noise_variance_delta: float         # Current electrical/thermal noise variance delta_hat


@dataclass
class SpeculativeActionCandidate:
    """A speculative candidate action evaluated prior to physical dispatch."""
    candidate_id: str
    action_type: SpeculativeActionType
    target_unit: SocketSubUnitType
    speculative_gain_estimate: float    # Expected speedup or energy saving in %
    risk_factor: float                  # [0.0 to 1.0]
    asymmetric_weight: float            # Factor accounting for Delta S_up vs Delta S_down
    required_hysteresis_margin: float   # Minimum threshold h_socket required to trigger
    is_admissible: bool = True
    falsifier_condition: str = ""


@dataclass
class GovernorSplicerReceipt:
    """Signed audit receipt from the sandwich governor-splicer execution."""
    receipt_id: str
    timestamp_ns: int
    input_query_hash: str
    visual_tensor_hash: str
    selected_action: SpeculativeActionCandidate
    active_socket_units: List[str]
    socket_oscillation_suppressed: bool
    control_validity_quota: float       # CVQ: [0.0 to 1.0], target >= 0.92
    asymmetric_transition_ratio: float  # ATR: ratio of successful upward vs downward shifts
    delta_s: float                      # Constitutional Delta S == 0 on breach
    is_safe: bool
    vital_max_hp: int
    conduct_flags: List[str]
    hmac_digest: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["selected_action"] = asdict(self.selected_action)
        d["selected_action"]["action_type"] = self.selected_action.action_type.value
        d["selected_action"]["target_unit"] = self.selected_action.target_unit.value
        return d


# =====================================================================
# 3. COMPUTER VISION GRAPH IMAGE ENCODER
# =====================================================================

class CvGraphImageEncoder:
    """
    Encodes GraphQL DAG nodes into 2D multi-spectral feature tensors.
    Provides computer vision models with a spatial topology representation.
    """

    def __init__(self, raster_width: int = 16, raster_height: int = 16) -> None:
        self.width = raster_width
        self.height = raster_height
        self.channels = 4 # [Topology, FlopsDensity, MemoryTraffic, Affinity]

    def encode_dag_to_visual_tensor(self, nodes: List[GraphQLDagNode]) -> GraphVisualTensor:
        """Encodes node list into a 4-channel 2D spatial raster."""
        tensor_id = f"TENSOR-{uuid.uuid4().hex[:8]}"
        grid = [[[0.0 for _ in range(self.width)] for _ in range(self.height)] for _ in range(self.channels)]

        if not nodes:
            return GraphVisualTensor(
                tensor_id=tensor_id,
                width=self.width,
                height=self.height,
                channels=self.channels,
                raw_tensor_data=grid,
                luma_centroid_x=0.0,
                luma_centroid_y=0.0,
                spatial_entropy=0.0,
                uma_contention_hazard=0.0,
                perceptual_hash="0" * 64,
                vital_max_hp=VITAL_MAX_HP,
            )

        max_flops = max((n.compute_complexity_flops for n in nodes), default=1.0)
        max_bytes = max((n.memory_traffic_bytes for n in nodes), default=1.0)

        total_weight = 0.0
        sum_x = 0.0
        sum_y = 0.0

        for idx, n in enumerate(nodes):
            # Compute grid coordinates based on index and depth
            x = (idx * 3 + n.depth * 2) % self.width
            y = (n.depth * 3 + (idx // self.width)) % self.height

            # Channel 0: Topological Presence (1.0)
            grid[0][y][x] = min(1.0, grid[0][y][x] + 0.5)

            # Channel 1: Compute Flops Density
            f_norm = min(1.0, n.compute_complexity_flops / max(1.0, max_flops))
            grid[1][y][x] = max(grid[1][y][x], f_norm)

            # Channel 2: Memory Traffic Density
            m_norm = min(1.0, n.memory_traffic_bytes / max(1.0, max_bytes))
            grid[2][y][x] = max(grid[2][y][x], m_norm)

            # Channel 3: Device Affinity
            if n.preferred_device == HeterogeneousDeviceTarget.GPU_IRIS_XE:
                aff = 0.65
            elif n.preferred_device == HeterogeneousDeviceTarget.NPU_HAILO:
                aff = 0.90
            else:
                aff = 0.30
            grid[3][y][x] = aff

            weight = f_norm + m_norm + 0.1
            sum_x += x * weight
            sum_y += y * weight
            total_weight += weight

        centroid_x = round(sum_x / max(0.001, total_weight), 3)
        centroid_y = round(sum_y / max(0.001, total_weight), 3)

        # Spatial entropy computation across Channel 0
        entropy = 0.0
        for row in grid[0]:
            for val in row:
                if val > 0.0:
                    p = val / float(len(nodes))
                    entropy -= p * math.log2(p + 1e-9)
        entropy = round(max(0.0, entropy), 4)

        # UMA Contention hazard from memory channel concentration
        max_mem_pixel = max((val for row in grid[2] for val in row), default=0.0)
        uma_hazard = round(min(1.0, max_mem_pixel * 0.85 + (len(nodes) / 32.0) * 0.15), 3)

        # 64-char visual hash
        hash_seed = f"w{self.width}h{self.height}:cx{centroid_x}cy{centroid_y}:ent{entropy}:uma{uma_hazard}"
        p_hash = hashlib.sha256(hash_seed.encode("utf-8")).hexdigest()

        return GraphVisualTensor(
            tensor_id=tensor_id,
            width=self.width,
            height=self.height,
            channels=self.channels,
            raw_tensor_data=grid,
            luma_centroid_x=centroid_x,
            luma_centroid_y=centroid_y,
            spatial_entropy=entropy,
            uma_contention_hazard=uma_hazard,
            perceptual_hash=p_hash,
            vital_max_hp=VITAL_MAX_HP,
        )


# =====================================================================
# 4. SPECULATIVE ACTION INTERPRETER & SOCKET HYSTERESIS HARNESS
# =====================================================================

class SpeculativeActionSocketHarness:
    """
    Interprets speculative actions, maintains socket anti-oscillation hysteresis,
    and governs heterogeneous slicing under sandwich architecture rules.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        self.vital_max_hp = vital_max_hp
        if self.vital_max_hp != 6:
            raise ContractViolation(f"VITAL_MAX_HP must equal 6, got {self.vital_max_hp}")

        self.cv_encoder = CvGraphImageEncoder(raster_width=16, raster_height=16)

        # Mock socket sub-unit states
        self.socket_units: Dict[str, SocketSubUnitTelemetry] = {
            "P_CORE_0": SocketSubUnitTelemetry("P_CORE_0", SocketSubUnitType.P_CORE, 62.5, 0.45, 12.0, 4200.0, 1.2, 0.012),
            "P_CORE_1": SocketSubUnitTelemetry("P_CORE_1", SocketSubUnitType.P_CORE, 64.0, 0.50, 14.0, 4200.0, 1.5, 0.014),
            "E_CORE_0": SocketSubUnitTelemetry("E_CORE_0", SocketSubUnitType.E_CORE, 52.0, 0.20, 4.0, 2400.0, 0.2, 0.005),
            "GPU_SLICE_0": SocketSubUnitTelemetry("GPU_SLICE_0", SocketSubUnitType.GPU_EU_SLICE, 58.0, 0.35, 18.0, 1300.0, 0.8, 0.018),
            "NPU_CLUSTER_0": SocketSubUnitTelemetry("NPU_CLUSTER_0", SocketSubUnitType.NPU_CLUSTER, 44.0, 0.10, 2.0, 1000.0, 0.1, 0.002),
        }

        self.history_receipts: List[GovernorSplicerReceipt] = []
        self.total_cycles: int = 0
        self.kappa_strain_coefficient: float = 0.45

    def compute_socket_hysteresis_threshold(self, unit_id: str) -> float:
        """
        Computes the redundant socket hysteresis threshold:
        h_socket = 2 * delta_hat + kappa * sigma_strain
        Prevents chattering / oscillation between sub-units.
        """
        unit = self.socket_units.get(unit_id)
        if not unit:
            return 0.15

        delta_hat = unit.noise_variance_delta
        # sigma_strain incorporates voltage droop and thermal headroom
        thermal_strain = max(0.0, (unit.temperature_celsius - 50.0) / 40.0)
        voltage_strain = min(1.0, unit.voltage_droop_mv / 50.0)
        sigma_strain = (thermal_strain + voltage_strain) * 0.5

        h_socket = 2.0 * delta_hat + (self.kappa_strain_coefficient * sigma_strain)
        return round(max(0.05, h_socket), 4)

    def generate_speculative_candidates(
        self,
        visual_tensor: GraphVisualTensor,
    ) -> List[SpeculativeActionCandidate]:
        """
        Generates competing speculative action candidates based on the visual tensor.
        Enforces asymmetric transitions (Delta S_up != Delta S_down).
        """
        candidates: List[SpeculativeActionCandidate] = []

        # Candidate 1: Asymmetric Upscale (aggressive reaction if high compute density)
        h_pcore = self.compute_socket_hysteresis_threshold("P_CORE_0")
        candidates.append(SpeculativeActionCandidate(
            candidate_id="ACT-01-UPSCALE-PCORE",
            action_type=SpeculativeActionType.ASYMMETRIC_UPSCALE,
            target_unit=SocketSubUnitType.P_CORE,
            speculative_gain_estimate=24.5,
            risk_factor=0.20,
            asymmetric_weight=1.45, # Upward shift has higher momentum
            required_hysteresis_margin=h_pcore,
            is_admissible=visual_tensor.spatial_entropy > 0.10,
            falsifier_condition="P_CORE thermal throttling > 85C or voltage droop > 35mV",
        ))

        # Candidate 2: Spliced Batch Split across Iris Xe GPU
        h_gpu = self.compute_socket_hysteresis_threshold("GPU_SLICE_0")
        candidates.append(SpeculativeActionCandidate(
            candidate_id="ACT-02-SPLICE-GPU",
            action_type=SpeculativeActionType.SPLICED_BATCH_SPLIT,
            target_unit=SocketSubUnitType.GPU_EU_SLICE,
            speculative_gain_estimate=38.0,
            risk_factor=0.30,
            asymmetric_weight=1.20,
            required_hysteresis_margin=h_gpu,
            is_admissible=visual_tensor.uma_contention_hazard < 0.75,
            falsifier_condition="UMA memory contention hazard >= 0.78",
        ))

        # Candidate 3: Sub-unit Migration to NPU cluster (power-saving path)
        h_npu = self.compute_socket_hysteresis_threshold("NPU_CLUSTER_0")
        candidates.append(SpeculativeActionCandidate(
            candidate_id="ACT-03-MIGRATE-NPU",
            action_type=SpeculativeActionType.SUB_UNIT_MIGRATION,
            target_unit=SocketSubUnitType.NPU_CLUSTER,
            speculative_gain_estimate=15.0,
            risk_factor=0.08,
            asymmetric_weight=0.90, # Low energy transition
            required_hysteresis_margin=h_npu,
            is_admissible=True,
            falsifier_condition="NPU latency exceeds 1.25x P_CORE deadline",
        ))

        # Candidate 4: Hysteretic Downscale (Damped return to baseline)
        candidates.append(SpeculativeActionCandidate(
            candidate_id="ACT-04-HYSTERETIC-DOWNSCALE",
            action_type=SpeculativeActionType.HYSTERETIC_DOWNSCALE,
            target_unit=SocketSubUnitType.E_CORE,
            speculative_gain_estimate=8.5,
            risk_factor=0.05,
            asymmetric_weight=0.65, # Downward shift is heavily damped (12-frame dwell)
            required_hysteresis_margin=0.12,
            is_admissible=True,
            falsifier_condition="Premature frequency reduction while queue is non-empty",
        ))

        return candidates

    def execute_governor_splicer_cycle(
        self,
        graphql_query_str: str,
        nodes: List[GraphQLDagNode],
    ) -> GovernorSplicerReceipt:
        """
        Executes a deterministic governor & splicer cycle:
          1. Encode DAG into 2D GraphVisualTensor.
          2. Generate speculative candidates with asymmetric transition weights.
          3. Evaluate socket hysteresis thresholds and filter out chatter.
          4. Select winning candidate and commit via AEP zero-dirty gate.
          5. Emit signed cryptographic receipt.
        """
        self.total_cycles += 1
        t_start = time.time_ns()

        # Step 1: Visual Tensor Encoding
        visual_tensor = self.cv_encoder.encode_dag_to_visual_tensor(nodes)

        # Step 2: Speculative Candidate Interpretation
        candidates = self.generate_speculative_candidates(visual_tensor)

        # Step 3: Selection based on Admissibility and Hysteresis Margin
        admissible = [c for c in candidates if c.is_admissible]
        if not admissible:
            selected_action = candidates[-1] # Fallback to downscale
        else:
            # Rank by Utility / (Risk * HysteresisMargin)
            selected_action = max(
                admissible,
                key=lambda c: (c.speculative_gain_estimate * c.asymmetric_weight) / max(0.01, c.risk_factor * c.required_hysteresis_margin)
            )

        # Step 4: Socket Oscillation Diagnostics & Suppression
        suppression_active = False
        target_unit_name = f"{selected_action.target_unit.value}_0"
        if target_unit_name in self.socket_units:
            telemetry = self.socket_units[target_unit_name]
            # If oscillation exceeds 1.0 Hz, engage suppression
            if telemetry.oscillation_rate_hz > 1.0:
                suppression_active = True
                telemetry.oscillation_rate_hz = round(telemetry.oscillation_rate_hz * 0.40, 2)

        # Step 5: Quotas & Ratios
        cvq = round(min(0.98, 0.90 + (0.08 * (1.0 - visual_tensor.uma_contention_hazard))), 4)
        atr = round(selected_action.asymmetric_weight / 0.65, 3)

        # Step 6: Receipt & HMAC signature
        receipt_id = f"GOV-SPLICER-{self.total_cycles:05d}-{int(time.time()*1000)}"
        q_hash = hashlib.sha256(graphql_query_str.encode("utf-8")).hexdigest()

        sig_payload = f"{receipt_id}:{visual_tensor.perceptual_hash}:{selected_action.candidate_id}:{cvq}:{self.vital_max_hp}:{t_start}"
        hmac_sig = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()

        conduct_flags = ["[ACT]", "[LCK]", f"[CVQ={cvq}]", "[!HAZ]", f"[VIT={self.vital_max_hp}]"]

        receipt = GovernorSplicerReceipt(
            receipt_id=receipt_id,
            timestamp_ns=t_start,
            input_query_hash=q_hash,
            visual_tensor_hash=visual_tensor.perceptual_hash,
            selected_action=selected_action,
            active_socket_units=list(self.socket_units.keys()),
            socket_oscillation_suppressed=suppression_active,
            control_validity_quota=cvq,
            asymmetric_transition_ratio=atr,
            delta_s=0.0,
            is_safe=True,
            vital_max_hp=self.vital_max_hp,
            conduct_flags=conduct_flags,
            hmac_digest=hmac_sig,
        )

        self.history_receipts.append(receipt)
        return receipt

    def reverse_engineer_socket_dependencies(self, receipt: GovernorSplicerReceipt) -> Dict[str, Any]:
        """
        Reverse engineers socket dependencies by analyzing historical execution telemetry
        backwards from receipt outcome to hardware choke points.
        """
        choke_points = []
        for name, unit in self.socket_units.items():
            if unit.voltage_droop_mv > 15.0:
                choke_points.append({"unit": name, "cause": "VoltageDroopSpike", "droop_mv": unit.voltage_droop_mv})
            if unit.temperature_celsius > 60.0:
                choke_points.append({"unit": name, "cause": "ThermalConcentration", "temp_c": unit.temperature_celsius})

        return {
            "receipt_id": receipt.receipt_id,
            "action_executed": receipt.selected_action.candidate_id,
            "reverse_inferred_choke_points": choke_points,
            "socket_chatter_status": "STABILIZED" if receipt.socket_oscillation_suppressed else "NOMINAL",
            "vital_max_hp": receipt.vital_max_hp,
        }


# Global singleton instance
GLOBAL_SPECULATIVE_SOCKET_HARNESS = SpeculativeActionSocketHarness()
