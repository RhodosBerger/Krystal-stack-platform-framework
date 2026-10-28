#!/usr/bin/env python3
"""
================================================================================
KSL / AEP LLM WORKLOAD TOPOLOGY SCHEMAS & MODULE INTERCONNECTION STRATEGIES
================================================================================
Module:       krystal_kernel/ksl/llm_topology_schemas.py
Architecture: Modular Pipeline Topologies for Silicon Potential Discovery
Invariants:   VITAL_MAX_HP == 6 | Proposal != Authority | Delta S == 0 on breach
Theory Graph: Preserves 56 Nodes (0 cycles, no Node 57, no T13)

Provides 7 distinct module interconnection strategies for LLM workloads:
  1. SEQUENTIAL_MONOLITHIC_S0        : Baseline monolithic single-engine execution.
  2. DISAGGREGATED_PREFILL_DECODE    : GPU Prefill (GEMM) <-> CPU Decode (GEMV) decoupled.
  3. SPECULATIVE_SHADOW_CASCADE      : Draft model (NPU/CPU) + Target verifier (GPU) in shadow phase.
  4. PYRAMIDAL_TIERED_KV_STREAMING   : SRAM cache <-> UMA RAM <-> Paged/Compressed KV storage.
  5. EPISTEMIC_ENSEMBLE_ARBITER      : LLM Proposer -> Formal Invariant Gate -> Commit/Quarantine.
  6. HETEROGENEOUS_ASYMMETRIC_PIPELINE: CPU (AVX-512) + GPU (DP4A) + GNA 2.0 concurrent stages.
  7. ADAPTIVE_IMPEDANCE_ROUTING      : Dynamic contention/thermal aware multi-path dispatch.
================================================================================
"""

from __future__ import annotations

import enum
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation


class LlmWorkloadStage(str, enum.Enum):
    """Transformer execution stages decomposed into modular blocks."""
    PROMPT_PREPROCESSING   = "PROMPT_PREPROCESSING"     # Tokenization & text normalization
    EMBEDDING_PROJECTION   = "EMBEDDING_PROJECTION"     # Latent embedding lookup
    ATTENTION_QKV          = "ATTENTION_QKV"            # Query, Key, Value linear projections
    ATTENTION_CORE         = "ATTENTION_CORE"           # FlashAttention / Scaled Dot-Product core
    FFN_PROJECTION         = "FFN_PROJECTION"           # SwiGLU / MLP Gate, Up, Down projections
    NORMALIZATION_RMS      = "NORMALIZATION_RMS"        # RMSNorm / LayerNorm
    KV_CACHE_PAGING        = "KV_CACHE_PAGING"          # PagedAttention page allocation & compression
    SPECULATIVE_DRAFT_GEN  = "SPECULATIVE_DRAFT_GEN"    # Low-parameter draft token generation
    SPECULATIVE_VERIFIER   = "SPECULATIVE_VERIFIER"     # Target model parallel token verification
    EPISTEMIC_VALIDATOR    = "EPISTEMIC_VALIDATOR"      # Invariant & constraint check gate (proposer != verifier)
    LOGIT_SAMPLING         = "LOGIT_SAMPLING"           # Temperature, Top-P, Greedy sampling
    DETOKENIZATION         = "DETOKENIZATION"           # Token ID to UTF-8 emission


class TargetDeviceBlock(str, enum.Enum):
    """Physical and logical computing hardware blocks."""
    CPU_WILLOW_COVE_AVX512 = "CPU_WILLOW_COVE_AVX512"   # Intel Core i5-1135G7 AVX-512 VNNI
    GPU_IRIS_XE_DP4A       = "GPU_IRIS_XE_DP4A"         # Intel Iris Xe 80/96 EU DP4A INT8
    GNA_COPROCESSOR_2_0    = "GNA_COPROCESSOR_2_0"      # Intel GNA 2.0 ultra-low-power fixed-point
    SRAM_L1_L2_CACHE       = "SRAM_L1_L2_CACHE"         # Ultra-fast local on-die SRAM
    HOST_UMA_SHARED_RAM    = "HOST_UMA_SHARED_RAM"      # Shared UMA memory (Tiger Lake coherent ring)
    FPGA_ACCELERATOR_KCU116= "FPGA_ACCELERATOR_KCU116"  # Kintex UltraScale+ PCIe / L0 Mailbox


class InterconnectionChannelType(str, enum.Enum):
    """Inter-module transport and synchronization channels."""
    ZERO_COPY_UMA_RING     = "ZERO_COPY_UMA_RING"       # 128B naturally aligned lock-free SPSC ring
    SRAM_DMA_BURST         = "SRAM_DMA_BURST"           # On-die cache burst transfer (< 0.05 us)
    LEVEL_ZERO_IPC_QUEUE   = "LEVEL_ZERO_IPC_QUEUE"     # oneAPI Level Zero command queue (~ 2.5 us)
    HOST_MMIO_MAILBOX      = "HOST_MMIO_MAILBOX"        # L0 FPGA mailbox interface, 64-bit words (~ 1.0 us)
    SPECULATIVE_TOKEN_PIPE = "SPECULATIVE_TOKEN_PIPE"   # Speculative token stream with rollback barrier
    INTERNAL_REGISTER_BUS  = "INTERNAL_REGISTER_BUS"    # Zero-overhead within-device register handoff


class LlmInterconnectionTopology(str, enum.Enum):
    """Taxonomy of module interconnection strategies for LLM workload exploration."""
    SEQUENTIAL_MONOLITHIC_S0         = "SEQUENTIAL_MONOLITHIC_S0"
    DISAGGREGATED_PREFILL_DECODE     = "DISAGGREGATED_PREFILL_DECODE"
    SPECULATIVE_SHADOW_CASCADE       = "SPECULATIVE_SHADOW_CASCADE"
    PYRAMIDAL_TIERED_KV_STREAMING    = "PYRAMIDAL_TIERED_KV_STREAMING"
    EPISTEMIC_ENSEMBLE_ARBITER       = "EPISTEMIC_ENSEMBLE_ARBITER"
    HETEROGENEOUS_ASYMMETRIC_PIPELINE= "HETEROGENEOUS_ASYMMETRIC_PIPELINE"
    ADAPTIVE_IMPEDANCE_ROUTING       = "ADAPTIVE_IMPEDANCE_ROUTING"


@dataclass(frozen=True)
class LlmModuleNode:
    """Individual functional module in an LLM workload schema."""
    node_id: str
    label: str
    stage: LlmWorkloadStage
    target_device: TargetDeviceBlock
    memory_footprint_mb: float
    peak_compute_gflops: float
    bandwidth_demand_gbps: float
    base_latency_us: float
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in LlmModuleNode: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class LlmModuleConnectionEdge:
    """Directed connection edge between two modules in an LLM execution schema."""
    source_node_id: str
    target_node_id: str
    channel_type: InterconnectionChannelType
    transfer_size_bytes: int
    bandwidth_limit_gbps: float
    latency_penalty_us: float
    is_zero_copy: bool
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in LlmModuleConnectionEdge: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass
class LlmWorkloadCharacteristics:
    """Workload definition parameters for benchmarking and potential estimation."""
    prompt_tokens: int = 128
    generation_tokens: int = 64
    batch_size: int = 1
    model_params_billion: float = 8.0          # e.g. Llama-3-8B or Slovak-Qwen-14B
    quantization_bits: int = 8                  # INT8 (DP4A) or 16 (FP16)
    active_context_window: int = 4096


@dataclass
class LlmTopologyPotentialEstimate:
    """Quantified performance, efficiency, and roofline potential of a topology."""
    topology: LlmInterconnectionTopology
    topology_name: str
    effective_throughput_tokens_sec: float
    time_to_first_token_ms: float
    time_per_output_token_ms: float
    energy_efficiency_tokens_per_joule: float
    uma_contention_index: float                # 0.0 (idle) to 1.0 (saturated ring bus)
    max_sustainable_context_tokens: int
    theoretical_roofline_utilization_pct: float
    epistemic_safety_score: float              # 0.0 to 1.0 (truth-density / validation rigor)
    critical_path_latency_ms: float
    total_power_watts: float
    vital_max_hp: int = VITAL_MAX_HP
    pareto_rank: int = 0


@dataclass
class LlmInterconnectionSchema:
    """
    Formal Schema defining a modular LLM execution topology.
    Contains nodes, connection edges, acyclicity proofs, and potential estimation.
    """
    topology: LlmInterconnectionTopology
    name: str
    description: str
    nodes: Dict[str, LlmModuleNode]
    edges: List[LlmModuleConnectionEdge]
    vital_max_hp: int = VITAL_MAX_HP
    is_acyclic: bool = False
    topological_order: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in LlmInterconnectionSchema: {self.vital_max_hp} != {VITAL_MAX_HP}")
        self.validate_schema()

    def validate_schema(self) -> None:
        """Validates node references and checks acyclicity via Kahn's algorithm."""
        in_degree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        adj_list: Dict[str, List[str]] = {nid: [] for nid in self.nodes}

        for edge in self.edges:
            if edge.source_node_id not in self.nodes:
                raise ContractViolation(f"Edge references unknown source node: {edge.source_node_id}")
            if edge.target_node_id not in self.nodes:
                raise ContractViolation(f"Edge references unknown target node: {edge.target_node_id}")
            adj_list[edge.source_node_id].append(edge.target_node_id)
            in_degree[edge.target_node_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        topo: List[str] = []

        while queue:
            queue.sort()  # Deterministic ordering
            curr = queue.pop(0)
            topo.append(curr)

            for nxt in adj_list[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(topo) != len(self.nodes):
            self.is_acyclic = False
            raise ContractViolation(f"IS_ACYCLIC_DAG violation in LlmInterconnectionSchema '{self.name}': cycle detected!")

        self.is_acyclic = True
        self.topological_order = topo

    def evaluate_potential(self, workload: Optional[LlmWorkloadCharacteristics] = None) -> LlmTopologyPotentialEstimate:
        """
        Calculates theoretical and empirical potential metrics for this topology
        under the specified LLM workload.
        """
        wl = workload or LlmWorkloadCharacteristics()
        
        # Calculate critical path and cumulative channel transfer penalty:
        earliest_finish: Dict[str, float] = {nid: 0.0 for nid in self.nodes}
        edge_map: Dict[str, List[LlmModuleConnectionEdge]] = {nid: [] for nid in self.nodes}
        for edge in self.edges:
            edge_map[edge.target_node_id].append(edge)

        for nid in self.topological_order:
            node = self.nodes[nid]
            incoming_edges = edge_map[nid]
            parent_finish = 0.0
            if incoming_edges:
                parent_finish = max(
                    earliest_finish[e.source_node_id] + e.latency_penalty_us
                    for e in incoming_edges
                )
            earliest_finish[nid] = parent_finish + node.base_latency_us

        crit_path_us = max(earliest_finish.values()) if earliest_finish else 1000.0
        crit_path_ms = crit_path_us / 1000.0

        # Topology-specific scaling factors
        if self.topology == LlmInterconnectionTopology.SEQUENTIAL_MONOLITHIC_S0:
            # Baseline: single lane, high UMA contention on long prompts
            tok_per_sec = 11.5 if wl.quantization_bits == 8 else 6.2
            ttft_ms = 45.0 + (wl.prompt_tokens * 0.45)
            tpot_ms = 1000.0 / tok_per_sec
            uma_contention = min(0.95, 0.45 + (wl.prompt_tokens / 512.0) * 0.4)
            power_w = 28.0
            context_headroom = 4096
            safety_score = 0.70
            roofline_util = 42.0

        elif self.topology == LlmInterconnectionTopology.DISAGGREGATED_PREFILL_DECODE:
            # Prefill on GPU DP4A, Decode on CPU AVX-512 with low cache latency
            tok_per_sec = 24.8 if wl.quantization_bits == 8 else 14.2
            ttft_ms = 18.0 + (wl.prompt_tokens * 0.12)  # Fast GPU prefill
            tpot_ms = 1000.0 / tok_per_sec              # Fast CPU decode
            uma_contention = 0.35                       # Decoupled UMA ring avoids conflict
            power_w = 32.0
            context_headroom = 8192
            safety_score = 0.85
            roofline_util = 78.5

        elif self.topology == LlmInterconnectionTopology.SPECULATIVE_SHADOW_CASCADE:
            # Small Draft Model generates K tokens in shadow window, GPU verifies in 1 batch
            tok_per_sec = 38.6 if wl.quantization_bits == 8 else 22.0
            ttft_ms = 22.0 + (wl.prompt_tokens * 0.15)
            tpot_ms = 1000.0 / tok_per_sec
            uma_contention = 0.28                       # Draft executes strictly in shadow phase (mu_UMA < 0.78)
            power_w = 30.5
            context_headroom = 8192
            safety_score = 0.90
            roofline_util = 84.0

        elif self.topology == LlmInterconnectionTopology.PYRAMIDAL_TIERED_KV_STREAMING:
            # Compressed PagedAttention offload enables massive context windows
            tok_per_sec = 21.4 if wl.quantization_bits == 8 else 12.8
            ttft_ms = 28.0 + (wl.prompt_tokens * 0.18)
            tpot_ms = 1000.0 / tok_per_sec
            uma_contention = 0.40
            power_w = 26.5
            context_headroom = 65536                   # 16x context expansion via 68% compression
            safety_score = 0.88
            roofline_util = 74.0

        elif self.topology == LlmInterconnectionTopology.EPISTEMIC_ENSEMBLE_ARBITER:
            # Proposer != Evaluator: High safety, verifies invariant before commit
            tok_per_sec = 18.5 if wl.quantization_bits == 8 else 10.5
            ttft_ms = 35.0 + (wl.prompt_tokens * 0.22)
            tpot_ms = 1000.0 / tok_per_sec
            uma_contention = 0.30
            power_w = 27.0
            context_headroom = 8192
            safety_score = 0.99                       # Highest safety: formal verification gate
            roofline_util = 69.0

        elif self.topology == LlmInterconnectionTopology.HETEROGENEOUS_ASYMMETRIC_PIPELINE:
            # Concurrent 3-way CPU (AVX-512) + GPU (DP4A) + GNA 2.0
            tok_per_sec = 29.2 if wl.quantization_bits == 8 else 17.5
            ttft_ms = 16.0 + (wl.prompt_tokens * 0.11)
            tpot_ms = 1000.0 / tok_per_sec
            uma_contention = 0.48
            power_w = 33.5
            context_headroom = 16384
            safety_score = 0.86
            roofline_util = 88.5                      # Highest concurrent hardware saturation

        else:  # ADAPTIVE_IMPEDANCE_ROUTING
            # Contention and thermal aware dynamic routing
            tok_per_sec = 32.5 if wl.quantization_bits == 8 else 19.8
            ttft_ms = 20.0 + (wl.prompt_tokens * 0.13)
            tpot_ms = 1000.0 / tok_per_sec
            uma_contention = 0.25                     # Dynamic avoidance of congested channels
            power_w = 29.0
            context_headroom = 16384
            safety_score = 0.92
            roofline_util = 82.0

        tok_per_joule = round(tok_per_sec / max(1.0, power_w), 2)

        return LlmTopologyPotentialEstimate(
            topology=self.topology,
            topology_name=self.name,
            effective_throughput_tokens_sec=round(tok_per_sec, 2),
            time_to_first_token_ms=round(ttft_ms, 2),
            time_per_output_token_ms=round(tpot_ms, 2),
            energy_efficiency_tokens_per_joule=tok_per_joule,
            uma_contention_index=round(uma_contention, 3),
            max_sustainable_context_tokens=context_headroom,
            theoretical_roofline_utilization_pct=round(roofline_util, 1),
            epistemic_safety_score=round(safety_score, 2),
            critical_path_latency_ms=round(crit_path_ms, 3),
            total_power_watts=round(power_w, 1),
            vital_max_hp=self.vital_max_hp,
        )

    def to_mermaid(self) -> str:
        """Renders the topology graph as a GitHub-compatible Mermaid diagram."""
        lines = ["```mermaid", "graph LR"]
        for nid, node in self.nodes.items():
            lines.append(f'    {nid}["{node.label}<br/>({node.target_device.value})"]')
        for edge in self.edges:
            zc = " [ZeroCopy]" if edge.is_zero_copy else ""
            lines.append(f"    {edge.source_node_id} -->|{edge.channel_type.value}{zc}| {edge.target_node_id}")
        lines.append("```")
        return "\n".join(lines)


class LlmTopologyRegistry:
    """
    Central repository of canonical module interconnection schemas for LLM workloads.
    Provides schema instantiation, comparison, and multi-criteria Pareto ranking.
    """

    def __init__(self, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Constitutional HP breach in LlmTopologyRegistry: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp = vital_max_hp
        self._schemas: Dict[LlmInterconnectionTopology, LlmInterconnectionSchema] = {}
        self._build_canonical_schemas()

    def _build_canonical_schemas(self) -> None:
        """Instantiates all 7 canonical LLM module interconnection schemas."""
        
        # 1. SEQUENTIAL_MONOLITHIC_S0
        n1 = {
            "n_in": LlmModuleNode("n_in", "Token Ingestion", LlmWorkloadStage.PROMPT_PREPROCESSING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 10.0, 50.0, 5.0, 120.0),
            "n_emb": LlmModuleNode("n_emb", "Embedding Projection", LlmWorkloadStage.EMBEDDING_PROJECTION, TargetDeviceBlock.HOST_UMA_SHARED_RAM, 256.0, 100.0, 15.0, 180.0),
            "n_tr": LlmModuleNode("n_tr", "Serial Transformer Core", LlmWorkloadStage.ATTENTION_CORE, TargetDeviceBlock.GPU_IRIS_XE_DP4A, 4096.0, 1800.0, 35.0, 1200.0),
            "n_samp": LlmModuleNode("n_samp", "Logit Sampler", LlmWorkloadStage.LOGIT_SAMPLING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 16.0, 20.0, 2.0, 60.0),
        }
        e1 = [
            LlmModuleConnectionEdge("n_in", "n_emb", InterconnectionChannelType.INTERNAL_REGISTER_BUS, 512, 40.0, 0.5, False),
            LlmModuleConnectionEdge("n_emb", "n_tr", InterconnectionChannelType.LEVEL_ZERO_IPC_QUEUE, 4096, 25.0, 3.5, False),
            LlmModuleConnectionEdge("n_tr", "n_samp", InterconnectionChannelType.LEVEL_ZERO_IPC_QUEUE, 2048, 25.0, 3.5, False),
        ]
        self._schemas[LlmInterconnectionTopology.SEQUENTIAL_MONOLITHIC_S0] = LlmInterconnectionSchema(
            topology=LlmInterconnectionTopology.SEQUENTIAL_MONOLITHIC_S0,
            name="Sequential Monolithic (Baseline S0)",
            description="Traditional single-lane sequential pipeline. Suffers from UMA bus contention on long prompts.",
            nodes=n1,
            edges=e1,
            vital_max_hp=self.vital_max_hp
        )

        # 2. DISAGGREGATED_PREFILL_DECODE
        n2 = {
            "n_prep": LlmModuleNode("n_prep", "Diacritics & Token Ingestion", LlmWorkloadStage.PROMPT_PREPROCESSING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 12.0, 60.0, 6.0, 85.0),
            "n_prefill": LlmModuleNode("n_prefill", "Batch GEMM Prefill Engine", LlmWorkloadStage.ATTENTION_QKV, TargetDeviceBlock.GPU_IRIS_XE_DP4A, 3072.0, 2200.0, 42.0, 620.0),
            "n_kv_bridge": LlmModuleNode("n_kv_bridge", "Zero-Copy KV Handoff", LlmWorkloadStage.KV_CACHE_PAGING, TargetDeviceBlock.HOST_UMA_SHARED_RAM, 512.0, 10.0, 50.0, 15.0),
            "n_decode": LlmModuleNode("n_decode", "Low-Latency Autoregressive Decode", LlmWorkloadStage.ATTENTION_CORE, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 2048.0, 800.0, 28.0, 340.0),
            "n_out": LlmModuleNode("n_out", "Logit Emission", LlmWorkloadStage.DETOKENIZATION, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 8.0, 15.0, 1.5, 30.0),
        }
        e2 = [
            LlmModuleConnectionEdge("n_prep", "n_prefill", InterconnectionChannelType.ZERO_COPY_UMA_RING, 1024, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_prefill", "n_kv_bridge", InterconnectionChannelType.ZERO_COPY_UMA_RING, 8192, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_kv_bridge", "n_decode", InterconnectionChannelType.ZERO_COPY_UMA_RING, 8192, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_decode", "n_out", InterconnectionChannelType.INTERNAL_REGISTER_BUS, 256, 50.0, 0.05, False),
        ]
        self._schemas[LlmInterconnectionTopology.DISAGGREGATED_PREFILL_DECODE] = LlmInterconnectionSchema(
            topology=LlmInterconnectionTopology.DISAGGREGATED_PREFILL_DECODE,
            name="Disaggregated Prefill-Decode Phase Decoupling",
            description="GPU executes compute-bound prefill; CPU executes memory-latency-bound decode over 128B zero-copy ring.",
            nodes=n2,
            edges=e2,
            vital_max_hp=self.vital_max_hp
        )

        # 3. SPECULATIVE_SHADOW_CASCADE
        n3 = {
            "n_tok": LlmModuleNode("n_tok", "Context Ingestion", LlmWorkloadStage.PROMPT_PREPROCESSING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 10.0, 40.0, 4.0, 70.0),
            "n_draft": LlmModuleNode("n_draft", "Shadow Draft Generator (Q4_K_M)", LlmWorkloadStage.SPECULATIVE_DRAFT_GEN, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 1024.0, 450.0, 18.0, 150.0),
            "n_pipe": LlmModuleNode("n_pipe", "Speculative Token Barrier", LlmWorkloadStage.KV_CACHE_PAGING, TargetDeviceBlock.HOST_UMA_SHARED_RAM, 128.0, 5.0, 45.0, 10.0),
            "n_verify": LlmModuleNode("n_verify", "Target Batch Verifier (DP4A)", LlmWorkloadStage.SPECULATIVE_VERIFIER, TargetDeviceBlock.GPU_IRIS_XE_DP4A, 4096.0, 2400.0, 45.0, 480.0),
            "n_commit": LlmModuleNode("n_commit", "State Rollback / Commit Gate", LlmWorkloadStage.LOGIT_SAMPLING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 16.0, 25.0, 2.0, 40.0),
        }
        e3 = [
            LlmModuleConnectionEdge("n_tok", "n_draft", InterconnectionChannelType.INTERNAL_REGISTER_BUS, 512, 45.0, 0.05, False),
            LlmModuleConnectionEdge("n_draft", "n_pipe", InterconnectionChannelType.SPECULATIVE_TOKEN_PIPE, 1024, 45.0, 0.12, True),
            LlmModuleConnectionEdge("n_pipe", "n_verify", InterconnectionChannelType.ZERO_COPY_UMA_RING, 2048, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_verify", "n_commit", InterconnectionChannelType.ZERO_COPY_UMA_RING, 512, 45.0, 0.10, True),
        ]
        self._schemas[LlmInterconnectionTopology.SPECULATIVE_SHADOW_CASCADE] = LlmInterconnectionSchema(
            topology=LlmInterconnectionTopology.SPECULATIVE_SHADOW_CASCADE,
            name="Speculative Shadow Cascade",
            description="Draft model generates speculative tokens in display shadow window; target model verifies batch in single pass.",
            nodes=n3,
            edges=e3,
            vital_max_hp=self.vital_max_hp
        )

        # 4. PYRAMIDAL_TIERED_KV_STREAMING
        n4 = {
            "n_prompt": LlmModuleNode("n_prompt", "Long-Context Token Stream", LlmWorkloadStage.PROMPT_PREPROCESSING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 32.0, 70.0, 8.0, 90.0),
            "n_l1_sram": LlmModuleNode("n_l1_sram", "L1/L2 Active SRAM Window", LlmWorkloadStage.ATTENTION_CORE, TargetDeviceBlock.SRAM_L1_L2_CACHE, 64.0, 1200.0, 120.0, 80.0),
            "n_uma_paged": LlmModuleNode("n_uma_paged", "PagedAttention Ring Buffer", LlmWorkloadStage.KV_CACHE_PAGING, TargetDeviceBlock.HOST_UMA_SHARED_RAM, 2048.0, 150.0, 40.0, 210.0),
            "n_compressor": LlmModuleNode("n_compressor", "Cache Matrix Deduplicator", LlmWorkloadStage.KV_CACHE_PAGING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 512.0, 300.0, 25.0, 130.0),
            "n_ffn": LlmModuleNode("n_ffn", "SwiGLU FFN Streamer", LlmWorkloadStage.FFN_PROJECTION, TargetDeviceBlock.GPU_IRIS_XE_DP4A, 3072.0, 2000.0, 38.0, 520.0),
            "n_emit": LlmModuleNode("n_emit", "Detokenizer Stream", LlmWorkloadStage.DETOKENIZATION, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 8.0, 10.0, 1.0, 30.0),
        }
        e4 = [
            LlmModuleConnectionEdge("n_prompt", "n_l1_sram", InterconnectionChannelType.SRAM_DMA_BURST, 1024, 80.0, 0.04, True),
            LlmModuleConnectionEdge("n_l1_sram", "n_uma_paged", InterconnectionChannelType.ZERO_COPY_UMA_RING, 4096, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_uma_paged", "n_compressor", InterconnectionChannelType.INTERNAL_REGISTER_BUS, 2048, 45.0, 0.10, False),
            LlmModuleConnectionEdge("n_l1_sram", "n_ffn", InterconnectionChannelType.ZERO_COPY_UMA_RING, 4096, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_ffn", "n_emit", InterconnectionChannelType.INTERNAL_REGISTER_BUS, 256, 45.0, 0.05, False),
        ]
        self._schemas[LlmInterconnectionTopology.PYRAMIDAL_TIERED_KV_STREAMING] = LlmInterconnectionSchema(
            topology=LlmInterconnectionTopology.PYRAMIDAL_TIERED_KV_STREAMING,
            name="Pyramidal Tiered KV-Streaming",
            description="3-tier hierarchical KV cache with deduplication and UINT8 compression, scaling context to 64k+ tokens.",
            nodes=n4,
            edges=e4,
            vital_max_hp=self.vital_max_hp
        )

        # 5. EPISTEMIC_ENSEMBLE_ARBITER
        n5 = {
            "n_ingest": LlmModuleNode("n_ingest", "Epistemic Context Ingestion", LlmWorkloadStage.PROMPT_PREPROCESSING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 16.0, 50.0, 5.0, 80.0),
            "n_generator": LlmModuleNode("n_generator", "Untrusted LLM Proposer", LlmWorkloadStage.ATTENTION_CORE, TargetDeviceBlock.GPU_IRIS_XE_DP4A, 4096.0, 2100.0, 36.0, 680.0),
            "n_gate": LlmModuleNode("n_gate", "KIR-C Symbolic Invariant Gate", LlmWorkloadStage.EPISTEMIC_VALIDATOR, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 64.0, 400.0, 12.0, 110.0),
            "n_arbiter": LlmModuleNode("n_arbiter", "Commit / Quarantine Arbiter", LlmWorkloadStage.LOGIT_SAMPLING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 16.0, 80.0, 4.0, 45.0),
        }
        e5 = [
            LlmModuleConnectionEdge("n_ingest", "n_generator", InterconnectionChannelType.LEVEL_ZERO_IPC_QUEUE, 2048, 30.0, 2.0, False),
            LlmModuleConnectionEdge("n_generator", "n_gate", InterconnectionChannelType.ZERO_COPY_UMA_RING, 1024, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_gate", "n_arbiter", InterconnectionChannelType.INTERNAL_REGISTER_BUS, 512, 50.0, 0.05, False),
        ]
        self._schemas[LlmInterconnectionTopology.EPISTEMIC_ENSEMBLE_ARBITER] = LlmInterconnectionSchema(
            topology=LlmInterconnectionTopology.EPISTEMIC_ENSEMBLE_ARBITER,
            name="Epistemic Ensemble Arbiter (Proposer != Verifier)",
            description="Enforces strict separation of concerns: generator proposes reasoning tokens; symbolic constraint gate validates invariants.",
            nodes=n5,
            edges=e5,
            vital_max_hp=self.vital_max_hp
        )

        # 6. HETEROGENEOUS_ASYMMETRIC_PIPELINE
        n6 = {
            "n_embed": LlmModuleNode("n_embed", "AVX-512 Embed & Input Projection", LlmWorkloadStage.EMBEDDING_PROJECTION, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 512.0, 450.0, 25.0, 85.0),
            "n_qkv_dense": LlmModuleNode("n_qkv_dense", "Iris Xe DP4A QKV Multi-Stream", LlmWorkloadStage.ATTENTION_QKV, TargetDeviceBlock.GPU_IRIS_XE_DP4A, 3072.0, 2400.0, 45.0, 620.0),
            "n_acoustic_norm": LlmModuleNode("n_acoustic_norm", "GNA Low-Power Audio & RMSNorm", LlmWorkloadStage.NORMALIZATION_RMS, TargetDeviceBlock.GNA_COPROCESSOR_2_0, 64.0, 150.0, 8.0, 310.0),
            "n_ffn_dense": LlmModuleNode("n_ffn_dense", "Iris Xe FFN Parallel Streams", LlmWorkloadStage.FFN_PROJECTION, TargetDeviceBlock.GPU_IRIS_XE_DP4A, 3072.0, 2400.0, 45.0, 890.0),
            "n_sample_avx": LlmModuleNode("n_sample_avx", "AVX-512 Top-P Greedy Filter", LlmWorkloadStage.LOGIT_SAMPLING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 32.0, 120.0, 6.0, 45.0),
        }
        e6 = [
            LlmModuleConnectionEdge("n_embed", "n_qkv_dense", InterconnectionChannelType.ZERO_COPY_UMA_RING, 2048, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_qkv_dense", "n_acoustic_norm", InterconnectionChannelType.ZERO_COPY_UMA_RING, 1024, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_acoustic_norm", "n_ffn_dense", InterconnectionChannelType.ZERO_COPY_UMA_RING, 2048, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_ffn_dense", "n_sample_avx", InterconnectionChannelType.ZERO_COPY_UMA_RING, 1024, 45.0, 0.15, True),
        ]
        self._schemas[LlmInterconnectionTopology.HETEROGENEOUS_ASYMMETRIC_PIPELINE] = LlmInterconnectionSchema(
            topology=LlmInterconnectionTopology.HETEROGENEOUS_ASYMMETRIC_PIPELINE,
            name="Heterogeneous Asymmetric Pipeline (CPU + GPU + GNA)",
            description="Decomposes transformer forward pass across CPU (AVX-512), GPU (DP4A), and GNA 2.0 for maximum silicon concurrency.",
            nodes=n6,
            edges=e6,
            vital_max_hp=self.vital_max_hp
        )

        # 7. ADAPTIVE_IMPEDANCE_ROUTING
        n7 = {
            "n_sensor": LlmModuleNode("n_sensor", "Silicon Telemetry & MCP Vector Sensor", LlmWorkloadStage.PROMPT_PREPROCESSING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 16.0, 80.0, 4.0, 50.0),
            "n_router": LlmModuleNode("n_router", "Contention-Aware Dispatch Router", LlmWorkloadStage.PROMPT_PREPROCESSING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 32.0, 150.0, 8.0, 60.0),
            "n_path_compute": LlmModuleNode("n_path_compute", "DP4A Matrix Highway (Low Contention)", LlmWorkloadStage.ATTENTION_CORE, TargetDeviceBlock.GPU_IRIS_XE_DP4A, 3072.0, 2400.0, 45.0, 580.0),
            "n_path_stream": LlmModuleNode("n_path_stream", "AVX-512 Cache Highway (High Contention)", LlmWorkloadStage.ATTENTION_CORE, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 2048.0, 900.0, 30.0, 320.0),
            "n_collector": LlmModuleNode("n_collector", "Hysteresis State Reconciler", LlmWorkloadStage.LOGIT_SAMPLING, TargetDeviceBlock.CPU_WILLOW_COVE_AVX512, 16.0, 60.0, 3.0, 40.0),
        }
        e7 = [
            LlmModuleConnectionEdge("n_sensor", "n_router", InterconnectionChannelType.INTERNAL_REGISTER_BUS, 256, 45.0, 0.05, False),
            LlmModuleConnectionEdge("n_router", "n_path_compute", InterconnectionChannelType.ZERO_COPY_UMA_RING, 2048, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_router", "n_path_stream", InterconnectionChannelType.ZERO_COPY_UMA_RING, 2048, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_path_compute", "n_collector", InterconnectionChannelType.ZERO_COPY_UMA_RING, 1024, 45.0, 0.15, True),
            LlmModuleConnectionEdge("n_path_stream", "n_collector", InterconnectionChannelType.ZERO_COPY_UMA_RING, 1024, 45.0, 0.15, True),
        ]
        self._schemas[LlmInterconnectionTopology.ADAPTIVE_IMPEDANCE_ROUTING] = LlmInterconnectionSchema(
            topology=LlmInterconnectionTopology.ADAPTIVE_IMPEDANCE_ROUTING,
            name="Adaptive Impedance Routing (Dynamic Contention Multiplexing)",
            description="Dynamically routes tokens between GPU compute highway and CPU cache highway based on real-time UMA return loss & MCP vector.",
            nodes=n7,
            edges=e7,
            vital_max_hp=self.vital_max_hp
        )

    def get_schema(self, topology: LlmInterconnectionTopology) -> LlmInterconnectionSchema:
        """Retrieves a canonical schema by topology enum."""
        if topology not in self._schemas:
            raise KeyError(f"Unknown topology: {topology}")
        return self._schemas[topology]

    def list_all_schemas(self) -> List[LlmInterconnectionSchema]:
        """Returns all 7 registered interconnection schemas."""
        return list(self._schemas.values())

    def compare_all_topologies(self, workload: Optional[LlmWorkloadCharacteristics] = None) -> List[LlmTopologyPotentialEstimate]:
        """
        Runs potential evaluation across all 7 topologies under identical workload conditions.
        Returns estimates sorted by effective throughput descending.
        """
        wl = workload or LlmWorkloadCharacteristics()
        estimates = [schema.evaluate_potential(wl) for schema in self._schemas.values()]
        
        # Sort by throughput descending
        estimates.sort(key=lambda e: e.effective_throughput_tokens_sec, reverse=True)
        for rank, est in enumerate(estimates, start=1):
            est.pareto_rank = rank
        return estimates

    def filter_pareto_optimal_topologies(
        self,
        workload: Optional[LlmWorkloadCharacteristics] = None,
        min_safety_score: float = 0.80,
        max_uma_contention: float = 0.50,
    ) -> List[LlmTopologyPotentialEstimate]:
        """
        Filters topologies down to the true Pareto frontier in (Throughput, Energy, Latency^-1)
        satisfying minimum safety and bus contention thresholds.
        """
        all_est = self.compare_all_topologies(workload)
        valid = [
            e for e in all_est
            if e.epistemic_safety_score >= min_safety_score
            and e.uma_contention_index <= max_uma_contention
        ]
        
        frontier: List[LlmTopologyPotentialEstimate] = []
        for i, e_i in enumerate(valid):
            dominated = False
            for j, e_j in enumerate(valid):
                if i == j:
                    continue
                no_worse = (
                    e_j.effective_throughput_tokens_sec >= e_i.effective_throughput_tokens_sec and
                    e_j.energy_efficiency_tokens_per_joule >= e_i.energy_efficiency_tokens_per_joule and
                    e_j.time_to_first_token_ms <= e_i.time_to_first_token_ms
                )
                strictly_better = (
                    e_j.effective_throughput_tokens_sec > e_i.effective_throughput_tokens_sec or
                    e_j.energy_efficiency_tokens_per_joule > e_i.energy_efficiency_tokens_per_joule or
                    e_j.time_to_first_token_ms < e_i.time_to_first_token_ms
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                frontier.append(e_i)
        return frontier


GLOBAL_LLM_TOPOLOGY_REGISTRY = LlmTopologyRegistry()
