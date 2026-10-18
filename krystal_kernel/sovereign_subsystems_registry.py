"""
KRYSTAL-STACK // SOVEREIGN SUBSYSTEMS REGISTRY & RUNTIME SINGLETONS
=============================================================================
Milestone: V-SOVEREIGN-FABRIC-01
Normative Reference: docs/architecture/AEP_CANONICAL_SPEC.md
Constitutional Invariant: VITAL_MAX_HP == 6

Centralizes authoritative runtime singletons across all four newly invented subsystems:
  1. AepPleaIntegratedEngine (V-AEP-PLEA-01)
  2. VulkanOpenVinoUsmBridge (V-VULKAN-USM-01)
  3. AepEconomyProcessor (V-AEP-ECONOMY-01)
  4. AepResearchActuator (V-AEP-RESEARCH-01)
"""

from __future__ import annotations

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.aep_plea_integrated_engine import AepPleaIntegratedEngine
from krystal_kernel.vulkan_openvino_usm_bridge import VulkanOpenVinoUsmBridge
from krystal_kernel.aep_economy_processor import AepEconomyProcessor, AssetType
from krystal_kernel.aep_research_actuator import AepResearchActuator

if VITAL_MAX_HP != 6:
    raise ContractViolation(f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}")

# 1. Sovereign PLEA-AEP Engine
GLOBAL_AEP_PLEA_ENGINE = AepPleaIntegratedEngine()

# 2. Direct Zero-Copy GPU USM Vulkan-OpenVINO Bridge
GLOBAL_VULKAN_USM_BRIDGE = VulkanOpenVinoUsmBridge()
# Pre-register default 540p viewport and 1080p target
GLOBAL_VULKAN_USM_BRIDGE.allocate_shared_tensor_buffer("viewport_540p", 960, 540)
GLOBAL_VULKAN_USM_BRIDGE.allocate_shared_tensor_buffer("target_1080p", 1920, 1080)

# 3. Transactional Double-Entry Economy Processor
GLOBAL_AEP_ECONOMY_PROCESSOR = AepEconomyProcessor()
# Seed initial test participants
GLOBAL_AEP_ECONOMY_PROCESSOR.deposit("alice", AssetType.CREDIT.value, 1500.0)
GLOBAL_AEP_ECONOMY_PROCESSOR.deposit("miner_bob", "RAW_ORE", 100.0)
GLOBAL_AEP_ECONOMY_PROCESSOR.deposit("miner_bob", "GAS_UNIT", 50.0)
GLOBAL_AEP_ECONOMY_PROCESSOR.deposit("miner_bob", AssetType.CREDIT.value, 200.0)

# 4. Autonomous Closed-Loop Research Actuator
GLOBAL_AEP_RESEARCH_ACTUATOR = AepResearchActuator(plea_aep_engine=GLOBAL_AEP_PLEA_ENGINE)
