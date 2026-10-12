"""
KRYSTAL-STACK // CENTRAL UNIT CORE MODULARITY AMPLIFIER & RISC-NPU BRIDGE
=============================================================================
Re-exports the Central Unit Core Modularity Amplifier from krystal_kernel
for economic engine modules and Mission Control HTTP dispatch.
"""

from krystal_kernel.central_unit_core_modularity_amplifier import (
    VITAL_MAX_HP,
    GOLDEN_RATIO,
    SAFE_VOLT_CEILING_V,
    KRiscOpcode,
    KRiscInstruction,
    KRiscCpuState,
    KRiscVirtualMachine,
    CorePartitionDomain,
    CoreModularityMetric,
    CentralUnitModularityProfile,
    CentralUnitModularityMeter,
    NpuCoProcessorState,
    NpuCoProcessorBridge,
    CompiledRiscAsset,
    RiscAssetAndInstructionSynthesizer,
    CentralUnitCoreModularityAmplifier,
    GLOBAL_CORE_MODULARITY_AMPLIFIER
)

__all__ = [
    "VITAL_MAX_HP",
    "GOLDEN_RATIO",
    "SAFE_VOLT_CEILING_V",
    "KRiscOpcode",
    "KRiscInstruction",
    "KRiscCpuState",
    "KRiscVirtualMachine",
    "CorePartitionDomain",
    "CoreModularityMetric",
    "CentralUnitModularityProfile",
    "CentralUnitModularityMeter",
    "NpuCoProcessorState",
    "NpuCoProcessorBridge",
    "CompiledRiscAsset",
    "RiscAssetAndInstructionSynthesizer",
    "CentralUnitCoreModularityAmplifier",
    "GLOBAL_CORE_MODULARITY_AMPLIFIER"
]
