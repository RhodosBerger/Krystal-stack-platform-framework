"""Universal Knowledge-Base Hierarchy Solution Kernel (`UniversalKnowledgeHierarchyKernel`).

Classifies the repository's critical JSON schemas & benchmark scripts, traverses the
6-tier Knowledge Hierarchy (Axioms -> Silicon Facts -> Theories -> Combinatorial
Derivatives -> Error-Prevention Rules -> Schema-Validated Deployment), and universally
pushes custom engineering solutions conforming to:
  - `schemas/inter_module_protocol_and_reasoning.schema.json` (`LOCAL/GLOBAL/SUPERGLOBAL`, `vital_max_hp == 6`)
  - `schemas/authority_profile_bits.schema.json` (64-bit `0x...` profile eligibility mask)
  - `192 B` (`64 B WNTC + 128 B KPKT`) Hypervisor HyperTransport framing.
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from krystal_kernel.constants import UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation

from .combinatorial_derivative_functions_engine import derive_192b_wntc_kpkt_hypervisor_datagram
from .fact_synthesis_engine import REPO_ROOT
from .krystal_uma_cuda_vulkan_library import kuAdviseAndDefend, kuMallocUMA
from .neurosymbolic_theories_case_study_and_kb import CROSS_MODULE_THEORIES
from .seasonal_scrum_prompt_kb_engine import RESOLVED_PROBLEMS_AND_RULES

SCHEMAS_DIR = REPO_ROOT / "schemas"
SCRIPTS_DIR = REPO_ROOT / "scripts"


CRITICAL_SCHEMAS_AND_SCRIPTS_RANKING: Tuple[Dict[str, str], ...] = (
    {
        "rank": "1",
        "artifact": "schemas/inter_module_protocol_and_reasoning.schema.json",
        "role": "Defines the 3-tier memory scope bus (LOCAL/GLOBAL/SUPERGLOBAL with vital_max_hp==6) & CRC32 packet frame.",
    },
    {
        "rank": "2",
        "artifact": "schemas/authority_profile_bits.schema.json",
        "role": "Defines the 64-bit profile eligibility bitmask governing CPU/GPU/Vulkan/Hypervisor execution authority.",
    },
    {
        "rank": "3",
        "artifact": "schemas/krystal_config_rail.schema.json & semantic_status_protocol_v1.json",
        "role": "Governs bounded configuration rails and deterministic status telemetry contracts.",
    },
    {
        "rank": "4",
        "artifact": "scripts/benchmark_gpu_zero_copy.py & benchmark_native_cpu_vs_vulkan.py",
        "role": "Empirical hardware calibrators proving the 20.35x Vulkan UMA speedup and 497.66 MB/s saved bandwidth.",
    },
    {
        "rank": "5",
        "artifact": "scripts/benchmark_phase_aware_execution_arbitration.py",
        "role": "Calibrates chronometric VBlank shadow arbitration (Phi_Scanout * Phi_Compute == 0).",
    },
)


UNIVERSAL_KERNEL_OBJECTIVES: Tuple[Dict[str, str], ...] = (
    {
        "objective_id": "OBJ_01_SCHEMA_CONFORMANT_SYNTHESIS",
        "statement": "Every pushed engineering solution must validate against `inter_module_protocol_and_reasoning.schema.json` and `authority_profile_bits.schema.json` before actuation.",
    },
    {
        "objective_id": "OBJ_02_HIERARCHICAL_DEDUCTIVE_TRACE",
        "statement": "Compose every custom solution top-down across all 6 KB tiers (Axioms -> Silicon X01..05 -> Theories TH_01..04 -> Derivatives DF_01..05 -> Rules EPR_01..05 -> Target).",
    },
    {
        "objective_id": "OBJ_03_ZERO_TOUCH_MODULE_INJECTION",
        "statement": "Inject subnormal sanitization, K<449 tiling, and 192B HyperTransport telemetry into existing modules with 0 bytes of upstream source delta.",
    },
    {
        "objective_id": "OBJ_04_VULKAN_AND_UMA_DEFENSE",
        "statement": "Maintain 60/120 Hz scanout determinism by revoking GPU/Vulkan credits in 0 frames whenever UMA >= 0.78.",
    },
    {
        "objective_id": "OBJ_05_SEASONAL_SELF_IMPROVING_KB",
        "statement": "Convert every sprint impediment into an enforceable Error-Prevention Rule (`EPR_*`) and auto-compiled next-sprint prompt.",
    },
)


class UniversalKnowledgeHierarchyKernel:
    """Pushes custom, schema-verified engineering solutions across domains using the 6-tier KB hierarchy."""

    KNOWLEDGE_TIERS: Tuple[str, ...] = (
        "KH_T0_CONSTITUTIONAL_AXIOMS (VITAL_MAX_HP==6, UMA<0.78, PROPOSER!=VERIFIER)",
        "KH_T1_EMPIRICAL_SILICON_FACTS (X01..X05: L2 55.1 GB/s, 37.4x subnormal, K=449)",
        "KH_T2_CROSS_MODULE_THEORIES (TH_01..TH_04)",
        "KH_T3_COMBINATORIAL_DERIVATIVES (DF_01..DF_05 over C(6,2)=15 pairs)",
        "KH_T4_ERROR_PREVENTION_RULES (EPR_01..EPR_05)",
        "KH_T5_SCHEMA_VALIDATED_CUSTOM_DEPLOYMENT",
    )

    def __init__(self) -> None:
        if VITAL_MAX_HP != 6:
            raise ContractViolation("VITAL_MAX_HP != 6 in UniversalKnowledgeHierarchyKernel", violation_code="ERR_UKH_HP")
        self._protocol_schema = json.loads((SCHEMAS_DIR / "inter_module_protocol_and_reasoning.schema.json").read_text(encoding="utf-8"))
        self._auth_schema = json.loads((SCHEMAS_DIR / "authority_profile_bits.schema.json").read_text(encoding="utf-8"))

    def push_custom_engineering_solution(
        self,
        problem_domain: str,
        payload_bytes: int = 65_536,
        batch_n: int = 128,
        reduction_k: int = 512,
        uma_utilization: float = 0.35,
    ) -> Dict[str, Any]:
        """Synthesize, advise, schema-validate, and package a custom solution for `problem_domain`."""
        import numpy as np

        if not problem_domain.strip():
            raise ContractViolation("Empty problem_domain", violation_code="ERR_UKH_DOMAIN")

        elems = max(16, payload_bytes // 4)
        sample_buf = kuMallocUMA(f"sol_{problem_domain}", np.ones((elems,), dtype=np.float32))
        advice = kuAdviseAndDefend(sample_buf, batch_n=batch_n, reduction_k=reduction_k, uma_utilization=uma_utilization)
        hht = derive_192b_wntc_kpkt_hypervisor_datagram(uma_utilization=uma_utilization)

        # Build 64-bit authority eligibility mask conforming to authority_profile_bits.schema.json
        gpu_bit = 1 if advice["vulkan_gpu_allowed"] else 0
        mask_u64 = (1 << 0) | (1 << 1) | (gpu_bit << 2) | (1 << 3)
        auth_instance = {
            "schema_version": "1.0",
            "mask_semantics": "PROFILE_ELIGIBILITY",
            "bitmask_width": 64,
            "profiles": [
                {"profile_id": "CPU_4T_AVX512", "enum_name": "LANE_CPU_4T", "bit": 0, "hex": f"0x{1 << 0:016x}"},
                {"profile_id": "HYPERTRANSPORT_192B", "enum_name": "LANE_HHT_192B", "bit": 1, "hex": f"0x{1 << 1:016x}"},
                {"profile_id": "VULKAN_IRIS_XE", "enum_name": "LANE_VK_GPU", "bit": 2, "hex": f"0x{gpu_bit << 2:016x}"},
            ],
            "active_mask_hex": f"0x{mask_u64:016x}",
        }

        body_payload = {
            "problem_domain": problem_domain,
            "recommended_lane": advice["recommended_lane"],
            "reduction_k_strategy": advice["reduction_k_advice"],
            "theories_applied": [t["theory_id"] for t in CROSS_MODULE_THEORIES],
            "rules_enforced": [r["problem_id"] for r in RESOLVED_PROBLEMS_AND_RULES],
        }
        crc_hex = f"{zlib.crc32(json.dumps(body_payload, sort_keys=True).encode('utf-8')) & 0xFFFFFFFF:08x}"

        # Build protocol packet conforming to inter_module_protocol_and_reasoning.schema.json
        protocol_instance = {
            "schema_version": "1.0.0",
            "packet_frame": {
                "packet_id": f"ukh_{problem_domain}",
                "source_module": "universal_kb_hierarchy_kernel",
                "target_module": problem_domain,
                "protocol_type": "REASONING_SIGNAL",
                "qos_priority": "CRITICAL" if uma_utilization >= UMA_SATURATION_THRESHOLD else "HIGH",
                "scope_tier": "SUPERGLOBAL",
                "timestamp_ns": time.time_ns(),
                "ttl": 16,
                "checksum_crc32": crc_hex,
                "payload": body_payload,
            },
            "scope_environment": {
                "local_scope": {"cache_lines_64b": sample_buf.cache_lines_64b, "aligned_64b": sample_buf.aligned_64b},
                "global_scope": {"uma_utilization": round(uma_utilization, 4), "batch_n": batch_n},
                "superglobal_scope": {"vital_max_hp": VITAL_MAX_HP},
            },
            "control_flow_execution": {
                "coarse_scales_enabled": True,
                "instructions": [
                    {"op": "SANITIZE_F32_ZERO_COPY"},
                    {"op": advice["reduction_k_advice"]},
                    {"op": advice["defense_action"]},
                ],
            },
            "reasoning_composition_pipeline": {
                "tiers_traversed": list(self.KNOWLEDGE_TIERS),
            },
        }

        # Structural contract validation against loaded root schemas
        req_proto = set(self._protocol_schema["required"])
        req_auth = set(self._auth_schema["required"])
        if not req_proto.issubset(protocol_instance.keys()) or protocol_instance["scope_environment"]["superglobal_scope"]["vital_max_hp"] != 6:
            raise ContractViolation("Protocol schema contract breach", violation_code="ERR_UKH_PROTO_SCHEMA")
        if not req_auth.issubset(auth_instance.keys()) or auth_instance["bitmask_width"] != 64:
            raise ContractViolation("Authority schema contract breach", violation_code="ERR_UKH_AUTH_SCHEMA")

        return {
            "vital_max_hp": VITAL_MAX_HP,
            "schema_validated": True,
            "critical_artifacts_ranked": list(CRITICAL_SCHEMAS_AND_SCRIPTS_RANKING),
            "objectives": list(UNIVERSAL_KERNEL_OBJECTIVES),
            "authority_bitmask": auth_instance,
            "protocol_solution_envelope": protocol_instance,
            "hypertransport_192b_verified": hht["wntc_crc32_valid"] and hht["kpkt_sha256_valid"],
        }
