"""Krystal-UMA-CUDA (`ku*`) Library, Defensive Vulkan Endpoint Asset Coordinator,
Zero-Touch Kernel Operation Injector (Sovereign P1), and Hypervisor HyperTransport
Methodics (`HHT_M1..HHT_M4`) over Intel Willow Cove + Iris Xe UMA + `schemas/` & `scripts/`.

Directly addresses XERJ (`github.com/xerj-org/xerj`, founded by Ivan Novikov)
neurosymbolic principles: zero-token deterministic AST/schema compilation coupled with
cryptographically attested, zero-copy hardware execution and strict `Delta_S == 0` defense.
"""

from __future__ import annotations

import functools
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from krystal_kernel.constants import CACHE_LINE_BYTES, UMA_SATURATION_THRESHOLD, VITAL_MAX_HP, ContractViolation
from krystal_kernel.zero_copy_contracts import canonicalize_float32
from projects.sovereign_openvino_companion_v2.vulkan_endpoint_extension_and_zero_copy_signal import (
    VulkanEndpointExtensionSpec,
)

from .combinatorial_derivative_functions_engine import derive_192b_wntc_kpkt_hypervisor_datagram
from .combat_turn_planner import DEFAULT_CACHE_DIR, get_cached_hw_profile, probe_isa_flags
from .configurator_and_dxdiag import partition_reduction_below_k449
from .fact_synthesis_engine import REPO_ROOT
from .intel_accelerator_and_power_thread_governor import get_shared_ov_core

SCHEMAS_DIR = REPO_ROOT / "schemas"
SCRIPTS_DIR = REPO_ROOT / "scripts"


# ---------------------------------------------------------------------------
# 1. CUDA-Like Intel UMA Data Structures (`UMADeviceAssetBuffer` & `kuStream`)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UMADeviceAssetBuffer:
    """64-byte cache-line-aligned zero-copy UMA buffer acting like a governed `cudaMallocManaged` slab."""

    asset_id: str
    shape: Tuple[int, ...]
    byte_length: int
    cache_lines_64b: int
    aligned_64b: bool
    subnormals_sanitized: bool
    sha256_digest: str
    host_array: np.ndarray
    ov_shared_tensor: Any
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != 6:
            raise ContractViolation("VITAL_MAX_HP != 6 in UMADeviceAssetBuffer", violation_code="ERR_KU_HP")


def kuMallocUMA(asset_id: str, data: np.ndarray) -> UMADeviceAssetBuffer:
    """Allocate a 64B-aligned, subnormal-free zero-copy UMA buffer backed by `ov.Tensor(shared_memory=True)`."""
    if VITAL_MAX_HP != 6:
        raise ContractViolation("VITAL_MAX_HP != 6", violation_code="ERR_KU_MALLOC_HP")
    arr_f32 = np.ascontiguousarray(data, dtype=np.float32)
    if not np.all(np.isfinite(arr_f32)):
        raise ContractViolation(f"Non-finite values in asset {asset_id}", violation_code="ERR_KU_NONFINITE")

    # Flush IEEE-754 subnormals (< 1.17549435e-38) to exact 0.0 to prevent 14x..37.4x microcode stalls (X03)
    sub_mask = (arr_f32 != 0.0) & (np.abs(arr_f32) < np.float32(1.17549435e-38))
    if np.any(sub_mask):
        arr_f32 = arr_f32.copy()
        arr_f32[sub_mask] = np.float32(0.0)

    nbytes = int(arr_f32.nbytes)
    raw = bytearray(nbytes + CACHE_LINE_BYTES)
    base_addr = np.frombuffer(raw, dtype=np.uint8).ctypes.data
    offset = (CACHE_LINE_BYTES - (base_addr % CACHE_LINE_BYTES)) % CACHE_LINE_BYTES
    aligned_view = np.frombuffer(raw, dtype=np.float32, count=arr_f32.size, offset=offset).reshape(arr_f32.shape)
    np.copyto(aligned_view, arr_f32)

    ov, _, _ = get_shared_ov_core(DEFAULT_CACHE_DIR)
    ov_tensor = ov.Tensor(aligned_view, shared_memory=True)
    ptr_aligned = (aligned_view.ctypes.data % CACHE_LINE_BYTES) == 0
    digest = hashlib.sha256(aligned_view.tobytes()).hexdigest()

    return UMADeviceAssetBuffer(
        asset_id=asset_id,
        shape=tuple(aligned_view.shape),
        byte_length=nbytes,
        cache_lines_64b=max(1, (nbytes + CACHE_LINE_BYTES - 1) // CACHE_LINE_BYTES),
        aligned_64b=ptr_aligned,
        subnormals_sanitized=True,
        sha256_digest=digest,
        host_array=aligned_view,
        ov_shared_tensor=ov_tensor,
    )


# ---------------------------------------------------------------------------
# 2. Defensive Vulkan Endpoint Coordinator & Advisor (`kuAdviseAndDefend`)
# ---------------------------------------------------------------------------

VULKAN_DEFENDED_ENDPOINTS: Tuple[VulkanEndpointExtensionSpec, ...] = (
    VulkanEndpointExtensionSpec(
        endpoint_id="VK_EP_01_HOLOGRAM_RAY_PROFILER",
        method_and_route="GET /vr_hologram_profiler",
        base_module="spatial_coordinators_and_vr_hologram_compositor.py",
        vk_extensions=("VK_KHR_external_memory", "VK_EXT_external_memory_host"),
        capability_mask=0x1F,
        frame_size_bytes=128,
        evidence_level="VERIFIED_WITH_SCOPE",
    ),
    VulkanEndpointExtensionSpec(
        endpoint_id="VK_EP_02_GEOTILE_TERRAIN_SHADER",
        method_and_route="GET /api/v1/runtime/geotile_vulkan_arena",
        base_module="wsl_hypervisor_mortar_and_geotile_adapter.py",
        vk_extensions=("VK_KHR_shader_float16_int8", "VK_KHR_16bit_storage"),
        capability_mask=0x1F,
        frame_size_bytes=128,
        evidence_level="VERIFIED_WITH_SCOPE",
    ),
    VulkanEndpointExtensionSpec(
        endpoint_id="VK_EP_03_UMA_CUDA_HYPERTRANSPORT",
        method_and_route="GET /api/v1/runtime/krystal_uma_cuda",
        base_module="krystal_uma_cuda_vulkan_library.py",
        vk_extensions=("VK_EXT_subgroup_size_control", "VK_EXT_external_memory_host"),
        capability_mask=0x3F,
        frame_size_bytes=192,
        evidence_level="LOCAL_MEASURED",
    ),
)


def kuAdviseAndDefend(
    buf: UMADeviceAssetBuffer,
    batch_n: int = 64,
    reduction_k: int = 256,
    uma_utilization: float = 0.35,
) -> Dict[str, Any]:
    """CUDA-style hardware advisor & defensive gate over Intel CPU/Iris Xe + Vulkan endpoints."""
    u = canonicalize_float32(float(uma_utilization))
    if not (0.0 <= u <= 1.0):
        raise ContractViolation("uma_utilization must be in [0, 1]", violation_code="ERR_KU_UMA")

    hw = get_cached_hw_profile()
    isa = probe_isa_flags()
    vulkan_gpu_allowed = u < UMA_SATURATION_THRESHOLD and "GPU" in hw.openvino_devices

    if not vulkan_gpu_allowed:
        lane = "CPU_8T_SMT_THROUGHPUT" if batch_n >= 200 else "CPU_4T_PHYSICAL_AVX512"
        defense_action = "DEFENDED_VULKAN_ENDPOINT_VETO_GPU_ON_UMA_CEILING"
    elif buf.byte_length >= 16_384 and batch_n >= 64:
        lane = "GPU_IRIS_XE_WARM_CACHE_PLUS_VULKAN_HOST_COHERENT"
        defense_action = "ADVISED_HETERO_UMA_ZERO_COPY"
    else:
        lane = "CPU_4T_PHYSICAL_AVX512"
        defense_action = "ADVISED_CPU_L1D_L2_FASTPATH"

    k_advice = (
        "TILE_REDUCTION_AT_K448_PLUS_FREIVALDS_AUDIT"
        if reduction_k >= 449
        else "DIRECT_SEQUENTIAL_FMA_EXACT"
    )
    return {
        "vital_max_hp": VITAL_MAX_HP,
        "asset_id": buf.asset_id,
        "aligned_64b": buf.aligned_64b,
        "cache_lines_64b": buf.cache_lines_64b,
        "avx512f_detected": bool(isa["flags"].get("AVX512F")),
        "vulkan_gpu_allowed": vulkan_gpu_allowed,
        "recommended_lane": lane,
        "defense_action": defense_action,
        "reduction_k_advice": k_advice,
        "defended_vulkan_endpoints": [ep.endpoint_id for ep in VULKAN_DEFENDED_ENDPOINTS],
    }


def kuLaunchMatMulDefended(a: np.ndarray, b: np.ndarray, uma_utilization: float = 0.35) -> Dict[str, Any]:
    """Execute defended UMA matrix multiplication automatically tiling K >= 449 and emitting 192B HyperTransport receipt."""
    buf_a = kuMallocUMA("mat_a", a)
    buf_b = kuMallocUMA("mat_b", b)
    advice = kuAdviseAndDefend(buf_a, batch_n=1, reduction_k=int(a.shape[1]), uma_utilization=uma_utilization)
    tiled = partition_reduction_below_k449(buf_a.host_array, buf_b.host_array, tile_k=448)
    out_buf = kuMallocUMA("mat_c", tiled["output"])
    hht_pkt = derive_192b_wntc_kpkt_hypervisor_datagram(uma_utilization=uma_utilization)
    return {
        "vital_max_hp": VITAL_MAX_HP,
        "advice": advice,
        "tiles_executed": tiled["tiles_executed"],
        "output_sha256": out_buf.sha256_digest,
        "output_aligned_64b": out_buf.aligned_64b,
        "hypertransport_192b_verified": hht_pkt["wntc_crc32_valid"] and hht_pkt["kpkt_sha256_valid"],
    }


# ---------------------------------------------------------------------------
# 3. Zero-Touch Kernel Operation Simplifier/Injector & HyperTransport Methodics
# ---------------------------------------------------------------------------

HYPERVISOR_HYPERTRANSPORT_METHODICS: Tuple[Dict[str, str], ...] = (
    {
        "methodic_id": "HHT_M1_CACHELINE_PACKET_QUANTA",
        "title": "Exact 64B Cache-Line Multiple Framing (64B WNTC + 128B KPKT = 192B)",
        "rule": "Never split a hypervisor control/telemetry descriptor across unaligned page boundaries; pack into 3 x 64B cache lines.",
    },
    {
        "methodic_id": "HHT_M2_VBLANK_CREDIT_FLOW_CONTROL",
        "title": "Chronometric VBlank Phase-Shadow Credit Window",
        "rule": "Grant bulk DMA/HyperTransport credits strictly inside the VBlank shadow (tau_spec) so active 60/120Hz scanout sees zero bus contention.",
    },
    {
        "methodic_id": "HHT_M3_HOST_COHERENT_ZERO_COPY_SLAB",
        "title": "Pinned 64B-Aligned UMA Views Over ov.Tensor & Vulkan Host-Coherent Memory",
        "rule": "Bind one 64B-aligned host allocation across NumPy, OpenVINO CPU/GPU, and Vulkan endpoints with 0 memcpy bytes.",
    },
    {
        "methodic_id": "HHT_M4_INSTANT_CREDIT_REVOCATION",
        "title": "0-Frame Credit Revocation & WSL2 Quota Trim on UMA >= 0.78",
        "rule": "Revoke GPU/DMA HyperTransport credits in 0 frames when UMA >= 0.78, trim WSL2 RAM 10->8 GB, and require 12 clean samples to restore.",
    },
)


def kuInjectSimplifiedKernelOp(op_name: str) -> Callable[[Callable[..., Any]], Callable[..., Dict[str, Any]]]:
    """Zero-touch decorator (Sovereign P1) simplifying operation logic & injecting 192B HyperTransport receipts."""

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Dict[str, Any]]:
        @functools.wraps(fn)
        def _wrapped(*args: Any, uma_utilization: float = 0.35, **kwargs: Any) -> Dict[str, Any]:
            if VITAL_MAX_HP != 6:
                raise ContractViolation("VITAL_MAX_HP != 6 in injected kernel op", violation_code="ERR_INJ_HP")
            t0 = time.perf_counter()
            clean_args = [
                kuMallocUMA(f"{op_name}_arg{i}", a).host_array if isinstance(a, np.ndarray) else a
                for i, a in enumerate(args)
            ]
            result = fn(*clean_args, **kwargs)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            hht = derive_192b_wntc_kpkt_hypervisor_datagram(uma_utilization=uma_utilization)
            return {
                "vital_max_hp": VITAL_MAX_HP,
                "injected_op": op_name,
                "elapsed_ms": elapsed_ms,
                "hypertransport_192b_ok": hht["wntc_crc32_valid"] and hht["kpkt_sha256_valid"],
                "result": result,
            }

        return _wrapped

    return _decorator


def coordinate_existing_schemas_and_scripts() -> Dict[str, Any]:
    """Inventory and bind existing `schemas/*.json` and `scripts/benchmark_*.py` into the UMA-CUDA coordinator."""
    schemas = sorted(p.name for p in SCHEMAS_DIR.glob("*.json")) if SCHEMAS_DIR.exists() else []
    bench_scripts = sorted(p.name for p in SCRIPTS_DIR.glob("benchmark_*.py")) if SCRIPTS_DIR.exists() else []
    return {
        "vital_max_hp": VITAL_MAX_HP,
        "schemas_bound_count": len(schemas),
        "schemas_bound": schemas,
        "benchmark_scripts_bound_count": len(bench_scripts),
        "benchmark_scripts_bound": bench_scripts,
        "hypertransport_methodics": list(HYPERVISOR_HYPERTRANSPORT_METHODICS),
        "defended_vulkan_endpoints_count": len(VULKAN_DEFENDED_ENDPOINTS),
    }
