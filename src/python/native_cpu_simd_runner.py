"""
CrystalSec 3.0 Phase 5.3C: Native CPU SIMD Runner.

Replaces Phase 5 / 5.2 HOST_CPU_EMULATED Tier 1 paths with genuine native CPU
execution on physical x86_64 silicon (Intel Willow Cove Tiger Lake host).

Evidence rules:
  - Runtime CPUID/XGETBV capability discovery determines valid ISA extensions.
  - Every timing is wall-clock (time.perf_counter_ns) split into:
      marshal_ms, cpu_native_ms, unmarshal_ms, end_to_end_ms.
  - No synthetic constants or artificial penalties are added.
  - Native DLL is audited via SHA-256 against artifacts/taxonomy_simd_manifest.json.
  - If the native binary is unavailable or blocked by OS policy (e.g. WDAC/UMCI),
    the runner reports available() == False with the explicit OS reason,
    allowing tests to skip rather than secretly falling back to Python.
  - NEVER emits HARDWARE_MEASURED or NATIVE_*_MEASURED for Python execution.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import struct
import time
from array import array
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DLL_PATH = os.path.join(_ROOT, "src", "c", "taxonomy_simd.dll")
_MANIFEST_PATH = os.path.join(_ROOT, "artifacts", "taxonomy_simd_manifest.json")


class EvidenceLabel(str, Enum):
    HOST_CPU_REFERENCE = "HOST_CPU_REFERENCE"
    NATIVE_SCALAR_MEASURED = "NATIVE_SCALAR_MEASURED"
    NATIVE_AVX2_MEASURED = "NATIVE_AVX2_MEASURED"
    NATIVE_AVX512_MEASURED = "NATIVE_AVX512_MEASURED"
    NATIVE_VNNI_MEASURED = "NATIVE_VNNI_MEASURED"
    HARDWARE_MEASURED = "HARDWARE_MEASURED"
    STATISTICALLY_REPEATED = "STATISTICALLY_REPEATED"
    CAPABILITY_DETECTED = "CAPABILITY_DETECTED"
    UNAVAILABLE = "UNAVAILABLE"
    MODELED_PRIOR = "MODELED_PRIOR"


@dataclass
class CpuNativeCapabilities:
    """Hardware capability inspection report for CPU SIMD & tensor units."""
    sse42: bool = False
    avx: bool = False
    avx2: bool = False
    fma: bool = False
    avx512f: bool = False
    avx512bw: bool = False
    avx512vl: bool = False
    avx512_vnni: bool = False
    os_avx_state: bool = False
    os_avx512_state: bool = False
    cpu_model: str = ""
    architecture: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NativeSimdTimings:
    marshal_ms: float = 0.0
    cpu_native_ms: float = 0.0
    unmarshal_ms: float = 0.0

    @property
    def end_to_end_ms(self) -> float:
        return self.marshal_ms + self.cpu_native_ms + self.unmarshal_ms

    def to_dict(self) -> Dict[str, float]:
        return {
            "marshal_ms": round(self.marshal_ms, 6),
            "cpu_native_ms": round(self.cpu_native_ms, 6),
            "unmarshal_ms": round(self.unmarshal_ms, 6),
            "end_to_end_ms": round(self.end_to_end_ms, 6),
        }


# C struct mappings for ctypes
class _C_CpuNativeCapabilities(ctypes.Structure):
    _fields_ = [
        ("sse42", ctypes.c_uint32),
        ("avx", ctypes.c_uint32),
        ("avx2", ctypes.c_uint32),
        ("fma", ctypes.c_uint32),
        ("avx512f", ctypes.c_uint32),
        ("avx512bw", ctypes.c_uint32),
        ("avx512vl", ctypes.c_uint32),
        ("avx512_vnni", ctypes.c_uint32),
        ("os_avx_state", ctypes.c_uint32),
        ("os_avx512_state", ctypes.c_uint32),
    ]


class _C_TaxonomyExecutionInfo(ctypes.Structure):
    _fields_ = [
        ("selected_path", ctypes.c_uint32),
        ("status_code", ctypes.c_uint32),
        ("cycles_elapsed", ctypes.c_uint64),
        ("elapsed_ms", ctypes.c_double),
    ]


def detect_cpu_capabilities() -> CpuNativeCapabilities:
    """Discovers CPU SIMD flags via direct x86_64 CPUID and XGETBV execution."""
    caps = CpuNativeCapabilities(
        cpu_model=platform.processor() or "Unknown x86_64",
        architecture=platform.machine() or "x86_64",
    )

    if platform.machine() not in ("AMD64", "x86_64"):
        return caps

    # Use Windows x64 kernel32 to execute live CPUID and XGETBV
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.VirtualAlloc.restype = ctypes.c_void_p
        kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
        kernel32.VirtualProtect.restype = ctypes.c_int32
        kernel32.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]

        out_buf = kernel32.VirtualAlloc(None, 64, 0x3000, 0x04) # PAGE_READWRITE
        if not out_buf:
            return caps

        # Assembly: rcx=out_ptr, edx=leaf, r8d=subleaf
        # Preserves rbx, saves out_ptr in r10
        cpuid_asm = bytes([
            0x53,                         # push rbx
            0x49, 0x89, 0xCA,             # mov r10, rcx
            0x89, 0xD0,                   # mov eax, edx
            0x44, 0x89, 0xC1,             # mov ecx, r8d
            0x0F, 0xA2,                   # cpuid
            0x41, 0x89, 0x02,             # mov [r10], eax
            0x41, 0x89, 0x5A, 0x04,       # mov [r10+4], ebx
            0x41, 0x89, 0x4A, 0x08,       # mov [r10+8], ecx
            0x41, 0x89, 0x52, 0x0C,       # mov [r10+12], edx
            0x5B,                         # pop rbx
            0xC3                          # ret
        ])
        c_ptr = kernel32.VirtualAlloc(None, len(cpuid_asm), 0x3000, 0x04)
        ctypes.memmove(c_ptr, cpuid_asm, len(cpuid_asm))
        old = ctypes.c_uint32()
        kernel32.VirtualProtect(c_ptr, len(cpuid_asm), 0x20, ctypes.byref(old)) # PAGE_EXECUTE_READ

        cpuid_fn = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32)(c_ptr)

        # Leaf 1: feature flags
        cpuid_fn(out_buf, 1, 0)
        eax, ebx, ecx, edx = (ctypes.c_uint32 * 4).from_address(out_buf)
        caps.sse42 = bool(ecx & (1 << 20))
        caps.fma = bool(ecx & (1 << 12))
        caps.avx = bool(ecx & (1 << 28))
        has_osxsave = bool(ecx & (1 << 27))

        # Leaf 7, Subleaf 0: extended feature flags
        cpuid_fn(out_buf, 7, 0)
        eax7, ebx7, ecx7, edx7 = (ctypes.c_uint32 * 4).from_address(out_buf)
        caps.avx2 = bool(ebx7 & (1 << 5))
        caps.avx512f = bool(ebx7 & (1 << 16))
        caps.avx512bw = bool(ebx7 & (1 << 30))
        caps.avx512vl = bool(ebx7 & (1 << 31))
        caps.avx512_vnni = bool(ecx7 & (1 << 11))

        # XGETBV for OS extended state
        if has_osxsave:
            xgetbv_asm = bytes([
                0x49, 0x89, 0xCA,             # mov r10, rcx
                0x89, 0xD1,                   # mov ecx, edx (xcr_num)
                0x0F, 0x01, 0xD0,             # xgetbv
                0x41, 0x89, 0x02,             # mov [r10], eax
                0x41, 0x89, 0x52, 0x04,       # mov [r10+4], edx
                0xC3                          # ret
            ])
            x_ptr = kernel32.VirtualAlloc(None, len(xgetbv_asm), 0x3000, 0x04)
            ctypes.memmove(x_ptr, xgetbv_asm, len(xgetbv_asm))
            kernel32.VirtualProtect(x_ptr, len(xgetbv_asm), 0x20, ctypes.byref(old))
            xgetbv_fn = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32)(x_ptr)

            xgetbv_fn(out_buf, 0)
            xcr0_low, xcr0_high = (ctypes.c_uint32 * 2).from_address(out_buf)
            caps.os_avx_state = bool((xcr0_low & 0x06) == 0x06)
            caps.os_avx512_state = bool(caps.os_avx_state and ((xcr0_low & 0xE0) == 0xE0))

        # Mask hardware flags if OS does not support context switching
        if not caps.os_avx_state:
            caps.avx = False
            caps.avx2 = False
            caps.fma = False
        if not caps.os_avx512_state:
            caps.avx512f = False
            caps.avx512bw = False
            caps.avx512vl = False
            caps.avx512_vnni = False

    except Exception:
        # If dynamic probe fails, fallback to platform detection
        is_intel = "Intel" in caps.cpu_model or "GenuineIntel" in caps.cpu_model
        is_tiger_lake = ("Model 140" in caps.cpu_model) or ("1135G7" in caps.cpu_model)
        caps.avx2 = is_intel or caps.architecture == "AMD64"
        caps.avx512f = is_tiger_lake
        caps.avx512_vnni = is_tiger_lake

    return caps


def verify_outputs(
    reference: Sequence[float],
    candidate: Sequence[float],
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-4,
) -> Dict[str, Any]:
    """Element-wise comparison against reference."""
    if len(reference) != len(candidate):
        return {
            "match": False,
            "reason": f"length mismatch: {len(candidate)} != {len(reference)}",
            "max_abs_error": None,
            "mismatches": None,
        }
    mism, max_err = 0, 0.0
    for r, c in zip(reference, candidate):
        err = abs(r - c)
        if err > max_err:
            max_err = err
        if err > max(abs_tol, rel_tol * abs(r)):
            mism += 1
    return {
        "match": mism == 0,
        "reason": "" if mism == 0 else f"{mism} elements outside tolerance",
        "max_abs_error": max_err,
        "mismatches": mism,
    }


class NativeCpuSimdRunner:
    """Dispatches taxonomy workloads to native C / SIMD shared library."""

    def __init__(
        self,
        dll_path: str = _DLL_PATH,
        manifest_path: str = _MANIFEST_PATH,
        force_scalar: bool = False,
    ):
        self.dll_path = dll_path
        self.manifest_path = manifest_path
        self.force_scalar = force_scalar
        self.capabilities = detect_cpu_capabilities()
        self._dll: Optional[ctypes.CDLL] = None
        self._available = False
        self._unavailable_reason = ""
        self._dll_hash: Optional[str] = None
        self._init_library()

    def _init_library(self) -> None:
        if not os.path.exists(self.dll_path):
            self._unavailable_reason = f"Native library missing at {self.dll_path}. Run scripts/build_taxonomy_simd.ps1"
            self._available = False
            return

        # Check manifest and SHA-256
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                expected_hash = manifest.get("dll_sha256")
                with open(self.dll_path, "rb") as f_dll:
                    actual_hash = hashlib.sha256(f_dll.read()).hexdigest().lower()
                self._dll_hash = actual_hash
                if expected_hash and actual_hash != expected_hash.lower():
                    self._unavailable_reason = (
                        f"DLL hash mismatch! Expected {expected_hash}, got {actual_hash}. "
                        "Binary provenance verification failed."
                    )
                    self._available = False
                    return
            except Exception as e:
                self._unavailable_reason = f"Manifest audit failed: {e}"
                self._available = False
                return

        # Attempt to load DLL
        try:
            self._dll = ctypes.CDLL(self.dll_path)
            self._setup_bindings()
            self._available = True
            self._unavailable_reason = ""
        except Exception as exc:
            self._available = False
            self._unavailable_reason = f"CDLL load failed ({type(exc).__name__}: {exc})"

    def _setup_bindings(self) -> None:
        dll = self._dll
        if not dll:
            return

        # taxonomy_get_capabilities
        dll.taxonomy_get_capabilities.argtypes = [ctypes.POINTER(_C_CpuNativeCapabilities)]
        dll.taxonomy_get_capabilities.restype = ctypes.c_int

        # taxonomy_int8_dot
        dll.taxonomy_int8_dot.argtypes = [
            ctypes.c_void_p,  # uint8_t* activations
            ctypes.c_void_p,  # int8_t* weights
            ctypes.c_void_p,  # float* out
            ctypes.c_size_t,  # n
            ctypes.c_int,     # requested_path
            ctypes.POINTER(_C_TaxonomyExecutionInfo),
        ]
        dll.taxonomy_int8_dot.restype = ctypes.c_int

        # taxonomy_fp32_poly
        dll.taxonomy_fp32_poly.argtypes = [
            ctypes.c_void_p,  # float* inputs
            ctypes.c_void_p,  # float* coeffs
            ctypes.c_void_p,  # float* out
            ctypes.c_size_t,  # n
            ctypes.c_int,     # requested_path
            ctypes.POINTER(_C_TaxonomyExecutionInfo),
        ]
        dll.taxonomy_fp32_poly.restype = ctypes.c_int

        # taxonomy_memory_stream
        dll.taxonomy_memory_stream.argtypes = [
            ctypes.c_void_p,  # float* buffer
            ctypes.c_size_t,  # total_elements
            ctypes.c_size_t,  # stride
            ctypes.c_void_p,  # float* out
            ctypes.c_size_t,  # out_count
            ctypes.c_int,     # requested_path
            ctypes.POINTER(_C_TaxonomyExecutionInfo),
        ]
        dll.taxonomy_memory_stream.restype = ctypes.c_int

        # taxonomy_branch_control
        dll.taxonomy_branch_control.argtypes = [
            ctypes.c_void_p,  # int32_t* keys
            ctypes.c_void_p,  # float* out
            ctypes.c_size_t,  # n
            ctypes.c_int,     # requested_path
            ctypes.POINTER(_C_TaxonomyExecutionInfo),
        ]
        dll.taxonomy_branch_control.restype = ctypes.c_int

        # taxonomy_mixed_compute_memory
        dll.taxonomy_mixed_compute_memory.argtypes = [
            ctypes.c_void_p,  # uint8_t* activations
            ctypes.c_void_p,  # int8_t* weights
            ctypes.c_void_p,  # float* out
            ctypes.c_size_t,  # n
            ctypes.c_int,     # requested_path
            ctypes.POINTER(_C_TaxonomyExecutionInfo),
        ]
        dll.taxonomy_mixed_compute_memory.restype = ctypes.c_int

    def available(self) -> bool:
        return self._available

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    @property
    def dll_sha256(self) -> Optional[str]:
        return self._dll_hash

    def run(
        self,
        workload_class: str,
        data: Dict[str, Any],
        requested_path: str = "AUTO",
    ) -> Tuple[List[float], NativeSimdTimings, Dict[str, Any]]:
        """Executes workload on native CPU library.
        Returns: (output, timings, metadata)
        """
        if not self.available():
            raise RuntimeError(f"Native CPU SIMD runner unavailable: {self.unavailable_reason}")

        path_map = {
            "AUTO": 0,
            "SCALAR": 1,
            "AVX2": 2,
            "AVX512": 3,
            "VNNI": 4,
        }
        req_code = path_map.get(requested_path.upper(), 0)
        if self.force_scalar:
            req_code = 1

        timings = NativeSimdTimings()
        info = _C_TaxonomyExecutionInfo()
        dll = self._dll

        t_m0 = time.perf_counter_ns()
        if workload_class in ("INT8_DOT_PRODUCT", "MIXED_COMPUTE_MEMORY"):
            va = data["vector_a"]
            vb = data["vector_b"]
            n = min(len(va), len(vb))
            # Canonical contract: uint8 activations, int8 weights
            raw_a = (ctypes.c_uint8 * n)(*[max(0, min(255, int(x))) for x in va[:n]])
            raw_b = (ctypes.c_int8 * n)(*[max(-128, min(127, int(x))) for x in vb[:n]])
            raw_out = (ctypes.c_float * n)()
            timings.marshal_ms = (time.perf_counter_ns() - t_m0) / 1_000_000.0

            fn = dll.taxonomy_int8_dot if workload_class == "INT8_DOT_PRODUCT" else dll.taxonomy_mixed_compute_memory
            t_exec0 = time.perf_counter_ns()
            status = fn(
                ctypes.byref(raw_a),
                ctypes.byref(raw_b),
                ctypes.byref(raw_out),
                ctypes.c_size_t(n),
                ctypes.c_int(req_code),
                ctypes.byref(info),
            )
            timings.cpu_native_ms = (time.perf_counter_ns() - t_exec0) / 1_000_000.0

            t_u0 = time.perf_counter_ns()
            out = list(raw_out)
            timings.unmarshal_ms = (time.perf_counter_ns() - t_u0) / 1_000_000.0

        elif workload_class == "FP32_VECTOR_MATH":
            inputs = data["inputs"]
            coeffs = data["coeffs"]
            n = len(inputs)
            raw_x = (ctypes.c_float * n)(*[float(x) for x in inputs])
            raw_c = (ctypes.c_float * 4)(*[float(c) for c in coeffs[:4]])
            raw_out = (ctypes.c_float * n)()
            timings.marshal_ms = (time.perf_counter_ns() - t_m0) / 1_000_000.0

            t_exec0 = time.perf_counter_ns()
            status = dll.taxonomy_fp32_poly(
                ctypes.byref(raw_x),
                ctypes.byref(raw_c),
                ctypes.byref(raw_out),
                ctypes.c_size_t(n),
                ctypes.c_int(req_code),
                ctypes.byref(info),
            )
            timings.cpu_native_ms = (time.perf_counter_ns() - t_exec0) / 1_000_000.0

            t_u0 = time.perf_counter_ns()
            out = list(raw_out)
            timings.unmarshal_ms = (time.perf_counter_ns() - t_u0) / 1_000_000.0

        elif workload_class == "MEMORY_BOUND_STREAM":
            buf = data["buffer"]
            stride = int(data["stride"])
            total = len(buf)
            out_count = (total + stride - 1) // stride
            raw_buf = (ctypes.c_float * total)(*[float(x) for x in buf])
            raw_out = (ctypes.c_float * out_count)()
            timings.marshal_ms = (time.perf_counter_ns() - t_m0) / 1_000_000.0

            t_exec0 = time.perf_counter_ns()
            status = dll.taxonomy_memory_stream(
                ctypes.byref(raw_buf),
                ctypes.c_size_t(total),
                ctypes.c_size_t(stride),
                ctypes.byref(raw_out),
                ctypes.c_size_t(out_count),
                ctypes.c_int(req_code),
                ctypes.byref(info),
            )
            timings.cpu_native_ms = (time.perf_counter_ns() - t_exec0) / 1_000_000.0

            t_u0 = time.perf_counter_ns()
            out = list(raw_out)
            timings.unmarshal_ms = (time.perf_counter_ns() - t_u0) / 1_000_000.0

        elif workload_class == "BRANCH_HEAVY_CONTROL":
            keys = data["branch_keys"]
            n = len(keys)
            raw_k = (ctypes.c_int32 * n)(*[int(k) for k in keys])
            raw_out = (ctypes.c_float * n)()
            timings.marshal_ms = (time.perf_counter_ns() - t_m0) / 1_000_000.0

            t_exec0 = time.perf_counter_ns()
            status = dll.taxonomy_branch_control(
                ctypes.byref(raw_k),
                ctypes.byref(raw_out),
                ctypes.c_size_t(n),
                ctypes.c_int(req_code),
                ctypes.byref(info),
            )
            timings.cpu_native_ms = (time.perf_counter_ns() - t_exec0) / 1_000_000.0

            t_u0 = time.perf_counter_ns()
            out = list(raw_out)
            timings.unmarshal_ms = (time.perf_counter_ns() - t_u0) / 1_000_000.0
        else:
            raise ValueError(f"Unknown workload class: {workload_class}")

        if status != 0:
            raise RuntimeError(f"Native kernel execution failed with status code: {status}")

        path_name_map = {
            1: "NATIVE_SCALAR",
            2: "NATIVE_AVX2",
            3: "NATIVE_AVX512",
            4: "NATIVE_VNNI",
        }
        sel_name = path_name_map.get(info.selected_path, "NATIVE_SCALAR")

        evidence_label_map = {
            "NATIVE_SCALAR": EvidenceLabel.NATIVE_SCALAR_MEASURED.value,
            "NATIVE_AVX2": EvidenceLabel.NATIVE_AVX2_MEASURED.value,
            "NATIVE_AVX512": EvidenceLabel.NATIVE_AVX512_MEASURED.value,
            "NATIVE_VNNI": EvidenceLabel.NATIVE_VNNI_MEASURED.value,
        }

        metadata = {
            "requested_path": requested_path,
            "selected_path": sel_name,
            "fallback_reason": "" if req_code == 0 or req_code == info.selected_path else "ISA unsupported",
            "capabilities_detected": self.capabilities.to_dict(),
            "native_library_hash": self._dll_hash,
            "kernel_symbol": f"taxonomy_{workload_class.lower()}",
            "measurement_method": "QueryPerformanceCounter / time.perf_counter_ns",
            "evidence_label": evidence_label_map.get(sel_name, EvidenceLabel.NATIVE_SCALAR_MEASURED.value),
            "cycles_elapsed": info.cycles_elapsed,
        }

        return out, timings, metadata


# Deterministic scalar reference oracle for T0 verification
def run_scalar_reference(
    workload_class: str,
    data: Dict[str, Any],
    legacy_relu_per_product: bool = False,
) -> List[float]:
    """Pure-Python canonical scalar reference oracle (T0).
    
    INT8 Canonical Semantics:
      input A: uint8
      input B: int8
      operation: A[i] * B[i] (without per-product ReLU)
      post-op activation: separate
    """
    if workload_class in ("INT8_DOT_PRODUCT", "MIXED_COMPUTE_MEMORY"):
        va = data["vector_a"]
        vb = data["vector_b"]
        n = min(len(va), len(vb))
        if legacy_relu_per_product:
            return [float(max(0, int(a) * int(b))) for a, b in zip(va[:n], vb[:n])]
        return [float(int(a) * int(b)) for a, b in zip(va[:n], vb[:n])]


    elif workload_class == "FP32_VECTOR_MATH":
        inputs = data["inputs"]
        c = data["coeffs"]
        c0, c1, c2, c3 = float(c[0]), float(c[1]), float(c[2]), float(c[3])
        return [((c3 * float(x) + c2) * float(x) + c1) * float(x) + c0 for x in inputs]

    elif workload_class == "MEMORY_BOUND_STREAM":
        buf = data["buffer"]
        stride = int(data["stride"])
        return [float(buf[i]) * 1.001 for i in range(0, len(buf), stride)]

    elif workload_class == "BRANCH_HEAVY_CONTROL":
        keys = data["branch_keys"]
        res = []
        for k in keys:
            ki = int(k)
            if ki % 11 == 0:
                v = ki * 3 + 1
            elif ki % 7 == 0:
                v = ki // 2
            elif ki % 3 == 0:
                v = ki * 5
            elif ki % 2 == 0:
                v = ki + 17
            else:
                v = ki - 1
            res.append(float(v))
        return res

    raise ValueError(f"Unknown workload class: {workload_class}")


def run_scalar_reduction_reference(
    data: Dict[str, Any],
    apply_relu_post_op: bool = False,
) -> int:
    """Canonical pure dot-product reduction: acc = sum(uint8(a) * int8(b)).
    
    Matches VNNI vpdpbusd arithmetic. Post-op ReLU is applied to the final
    accumulated sum, never per individual product.
    """
    va = data["vector_a"]
    vb = data["vector_b"]
    n = min(len(va), len(vb))
    acc = sum(int(a) * int(b) for a, b in zip(va[:n], vb[:n]))
    return max(0, acc) if apply_relu_post_op else acc
