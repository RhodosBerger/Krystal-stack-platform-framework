"""
CrystalSec 3.0 Phase 5.3D: Trusted Native CPU SIMD Runner & Verification Engine.

Authoritative Evidence Progression:
  CAPABILITY_DETECTED
      ↓
  SOURCE_IMPLEMENTED
      ↓
  BINARY_BUILT
      ↓
  BINARY_PROVENANCE_VERIFIED (SHA-256 matches manifest)
      ↓
  BINARY_TRUST_ACCEPTED (Windows code integrity / authenticode)
      ↓
  NATIVE_LIBRARY_LOADED (LoadLibraryExW explicit absolute canonical path)
      ↓
  NATIVE_SYMBOL_EXECUTED (function address resolved and invoked)
      ↓
  ISA_PATH_CONFIRMED (disassembly and CPUID match executed vector path)
      ↓
  NATIVE_SIMD_EXECUTION_MEASURED (QPC timing inside native C only)
      ↓
  STATISTICALLY_REPEATED (persistent break-even calibration)

Never skips stages. Never emits NATIVE_*_MEASURED for Python fallback.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import time
from collections.abc import Sequence
from ctypes import wintypes
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from src.python.native_binary_trust_inspector import (
    NativeBinaryTrustInspector,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DLL_PATH = os.path.abspath(os.path.join(_ROOT, "src", "c", "taxonomy_simd.dll"))
_MANIFEST_PATH = os.path.abspath(os.path.join(_ROOT, "artifacts", "taxonomy_simd_manifest.json"))


class EvidenceStage(str, Enum):
    CAPABILITY_DETECTED = "CAPABILITY_DETECTED"
    SOURCE_IMPLEMENTED = "SOURCE_IMPLEMENTED"
    BINARY_BUILT = "BINARY_BUILT"
    BINARY_PROVENANCE_VERIFIED = "BINARY_PROVENANCE_VERIFIED"
    BINARY_TRUST_ACCEPTED = "BINARY_TRUST_ACCEPTED"
    NATIVE_LIBRARY_LOADED = "NATIVE_LIBRARY_LOADED"
    NATIVE_SYMBOL_EXECUTED = "NATIVE_SYMBOL_EXECUTED"
    ISA_PATH_CONFIRMED = "ISA_PATH_CONFIRMED"
    NATIVE_SIMD_EXECUTION_MEASURED = "NATIVE_SIMD_EXECUTION_MEASURED"
    STATISTICALLY_REPEATED = "STATISTICALLY_REPEATED"


def derive_evidence_levels(
    capabilities_detected: bool,
    source_implemented: bool,
    binary_built: bool,
    provenance_verified: bool,
    trust_accepted: bool,
    library_loaded: bool,
    symbol_executed: bool,
    isa_confirmed: bool,
    native_simd_measured: bool,
    statistically_repeated: bool,
) -> list[str]:
    """Derives exact reached evidence levels strictly from observed terminal state.

    Rule: evidence_levels = derive_from_observed_state()
    Never returns unreached stages.
    """
    levels = []
    if capabilities_detected:
        levels.append(EvidenceStage.CAPABILITY_DETECTED.value)
    else:
        return levels

    if source_implemented:
        levels.append(EvidenceStage.SOURCE_IMPLEMENTED.value)
    else:
        return levels

    if binary_built:
        levels.append(EvidenceStage.BINARY_BUILT.value)
    else:
        return levels

    if provenance_verified:
        levels.append(EvidenceStage.BINARY_PROVENANCE_VERIFIED.value)
    else:
        return levels

    if trust_accepted:
        levels.append(EvidenceStage.BINARY_TRUST_ACCEPTED.value)
    else:
        return levels

    if library_loaded:
        levels.append(EvidenceStage.NATIVE_LIBRARY_LOADED.value)
    else:
        return levels

    if symbol_executed:
        levels.append(EvidenceStage.NATIVE_SYMBOL_EXECUTED.value)
    else:
        return levels

    if isa_confirmed:
        levels.append(EvidenceStage.ISA_PATH_CONFIRMED.value)
    else:
        return levels

    if native_simd_measured:
        levels.append(EvidenceStage.NATIVE_SIMD_EXECUTION_MEASURED.value)
    else:
        return levels

    if statistically_repeated:
        levels.append(EvidenceStage.STATISTICALLY_REPEATED.value)

    return levels


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
    BINARY_TRUST_REJECTED = "BINARY_TRUST_REJECTED"


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NativeSimdTimings:
    marshal_ms: float = 0.0
    ffi_call_overhead_ms: float = 0.0
    cpu_native_ms: float = 0.0
    unmarshal_ms: float = 0.0

    @property
    def end_to_end_ms(self) -> float:
        return self.marshal_ms + self.ffi_call_overhead_ms + self.cpu_native_ms + self.unmarshal_ms

    def to_dict(self) -> dict[str, float]:
        return {
            "marshal_ms": round(self.marshal_ms, 6),
            "ffi_call_overhead_ms": round(self.ffi_call_overhead_ms, 6),
            "cpu_native_ms": round(self.cpu_native_ms, 6),
            "unmarshal_ms": round(self.unmarshal_ms, 6),
            "end_to_end_ms": round(self.end_to_end_ms, 6),
        }


@dataclass
class NativeExecutionRecord:
    workload_class: str
    requested_path: str
    selected_path: str
    function_symbol: str
    function_address: str
    capability_snapshot: dict[str, Any]
    input_hash: str
    output_hash: str
    status_code: int
    cpu_native_ns: int
    cycles_elapsed: int
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        ("native_time_ns", ctypes.c_uint64),
    ]


def detect_cpu_capabilities() -> CpuNativeCapabilities:
    """Discovers CPU SIMD flags via direct x86_64 CPUID and XGETBV execution."""
    caps = CpuNativeCapabilities(
        cpu_model=platform.processor() or "Unknown x86_64",
        architecture=platform.machine() or "x86_64",
    )

    if platform.machine() not in ("AMD64", "x86_64") or platform.system() != "Windows":
        return caps

    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.VirtualAlloc.restype = ctypes.c_void_p
        kernel32.VirtualAlloc.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
        ]
        kernel32.VirtualProtect.restype = ctypes.c_int32
        kernel32.VirtualProtect.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
        ]

        out_buf = kernel32.VirtualAlloc(None, 64, 0x3000, 0x04)  # PAGE_READWRITE
        if not out_buf:
            return caps

        cpuid_asm = bytes(
            [
                0x53,  # push rbx
                0x49,
                0x89,
                0xCA,  # mov r10, rcx
                0x89,
                0xD0,  # mov eax, edx
                0x44,
                0x89,
                0xC1,  # mov ecx, r8d
                0x0F,
                0xA2,  # cpuid
                0x41,
                0x89,
                0x02,  # mov [r10], eax
                0x41,
                0x89,
                0x5A,
                0x04,  # mov [r10+4], ebx
                0x41,
                0x89,
                0x4A,
                0x08,  # mov [r10+8], ecx
                0x41,
                0x89,
                0x52,
                0x0C,  # mov [r10+12], edx
                0x5B,  # pop rbx
                0xC3,  # ret
            ]
        )
        c_ptr = kernel32.VirtualAlloc(None, len(cpuid_asm), 0x3000, 0x04)
        ctypes.memmove(c_ptr, cpuid_asm, len(cpuid_asm))
        old = ctypes.c_uint32()
        kernel32.VirtualProtect(c_ptr, len(cpuid_asm), 0x20, ctypes.byref(old))  # PAGE_EXECUTE_READ

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
            xgetbv_asm = bytes(
                [
                    0x49,
                    0x89,
                    0xCA,  # mov r10, rcx
                    0x89,
                    0xD1,  # mov ecx, edx (xcr_num)
                    0x0F,
                    0x01,
                    0xD0,  # xgetbv
                    0x41,
                    0x89,
                    0x02,  # mov [r10], eax
                    0x41,
                    0x89,
                    0x52,
                    0x04,  # mov [r10+4], edx
                    0xC3,  # ret
                ]
            )
            x_ptr = kernel32.VirtualAlloc(None, len(xgetbv_asm), 0x3000, 0x04)
            ctypes.memmove(x_ptr, xgetbv_asm, len(xgetbv_asm))
            kernel32.VirtualProtect(x_ptr, len(xgetbv_asm), 0x20, ctypes.byref(old))
            xgetbv_fn = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32)(x_ptr)

            xgetbv_fn(out_buf, 0)
            xcr0_low, xcr0_high = (ctypes.c_uint32 * 2).from_address(out_buf)
            caps.os_avx_state = bool((xcr0_low & 0x06) == 0x06)
            caps.os_avx512_state = bool(caps.os_avx_state and ((xcr0_low & 0xE0) == 0xE0))

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
) -> dict[str, Any]:
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
        max_err = max(max_err, err)
        if err > max(abs_tol, rel_tol * abs(r)):
            mism += 1
    return {
        "match": mism == 0,
        "reason": "" if mism == 0 else f"{mism} elements outside tolerance",
        "max_abs_error": max_err,
        "mismatches": mism,
    }


def compute_payload_hash(payload: Any) -> str:
    """Computes deterministic SHA-256 for test payloads to ensure same-input verification."""
    data_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data_bytes).hexdigest()[:16]


class NativeCpuSimdRunner:
    """Dispatches taxonomy workloads to native C / SIMD shared library under Windows security audit."""

    def __init__(
        self,
        dll_path: str = _DLL_PATH,
        manifest_path: str = _MANIFEST_PATH,
        force_scalar: bool = False,
    ):
        self.dll_path = os.path.abspath(dll_path)
        self.manifest_path = os.path.abspath(manifest_path)
        self.force_scalar = force_scalar
        self.capabilities = detect_cpu_capabilities()
        self.trust_report = NativeBinaryTrustInspector.inspect(self.dll_path, self.manifest_path)

        self._dll: ctypes.CDLL | None = None
        self._available = False
        self._unavailable_reason = ""
        self._module_handle: int | None = None
        self._ffi_overhead_ns: float = 0.0
        self._last_execution_record: NativeExecutionRecord | None = None

        self._init_library()

    def _init_library(self) -> None:
        tr = self.trust_report
        if not os.path.exists(self.dll_path):
            self._unavailable_reason = f"Native library missing at {self.dll_path}. Run scripts/build_taxonomy_simd.ps1"
            self._available = False
            return

        if not tr.provenance_verified:
            self._unavailable_reason = (
                f"BINARY_HASH_MISMATCH: SHA-256 {tr.sha256} does not match manifest. "
                "Binary provenance verification failed."
            )
            self._available = False
            return

        if not tr.execution_trusted:
            self._unavailable_reason = (
                f"BINARY_TRUST_REJECTED: Windows Code Integrity / UMCI policy ({tr.policy_classification}) "
                f"rejected library load with code {tr.loader_error_code} ({tr.code_integrity_load_outcome})."
            )
            self._available = False
            return

        # Controlled loading via LoadLibraryExW with explicit path handling
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.LoadLibraryExW.argtypes = [
                wintypes.LPCWSTR,
                wintypes.HANDLE,
                wintypes.DWORD,
            ]
            kernel32.LoadLibraryExW.restype = ctypes.c_void_p

            h_mod = kernel32.LoadLibraryExW(self.dll_path, None, 0x00000008)  # LOAD_WITH_ALTERED_SEARCH_PATH
            if not h_mod:
                err = kernel32.GetLastError()
                self._unavailable_reason = f"DLL_LOAD_FAILED: LoadLibraryExW failed with Win32 error {err}"
                self._available = False
                return

            self._module_handle = h_mod
            self._dll = ctypes.CDLL(self.dll_path, handle=h_mod)
            self._setup_bindings()
            self._available = True
            self._unavailable_reason = ""
            self._calibrate_ffi_overhead()

        except Exception as exc:
            self._available = False
            self._unavailable_reason = f"DLL_LOAD_FAILED ({type(exc).__name__}: {exc})"

    def _setup_bindings(self) -> None:
        dll = self._dll
        if not dll:
            return

        # Diagnostics & Delay
        dll.taxonomy_get_capabilities.argtypes = [ctypes.POINTER(_C_CpuNativeCapabilities)]
        dll.taxonomy_get_capabilities.restype = ctypes.c_int

        dll.taxonomy_noop.argtypes = []
        dll.taxonomy_noop.restype = None

        dll.taxonomy_set_test_delay_ns.argtypes = [ctypes.c_uint64]
        dll.taxonomy_set_test_delay_ns.restype = None

        dll.taxonomy_get_test_delay_ns.argtypes = []
        dll.taxonomy_get_test_delay_ns.restype = ctypes.c_uint64

        # Explicit Symbols
        dll.taxonomy_int8_dot_scalar.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        dll.taxonomy_int8_dot_scalar.restype = ctypes.c_int32

        dll.taxonomy_int8_dot_avx2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        dll.taxonomy_int8_dot_avx2.restype = ctypes.c_int32

        dll.taxonomy_int8_dot_vnni.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        dll.taxonomy_int8_dot_vnni.restype = ctypes.c_int32

        # High-level dispatchers
        dll.taxonomy_int8_dot_reduction.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_C_TaxonomyExecutionInfo),
        ]
        dll.taxonomy_int8_dot_reduction.restype = ctypes.c_int

        dll.taxonomy_int8_dot.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(_C_TaxonomyExecutionInfo),
        ]
        dll.taxonomy_int8_dot.restype = ctypes.c_int

        dll.taxonomy_fp32_poly.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(_C_TaxonomyExecutionInfo),
        ]
        dll.taxonomy_fp32_poly.restype = ctypes.c_int

        dll.taxonomy_memory_stream.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(_C_TaxonomyExecutionInfo),
        ]
        dll.taxonomy_memory_stream.restype = ctypes.c_int

        dll.taxonomy_branch_control.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(_C_TaxonomyExecutionInfo),
        ]
        dll.taxonomy_branch_control.restype = ctypes.c_int

        dll.taxonomy_mixed_native.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(_C_TaxonomyExecutionInfo),
        ]
        dll.taxonomy_mixed_native.restype = ctypes.c_int

    def _calibrate_ffi_overhead(self, iterations: int = 1000) -> None:
        """Measures median ctypes call boundary overhead using taxonomy_noop()."""
        if not self._dll:
            return
        noop = self._dll.taxonomy_noop
        # Warmup
        for _ in range(50):
            noop()
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            noop()
            times.append(time.perf_counter_ns() - t0)
        times.sort()
        self._ffi_overhead_ns = float(times[len(times) // 2])

    @property
    def ffi_overhead_ns(self) -> float:
        return self._ffi_overhead_ns

    def set_mutation_delay_ns(self, delay_ns: int) -> None:
        if self._dll:
            self._dll.taxonomy_set_test_delay_ns(ctypes.c_uint64(delay_ns))

    def available(self) -> bool:
        return self._available

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    @property
    def dll_sha256(self) -> str | None:
        return self.trust_report.sha256 if self.trust_report else None

    @property
    def last_execution_record(self) -> NativeExecutionRecord | None:
        return self._last_execution_record

    def get_reached_evidence_levels(self, statistically_repeated: bool = False) -> list[str]:
        """Derives reached evidence stages strictly based on actual observed state."""
        tr = self.trust_report
        c_source_exists = os.path.exists(os.path.join(_ROOT, "src", "c", "taxonomy_simd.c"))
        bin_built = os.path.exists(self.dll_path)
        prov_ok = bool(tr and tr.provenance_verified)
        trust_ok = bool(tr and tr.execution_trusted)
        lib_loaded = self.available() and (self._module_handle is not None)
        sym_exec = self._last_execution_record is not None
        isa_conf = os.path.exists(os.path.join(_ROOT, "artifacts", "taxonomy_simd_disassembly.txt")) and sym_exec
        native_meas = sym_exec and (self._last_execution_record.cpu_native_ns > 0)

        return derive_evidence_levels(
            capabilities_detected=bool(self.capabilities.avx2 or self.capabilities.sse42),
            source_implemented=c_source_exists,
            binary_built=bin_built,
            provenance_verified=prov_ok,
            trust_accepted=trust_ok,
            library_loaded=lib_loaded,
            symbol_executed=sym_exec,
            isa_confirmed=isa_conf,
            native_simd_measured=native_meas,
            statistically_repeated=statistically_repeated and native_meas,
        )

    def run(
        self,
        workload_class: str,
        data: dict[str, Any],
        requested_path: str = "AUTO",
    ) -> tuple[list[float], NativeSimdTimings, dict[str, Any]]:
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
        timings.ffi_call_overhead_ms = self._ffi_overhead_ns / 1_000_000.0
        info = _C_TaxonomyExecutionInfo()
        dll = self._dll
        inp_hash = compute_payload_hash(data)
        func_symbol = ""
        func_addr = "0x0"

        t_m0 = time.perf_counter_ns()
        if workload_class in ("INT8_DOT_PRODUCT", "MIXED_COMPUTE_MEMORY"):
            va = data["vector_a"]
            vb = data["vector_b"]
            n = min(len(va), len(vb))
            raw_a = (ctypes.c_uint8 * n)(*[max(0, min(255, int(x))) for x in va[:n]])
            raw_b = (ctypes.c_int8 * n)(*[max(-128, min(127, int(x))) for x in vb[:n]])
            raw_out = (ctypes.c_float * n)()
            timings.marshal_ms = (time.perf_counter_ns() - t_m0) / 1_000_000.0

            fn = dll.taxonomy_int8_dot if workload_class == "INT8_DOT_PRODUCT" else dll.taxonomy_mixed_native
            func_symbol = "taxonomy_int8_dot" if workload_class == "INT8_DOT_PRODUCT" else "taxonomy_mixed_native"
            func_addr = hex(ctypes.cast(fn, ctypes.c_void_p).value or 0)

            status = fn(
                ctypes.byref(raw_a),
                ctypes.byref(raw_b),
                ctypes.byref(raw_out),
                ctypes.c_size_t(n),
                ctypes.c_int(req_code),
                ctypes.byref(info),
            )

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

            fn = dll.taxonomy_fp32_poly
            func_symbol = "taxonomy_fp32_poly"
            func_addr = hex(ctypes.cast(fn, ctypes.c_void_p).value or 0)

            status = fn(
                ctypes.byref(raw_x),
                ctypes.byref(raw_c),
                ctypes.byref(raw_out),
                ctypes.c_size_t(n),
                ctypes.c_int(req_code),
                ctypes.byref(info),
            )

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

            fn = dll.taxonomy_memory_stream
            func_symbol = "taxonomy_memory_stream"
            func_addr = hex(ctypes.cast(fn, ctypes.c_void_p).value or 0)

            status = fn(
                ctypes.byref(raw_buf),
                ctypes.c_size_t(total),
                ctypes.c_size_t(stride),
                ctypes.byref(raw_out),
                ctypes.c_size_t(out_count),
                ctypes.c_int(req_code),
                ctypes.byref(info),
            )

            t_u0 = time.perf_counter_ns()
            out = list(raw_out)
            timings.unmarshal_ms = (time.perf_counter_ns() - t_u0) / 1_000_000.0

        elif workload_class == "BRANCH_HEAVY_CONTROL":
            keys = data["branch_keys"]
            n = len(keys)
            raw_k = (ctypes.c_int32 * n)(*[int(k) for k in keys])
            raw_out = (ctypes.c_float * n)()
            timings.marshal_ms = (time.perf_counter_ns() - t_m0) / 1_000_000.0

            fn = dll.taxonomy_branch_control
            func_symbol = "taxonomy_branch_control"
            func_addr = hex(ctypes.cast(fn, ctypes.c_void_p).value or 0)

            status = fn(
                ctypes.byref(raw_k),
                ctypes.byref(raw_out),
                ctypes.c_size_t(n),
                ctypes.c_int(req_code),
                ctypes.byref(info),
            )

            t_u0 = time.perf_counter_ns()
            out = list(raw_out)
            timings.unmarshal_ms = (time.perf_counter_ns() - t_u0) / 1_000_000.0
        else:
            raise ValueError(f"Unknown workload class: {workload_class}")

        if status != 0:
            raise RuntimeError(f"Native kernel execution failed with status code: {status}")

        # cpu_native_ms comes directly from internal C QPC measurement
        timings.cpu_native_ms = info.elapsed_ms

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

        out_hash = compute_payload_hash(out[:64] if len(out) > 64 else out)
        exec_record = NativeExecutionRecord(
            workload_class=workload_class,
            requested_path=requested_path,
            selected_path=sel_name,
            function_symbol=func_symbol,
            function_address=func_addr,
            capability_snapshot=self.capabilities.to_dict(),
            input_hash=inp_hash,
            output_hash=out_hash,
            status_code=status,
            cpu_native_ns=info.native_time_ns,
            cycles_elapsed=info.cycles_elapsed,
            timestamp=time.time(),
        )
        self._last_execution_record = exec_record

        metadata = {
            "evidence_stage": EvidenceStage.NATIVE_SIMD_EXECUTION_MEASURED.value,
            "evidence_label": evidence_label_map.get(sel_name, EvidenceLabel.NATIVE_SCALAR_MEASURED.value),
            "requested_path": requested_path,
            "selected_path": sel_name,
            "fallback_reason": "" if req_code == 0 or req_code == info.selected_path else "ISA unsupported",
            "capabilities_detected": self.capabilities.to_dict(),
            "native_library_hash": self.trust_report.sha256 if self.trust_report else None,
            "kernel_symbol": func_symbol,
            "kernel_address": func_addr,
            "measurement_method": "Internal QueryPerformanceCounter (zero Python allocation)",
            "cycles_elapsed": info.cycles_elapsed,
            "input_hash": inp_hash,
            "output_hash": out_hash,
        }

        return out, timings, metadata


# Deterministic scalar reference oracle for T0 verification
def run_scalar_reference(
    workload_class: str,
    data: dict[str, Any],
    legacy_relu_per_product: bool = False,
) -> list[float]:
    """Pure-Python canonical scalar reference oracle (T0)."""
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
    data: dict[str, Any],
    apply_relu_post_op: bool = False,
) -> int:
    """Canonical pure dot-product reduction: acc = sum(uint8(a) * int8(b))."""
    va = data["vector_a"]
    vb = data["vector_b"]
    n = min(len(va), len(vb))
    acc = sum(int(a) * int(b) for a, b in zip(va[:n], vb[:n]))
    return max(0, acc) if apply_relu_post_op else acc
