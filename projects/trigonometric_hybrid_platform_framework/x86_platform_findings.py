"""Empirical x86-64 platform computation & microarchitectural findings suite.

Executes 5 calibrated discriminating tests on the host x86-64 processor:
  X01: Cache-hierarchy bandwidth cliff (L1d 32 KB -> L2 512 KB -> L3 4 MB -> DRAM 32 MB)
  X02: Willow Cove physical cores (1T/2T/4T) vs SMT hyper-threading (8T) under AVX-512 conv
  X03: IEEE-754 subnormal (denormal ~1e-42) microcode assist vs oneDNN MXCSR FTZ/DAZ behavior
  X04: 64-byte cache-line stride efficiency & OpenVINO shared_memory alignment contract
  X05: Little-endian 128 B binary struct+SHA256 frame vs JSON serialization throughput

Every measurement reports sample size n, median, p95, p99, and stdev (LOCAL_MEASURED).
No hardware counter (RAPL/PMC) is fabricated; unmeasured quantities are labeled UNKNOWN.
"""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from krystal_kernel.constants import CACHE_LINE_BYTES, FRAME_SIZE_BYTES, VITAL_MAX_HP, ContractViolation

from .combat_turn_planner import DEFAULT_CACHE_DIR, get_cached_hw_profile, probe_isa_flags
from .fact_synthesis_engine import REPORTS
from .intel_accelerator_and_power_thread_governor import get_shared_ov_core
from .virtual_convolution_pipeline import _weights, build_openvino_model


def _stats(samples_ms: List[float]) -> Dict[str, Any]:
    s = sorted(samples_ms)
    n = len(s)
    p95 = s[min(n - 1, int(0.95 * n))]
    p99 = s[min(n - 1, int(0.99 * n))]
    med = statistics.median(s)
    sd = statistics.stdev(s) if n > 1 else 0.0
    return {
        "n": n,
        "median_ms": round(med, 4),
        "p95_ms": round(p95, 4),
        "p99_ms": round(p99, 4),
        "stdev_ms": round(sd, 4),
        "jitter_p95_over_median": round(p95 / max(med, 1e-9), 3),
    }


def measure_x01_cache_bandwidth_hierarchy(runs: int = 30) -> Dict[str, Any]:
    """X01: Measure effective FP32 SIMD dot-product bandwidth across L1d, L2, L3, and UMA DRAM footprints."""
    # Single compiled BLAS sdot reads 2 contiguous float32 vectors (8 bytes/elem) inside native SIMD.
    tiers = [
        ("L1d_32KB", 4_096, 256),       # 2 * 4096 * 4B = 32 KB (< 48 KB Willow Cove L1d)
        ("L2_512KB", 65_536, 32),       # 2 * 65536 * 4B = 512 KB (< 1.25 MB L2)
        ("L3_4MB", 524_288, 6),         # 2 * 524288 * 4B = 4.0 MB (< 8 MB L3)
        ("DRAM_32MB", 4_194_304, 2),    # 2 * 4194304 * 4B = 32 MB (> 8 MB L3 -> UMA DRAM)
    ]
    rng = np.random.default_rng(6)
    results: Dict[str, Any] = {}

    for label, elems, inner_loops in tiers:
        a = rng.standard_normal(elems).astype(np.float32)
        b = rng.standard_normal(elems).astype(np.float32)
        acc_guard = 0.0

        for _ in range(3):
            acc_guard += float(np.dot(a, b))

        samples_ms: List[float] = []
        gbps_samples: List[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            for _ in range(inner_loops):
                acc_guard += float(np.dot(a, b))
            dt_s = max(time.perf_counter() - t0, 1e-9)
            logical_bytes = elems * 8 * inner_loops
            samples_ms.append(dt_s * 1000.0)
            gbps_samples.append((logical_bytes / (1024 ** 3)) / dt_s)

        results[label] = {
            "working_set_kb": round((elems * 8) / 1024.0, 1),
            "logical_gbps_median": round(statistics.median(gbps_samples), 2),
            "logical_gbps_p95": round(sorted(gbps_samples)[min(len(gbps_samples) - 1, int(0.95 * len(gbps_samples)))], 2),
            "timing": _stats(samples_ms),
            "checksum_guard": round(acc_guard, 3),
            "bytes_formula_note": "8 B/elem across two contiguous float32 vectors in native SIMD sdot",
        }

    l1_bw = results["L1d_32KB"]["logical_gbps_median"]
    dram_bw = results["DRAM_32MB"]["logical_gbps_median"]
    return {
        "finding_id": "X01_CACHE_HIERARCHY_CLIFF",
        "tiers": results,
        "l1d_over_dram_bandwidth_ratio": round(l1_bw / max(dram_bw, 1e-6), 2),
        "evidence": "LOCAL_MEASURED",
    }


def measure_x02_physical_vs_smt_avx512_scaling(runs: int = 30) -> Dict[str, Any]:
    """X02: Compare 1T, 2T, 4T (physical cores) vs 8T (SMT siblings) on OpenVINO AVX-512 conv."""
    hw = get_cached_hw_profile()
    _, core, _ = get_shared_ov_core(DEFAULT_CACHE_DIR)

    shape = (1, 16, 96, 96)
    rng = np.random.default_rng(6)
    x = rng.standard_normal(shape).astype(np.float32)
    w1 = _weights(16, 32, 6)
    w2 = _weights(32, 32, 7)
    model = build_openvino_model(shape, w1, w2)

    thread_counts = sorted({1, 2, max(1, hw.physical_cores), max(1, hw.logical_cores)})
    by_threads: Dict[str, Any] = {}
    kernel_seen = "unknown"

    for t in thread_counts:
        compiled = core.compile_model(model, "CPU", {"INFERENCE_PRECISION_HINT": "f32", "INFERENCE_NUM_THREADS": t})
        for op in compiled.get_runtime_model().get_ordered_ops():
            rt = op.get_rt_info()
            if "primitiveType" in rt and "conv" in str(rt["primitiveType"].astype(str)).lower():
                kernel_seen = str(rt["primitiveType"].astype(str))
                break
        req = compiled.create_infer_request()
        for _ in range(3):
            req.infer({0: x})
        samples: List[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            req.infer({0: x})
            samples.append((time.perf_counter() - t0) * 1000.0)
        by_threads[f"{t}T"] = _stats(samples)

    t1 = by_threads["1T"]["median_ms"]
    phys_key = f"{hw.physical_cores}T"
    logi_key = f"{hw.logical_cores}T"
    t_phys = by_threads[phys_key]["median_ms"]
    t_logi = by_threads[logi_key]["median_ms"]

    return {
        "finding_id": "X02_PHYSICAL_VS_SMT_AVX512_SCALING",
        "cpu_kernel": kernel_seen,
        "physical_cores": hw.physical_cores,
        "logical_cores": hw.logical_cores,
        "threads_table": by_threads,
        "speedup_1T_to_physical": round(t1 / max(t_phys, 1e-9), 3),
        "speedup_physical_to_smt": round(t_phys / max(t_logi, 1e-9), 3),
        "smt_faster_than_physical": t_logi < t_phys,
        "evidence": "LOCAL_MEASURED",
    }


def measure_x03_subnormal_assist_vs_mxcsr_ftz(runs: int = 30) -> Dict[str, Any]:
    """X03: Test x86 IEEE-754 subnormal (~1e-42) microcode penalty in NumPy vs OpenVINO oneDNN FTZ/DAZ."""
    import openvino.opset13 as ops

    ov, core, _ = get_shared_ov_core(DEFAULT_CACHE_DIR)
    shape = (1, 8, 64, 64)
    rng = np.random.default_rng(6)
    x_norm = (rng.uniform(0.5, 1.5, size=shape)).astype(np.float32)
    # Construct true IEEE-754 float32 subnormals (< 1.17549435e-38, nonzero)
    x_sub = np.ldexp(np.ones(shape, dtype=np.float32), -138)  # 2^-138 ~= 3.58e-42 (subnormal, != 0)
    w_unit = np.full((8, 8, 3, 3), 0.5, dtype=np.float32)

    # 1. NumPy scalar/elementwise iterative decay entering subnormals vs staying normal
    vec_norm = np.full(16_384, 1.0, dtype=np.float32)
    vec_sub = np.ldexp(np.ones(16_384, dtype=np.float32), -135)
    decay = np.float32(0.9999)
    add_tiny = np.ldexp(np.float32(1.0), -140)
    add_norm = np.float32(1e-6)

    np_norm_ms: List[float] = []
    np_sub_ms: List[float] = []
    for _ in range(runs):
        vn = vec_norm.copy()
        t0 = time.perf_counter()
        for _ in range(16):
            vn = vn * decay + add_norm
        np_norm_ms.append((time.perf_counter() - t0) * 1000.0)

        vs = vec_sub.copy()
        t1 = time.perf_counter()
        for _ in range(16):
            vs = vs * decay + add_tiny
        np_sub_ms.append((time.perf_counter() - t1) * 1000.0)

    numpy_preserves_subnormals = bool(np.any((vs > 0.0) & (vs < np.finfo(np.float32).tiny)))

    # 2. OpenVINO CPU oneDNN convolution on normal vs subnormal inputs
    param = ops.parameter(list(shape), np.float32, name="x")
    conv_model = ov.Model([ops.convolution(param, ops.constant(w_unit), [1, 1], [1, 1], [1, 1], [1, 1])], [param], "subnorm_probe")
    compiled = core.compile_model(conv_model, "CPU", {"INFERENCE_PRECISION_HINT": "f32", "INFERENCE_NUM_THREADS": 1})
    req = compiled.create_infer_request()

    ov_norm_ms: List[float] = []
    ov_sub_ms: List[float] = []
    out_sub = None
    for _ in range(runs):
        t0 = time.perf_counter()
        req.infer({0: x_norm})
        ov_norm_ms.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        res_s = req.infer({0: x_sub})
        ov_sub_ms.append((time.perf_counter() - t1) * 1000.0)
        if out_sub is None:
            out_sub = np.array(res_s[compiled.output(0)], copy=True)

    if out_sub is None:
        raise ContractViolation("Missing subnormal OpenVINO output", violation_code="ERR_X03_OUT_NONE")
    ov_flushed_to_zero = bool(np.all(out_sub == 0.0))
    np_n_stat, np_s_stat = _stats(np_norm_ms), _stats(np_sub_ms)
    ov_n_stat, ov_s_stat = _stats(ov_norm_ms), _stats(ov_sub_ms)

    return {
        "finding_id": "X03_SUBNORMAL_ASSIST_VS_FTZ_DAZ",
        "subnormal_input_nonzero": bool(np.all(x_sub > 0.0)),
        "numpy_vector_fma": {
            "normal": np_n_stat,
            "subnormal": np_s_stat,
            "subnormal_slowdown_ratio": round(np_s_stat["median_ms"] / max(np_n_stat["median_ms"], 1e-9), 3),
            "preserves_ieee754_subnormals": numpy_preserves_subnormals,
        },
        "openvino_onednn_cpu": {
            "normal": ov_n_stat,
            "subnormal": ov_s_stat,
            "subnormal_slowdown_ratio": round(ov_s_stat["median_ms"] / max(ov_n_stat["median_ms"], 1e-9), 3),
            "flushes_subnormals_to_zero_ftz_daz": ov_flushed_to_zero,
        },
        "evidence": "LOCAL_MEASURED",
    }


def measure_x04_cacheline_stride_and_alignment(runs: int = 30) -> Dict[str, Any]:
    """X04: Measure 64 B cache-line stride penalty (stride 1 vs stride 16) and ov.Tensor alignment."""
    ov, _, _ = get_shared_ov_core(DEFAULT_CACHE_DIR)
    # Compare summing 262,144 floats contiguously (stride 1) vs spaced by 16 floats (1 per 64 B cache line)
    count = 262_144
    big = np.ones(count * 16, dtype=np.float32)
    contig = big[:count]
    strided_64b = big[::16]  # same element count (262,144), touches 16x more 64 B cache lines (16 MB span)

    contig_ms: List[float] = []
    stride_ms: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = float(np.add.reduce(contig))
        contig_ms.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        _ = float(np.add.reduce(strided_64b))
        stride_ms.append((time.perf_counter() - t1) * 1000.0)

    # Probe ov.Tensor(shared_memory=True) on 64 B-aligned vs +4 B unaligned slice
    raw_bytes = bytearray(1024 + 128)
    base_addr = np.frombuffer(raw_bytes, dtype=np.uint8).ctypes.data
    pad_64 = (64 - (base_addr % 64)) % 64
    aligned_arr = np.frombuffer(raw_bytes, dtype=np.float32, count=64, offset=pad_64)
    unaligned_arr = np.frombuffer(raw_bytes, dtype=np.float32, count=64, offset=pad_64 + 4)

    t_aligned = ov.Tensor(aligned_arr, shared_memory=True)
    aligned_arr[0] = 42.5
    aligned_zero_copy = bool(float(t_aligned.data[0]) == 42.5)

    unaligned_zero_copy = False
    unaligned_note = ""
    try:
        t_unaligned = ov.Tensor(unaligned_arr, shared_memory=True)
        unaligned_arr[0] = 19.25
        unaligned_zero_copy = bool(float(t_unaligned.data[0]) == 19.25)
        unaligned_note = "shared_memory=True accepted +4B offset"
    except Exception as exc:
        unaligned_note = f"shared_memory=True rejected +4B offset: {type(exc).__name__}"

    c_stat, s_stat = _stats(contig_ms), _stats(stride_ms)
    return {
        "finding_id": "X04_CACHELINE_64B_STRIDE_AND_ALIGNMENT",
        "cache_line_bytes": CACHE_LINE_BYTES,
        "elements_summed": count,
        "stride_1_contiguous": c_stat,
        "stride_16_every_64B_line": s_stat,
        "stride_16_slowdown_ratio": round(s_stat["median_ms"] / max(c_stat["median_ms"], 1e-9), 3),
        "ov_tensor_64B_aligned_zero_copy": aligned_zero_copy,
        "ov_tensor_plus4B_offset_zero_copy": unaligned_zero_copy,
        "ov_alignment_note": unaligned_note,
        "evidence": "LOCAL_MEASURED",
    }


def measure_x05_128b_binary_frame_vs_json(runs: int = 30, batch: int = 500) -> Dict[str, Any]:
    """X05: Compare x86 little-endian struct.pack 128 B frame + SHA256 vs JSON serialization + SHA256."""
    # Little-endian 96 B header+payload (<4I2f72s = 16 + 8 + 72 = 96 B) + 32 B SHA-256 = 128 B
    fmt = "<4I2f72s"
    if struct.calcsize(fmt) != 96:
        raise ContractViolation("X05 binary header+payload != 96 B", violation_code="ERR_X05_FMT")
    bin_ms: List[float] = []
    json_ms: List[float] = []

    sample_dict = {
        "magic": 0x4B525953,
        "round": 3,
        "hp": VITAL_MAX_HP,
        "threads": 4,
        "uma": 0.35,
        "lat_ms": 3.24,
        "action": "MOVE C01->(2,0)|VERIFY",
    }
    action_bytes = sample_dict["action"].encode("utf-8").ljust(72, b"\x00")

    for _ in range(runs):
        t0 = time.perf_counter()
        for k in range(batch):
            raw96 = struct.pack(fmt, 0x4B525953, k, VITAL_MAX_HP, 4, 0.35, 3.24, action_bytes)
            frame128 = raw96 + hashlib.sha256(raw96).digest()
        bin_ms.append((time.perf_counter() - t0) * 1000.0)
        if len(frame128) != FRAME_SIZE_BYTES:
            raise ContractViolation("X05 frame != 128 B", violation_code="ERR_X05_FRAME")

        t1 = time.perf_counter()
        for k in range(batch):
            sample_dict["round"] = k
            enc = json.dumps(sample_dict, sort_keys=True).encode("utf-8")
            _ = hashlib.sha256(enc).hexdigest()
        json_ms.append((time.perf_counter() - t1) * 1000.0)

    b_stat, j_stat = _stats(bin_ms), _stats(json_ms)
    frames_per_sec = round((batch / (b_stat["median_ms"] / 1000.0)), 1)
    return {
        "finding_id": "X05_LITTLE_ENDIAN_128B_VS_JSON",
        "byte_order": sys.byteorder,
        "frame_bytes": FRAME_SIZE_BYTES,
        "cache_lines_per_frame": FRAME_SIZE_BYTES // CACHE_LINE_BYTES,
        "batch_frames": batch,
        "binary_128B_struct_sha256": b_stat,
        "utf8_json_sha256": j_stat,
        "binary_speedup_over_json": round(j_stat["median_ms"] / max(b_stat["median_ms"], 1e-9), 3),
        "binary_frames_per_sec_median": frames_per_sec,
        "binary_throughput_mb_per_sec": round((frames_per_sec * FRAME_SIZE_BYTES) / (1024 ** 2), 2),
        "evidence": "LOCAL_MEASURED",
    }


def run_x86_platform_findings_computation(
    runs: int = 30,
    write_reports: bool = True,
    json_path: Path = REPORTS / "x86_platform_findings.json",
    md_path: Path = REPORTS / "x86_platform_findings.md",
) -> Dict[str, Any]:
    """Run all 5 x86-64 platform computations and export structured JSON + Markdown reports."""
    hw = get_cached_hw_profile()
    isa = probe_isa_flags()
    x01 = measure_x01_cache_bandwidth_hierarchy(runs=runs)
    x02 = measure_x02_physical_vs_smt_avx512_scaling(runs=runs)
    x03 = measure_x03_subnormal_assist_vs_mxcsr_ftz(runs=runs)
    x04 = measure_x04_cacheline_stride_and_alignment(runs=runs)
    x05 = measure_x05_128b_binary_frame_vs_json(runs=runs)

    report = {
        "vital_max_hp": VITAL_MAX_HP,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "arch": platform.machine(),
            "byte_order": sys.byteorder,
            "cpu_name": hw.cpu_name,
            "physical_cores": hw.physical_cores,
            "logical_cores": hw.logical_cores,
            "ram_gb": hw.ram_gb,
            "os": hw.os_name,
            "python": hw.python,
            "isa_flags": isa["flags"],
        },
        "findings": {
            "X01_cache_hierarchy": x01,
            "X02_smt_vs_physical_avx512": x02,
            "X03_subnormals_and_ftz_daz": x03,
            "X04_cacheline_stride_and_alignment": x04,
            "X05_binary_128b_vs_json": x05,
        },
        "unmeasured_boundaries": [
            "Priame hardvérové počítadlá RAPL (Joule) a IMC šírky pásma nie sú v používateľskom priestore Windows čítané (UNKNOWN).",
            "Frekvencia jednotlivých jadier pod AVX-512 downclockingom nie je odčítaná z MSR IA32_PERF_STATUS (UNKNOWN).",
        ],
    }

    if write_reports:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        t01 = x01["tiers"]
        lines = [
            "# Výpočtová správa: Nové zistenia pre platformu x86-64 (Willow Cove / AVX-512)",
            "",
            f"- **Procesor**: `{hw.cpu_name}` (`{hw.physical_cores}` fyzické / `{hw.logical_cores}` logické jadrá, `{ sys.byteorder }`-endian)",
            f"- **ISA príznaky (`WIN32_IsProcessorFeaturePresent`)**: `{json.dumps(isa['flags'])}`",
            f"- **Vzorky na meranie**: `n = {runs}` (`LOCAL_MEASURED`)",
            "",
            "## Prehľad 5 empirických zistení",
            "",
            "| ID | Skúmaný jav na x86-64 | Kľúčová nameraná hodnota | Dôsledok pre Krystal Stack |",
            "|---|---|---|---|",
            f"| `X01` | Priepustnosť L1d -> L2 -> L3 -> UMA DRAM | `{t01['L1d_32KB']['logical_gbps_median']}` -> `{t01['L2_512KB']['logical_gbps_median']}` -> `{t01['L3_4MB']['logical_gbps_median']}` -> `{t01['DRAM_32MB']['logical_gbps_median']}` GB/s (`{x01['l1d_over_dram_bandwidth_ratio']}x` L1d/DRAM) | Držať mikro-tenzory `< 48 KB` v L1d na CPU namiesto prenosu cez UMA |",
            f"| `X02` | Škálovanie `1T -> {hw.physical_cores}T` (fyzické) vs. `{hw.logical_cores}T` (SMT) pod `{x02['cpu_kernel']}` | `1T->{hw.physical_cores}T`: `{x02['speedup_1T_to_physical']}x`, `{hw.physical_cores}T->{hw.logical_cores}T`: `{x02['speedup_physical_to_smt']}x` | Pri nízkolatenčnom AVX-512 behu viazať vlákna na fyzické jadrá, ak SMT neprináša zisk |",
            f"| `X03` | Subnormálne čísla (`~3.6e-42`) v NumPy vs. OpenVINO `oneDNN` | NumPy spomalenie `{x03['numpy_vector_fma']['subnormal_slowdown_ratio']}x` (zachová subnormály=`{x03['numpy_vector_fma']['preserves_ieee754_subnormals']}`); oneDNN spomalenie `{x03['openvino_onednn_cpu']['subnormal_slowdown_ratio']}x` (FTZ/DAZ nulovanie=`{x03['openvino_onednn_cpu']['flushes_subnormals_to_zero_ftz_daz']}`) | Vysvetľuje odlišné správanie pod prahom `1.18e-38` a chráni oneDNN pred mikrokódovou pascou |",
            f"| `X04` | Krok po `64 B` cache linajkach (`stride 16`) a zarovnanie `ov.Tensor` | Spomalenie `stride 16 / stride 1`: `{x04['stride_16_slowdown_ratio']}x`; zero-copy `64B`=`{x04['ov_tensor_64B_aligned_zero_copy']}`, `+4B`=`{x04['ov_tensor_plus4B_offset_zero_copy']}` | Susediace uloženie (`C-contiguous`) je kritické; rozptýlené indexy ničia využitie 64 B linajok |",
            f"| `X05` | Little-endian binárny `128 B` rámec (`2x 64 B`) vs. UTF-8 JSON | `{x05['binary_speedup_over_json']}x` rýchlejšie (`{x05['binary_frames_per_sec_median']}` rámcov/s, `{x05['binary_throughput_mb_per_sec']} MB/s`) | Potvrdzuje zmysel 128 B zmluvných rámcov oproti textovému JSON v kritickej slučke |",
            "",
            "## Neodmerané hranice (`UNKNOWN`)",
            "",
        ]
        for u in report["unmeasured_boundaries"]:
            lines.append(f"- {u}")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report


if __name__ == "__main__":
    r = run_x86_platform_findings_computation(runs=30, write_reports=True)
    print(json.dumps(r["findings"], indent=2, ensure_ascii=False))
