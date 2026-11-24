"""Reproducible build & PE64/ELF verification script for Krystal Stack `#![no_std]`
concurrent low-footprint Rust modules (`krystal_concurrent_rust_core` & `pulse_logger_core`).

Produces:
  - `native/krystal_concurrent_rust_core.dll` (or `.so` on Linux) + `artifacts/krystal_concurrent_rust_core_manifest.json`
  - `native/pulse_logger_core.dll` (or `.so` on Linux) + `artifacts/pulse_logger_core_manifest.json`
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


def parse_pe64_exports_and_isa_opcodes(pe_bytes: bytes) -> Dict[str, Any]:
    """Parses PE32+ (x86-64) COFF headers, `.edata` Export Directory, and `.text` machine
    instructions directly from raw bytes without invoking `LoadLibraryExW`.
    """
    if len(pe_bytes) < 0x100 or pe_bytes[:2] != b"MZ":
        return {"format": "NON_PE_OR_ELF", "exports": [], "isa_opcodes": {}}

    pe_off = struct.unpack_from("<I", pe_bytes, 0x3C)[0]
    if pe_bytes[pe_off : pe_off + 4] != b"PE\x00\x00":
        return {"format": "INVALID_PE", "exports": [], "isa_opcodes": {}}

    machine, num_sections = struct.unpack_from("<HH", pe_bytes, pe_off + 4)
    opt_hdr_size = struct.unpack_from("<H", pe_bytes, pe_off + 20)[0]
    opt_off = pe_off + 24
    magic = struct.unpack_from("<H", pe_bytes, opt_off)[0]
    if magic != 0x20B:  # PE32+ (64-bit)
        return {"format": f"PE_MAGIC_{hex(magic)}", "exports": [], "isa_opcodes": {}}

    export_rva, export_size = struct.unpack_from("<II", pe_bytes, opt_off + 112)
    sec_table_off = opt_off + opt_hdr_size

    sections: List[Tuple[str, int, int, int, int]] = []
    text_bytes = b""
    for idx in range(num_sections):
        soff = sec_table_off + idx * 40
        raw_name = pe_bytes[soff : soff + 8].rstrip(b"\x00").decode("ascii", errors="replace")
        virt_size, virt_addr, raw_size, raw_ptr = struct.unpack_from("<IIII", pe_bytes, soff + 8)
        sections.append((raw_name, virt_addr, max(virt_size, raw_size), raw_ptr, raw_size))
        if raw_name == ".text":
            text_bytes = pe_bytes[raw_ptr : raw_ptr + raw_size]

    def rva_to_offset(rva: int) -> int:
        for _name, va, span, ptr, _rsize in sections:
            if va <= rva < va + span:
                return ptr + (rva - va)
        return -1

    exports: List[str] = []
    if export_rva > 0 and export_size > 0:
        exp_off = rva_to_offset(export_rva)
        if exp_off >= 0 and exp_off + 40 <= len(pe_bytes):
            num_names = struct.unpack_from("<I", pe_bytes, exp_off + 24)[0]
            names_rva = struct.unpack_from("<I", pe_bytes, exp_off + 32)[0]
            names_off = rva_to_offset(names_rva)
            if names_off >= 0:
                for i in range(min(num_names, 256)):
                    str_rva = struct.unpack_from("<I", pe_bytes, names_off + i * 4)[0]
                    str_off = rva_to_offset(str_rva)
                    if str_off >= 0:
                        end = pe_bytes.find(b"\x00", str_off)
                        if end > str_off:
                            exports.append(pe_bytes[str_off:end].decode("ascii", errors="replace"))

    # Scan `.text` section for key x86-64 hardware instructions:
    # - `F0 ... 0F B1` / `F0 48 0F B1` : LOCK CMPXCHG (atomic CAS)
    # - `F3 0F BC` / `F3 48 0F BC`     : TZCNT (hardware trailing_zeros)
    # - `0F 31`                        : RDTSC (hardware cycle counter)
    # - `48 0F C3` / `0F C3`           : MOVNTI (non-temporal cache-bypassing store)
    isa_opcodes = {
        "has_lock_cmpxchg_cas": (b"\xf0\x48\x0f\xb1" in text_bytes) or (b"\x0f\xb1" in text_bytes),
        "has_tzcnt_bmi1": (b"\xf3\x48\x0f\xbc" in text_bytes) or (b"\x0f\xbc" in text_bytes),
        "has_rdtsc": b"\x0f\x31" in text_bytes,
        "has_movnti_stream": (b"\x48\x0f\xc3" in text_bytes) or (b"\x0f\xc3" in text_bytes),
        "text_section_bytes": len(text_bytes),
    }

    return {
        "format": "PE32+_X86_64",
        "machine_hex": hex(machine),
        "sections": [s[0] for s in sections],
        "exports": exports,
        "isa_opcodes": isa_opcodes,
    }


def build_all_concurrent_rust_crates() -> Dict[str, Any]:
    is_win = platform.system() == "Windows"
    cargo_bin = shutil.which("cargo") or str(Path.home() / ".cargo" / "bin" / ("cargo.exe" if is_win else "cargo"))
    env = os.environ.copy()
    cargo_dir = str(Path(cargo_bin).parent)
    env["PATH"] = f"{cargo_dir}{os.pathsep}{env.get('PATH', '')}"

    # Tiger Lake / Willow Cove + Kisak/Fedora Rust codegen optimizations:
    # Enables AVX2, FMA, BMI1 (TZCNT), BMI2, LZCNT, POPCNT, SHA-NI, and AES-NI vector/crypto instructions.
    tigerlake_rustflags = "-C target-feature=+avx2,+fma,+bmi1,+bmi2,+lzcnt,+popcnt,+sha,+aes"
    existing_rf = env.get("RUSTFLAGS", "").strip()
    env["RUSTFLAGS"] = f"{existing_rf} {tigerlake_rustflags}".strip()

    if is_win:
        zig_pkg = (
            Path.home()
            / "AppData/Local/Microsoft/WinGet/Packages/zig.zig_Microsoft.Winget.Source_8wekyb3d8bbwe"
            / "zig-x86_64-windows-0.17.0"
        )
        if zig_pkg.exists():
            env["PATH"] = f"{zig_pkg}{os.pathsep}{env['PATH']}"
        linker_cmd = REPO_ROOT / "scripts" / "zig_x86_64_windows_gnu_cc.cmd"
        target_triple = "x86_64-pc-windows-gnullvm"
        env["CARGO_TARGET_X86_64_PC_WINDOWS_GNULLVM_LINKER"] = str(linker_cmd)
        ext_prefix, ext_suffix = "", ".dll"
    else:
        target_triple = ""
        ext_prefix, ext_suffix = "lib", ".so"

    crates = [
        ("krystal_concurrent_rust_core", REPO_ROOT / "native" / "krystal_concurrent_rust_core"),
        ("pulse_logger_core", REPO_ROOT / "native" / "pulse_logger_core"),
        ("krystal_sovereign_firmware_rs", REPO_ROOT / "native" / "krystal_sovereign_firmware_rs"),
    ]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"built_crates": []}

    for crate_name, crate_dir in crates:
        cmd = [cargo_bin, "build", "--release", "--manifest-path", str(crate_dir / "Cargo.toml")]
        if target_triple:
            cmd.extend(["--target", target_triple])

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        elapsed_s = round(time.perf_counter() - t0, 3)
        if proc.returncode != 0:
            raise RuntimeError(f"Cargo build failed for {crate_name}:\n{proc.stderr}")

        rel_dir = (
            crate_dir / "target" / target_triple / "release"
            if target_triple
            else crate_dir / "target" / "release"
        )
        built_bin = rel_dir / f"{ext_prefix}{crate_name}{ext_suffix}"
        staged_bin = REPO_ROOT / "native" / f"{ext_prefix}{crate_name}{ext_suffix}"
        shutil.copy2(built_bin, staged_bin)

        raw = staged_bin.read_bytes()
        sha256_hex = hashlib.sha256(raw).hexdigest()
        src_rs = crate_dir / "src" / "lib.rs"
        src_sha256 = hashlib.sha256(src_rs.read_bytes()).hexdigest()
        pe_info = parse_pe64_exports_and_isa_opcodes(raw) if is_win else {"format": "ELF_SO", "exports": []}

        manifest = {
            "schema_version": "1.1.0",
            "crate_name": crate_name,
            "target_binary": staged_bin.name,
            "target_triple": target_triple or "host-linux",
            "no_std_cdylib": True,
            "external_dependencies": 0,
            "compile_time_s": elapsed_s,
            "binary_size_bytes": len(raw),
            "binary_size_kb": round(len(raw) / 1024.0, 2),
            "dll_sha256": sha256_hex,
            "source_sha256": src_sha256,
            "pe_inspection": pe_info,
        }
        manifest_path = ARTIFACTS_DIR / f"{crate_name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        results["built_crates"].append(manifest)

    return results


if __name__ == "__main__":
    summary = build_all_concurrent_rust_crates()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
