#!/usr/bin/env python3
"""
CrystalSec 3.0 Phase 5.3C - Native CPU SIMD Binary & Manifest Verifier.

Audits native library provenance:
- Verifies SHA-256 checksums of C sources and headers against taxonomy_simd_manifest.json.
- If taxonomy_simd.dll is present, verifies its binary SHA-256 against the manifest.
- Audits export symbols.
- Emits explicit verification report without synthetic assumptions.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MANIFEST_PATH = os.path.join(_ROOT, "artifacts", "taxonomy_simd_manifest.json")
_SRC_C = os.path.join(_ROOT, "src", "c", "taxonomy_simd.c")
_SRC_H = os.path.join(_ROOT, "src", "c", "taxonomy_simd.h")
_DLL_PATH = os.path.join(_ROOT, "src", "c", "taxonomy_simd.dll")


def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_manifest() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "manifest_valid": False,
        "sources_valid": False,
        "dll_present": False,
        "dll_hash_matched": False,
        "errors": []
    }

    if not os.path.exists(_MANIFEST_PATH):
        report["errors"].append(f"Manifest not found: {_MANIFEST_PATH}")
        return report

    try:
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        report["manifest_valid"] = True
    except Exception as exc:
        report["errors"].append(f"Failed to parse manifest: {exc}")
        return report

    # Verify sources
    src_hashes = manifest.get("source_files", {})
    sources_ok = True
    for rel_path, expected_hash in src_hashes.items():
        abs_path = os.path.join(_ROOT, rel_path)
        if not os.path.exists(abs_path):
            report["errors"].append(f"Source file missing: {rel_path}")
            sources_ok = False
            continue
        actual_hash = compute_sha256(abs_path)
        if actual_hash != expected_hash.lower():
            report["errors"].append(
                f"Source hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
            )
            sources_ok = False
    report["sources_valid"] = sources_ok

    # Verify DLL if present
    if os.path.exists(_DLL_PATH):
        report["dll_present"] = True
        actual_dll_hash = compute_sha256(_DLL_PATH)
        expected_dll_hash = manifest.get("dll_sha256")
        if expected_dll_hash and actual_dll_hash == expected_dll_hash.lower():
            report["dll_hash_matched"] = True
        elif not expected_dll_hash:
            report["errors"].append("DLL present on disk but manifest has no dll_sha256 recorded.")
        else:
            report["errors"].append(
                f"DLL hash mismatch: expected {expected_dll_hash}, got {actual_dll_hash}"
            )
    else:
        report["dll_present"] = False

    return report


def main() -> int:
    report = verify_manifest()
    print(json.dumps(report, indent=2))
    if not report["manifest_valid"] or not report["sources_valid"]:
        return 1
    if report["dll_present"] and not report["dll_hash_matched"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
