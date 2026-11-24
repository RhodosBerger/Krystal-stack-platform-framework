#!/usr/bin/env python3
r"""Terminálové CLI rozhranie pre výhradnú hardvérovú akceleráciu zvoleného subsystému:
  - `BLENDER_MCP_PROGRESSIVE` (postupné renderovanie CAD/3D scén 4 -> 16 -> 64 -> 256 vzoriek cez MCP),
  - `DUSAN_GAME_AXIOM` (`19-hex` dávková matica pre engine projektu `dusan-game`),
  - `GEMMA_2B_AST_EDITING` (`14 GB` UMA VRAM + sémantická L1d cache pre `Gemma 4 E2B`),
  - `VULKAN_KRYSTAL_DLSS` (`768` FP32 stream procesorov + presety herných enginov).

Použitie z terminálu:
  .\.venv\Scripts\python.exe -O scripts\krystal_exclusive_terminal_accel_cli.py --target BLENDER_MCP_PROGRESSIVE --vram-mb 14336
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation


def run_exclusive_terminal_acceleration(target: str = "BLENDER_MCP_PROGRESSIVE", vram_mb: int = 14_336) -> dict:
    if VITAL_MAX_HP != 6:
        raise ContractViolation("Porušený VITAL_MAX_HP != 6", violation_code="ERR_CLI_HP")

    from projects.trigonometric_hybrid_platform_framework.dissertation_eula_nextcloud_cad_blender_kb_suite import (
        execute_dissertation_eula_nextcloud_cad_blender_suite,
    )
    from projects.trigonometric_hybrid_platform_framework.fedora_dlss_vram_bionic_mcp_and_2b_ast import (
        profile_safe_chipset_vram_bionic_and_swap,
    )
    from projects.trigonometric_hybrid_platform_framework.metal_vulkan_endpoints_and_speculative_bench_loop import (
        optimize_win32_gen11_processes_and_advise_vulkan_quota,
    )

    hw_lock = optimize_win32_gen11_processes_and_advise_vulkan_quota(uma_util=0.30, junction_temp_c=60.0)
    vram_cfg = profile_safe_chipset_vram_bionic_and_swap(
        cooling_mode="TIGERLAKE_TURBO_SAFE_28W",
        requested_cpu_mhz=3800,
        requested_vram_mb=vram_mb,
        uma_util=0.30,
    )
    cad_blender = execute_dissertation_eula_nextcloud_cad_blender_suite()

    return {
        "vital_max_hp": VITAL_MAX_HP,
        "exclusive_acceleration_target": target,
        "p_core_affinity_hex": hw_lock["gen11_multithread_4p_tuning"]["physical_cores_mask"],
        "single_core_pinned_ms": hw_lock["gen11_single_core_turbo_tuning"]["pinned_core0_ms"],
        "granted_uma_vram_mb": vram_cfg["vram_and_bionic_lmstudio_bundle"]["uma_shared_vram_allocation"][
            "granted_llm_vram_mb"
        ],
        "allowed_iris_xe_eus_of_96": hw_lock["rust_vulkan_acceleration_quota_64b"]["allowed_iris_xe_eus_of_96"],
        "progressive_blender_mcp_summary": cad_blender["cad_drafter_and_progressive_blender_mcp"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Krystal Exclusive Hardware Acceleration CLI")
    parser.add_argument(
        "--target",
        default="BLENDER_MCP_PROGRESSIVE",
        choices=["BLENDER_MCP_PROGRESSIVE", "DUSAN_GAME_AXIOM", "GEMMA_2B_AST_EDITING", "VULKAN_KRYSTAL_DLSS"],
    )
    parser.add_argument("--vram-mb", type=int, default=14336)
    args = parser.parse_args()
    out = run_exclusive_terminal_acceleration(target=args.target, vram_mb=args.vram_mb)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
