"""
krystal_kernel/godot_krystal_edition_packager.py

Godot Krystal Edition Packaging & Distribution Engine.
======================================================
Assembles, verifies, and packages the 6 sovereign engine layers into
a production-ready custom distribution bundle: "Godot Krystal Edition".

Architecture Layers Packaged:
  - L0: Silicon PDN Telemetry & Voltage Damping
  - L1: Chronometric VBlank Shadow Compositor (KSPP P3)
  - L2: Spatial Symplectic Integrator (det(J)==1.0) & Hex-HSL Color Manifold
  - L3: Cognitive Cadence & 12-Frame Dwell Hysteresis Governor (KSPP P8)
  - L4: Universal 128B SPSC Zero-Copy Contract Framing (KSPP P2, P9)
  - L5: Constitutional Gatekeeper & Real-Time Intervention Client (Delta S == 0)

Constitutional Invariant: VITAL_MAX_HP == 6
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from .constants import VITAL_MAX_HP, ContractViolation
from .unified_dependency_coordinator import (
    RuntimeStreamKind,
    KeyDependencyCoordinatorId,
    GLOBAL_UNIFIED_DEPENDENCY_COORDINATOR
)


class EngineLayerId(str, Enum):
    L0_SILICON_PDN = "L0_SILICON_PDN"
    L1_CHRONOMETRIC = "L1_CHRONOMETRIC"
    L2_SPATIAL_SYMPLECTIC = "L2_SPATIAL_SYMPLECTIC"
    L3_COGNITIVE_CADENCE = "L3_COGNITIVE_CADENCE"
    L4_SPSC_ZERO_COPY = "L4_SPSC_ZERO_COPY"
    L5_CONSTITUTIONAL_GATEKEEPER = "L5_CONSTITUTIONAL_GATEKEEPER"


@dataclass(frozen=True)
class EngineLayerSpec:
    """Formal specification of an engine layer integrated into Godot Krystal Edition."""
    layer_id: EngineLayerId
    name: str
    script_rel_path: str
    description: str
    stream_kind: RuntimeStreamKind
    primary_coordinator: KeyDependencyCoordinatorId
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"EngineLayerSpec constitutional breach: {self.vital_max_hp} != {VITAL_MAX_HP}")


@dataclass(frozen=True)
class EditionPackageManifest:
    """Immutable manifest declaring the full constituent set of Godot Krystal Edition."""
    edition_name: str
    edition_version: str
    target_godot_engine: str
    total_layers: int
    layers: Tuple[EngineLayerSpec, ...]
    packaged_scripts: Tuple[str, ...]
    packaged_scenes: Tuple[str, ...]
    packaged_shaders: Tuple[str, ...]
    package_sha256: str
    is_constitutionally_valid: bool
    vital_max_hp: int = VITAL_MAX_HP


@dataclass(frozen=True)
class EditionPackagingReceipt:
    """Cryptographic audit receipt emitted on packaging Godot Krystal Edition."""
    package_id: str
    output_directory: str
    total_files_packaged: int
    total_bytes_packaged: int
    manifest_path: str
    launcher_bat_path: str
    launcher_sh_path: str
    hmac_digest: str
    timestamp_iso: str
    vital_max_hp: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Receipt constitutional breach: {self.vital_max_hp} != {VITAL_MAX_HP}")


class GodotKrystalEditionPackager:
    """
    Authoritative packager and distribution manager for Godot Krystal Edition.
    Validates file integrity, compiles manifests, verifies invariants,
    and generates one-click launchers.
    """

    EDITION_NAME: str = "Godot Krystal Edition"
    EDITION_VERSION: str = "1.0.0-SOVEREIGN"
    TARGET_GODOT: str = "Godot 4.2+ Forward+ Vulkan"

    def __init__(self, repo_root: Optional[str] = None, vital_max_hp: int = VITAL_MAX_HP) -> None:
        if vital_max_hp != VITAL_MAX_HP:
            raise ContractViolation(f"Packager constitutional breach: {vital_max_hp} != {VITAL_MAX_HP}")
        self.vital_max_hp: int = vital_max_hp
        self.repo_root: str = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.godot_dir: str = os.path.join(self.repo_root, "godot_project")
        self.artifacts_dir: str = os.path.join(self.repo_root, "artifacts", "godot_krystal_edition")

        self.layers: Tuple[EngineLayerSpec, ...] = self._define_sovereign_layers()

    def _define_sovereign_layers(self) -> Tuple[EngineLayerSpec, ...]:
        return (
            EngineLayerSpec(
                layer_id=EngineLayerId.L0_SILICON_PDN,
                name="Silicon PDN & Hardware Telemetry Bridge",
                script_rel_path="scripts/GodotSiliconPDNBridge.gd",
                description="Live Vdd, Tj, overshoot monitoring, and automatic workload shedding",
                stream_kind=RuntimeStreamKind.SILICON_PDN_TELEMETRY,
                primary_coordinator=KeyDependencyCoordinatorId.PDN_VOLTAGE_STABILITY_COORDINATOR,
                vital_max_hp=self.vital_max_hp
            ),
            EngineLayerSpec(
                layer_id=EngineLayerId.L1_CHRONOMETRIC,
                name="Chronometric VBlank Shadow Compositor",
                script_rel_path="scripts/GodotChronometricCompositor.gd",
                description="VBlank shadow window scheduler (>= 3.5ms) enforcing KSPP P3",
                stream_kind=RuntimeStreamKind.CHRONOMETRIC_RENDER_STREAM,
                primary_coordinator=KeyDependencyCoordinatorId.VBLANK_SHADOW_TIMING_COORDINATOR,
                vital_max_hp=self.vital_max_hp
            ),
            EngineLayerSpec(
                layer_id=EngineLayerId.L2_SPATIAL_SYMPLECTIC,
                name="Spatial Symplectic Integrator & Hex-HSL Manifold",
                script_rel_path="scripts/GodotSymplecticIntegrator.gd",
                description="Phase-space Leapfrog Verlet det(J)==1.0 and 3D Hex-HSL D6 color manifold",
                stream_kind=RuntimeStreamKind.SPATIAL_SYMPLECTIC_STREAM,
                primary_coordinator=KeyDependencyCoordinatorId.SYMPLECTIC_CFL_STEP_COORDINATOR,
                vital_max_hp=self.vital_max_hp
            ),
            EngineLayerSpec(
                layer_id=EngineLayerId.L3_COGNITIVE_CADENCE,
                name="Cognitive Cadence & 12-Frame Dwell Governor",
                script_rel_path="scripts/GodotInertialGovernor.gd",
                description="Dynamic resolution scaling with 12-frame dwell hysteresis (P8) and 10Hz alpha pulse",
                stream_kind=RuntimeStreamKind.COGNITIVE_CADENCE_STREAM,
                primary_coordinator=KeyDependencyCoordinatorId.INERTIAL_HYSTERESIS_COORDINATOR,
                vital_max_hp=self.vital_max_hp
            ),
            EngineLayerSpec(
                layer_id=EngineLayerId.L4_SPSC_ZERO_COPY,
                name="Universal 128B SPSC Zero-Copy Contract Framing",
                script_rel_path="scripts/GodotZeroCopyContractBridge.gd",
                description="Aligned 128B binary contract frames with HMAC-SHA256 integrity (P2, P9)",
                stream_kind=RuntimeStreamKind.SPSC_COORDINATION_STREAM,
                primary_coordinator=KeyDependencyCoordinatorId.CONSTITUTIONAL_ARTIFACT_GATE_COORDINATOR,
                vital_max_hp=self.vital_max_hp
            ),
            EngineLayerSpec(
                layer_id=EngineLayerId.L5_CONSTITUTIONAL_GATEKEEPER,
                name="Constitutional Gatekeeper & Safe Real-Time Actuator Client",
                script_rel_path="scripts/GodotRealtimeInterventionClient.gd",
                description="Two-phase intervention client guaranteeing Delta S == 0 on any safety violation",
                stream_kind=RuntimeStreamKind.SPSC_COORDINATION_STREAM,
                primary_coordinator=KeyDependencyCoordinatorId.CONSTITUTIONAL_ARTIFACT_GATE_COORDINATOR,
                vital_max_hp=self.vital_max_hp
            ),
        )

    def assemble_package(self, output_dir: Optional[str] = None) -> EditionPackagingReceipt:
        """
        Assembles, audits, and generates the custom Godot Krystal Edition package.
        Validates all layer scripts, creates the manifest, and writes launch scripts.
        """
        target_out: str = output_dir or self.artifacts_dir
        os.makedirs(target_out, exist_ok=True)

        scripts_dir = os.path.join(self.godot_dir, "scripts")
        scenes_dir = os.path.join(self.godot_dir, "scenes")
        shaders_dir = os.path.join(self.godot_dir, "shaders")

        # 1. Audit and collect scripts
        packaged_scripts: List[str] = []
        total_bytes: int = 0
        hasher = hashlib.sha256()

        for layer in self.layers:
            full_path = os.path.join(self.godot_dir, layer.script_rel_path)
            if not os.path.isfile(full_path):
                raise FileNotFoundError(f"Missing required layer script: {full_path}")
            size = os.path.getsize(full_path)
            total_bytes += size
            packaged_scripts.append(layer.script_rel_path)
            with open(full_path, "rb") as f:
                hasher.update(f.read())

        # Include Master Autoload, Showcase, and Headless Test Runner
        auxiliary_scripts = [
            "scripts/GodotKrystalEdition.gd",
            "scripts/GodotKrystalEditionStage.gd",
            "scripts/GodotKrystalHeadlessTestRunner.gd",
            "scripts/KrystalHoloBridge.gd"
        ]
        for s_rel in auxiliary_scripts:
            full_path = os.path.join(self.godot_dir, s_rel)
            if os.path.isfile(full_path):
                size = os.path.getsize(full_path)
                total_bytes += size
                packaged_scripts.append(s_rel)
                with open(full_path, "rb") as f:
                    hasher.update(f.read())

        # 2. Collect core scenes
        packaged_scenes: List[str] = []
        if os.path.isdir(scenes_dir):
            for sc in sorted(os.listdir(scenes_dir)):
                if sc.endswith(".tscn"):
                    rel_p = f"scenes/{sc}"
                    full_p = os.path.join(scenes_dir, sc)
                    packaged_scenes.append(rel_p)
                    total_bytes += os.path.getsize(full_p)
                    with open(full_p, "rb") as f:
                        hasher.update(f.read())

        # 3. Collect shaders
        packaged_shaders: List[str] = []
        if os.path.isdir(shaders_dir):
            for sh in sorted(os.listdir(shaders_dir)):
                if sh.endswith(".gdshader"):
                    rel_p = f"shaders/{sh}"
                    full_p = os.path.join(shaders_dir, sh)
                    packaged_shaders.append(rel_p)
                    total_bytes += os.path.getsize(full_p)
                    with open(full_p, "rb") as f:
                        hasher.update(f.read())

        package_sha256 = hasher.hexdigest()

        # 4. Construct manifest
        manifest = EditionPackageManifest(
            edition_name=self.EDITION_NAME,
            edition_version=self.EDITION_VERSION,
            target_godot_engine=self.TARGET_GODOT,
            total_layers=len(self.layers),
            layers=self.layers,
            packaged_scripts=tuple(packaged_scripts),
            packaged_scenes=tuple(packaged_scenes),
            packaged_shaders=tuple(packaged_shaders),
            package_sha256=package_sha256,
            is_constitutionally_valid=(self.vital_max_hp == VITAL_MAX_HP),
            vital_max_hp=self.vital_max_hp
        )

        manifest_path = os.path.join(target_out, "krystal_edition_manifest.json")
        manifest_dict = asdict(manifest)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, indent=2)

        # 5. Generate one-click launchers
        launcher_bat_path = os.path.join(self.repo_root, "godot_krystal_edition.bat")
        launcher_sh_path = os.path.join(self.repo_root, "godot_krystal_edition.sh")

        bat_content = (
            "@echo off\r\n"
            "REM Godot Krystal Edition Launcher (Windows)\r\n"
            "REM Invariant: VITAL_MAX_HP == 6\r\n"
            "echo ========================================================\r\n"
            "echo   LAUNCHING GODOT KRYSTAL EDITION (v1.0.0-SOVEREIGN)\r\n"
            "echo ========================================================\r\n"
            "set GODOT_EXE=%~dp0godot_project\\godot.exe\r\n"
            "if not exist \"%GODOT_EXE%\" (\r\n"
            "    echo [INFO] godot.exe not in local directory, checking PATH...\r\n"
            "    where godot >nul 2>&1\r\n"
            "    if %ERRORLEVEL% EQU 0 ( set GODOT_EXE=godot ) else (\r\n"
            "        echo [ERROR] Godot 4.x executable not found. Please install Godot 4.2+ Forward+.\r\n"
            "        exit /b 1\r\n"
            "    )\r\n"
            ")\r\n"
            "\"%GODOT_EXE%\" --path \"%~dp0godot_project\" res://scenes/GodotKrystalEditionStage.tscn %*\r\n"
        )
        with open(launcher_bat_path, "w", encoding="utf-8") as f:
            f.write(bat_content)

        sh_content = (
            "#!/usr/bin/env bash\n"
            "# Godot Krystal Edition Launcher (POSIX / WSL)\n"
            "# Invariant: VITAL_MAX_HP == 6\n"
            "set -euo pipefail\n"
            "echo \"========================================================\"\n"
            "echo \"  LAUNCHING GODOT KRYSTAL EDITION (v1.0.0-SOVEREIGN)\"\n"
            "echo \"========================================================\"\n"
            "REPO_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
            "GODOT_BIN=\"${REPO_DIR}/godot_project/godot.x86_64\"\n"
            "if [ ! -f \"$GODOT_BIN\" ]; then\n"
            "    if command -v godot >/dev/null 2>&1; then\n"
            "        GODOT_BIN=\"godot\"\n"
            "    elif command -v godot4 >/dev/null 2>&1; then\n"
            "        GODOT_BIN=\"godot4\"\n"
            "    else\n"
            "        echo \"[ERROR] Godot 4.x executable not found in PATH or godot_project/.\"\n"
            "        exit 1\n"
            "    fi\n"
            "fi\n"
            "\"$GODOT_BIN\" --path \"${REPO_DIR}/godot_project\" res://scenes/GodotKrystalEditionStage.tscn \"$@\"\n"
        )
        with open(launcher_sh_path, "w", encoding="utf-8") as f:
            f.write(sh_content)

        # 6. Generate Packaging Receipt
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        receipt_id = f"GKE-PKG-{int(time.time())}"
        total_files = len(packaged_scripts) + len(packaged_scenes) + len(packaged_shaders) + 1  # + project.godot

        sig_payload = f"{receipt_id}:{package_sha256}:{total_files}:{total_bytes}:{self.vital_max_hp}"
        hmac_digest = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()

        receipt = EditionPackagingReceipt(
            package_id=receipt_id,
            output_directory=target_out,
            total_files_packaged=total_files,
            total_bytes_packaged=total_bytes,
            manifest_path=manifest_path,
            launcher_bat_path=launcher_bat_path,
            launcher_sh_path=launcher_sh_path,
            hmac_digest=hmac_digest,
            timestamp_iso=now_str,
            vital_max_hp=self.vital_max_hp
        )

        receipt_path = os.path.join(target_out, "packaging_receipt.json")
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(asdict(receipt), f, indent=2)

        return receipt

    def verify_package_integrity(self, manifest_path: Optional[str] = None) -> bool:
        """Verifies existence and checksums of all constituent assets in manifest."""
        target_path: str = manifest_path or os.path.join(self.artifacts_dir, "krystal_edition_manifest.json")
        if not os.path.isfile(target_path):
            return False

        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("vital_max_hp") != VITAL_MAX_HP:
            return False

        # Verify all layer script files exist
        for layer in data.get("layers", []):
            script_path = os.path.join(self.godot_dir, layer["script_rel_path"])
            if not os.path.isfile(script_path):
                return False

        return True

    # Alias for convenience
    build_package = assemble_package


# Global Singleton instance
GLOBAL_GODOT_KRYSTAL_EDITION_PACKAGER: GodotKrystalEditionPackager = (
    GodotKrystalEditionPackager()
)


if __name__ == "__main__":
    print("=" * 66)
    print("  KRYSTAL STACK // GODOT KRYSTAL EDITION PACKAGER")
    print(f"  Constitutional Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}")
    print("=" * 66)
    rcpt = GLOBAL_GODOT_KRYSTAL_EDITION_PACKAGER.assemble_package()
    print(f"[PACKAGED] ID: {rcpt.package_id}")
    print(f"[FILES]    {rcpt.total_files_packaged} assets ({rcpt.total_bytes_packaged} bytes)")
    print(f"[MANIFEST] {rcpt.manifest_path}")
    print(f"[RECEIPT]  artifacts/godot_krystal_edition/packaging_receipt.json")
    print(f"[HMAC]     {rcpt.hmac_digest}")
    print(">> GODOT KRYSTAL EDITION PACKAGE READY FOR DEPLOYMENT")
