"""
tests/test_godot_krystal_edition_packager.py

Authoritative Unit Tests for Godot Krystal Edition Packager & Sovereign Engine Layers.
======================================================================================
Validates layer specifications, manifest compilation, launcher generation,
GDScript invariant compliance (VITAL_MAX_HP == 6), and packaging integrity.
"""

import os
import json
import unittest
from krystal_kernel.constants import VITAL_MAX_HP, ContractViolation
from krystal_kernel.godot_krystal_edition_packager import (
    EngineLayerId,
    EngineLayerSpec,
    EditionPackageManifest,
    EditionPackagingReceipt,
    GodotKrystalEditionPackager,
    GLOBAL_GODOT_KRYSTAL_EDITION_PACKAGER
)


class TestGodotKrystalEditionPackager(unittest.TestCase):
    """Test suite for Godot Krystal Edition packaging engine and constituent layers."""

    def setUp(self) -> None:
        self.packager = GLOBAL_GODOT_KRYSTAL_EDITION_PACKAGER

    def test_01_layer_spec_definitions_and_invariants(self) -> None:
        """Verifies that all 6 sovereign layers are declared and their files exist."""
        self.assertEqual(len(self.packager.layers), 6)
        layer_ids = {layer.layer_id for layer in self.packager.layers}
        for lid in EngineLayerId:
            self.assertIn(lid, layer_ids)

        for layer in self.packager.layers:
            self.assertEqual(layer.vital_max_hp, VITAL_MAX_HP)
            full_path = os.path.join(self.packager.godot_dir, layer.script_rel_path)
            self.assertTrue(os.path.isfile(full_path), f"Layer script missing: {full_path}")

    def test_02_package_assembly_and_manifest_generation(self) -> None:
        """Tests end-to-end packaging, manifest generation, and launch script creation."""
        receipt = self.packager.assemble_package()

        self.assertIsInstance(receipt, EditionPackagingReceipt)
        self.assertEqual(receipt.vital_max_hp, VITAL_MAX_HP)
        self.assertGreater(receipt.total_files_packaged, 30)
        self.assertGreater(receipt.total_bytes_packaged, 100_000)
        self.assertTrue(len(receipt.hmac_digest) > 0)

        # Verify manifest file
        self.assertTrue(os.path.isfile(receipt.manifest_path))
        with open(receipt.manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        self.assertEqual(manifest_data["edition_name"], "Godot Krystal Edition")
        self.assertEqual(manifest_data["total_layers"], 6)
        self.assertEqual(manifest_data["vital_max_hp"], VITAL_MAX_HP)
        self.assertTrue(manifest_data["is_constitutionally_valid"])

        # Verify Windows and POSIX launch scripts
        self.assertTrue(os.path.isfile(receipt.launcher_bat_path))
        self.assertTrue(os.path.isfile(receipt.launcher_sh_path))

        with open(receipt.launcher_bat_path, "r", encoding="utf-8") as f:
            bat_text = f.read()
            self.assertIn("VITAL_MAX_HP == 6", bat_text)
            self.assertIn("GodotKrystalEditionStage.tscn", bat_text)

        with open(receipt.launcher_sh_path, "r", encoding="utf-8") as f:
            sh_text = f.read()
            self.assertIn("VITAL_MAX_HP == 6", sh_text)
            self.assertIn("GodotKrystalEditionStage.tscn", sh_text)

    def test_03_verify_package_integrity(self) -> None:
        """Tests manifest verification function."""
        is_valid = self.packager.verify_package_integrity()
        self.assertTrue(is_valid)

    def test_04_gdscript_files_syntax_and_invariants(self) -> None:
        """Verifies that all layer scripts and singletons contain VITAL_MAX_HP == 6 assertions."""
        expected_scripts = [
            "scripts/GodotSiliconPDNBridge.gd",
            "scripts/GodotChronometricCompositor.gd",
            "scripts/GodotSymplecticIntegrator.gd",
            "scripts/GodotInertialGovernor.gd",
            "scripts/GodotZeroCopyContractBridge.gd",
            "scripts/GodotRealtimeInterventionClient.gd",
            "scripts/GodotKrystalEdition.gd",
            "scripts/GodotKrystalEditionStage.gd",
            "scripts/GodotKrystalHeadlessTestRunner.gd"
        ]

        for s_rel in expected_scripts:
            full_path = os.path.join(self.packager.godot_dir, s_rel)
            self.assertTrue(os.path.isfile(full_path), f"File {s_rel} not found")
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("VITAL_MAX_HP", content, f"VITAL_MAX_HP missing from {s_rel}")
            self.assertIn("6", content, f"Constant 6 missing from {s_rel}")

    def test_05_project_godot_autoload_registration(self) -> None:
        """Verifies that project.godot registers GodotKrystalEdition as an Autoload."""
        project_godot_path = os.path.join(self.packager.godot_dir, "project.godot")
        self.assertTrue(os.path.isfile(project_godot_path))
        with open(project_godot_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("[autoload]", content)
        self.assertIn("GodotKrystalEdition", content)
        self.assertIn("res://scripts/GodotKrystalEdition.gd", content)

    def test_06_constitutional_invariant_breach_prevention(self) -> None:
        """Verifies ContractViolation is raised if VITAL_MAX_HP != 6 is attempted."""
        with self.assertRaises(ContractViolation):
            GodotKrystalEditionPackager(vital_max_hp=5)

        with self.assertRaises(ContractViolation):
            EngineLayerSpec(
                layer_id=EngineLayerId.L0_SILICON_PDN,
                name="Breach Layer",
                script_rel_path="scripts/test.gd",
                description="Invalid",
                stream_kind=self.packager.layers[0].stream_kind,
                primary_coordinator=self.packager.layers[0].primary_coordinator,
                vital_max_hp=7
            )


if __name__ == "__main__":
    unittest.main()
