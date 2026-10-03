"""
Unit Tests for Procedural Instance Generator & Engine Bootstrap Schemas
======================================================================
Tests:
1. Gielis 3D Superformula generation & mathematical sanity
2. Modular Cyber Spire generator
3. Sacred Alchemical Polyhedron generator
4. Schema structural validation against JSON schemas
5. Krystal-Bootstrap Web Specification compliance
"""

import unittest
import os
import sys
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from instances.procedural_generator import ProceduralInstanceGenerator

class TestProceduralBootstrap(unittest.TestCase):

    def test_01_superformula_generation(self):
        inst = ProceduralInstanceGenerator.generate_superformula_instance(seed=42)
        self.assertTrue(inst["generated"])
        self.assertEqual(inst["category"], "superformula_organic")
        self.assertIn("m", inst["default_params"])
        self.assertIn("n1", inst["default_params"])
        self.assertGreater(inst["default_params"]["bound_radius"], 0.5)
        self.assertTrue(len(inst["preferred_glyphs"]) >= 4)

    def test_02_cyber_spire_generation(self):
        inst = ProceduralInstanceGenerator.generate_cyber_spire_instance(seed=101)
        self.assertTrue(inst["generated"])
        self.assertEqual(inst["category"], "cybernetic_spire")
        self.assertGreaterEqual(inst["default_params"]["decks"], 3)
        self.assertGreaterEqual(inst["default_params"]["fin_count"], 3)
        self.assertGreater(inst["default_params"]["height"], 3.0)

    def test_03_alchemical_polyhedron_generation(self):
        inst = ProceduralInstanceGenerator.generate_alchemical_polyhedron_instance(seed=202)
        self.assertTrue(inst["generated"])
        self.assertEqual(inst["category"], "alchemical_polyhedron")
        self.assertIn(inst["default_params"]["faces"], [6, 8, 12, 20])
        self.assertGreater(inst["default_params"]["twist_rate"], 0.0)

    def test_04_instance_json_schema_validation(self):
        schema_path = os.path.join(REPO_ROOT, "schemas", "krystal_instance.schema.json")
        self.assertTrue(os.path.exists(schema_path))
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        required_keys = schema["required"]
        inst = ProceduralInstanceGenerator.generate_random_instance(seed=999)
        for k in required_keys:
            self.assertIn(k, inst, f"Generated instance missing required schema key: '{k}'")

    def test_05_engine_bootstrap_schema_validation(self):
        schema_path = os.path.join(REPO_ROOT, "schemas", "krystal_engine_bootstrap.schema.json")
        self.assertTrue(os.path.exists(schema_path))
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        self.assertIn("viewport", schema["properties"])
        self.assertIn("scene", schema["properties"])
        self.assertEqual(schema["title"], "Krystal Engine Bootstrap Web Specification")


if __name__ == "__main__":
    unittest.main()
