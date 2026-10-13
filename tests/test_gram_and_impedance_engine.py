# ==============================================================================
# KRYSTAL-STACK: UNIT TESTS PRE ČÍTAČKY GRAMOV A HODNOTIČ IMPEDANCIE
# ==============================================================================
import unittest
import math
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from krystal_kernel.gram_and_impedance_engine import (
    NGramReader,
    GramianMatrixReader,
    SpectrogramFeatureReader,
    PhysicalMassGramReader,
    ComplexImpedanceEvaluator,
    SiliconThermalImpedanceEvaluator,
    CognitiveFlowImpedanceEvaluator,
    GramAndImpedanceEngine,
    GLOBAL_GRAM_AND_IMPEDANCE_ENGINE,
    VITAL_MAX_HP
)
from krystal_web_hub.hybrid_mcp_antigravity_connector import GLOBAL_HYBRID_MCP_CONNECTOR


class TestGramAndImpedanceEngine(unittest.TestCase):
    def setUp(self):
        self.engine = GLOBAL_GRAM_AND_IMPEDANCE_ENGINE
        self.mcp_conn = GLOBAL_HYBRID_MCP_CONNECTOR

    # ─── 1. INVARIANT TESTS ──────────────────────────────────────────────────

    def test_01_vital_max_hp_invariant(self):
        """Strictly asserts that platform-wide invariant VITAL_MAX_HP is 6."""
        self.assertEqual(VITAL_MAX_HP, 6, "Inviolable system invariant: VITAL_MAX_HP must equal 6")

    # ─── 2. ČÍTAČKY GRAMOV (GRAM READERS) ────────────────────────────────────

    def test_02_ngram_reader_extraction_and_entropy(self):
        """Verifies N-gram extraction, Shannon entropy, perplexity, and TTR."""
        text = "posledni kmen alchymia posledni kmen geometria posledni kmen"
        bigrams = NGramReader.read_ngrams(text, n=2)
        self.assertEqual(len(bigrams), 7)
        self.assertEqual(bigrams[0], ("posledni", "kmen"))

        stats = NGramReader.analyze_ngram_distribution(text, n=2)
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["total_ngrams"], 7)
        self.assertGreater(stats["entropy"], 0.0)
        self.assertGreaterEqual(stats["perplexity"], 1.0)
        self.assertEqual(stats["vital_max_hp"], 6)

        # Markov transition matrix
        trans = NGramReader.build_transition_matrix(text)
        self.assertIn("posledni", trans)
        self.assertIn("kmen", trans["posledni"])
        self.assertEqual(trans["posledni"]["kmen"], 1.0)

    def test_03_gramian_matrix_reader_orthogonality_and_volume(self):
        """Verifies Gram matrix G=V^T V, symmetry, Gram determinant, and parallelotope volume."""
        # 1. Štandardná ortonormálna báza v R^3 -> det(G) == 1.0, Volume == 1.0
        basis_vecs = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        res = GramianMatrixReader.compute_gram_matrix(basis_vecs, normalize=True)
        self.assertEqual(res["matrix_dimension"], 3)
        self.assertEqual(res["vector_dimension"], 3)
        self.assertAlmostEqual(res["gram_determinant"], 1.0, places=5)
        self.assertAlmostEqual(res["parallelotope_volume"], 1.0, places=5)
        self.assertTrue(res["is_linearly_independent"])

        # Symetria Gramovej matice: G_ij == G_ji
        g = res["gram_matrix"]
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(g[i][j], g[j][i], places=6)

        # 2. Lineárne závislé vektory -> det(G) == 0.0
        dep_vecs = [
            [1.0, 2.0],
            [2.0, 4.0]  # v2 = 2 * v1
        ]
        dep_res = GramianMatrixReader.compute_gram_matrix(dep_vecs)
        self.assertAlmostEqual(dep_res["gram_determinant"], 0.0, places=5)
        self.assertFalse(dep_res["is_linearly_independent"])

    def test_04_spectrogram_feature_reader(self):
        """Verifies acoustic spectrogram time-frequency frame analysis and spectral centroid."""
        # Syntetický frekvenčný kôš (dominantný signál okolo 1000 Hz)
        mags = [0.1] * 64
        mags[8] = 5.0  # výrazný pík na 8. koši
        spec = SpectrogramFeatureReader.read_spectral_frame(mags, sample_rate=16000)

        self.assertEqual(spec["num_bins"], 64)
        self.assertEqual(spec["sample_rate_hz"], 16000)
        self.assertGreater(spec["spectral_centroid_hz"], 0.0)
        self.assertIn("sub_bass", spec["spectral_bands"])
        self.assertIn("low_mid", spec["spectral_bands"])
        self.assertIn("high_mid", spec["spectral_bands"])
        self.assertIn("presence", spec["spectral_bands"])
        self.assertEqual(spec["vital_max_hp"], 6)

    def test_05_physical_mass_gram_reader(self):
        """Verifies substrate density mass calculation and assembly center of mass."""
        # 1. Jednotlivý komponent: 100 cm^3 kryštálového kremeňa (hustota 2.65 g/cm^3)
        comp = PhysicalMassGramReader.calculate_component_mass("crystal_quartz", 100.0, [10.0, 0.0, 0.0])
        self.assertAlmostEqual(comp["mass_grams"], 265.0, places=2)
        self.assertAlmostEqual(comp["mass_kilograms"], 0.265, places=4)

        # 2. Zložená zostava s výpočtom ťažiska
        assembly = PhysicalMassGramReader.calculate_aggregate_assembly([
            {"material": "crystal_quartz", "volume_cm3": 100.0, "position": [10.0, 0.0, 0.0]},
            {"material": "crystal_quartz", "volume_cm3": 100.0, "position": [-10.0, 0.0, 0.0]}
        ])
        self.assertEqual(assembly["component_count"], 2)
        self.assertAlmostEqual(assembly["total_mass_grams"], 530.0, places=2)
        # Symetrické ťažisko musí byť v bode x=0
        self.assertAlmostEqual(assembly["center_of_mass"][0], 0.0, places=4)

    # ─── 3. HODNOTIČ IMPEDANCIE (IMPEDANCE EVALUATORS) ───────────────────────

    def test_06_complex_electrical_impedance(self):
        """Verifies complex electrical impedance RLC, admittance, Q-factor, and reflection coefficient."""
        # R = 50 Ohm, L = 1 uH, C = 100 pF, f = 10 MHz
        res = ComplexImpedanceEvaluator.evaluate_circuit_impedance(
            resistance=50.0,
            inductance_h=1e-6,
            capacitance_f=100e-12,
            frequency_hz=10e6,
            reference_z0=50.0
        )
        self.assertEqual(res["resistance_ohms"], 50.0)
        self.assertGreater(res["impedance_magnitude_ohms"], 0.0)
        self.assertGreaterEqual(res["vswr"], 1.0)
        self.assertGreater(res["return_loss_db"], 0.0)
        self.assertIn("matching_network", res)
        self.assertEqual(res["vital_max_hp"], 6)

        # Dokonalé prispôsobenie Z = 50 + j0 k Z0 = 50
        matched = ComplexImpedanceEvaluator.evaluate_circuit_impedance(50.0, 0.0, 0.0, 1000.0, 50.0)
        self.assertAlmostEqual(matched["reflection_coefficient"]["magnitude"], 0.0, places=4)
        self.assertAlmostEqual(matched["vswr"], 1.0, places=2)
        self.assertEqual(matched["matching_network"]["topology"], "NONE_REQUIRED")

    def test_07_silicon_thermal_impedance(self):
        """Verifies transient silicon thermal impedance Z_th(t) and junction temperature rise."""
        # Test pri t = 0 s (žiadny vzostup)
        t0_res = SiliconThermalImpedanceEvaluator.evaluate_transient_thermal_impedance(
            time_seconds=0.0, power_watts=28.0, ambient_temp_c=25.0
        )
        self.assertEqual(t0_res["transient_zth_k_w"], 0.0)
        self.assertEqual(t0_res["temperature_rise_c"], 0.0)
        self.assertEqual(t0_res["junction_temperature_c"], 25.0)

        # Test pri t = 5 s (prechodový stav)
        t5_res = SiliconThermalImpedanceEvaluator.evaluate_transient_thermal_impedance(
            time_seconds=5.0, power_watts=28.0, ambient_temp_c=25.0
        )
        self.assertGreater(t5_res["transient_zth_k_w"], 0.0)
        self.assertGreater(t5_res["junction_temperature_c"], 25.0)
        self.assertIn(t5_res["thermal_state"], ["NOMINAL_ACTIVE", "ELEVATED_HEAVY_LOAD", "COOL_IDLE", "CRITICAL_THROTTLE"])
        self.assertEqual(t5_res["vital_max_hp"], 6)

    def test_08_cognitive_flow_impedance(self):
        """Verifies cognitive flow impedance, memory contention inductance, and bounded strain."""
        # Nízka zložitosť promptu
        cog_low = CognitiveFlowImpedanceEvaluator.evaluate_cognitive_flow(
            prompt_complexity_score=1.5,
            model_tier="local_fast",
            active_memory_load_pct=20.0
        )
        self.assertLess(cog_low["normalized_strain_index"], 0.5)
        self.assertIn("OPTIMAL", cog_low["recommendation"])

        # Vysoká zložitosť promptu a pamäťové preťaženie
        cog_high = CognitiveFlowImpedanceEvaluator.evaluate_cognitive_flow(
            prompt_complexity_score=8.5,
            model_tier="cloud_frontier",
            active_memory_load_pct=95.0
        )
        self.assertGreater(cog_high["normalized_strain_index"], 0.5)
        self.assertLess(cog_high["normalized_strain_index"], 1.0)  # Bounded [0, 1)

    # ─── 4. MCP PROTOCOL INTEGRATION ─────────────────────────────────────────

    def test_09_mcp_krystal_read_grams_tool(self):
        """Verifies krystal-read-grams execution through the Hybrid MCP Connector."""
        # 1. N-Gram mode
        res_ng = self.mcp_conn.execute_tool("krystal-read-grams", {
            "mode": "ngram",
            "payload": {"text": "alchymia krystal alchymia", "n": 1}
        })
        self.assertEqual(res_ng["total_ngrams"], 3)
        self.assertEqual(res_ng["unique_ngrams"], 2)

        # 2. Gramian mode
        res_gm = self.mcp_conn.execute_tool("krystal-read-grams", {
            "mode": "gramian",
            "payload": {"vectors": [[1.0, 0.0], [0.0, 1.0]], "normalize": True}
        })
        self.assertEqual(res_gm["matrix_dimension"], 2)
        self.assertAlmostEqual(res_gm["gram_determinant"], 1.0, places=5)

    def test_10_mcp_krystal_evaluate_impedance_tool(self):
        """Verifies krystal-evaluate-impedance execution through the Hybrid MCP Connector."""
        # Electrical domain
        res_el = self.mcp_conn.execute_tool("krystal-evaluate-impedance", {
            "domain": "electrical",
            "payload": {"resistance": 75.0, "inductance_h": 2e-6, "frequency_hz": 1e6}
        })
        self.assertEqual(res_el["resistance_ohms"], 75.0)
        self.assertGreater(res_el["impedance_magnitude_ohms"], 75.0)

        # Thermal domain
        res_th = self.mcp_conn.execute_tool("krystal-evaluate-impedance", {
            "domain": "thermal",
            "payload": {"time_seconds": 1.0, "power_watts": 35.0}
        })
        self.assertEqual(res_th["power_watts"], 35.0)
        self.assertGreater(res_th["junction_temperature_c"], 25.0)

    def test_11_mcp_tool_catalog_enumeration(self):
        """Verifies that all 14 tools are registered and enumerate properly."""
        tools = self.mcp_conn.mcp_tool_schemas
        tool_names = [t["name"] for t in tools]
        self.assertEqual(len(tools), 16)
        self.assertIn("krystal-read-grams", tool_names)
        self.assertIn("krystal-evaluate-impedance", tool_names)
        self.assertIn("krystal-unified-orchestrate", tool_names)
        self.assertIn("krystal-karaf-evaluate", tool_names)


if __name__ == "__main__":
    unittest.main()
