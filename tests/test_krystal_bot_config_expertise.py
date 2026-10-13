#!/usr/bin/env python3
"""
==============================================================================
UNIT TESTS: KRYSTAL BOT CONFIGURATION CHEATSHEET EXPERTISE & CHATBOT
==============================================================================
Module: tests/test_krystal_bot_config_expertise.py
Description:
    Validates that the LLM / SLM bot and chatbot system has complete cheatsheet
    expertise across all alternative valid configurations:
      - 6 Canonical Configuration Profiles & Trade-Offs
      - 9 Coordination Templates (including 4 Next-Gen Algorithms)
      - 8 Cascade Execution Strategies & Multi-Model Routing
      - 8 Categorized Linux Hardware Allocation Domains
      - Conversational Chatbot Queries & Natural Language Routing
      - REST API endpoints under /api/bot/
      - Strict Invariant Verification: assert VITAL_MAX_HP == 6
==============================================================================
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from krystal_kernel.scene_constructor_presets import VITAL_MAX_HP
from krystal_bot.config_cheatsheet_expert import CONFIG_CHEATSHEET_EXPERT, ConfigCheatsheetExpert
from krystal_bot.service import BotService
from krystal_bot import api


class TestConfigCheatsheetExpert(unittest.TestCase):
    """Tests the core logic of ConfigCheatsheetExpert."""

    def setUp(self):
        assert VITAL_MAX_HP == 6
        self.expert = CONFIG_CHEATSHEET_EXPERT

    def test_vital_invariant(self):
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_list_profiles_cheatsheet(self):
        sheet = self.expert.list_profiles_cheatsheet()
        self.assertIn("REALTIME_LOW_LATENCY", sheet)
        self.assertIn("BALANCED_DEVELOPMENT", sheet)
        self.assertIn("ECO_THERMAL_PRESERVATION", sheet)
        self.assertIn("SHADOW_COUNTERFACTUAL_RESEARCH", sheet)
        self.assertIn("MISSION_CONTROL_STUDIO_KIOSK", sheet)
        self.assertIn("HEADLESS_CONTAINER_CI", sheet)
        self.assertIn("assert VITAL_MAX_HP == 6", sheet)

    def test_get_profile_details(self):
        # Valid profile
        details = self.expert.get_profile_details("REALTIME_LOW_LATENCY")
        self.assertIn("PROFIL: REALTIME_LOW_LATENCY", details)
        self.assertIn("1.0 ms", details)
        self.assertIn("assert VITAL_MAX_HP == 6", details)

        # Fuzzy match
        details_eco = self.expert.get_profile_details("eco")
        self.assertIn("ECO_THERMAL_PRESERVATION", details_eco)

        # Unknown profile
        unknown = self.expert.get_profile_details("NON_EXISTENT_PROFILE_XYZ")
        self.assertIn("Neznámy profil", unknown)

    def test_compare_profiles(self):
        comp = self.expert.compare_profiles("REALTIME_LOW_LATENCY", "BALANCED_DEVELOPMENT")
        self.assertIn("POROVNANIE PROFILOV", comp)
        self.assertIn("1.0 ms", comp)
        self.assertIn("10.0 ms", comp)
        self.assertIn("VITAL_MAX_HP == 6", comp)

    def test_recommend_profile(self):
        # Battery / thermal preservation
        rec_battery = self.expert.recommend_profile("potrebujem šetriť batériu a uchladiť procesor")
        self.assertIn("ECO_THERMAL_PRESERVATION", rec_battery)

        # Real-time / low latency
        rec_rt = self.expert.recommend_profile("chcem sub-millisecond latenciu a realtime odozvu pre bojovú arénu")
        self.assertIn("REALTIME_LOW_LATENCY", rec_rt)

        # Kiosk / studio
        rec_kiosk = self.expert.recommend_profile("nastavujem kiosk pre localhost Mission Control studio s HUD displejom")
        self.assertIn("MISSION_CONTROL_STUDIO_KIOSK", rec_kiosk)

        # CI / Docker
        rec_ci = self.expert.recommend_profile("bežím na Linux runneri v Docker kontajneri bez displeja")
        self.assertIn("HEADLESS_CONTAINER_CI", rec_ci)

        # Shadow / safe
        rec_shadow = self.expert.recommend_profile("chcem bezpečne simulovať hypotézy bez zápisu a bez mutácií")
        self.assertIn("SHADOW_COUNTERFACTUAL_RESEARCH", rec_shadow)

    def test_validate_configuration_dict(self):
        # Valid configuration
        valid_cfg = {
            "vital_max_hp": 6,
            "target_cadence_ms": 1.0,
            "target_pl1_watts": 32.0,
            "cpu_affinity_mask": 0x000F
        }
        ok, errs = self.expert.validate_configuration_dict(valid_cfg)
        self.assertTrue(ok)
        self.assertEqual(len(errs), 0)

        # Invalid: vital_max_hp != 6
        invalid_hp = dict(valid_cfg, vital_max_hp=5)
        ok_hp, errs_hp = self.expert.validate_configuration_dict(invalid_hp)
        self.assertFalse(ok_hp)
        self.assertTrue(any("vital_max_hp" in e for e in errs_hp))

        # Invalid: dangerous PL1 wattage
        invalid_pl1 = dict(valid_cfg, target_pl1_watts=120.0)
        ok_pl1, errs_pl1 = self.expert.validate_configuration_dict(invalid_pl1)
        self.assertFalse(ok_pl1)
        self.assertTrue(any("PL1" in e for e in errs_pl1))

    def test_list_templates_cheatsheet(self):
        sheet = self.expert.list_templates_cheatsheet()
        self.assertIn("TEMPLATE_ULTRA_LOW_JITTER_REALTIME", sheet)
        self.assertIn("TEMPLATE_BALANCED_ADAPTIVE_CADENCE", sheet)
        self.assertIn("TEMPLATE_NEXTGEN_SYMPLECTIC_TENSOR_HARMONY", sheet)
        self.assertIn("TEMPLATE_NEXTGEN_HETEROGENEOUS_ROOFLINE_DISPATCH", sheet)
        self.assertIn("TEMPLATE_NEXTGEN_EPISTEMIC_CALIBRATION_LEDGER", sheet)
        self.assertIn("TEMPLATE_NEXTGEN_MULTI_TENANT_ISOLATION_SLICE", sheet)

    def test_get_template_details(self):
        det = self.expert.get_template_details("TEMPLATE_NEXTGEN_SYMPLECTIC_TENSOR_HARMONY")
        self.assertIn("TEMPLATE_NEXTGEN_SYMPLECTIC_TENSOR_HARMONY", det)
        self.assertIn("Golden Ratio Φ", det)
        self.assertIn("VITAL_MAX_HP == 6", det)

    def test_list_cascade_cheatsheet(self):
        sheet = self.expert.list_cascade_cheatsheet()
        self.assertIn("DIRECT_LOCAL", sheet)
        self.assertIn("SPECULATIVE_VERIFY", sheet)
        self.assertIn("assert VITAL_MAX_HP == 6", sheet)

    def test_get_hardware_cheatsheet(self):
        sheet = self.expert.get_hardware_cheatsheet()
        self.assertIsInstance(sheet, str)
        self.assertTrue(len(sheet) > 50)
        self.assertIn("VITAL_MAX_HP == 6", sheet)

    def test_answer_configuration_chat(self):
        # Recommendation question
        ans_rec = self.expert.answer_configuration_chat("akú konfiguráciu odporúčaš na šetrenie batérie?")
        self.assertIn("ECO_THERMAL_PRESERVATION", ans_rec)

        # Comparison question
        ans_comp = self.expert.answer_configuration_chat("porovnaj REALTIME_LOW_LATENCY a BALANCED_DEVELOPMENT")
        self.assertIn("POROVNANIE PROFILOV", ans_comp)

        # Template question
        ans_tmpl = self.expert.answer_configuration_chat("aké sú koordinačné templáty v systéme?")
        self.assertIn("TEMPLATE_ULTRA_LOW_JITTER_REALTIME", ans_tmpl)

        # Hardware question
        ans_hw = self.expert.answer_configuration_chat("aká je alokácia hardware prostriedkov a monitora?")
        self.assertIn("VITAL_MAX_HP == 6", ans_hw)


class TestBotServiceSkills(unittest.TestCase):
    """Tests the BotService with newly registered skills and chat routing."""

    def setUp(self):
        assert VITAL_MAX_HP == 6
        self.td = tempfile.mkdtemp(prefix="krystal_bot_skills_test_")
        self.svc = BotService(self.td, use_keras=False)

    def tearDown(self):
        self.svc.shutdown()
        shutil.rmtree(self.td, ignore_errors=True)

    def test_skills_registration(self):
        skill_names = {s["name"] for s in self.svc.gateway.skills()}
        self.assertIn("config", skill_names)
        self.assertIn("template", skill_names)
        self.assertIn("cascade", skill_names)
        self.assertIn("hw", skill_names)
        self.assertIn("ask", skill_names)

    def test_config_command_dispatch(self):
        # /config list
        res_list = self.svc.message("/config list")
        self.assertTrue(res_list["ok"])
        self.assertIn("REALTIME_LOW_LATENCY", res_list["reply"])

        # /config get
        res_get = self.svc.message("/config get REALTIME_LOW_LATENCY")
        self.assertTrue(res_get["ok"])
        self.assertIn("1.0 ms", res_get["reply"])

        # /config compare
        res_comp = self.svc.message("/config compare REALTIME_LOW_LATENCY ECO_THERMAL_PRESERVATION")
        self.assertTrue(res_comp["ok"])
        self.assertIn("POROVNANIE PROFILOV", res_comp["reply"])

        # /config recommend
        res_rec = self.svc.message("/config recommend bateria a nizka spotreba")
        self.assertTrue(res_rec["ok"])
        self.assertIn("ECO_THERMAL_PRESERVATION", res_rec["reply"])

        # /config validate
        res_val = self.svc.message("/config validate REALTIME_LOW_LATENCY")
        self.assertTrue(res_val["ok"])
        self.assertIn("VITAL_MAX_HP == 6 overený", res_val["reply"])

    def test_template_command_dispatch(self):
        # /template list
        res_list = self.svc.message("/template list")
        self.assertTrue(res_list["ok"])
        self.assertIn("TEMPLATE_ULTRA_LOW_JITTER_REALTIME", res_list["reply"])
        self.assertIn("TEMPLATE_NEXTGEN_SYMPLECTIC_TENSOR_HARMONY", res_list["reply"])

        # /template get
        res_get = self.svc.message("/template get TEMPLATE_BALANCED_ADAPTIVE_CADENCE")
        self.assertTrue(res_get["ok"])
        self.assertIn("10.0 ms", res_get["reply"])

        # /template dispatch
        res_disp = self.svc.message("/template dispatch TEMPLATE_BALANCED_ADAPTIVE_CADENCE")
        self.assertTrue(res_disp["ok"])
        self.assertIn("VITAL_MAX_HP == 6", res_disp["reply"])

    def test_cascade_command_dispatch(self):
        # /cascade list
        res_list = self.svc.message("/cascade list")
        self.assertTrue(res_list["ok"])
        self.assertIn("DIRECT_LOCAL", res_list["reply"])

        # /cascade run
        res_run = self.svc.message("/cascade run Otestuj invariant VITAL_MAX_HP == 6")
        self.assertTrue(res_run["ok"])
        self.assertIn("VITAL_MAX_HP == 6", res_run["reply"])

    def test_hw_command_dispatch(self):
        res_hw = self.svc.message("/hw summary")
        self.assertTrue(res_hw["ok"])
        self.assertIn("VITAL_MAX_HP == 6", res_hw["reply"])

    def test_natural_language_fallback_to_cheatsheet_expertise(self):
        # Sending a message without slash command that asks about config
        res = self.svc.message("akú konfiguráciu odporúčaš na šetrenie batérie a chladenie?")
        self.assertTrue(res["ok"])
        self.assertIn("ECO_THERMAL_PRESERVATION", res["reply"])
        self.assertIn("assert VITAL_MAX_HP == 6", res["reply"])


class TestBotApiEndpoints(unittest.TestCase):
    """Tests the HTTP surface under /api/bot/ for the new config cheatsheet features."""

    def setUp(self):
        assert VITAL_MAX_HP == 6
        self.td = tempfile.mkdtemp(prefix="krystal_bot_api_test_")
        self.svc = BotService(self.td, use_keras=False)

    def tearDown(self):
        self.svc.shutdown()
        shutil.rmtree(self.td, ignore_errors=True)

    def _call(self, method: str, path: str, body: dict = None) -> tuple:
        raw_body = json.dumps(body).encode("utf-8") if body is not None else b""
        headers = {"Content-Type": "application/json"}
        return api.handle(self.svc, method, path, headers, raw_body)

    def test_get_configs(self):
        status, headers, body = self._call("GET", "/api/bot/configs")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["status"], "OK")
        self.assertIn("REALTIME_LOW_LATENCY", data["profiles"])
        self.assertEqual(data["vital_max_hp"], 6)

    def test_get_templates(self):
        status, headers, body = self._call("GET", "/api/bot/templates")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["status"], "OK")
        self.assertGreaterEqual(len(data["templates"]), 5)
        self.assertEqual(data["vital_max_hp"], 6)

    def test_get_cascade(self):
        status, headers, body = self._call("GET", "/api/bot/cascade")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["status"], "OK")
        self.assertEqual(data["vital_max_hp"], 6)

    def test_get_hw(self):
        status, headers, body = self._call("GET", "/api/bot/hw")
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["status"], "OK")
        self.assertEqual(data["vital_max_hp"], 6)

    def test_post_config_recommend(self):
        status, headers, body = self._call("POST", "/api/bot/config/recommend", {"query": "chcem maximálny výkon na P-Cores"})
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["status"], "OK")
        self.assertIn("REALTIME_LOW_LATENCY", data["recommendation"])
        self.assertEqual(data["vital_max_hp"], 6)

    def test_post_config_validate(self):
        # Valid config
        status, headers, body = self._call("POST", "/api/bot/config/validate", {"config": {"vital_max_hp": 6, "target_cadence_ms": 1.0}})
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertTrue(data["valid"])

        # Invalid config
        status_bad, headers_bad, body_bad = self._call("POST", "/api/bot/config/validate", {"config": {"vital_max_hp": 999}})
        self.assertEqual(status_bad, 400)
        data_bad = json.loads(body_bad.decode("utf-8"))
        self.assertFalse(data_bad["valid"])

    def test_post_template_dispatch(self):
        status, headers, body = self._call("POST", "/api/bot/template/dispatch", {"template_id": "TEMPLATE_BALANCED_ADAPTIVE_CADENCE"})
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["vital_max_hp"], 6)

    def test_post_cascade_run(self):
        status, headers, body = self._call("POST", "/api/bot/cascade/run", {"prompt": "Generuj odozvu", "objective": "BALANCED"})
        self.assertEqual(status, 200)
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["status"], "OK")
        self.assertEqual(data["vital_max_hp"], 6)


if __name__ == "__main__":
    unittest.main()
