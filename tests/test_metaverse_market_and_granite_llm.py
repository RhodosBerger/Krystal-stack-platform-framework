import unittest
import os
import time

from krystal_web_hub.economic_engine.metaverse_market_and_granite_llm import (
    MetaverseAssetType,
    OrderType,
    MetaverseMarketplaceEngine,
    GraniteAndEdgeLLMEngine
)
from krystal_janet.janet_bridge import JanetValidator


class TestMetaverseMarketAndGraniteLLM(unittest.TestCase):
    """
    Unit test suite verifying:
    1. Metaverse Marketplace Engine (Order Book matching & AMM Constant Product Swaps)
    2. Max 2B parameter edge LLM budgeting & 4B IBM Granite 2026 models
    3. Market sentiment analysis and structured Granite reasoning
    4. Autonomous NPC merchant barter negotiation
    5. Anti-fraud / wash-trading detection
    6. Janet DSL syntax and validation
    """

    def setUp(self):
        self.market = MetaverseMarketplaceEngine()
        self.llm = GraniteAndEdgeLLMEngine()

    # ── 1. METAVERSE MARKETPLACE ENGINE ───────────────────────────────────────
    def test_catalog_assets_and_types(self):
        catalog = self.market.get_catalog()
        self.assertGreaterEqual(len(catalog), 5)
        asset_types = {item["asset_type"] for item in catalog}
        self.assertIn(MetaverseAssetType.VIRTUAL_LAND.value, asset_types)
        self.assertIn(MetaverseAssetType.AVATAR_COSMETIC.value, asset_types)
        self.assertIn(MetaverseAssetType.COMPUTE_EPOCH.value, asset_types)

    def test_order_book_bids_asks_and_matching(self):
        pair = "LAND_042/AET"
        ob = self.market.get_order_book(pair)
        self.assertGreater(ob["best_bid"], 0)
        self.assertGreater(ob["best_ask"], ob["best_bid"])
        self.assertGreater(ob["spread_percentage"], 0.0)

        # Place a matching limit buy order at ask price
        match_price = ob["best_ask"]
        res = self.market.place_order(
            pair=pair,
            order_type=OrderType.LIMIT_BUY.value,
            price=match_price,
            amount=1.0,
            trader="buyer_test"
        )
        self.assertTrue(res["success"])
        self.assertTrue(res["fully_filled"])
        self.assertGreaterEqual(res["executed_trades_count"], 1)

    def test_amm_constant_product_swap_and_slippage(self):
        pool_id = "pool_aet_gold"
        pools = self.market.get_amm_pools()
        pool = next(p for p in pools if p["pool_id"] == pool_id)
        k_before = pool["k_constant"]

        # Swap 10 AET for GOLD
        swap_res = self.market.execute_amm_swap(
            pool_id=pool_id,
            token_in="AET",
            amount_in=10.0,
            slippage_tolerance=0.05
        )
        self.assertTrue(swap_res["success"])
        self.assertEqual(swap_res["token_out"], "GOLD")
        self.assertGreater(swap_res["amount_out"], 80.0)

        # Verify slippage protection rejects huge swap relative to pool
        fail_swap = self.market.execute_amm_swap(
            pool_id=pool_id,
            token_in="AET",
            amount_in=8000.0,
            slippage_tolerance=0.01  # Tight 1% tolerance
        )
        self.assertFalse(fail_swap["success"])
        self.assertIn("Slippage limit", fail_swap["error"])

    # ── 2. 2B PARAMETER & 4B IBM GRANITE 2026 LLM EVALUATION ─────────────────
    def test_registered_models_and_budget_constraints(self):
        models = self.llm.get_registered_models()
        self.assertGreaterEqual(len(models), 4)

        # Verify 2B model constraint (<= 2B parameters)
        b2b = self.llm.verify_hardware_budget("ibm-granite-3.0-2b-instruct")
        self.assertTrue(b2b["parameter_constraint_satisfied"])
        self.assertLessEqual(b2b["parameters_exact"], 2000000000)
        self.assertTrue(b2b["fits_in_edge_vram"])
        self.assertEqual(b2b["hardware_verdict"], "APPROVED_FOR_EDGE_METAVERSE_RUNTIME")

        # Verify 4B IBM Granite 2026 model
        b4b = self.llm.verify_hardware_budget("ibm-granite-3.1-4b-instruct-2026")
        self.assertTrue(b4b["parameter_constraint_satisfied"])
        self.assertLessEqual(b4b["parameters_exact"], 4200000000)
        self.assertGreaterEqual(b4b["context_window"], 65536)
        self.assertEqual(b4b["hardware_verdict"], "APPROVED_FOR_EDGE_METAVERSE_RUNTIME")

    def test_market_sentiment_analysis(self):
        pair = "LAND_042/AET"
        ob = self.market.get_order_book(pair)
        eval_res = self.llm.evaluate_market_sentiment(pair, ob, "ibm-granite-3.1-4b-instruct-2026")

        self.assertEqual(eval_res["pair"], pair)
        self.assertIn(eval_res["sentiment"], ["BULLISH", "BEARISH", "NEUTRAL_CONSOLIDATING"])
        self.assertGreaterEqual(eval_res["confidence_score_pct"], 50.0)
        self.assertIn("IBM Granite 2026 Reasoning", eval_res["granite_structured_reasoning"])

    def test_npc_merchant_barter_simulation(self):
        # 1. Fair/Generous offer should be ACCEPTED
        accept_res = self.llm.simulate_merchant_barter(
            offered_item="Kryštálový Meč",
            offered_nominal_val=150.0,
            requested_item="Aéterová Batéria",
            requested_nominal_val=100.0,
            merchant_archetype="Aéterový Kováč",
            greed_factor=0.1,
            model_key="ibm-granite-3.0-2b-instruct"
        )
        self.assertEqual(accept_res["verdict"], "ACCEPTED")
        self.assertIn("Prijímam výmenu", accept_res["merchant_dialogue"])

        # 2. Low-ball offer should be REJECTED
        reject_res = self.llm.simulate_merchant_barter(
            offered_item="Hrdzavý Nôž",
            offered_nominal_val=20.0,
            requested_item="Pozemok #042",
            requested_nominal_val=450.0,
            merchant_archetype="Pozemkový Maklér",
            model_key="ibm-granite-3.1-4b-instruct-2026"
        )
        self.assertEqual(reject_res["verdict"], "REJECTED")

        # 3. Near-fair offer should receive COUNTER_OFFER
        counter_res = self.llm.simulate_merchant_barter(
            offered_item="Kryštálové Brnenie",
            offered_nominal_val=95.0,
            requested_item="Aéterová Batéria",
            requested_nominal_val=100.0,
            greed_factor=0.1
        )
        self.assertEqual(counter_res["verdict"], "COUNTER_OFFER")
        self.assertIsNotNone(counter_res["counter_offer_aet"])

    def test_market_manipulation_detection(self):
        # Suspicious events with self-trades and rapid cancels
        suspicious_events = [
            {"action": "CANCEL", "lifespan_sec": 0.4, "trader": "bot_1"},
            {"action": "CANCEL", "lifespan_sec": 0.3, "trader": "bot_1"},
            {"action": "CANCEL", "lifespan_sec": 0.5, "trader": "bot_1"},
            {"buyer": "whale_wash", "seller": "whale_wash", "amount": 100, "price": 50}
        ]
        audit = self.llm.detect_market_manipulation(suspicious_events)
        self.assertTrue(audit["wash_trading_flag"])
        self.assertTrue(audit["spoofing_flag"])
        self.assertGreaterEqual(audit["risk_score_pct"], 60)
        self.assertIn("MANIPULATION_DETECTED", audit["audit_verdict"])

    # ── 3. JANET DSL VALIDATION ───────────────────────────────────────────────
    def test_janet_metaverse_market_script_syntax(self):
        janet_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "krystal_janet",
            "metaverse_market_granite.janet"
        )
        self.assertTrue(os.path.exists(janet_path))
        val_res = JanetValidator.validate_file(janet_path)
        self.assertTrue(val_res["valid"], f"Validation error: {val_res.get('error')}")
        self.assertGreaterEqual(val_res["line_count"], 40)


if __name__ == '__main__':
    unittest.main()
