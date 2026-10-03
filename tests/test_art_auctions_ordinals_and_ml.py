import unittest
import time
import os

from krystal_web_hub.economic_engine.art_auctions_ordinals_and_ml import (
    PaintingAuctionHouseEngine,
    AetherOrdinalsProtocolEngine,
    ContentReplayEngine,
    HeroMatrixEngine,
    FrameRateEncodingProtocol,
    EvolutionaryPhysicsEngine,
    EventStreamCongestionController
)
from krystal_janet.janet_bridge import JanetValidator, JanetSExpressionParser


class TestArtAuctionsOrdinalsAndML(unittest.TestCase):
    """
    Unit test suite verifying:
    1. Art/Painting Auctions & Bidding Engine
    2. Aether Ordinals Protocol & Bitcoin Witness Script Export
    3. Content Replay Engine with Unique Perceptual Level Snapshots
    4. Multi-dimensional Hero Matrices (5x2, 4x5, 30x20, 90x120)
    5. Variable Sampling Frequency Encoding Protocol
    6. Evolutionary Physics & Kinematics Engine
    7. Event Stream AIMD Congestion Controller
    8. Janet Evolutionary Encoding DSL
    """

    def setUp(self):
        self.auction_house = PaintingAuctionHouseEngine()
        self.ordinals = AetherOrdinalsProtocolEngine()
        self.replay = ContentReplayEngine()
        self.congestion = EventStreamCongestionController(initial_window=10, max_queue=25)

    # ── 1. ART AUCTIONS & BIDDING ─────────────────────────────────────────────
    def test_auction_house_lots_and_bidding(self):
        lots = self.auction_house.list_lots(status="active")
        self.assertGreaterEqual(len(lots), 3)

        new_lot = self.auction_house.create_lot(
            title="Kryštálové Hory v Hmle",
            artist="Aetheric Master",
            style="Ethereal Realism",
            rarity="Epic",
            starting_bid=100,
            buyout_price=300
        )
        lot_id = new_lot["id"]
        self.assertEqual(new_lot["current_bid"], 100)
        self.assertEqual(new_lot["status"], "active")

        # Bid below minimum should fail
        fail_bid = self.auction_house.place_bid(lot_id, "Tester1", 90)
        self.assertFalse(fail_bid["success"])

        # Valid bid
        valid_bid = self.auction_house.place_bid(lot_id, "Tester1", 150)
        self.assertTrue(valid_bid["success"])
        self.assertEqual(valid_bid["current_bid"], 150)
        self.assertEqual(valid_bid["highest_bidder"], "Tester1")

        # Buyout bid
        buyout = self.auction_house.place_bid(lot_id, "WhaleBuyer", 300)
        self.assertTrue(buyout["success"])
        self.assertTrue(buyout["buyout_triggered"])
        self.assertEqual(buyout["status"], "sold")

        # After buyout, lot is not in active lots
        active_now = [l["id"] for l in self.auction_house.list_lots(status="active")]
        self.assertNotIn(lot_id, active_now)

    # ── 2. AETHER ORDINALS PROTOCOL & WITNESS SCRIPT ─────────────────────────
    def test_ordinals_inscription_and_export(self):
        payload = "KRYSTAL-STACK GENESIS ARTIFACT // TRANSCENDENT RELIC"
        insc = self.ordinals.inscribe(
            content_payload=payload,
            content_type="text/plain;charset=utf-8",
            owner_address="bc1p_hero_collector_42"
        )
        self.assertIn("id", insc)
        self.assertEqual(insc["content_size_bytes"], len(payload.encode("utf-8")))
        self.assertTrue(len(insc["content_hash"]) == 64)

        # Export ordinal envelope
        export_data = self.ordinals.export_ordinal(insc["id"])
        self.assertIsNotNone(export_data)
        self.assertIn("witness_script_hex", export_data)
        self.assertIn("witness_asm", export_data)

        # Verify Bitcoin Ordinals envelope prefix (00 63 03 6f 72 64 = OP_0 OP_IF OP_PUSH("ord"))
        self.assertTrue(export_data["witness_script_hex"].startswith("0063036f7264"))
        self.assertTrue(export_data["witness_script_hex"].endswith("68")) # OP_ENDIF
        self.assertIn("OP_IF", export_data["witness_asm"])
        self.assertIn("ord", export_data["witness_asm"])

    # ── 3. CONTENT REPLAY & LEVEL SNAPSHOTS ──────────────────────────────────
    def test_content_replay_fragmentation_and_snapshots(self):
        m_id = "match_test_omega"
        seg1 = self.replay.record_level_segment(
            match_id=m_id,
            level_name="Les Duchov",
            level_idx=1,
            level_seed=1001,
            hero_count=3,
            action_events=[{"action": "spawn", "hero": "druid_1"}],
            visual_layers={"fog": 0.2, "sun": 0.8}
        )
        self.assertEqual(seg1["level_idx"], 1)
        self.assertIn("perceptual_snapshot", seg1)
        snap1_hash = seg1["perceptual_snapshot"]["snapshot_hash"]
        self.assertEqual(len(snap1_hash), 64)

        # Record second level
        seg2 = self.replay.record_level_segment(
            match_id=m_id,
            level_name="Kryštálová Baňa",
            level_idx=2,
            level_seed=1002,
            hero_count=3,
            action_events=[{"action": "crystal_mine", "yield": 50}],
            visual_layers={"ambient": "blue_luminescence"}
        )
        snap2_hash = seg2["perceptual_snapshot"]["snapshot_hash"]
        # Different levels must produce unique perceptual snapshot hashes
        self.assertNotEqual(snap1_hash, snap2_hash)

        # Retrieve match manifest
        manifest = self.replay.get_match_manifest(m_id)
        self.assertIsNotNone(manifest)
        self.assertEqual(manifest["total_levels"], 2)

    # ── 4. MULTI-DIMENSIONAL HERO MATRICES ────────────────────────────────────
    def test_hero_matrices_dimensions_and_scaling(self):
        # Test 5x2 (10 cells)
        m_5x2 = HeroMatrixEngine.generate_matrix_for_hero_count(hero_count=4, dim_key="5x2")
        self.assertEqual(m_5x2["rows"], 5)
        self.assertEqual(m_5x2["cols"], 2)
        self.assertEqual(len(m_5x2["matrix"]), 5)
        self.assertEqual(len(m_5x2["matrix"][0]), 2)

        # Test 4x5 (20 cells)
        m_4x5 = HeroMatrixEngine.generate_matrix_for_hero_count(hero_count=2, dim_key="4x5")
        self.assertEqual(m_4x5["rows"], 4)
        self.assertEqual(m_4x5["cols"], 5)

        # Test 30x20 (600 cells)
        m_30x20 = HeroMatrixEngine.generate_matrix_for_hero_count(hero_count=6, dim_key="30x20")
        self.assertEqual(m_30x20["rows"], 30)
        self.assertEqual(m_30x20["cols"], 20)

        # Test 90x120 (10800 cells)
        m_90x120 = HeroMatrixEngine.generate_matrix_for_hero_count(hero_count=8, dim_key="90x120")
        self.assertEqual(m_90x120["rows"], 90)
        self.assertEqual(m_90x120["cols"], 120)
        self.assertEqual(m_90x120["total_cells"], 10800)
        self.assertIn("preview_sample", m_90x120)

    # ── 5. VARIABLE SAMPLING FREQUENCY ENCODING ──────────────────────────────
    def test_framerate_encoding_protocol(self):
        # 1 hero, peaceful intensity
        sched_low = FrameRateEncodingProtocol.compute_sampling_frequency_schedule(hero_count=1, battle_intensity=0.1)
        self.assertGreaterEqual(sched_low["effective_hz"], 15)
        self.assertLessEqual(sched_low["effective_hz"], 60)

        # 12 heroes, chaotic intensity
        sched_high = FrameRateEncodingProtocol.compute_sampling_frequency_schedule(hero_count=12, battle_intensity=0.95)
        self.assertGreaterEqual(sched_high["effective_hz"], 120)
        self.assertLessEqual(sched_high["effective_hz"], 240)

        # Header packing
        header = FrameRateEncodingProtocol.pack_stream_header(total_frames=240, base_hz=120, channel_count=4)
        self.assertTrue(header["magic"].startswith("KRYS_FPS"))
        self.assertEqual(header["base_frequency_hz"], 120)
        self.assertEqual(header["total_frames"], 240)
        self.assertIn("crc32_hex", header)

    # ── 6. EVOLUTIONARY PHYSICS & KINEMATICS ──────────────────────────────────
    def test_evolutionary_physics_engine(self):
        evo = EvolutionaryPhysicsEngine(population_size=12, trajectory_steps=15)
        evo.collision_target = (10.0, 0.0, 5.0)

        initial_best = evo.get_best_individual()
        initial_fitness = initial_best["fitness"]

        # Run 4 generations
        for _ in range(4):
            gen_res = evo.evolve_generation()
            self.assertIn("best_fitness", gen_res)

        final_best = evo.get_best_individual()
        # Elitism guarantees best fitness never worsens
        self.assertGreaterEqual(final_best["fitness"], initial_fitness)
        self.assertGreater(len(final_best["trajectory"]), 0)

    # ── 7. AIMD CONGESTION CONTROLLER ────────────────────────────────────────
    def test_congestion_controller_aimd(self):
        ctrl = EventStreamCongestionController(initial_window=10, max_queue=20)

        # Push items up to threshold
        for i in range(15):
            res = ctrl.push_event({"event_id": i, "data": "metric"})
            self.assertTrue(res)

        # Push beyond capacity should trigger multiplicative decrease
        for i in range(10):
            ctrl.push_event({"overflow_id": i})

        status = ctrl.get_status()
        self.assertGreater(status["congestion_events_count"], 0)
        self.assertLess(status["window_size"], 10)

        # Dispatch batch clears items and starts additive recovery
        batch = ctrl.dispatch_batch()
        self.assertGreater(batch["dispatched_count"], 0)
        self.assertEqual(status["total_dispatched"] + batch["dispatched_count"], ctrl.total_dispatched)

    # ── 8. JANET EVOLUTIONARY ENCODING SCRIPT ────────────────────────────────
    def test_janet_evolutionary_script_syntax_and_dsl(self):
        janet_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "krystal_janet",
            "evolutionary_encoding.janet"
        )
        self.assertTrue(os.path.exists(janet_path), f"Janet file missing: {janet_path}")
        with open(janet_path, "r", encoding="utf-8") as f:
            code = f.read()

        val_res = JanetValidator.validate_file(janet_path)
        self.assertTrue(val_res["valid"], f"Validation errors: {val_res.get('error')}")
        self.assertGreater(val_res["line_count"], 50)
        self.assertIn("create-sampling-frequency-schedule", code)
        self.assertIn("encode-frame-stream-header", code)
        self.assertIn("evaluate-trajectory-fitness", code)
        self.assertIn("step-congestion-window", code)
        self.assertIn("HERO-MATRIX-DIMS", code)


if __name__ == '__main__':
    unittest.main()
