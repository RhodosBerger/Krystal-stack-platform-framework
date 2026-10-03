import unittest
from krystal_web_hub.economic_engine import (
    CosmologicalPlane, COSMOLOGICAL_PLANE_DATA,
    BotTransmitter, NocturnalAtmosphereEngine,
    EntropyWeatherEngine, ArborMycorrhizalNetwork,
    DimensionalPortalsAndMirrors, TWELVE_APOSTLES,
    ANGELIC_GUARDIANS, ApostlesAndAngelsRegistry
)

class TestJanetTransmittersAndCosmology(unittest.TestCase):

    def setUp(self):
        self.bot = BotTransmitter("bot_explorer_01", channel_freq_mhz=433.92)

    # -------------------------------------------------------------------------
    # 1. BOT TRANSMITTER & PREMIUM TRAVEL STEP TESTS
    # -------------------------------------------------------------------------
    def test_bot_transmitter_step_and_rewind(self):
        self.assertEqual(self.bot.step_index, 0)
        self.assertEqual(self.bot.position, (0.0, 0.0, 0.0))

        # Step 1
        res1 = self.bot.step_forward((1.5, 0.0, 2.0))
        self.assertTrue(res1["success"])
        self.assertEqual(self.bot.step_index, 1)
        self.assertEqual(self.bot.position, (1.5, 0.0, 2.0))

        # Step 2
        res2 = self.bot.step_forward((0.5, 1.0, -1.0))
        self.assertTrue(res2["success"])
        self.assertEqual(self.bot.step_index, 2)
        self.assertEqual(self.bot.position, (2.0, 1.0, 1.0))

        # Rewind to Step 1
        rew1 = self.bot.rewind_step()
        self.assertTrue(rew1["success"])
        self.assertEqual(self.bot.step_index, 1)
        self.assertEqual(self.bot.position, (1.5, 0.0, 2.0))

        # Rewind to Step 0
        rew2 = self.bot.rewind_step()
        self.assertTrue(rew2["success"])
        self.assertEqual(self.bot.step_index, 0)
        self.assertEqual(self.bot.position, (0.0, 0.0, 0.0))

        # Rewind on empty stack fails gracefully
        rew_empty = self.bot.rewind_step()
        self.assertFalse(rew_empty["success"])
        self.assertEqual(rew_empty["error"], "HISTORY_STACK_EMPTY")

    def test_bot_transmitter_teleport_and_god_mode(self):
        tp_res = self.bot.teleport_bot((50.0, 120.0, -30.0), CosmologicalPlane.AETHERIC_SKY)
        self.assertTrue(tp_res["success"])
        self.assertEqual(self.bot.position, (50.0, 120.0, -30.0))
        self.assertEqual(self.bot.plane, CosmologicalPlane.AETHERIC_SKY)

        # Toggle God Mode
        gm = self.bot.toggle_god_mode()
        self.assertTrue(gm)
        self.assertEqual(self.bot.active_ward, 999)
        self.assertEqual(self.bot.active_mana, 999)

        # Toggle off
        gm_off = self.bot.toggle_god_mode()
        self.assertFalse(gm_off)
        self.assertEqual(self.bot.active_ward, 6)

    # -------------------------------------------------------------------------
    # 2. NOCTURNAL SKY & SPIRIT ENTITY TESTS
    # -------------------------------------------------------------------------
    def test_nocturnal_sky_optics_and_spirits(self):
        sky = NocturnalAtmosphereEngine.calculate_nocturnal_sky_optics(celestial_time_sec=12.5, lunar_phase=0.75)
        self.assertGreater(sky["lunar_intensity"], 0.0)
        self.assertIn("volumetric_spirit_rays", sky)
        self.assertGreaterEqual(sky["volumetric_spirit_rays"]["count"], 1)
        self.assertTrue(sky["aurora_borealis_ribbon_color"].startswith("#"))
        self.assertEqual(len(sky["spectral_spirit_entities"]), 3)
        self.assertEqual(sky["spectral_spirit_entities"][0]["name"], "Bludné Svetielko (Will-o'-the-Wisp)")

    # -------------------------------------------------------------------------
    # 3. WEATHER ENTROPY & CATASTROPHE TESTS
    # -------------------------------------------------------------------------
    def test_entropy_weather_states(self):
        # Calm starry night
        calm = EntropyWeatherEngine.calculate_weather_entropy(0.0, 10.0, 5.0)
        self.assertEqual(calm["weather_state"], "CLEAR_STARRY_NIGHT")
        self.assertFalse(calm["is_catastrophe_active"])
        self.assertEqual(calm["hazard_damage_per_turn"], 0)

        # Severe cataclysm rift
        storm = EntropyWeatherEngine.calculate_weather_entropy(95.0, 95.0, 95.0)
        self.assertEqual(storm["weather_state"], "AETHERIC_CATACLYSM_RIFT")
        self.assertTrue(storm["is_catastrophe_active"])
        self.assertEqual(storm["hazard_damage_per_turn"], 3)

    # -------------------------------------------------------------------------
    # 4. ARBOR NETWORK, LAKE MIRRORS & CAVERNS TESTS
    # -------------------------------------------------------------------------
    def test_arbor_root_network(self):
        net = ArborMycorrhizalNetwork.evaluate_root_network(connected_nodes_count=4)
        self.assertEqual(net["network_status"], "INTERTWINED_HEALTHY")
        self.assertEqual(net["active_tree_nodes_count"], 4)
        self.assertGreater(net["total_subterranean_mana_pool"], 100)
        self.assertGreater(net["collective_ward_barrier"], 5)
        self.assertEqual(len(net["mycorrhizal_channels"]), 4)

    def test_lake_mirrors_and_caverns(self):
        portals = DimensionalPortalsAndMirrors.get_portals()
        self.assertEqual(len(portals), 4)
        mirror_portal = portals[0]
        self.assertEqual(mirror_portal["portal_id"], "lake_mirror_portal_01")
        self.assertIn("Lake Mirror", mirror_portal["name"])

        cavern = DimensionalPortalsAndMirrors.inspect_cavern_depth(cavern_depth_meters=50.0)
        self.assertEqual(cavern["cavern_depth_m"], 50.0)
        self.assertEqual(cavern["darkness_factor"], 0.50)
        self.assertGreater(cavern["acoustic_echo_delay_ms"], 0)

    # -------------------------------------------------------------------------
    # 5. COSMOLOGICAL PLANES HIERARCHY TESTS
    # -------------------------------------------------------------------------
    def test_cosmological_planes_tiers(self):
        self.assertEqual(len(COSMOLOGICAL_PLANE_DATA), 5)
        self.assertEqual(COSMOLOGICAL_PLANE_DATA[CosmologicalPlane.ABYSSAL_INFERNO]["tier"], -2)
        self.assertEqual(COSMOLOGICAL_PLANE_DATA[CosmologicalPlane.SUBTERRANEAN_CAVERNS]["tier"], -1)
        self.assertEqual(COSMOLOGICAL_PLANE_DATA[CosmologicalPlane.MORTAL_TERRESTRIAL]["tier"], 0)
        self.assertEqual(COSMOLOGICAL_PLANE_DATA[CosmologicalPlane.AETHERIC_SKY]["tier"], 1)
        self.assertEqual(COSMOLOGICAL_PLANE_DATA[CosmologicalPlane.EMPYREAN_HEAVEN]["tier"], 2)

    # -------------------------------------------------------------------------
    # 6. TWELVE APOSTLES & ANGELS TESTS
    # -------------------------------------------------------------------------
    def test_twelve_apostles_registry(self):
        apostles = ApostlesAndAngelsRegistry.get_apostles()
        self.assertEqual(len(apostles), 12)

        # Verify Peter (Cephas)
        peter = ApostlesAndAngelsRegistry.get_apostle_by_id("apostle_peter_01")
        self.assertIsNotNone(peter)
        self.assertEqual(peter["name"], "Šimon Peter")
        self.assertEqual(peter["warhammer_stats"]["W"], 6)
        self.assertEqual(peter["warhammer_stats"]["Sv"], "2+")
        self.assertGreater(peter["duel_stats"]["toughness"], 30)
        self.assertIn("Pevná Skala", peter["signature_divine_spell"]["name"])

        # Verify Jude Thaddaeus
        jude = ApostlesAndAngelsRegistry.get_apostle_by_id("apostle_jude_thaddaeus_10")
        self.assertIsNotNone(jude)
        self.assertIn("Zázrak v Poslednej Sekunde", jude["signature_divine_spell"]["name"])

        # Verify Angels
        angels = ApostlesAndAngelsRegistry.get_angels()
        self.assertEqual(len(angels), 5)
        michael = angels[0]
        self.assertEqual(michael["id"], "archangel_michael")
        self.assertIn("Plamenný Meč", michael["weapon"])

if __name__ == '__main__':
    unittest.main()
