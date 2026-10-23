#!/usr/bin/env python3
# ==============================================================================
# TESTS FOR VPP GRID BROKER FLEET COORDINATOR & SWITCHED BATTERY MATRIX
# ==============================================================================
# Target Invariant: VITAL_MAX_HP == 6
# Standard: EN 50160 (230 V +/- 10% -> 207 V <= V <= 253 V)
# ==============================================================================

import unittest
from krystal_kernel.vpp_grid_broker_fleet import (
    SwitchedBatteryMatrixController,
    SubstationFeederModel,
    VppGridBrokerFleetCoordinator,
    BatteryTopology,
    VppFleetMode,
    VITAL_MAX_HP,
    VOLTAGE_MAX_ALLOWED_V,
    VOLTAGE_NOMINAL_V
)


class TestVppGridBrokerFleet(unittest.TestCase):
    """Verifies VPP fleet coordination, 253V phased array protection, and SiC MOSFET switching."""

    def test_01_constitutional_invariant_holds(self):
        """Constitutional invariant must strictly equal 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_sic_mosfet_switched_matrix_transition_and_loss_reduction(self):
        """Switching to 800V series DC bus must reduce I^2*R cable losses by ~277x."""
        ctrl = SwitchedBatteryMatrixController()
        self.assertEqual(ctrl.topology, BatteryTopology.PARALLEL_48V)

        # 1. Výpočet strát pre 15 kW pri 48V
        curr_48v, vdrop_48v, loss_48v = SwitchedBatteryMatrixController.calculate_cable_loss(
            power_w=15000.0,
            topology=BatteryTopology.PARALLEL_48V,
            cable_resistance_ohm=0.045
        )
        self.assertAlmostEqual(curr_48v, 312.5, places=1)
        self.assertGreater(loss_48v, 4000.0) # > 4 kW tepelných strát v kábli!

        # 2. Prepnutie do 800V GRID BOOST režimu
        res_switch = ctrl.switch_topology(BatteryTopology.SERIES_800V)
        self.assertEqual(ctrl.topology, BatteryTopology.SERIES_800V)
        self.assertTrue(res_switch["is_grid_boost_active"])
        self.assertLess(res_switch["switching_latency_ms"], 5.0) # < 5 ms zopnutie

        # 3. Výpočet strát pre 15 kW pri 800V
        curr_800v, vdrop_800v, loss_800v = SwitchedBatteryMatrixController.calculate_cable_loss(
            power_w=15000.0,
            topology=BatteryTopology.SERIES_800V,
            cable_resistance_ohm=0.045
        )
        self.assertAlmostEqual(curr_800v, 18.75, delta=0.5)
        self.assertLess(loss_800v, 20.0) # Len ~15.8 W tepelných strát!

        # 4. Pomer redukcie strát
        loss_ratio = loss_48v / loss_800v
        self.assertGreater(loss_ratio, 250.0)
        self.assertLess(loss_ratio, 300.0)
        self.assertEqual(res_switch["vital_max_hp"], 6)

    def test_03_substation_feeder_voltage_rise_modeling(self):
        """Feeder model must compute progressive voltage rise along 800m cable."""
        feeder = SubstationFeederModel(node_count=100)
        # Všetkých 100 domov exportuje po 6 kW
        active_p = [6.0] * 100
        reactive_q = [0.0] * 100

        profile = feeder.solve_voltage_profile(active_p, reactive_q, substation_voltage_v=230.0)

        self.assertEqual(len(profile), 100)
        # Napätie na začiatku vedenia blízko trafostanice (uzol 0)
        self.assertGreater(profile[0].voltage_v, 230.0)
        # Napätie musí monotónne rásť pozdĺž vedenia smerom ku koncu
        self.assertGreater(profile[99].voltage_v, profile[0].voltage_v)
        self.assertGreaterEqual(profile[99].distance_meters, 390.0)

    def test_04_phased_array_prevents_253v_overvoltage(self):
        """Phased Array coordination must keep all 100 nodes <= 253.0V during massive 0.8 MW export."""
        broker = VppGridBrokerFleetCoordinator(node_count=100)

        # Požiadavka na +0.8 MW export do siete pri vysokej spotovej cene
        dec = broker.dispatch_fleet_request(
            requested_mw=0.80,
            spot_price_eur_mwh=160.0
        )

        self.assertEqual(dec.fleet_mode, VppFleetMode.VOLTAGE_PROTECT_PHASED)
        self.assertTrue(dec.overvoltage_prevented_by_phased_array)
        # Záväzný limit EN 50160: žiadny uzol nesmie prekročiť 253.0 V!
        self.assertLessEqual(dec.max_feeder_voltage_v, VOLTAGE_MAX_ALLOWED_V)
        self.assertEqual(dec.voltage_violations_count, 0)
        # Všetky uzly musia byť v 800V sérii
        self.assertEqual(dec.switched_matrix_800v_nodes_count, 100)
        # Jalový výkon musí byť induktívny (záporný) pre stiahnutie napätia
        self.assertLess(dec.total_reactive_power_mvar, 0.0)
        # Koncové uzly museli aktivovať CLU tepelné tlmenie
        self.assertGreater(dec.total_clu_thermal_dampening_kw, 0.0)
        self.assertEqual(dec.vital_max_hp, 6)

    def test_05_afrr_down_negative_price_absorption(self):
        """Negative price event (-1.2 MW demand) must switch nodes to 48V parallel and absorb power."""
        broker = VppGridBrokerFleetCoordinator(node_count=100)

        dec = broker.dispatch_fleet_request(
            requested_mw=-1.20,
            spot_price_eur_mwh=-15.0
        )

        self.assertEqual(dec.fleet_mode, VppFleetMode.AFRR_DOWN_MAX_ABSORPTION)
        # Prepnutie do paralelného režimu pre šetrné nabíjanie (0 uzlov v 800V)
        self.assertEqual(dec.switched_matrix_800v_nodes_count, 0)
        # Absorpcia výkonu zo siete
        self.assertGreater(dec.total_active_absorption_mw, 1.0)
        # Napätie na vedení nesmie padnúť pod 207.0 V
        self.assertGreaterEqual(dec.min_feeder_voltage_v, 207.0)
        self.assertEqual(dec.voltage_violations_count, 0)
        self.assertEqual(dec.vital_max_hp, 6)

    def test_06_state_hash_determinism(self):
        """Identical dispatch requests must yield bit-exact identical state hashes."""
        b1 = VppGridBrokerFleetCoordinator(node_count=100)
        b2 = VppGridBrokerFleetCoordinator(node_count=100)

        d1 = b1.dispatch_fleet_request(requested_mw=0.65, spot_price_eur_mwh=140.0)
        d2 = b2.dispatch_fleet_request(requested_mw=0.65, spot_price_eur_mwh=140.0)

        self.assertEqual(d1.state_hash, d2.state_hash)
        self.assertEqual(d1.max_feeder_voltage_v, d2.max_feeder_voltage_v)
        self.assertEqual(d1.achieved_grid_power_mw, d2.achieved_grid_power_mw)


if __name__ == "__main__":
    unittest.main()
