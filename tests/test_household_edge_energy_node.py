#!/usr/bin/env python3
# ==============================================================================
# TESTS FOR GRID-TO-HOUSEHOLD EDGE ENERGY NODE & ASYMMETRIC BALANCER
# ==============================================================================
# Target Invariant: VITAL_MAX_HP == 6
# System Model: (S, A, T, I, O, E, H, W)
# ==============================================================================

import unittest
from krystal_kernel.household_edge_energy_node import (
    HouseholdEdgeEnergyNode,
    PhaseLoads,
    HouseholdEdgeMode,
    HouseholdContractViolation,
    VITAL_MAX_HP,
    FRAME_SIZE_BYTES,
    MAX_POWER_PER_PHASE_W,
    BATTERY_CAPACITY_KWH
)


class TestHouseholdEdgeEnergyNode(unittest.TestCase):
    """Verifies Household Edge 128B frames, asymmetric phase balancing, and breaker protection."""

    def test_01_constitutional_invariant_holds(self):
        """Constitutional invariant must strictly equal 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_128b_frame_packing_and_alignment(self):
        """Packet frame must strictly align to 128 bytes with valid magic code."""
        frame = HouseholdEdgeEnergyNode.pack_128b_household_frame(
            mode=HouseholdEdgeMode.GRID_CUT_ISLAND,
            i_grid_cma=(0, 0, 0),
            v_grid_dmv=(2300, 2300, 2300),
            cos_phi_bp=(9950, 9900, 9920),
            p_inv_cmd_w=(2500, 400, 300),
            soc_bp=6500,
            p_compute_w=500,
            q_thermal_dhw_w=460,
            eta_ppc_bp=45000,
            c_spot_cents_mwh=14500,
            c_dist_cents_mwh=6500,
            breaker_warning=False,
            timestamp_us=98765432
        )
        self.assertEqual(len(frame), FRAME_SIZE_BYTES)
        self.assertEqual(len(frame), 128)

        unpacked = HouseholdEdgeEnergyNode.unpack_128b_household_frame(frame)
        self.assertEqual(unpacked["vital_max_hp"], 6)
        self.assertEqual(unpacked["mode"], HouseholdEdgeMode.GRID_CUT_ISLAND)
        self.assertAlmostEqual(unpacked["soc_percent"], 65.0, places=2)
        self.assertEqual(unpacked["p_compute_w"], 500)
        self.assertAlmostEqual(unpacked["c_spot_eur_mwh"], 145.0, places=2)
        self.assertEqual(unpacked["transformer_id"], 101)
        self.assertEqual(unpacked["battery_topology_mode"], 0)
        self.assertTrue(unpacked["is_128b_aligned"])

    def test_08_l3_grid_broker_extension_fields(self):
        """Bytes 74..95 must accurately pack and unpack L3 Grid Broker & Fleet Phased-Array fields."""
        frame = HouseholdEdgeEnergyNode.pack_128b_household_frame(
            mode=HouseholdEdgeMode.AGGRESSIVE_CHARGE_COMPUTE,
            i_grid_cma=(2200, 2100, 2000),
            v_grid_dmv=(2300, 2300, 2300),
            cos_phi_bp=(9900, 9900, 9900),
            p_inv_cmd_w=(-3000, -3000, -3000),
            soc_bp=4200,
            p_compute_w=5400,
            q_thermal_dhw_w=4968,
            eta_ppc_bp=55000,
            c_spot_cents_mwh=-1500,
            c_dist_cents_mwh=5500,
            breaker_warning=False,
            timestamp_us=55555555,
            transformer_id=2204,
            phase_topology_mask=0b00100100,
            battery_topology_mode=1, # 800V Series Grid-Boost
            q_reactive_cmd_var=(-1200, -800, -400),
            vpp_dispatch_target_w=7500,
            broker_contract_id=0x41465252 # "AFRR"
        )
        self.assertEqual(len(frame), 128)

        unpacked = HouseholdEdgeEnergyNode.unpack_128b_household_frame(frame)
        self.assertEqual(unpacked["transformer_id"], 2204)
        self.assertEqual(unpacked["phase_topology_mask"], 0b00100100)
        self.assertEqual(unpacked["battery_topology_mode"], 1)
        self.assertEqual(unpacked["q_reactive_cmd_var"], (-1200, -800, -400))
        self.assertEqual(unpacked["vpp_dispatch_target_w"], 7500)
        self.assertEqual(unpacked["broker_contract_id"], 0x41465252)
        self.assertEqual(unpacked["vital_max_hp"], 6)

    def test_03_asymmetric_phase_balancing_zero_grid_draw(self):
        """Heavy single-phase load on L1 must be covered from battery, leaving L1 grid draw at 0 W."""
        node = HouseholdEdgeEnergyNode(initial_soc=0.75)
        # L1 = 3500 W (Induction cooker), L2 = 200 W (TV), L3 = 150 W (Fridge)
        loads = PhaseLoads(l1_w=3500.0, l2_w=200.0, l3_w=150.0)
        # Spot price = 155 EUR/MWh (Expensive peak)
        res = node.step_15min_slot(
            slot_index=32,
            household_loads=loads,
            spot_price_eur_mwh=155.0
        )

        self.assertEqual(res.mode, HouseholdEdgeMode.GRID_CUT_ISLAND)
        # Grid draw on all phases should be reduced to 0 W!
        self.assertAlmostEqual(res.grid_draw_per_phase_w[0], 0.0, places=1)
        self.assertAlmostEqual(res.grid_draw_per_phase_w[1], 0.0, places=1)
        self.assertAlmostEqual(res.grid_draw_per_phase_w[2], 0.0, places=1)
        self.assertAlmostEqual(res.total_grid_draw_w, 0.0, places=1)
        # Battery must be discharging
        self.assertLess(res.battery_flow_w, 0.0)
        self.assertEqual(res.vital_max_hp, 6)

    def test_04_dynamic_peak_shaving_protects_25a_breaker(self):
        """Overload on L1 (e.g. 7.3 kW total) must be dynamically throttled to prevent breaker trip."""
        node = HouseholdEdgeEnergyNode(initial_soc=0.20)
        # L1 = 4000 W heavy appliance while charging is active at spot price = 15 EUR/MWh
        loads = PhaseLoads(l1_w=4000.0, l2_w=500.0, l3_w=500.0)
        res = node.step_15min_slot(
            slot_index=4,
            household_loads=loads,
            spot_price_eur_mwh=15.0  # Would trigger max charging + compute
        )

        # Total on L1 must strictly stay below 5,500 W safe ceiling
        self.assertLessEqual(res.grid_draw_per_phase_w[0], 5500.0)
        self.assertTrue(res.breaker_overload_prevented)
        self.assertEqual(res.vital_max_hp, 6)

    def test_05_thermal_dhw_recovery_accumulates_heat(self):
        """CLU 5.4 kW waste heat must accumulate in domestic hot water storage cylinder."""
        node = HouseholdEdgeEnergyNode(initial_soc=0.30, initial_dhw_kwh=0.0)
        loads = PhaseLoads(l1_w=300.0, l2_w=300.0, l3_w=300.0)

        # Run 4 intervals (1 hour) of aggressive charge & compute at cheap price
        for t in range(4):
            res = node.step_15min_slot(slot_index=t, household_loads=loads, spot_price_eur_mwh=12.0)

        self.assertGreater(res.dhw_stored_kwh, 4.0)
        self.assertGreater(res.thermal_recovered_w, 4000.0)
        self.assertEqual(res.vital_max_hp, 6)

    def test_06_state_hash_determinism(self):
        """Identical inputs must yield bit-exact identical state hashes."""
        n1 = HouseholdEdgeEnergyNode(initial_soc=0.50)
        n2 = HouseholdEdgeEnergyNode(initial_soc=0.50)
        loads = PhaseLoads(l1_w=1200.0, l2_w=800.0, l3_w=400.0)

        r1 = n1.step_15min_slot(slot_index=10, household_loads=loads, spot_price_eur_mwh=45.0)
        r2 = n2.step_15min_slot(slot_index=10, household_loads=loads, spot_price_eur_mwh=45.0)

        self.assertEqual(r1.state_hash, r2.state_hash)
        self.assertEqual(r1.frame_128b_hex, r2.frame_128b_hex)
        self.assertEqual(r1.net_financial_gain_eur, r2.net_financial_gain_eur)

    def test_07_full_24h_cycle_simulation(self):
        """24-hour simulation across 96 slots must yield positive financial gain and healthy PPC."""
        node = HouseholdEdgeEnergyNode(initial_soc=0.40)

        # Realistic 96-slot curves:
        # Slots 0..24 (00:00 - 06:00): cheap price ~15 EUR/MWh
        # Slots 24..36 (06:00 - 09:00): morning peak ~160 EUR/MWh
        # Slots 36..64 (09:00 - 16:00): solar dip ~30 EUR/MWh
        # Slots 64..84 (16:00 - 21:00): evening peak ~180 EUR/MWh
        # Slots 84..96 (21:00 - 24:00): night drop ~40 EUR/MWh
        spot_curve = []
        for s in range(96):
            if s < 24:
                spot_curve.append(15.0)
            elif s < 36:
                spot_curve.append(160.0)
            elif s < 64:
                spot_curve.append(30.0)
            elif s < 84:
                spot_curve.append(180.0)
            else:
                spot_curve.append(40.0)

        # Household load curve (base 400W/phase + morning/evening cooking & heating)
        l1_curve = [500.0 + (2500.0 if (28 <= s <= 34 or 68 <= s <= 76) else 0.0) for s in range(96)]
        l2_curve = [400.0 + (1800.0 if 70 <= s <= 78 else 0.0) for s in range(96)]
        l3_curve = [350.0 + (1200.0 if 30 <= s <= 60 else 0.0) for s in range(96)]

        sim = node.simulate_24h_cycle(
            spot_price_curve_96=spot_curve,
            load_curve_l1_96=l1_curve,
            load_curve_l2_96=l2_curve,
            load_curve_l3_96=l3_curve
        )

        self.assertEqual(sim["slots_count"], 96)
        self.assertGreater(sim["total_net_gain_eur"], 0.0)
        self.assertGreater(sim["average_eta_ppc"], 1.5)
        self.assertEqual(sim["vital_max_hp"], 6)


if __name__ == "__main__":
    unittest.main()
