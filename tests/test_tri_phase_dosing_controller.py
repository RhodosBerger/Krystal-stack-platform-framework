#!/usr/bin/env python3
# ==============================================================================
# TESTS FOR TRI-PHASE POWER DOSING, INVERTER CASCADE & GPU SHARDING
# ==============================================================================
# Target Invariant: VITAL_MAX_HP == 6
# Standard: Zero-Copy 128B Lingua Franca (P2), Thermal Thresholds, EN 50160
# ==============================================================================

import unittest
from krystal_kernel.tri_phase_dosing_controller import (
    TriPhaseDosingController,
    CableThermalState,
    TriPhaseDosingOutput,
    GpuPhasePriority,
    ThermalValveState,
    TriPhaseDosingException,
    VITAL_MAX_HP,
    FRAME_SIZE_BYTES,
    MAX_CABLE_SAFE_TEMP_C,
    MAX_NEUTRAL_SAFE_TEMP_C,
    BOILER_MAX_TEMP_C
)


class TestTriPhaseDosingController(unittest.TestCase):
    """Verifies Tri-Phase power dosing, cable temperature monitoring, and GPU phase sharding."""

    def setUp(self):
        self.controller = TriPhaseDosingController()

    def test_01_constitutional_invariant_holds(self):
        """Constitutional invariant must strictly equal 6."""
        self.assertEqual(VITAL_MAX_HP, 6)

    def test_02_128b_dosed_frame_packing_and_unpacking_roundtrip(self):
        """128-byte frame with TriPhase Dosing extension must roundtrip with bit-exact alignment."""
        cable_state = CableThermalState(
            t_l1_c=38.5,
            t_l2_c=39.2,
            t_l3_c=37.0,
            t_neutral_c=35.4
        )
        p_gpu_w = (1240, 1240, 1660) # u16
        gpu_priorities = (
            int(GpuPhasePriority.CRYPTO_MINING),
            int(GpuPhasePriority.CRYPTO_MINING),
            int(GpuPhasePriority.CRYPTO_MINING)
        )
        inverter_duty = (90, 90, 90)
        valve_state = int(ThermalValveState.BOILER_DHW)

        # 1. Zabalenie do 128B
        frame_bytes = self.controller.pack_128b_dosed_frame(
            cable_state=cable_state,
            p_gpu_w=p_gpu_w,
            gpu_priorities=gpu_priorities,
            inverter_duty_pct=inverter_duty,
            valve_state=valve_state,
            timestamp_us=123456789
        )
        self.assertEqual(len(frame_bytes), FRAME_SIZE_BYTES)
        self.assertEqual(len(frame_bytes), 128)

        # 2. Rozbalenie a verifikácia polí (offset 96..117)
        unpacked = self.controller.unpack_128b_dosed_frame(frame_bytes)

        self.assertAlmostEqual(unpacked["t_cable_l1_c"], 38.5, places=1)
        self.assertAlmostEqual(unpacked["t_cable_l2_c"], 39.2, places=1)
        self.assertAlmostEqual(unpacked["t_cable_l3_c"], 37.0, places=1)
        self.assertAlmostEqual(unpacked["t_cable_neutral_c"], 35.4, places=1)

        self.assertEqual(unpacked["p_gpu_w"], (1240, 1240, 1660))
        self.assertEqual(
            unpacked["gpu_priorities"],
            (GpuPhasePriority.CRYPTO_MINING, GpuPhasePriority.CRYPTO_MINING, GpuPhasePriority.CRYPTO_MINING)
        )
        self.assertEqual(unpacked["inverter_dosing_duty_pct"], (90, 90, 90))
        self.assertEqual(unpacked["thermal_valve_state"], ThermalValveState.BOILER_DHW)
        self.assertEqual(unpacked["vital_max_hp"], 6)
        self.assertTrue(unpacked["is_128b_aligned"])

    def test_03_cable_thermal_alarm_thresholds(self):
        """Cable alarm must trigger at >= 45.0 °C on phase terminals or >= 50.0 °C on Neutral."""
        # Bezpečný stav
        safe_state = CableThermalState(44.9, 44.8, 44.5, 49.5)
        self.assertFalse(safe_state.is_alarm_active)

        # Prehriatie fázy L1
        hot_l1 = CableThermalState(45.0, 38.0, 38.0, 35.0)
        self.assertTrue(hot_l1.is_alarm_active)

        # Prehriatie fázy L2
        hot_l2 = CableThermalState(38.0, 45.2, 38.0, 35.0)
        self.assertTrue(hot_l2.is_alarm_active)

        # Prehriatie fázy L3
        hot_l3 = CableThermalState(38.0, 38.0, 46.1, 35.0)
        self.assertTrue(hot_l3.is_alarm_active)

        # Prehriatie Nulového vodiča (N)
        hot_neutral = CableThermalState(42.0, 42.0, 42.0, 50.0)
        self.assertTrue(hot_neutral.is_alarm_active)

    def test_04_phase_overload_reaction_and_gpu_throttling(self):
        """Under sudden phase L1 overload (stove 3 kW + T=46°C), GPU 0,1,2 must throttle to AI_INFERENCE."""
        # Scenár: Bežný stav - nízky domáci odber, studené káble
        normal_cables = CableThermalState(35.0, 35.0, 35.0, 32.0)
        out_normal = self.controller.evaluate_control_cycle(
            household_load_per_phase_w=(500.0, 500.0, 500.0),
            cable_state=normal_cables,
            spot_price_eur_mwh=180.0,
            boiler_temp_c=50.0
        )
        self.assertFalse(out_normal.phase_relief_applied)
        self.assertFalse(out_normal.cable_alarm_triggered)
        self.assertEqual(out_normal.gpu_priorities[0], GpuPhasePriority.CRYPTO_MINING)
        self.assertEqual(out_normal.gpu_priorities[1], GpuPhasePriority.CRYPTO_MINING)
        self.assertEqual(out_normal.gpu_priorities[2], GpuPhasePriority.CRYPTO_MINING)
        # Príkon GPU: 3*400+40 = 1240 W na L1, L2; 4*400+60 = 1660 W na L3
        self.assertAlmostEqual(out_normal.p_gpu_w[0], 1240.0)
        self.assertAlmostEqual(out_normal.p_gpu_w[1], 1240.0)
        self.assertAlmostEqual(out_normal.p_gpu_w[2], 1660.0)
        self.assertAlmostEqual(out_normal.total_compute_power_w, 4140.0)

        # Scenár: Skokový odber na L1 (sporák 3 000 W) a oteplenie svorky na 46.0 °C
        overload_cables = CableThermalState(46.0, 36.0, 36.0, 34.0)
        out_overload = self.controller.evaluate_control_cycle(
            household_load_per_phase_w=(3000.0, 500.0, 500.0),
            cable_state=overload_cables,
            spot_price_eur_mwh=350.0, # Špičková cena
            boiler_temp_c=55.0
        )

        # Reakcia kontroléra:
        self.assertTrue(out_overload.phase_relief_applied)
        self.assertTrue(out_overload.cable_alarm_triggered)
        # 1. GPU na L1 okamžite znížené na AI_INFERENCE
        self.assertEqual(out_overload.gpu_priorities[0], GpuPhasePriority.AI_INFERENCE)
        # 3*85 + 40 = 295.0 W (zníženie z 1240 W o 945 W!)
        self.assertAlmostEqual(out_overload.p_gpu_w[0], 295.0)
        # 2. Menič #1 znížený na duty 40 % pre ochranu kábla L1
        self.assertEqual(out_overload.inverter_dosing_duty_pct[0], 40)
        # 3. Meniče #2 a #3 vykompenzujú export na 95 %
        self.assertEqual(out_overload.inverter_dosing_duty_pct[1], 95)
        self.assertEqual(out_overload.inverter_dosing_duty_pct[2], 95)
        # 4. GPU na L2 a L3 pokračujú na plný výkon CRYPTO_MINING
        self.assertEqual(out_overload.gpu_priorities[1], GpuPhasePriority.CRYPTO_MINING)
        self.assertEqual(out_overload.gpu_priorities[2], GpuPhasePriority.CRYPTO_MINING)
        self.assertEqual(out_overload.vital_max_hp, 6)

    def test_05_thermal_valve_boiler_diversion(self):
        """When DHW boiler reaches >= 65.0 °C, 3-way valve must divert heat to Dry-Cooler."""
        cables = CableThermalState(36.0, 36.0, 36.0, 33.0)

        # Pod 65 °C -> Ohrev bojlera
        out_cold = self.controller.evaluate_control_cycle(
            household_load_per_phase_w=(800.0, 800.0, 800.0),
            cable_state=cables,
            boiler_temp_c=64.8
        )
        self.assertEqual(out_cold.thermal_valve_state, ThermalValveState.BOILER_DHW)

        # Pri 65.0 °C -> Presmerovanie do suchého chladiča
        out_hot = self.controller.evaluate_control_cycle(
            household_load_per_phase_w=(800.0, 800.0, 800.0),
            cable_state=cables,
            boiler_temp_c=65.0
        )
        self.assertEqual(out_hot.thermal_valve_state, ThermalValveState.EXTERNAL_DRY_COOLER)

        # Nad 65 °C -> Zostáva v suchom chladiči
        out_superhot = self.controller.evaluate_control_cycle(
            household_load_per_phase_w=(800.0, 800.0, 800.0),
            cable_state=cables,
            boiler_temp_c=72.0
        )
        self.assertEqual(out_superhot.thermal_valve_state, ThermalValveState.EXTERNAL_DRY_COOLER)

    def test_06_neutral_wire_harmonic_current_remains_safe(self):
        """Neutral wire current must stay well within safe limits (< 5 A) even during unbalanced phase loads."""
        cables = CableThermalState(36.0, 36.0, 36.0, 33.0)
        out = self.controller.evaluate_control_cycle(
            household_load_per_phase_w=(2500.0, 400.0, 600.0),
            cable_state=cables,
            boiler_temp_c=52.0
        )
        self.assertLess(out.neutral_current_est_a, 5.0)
        self.assertGreaterEqual(out.neutral_current_est_a, 1.2)

    def test_07_invalid_frame_size_and_contract_violations(self):
        """Any invalid frame size or contract violation must raise TriPhaseDosingException."""
        # Neplatná dĺžka rámca
        with self.assertRaises(TriPhaseDosingException):
            self.controller.unpack_128b_dosed_frame(bytes(127))

        with self.assertRaises(TriPhaseDosingException):
            self.controller.unpack_128b_dosed_frame(bytes(129))
    def test_08_round_robin_duty_dosing_rotation(self):
        """Under balanced conditions, duty cycles must rotate in round-robin fashion (T1->L1, T2->L2, T3->L3)."""
        ctrl = TriPhaseDosingController()
        cables = CableThermalState(35.0, 35.0, 35.0, 32.0)
        loads = (500.0, 500.0, 500.0)

        # Slice 1: time_slice_counter % 3 == 1 -> L2 gets boost
        out1 = ctrl.evaluate_control_cycle(loads, cables)
        self.assertEqual(out1.inverter_dosing_duty_pct, (70, 95, 70))

        # Slice 2: time_slice_counter % 3 == 2 -> L3 gets boost
        out2 = ctrl.evaluate_control_cycle(loads, cables)
        self.assertEqual(out2.inverter_dosing_duty_pct, (70, 70, 95))

        # Slice 3: time_slice_counter % 3 == 0 -> L1 gets boost
        out3 = ctrl.evaluate_control_cycle(loads, cables)
        self.assertEqual(out3.inverter_dosing_duty_pct, (95, 70, 70))

    def test_09_telemetry_text_representation_home_station_and_compute(self):
        """Textual representation must explicitly format original measured physical quantities."""
        ctrl = TriPhaseDosingController()
        cables = CableThermalState(38.0, 37.5, 36.8, 33.2)
        loads = (1200.0, 800.0, 600.0)

        out = ctrl.evaluate_control_cycle(loads, cables, boiler_temp_c=58.0)
        report = out.to_text_report()

        # Overenie pôvodných meraných hodnôt pre domáce stanice a počítače
        self.assertIn("TELEMETRIA MERANIA", report)
        self.assertIn("230.0V", report) # Sieťové napätie domácej stanice
        self.assertIn("1240W", report)  # Pôvodný meraný príkon GPU
        self.assertIn("Prúd =", report) # Prúd nulákom
        self.assertIn("120° PWM fázový offset", report) # Rušenie 3. harmonickej 150 Hz
        self.assertIn("Celkový príkon 10x GPU", report)
        self.assertIn("BOILER_DHW", report) # TÚV chladenie
        self.assertIn("65.0°C", report)     # Bezpečnostný prah bojlera
        self.assertIn("45.0°C", report)     # Bezpečnostný prah svoriek medi
        self.assertIn("VITAL_MAX_HP == 6", report)
        self.assertEqual(str(out), report)


if __name__ == "__main__":
    unittest.main()
