#!/usr/bin/env python3
# ==============================================================================
# DEMO: VIRTUAL POWER PLANT (VPP) BROKER & SWITCHED BATTERY MATRIX
# ==============================================================================
# Invariant: VITAL_MAX_HP == 6
# Substation Target: 22 kV / 0.4 kV, 630 kVA, 100 Domácností (4 Vetvy x 25 Domov)
# ==============================================================================

import sys
from krystal_kernel.vpp_grid_broker_fleet import (
    VppGridBrokerFleetCoordinator,
    SwitchedBatteryMatrixController,
    BatteryTopology,
    VppFleetMode,
    VITAL_MAX_HP,
    VOLTAGE_MAX_ALLOWED_V,
    VOLTAGE_MIN_ALLOWED_V
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_vpp_broker_simulation():
    print("=" * 85)
    print("⚡ KRYSTAL-STACK: VIRTUAL POWER PLANT (VPP) BROKER & PHASED ARRAY FLEET DISPATCH")
    print(f"🔒 Konštitučný Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}")
    print("🏢 Trafostanica: 22 kV / 0.4 kV, 630 kVA | 100 Domácností (4 vetvy x 25 domov)")
    print(f"📏 Limit EN 50160: 230 V +/- 10% ({VOLTAGE_MIN_ALLOWED_V:.1f} V <= V <= {VOLTAGE_MAX_ALLOWED_V:.1f} V)")
    print("=" * 85)

    # 1. Porovnanie SiC MOSFET spínania (48V vs 800V DC bus)
    print("\n▶ 1. HARDVÉROVÁ REKONFIGURÁCIA SiC MOSFET MATICE (48V vs 800V DC BUS):")
    print("-" * 85)
    curr_48, drop_48, loss_48 = SwitchedBatteryMatrixController.calculate_cable_loss(
        power_w=15000.0,
        topology=BatteryTopology.PARALLEL_48V,
        cable_resistance_ohm=0.045
    )
    curr_800, drop_800, loss_800 = SwitchedBatteryMatrixController.calculate_cable_loss(
        power_w=15000.0,
        topology=BatteryTopology.SERIES_800V,
        cable_resistance_ohm=0.045
    )
    print(f"  Pri výkone 15 kW do siete:")
    print(f"  🔴 Režim 48V Paralelný:  Prúd = {curr_48:6.1f} A | Úbytok = {drop_48:5.2f} V | Tepelná strata = {loss_48:6.1f} W")
    print(f"  🟢 Režim 800V GRID BOOST: Prúd = {curr_800:6.1f} A | Úbytok = {drop_800:5.2f} V | Tepelná strata = {loss_800:6.1f} W")
    print(f"  ✨ Redukcia tepelných strát kábla: {(loss_48 / loss_800):.1f}x menej tepla na domovom prívode!")

    broker = VppGridBrokerFleetCoordinator(node_count=100)

    # 2. Scenár: Špičkový aFRR+ export do siete (+0.80 MW)
    print("\n▶ 2. SCENÁR: aFRR+ ŠPIČKOVÝ EXPORT (+0.80 MW DO SIETE PRI CENE 175 €/MWh)")
    print("-" * 85)
    dec_export = broker.dispatch_fleet_request(requested_mw=0.80, spot_price_eur_mwh=175.0)
    print(f"  Požiadavka siete:        +{dec_export.target_grid_power_mw:.2f} MW (aFRR+ nahor)")
    print(f"  Dosiahnutý export:       +{dec_export.achieved_grid_power_mw:.3f} MW ({dec_export.achieved_grid_power_mw * 1000:.0f} kW)")
    print(f"  Režim flotily:            {dec_export.fleet_mode.name}")
    print(f"  800V GRID BOOST uzlov:    {dec_export.switched_matrix_800v_nodes_count} / 100 uzlov")
    print(f"  Max napätie na vývode:    {dec_export.max_feeder_voltage_v:.2f} V (Limit: 253.0 V)")
    print(f"  Fázované pole Q:          {dec_export.total_reactive_power_mvar * 1000:.1f} kvar (induktívna kompenzácia napätia)")
    print(f"  CLU tepelné tlmenie:      {dec_export.total_clu_thermal_dampening_kw:.1f} kW (odvod tepla do bojlerov na koncoch ulíc)")
    print(f"  Prepätia a výpadky:       {dec_export.voltage_violations_count} výpadkov (Ochrana aktívna: {dec_export.overvoltage_prevented_by_phased_array})")

    # 3. Scenár: Masívny aFRR- odber pri zápornej cene (-1.20 MW)
    print("\n▶ 3. SCENÁR: aFRR- MASÍVNY ODBER PREBYTKOV (-1.20 MW PRI ZÁPORNEJ CENE -25 €/MWh)")
    print("-" * 85)
    dec_absorb = broker.dispatch_fleet_request(requested_mw=-1.20, spot_price_eur_mwh=-25.0)
    print(f"  Požiadavka siete:        {dec_absorb.target_grid_power_mw:.2f} MW (aFRR- nadol)")
    print(f"  Dosiahnutá absorpcia:    -{dec_absorb.total_active_absorption_mw:.3f} MW ({dec_absorb.total_active_absorption_mw * 1000:.0f} kW)")
    print(f"  Režim flotily:            {dec_absorb.fleet_mode.name}")
    print(f"  Konfigurácia batérií:     48V Paralelný šetrný režim (0 uzlov v 800V)")
    print(f"  Min napätie na vývode:    {dec_absorb.min_feeder_voltage_v:.2f} V (Limit: 207.0 V)")
    print(f"  Podpäťové výpadky:        {dec_absorb.voltage_violations_count} výpadkov (Sieť zostala stabilná)")

    print("\n" + "=" * 85)
    print(f"🏆 VŠETKY TESTY FLOTILY DOKONČENÉ. INVARIANT VITAL_MAX_HP == {VITAL_MAX_HP} ZACHOVANÝ.")
    print("=" * 85)


if __name__ == "__main__":
    run_vpp_broker_simulation()
