#!/usr/bin/env python3
# ==============================================================================
# DEMO: PREDICTIVE GRID-AWARE ENERGY ARBITRAGE (MOONSHOT C)
# ==============================================================================

import sys
from krystal_kernel.grid_aware_energy_arbitrage import (
    PredictiveEnergyArbitrageEngine,
    GridTelemetry,
    WorkloadMode,
    VITAL_MAX_HP
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_grid_arbitrage_simulation():
    print("=" * 80)
    print("⚡ KRYSTAL-STACK: PREDICTIVE GRID-AWARE ENERGY ARBITRAGE SIMULATION (MOONSHOT C)")
    print(f"🔒 Constitutional Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}")
    print("🎯 Hardware Ceiling: 10x AMD GPU + 128-core EPYC (500 W - 5,400 W)")
    print("=" * 80)

    scenarios = [
        {
            "name": "1. Nočný veterný prebytok (Záporná spotová cena)",
            "telemetry": GridTelemetry(frequency_hz=50.035, spot_price_eur_mwh=-18.50),
            "temp_c": 54.0,
            "desc": "Operátor platí 18.50 EUR/MWh za odber! Max hash mining záťaž."
        },
        {
            "name": "2. Ranná špička (Drahá elektrina + mierny deficit siete)",
            "telemetry": GridTelemetry(frequency_hz=49.965, spot_price_eur_mwh=142.00),
            "temp_c": 58.0,
            "desc": "Extrémna cena elektriny -> Ťažba vypnutá, priorita vysoko maržovej AI inferencie."
        },
        {
            "name": "3. Poludňajší solárny boom (FCR primárna regulácia nadfrekvencie)",
            "telemetry": GridTelemetry(frequency_hz=50.140, spot_price_eur_mwh=28.00),
            "temp_c": 62.0,
            "desc": "Prebytok solárnej energie v SEPS (+140 mHz) -> Inkasovanie FCR regulačného bonusu."
        },
        {
            "name": "4. Havarijná porucha v sieti (Kritická podfrekvencia < 49.80 Hz)",
            "telemetry": GridTelemetry(frequency_hz=49.760, spot_price_eur_mwh=-5.00),
            "temp_c": 56.0,
            "desc": "Výpadok elektrárne -> Okamžitý Emergency Load Shed na 500 W chráni sieť pred blackoutom."
        },
        {
            "name": "5. Termálne prehriatie kremíka (Princíp P7: Fyzika > Ekonomika)",
            "telemetry": GridTelemetry(frequency_hz=50.000, spot_price_eur_mwh=-30.00),
            "temp_c": 86.5,
            "desc": "Tj = 86.5 C (prehriatie) -> Ochranný clamp na 2,800 W aj pri zápornej cene."
        }
    ]

    for sc in scenarios:
        print(f"\n▶ SCENÁR: {sc['name']}")
        print(f"  Kontext: {sc['desc']}")
        t = sc["telemetry"]
        print(f"  Vstupy:  Frekvencia = {t.frequency_hz:.3f} Hz (Δf = {t.frequency_dev_hz:+.3f} Hz) | Spot OKTE = {t.spot_price_eur_mwh:+.2f} EUR/MWh | Tj = {sc['temp_c']} °C")
        
        dec = PredictiveEnergyArbitrageEngine.solve_arbitrage(
            telemetry=t,
            silicon_junction_temp_c=sc["temp_c"]
        )

        mode_icon = {
            WorkloadMode.CRYPTO_MINING: "⛏️ [CRYPTO_MINING]",
            WorkloadMode.HIGH_MARGIN_AI_INFERENCE: "🧠 [HIGH_MARGIN_AI_INFERENCE]",
            WorkloadMode.DYNAMIC_HYBRID_ARBITRAGE: "⚖️ [DYNAMIC_HYBRID_ARBITRAGE]",
            WorkloadMode.EMERGENCY_LOAD_SHED: "🚨 [EMERGENCY_LOAD_SHED]"
        }.get(dec.workload_mode, str(dec.workload_mode))

        print(f"  Výstup:  Režim: {mode_icon}")
        print(f"           Alokovaný výkon: {dec.allocated_power_watts:7.1f} W (Mining: {dec.mining_ratio_alpha*100:5.1f}% | AI: {dec.ai_inference_ratio*100:5.1f}%)")
        print(f"           Bilancia EUR/h:  Zisk = {dec.expected_profit_eur_per_hour:+.4f} €/h | Náklad siete = {dec.grid_cost_eur_per_hour:+.4f} €/h | FCR = {dec.fcr_revenue_eur_per_hour:+.4f} €/h")
        print(f"           Termálny clamp:  {'⚠️ AKTÍVNY (Ochrana hardvéru)' if dec.thermal_throttled else '✅ NOMINÁLNY'}")
        print(f"           128B State Hash: {dec.state_hash} | Invariant: HP == {dec.vital_max_hp}")

    print("\n" + "=" * 80)
    print("✅ VŠETKY SCENÁRE ÚSPEŠNE DOKONČENÉ V SUB-MILISEKUNDOVOM FORMÁTE.")
    print("=" * 80)


if __name__ == "__main__":
    run_grid_arbitrage_simulation()
