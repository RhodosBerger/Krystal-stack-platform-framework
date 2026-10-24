"""
krystal_bot/sovereign_layer_governor.py

Sovereign Layer Governor & Bot Scale Expert for krystal_bot.
=============================================================================
Provides interactive commands and cheatsheets for the BotService to query,
inspect, tune, and optimize all 28 governing variables across the 6 architectural
layers (L0 to L5) and 4 scale domains (Micro, Meso, Macro, Epoch).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Any

from krystal_kernel.sovereign_bot_scale_dimensioner import (
    ScaleDomain,
    LayerSystemId,
    ScaleProfile,
    LayerGoverningVariable,
    VariableTuningReceipt,
    OptimalOperatingReceipt,
    SovereignBotScaleDimensioner,
    GLOBAL_SOVEREIGN_BOT_SCALE_DIMENSIONER
)
from krystal_kernel.constants import VITAL_MAX_HP


class SovereignLayerGovernor:
    """Expert helper integrating multi-layer variable governance into krystal_bot."""

    def __init__(self, dimensioner: Optional[SovereignBotScaleDimensioner] = None):
        self.dimensioner = dimensioner or GLOBAL_SOVEREIGN_BOT_SCALE_DIMENSIONER

    def list_variables_cheatsheet(self, layer_query: Optional[str] = None) -> str:
        """Formats all 28 governing variables grouped by layer."""
        target_layer: Optional[LayerSystemId] = None
        if layer_query:
            q = layer_query.strip().upper()
            for lid in LayerSystemId:
                if q in lid.value or q in lid.name:
                    target_layer = lid
                    break

        variables = self.dimensioner.list_variables_by_layer(target_layer)
        lines = [
            "=" * 78,
            "🏛️  KRYSTAL STACK: PREHĽAD PREMENNÝCH VRSTIEV A ŠKÁLOVANIA BOTA",
            f"🔒 Ústavný Invariant: VITAL_MAX_HP == {VITAL_MAX_HP} (Delta S == 0 pri odmietnutí)",
            "=" * 78,
            f"{'ID PREMENNEJ':<32} | {'AKTUÁLNA':<10} | {'JEDNOTKA':<8} | {'ŠKÁLA':<18} | {'STAV'}",
            "-" * 78
        ]

        current_layer = None
        for v in variables:
            if v.layer_id != current_layer:
                current_layer = v.layer_id
                lines.append(f"\n▶ [{current_layer.value}]")

            imm_tag = "🔒 ZÁKON" if v.is_constitutional_immutable else "⚙️ LADITEĽNÁ"
            val_str = f"{v.current_value:.4g}" if isinstance(v.current_value, float) else str(v.current_value)
            lines.append(
                f"{v.variable_id:<32} | {val_str:<10} | {v.unit:<8} | {v.scale_domain.name:<18} | {imm_tag}"
            )

        lines.extend([
            "-" * 78,
            f"Celkom registrovaných riadiacich premenných: {len(variables)}",
            "Príkazy: /layer get <var_id> | /layer tune <var_id> <hodnota> | /layer optimal | /layer scales",
            "=" * 78
        ])
        return "\n".join(lines)

    def get_variable_details(self, variable_id: str) -> str:
        """Returns comprehensive physical and architectural details for a specific variable."""
        vid = variable_id.strip().upper()
        # Allow shorthand search
        match_var = None
        for v in self.dimensioner.list_variables_by_layer():
            if vid == v.variable_id or vid in v.variable_id:
                match_var = v
                break

        if not match_var:
            return f"❌ Neznáma riadiaca premenná: '{variable_id}'. Použite '/layer list' pre zoznam."

        v = match_var
        lines = [
            "=" * 70,
            f"📄 DETAIL PREMENNEJ: {v.variable_id}",
            "=" * 70,
            f"  Názov:                 {v.name}",
            f"  Architektonická vrstva:{v.layer_id.value}",
            f"  Škálovacia doména:     {v.scale_domain.value}",
            f"  Aktuálna hodnota:      {v.current_value} {v.unit}",
            f"  Nominálna hodnota:     {v.nominal_value} {v.unit}",
            f"  Povolený rozsah:       [{v.min_value}, {v.max_value}] {v.unit}",
            f"  Ústavný status:        {'NEMENNÝ ZÁKON (IMMUTABLE)' if v.is_constitutional_immutable else 'LADITEĽNÁ PREVÁDZKOVÁ HODNOTA'}",
            f"  Riadica rovnica:       {v.governing_equation}",
            f"  Fyzikálny popis:       {v.description}",
            f"  Invariant integrity:   VITAL_MAX_HP == {v.vital_max_hp}",
            "=" * 70
        ]
        return "\n".join(lines)

    def tune_variable(self, variable_id: str, new_value_str: str) -> str:
        """Attempts to tune a variable and returns an authoritative result string."""
        vid = variable_id.strip().upper()
        target_var = None
        for v in self.dimensioner.list_variables_by_layer():
            if vid == v.variable_id or vid in v.variable_id:
                target_var = v
                break

        if not target_var:
            return f"❌ Premenná '{variable_id}' nebola nájdená."

        try:
            val = float(new_value_str.strip())
        except ValueError:
            return f"❌ Neplatná číselná hodnota: '{new_value_str}'"

        receipt = self.dimensioner.tune_variable(
            variable_id=target_var.variable_id,
            new_value=val,
            proposer_role="BOT_OPERATOR",
            evaluator_role="CONSTITUTIONAL_GATEKEEPER"
        )

        if receipt.success:
            return (
                f"✅ PREMENNÁ ÚSPEŠNE UPRAVENÁ\n"
                f"  Premenná:    {receipt.variable_id}\n"
                f"  Pôvodná:     {receipt.old_value} {target_var.unit}\n"
                f"  Nová hodnota: {receipt.committed_value} {target_var.unit}\n"
                f"  Zmena stavu: {receipt.mutation_delta_bytes} bajtov zapísaných\n"
                f"  Audit Hash:  {receipt.receipt_hash[:16]}...\n"
                f"  Invariant:   VITAL_MAX_HP == {receipt.vital_max_hp} zachovaný"
            )
        else:
            return (
                f"❌ ODMIETNUTÉ ZÁKONOM (Delta S == 0)\n"
                f"  Premenná:    {receipt.variable_id}\n"
                f"  Dôvod:       {receipt.reason}\n"
                f"  Hodnota:     Zostáva {receipt.old_value} {target_var.unit}\n"
                f"  Audit Hash:  {receipt.receipt_hash[:16]}..."
            )

    def format_optimal_operating_cheatsheet(self, intensity_str: Optional[str] = None) -> str:
        """Calculates and displays the Pareto-optimal operating profile."""
        intensity = 0.5
        if intensity_str:
            try:
                intensity = float(intensity_str.strip())
            except ValueError:
                intensity = 0.5

        rec = self.dimensioner.solve_optimal_operating_point(intensity)
        lines = [
            "=" * 70,
            "⚡ OPTIMÁLNY MULTI-VRSTVOVÝ PREVÁDZKOVÝ BOD (PARETO-OPTIMAL)",
            f"🔒 Ústavný Invariant: VITAL_MAX_HP == {rec.vital_max_hp}",
            "=" * 70,
            f"  ID Certifikátu:        {rec.operating_point_id}",
            f"  Cieľová snímková frekv:{rec.target_fps} Hz",
            f"  VBlank tieňové okno:   {rec.allocated_shadow_ms} ms (Požiadavka >= 3.50 ms)",
            f"  Projektovaný príkon PL1:{rec.projected_power_w} W",
            f"  Saturácia zbernice UMA:{rec.uma_utilization * 100:.1f} % (Strop < 78.0 %)",
            f"  Napäťová rezerva (PDN):{rec.voltage_stability_margin_v} V (Limit +0.15 V)",
            f"  Symplektická konzervácia:det(J) == {rec.det_j_conservation:.12f}",
            f"  Hysterézny dwell takt: {rec.dwell_filter_cadence_ms} ms",
            f"  Koordinácia vrstiev:   {rec.cross_layer_coordination_status}",
            f"  Všetky invarianty:     {'✅ SPLNENÉ' if rec.all_invariants_satisfied else '❌ PORUŠENÉ'}",
            f"  HMAC-SHA256 podpis:    {rec.audit_hmac[:32]}...",
            "=" * 70
        ]
        return "\n".join(lines)

    def format_scales_cheatsheet(self) -> str:
        """Formats the 4 hierarchical scale domains dimensioning the Sovereign Bot."""
        profiles = self.dimensioner.list_scale_profiles()
        lines = [
            "=" * 78,
            "🌐 ŠKÁLOVACIA DIMENZIA SOVEREIGN BOTA (4 ČASOVO-KAPACITNÉ DOMÉNY)",
            f"🔒 Ústavný Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}",
            "=" * 78,
            f"{'DOMÉNA':<22} | {'GRANULARITA':<18} | {'KONKURENCIA':<12} | {'PRIEPUSTNOSŤ':<14}",
            "-" * 78
        ]

        for p in profiles:
            tp_str = f"{p.throughput_events_per_sec:,.0f} ev/s" if p.throughput_events_per_sec >= 100 else f"{p.throughput_events_per_sec:.1f} ev/s"
            lines.append(
                f"{p.domain.name:<22} | {p.time_granularity:<18} | {p.max_concurrency:<12} | {tp_str:<14}"
            )

        lines.extend([
            "-" * 78,
            "Popis domén:",
            "  • MICRO_SILICON_NS:   Sub-50ns TSC telemetria, non-temporal MOVNTI zápisy, réžia < 0.12%",
            "  • MESO_VBLANK_MS:     3.5ms VBlank tieň, Level Zero USM (0.748us), UMA zbernica < 78%",
            "  • MACRO_CADENCE_MS:   12-rámcová dwell hysterézia (200ms), 100ms kognitívne vlny, DVFS PL1",
            "  • EPOCH_GOVERNANCE_S: VPP energetická arbitráž, kaskády modelov, HMAC blockchain ledger",
            "=" * 78
        ])
        return "\n".join(lines)


# Global Singleton instance
SOVEREIGN_LAYER_GOVERNOR: SovereignLayerGovernor = SovereignLayerGovernor()
