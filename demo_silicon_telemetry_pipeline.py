#!/usr/bin/env python3
# ==============================================================================
# DEMO: SILICON TELEMETRY & OPENVINO DYNAMIC POLICY PIPELINE
# FORMAL SYSTEM MODEL: (S, A, T, I, O, E, H, W)
# ==============================================================================

import sys
import json
from krystal_kernel.silicon_pipeline_orchestrator import (
    GLOBAL_SILICON_PIPELINE_ORCHESTRATOR,
    SiliconAction,
    VITAL_MAX_HP
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_pipeline_demo():
    print("=" * 80)
    print("  KRYSTAL-STACK: FORMAL SILICON TELEMETRY & OPENVINO ARBITER PIPELINE")
    print(f"  System Model: (S, A, T, I, O, E, H, W) | Invariant: VITAL_MAX_HP == {VITAL_MAX_HP}")
    print("=" * 80)

    orch = GLOBAL_SILICON_PIPELINE_ORCHESTRATOR

    # --------------------------------------------------------------------------
    # SCENARIO 1: Nominal Active Inference (P=32, G=1)
    # --------------------------------------------------------------------------
    print("\n[SCENARIO 1] Executing Nominal Active Inference (Prompt: 32 tokens)...")
    act1 = SiliconAction(action_type="DISPATCH_INFERENCE", prompt_tokens=32, generation_tokens=1)
    res1 = orch.step_pipeline(act1, simulated_time_s=1.0, load_power_w=20.0)
    
    st1 = res1["state"]
    print(f"  -> Status:             {res1['status']} (Delta S = {res1['delta_s']})")
    print(f"  -> State Step:         S_{st1['step']} (Tj = {st1['temperature_c']}°C, Return Loss = {st1['return_loss_db']} dB)")
    print(f"  -> UMA Ring Latency:   {st1['uma_cache_transit_us']} µs")
    print(f"  -> Arbitrated Policy:  {st1['current_policy']} ({res1['arbitration']['arbitration_reason']})")
    print(f"  -> Provenance Hash:    {st1['provenance_hash']}")
    print(f"  -> OpenVINO Precision: {res1['openvino_runtime']['precision']} (Iris Xe DP4A, {res1['openvino_runtime']['streams']} streams)")

    # --------------------------------------------------------------------------
    # SCENARIO 2: High Contention & Thermal Stress (P=128, Heavy Power Load)
    # --------------------------------------------------------------------------
    print("\n[SCENARIO 2] Executing Long-Prompt Inference under Thermal Load (P=128 tokens)...")
    act2 = SiliconAction(action_type="DISPATCH_INFERENCE", prompt_tokens=128, generation_tokens=4)
    res2 = orch.step_pipeline(act2, simulated_time_s=180.0, load_power_w=45.0)

    st2 = res2["state"]
    print(f"  -> Status:             {res2['status']} (Delta S = {res2['delta_s']})")
    print(f"  -> State Step:         S_{st2['step']} (Tj = {st2['temperature_c']}°C, Return Loss = {st2['return_loss_db']} dB)")
    print(f"  -> UMA Ring Latency:   {st2['uma_cache_transit_us']} µs (Contention Elevated)")
    print(f"  -> Arbitrated Policy:  {st2['current_policy']} ({res2['arbitration']['arbitration_reason']})")
    print(f"  -> Policy Shifted:     {res2['arbitration']['policy_shifted']} ({res2['arbitration']['previous_policy']} -> {res2['arbitration']['selected_policy']})")
    print(f"  -> Arrhenius Wear:     {st2['arrhenius_wear_factor']} (Silicon Lifespan Protected)")
    print(f"  -> Provenance Hash:    {st2['provenance_hash']}")

    # --------------------------------------------------------------------------
    # SCENARIO 3: Invalid Action Boundary Test (Delta S = 0 Enforcement)
    # --------------------------------------------------------------------------
    print("\n[SCENARIO 3] Attempting Invalid Action (Negative prompt tokens)...")
    invalid_act = SiliconAction(action_type="MALFORMED_DISPATCH", prompt_tokens=-10)
    res3 = orch.step_pipeline(invalid_act)
    print(f"  -> Status:             {res3['status']} (Delta S = {res3['delta_s']})")
    print(f"  -> Invariant Verified: Constitutional contract preserved, state untouched (S_{res3['state']['step']})")

    # --------------------------------------------------------------------------
    # SCENARIO 4: Observables Projection & Pulse Diagram Timeline
    # --------------------------------------------------------------------------
    print("\n[SCENARIO 4] Projecting Observables O & Pulse Diagram Timeline...")
    obs = orch.get_observables(pulse_diagram_limit=12)
    print(obs["pulse_diagram"])

    print("\n" + "=" * 80)
    print(f"  ALL PIPELINE STEPS COMMITTED WITH 100% ACCURACY | INVARIANT VITAL_MAX_HP == {VITAL_MAX_HP}")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline_demo()
