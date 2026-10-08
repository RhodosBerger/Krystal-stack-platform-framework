"""
Krystal-Stack Standalone Verification & Profiling: Extended Theoretical Functions (Formulas 7 - 14)
===================================================================================================
Executes exhaustive tests, mathematical invariant checks, deterministic seed verification,
profiling latency benchmarks, and ASCII visual comparisons.
"""

import sys
import os
import time
import math
import json

# Setup encoding
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from krystal_web_hub.economic_engine.extended_theoretical_functions import (
    VITAL_MAX_HP,
    GOLDEN_RATIO_PHI,
    evaluate_quadratic_transduction,
    solve_quadratic_latent_x,
    calculate_plunging_artillery_ballistics,
    evaluate_mortar_dispersion_ellipse,
    detect_minkowski_spacetime_intersection,
    evaluate_bullet_time_dilation,
    evaluate_multioctave_terrain_elevation,
    calculate_coupled_terrain_erosion,
    evaluate_dihedral_coxeter_fold,
    calculate_schlick_fresnel_and_chromatic,
    evaluate_dopamine_set_matrix_criticality,
    calculate_cadence_combo_multiplier,
    calculate_warhammer_wound_probability,
    calculate_damage_expectation_and_variance,
    evaluate_biome_partition_of_unity,
    classify_whittaker_biome_phase_space
)

def run_all_extended_verifications() -> Dict[str, Any]:
    audit_results: Dict[str, Any] = {
        "timestamp": time.time(),
        "status": "PASS",
        "formulas_tested": 8,
        "benchmarks_us": {},
        "visual_ascii": {},
        "invariants_verified": []
    }

    print("=" * 80)
    print(" [KRYSTAL-STACK] EXTENDED THEORETICAL FORMULAS VERIFICATION & PROFILING")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # FORMULA 7: Quadratic Cross-Domain Transduction
    # --------------------------------------------------------------------------
    print("\n--- [FORMULA 7] Quadratic Cross-Domain Transduction ---")
    t0 = time.perf_counter()
    # Check Max HP rule at 100 intervals
    for i in range(101):
        x = i / 100.0
        hp = evaluate_quadratic_transduction("vital_hp", x)
        assert hp <= VITAL_MAX_HP, f"HP {hp} exceeded VITAL_MAX_HP {VITAL_MAX_HP}"
        assert hp >= 0.0, f"HP {hp} fell below zero"

    # Forward-inverse consistency
    val_mem = evaluate_quadratic_transduction("memory_latency_ns", 0.65)
    rec_x, delta, is_real = solve_quadratic_latent_x("memory_latency_ns", val_mem)
    assert is_real and abs(rec_x - 0.65) < 0.01

    # Apex projection for out-of-envelope value
    apex_x, delta_neg, is_real_apex = solve_quadratic_latent_x("gpu_clock_mhz", 3500.0)
    assert not is_real_apex and delta_neg < 0.0 and 0.0 <= apex_x <= 1.0
    t7_us = (time.perf_counter() - t0) * 1e6
    audit_results["benchmarks_us"]["Formula_7_QuadraticTransduction"] = round(t7_us, 2)
    audit_results["invariants_verified"].append("Vital_Max_HP_6_Invariant_Preserved")
    print(f"  [PASS] Formula 7: HP <= 6 strictly clamped, inverse solved with apex projection. ({t7_us:.1f} us)")

    # --------------------------------------------------------------------------
    # FORMULA 8: Plunging Artillery Ballistics & Elliptical CEP
    # --------------------------------------------------------------------------
    print("\n--- [FORMULA 8] Plunging Artillery Ballistics & Elliptical CEP ---")
    t0 = time.perf_counter()
    b_res = calculate_plunging_artillery_ballistics(v0=45.0, elevation_deg=68.0, distance_m=150.0)
    assert b_res["is_plunging"]
    assert b_res["flight_time_sec"] > 8.0
    assert b_res["apex_height_meters"] > 80.0
    cep_lat, cep_long = evaluate_mortar_dispersion_ellipse(150.0, 68.0)
    assert cep_lat > 0.0 and cep_long > 0.0
    t8_us = (time.perf_counter() - t0) * 1e6
    audit_results["benchmarks_us"]["Formula_8_PlungingBallistics"] = round(t8_us, 2)
    print(f"  [PASS] Formula 8: Plunging ballistic flight time={b_res['flight_time_sec']}s, Apex={b_res['apex_height_meters']}m, CEP=({cep_lat}m, {cep_long}m). ({t8_us:.1f} us)")

    # --------------------------------------------------------------------------
    # FORMULA 9: 4D Minkowski Space-Time Collision & Time-Dilation
    # --------------------------------------------------------------------------
    print("\n--- [FORMULA 9] 4D Minkowski Space-Time Collision & Time-Dilation ---")
    t0 = time.perf_counter()
    traj_melee = [(float(i) * 0.5, 1.0, 0.0) for i in range(21)]
    traj_shell = [(10.0 - float(i) * 0.5, 1.0, 0.0) for i in range(21)]
    hit = detect_minkowski_spacetime_intersection(traj_melee, traj_shell, radius=0.8)
    assert hit is not None and hit["intersects"]
    assert abs(hit["impact_time_normalized"] - 0.50) < 0.05

    mu_slow = evaluate_bullet_time_dilation(0.50, t_impact=0.50)
    mu_normal = evaluate_bullet_time_dilation(1.20, t_impact=0.50)
    assert mu_slow == 0.12
    assert mu_normal == 1.0
    t9_us = (time.perf_counter() - t0) * 1e6
    audit_results["benchmarks_us"]["Formula_9_MinkowskiBulletTime"] = round(t9_us, 2)
    print(f"  [PASS] Formula 9: Intersection detected at t={hit['impact_time_normalized']}, Focus={hit['focus_center']}, Time-Dilation mu in [{mu_slow}, {mu_normal}]. ({t9_us:.1f} us)")

    # --------------------------------------------------------------------------
    # FORMULA 10: Multi-Octave Continuous Terrain Manifold & Coupled Erosion
    # --------------------------------------------------------------------------
    print("\n--- [FORMULA 10] Multi-Octave Terrain & Coupled Erosion ---")
    t0 = time.perf_counter()
    # Test deterministic seed replay
    h1 = evaluate_multioctave_terrain_elevation(25.4, 78.1, seed=42)
    h2 = evaluate_multioctave_terrain_elevation(25.4, 78.1, seed=42)
    assert h1 == h2, "Seed determinism failed!"

    eroded_h, slope, sediment = calculate_coupled_terrain_erosion(25.4, 78.1, seed=42)
    assert 0.0 <= sediment <= 1.0
    t10_us = (time.perf_counter() - t0) * 1e6
    audit_results["benchmarks_us"]["Formula_10_ContinuousTerrainErosion"] = round(t10_us, 2)
    audit_results["invariants_verified"].append("Seed_Hygiene_Deterministic_Replay")
    print(f"  [PASS] Formula 10: Deterministic seed replay verified. Eroded H={eroded_h}m, Slope={slope}, Sediment={sediment}. ({t10_us:.1f} us)")

    # --------------------------------------------------------------------------
    # FORMULA 11: Dihedral Coxeter Group Reflections & Fresnel Ray Optics
    # --------------------------------------------------------------------------
    print("\n--- [FORMULA 11] Dihedral Coxeter Group Reflections & Fresnel Optics ---")
    t0 = time.perf_counter()
    fx, fy = evaluate_dihedral_coxeter_fold(3.0, 4.0, num_folds=8)
    assert abs(math.hypot(fx, fy) - 5.0) < 0.01, "Radius metric not preserved in fold!"

    fres = calculate_schlick_fresnel_and_chromatic(cos_theta=0.10, r0=0.04, dispersion_factor=0.03)
    assert fres["fresnel_blue"] > fres["fresnel_base"] > fres["fresnel_red"]
    t11_us = (time.perf_counter() - t0) * 1e6
    audit_results["benchmarks_us"]["Formula_11_DihedralCoxeterFresnel"] = round(t11_us, 2)
    print(f"  [PASS] Formula 11: Coxeter D8 symmetry fold preserves norm r=5.0; Fresnel chromatic shift blue={fres['fresnel_blue']}, red={fres['fresnel_red']}. ({t11_us:.1f} us)")

    # --------------------------------------------------------------------------
    # FORMULA 12: Set-Theoretic Dopamine Cadence & Combo Cascade
    # --------------------------------------------------------------------------
    print("\n--- [FORMULA 12] Set-Theoretic Dopamine Cadence & Combo Cascade ---")
    t0 = time.perf_counter()
    crit_eval = evaluate_dopamine_set_matrix_criticality(
        allied=["hero_vanguard"],
        enemies=["cultist_1", "cultist_2", "beast_lord"],
        uncovered=["cultist_1"],
        cc=["beast_lord"]
    )
    assert crit_eval["critical_vulnerable_units"] == ["beast_lord", "cultist_1"]

    combo = calculate_cadence_combo_multiplier([10.0, 10.11, 10.22, 10.35])
    assert combo["dopamine_overdrive"]
    assert combo["multiplier"] > 1.3
    assert combo["mana_refund"] >= 2
    t12_us = (time.perf_counter() - t0) * 1e6
    audit_results["benchmarks_us"]["Formula_12_DopamineCadenceLattice"] = round(t12_us, 2)
    print(f"  [PASS] Formula 12: Set matrix U_crit verified, Cadence={combo['cadence_rating']}, Mult={combo['multiplier']}x, Refund={combo['mana_refund']}. ({t12_us:.1f} us)")

    # --------------------------------------------------------------------------
    # FORMULA 13: Warhammer Stochastic Wound Probability & Damage Expectation
    # --------------------------------------------------------------------------
    print("\n--- [FORMULA 13] Warhammer Stochastic Wound Probability & Expectation ---")
    t0 = time.perf_counter()
    assert abs(calculate_warhammer_wound_probability(8, 4) - 5.0 / 6.0) < 1e-4
    assert abs(calculate_warhammer_wound_probability(3, 4) - 2.0 / 6.0) < 1e-4

    stat = calculate_damage_expectation_and_variance(
        attacks=12, bs_ws_skill=3, strength=6, toughness=4, sv=3, ap=2, invuln=4, damage_per_wound=3
    )
    assert stat["expected_damage"] > 0.0
    assert abs(stat["std_dev"] - math.sqrt(stat["variance"])) < 1e-4
    t13_us = (time.perf_counter() - t0) * 1e6
    audit_results["benchmarks_us"]["Formula_13_WarhammerStochasticWound"] = round(t13_us, 2)
    print(f"  [PASS] Formula 13: Piecewise S/T matrix verified, E[D]={stat['expected_damage']}, Var={stat['variance']}, StdDev={stat['std_dev']}. ({t13_us:.1f} us)")

    # --------------------------------------------------------------------------
    # FORMULA 14: Whittaker 3D Biome Phase Space Metric
    # --------------------------------------------------------------------------
    print("\n--- [FORMULA 14] Whittaker 3D Biome Phase Space Metric ---")
    t0 = time.perf_counter()
    unity = evaluate_biome_partition_of_unity(temp=0.85, moisture=-0.75, anomaly=0.25)
    sum_unity = sum(unity.values())
    assert abs(sum_unity - 1.0) < 0.02, f"Partition of unity sum {sum_unity} != 1.0"

    biome_class = classify_whittaker_biome_phase_space(temp=0.85, moisture=-0.75, anomaly=0.25)
    assert biome_class["dominant_biome"] == "VOLCANIC_CRAGS"
    t14_us = (time.perf_counter() - t0) * 1e6
    audit_results["benchmarks_us"]["Formula_14_WhittakerBiomeMetric"] = round(t14_us, 2)
    audit_results["invariants_verified"].append("Partition_of_Unity_Sum_Equals_One")
    print(f"  [PASS] Formula 14: Partition of unity sum={sum_unity:.4f}, Dominant={biome_class['dominant_biome']} ({biome_class['confidence']*100:.1f}%). ({t14_us:.1f} us)")

    # --------------------------------------------------------------------------
    # VISUAL COMPARISON: ASCII RENDERING
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" [VISUAL COMPARISON] MATHEMATICAL TERRAIN & TRAJECTORY ASCII RASTER")
    print("=" * 80)

    # 1. 2D Terrain Elevation Profile
    terrain_profile = []
    width = 40
    for col in range(width):
        wx = (col - width // 2) * 2.0
        elev = evaluate_multioctave_terrain_elevation(wx, 0.0, seed=42, height_scale=5.0)
        # map elev from [-5, 5] to [0, 10]
        char_idx = int(max(0, min(9, (elev + 5.0))))
        terrain_profile.append(" .:-=+*#%@"[char_idx])
    terrain_str = "".join(terrain_profile)
    print(f"Terrain Slice (X in [-40, 40]m, Z=0): [{terrain_str}]")
    audit_results["visual_ascii"]["terrain_slice"] = terrain_str

    # 2. Mortar Plunging Arc
    arc_lines = []
    h_max = b_res["apex_height_meters"]
    t_flight = b_res["flight_time_sec"]
    for row in range(6, -1, -1):
        line = ""
        thresh_h = (row / 6.0) * h_max
        for step in range(30):
            t = (step / 29.0) * t_flight
            # Parabolic height at time t: y(t) = v0*sin(phi)*t - 0.5*g*t^2
            phi_rad = math.radians(68.0)
            y_t = max(0.0, 45.0 * math.sin(phi_rad) * t - 0.5 * 9.81 * (t ** 2))
            if abs(y_t - thresh_h) < (h_max / 10.0):
                line += "*"
            else:
                line += " "
        arc_lines.append(f"{thresh_h:5.1f}m |{line}|")
    mortar_ascii = "\n".join(arc_lines)
    print("\nMortar Plunging Arc (Apex ~88m):")
    print(mortar_ascii)
    audit_results["visual_ascii"]["mortar_arc"] = mortar_ascii

    # Write audit JSON to scratch
    scratch_dir = os.path.join(ROOT_DIR, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    out_path = os.path.join(scratch_dir, "extended_formulas_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f" [COMPLETE] All 8 extended theoretical formulas verified and profiled.")
    print(f" Audit log saved to: {out_path}")
    print("=" * 80)
    return audit_results

if __name__ == "__main__":
    run_all_extended_verifications()
