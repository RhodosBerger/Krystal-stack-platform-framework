# ==============================================================================
# KRYSTAL-STACK: EXTENDED THEORETICAL FORMULAS (FORMULAS 7 - 14)
# ==============================================================================
# Mathematical specification implementing the second wave of closed-form formulas:
#   Formula 7:  Quadratic Cross-Domain Transduction & Phase Space Invariant Mapping
#   Formula 8:  Plunging Artillery Ballistics & Elliptical CEP Dispersion Tensor
#   Formula 9:  4D Minkowski Space-Time Collision & Dynamic Time-Dilation Manifold
#   Formula 10: Multi-Octave Continuous Terrain Manifold & Coupled Erosion PDE
#   Formula 11: Dihedral Coxeter Group Reflections & Recursive AR Fresnel Ray Optics
#   Formula 12: Set-Theoretic Dopamine Cadence & Micro-Timing Burst Cascade
#   Formula 13: Warhammer Stochastic Wound Probability & Damage Expectation Tensor
#   Formula 14: Continuous 3D Biome Phase Space Whittaker Centroid Metric
#
# Core Invariants:
#   1. Immutable 6 Max HP Vital Rule: HP <= 6 and MaxHP = 6
#   2. Deterministic Seed Hygiene: Seed1 == Seed2 -> Output1 == Output2
#   3. Singularity-Free Continuous Operations: Apex projection for Delta < 0
# ==============================================================================

import math
from typing import Dict, List, Tuple, Any, Optional, Callable, Set

VITAL_MAX_HP: int = 6
GOLDEN_RATIO_PHI: float = 1.61803398875
INV_GOLDEN_RATIO: float = 0.61803398875
GRAVITY_G: float = 9.81


# ==============================================================================
# FORMULA 7: QUADRATIC CROSS-DOMAIN TRANSDUCTION
# ==============================================================================

# Domain parameter registry: a * x^2 + b * x + c
QUADRATIC_DOMAIN_REGISTRY: Dict[str, Dict[str, Any]] = {
    "memory_latency_ns": {"a": 85.0, "b": 15.0, "c": 5.2, "v_min": 5.0, "v_max": 120.0},
    "vram_allocation_mb": {"a": 3200.0, "b": 4800.0, "c": 256.0, "v_min": 256.0, "v_max": 8256.0},
    "gpu_clock_mhz": {"a": -400.0, "b": 1050.0, "c": 800.0, "v_min": 800.0, "v_max": 1450.0},
    "hamiltonian_h": {"a": 0.85, "b": 0.20, "c": 0.05, "v_min": 0.05, "v_max": 1.20},
    "market_price_credits": {"a": 120.0, "b": 30.0, "c": 10.0, "v_min": 10.0, "v_max": 160.0},
    "vital_hp": {"a": -3.5, "b": -1.5, "c": 6.0, "v_min": 0.0, "v_max": 6.0},
    "terrain_altitude_m": {"a": 1200.0, "b": 600.0, "c": 50.0, "v_min": 50.0, "v_max": 1850.0},
    "system_entropy_s": {"a": 0.45, "b": 0.55, "c": 0.02, "v_min": 0.02, "v_max": 1.02}
}

def evaluate_quadratic_transduction(domain_id: str, x: float) -> float:
    """
    Forward projection: V_i(x) = a_i * x^2 + b_i * x + c_i.
    Strictly clamps x in [0.0, 1.0] and enforces VITAL_MAX_HP = 6.
    """
    if domain_id not in QUADRATIC_DOMAIN_REGISTRY:
        raise KeyError(f"Unknown quadratic domain: {domain_id}")

    spec = QUADRATIC_DOMAIN_REGISTRY[domain_id]
    x_clamped = max(0.0, min(1.0, float(x)))
    val = spec["a"] * (x_clamped ** 2) + spec["b"] * x_clamped + spec["c"]
    bounded = max(spec["v_min"], min(spec["v_max"], val))

    # Strict invariant enforcement for vital health
    if domain_id == "vital_hp":
        bounded = min(float(VITAL_MAX_HP), max(0.0, bounded))

    return round(bounded, 4)

def solve_quadratic_latent_x(domain_id: str, value: float) -> Tuple[float, float, bool]:
    """
    Inverse extraction: a * x^2 + b * x + (c - V) = 0.
    Returns: (selected_x, discriminant_delta, is_real)
    If Delta < 0, projects onto parabolic apex x = -b / (2a) to prevent complex singularities.
    """
    if domain_id not in QUADRATIC_DOMAIN_REGISTRY:
        raise KeyError(f"Unknown quadratic domain: {domain_id}")

    spec = QUADRATIC_DOMAIN_REGISTRY[domain_id]
    v_target = max(spec["v_min"], min(spec["v_max"], float(value)))
    a = spec["a"]
    b = spec["b"]
    c_eff = spec["c"] - v_target

    # Degenerate linear case
    if abs(a) < 1e-12:
        if abs(b) < 1e-12:
            return (0.0, 0.0, True)
        x_lin = -c_eff / b
        return (round(max(0.0, min(1.0, x_lin)), 5), 0.0, True)

    delta = (b ** 2) - 4.0 * a * c_eff

    if delta < 0.0:
        # Apex projection
        apex_x = -b / (2.0 * a)
        clamped_apex = max(0.0, min(1.0, apex_x))
        return (round(clamped_apex, 5), round(delta, 5), False)

    sqrt_delta = math.sqrt(delta)
    r1 = (-b + sqrt_delta) / (2.0 * a)
    r2 = (-b - sqrt_delta) / (2.0 * a)

    # Pick valid root in [0, 1]
    r1_valid = 0.0 <= r1 <= 1.0
    r2_valid = 0.0 <= r2 <= 1.0

    if r1_valid and r2_valid:
        chosen = r1 if a > 0 else r2
    elif r1_valid:
        chosen = r1
    elif r2_valid:
        chosen = r2
    else:
        # Clamped nearest
        d1 = min(abs(r1 - 0.0), abs(r1 - 1.0))
        d2 = min(abs(r2 - 0.0), abs(r2 - 1.0))
        chosen = r1 if d1 <= d2 else r2
        chosen = max(0.0, min(1.0, chosen))

    return (round(chosen, 5), round(delta, 5), True)


# ==============================================================================
# FORMULA 8: PLUNGING ARTILLERY BALLISTICS & ELLIPTICAL CEP DISPERSION
# ==============================================================================

def calculate_plunging_artillery_ballistics(
    v0: float = 45.0,
    elevation_deg: float = 65.0,
    distance_m: float = 120.0
) -> Dict[str, Any]:
    """
    Computes high-angle plunging artillery flight dynamics:
    - Flight time: t = (2 * v0 * sin(phi)) / g
    - Apex height: h = (v0^2 * sin^2(phi)) / (2 * g)
    - Elliptical CEP axes (lateral and longitudinal)
    """
    phi_clamped = max(15.0, min(89.0, float(elevation_deg)))
    phi_rad = math.radians(phi_clamped)
    sin_phi = math.sin(phi_rad)
    cos_phi = math.cos(phi_rad)

    flight_time = (2.0 * v0 * sin_phi) / GRAVITY_G
    apex_height = ((v0 * sin_phi) ** 2) / (2.0 * GRAVITY_G)

    cot_phi = cos_phi / max(0.01, sin_phi)
    cep_longitudinal = round(distance_m * 0.08 * cot_phi, 3)
    cep_lateral = round(distance_m * 0.05, 3)

    return {
        "flight_time_sec": round(flight_time, 3),
        "apex_height_meters": round(apex_height, 3),
        "cep_lateral_meters": cep_lateral,
        "cep_longitudinal_meters": cep_longitudinal,
        "is_plunging": phi_clamped >= 60.0
    }

def evaluate_mortar_dispersion_ellipse(
    distance_m: float,
    elevation_deg: float
) -> Tuple[float, float]:
    """Returns (CEP_lateral, CEP_longitudinal) in meters."""
    res = calculate_plunging_artillery_ballistics(elevation_deg=elevation_deg, distance_m=distance_m)
    return (res["cep_lateral_meters"], res["cep_longitudinal_meters"])


# ==============================================================================
# FORMULA 9: MINKOWSKI 4D SPACE-TIME COLLISION & TIME-DILATION
# ==============================================================================

def detect_minkowski_spacetime_intersection(
    p1_samples: List[Tuple[float, float, float]],
    p2_samples: List[Tuple[float, float, float]],
    radius: float = 0.75
) -> Optional[Dict[str, Any]]:
    """
    Checks space-time intersection between two trajectories sampled across t in [0, 1].
    Returns intersection timestamp and focus center if within radius.
    """
    n = min(len(p1_samples), len(p2_samples))
    if n < 2:
        return None

    min_dist = float("inf")
    best_idx = -1

    for i in range(n):
        p1 = p1_samples[i]
        p2 = p2_samples[i]
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dz = p1[2] - p2[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist < min_dist:
            min_dist = dist
            best_idx = i

    if min_dist <= radius and best_idx >= 0:
        t_impact = best_idx / float(n - 1)
        p1_hit = p1_samples[best_idx]
        p2_hit = p2_samples[best_idx]
        focus = (
            round((p1_hit[0] + p2_hit[0]) / 2.0, 3),
            round((p1_hit[1] + p2_hit[1]) / 2.0, 3),
            round((p1_hit[2] + p2_hit[2]) / 2.0, 3)
        )
        return {
            "intersects": True,
            "impact_time_normalized": round(t_impact, 4),
            "spatial_distance": round(min_dist, 4),
            "focus_center": focus
        }

    return None

def evaluate_bullet_time_dilation(
    t: float,
    t_impact: float,
    window_sec: float = 0.50,
    min_dilation: float = 0.12
) -> float:
    """
    Computes continuous time-dilation scale mu(t) in [min_dilation, 1.0].
    At t == t_impact, simulation time dilates to min_dilation.
    """
    dt = abs(t - t_impact)
    if dt >= window_sec:
        return 1.0

    normalized_dt = dt / max(1e-4, window_sec)
    # Quadratic recovery from slow-motion
    scale = min_dilation + (1.0 - min_dilation) * (normalized_dt ** 2)
    return round(max(min_dilation, min(1.0, scale)), 4)


# ==============================================================================
# FORMULA 10: MULTI-OCTAVE TERRAIN MANIFOLD & COUPLED EROSION
# ==============================================================================

def _pseudo_hash2d(x: float, y: float, seed: int) -> float:
    """Deterministic integer hash in [0, 1)."""
    xi = int(math.floor(x)) + seed
    yi = int(math.floor(y)) + seed * 31
    n = (xi * 374761393 + yi * 668265263) ^ 0x5bf03635
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return (n & 0x7fffffff) / 2147483648.0

def _quintic_smoothstep(t: float) -> float:
    t_c = max(0.0, min(1.0, t))
    return t_c * t_c * t_c * (t_c * (t_c * 6.0 - 15.0) + 10.0)

def _value_noise_2d(x: float, y: float, seed: int) -> float:
    x0 = math.floor(x)
    y0 = math.floor(y)
    tx = _quintic_smoothstep(x - x0)
    ty = _quintic_smoothstep(y - y0)

    v00 = _pseudo_hash2d(x0, y0, seed)
    v10 = _pseudo_hash2d(x0 + 1.0, y0, seed)
    v01 = _pseudo_hash2d(x0, y0 + 1.0, seed)
    v11 = _pseudo_hash2d(x0 + 1.0, y0 + 1.0, seed)

    vx0 = v00 + tx * (v10 - v00)
    vx1 = v01 + tx * (v11 - v01)
    return vx0 + ty * (vx1 - vx0)

def evaluate_multioctave_terrain_elevation(
    x: float,
    z: float,
    seed: int = 42,
    octaves: int = 5,
    height_scale: float = 2.5
) -> float:
    """
    Evaluates multi-octave fBm procedural elevation with deterministic seed.
    """
    total = 0.0
    freq = 0.06
    amp = 1.0
    max_amp = 0.0

    for i in range(octaves):
        n = _value_noise_2d(x * freq, z * freq, seed + i * 17)
        total += n * amp
        max_amp += amp
        amp *= 0.5
        freq *= 2.0

    norm_h = total / max_amp if max_amp > 0 else 0.0
    return round((norm_h * 2.0 - 1.0) * height_scale, 4)

def calculate_coupled_terrain_erosion(
    x: float,
    z: float,
    seed: int = 42,
    erosion_strength: float = 0.4
) -> Tuple[float, float, float]:
    """
    Evaluates terrain elevation with coupled thermal weathering and hydraulic incision.
    Returns: (eroded_height, slope, sediment_factor)
    """
    eps = 0.15
    raw_h = evaluate_multioctave_terrain_elevation(x, z, seed=seed)
    hx_p = evaluate_multioctave_terrain_elevation(x + eps, z, seed=seed)
    hx_m = evaluate_multioctave_terrain_elevation(x - eps, z, seed=seed)
    hz_p = evaluate_multioctave_terrain_elevation(x, z + eps, seed=seed)
    hz_m = evaluate_multioctave_terrain_elevation(x, z - eps, seed=seed)

    dx = (hx_p - hx_m) / (2.0 * eps)
    dz = (hz_p - hz_m) / (2.0 * eps)
    slope = math.sqrt(dx * dx + dz * dz)

    # Thermal talus slip
    talus = max(0.0, slope - 0.68) * 0.40

    # Hydraulic Laplacian curvature incision
    laplacian = ((hx_p + hx_m - 2.0 * raw_h) + (hz_p + hz_m - 2.0 * raw_h)) / (eps * eps)
    incision = max(-1.0, min(1.0, laplacian * 0.15)) * erosion_strength

    eroded_h = raw_h - talus * erosion_strength + incision * 0.25
    sediment = max(0.0, min(1.0, 1.0 - slope))

    return (round(eroded_h, 4), round(slope, 4), round(sediment, 4))


# ==============================================================================
# FORMULA 11: DIHEDRAL COXETER REFLECTIONS & FRESNEL RAY OPTICS
# ==============================================================================

def evaluate_dihedral_coxeter_fold(
    x: float,
    y: float,
    num_folds: int = 6
) -> Tuple[float, float]:
    """
    Folds 2D coordinates into the fundamental wedge domain of dihedral group D_N:
    theta_period = 2 * pi / N.
    """
    n_folds = max(2, min(16, int(num_folds)))
    r = math.hypot(x, y)
    if r < 1e-6:
        return (0.0, 0.0)

    theta = math.atan2(y, x)
    if theta < 0.0:
        theta += 2.0 * math.pi

    period = (2.0 * math.pi) / float(n_folds)
    half_period = period * 0.5

    # Modulo fold into fundamental domain
    mod_theta = theta % period
    fold_theta = abs(mod_theta - half_period)

    fx = r * math.cos(fold_theta)
    fy = r * math.sin(fold_theta)
    return (round(fx, 4), round(fy, 4))

def calculate_schlick_fresnel_and_chromatic(
    cos_theta: float,
    r0: float = 0.04,
    dispersion_factor: float = 0.02
) -> Dict[str, float]:
    """
    Computes Schlick's Fresnel reflectance with Cauchy chromatic separation:
    R(theta) = R0 + (1 - R0) * (1 - cos(theta))^5
    """
    c = max(0.0, min(1.0, float(cos_theta)))
    fresnel_base = r0 + (1.0 - r0) * ((1.0 - c) ** 5)

    # Chromatic RGB shift
    fresnel_red = max(0.0, min(1.0, fresnel_base * (1.0 - dispersion_factor)))
    fresnel_green = max(0.0, min(1.0, fresnel_base))
    fresnel_blue = max(0.0, min(1.0, fresnel_base * (1.0 + dispersion_factor)))

    return {
        "fresnel_base": round(fresnel_base, 4),
        "fresnel_red": round(fresnel_red, 4),
        "fresnel_green": round(fresnel_green, 4),
        "fresnel_blue": round(fresnel_blue, 4)
    }


# ==============================================================================
# FORMULA 12: SET-THEORETIC DOPAMINE CADENCE & COMBO CASCADE
# ==============================================================================

def evaluate_dopamine_set_matrix_criticality(
    allied: List[str],
    enemies: List[str],
    uncovered: List[str],
    cc: List[str]
) -> Dict[str, Any]:
    """
    Computes set-theoretic critical vulnerability:
    U_crit = (U_enemy ∩ U_uncovered) ∪ (U_enemy ∩ U_cc)
    """
    set_enemies = set(enemies)
    set_uncovered = set(uncovered)
    set_cc = set(cc)

    crit_uncovered = set_enemies & set_uncovered
    crit_cc = set_enemies & set_cc
    crit_all = crit_uncovered | crit_cc

    return {
        "critical_vulnerable_units": sorted(list(crit_all)),
        "uncovered_enemy_count": len(crit_uncovered),
        "crowd_controlled_enemy_count": len(crit_cc),
        "shatter_ready": len(crit_uncovered & crit_cc) > 0
    }

def calculate_cadence_combo_multiplier(timestamps_sec: List[float]) -> Dict[str, Any]:
    """
    Evaluates micro-timing intervals to award dopamine combo multiplier and mana refund.
    Thresholds: Perfect Parry (<= 0.12s), Flow State (<= 0.35s), Rapid Tempo (<= 0.85s).
    """
    if len(timestamps_sec) < 2:
        return {
            "cadence_rating": "INITIAL_ACTION",
            "combo_count": len(timestamps_sec),
            "multiplier": 1.0,
            "mana_refund": 0,
            "dopamine_overdrive": False
        }

    deltas = [timestamps_sec[i] - timestamps_sec[i - 1] for i in range(1, len(timestamps_sec))]
    avg_delta = sum(deltas) / len(deltas)
    rapid_count = sum(1 for dt in deltas if 0.0 < dt <= 0.85)

    if avg_delta <= 0.12:
        rating = "PERFECT_PARRY_OVERDRIVE"
    elif avg_delta <= 0.35:
        rating = "FLOW_STATE_CADENCE"
    elif avg_delta <= 0.85:
        rating = "RAPID_TEMPO"
    else:
        rating = "STANDARD_CADENCE"

    is_overdrive = rapid_count >= 2 and avg_delta <= 0.85
    multiplier = 1.0 + (0.15 * rapid_count if is_overdrive else 0.0)
    mana_refund = min(4, math.floor(0.75 * rapid_count)) if is_overdrive else 0

    return {
        "cadence_rating": rating,
        "combo_count": len(timestamps_sec),
        "avg_interval_sec": round(avg_delta, 3),
        "multiplier": round(multiplier, 2),
        "mana_refund": mana_refund,
        "dopamine_overdrive": is_overdrive
    }


# ==============================================================================
# FORMULA 13: WARHAMMER STOCHASTIC WOUND PROBABILITY & DAMAGE EXPECTATION
# ==============================================================================

def calculate_warhammer_wound_probability(strength: int, toughness: int) -> float:
    """
    Evaluates classical S vs T piecewise wound probability:
      S >= 2T  -> 5/6 (2+)
      S > T    -> 4/6 (3+)
      S == T   -> 3/6 (4+)
      S < T    -> 2/6 (5+)
      S <= T//2-> 1/6 (6+)
    """
    s = max(1, int(strength))
    t = max(1, int(toughness))

    if s >= 2 * t:
        return 5.0 / 6.0
    elif s > t:
        return 4.0 / 6.0
    elif s == t:
        return 3.0 / 6.0
    elif s <= (t // 2):
        return 1.0 / 6.0
    else:
        return 2.0 / 6.0

def calculate_damage_expectation_and_variance(
    attacks: int,
    bs_ws_skill: int,
    strength: int,
    toughness: int,
    sv: int,
    ap: int,
    invuln: Optional[int] = None,
    damage_per_wound: int = 1
) -> Dict[str, float]:
    """
    Evaluates analytic damage expectation E[D] and variance Var(D):
    p_conv = P(Hit) * P(Wound) * P(Fail Save)
    E[D] = A * p_conv * D
    Var(D) = A * p_conv * (1 - p_conv) * D^2
    """
    a = max(1, int(attacks))
    # Hit probability: e.g. BS/WS 3+ -> 4/6
    skill = max(2, min(6, int(bs_ws_skill)))
    p_hit = (7.0 - skill) / 6.0

    p_wound = calculate_warhammer_wound_probability(strength, toughness)

    # Save threshold
    modified_sv = sv - ap
    effective_sv = min(modified_sv, invuln) if invuln is not None else modified_sv
    save_threshold = max(2, min(7, effective_sv))

    if save_threshold >= 7:
        p_save_success = 0.0
    else:
        p_save_success = (7.0 - save_threshold) / 6.0

    p_fail_save = 1.0 - p_save_success
    p_conv = p_hit * p_wound * p_fail_save

    d_val = max(1, int(damage_per_wound))
    expected_damage = a * p_conv * d_val
    variance = a * p_conv * (1.0 - p_conv) * (d_val ** 2)
    std_dev = math.sqrt(variance)

    return {
        "p_hit": round(p_hit, 4),
        "p_wound": round(p_wound, 4),
        "p_fail_save": round(p_fail_save, 4),
        "p_conversion": round(p_conv, 4),
        "expected_damage": round(expected_damage, 4),
        "variance": round(variance, 4),
        "std_dev": round(std_dev, 4)
    }


# ==============================================================================
# FORMULA 14: WHITTAKER 3D BIOME PHASE SPACE METRIC
# ==============================================================================

WHITTAKER_BIOME_CENTROIDS: Dict[str, Tuple[float, float, float]] = {
    "CYBERPUNK_WASTELAND": (0.10, -0.70, 0.95),
    "CRYSTALLINE_HIGHLANDS": (-0.60, 0.40, 0.60),
    "VOLCANIC_CRAGS": (0.90, -0.80, 0.20),
    "BIOMECHANICAL_HIVE": (0.40, 0.80, 0.90),
    "ALCHEMICAL_ETHER_PLAINS": (-0.20, 0.10, 0.40)
}

def evaluate_biome_partition_of_unity(
    temp: float,
    moisture: float,
    anomaly: float
) -> Dict[str, float]:
    """
    Computes inverse distance squared weights for all biomes in phase space:
    Normalized so sum(weights) == 1.0.
    """
    t = max(-1.0, min(1.0, float(temp)))
    m = max(-1.0, min(1.0, float(moisture)))
    a = max(0.0, min(1.0, float(anomaly)))

    raw_weights: Dict[str, float] = {}
    for b_id, centroid in WHITTAKER_BIOME_CENTROIDS.items():
        dist_sq = (t - centroid[0]) ** 2 + (m - centroid[1]) ** 2 + (a - centroid[2]) ** 2
        raw_weights[b_id] = 1.0 / (dist_sq + 1e-4)

    total_w = sum(raw_weights.values())
    norm_weights = {b_id: round(w / total_w, 4) for b_id, w in raw_weights.items()}
    return norm_weights

def classify_whittaker_biome_phase_space(
    temp: float,
    moisture: float,
    anomaly: float
) -> Dict[str, Any]:
    """
    Determines dominant biome and returns normalized partition weights.
    """
    weights = evaluate_biome_partition_of_unity(temp, moisture, anomaly)
    dominant_biome = max(weights.items(), key=lambda item: item[1])[0]

    return {
        "dominant_biome": dominant_biome,
        "confidence": weights[dominant_biome],
        "weights": weights
    }
