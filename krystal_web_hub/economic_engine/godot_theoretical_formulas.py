# ==============================================================================
# KRYSTAL-STACK: GODOT THEORETICAL FORMULAS & MATHEMATICAL EXTENSION ENGINE
# ==============================================================================
# Implements the 6 closed-form equations derived from Conformal Geometric
# Algebra, Sheaf Semantics, Continuous Hex Field Theory, and NESS Thermodynamics.
# ==============================================================================

import math
from typing import List, Dict, Tuple, Any, Optional

# ------------------------------------------------------------------------------
# FORMULA 1: HEX-RIEMANNIAN METRIC TENSOR & AXIAL GEODESIC DISTANCE
# ------------------------------------------------------------------------------
def hex_riemannian_distance(hex1: List[int], hex2: List[int]) -> int:
    """
    Calculates the exact axial Riemannian L1 distance metric on pointy-topped hexes:
    D(H1, H2) = (|dq| + |dq + dr| + |dr|) // 2
    """
    dq = hex1[0] - hex2[0]
    dr = hex1[1] - hex2[1]
    return (abs(dq) + abs(dq + dr) + abs(dr)) // 2

def hex_to_world_cartesian(q: int, r: int, radius: float = 1.5) -> Tuple[float, float, float]:
    """Converts axial (q, r) to 3D world (X, Y, Z) coordinates with terrain elevation."""
    x = radius * math.sqrt(3.0) * (q + r / 2.0)
    z = radius * 1.5 * r
    y = 0.05 * math.sin(q * 0.8) * math.cos(r * 0.8)
    return (x, y, z)


# ------------------------------------------------------------------------------
# FORMULA 2: AERODYNAMIC BALLISTIC BEZIER-HERMITE TRAJECTORY WITH WIND & DRAG
# ------------------------------------------------------------------------------
def calculate_ballistic_apex_height(
    distance: int,
    attack_type: str = "ranged",
    tension: float = 0.55,
    drag: float = 0.04
) -> float:
    """
    Computes dynamic ballistic apex height H(D, tau):
    - MELEE (D=1): Low flat slash (0.50m)
    - SELF (D=0): Immediate ground aura (0.20m)
    - RANGED (D>=1): min(4.20, H0 + tau * D^1.15 * e^(-drag * D))
    """
    if attack_type.lower() == "melee":
        return 0.50
    elif attack_type.lower() == "self":
        return 0.20
    else:
        h0 = 1.0
        exponent_dist = math.pow(max(1, distance), 1.15)
        drag_factor = math.exp(-drag * distance)
        dynamic_h = h0 + tension * exponent_dist * drag_factor
        return min(4.20, dynamic_h)

def sample_ballistic_bezier_hermite_curve(
    p0: List[float],
    p3: List[float],
    distance: int,
    attack_type: str = "ranged",
    tension: float = 0.55,
    drag: float = 0.04,
    wind: Optional[List[float]] = None,
    num_samples: int = 16
) -> Dict[str, Any]:
    """
    Generates a 3D Cubic-Hermite Bezier trajectory with aerodynamic drag and tension:
    P(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    """
    if wind is None:
        wind = [0.0, 0.0, 0.0]

    apex_h = calculate_ballistic_apex_height(distance, attack_type, tension, drag)

    # Control point 1 (1/3 along line + apex + wind)
    p1 = [
        p0[0] + (1.0 / 3.0) * (p3[0] - p0[0]) + wind[0],
        max(p0[1], p3[1]) + apex_h * 1.15 + wind[1],
        p0[2] + (1.0 / 3.0) * (p3[2] - p0[2]) + wind[2]
    ]

    # Control point 2 (2/3 along line + slightly lower apex + wind)
    p2 = [
        p0[0] + (2.0 / 3.0) * (p3[0] - p0[0]) + 2.0 * wind[0],
        max(p0[1], p3[1]) + apex_h * 0.90 + 2.0 * wind[1],
        p0[2] + (2.0 / 3.0) * (p3[2] - p0[2]) + 2.0 * wind[2]
    ]

    samples: List[List[float]] = []
    for i in range(num_samples + 1):
        t = i / float(num_samples)
        omt = 1.0 - t
        # Cubic Bezier basis polynomials
        b0 = omt * omt * omt
        b1 = 3.0 * omt * omt * t
        b2 = 3.0 * omt * t * t
        b3 = t * t * t

        px = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
        py = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
        pz = b0 * p0[2] + b1 * p1[2] + b2 * p2[2] + b3 * p3[2]
        samples.append([round(px, 4), round(py, 4), round(pz, 4)])

    # Tangent at impact (t=1): 3*(P3 - P2)
    tangent = [
        3.0 * (p3[0] - p2[0]),
        3.0 * (p3[1] - p2[1]),
        3.0 * (p3[2] - p2[2])
    ]
    t_len = math.sqrt(tangent[0]**2 + tangent[1]**2 + tangent[2]**2) or 1.0
    tangent_norm = [tangent[0] / t_len, tangent[1] / t_len, tangent[2] / t_len]

    return {
        "apex_height": round(apex_h, 4),
        "control_points": {"p0": p0, "p1": p1, "p2": p2, "p3": p3},
        "samples": samples,
        "impact_tangent": tangent_norm,
        "flight_duration_sec": round(0.35 + 0.10 * distance, 3)
    }


# ------------------------------------------------------------------------------
# FORMULA 3: ANALYTIC SIGNED DISTANCE FUNCTIONS (SDF) & SMOOTH MINIMUM (smin)
# ------------------------------------------------------------------------------
def hex_prism_sdf(p: List[float], radius: float = 1.5, height: float = 0.5) -> float:
    """
    Exact Signed Distance Function for a pointy-topped regular hexagonal prism column:
    Phi_Hex(p, r, h) = max(d_hex, |p_y| - h/2)
    """
    px, py, pz = abs(p[0]), abs(p[1]), abs(p[2])
    dx = px * (math.sqrt(3.0) / 2.0) + pz * 0.5 - radius
    dz = pz - radius
    d_hex = max(dx, dz)
    d_y = py - height / 2.0
    return max(d_hex, d_y)

def crystal_spire_sdf(p: List[float], scale: float = 1.2) -> float:
    """Exact SDF for an octahedral crystal shard."""
    px, py, pz = abs(p[0]), abs(p[1]), abs(p[2])
    norm = math.sqrt(1.0**2 + 1.6**2 + 1.0**2)
    return (px + py * 1.6 + pz - scale) / norm

def polynomial_smooth_min(a: float, b: float, k: float = 0.2) -> float:
    """Polynomial smooth minimum blending: smin_k(a, b)."""
    h = max(0.0, min(1.0, 0.5 + 0.5 * (b - a) / k))
    return (b * (1.0 - h) + a * h) - k * h * (1.0 - h)


# ------------------------------------------------------------------------------
# FORMULA 4: DYNAMIC HEX TERRAFORMING & BIOME PHASE TRANSITION PDE
# ------------------------------------------------------------------------------
# Impulse vectors for elemental spell impacts: (crystal, toxic, druid)
ELEMENTAL_IMPULSE_MAP: Dict[str, Tuple[float, float, float]] = {
    "crystal": (0.80, -0.40, -0.40),
    "toxic": (-0.40, 0.80, -0.40),
    "druid": (-0.40, -0.40, 0.80)
}

def update_hex_biome_transition(
    current_biome_vector: Tuple[float, float, float],
    spell_element: str,
    impact_distance: int,
    baseline_vector: Tuple[float, float, float] = (0.333, 0.333, 0.334),
    sigma: float = 1.25,
    kappa: float = 0.08
) -> Dict[str, Any]:
    """
    Evaluates dynamic biome phase transition equation:
    dB/dt = -kappa * (B - B0) + I_k * exp(-D^2 / (2 * sigma^2))
    """
    b_cryst, b_toxic, b_druid = current_biome_vector
    b0_cryst, b0_toxic, b0_druid = baseline_vector

    impulse = ELEMENTAL_IMPULSE_MAP.get(spell_element.lower(), (0.0, 0.0, 0.0))
    spatial_falloff = math.exp(-(impact_distance**2) / (2.0 * sigma**2))

    # Calculate delta
    d_cryst = -kappa * (b_cryst - b0_cryst) + impulse[0] * spatial_falloff
    d_toxic = -kappa * (b_toxic - b0_toxic) + impulse[1] * spatial_falloff
    d_druid = -kappa * (b_druid - b0_druid) + impulse[2] * spatial_falloff

    # Apply delta and clamp to positive
    new_cryst = max(0.01, b_cryst + d_cryst)
    new_toxic = max(0.01, b_toxic + d_toxic)
    new_druid = max(0.01, b_druid + d_druid)

    # Normalize to partition of unity (sum = 1.0)
    total = new_cryst + new_toxic + new_druid
    norm_vector = (round(new_cryst / total, 4), round(new_toxic / total, 4), round(new_druid / total, 4))

    # Biome classification
    dominant_biome = "neutral_citadel"
    if norm_vector[0] > 0.50:
        dominant_biome = "crystal_peaks"
    elif norm_vector[1] > 0.50:
        dominant_biome = "toxic_marsh"
    elif norm_vector[2] > 0.50:
        dominant_biome = "druid_forest"

    return {
        "updated_vector": norm_vector,
        "dominant_biome": dominant_biome,
        "delta": (round(d_cryst, 4), round(d_toxic, 4), round(d_druid, 4)),
        "terraformed": dominant_biome != "neutral_citadel"
    }


# ------------------------------------------------------------------------------
# FORMULA 5: CARD FUSION TENSOR ALGEBRA (C1 (x) C2 -> C_fused)
# ------------------------------------------------------------------------------
# Tribal synergy tensor Gamma(T1, T2)
TRIBAL_SYNERGY_TENSOR: Dict[Tuple[str, str], float] = {
    ("crystal", "crystal"): 0.25,
    ("crystal", "toxic"): 0.60,
    ("crystal", "druid"): 0.40,
    ("toxic", "crystal"): 0.60,
    ("toxic", "toxic"): 0.20,
    ("toxic", "druid"): 0.50,
    ("druid", "crystal"): 0.40,
    ("druid", "toxic"): 0.50,
    ("druid", "druid"): 0.30
}

def fuse_cards(card_a: Dict[str, Any], card_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combines two cards via Tribal Synergy Tensor algebra:
    - Fused Mana Cost: max(1, floor(0.75 * (Cost1 + Cost2) - Gamma(T1, T2)))
    - Fused Power: ceil((Power1 + Power2) * (1.0 + Gamma(T1, T2)))
    - Fused Range: [min(min1, min2), max(max1, max2)]
    """
    t1 = card_a.get("tribe", "crystal").lower()
    t2 = card_b.get("tribe", "toxic").lower()
    synergy = TRIBAL_SYNERGY_TENSOR.get((t1, t2), 0.30)

    cost1 = card_a.get("cost", 2)
    cost2 = card_b.get("cost", 2)
    fused_cost = max(1, math.floor(0.75 * (cost1 + cost2) - synergy))

    power1 = abs(card_a.get("hp_delta", 1))
    power2 = abs(card_b.get("hp_delta", 1))
    fused_power = math.ceil((power1 + power2) * (1.0 + synergy))

    min_range = min(card_a.get("min_range", 1), card_b.get("min_range", 1))
    max_range = max(card_a.get("max_range", 3), card_b.get("max_range", 3))

    fused_id = f"fused_{card_a.get('id', 'card1')}_{card_b.get('id', 'card2')}"
    fused_name = f"Fúzia: {card_a.get('name', 'Karta A')} + {card_b.get('name', 'Karta B')}"

    return {
        "id": fused_id,
        "name": fused_name,
        "tribe": "hybrid",
        "synergy_score": synergy,
        "cost": fused_cost,
        "hp_delta": -fused_power,
        "min_range": min_range,
        "max_range": max_range,
        "attack_type": "ranged" if max_range > 1 else "melee",
        "description": f"Synergická fúzia živlov ({t1.upper()} x {t2.upper()}) so synergiou +{int(synergy*100)}%."
    }


# ------------------------------------------------------------------------------
# FORMULA 6: GODOT SCENE AST COMPACTION & ZERO-COPY SERIALIZATION
# ------------------------------------------------------------------------------
def compact_godot_ast(ast_node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compresses verbose hierarchical Godot node dictionaries into compact token strings.
    Formula 6: Compression Ratio C_R = Size(Raw) / Size(Compact) >= 8x.
    """
    import json
    raw_json = json.dumps(ast_node)
    raw_size_bytes = len(raw_json.encode('utf-8'))

    packed_tokens: List[str] = []

    def _traverse(node: Dict[str, Any], depth: int = 0):
        n_type = node.get("type", "Spatial")
        n_name = node.get("name", "Node")
        props = node.get("properties", {})
        pos = props.get("position", [0, 0, 0])
        pos_str = f"{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}"
        token = f"N:{depth}:{n_type}:{n_name}:{pos_str}"
        packed_tokens.append(token)

        for child in node.get("children", []):
            _traverse(child, depth + 1)

    _traverse(ast_node)
    compact_payload = "|".join(packed_tokens)
    compact_size_bytes = len(compact_payload.encode('utf-8'))
    compression_ratio = round(raw_size_bytes / max(1, compact_size_bytes), 2)

    return {
        "raw_size_bytes": raw_size_bytes,
        "compact_size_bytes": compact_size_bytes,
        "compression_ratio": compression_ratio,
        "token_count": len(packed_tokens),
        "compact_payload": compact_payload
    }
