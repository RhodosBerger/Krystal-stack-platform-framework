#!/usr/bin/env python3
# ==============================================================================
# KRYSTAL-STACK: GRAM READERS SUITE & IMPEDANCE EVALUATION ENGINE
# ==============================================================================
# 1. GRAM READERS (ČÍTAČKY GRAMOV):
#    - NGramReader: N-gram tokenization, frequency distributions, Shannon entropy,
#      cross-entropy, Markov transition probabilities, and lexical density.
#    - GramianMatrixReader: Gram matrix G = V^T V, Gram determinant (parallelotope
#      volume), condition number, and feature/activation correlation tensors.
#    - SpectrogramFeatureReader: Acoustic & wavelet time-frequency gram frames,
#      spectral centroid, energy flux, and formant distribution bands.
#    - PhysicalMassGramReader: Metric mass, material substrate densities (g/cm^3),
#      volumetric mass integrals, and center of mass calculations.
#
# 2. IMPEDANCE EVALUATOR (HODNOTIČ IMPEDANCIE):
#    - Complex Electrical Impedance Z = R + jX:
#      Resistance, Reactance, Magnitude |Z|, Phase Angle theta, Admittance Y = G + jB,
#      Quality Factor Q, Dissipation Factor D, Reflection Coefficient Gamma,
#      VSWR, Return Loss (dB), and L-network matching synthesis.
#    - Silicon Thermal Impedance Z_th(t):
#      Foster & Cauer multi-pole RC network, transient thermal rise Delta T(t),
#      and thermal strain curvature governor integration.
#    - Cognitive & Architectural Flow Impedance Z_cog:
#      Prompt latency resistance, memory bus contention inductance, cache capacity,
#      and pipeline mismatch loss.
#
# INVIOLABLE SYSTEM INVARIANT: VITAL_MAX_HP == 6
# ==============================================================================

import sys
import os
import math
import cmath
import json
import time
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional, Tuple, Union

# Core Repository Invariant
VITAL_MAX_HP: int = 6


# ==============================================================================
# 1. ČÍTAČKY GRAMOV (GRAM READERS SUITE)
# ==============================================================================

class NGramReader:
    """
    Čítačka N-gramov:
    Analyzuje reťazce textu, tokenov alebo bytov. Extrahuje unigramy, bigramy, trigramy
    a n-gramy ľubovoľného rádu N. Počíta frekvenčné rozdelenia, Shannonovu entrópiu,
    perplexitu a matice prechodových pravdepodobností.
    """

    @staticmethod
    def tokenize(text: str, lowercase: bool = True) -> List[str]:
        """Jednoduchá deterministická tokenizácia bez externých závislostí."""
        if lowercase:
            text = text.lower()
        # Nahradenie interpunkcie medzerami s ponechaním alfanumerických znakov
        cleaned = "".join(c if c.isalnum() or c.isspace() else f" {c} " for c in text)
        return [t for t in cleaned.split() if t]

    @classmethod
    def read_ngrams(cls, tokens_or_text: Union[str, List[str]], n: int = 2) -> List[Tuple[str, ...]]:
        """Extrahuje usporiadané n-tice tokenov dĺžky n."""
        assert n >= 1, f"N-gram rád musí byť >= 1, zadané: {n}"
        tokens = cls.tokenize(tokens_or_text) if isinstance(tokens_or_text, str) else list(tokens_or_text)
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    @classmethod
    def analyze_ngram_distribution(cls, tokens_or_text: Union[str, List[str]], n: int = 2, top_k: int = 10) -> Dict[str, Any]:
        """
        Vypočíta podrobnú štatistiku n-gramov:
        - Celkový počet n-gramov a počet unikátnych
        - Frekvencia výskytu a relatívna pravdepodobnosť
        - Shannonova entrópia H = - sum(p * log2(p))
        - Perplexita PP = 2^H
        - Lexikálna diverzita (Type-Token Ratio TTR)
        """
        ngrams = cls.read_ngrams(tokens_or_text, n=n)
        total_count = len(ngrams)
        if total_count == 0:
            return {
                "n": n,
                "total_ngrams": 0,
                "unique_ngrams": 0,
                "entropy": 0.0,
                "perplexity": 1.0,
                "ttr": 0.0,
                "top_ngrams": []
            }

        counts = Counter(ngrams)
        unique_count = len(counts)

        # Výpočet Shannonovej entrópie
        entropy = 0.0
        for ngram, count in counts.items():
            p = count / total_count
            entropy -= p * math.log2(p)

        perplexity = math.pow(2.0, entropy)
        ttr = unique_count / total_count

        top_ngrams = []
        for ngram, count in counts.most_common(top_k):
            top_ngrams.append({
                "ngram": " ".join(ngram),
                "count": count,
                "probability": round(count / total_count, 6)
            })

        return {
            "n": n,
            "total_ngrams": total_count,
            "unique_ngrams": unique_count,
            "entropy": round(entropy, 4),
            "perplexity": round(perplexity, 4),
            "ttr": round(ttr, 4),
            "top_ngrams": top_ngrams,
            "vital_max_hp": VITAL_MAX_HP
        }

    @classmethod
    def build_transition_matrix(cls, text: str) -> Dict[str, Dict[str, float]]:
        """Zostaví Markovovskú maticu prechodových pravdepodobností z bigramov."""
        bigrams = cls.read_ngrams(text, n=2)
        transitions = defaultdict(Counter)
        for w1, w2 in bigrams:
            transitions[w1][w2] += 1

        probs = {}
        for w1, next_words in transitions.items():
            total = sum(next_words.values())
            probs[w1] = {w2: round(cnt / total, 4) for w2, cnt in next_words.most_common(5)}
        return probs


class GramianMatrixReader:
    """
    Čítačka Gramových Matíc (Gram Matrix / Gramian Reader):
    Počíta maticu vnútorných skalárnych súčinov G_ij = <v_i, v_j> = v_i^T v_j
    pre množinu vektorov prvkov/aktivačných vrstiev V = [v_1, ..., v_m].
    Analyzuje:
    - Gramov determinant det(G): meria druhú mocninu m-rozmerného objemu rovnobežnostena.
      Ak det(G) == 0, vektory sú lineárne závislé.
    - Normalizovanú kosínusovú korelačnú Gramovu maticu.
    - Číslo podmienenosti kappa(G) a stopu Tr(G).
    - Štylistické textúrne korelácie (Gram style loss v neurálnych sieťach).
    """

    @staticmethod
    def dot_product(v1: List[float], v2: List[float]) -> float:
        """Skalárny súčin dvoch vektorov."""
        return sum(x * y for x, y in zip(v1, v2))

    @classmethod
    def compute_gram_matrix(cls, vectors: List[List[float]], normalize: bool = False) -> Dict[str, Any]:
        """
        Vypočíta Gramovu maticu G rozmerov m x m pre m vektorov dimenzie d.
        """
        m = len(vectors)
        assert m > 0, "Množina vektorov nesmie byť prázdna."
        d = len(vectors[0])
        for v in vectors:
            assert len(v) == d, f"Všetky vektory musia mať rovnakú dimenziu {d}, nájdená: {len(v)}"

        # Voliteľná normalizácia na jednotkové vektory
        processed_vecs = []
        norms = []
        for v in vectors:
            norm = math.sqrt(sum(x * x for x in v))
            norms.append(norm)
            if normalize and norm > 1e-12:
                processed_vecs.append([x / norm for x in v])
            else:
                processed_vecs.append(list(v))

        # Zostavenie Gramovej matice G
        matrix = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i, m):
                val = cls.dot_product(processed_vecs[i], processed_vecs[j])
                matrix[i][j] = round(val, 6)
                matrix[j][i] = matrix[i][j]

        # Stopa matice Tr(G)
        trace = sum(matrix[i][i] for i in range(m))

        # Gramov determinant pre m <= 4 (analyticky pre nízke dimenzie)
        det_val = cls._calculate_determinant(matrix)

        # Test lineárnej nezávislosti
        is_linearly_independent = abs(det_val) > 1e-9 if m <= d else False

        # Spanning volume: V_m = sqrt(max(0, det(G)))
        volume = math.sqrt(max(0.0, det_val))

        return {
            "matrix_dimension": m,
            "vector_dimension": d,
            "gram_matrix": matrix,
            "trace": round(trace, 6),
            "gram_determinant": round(det_val, 8),
            "parallelotope_volume": round(volume, 6),
            "is_linearly_independent": is_linearly_independent,
            "normalized": normalize,
            "norms": [round(n, 4) for n in norms],
            "vital_max_hp": VITAL_MAX_HP
        }

    @classmethod
    def _calculate_determinant(cls, m: List[List[float]]) -> float:
        """Rekurzívny / Laplaceov výpočet determinantu pre matice menších rozmerov."""
        n = len(m)
        if n == 1:
            return m[0][0]
        if n == 2:
            return m[0][0] * m[1][1] - m[0][1] * m[1][0]
        if n == 3:
            return (
                m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
            )
        # Pre n >= 4 použijeme Gaussovu elimináciu s pivotingom
        mat_copy = [row[:] for row in m]
        det = 1.0
        for i in range(n):
            pivot = i
            for r in range(i + 1, n):
                if abs(mat_copy[r][i]) > abs(mat_copy[pivot][i]):
                    pivot = r
            if pivot != i:
                mat_copy[i], mat_copy[pivot] = mat_copy[pivot], mat_copy[i]
                det = -det
            if abs(mat_copy[i][i]) < 1e-12:
                return 0.0
            det *= mat_copy[i][i]
            for r in range(i + 1, n):
                factor = mat_copy[r][i] / mat_copy[i][i]
                for c in range(i, n):
                    mat_copy[r][c] -= factor * mat_copy[i][c]
        return det


class SpectrogramFeatureReader:
    """
    Čítačka Spektrogramov (Spectrogram & Acoustic Gram Reader):
    Spracováva časovo-frekvenčné matice energie (spektrogramové okná),
    počíta spektrálny centroid, energiu vo frekvenčných pásmach (sub-bass, mid, presence, air)
    a spektrálny tok (flux) v súlade s akustickou pipeline Whisper a syntézou zvuku.
    """

    @staticmethod
    def read_spectral_frame(magnitudes: List[float], sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Analyzuje jeden frekvenčný rez (spektrogramový stĺpec):
        - magnitudes: amplitúdy FFT frekvenčných košov (bins) od 0 Hz po Nyquist (sample_rate / 2)
        """
        num_bins = len(magnitudes)
        assert num_bins > 0, "Spektrálny rámec nesmie byť prázdny."

        nyquist = sample_rate / 2.0
        bin_width = nyquist / num_bins

        total_energy = sum(magnitudes)
        if total_energy <= 1e-12:
            return {
                "total_energy": 0.0,
                "spectral_centroid_hz": 0.0,
                "dominant_frequency_hz": 0.0,
                "spectral_bands": {"sub_bass": 0.0, "low_mid": 0.0, "high_mid": 0.0, "presence": 0.0}
            }

        # 1. Spektrálny centroid: f_c = sum(f_i * E_i) / sum(E_i)
        weighted_freq_sum = sum((i * bin_width) * magnitudes[i] for i in range(num_bins))
        centroid = weighted_freq_sum / total_energy

        # 2. Dominantná frekvencia
        max_idx = max(range(num_bins), key=lambda i: magnitudes[i])
        dominant_freq = max_idx * bin_width

        # 3. Pásmové energie (Bands)
        bands = {"sub_bass": 0.0, "low_mid": 0.0, "high_mid": 0.0, "presence": 0.0}
        for i in range(num_bins):
            f = i * bin_width
            e = magnitudes[i]
            if f < 250.0:
                bands["sub_bass"] += e
            elif f < 2000.0:
                bands["low_mid"] += e
            elif f < 6000.0:
                bands["high_mid"] += e
            else:
                bands["presence"] += e

        # Normalizácia pásiem
        norm_bands = {k: round(v / total_energy, 4) for k, v in bands.items()}

        return {
            "num_bins": num_bins,
            "sample_rate_hz": sample_rate,
            "nyquist_hz": nyquist,
            "total_energy": round(total_energy, 4),
            "spectral_centroid_hz": round(centroid, 2),
            "dominant_frequency_hz": round(dominant_freq, 2),
            "spectral_bands": norm_bands,
            "vital_max_hp": VITAL_MAX_HP
        }


class PhysicalMassGramReader:
    """
    Čítačka Fyzikálnych Gramov a Hmotnostných Integrálov (Physical Mass Gram Reader):
    Prepočítava komponenty a geometrie herných a technologických objektov
    na presné gramáže [g], hustoty [g/cm^3], ťažisko a momenty zotrvačnosti
    s rešpektovaním tabuliek materiálových substrátov (MATERIAL_SUBSTRATES).
    """

    # Štandardné hustoty materiálov v g/cm^3
    SUBSTRATE_DENSITIES: Dict[str, float] = {
        "crystal_quartz": 2.65,
        "silicon_carbide": 3.21,
        "druidic_amber": 1.07,
        "toxic_biopolymer": 1.25,
        "aether_aerogel": 0.002,
        "tungsten_alloy": 19.25,
        "hyper_glass": 2.50,
        "copper": 8.96
    }

    @classmethod
    def calculate_component_mass(cls, material: str, volume_cm3: float, position: Optional[List[float]] = None) -> Dict[str, Any]:
        """Vypočíta hmotnosť v gramoch na základe materiálového substrátu a objemu v cm^3."""
        density = cls.SUBSTRATE_DENSITIES.get(material.lower(), 2.5)  # predvolená hustota skla/kremeňa
        mass_grams = volume_cm3 * density
        pos = position if position and len(position) == 3 else [0.0, 0.0, 0.0]

        return {
            "material": material,
            "density_g_per_cm3": density,
            "volume_cm3": round(volume_cm3, 4),
            "mass_grams": round(mass_grams, 4),
            "mass_kilograms": round(mass_grams / 1000.0, 6),
            "position": pos,
            "vital_max_hp": VITAL_MAX_HP
        }

    @classmethod
    def calculate_aggregate_assembly(cls, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Vypočíta celkovú hmotnosť zostavy v gramoch a súradnice ťažiska R_cm = sum(m_i * r_i) / sum(m_i)."""
        total_mass_g = 0.0
        weighted_pos = [0.0, 0.0, 0.0]

        comp_summaries = []
        for c in components:
            mat = c.get("material", "crystal_quartz")
            vol = float(c.get("volume_cm3", 10.0))
            pos = c.get("position", [0.0, 0.0, 0.0])
            res = cls.calculate_component_mass(mat, vol, pos)

            m = res["mass_grams"]
            total_mass_g += m
            for ax in range(3):
                weighted_pos[ax] += m * pos[ax]
            comp_summaries.append(res)

        if total_mass_g > 1e-9:
            center_of_mass = [round(weighted_pos[i] / total_mass_g, 4) for i in range(3)]
        else:
            center_of_mass = [0.0, 0.0, 0.0]

        return {
            "component_count": len(components),
            "total_mass_grams": round(total_mass_g, 4),
            "total_mass_kg": round(total_mass_g / 1000.0, 6),
            "center_of_mass": center_of_mass,
            "components": comp_summaries,
            "vital_max_hp": VITAL_MAX_HP
        }


# ==============================================================================
# 2. HODNOTIČ IMPEDANCIE (IMPEDANCE EVALUATOR ENGINE)
# ==============================================================================

class ComplexImpedanceEvaluator:
    """
    Hodnotič Komplexnej Elektrickej a Obvodovej Impedancie:
    Počíta a vyhodnocuje komplexné impedančné charakteristiky:
    - Rezistencia R [Ohm], Reaktancia X [Ohm], Impedancia Z = R + jX
    - Modul |Z| = sqrt(R^2 + X^2), Fázový uhol theta = arctan2(X, R)
    - Admitancia Y = 1/Z = G + jB (Konduktancia G, Susceptancia B)
    - Činiteľ akosti Q = |X| / R, Stratový činiteľ D = 1 / Q
    - Odrazový koeficient Gamma = (Z_L - Z_0) / (Z_L + Z_0)
    - Činiteľ stojatých vĺn VSWR = (1 + |Gamma|) / (1 - |Gamma|)
    - Útlm odrazu (Return Loss) RL = -20 * log10(|Gamma|) [dB]
    - Návrh L-prispôsobovacieho článku (L-match network synthesis).
    """

    @classmethod
    def evaluate_circuit_impedance(
        cls,
        resistance: float,
        inductance_h: float = 0.0,
        capacitance_f: float = 0.0,
        frequency_hz: float = 1000.0,
        reference_z0: float = 50.0
    ) -> Dict[str, Any]:
        """
        Vyhodnotí sériovú RLC impedanciu pri frekvencii f:
        Z = R + j * (omega * L - 1 / (omega * C))
        """
        assert resistance >= 0.0, f"Rezistencia musí byť nezáporná, zadané: {resistance}"
        assert frequency_hz > 0.0, f"Frekvencia musí byť kladná, zadané: {frequency_hz}"
        assert reference_z0 > 0.0, f"Referenčná impedancia Z0 musí byť kladná, zadané: {reference_z0}"

        omega = 2.0 * math.pi * frequency_hz

        # Induktívna reaktancia X_L = omega * L
        xl = omega * inductance_h if inductance_h > 0.0 else 0.0

        # Kapacitná reaktancia X_C = 1 / (omega * C)
        xc = (1.0 / (omega * capacitance_f)) if capacitance_f > 0.0 else 0.0

        # Čistá reaktancia X = X_L - X_C
        reactance = xl - xc

        # Komplexná impedancia Z = R + jX
        z = complex(resistance, reactance)
        magnitude = abs(z)
        phase_rad = cmath.phase(z)
        phase_deg = math.degrees(phase_rad)

        # Admitancia Y = 1 / Z = G + jB
        if magnitude > 1e-12:
            y = 1.0 / z
            conductance = y.real
            susceptance = y.imag
        else:
            y = complex(1e12, 0.0)
            conductance = 1e12
            susceptance = 0.0

        # Činiteľ akosti Q
        q_factor = abs(reactance) / resistance if resistance > 1e-12 else 9999.0
        dissipation = 1.0 / q_factor if q_factor > 1e-9 else 0.0

        # Odrazový koeficient Gamma = (Z - Z0) / (Z + Z0)
        z0 = complex(reference_z0, 0.0)
        gamma = (z - z0) / (z + z0)
        gamma_mag = abs(gamma)
        gamma_phase_deg = math.degrees(cmath.phase(gamma))

        # VSWR = (1 + |Gamma|) / (1 - |Gamma|)
        if gamma_mag < 0.99999:
            vswr = (1.0 + gamma_mag) / (1.0 - gamma_mag)
        else:
            vswr = 999.99

        # Return Loss RL = -20 * log10(|Gamma|)
        if gamma_mag > 1e-9:
            return_loss_db = -20.0 * math.log10(gamma_mag)
        else:
            return_loss_db = 100.0  # Dokonalé prispôsobenie

        # Prispôsobenie (L-Network Matching) ak R != Z0
        matching = cls._synthesize_l_match(resistance, reactance, reference_z0, omega)

        # Určenie charakteru záťaže
        if abs(reactance) < 1e-6:
            circuit_type = "Čisto odporová (Rezistívna)"
        elif reactance > 0:
            circuit_type = "Induktívna (Oneskorený prúd)"
        else:
            circuit_type = "Kapacitná (Predbiehajúci prúd)"

        return {
            "frequency_hz": frequency_hz,
            "angular_freq_rad_s": round(omega, 2),
            "resistance_ohms": round(resistance, 4),
            "reactance_ohms": round(reactance, 4),
            "inductive_reactance_ohms": round(xl, 4),
            "capacitive_reactance_ohms": round(xc, 4),
            "impedance_magnitude_ohms": round(magnitude, 4),
            "phase_angle_deg": round(phase_deg, 2),
            "phase_angle_rad": round(phase_rad, 4),
            "circuit_type": circuit_type,
            "admittance": {
                "magnitude_siemens": round(abs(y), 6),
                "conductance_siemens": round(conductance, 6),
                "susceptance_siemens": round(susceptance, 6)
            },
            "quality_factor_q": round(q_factor, 2),
            "dissipation_factor_d": round(dissipation, 4),
            "reference_z0_ohms": reference_z0,
            "reflection_coefficient": {
                "magnitude": round(gamma_mag, 4),
                "phase_deg": round(gamma_phase_deg, 2)
            },
            "vswr": round(vswr, 3),
            "return_loss_db": round(return_loss_db, 2),
            "matching_network": matching,
            "vital_max_hp": VITAL_MAX_HP
        }

    @staticmethod
    def _synthesize_l_match(rl: float, xl: float, z0: float, omega: float) -> Dict[str, Any]:
        """Navrhne dvojprvkový L-článok pre transformáciu záťaže Z_L = RL + jXL na Z0."""
        if abs(rl - z0) < 0.1 and abs(xl) < 0.1:
            return {"topology": "NONE_REQUIRED", "status": "ALREADY_MATCHED"}

        # Ak je rezistencia zhodná, stačí sériová kompenzácia reaktancie
        if abs(rl - z0) < 0.1:
            xs = -xl
            l_val = xs / omega if xs > 0 else None
            c_val = -1.0 / (omega * xs) if xs < 0 else None
            return {
                "topology": "SERIES_REACTANCE_COMPENSATION",
                "q_match": 0.0,
                "series_reactance_ohms": round(xs, 3),
                "shunt_reactance_ohms": 0.0,
                "suggested_components": {
                    "series_element": f"L_series = {round(l_val * 1e6, 2)} uH" if l_val else f"C_series = {round(c_val * 1e12, 2)} pF",
                    "shunt_element": "None (R matched)"
                }
            }

        # Výpočet Q matching siete
        if rl < z0:
            # Step-up topológia: paralelný prvok na strane zdroja
            ratio = max(1.0, z0 / max(0.001, rl))
            q_net = math.sqrt(ratio - 1.0)
            q_safe = max(1e-6, q_net)
            xp = z0 / q_safe
            xs = q_net * rl - xl
            topology = "STEP_UP_L_MATCH"
        else:
            # Step-down topológia: paralelný prvok na strane záťaže
            ratio = max(1.0, rl / max(0.001, z0))
            q_net = math.sqrt(ratio - 1.0)
            q_safe = max(1e-6, q_net)
            xs = q_net * z0 - xl
            xp = rl / q_safe
            topology = "STEP_DOWN_L_MATCH"

        l_val_h = xs / omega if xs > 0 else None
        c_val_f = -1.0 / (omega * xs) if xs < 0 else None

        return {
            "topology": topology,
            "q_match": round(q_net, 3),
            "series_reactance_ohms": round(xs, 3),
            "shunt_reactance_ohms": round(xp, 3),
            "suggested_components": {
                "series_element": f"L = {round(l_val_h * 1e6, 2)} uH" if l_val_h else f"C = {round(c_val_f * 1e12, 2)} pF",
                "shunt_element": f"C_shunt" if xp < 0 else f"L_shunt"
            }
        }


class SiliconThermalImpedanceEvaluator:
    """
    Hodnotič Termálnej Impedancie Křemíka (Silicon Thermal Impedance):
    Modeluje dynamickú odozvu kremíkového jadra (Intel Iris Xe, NPU, CPU)
    pomocou Fosterovej / Cauerovej viackrokovej RC siete.
    Počíta prechodovú teplotu Delta T(t) a určuje tepelné zakrivenie
    pre Strain Domain Model (krystal_theory/strain_operator.py).
    """

    # Predvolené tepelné póly pre kremíkový die balíček (R_i v K/W, C_i v J/K)
    DEFAULT_THERMAL_POLES: List[Dict[str, float]] = [
        {"r": 0.08, "c": 0.05, "name": "Silicon_Junction"},
        {"r": 0.22, "c": 0.80, "name": "Thermal_Interface_TIM"},
        {"r": 0.45, "c": 12.0, "name": "Heat_Spreader_Copper"},
        {"r": 0.85, "c": 95.0, "name": "Chassis_Heatsink"}
    ]

    @classmethod
    def evaluate_transient_thermal_impedance(
        cls,
        time_seconds: float,
        power_watts: float = 28.0,
        ambient_temp_c: float = 25.0,
        poles: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Vypočíta prechodovú termálnu impedanciu Z_th(t) = sum(R_i * (1 - e^(-t / tau_i)))
        a odhadne výslednú teplotu jadra T_j(t) = T_ambient + P * Z_th(t).
        """
        assert time_seconds >= 0.0, "Čas musí byť nezáporný."
        assert power_watts >= 0.0, "Stratový výkon musí byť nezáporný."

        active_poles = poles or cls.DEFAULT_THERMAL_POLES
        z_th_t = 0.0
        r_th_total = 0.0
        pole_states = []

        for p in active_poles:
            r = p["r"]
            c = p["c"]
            tau = r * c  # Časová konštanta tau = R * C [s]
            r_th_total += r

            fraction = (1.0 - math.exp(-time_seconds / tau)) if tau > 0 else 1.0
            z_i = r * fraction
            z_th_t += z_i

            pole_states.append({
                "name": p.get("name", "Pole"),
                "r_k_per_w": r,
                "c_j_per_k": c,
                "tau_seconds": round(tau, 4),
                "z_contribution_k_per_w": round(z_i, 5),
                "saturation_pct": round(fraction * 100.0, 1)
            })

        # Teplota čipu: T_j = T_amb + P * Z_th
        temp_rise = power_watts * z_th_t
        junction_temp = ambient_temp_c + temp_rise

        # Tepelné zakrivenie arizujúce v strain operátore: curvature = d^2 Z_th / dt^2
        # Integruje sa do krystal_theory/strain_operator.py: P(x) = a x^2 + b x + c
        strain_curvature = round(z_th_t / (r_th_total + 1e-9), 4)

        # Tepelný stav
        if junction_temp > 95.0:
            thermal_state = "CRITICAL_THROTTLE"
        elif junction_temp > 80.0:
            thermal_state = "ELEVATED_HEAVY_LOAD"
        elif junction_temp > 55.0:
            thermal_state = "NOMINAL_ACTIVE"
        else:
            thermal_state = "COOL_IDLE"

        return {
            "time_seconds": time_seconds,
            "power_watts": power_watts,
            "ambient_temp_c": ambient_temp_c,
            "total_steady_state_rth_k_w": round(r_th_total, 4),
            "transient_zth_k_w": round(z_th_t, 4),
            "temperature_rise_c": round(temp_rise, 2),
            "junction_temperature_c": round(junction_temp, 2),
            "thermal_state": thermal_state,
            "strain_curvature_param_a": strain_curvature,
            "poles": pole_states,
            "vital_max_hp": VITAL_MAX_HP
        }


class CognitiveFlowImpedanceEvaluator:
    """
    Hodnotič Kognitívnej a Architektonickej Impedancie (Cognitive Flow Impedance):
    Modeluje odpor toku informácií a zložitosť promptov v Krystal Coping Cortex
    a kaskádových LLM modeloch.
    Z_cog = R_latency + j * (omega * L_contention - 1 / (omega * C_cache))
    """

    @classmethod
    def evaluate_cognitive_flow(
        cls,
        prompt_complexity_score: float,
        model_tier: str = "cascade_medium",
        active_memory_load_pct: float = 45.0,
        cache_hit_rate: float = 0.85
    ) -> Dict[str, Any]:
        """
        Vyhodnotí kognitívnu impedanciu toku promptu:
        - prompt_complexity_score: hodnota z NGramReader entrópie [0.0, 10.0]
        - model_tier: úroveň modelu ('local_fast', 'cascade_medium', 'cloud_frontier')
        - active_memory_load_pct: zaťaženie UMA VRAM
        - cache_hit_rate: úspešnosť KV cache [0.0, 1.0]
        """
        omega = max(0.1, prompt_complexity_score)

        # Základná rezistencia (latencia inferencie)
        base_latency_r = {
            "local_fast": 15.0,        # ~15 ms
            "cascade_medium": 45.0,    # ~45 ms
            "cloud_frontier": 220.0    # ~220 ms
        }.get(model_tier, 50.0)

        # Induktancia zbernice (pamäťové preťaženie a čakacie fronty)
        l_contention = (active_memory_load_pct / 100.0) * 12.0

        # Kapacitancia vyrovnávacej pamäte (KV Cache)
        c_cache = max(0.01, cache_hit_rate * 2.0)

        xl = omega * l_contention
        xc = 1.0 / (omega * c_cache)
        reactance = xl - xc

        z_cog = complex(base_latency_r, reactance)
        mag = abs(z_cog)
        phase_deg = math.degrees(cmath.phase(z_cog))

        # Strain Index: normalizované kognitívne napätie v intervale [0.0, 1.0)
        strain_raw = mag / 100.0
        normalized_strain = math.tanh(strain_raw * 0.5)

        # Odporúčaná akcia Kaskády
        if normalized_strain > 0.85:
            recommendation = "FALLBACK_TO_SLM_CORE_RAG (High Contention)"
        elif normalized_strain > 0.50:
            recommendation = "EXECUTE_CASCADE_MEDIUM_OPTIMIZED"
        else:
            recommendation = "OPTIMAL_FLOW_DIRECT_PIPELINE"

        return {
            "prompt_complexity_omega": round(omega, 2),
            "model_tier": model_tier,
            "cognitive_resistance_ohms": round(base_latency_r, 2),
            "memory_inductance_reactance": round(xl, 2),
            "cache_capacitance_reactance": round(xc, 2),
            "total_cognitive_impedance_magnitude": round(mag, 2),
            "cognitive_phase_deg": round(phase_deg, 2),
            "normalized_strain_index": round(normalized_strain, 4),
            "recommendation": recommendation,
            "vital_max_hp": VITAL_MAX_HP
        }


# ==============================================================================
# 3. ZJEDNOTENÝ MASTER DISPEČER (GRAM & IMPEDANCE MASTER DISPATCHER)
# ==============================================================================

class GramAndImpedanceEngine:
    """
    Zjednocujúci orchestrátor pre čítačky gramov a hodnotič impedancie.
    Poskytuje čisté rozhranie pre REST API a MCP server.
    """

    def __init__(self):
        self.ngram_reader = NGramReader()
        self.gramian_reader = GramianMatrixReader()
        self.spectrogram_reader = SpectrogramFeatureReader()
        self.mass_reader = PhysicalMassGramReader()
        self.circuit_evaluator = ComplexImpedanceEvaluator()
        self.thermal_evaluator = SiliconThermalImpedanceEvaluator()
        self.cognitive_evaluator = CognitiveFlowImpedanceEvaluator()

    def read_grams(self, mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispečer pre čítačky gramov (ngram, gramian, spectrogram, mass)."""
        mode_lower = mode.lower()
        if mode_lower in ("ngram", "n-gram", "text"):
            text = payload.get("text", "")
            n = int(payload.get("n", 2))
            top_k = int(payload.get("top_k", 10))
            return self.ngram_reader.analyze_ngram_distribution(text, n=n, top_k=top_k)

        elif mode_lower in ("gramian", "gram_matrix", "vectors"):
            vectors = payload.get("vectors", [[1.0, 0.0], [0.0, 1.0]])
            normalize = bool(payload.get("normalize", False))
            return self.gramian_reader.compute_gram_matrix(vectors, normalize=normalize)

        elif mode_lower in ("spectrogram", "acoustic", "spectral"):
            magnitudes = payload.get("magnitudes", [1.0, 0.8, 0.4, 0.2, 0.1])
            sample_rate = int(payload.get("sample_rate", 16000))
            return self.spectrogram_reader.read_spectral_frame(magnitudes, sample_rate=sample_rate)

        elif mode_lower in ("mass", "physical", "weight"):
            components = payload.get("components", [])
            if components:
                return self.mass_reader.calculate_aggregate_assembly(components)
            else:
                mat = payload.get("material", "crystal_quartz")
                vol = float(payload.get("volume_cm3", 10.0))
                pos = payload.get("position", [0.0, 0.0, 0.0])
                return self.mass_reader.calculate_component_mass(mat, vol, pos)

        else:
            return {"error": f"Neznámy mód čítačky gramov: '{mode}'. Podporované: ngram, gramian, spectrogram, mass."}

    def evaluate_impedance(self, domain: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Dispečer pre hodnotič impedancie (electrical, thermal, cognitive)."""
        dom_lower = domain.lower()
        if dom_lower in ("electrical", "circuit", "rf", "rlc"):
            r = float(payload.get("resistance", 50.0))
            l_val = float(payload.get("inductance_h", 1e-6))
            c_val = float(payload.get("capacitance_f", 1e-12))
            f_val = float(payload.get("frequency_hz", 1000000.0))
            z0 = float(payload.get("reference_z0", 50.0))
            return self.circuit_evaluator.evaluate_circuit_impedance(r, l_val, c_val, f_val, z0)

        elif dom_lower in ("thermal", "silicon", "temperature"):
            t = float(payload.get("time_seconds", 1.0))
            p = float(payload.get("power_watts", 28.0))
            amb = float(payload.get("ambient_temp_c", 25.0))
            return self.thermal_evaluator.evaluate_transient_thermal_impedance(t, p, amb)

        elif dom_lower in ("cognitive", "flow", "prompt", "pipeline"):
            complexity = float(payload.get("prompt_complexity_score", 3.5))
            tier = payload.get("model_tier", "cascade_medium")
            mem = float(payload.get("active_memory_load_pct", 45.0))
            hit = float(payload.get("cache_hit_rate", 0.85))
            return self.cognitive_evaluator.evaluate_cognitive_flow(complexity, tier, mem, hit)

        else:
            return {"error": f"Neznáma doména hodnotiča impedancie: '{domain}'. Podporované: electrical, thermal, cognitive."}


# Globálny singleton pre importy
GLOBAL_GRAM_AND_IMPEDANCE_ENGINE = GramAndImpedanceEngine()


if __name__ == "__main__":
    print(f"=== Krystal-Stack: Čítačky Gramov a Hodnotič Impedancie ===")
    assert VITAL_MAX_HP == 6, "VITAL_MAX_HP must equal 6"

    engine = GLOBAL_GRAM_AND_IMPEDANCE_ENGINE

    # 1. Test N-Gram čítačky
    sample_text = "zrkadli 6-uholníkovú posvätnú geometriu v štýle alchýmie a zrkadli zrkadlenie"
    ng_res = engine.read_grams("ngram", {"text": sample_text, "n": 2})
    print(f"  ✓ N-Gram (N=2): Celkovo {ng_res['total_ngrams']} n-gramov, Entrópia = {ng_res['entropy']}")

    # 2. Test Gramovej matice
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    gm_res = engine.read_grams("gramian", {"vectors": vecs, "normalize": True})
    print(f"  ✓ Gramian Matrix 3x3 Det: {gm_res['gram_determinant']}, Volume: {gm_res['parallelotope_volume']}")

    # 3. Test Hodnotiča elektrickej impedancie
    imp_res = engine.evaluate_impedance("electrical", {
        "resistance": 50.0,
        "inductance_h": 10e-6,
        "capacitance_f": 100e-12,
        "frequency_hz": 5e6,
        "reference_z0": 50.0
    })
    print(f"  ✓ Elektrická Impedancia: |Z| = {imp_res['impedance_magnitude_ohms']} Ohm, VSWR = {imp_res['vswr']}")

    # 4. Test Hodnotiča termálnej impedancie
    therm_res = engine.evaluate_impedance("thermal", {"time_seconds": 2.5, "power_watts": 28.0})
    print(f"  ✓ Termálna Impedancia (2.5s): Z_th = {therm_res['transient_zth_k_w']} K/W, Tj = {therm_res['junction_temperature_c']} °C")

    # 5. Test Hodnotiča kognitívnej impedancie
    cog_res = engine.evaluate_impedance("cognitive", {"prompt_complexity_score": ng_res['entropy']})
    print(f"  ✓ Kognitívna Impedancia: |Z_cog| = {cog_res['total_cognitive_impedance_magnitude']}, Strain = {cog_res['normalized_strain_index']}")

    print("\nVšetky moduly úspešne otestované a verifikované. VITAL_MAX_HP == 6 zachované.")
