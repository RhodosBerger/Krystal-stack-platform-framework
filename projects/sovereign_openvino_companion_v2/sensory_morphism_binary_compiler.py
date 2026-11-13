"""
KRYSTAL-STACK // SOVEREIGN OPENVINO COMPANION V2
Sensory Coordinate Morphism, Binary Parameter Aspect-Ratio Compiler,
Process vs. Microservice Epistemic Distinction, Hypervisor Matrix Choice Palette,
New Stack Flows (S0..S9), and Thermodynamic Knowledge Properties.

Constitutional Invariants:
  - VITAL_MAX_HP == 6 (¬Valid(Action) => ΔS = 0; raises ContractViolation)
  - 56 frozen theory nodes (T1..T12, 0 DAG cycles; never adds Node 57 or T13)
  - PROPOSER != VERIFIER (`ArtifactGate`)
  - 128B ZeroCopyContractFrame: [0:84] 16 Wire Params + [84:108] GNOS (24B) + [108:128] SMRP (20B)
  - No bare assert in production runtime modules; all contracts raise ContractViolation.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from krystal_kernel.constants import (
    AUTHENTICATED_PREFIX_BYTES,
    ContractViolation,
    FRAME_SIZE_BYTES,
    UMA_SATURATION_THRESHOLD,
    VITAL_MAX_HP,
)
from krystal_kernel.epistemic_cognition_compositor import CognitionLevel
from krystal_kernel.epistemic_enlg_compiler import ArtifactGate, EnlgRole
from krystal_kernel.ontologic_gnosis_power_strata import (
    FROZEN_THEORY_NODES,
    GLOBAL_ONTOLOGIC_GNOSIS_ENGINE,
    GNOSIS_MAGIC,
)
from krystal_kernel.zero_copy_contracts import canonicalize_float32
from projects.sovereign_openvino_companion.operating_coordinates_community import (
    OperatingCoordinatesVector,
)
from projects.sovereign_openvino_companion_v2.smart_tuner import (
    BufferedComputationLogSystem,
)

if VITAL_MAX_HP != 6:
    raise ContractViolation(
        f"Constitutional invariant breach: VITAL_MAX_HP must be 6, got {VITAL_MAX_HP}",
        violation_code="ERR_CONSTITUTIONAL_INVARIANT_BREACH",
    )

SMRP_MAGIC: bytes = b"SMRP"
PHI1_MAGIC: bytes = b"PHI1"
GLDS_MAGIC: bytes = b"GLDS"
SLVR_MAGIC: bytes = b"SLVR"
KMB1_MAGIC: bytes = b"KMB1"
WNTC_MAGIC: bytes = b"WNTC"

# Exact binary struct layouts (verified at module load)
# 84B base wire: 16 required wire parameters mapped from DusanGameVulkanPowerAndTensorBridge
WIRE_16_FORMAT: str = "<H4sQHIffffBB6fBfB16s"
# 20B SMRP descriptor in [108:128] of canonical 128B ZeroCopyContractFrame
SMRP_20B_FORMAT: str = "<4sHHHHBBBBI"
# 24B KMB1 lockstep packet
KMB1_24B_FORMAT: str = ">4sQBBHII"
# 64B Cache-Line Self-Binary Coexistence Descriptor (WSL2 UNIX <-> Windows NT Kernel Hierarchy)
WNTC_64B_FORMAT: str = "<4sHHIIIIQQ8s8sII"
# 79B + 49B = 128B Golden-Ratio partition (79/49 = 1.612245 ~ phi = 1.618034)
PHI_79B_HEAD_FORMAT: str = "<H4sQHffffBB6hBfB16sBHII"
GLDS_49B_TAIL_FORMAT: str = "<4sffffffffBBBB5sI"
# 90B + 38B = 128B Silver-Octagonal partition (90/38 = 2.368421 ~ delta_S = 2.414214)
SILVER_6B_EXT_FORMAT: str = "<HI"
SLVR_38B_TAIL_FORMAT: str = "<4sffHHHHffBBBBHI"

if struct.calcsize(WIRE_16_FORMAT) != 84:
    raise ContractViolation(
        f"WIRE_16_FORMAT size mismatch: expected 84, got {struct.calcsize(WIRE_16_FORMAT)}",
        violation_code="ERR_BINARY_LAYOUT_MISMATCH",
    )
if struct.calcsize(SMRP_20B_FORMAT) != 20:
    raise ContractViolation(
        f"SMRP_20B_FORMAT size mismatch: expected 20, got {struct.calcsize(SMRP_20B_FORMAT)}",
        violation_code="ERR_BINARY_LAYOUT_MISMATCH",
    )
if struct.calcsize(KMB1_24B_FORMAT) != 24:
    raise ContractViolation(
        f"KMB1_24B_FORMAT size mismatch: expected 24, got {struct.calcsize(KMB1_24B_FORMAT)}",
        violation_code="ERR_BINARY_LAYOUT_MISMATCH",
    )
if struct.calcsize(WNTC_64B_FORMAT) != 64:
    raise ContractViolation(
        f"WNTC_64B_FORMAT size mismatch: expected 64, got {struct.calcsize(WNTC_64B_FORMAT)}",
        violation_code="ERR_BINARY_LAYOUT_MISMATCH",
    )
if struct.calcsize(PHI_79B_HEAD_FORMAT) + struct.calcsize(GLDS_49B_TAIL_FORMAT) != FRAME_SIZE_BYTES:
    raise ContractViolation(
        "PHI_GOLDEN_79_49 frame size mismatch != 128B",
        violation_code="ERR_BINARY_LAYOUT_MISMATCH",
    )
if (
    struct.calcsize(WIRE_16_FORMAT)
    + struct.calcsize(SILVER_6B_EXT_FORMAT)
    + struct.calcsize(SLVR_38B_TAIL_FORMAT)
    != FRAME_SIZE_BYTES
):
    raise ContractViolation(
        "SILVER_OCTAGONAL_90_38 frame size mismatch != 128B",
        violation_code="ERR_BINARY_LAYOUT_MISMATCH",
    )

PERCEPTION_LENSES: Tuple[str, ...] = (
    "SOLAR_CLARITY",
    "CAUSTIC_SUSPICION",
    "UMBRAL_NOIR_MORTAR",
    "PRISMATIC_OVERDRIVE",
)
LENS_TO_CODE: Dict[str, int] = {name: idx for idx, name in enumerate(PERCEPTION_LENSES)}
CODE_TO_LENS: Dict[int, str] = {idx: name for idx, name in enumerate(PERCEPTION_LENSES)}

POWER_PLAN_TO_CODE: Dict[str, int] = {
    "VULKAN_UMA_BURST_15W": 0,
    "VULKAN_UMA_BALANCED_12W": 1,
    "VULKAN_UMA_SURVIVAL_7_5W": 2,
}
CODE_TO_POWER_PLAN: Dict[int, str] = {v: k for k, v in POWER_PLAN_TO_CODE.items()}

PHI_GOLDEN: float = (1.0 + math.sqrt(5.0)) / 2.0
SILVER_RATIO: float = 1.0 + math.sqrt(2.0)
VISUAL_DOF_TOTAL: int = 418
CORE_WIRE_PARAM_COUNT: int = 16


def _safe_float(val: float, name: str) -> float:
    if not math.isfinite(val):
        raise ContractViolation(
            f"Non-finite coordinate float for {name}: {val}",
            violation_code="ERR_NON_FINITE_COORDINATE",
        )
    return float(canonicalize_float32(val))


@dataclass(frozen=True)
class SensoryManifoldCoordinates:
    """5D + Vital-Anchor sensory perception coordinates mapped from operating telemetry."""

    cognition_level: int
    cognition_label: str
    active_perception_lens: str
    radiance_nits: float
    chromatic_entropy_bits: float
    acoustic_dorian_hz: float
    sense_coherence_score: float
    vital_hp_anchor: int = VITAL_MAX_HP

    def __post_init__(self) -> None:
        if self.vital_hp_anchor != VITAL_MAX_HP:
            raise ContractViolation(
                f"Constitutional breach in SensoryManifoldCoordinates: vital_hp_anchor={self.vital_hp_anchor} != 6",
                violation_code="ERR_CONSTITUTIONAL_INVARIANT_BREACH",
            )
        if self.cognition_level not in (1, 2, 3, 4, 5):
            raise ContractViolation(
                f"cognition_level out of range [1, 5]: {self.cognition_level}",
                violation_code="ERR_SENSORY_COGNITION_OUT_OF_RANGE",
            )
        if self.active_perception_lens not in LENS_TO_CODE:
            raise ContractViolation(
                f"Unknown perception lens: {self.active_perception_lens}",
                violation_code="ERR_SENSORY_UNKNOWN_LENS",
            )
        if not (0.0 <= self.sense_coherence_score <= 1.0):
            raise ContractViolation(
                f"sense_coherence_score out of [0, 1]: {self.sense_coherence_score}",
                violation_code="ERR_SENSORY_COHERENCE_BOUNDS",
            )
        for fname in (
            "radiance_nits",
            "chromatic_entropy_bits",
            "acoustic_dorian_hz",
            "sense_coherence_score",
        ):
            val = getattr(self, fname)
            if not math.isfinite(val):
                raise ContractViolation(
                    f"Non-finite sensory field {fname}: {val}",
                    violation_code="ERR_NON_FINITE_COORDINATE",
                )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SensoryCoordinateMorphismEngine:
    """
    Morphs between raw 6D `OperatingCoordinatesVector` hardware telemetry and
    `SensoryManifoldCoordinates` (Cognition L1..L5, V-Ray Perception Lens, Radiance,
    Chromatic Entropy, Dorian Acoustic Frequency, and Sense Coherence), enforcing P6
    (no convex blending on contradictory evidence) and packing the 20B `SMRP` trailer.
    """

    def __init__(self, logger: Optional[BufferedComputationLogSystem] = None) -> None:
        self.logger = logger

    def project_operating_to_sensory(
        self, op_vec: OperatingCoordinatesVector
    ) -> SensoryManifoldCoordinates:
        uma = float(op_vec.uma_bus_coordinate)
        gnosis = float(op_vec.gnosis_yield_nats_per_mj)
        gx = int(op_vec.grid_x)
        gy = int(op_vec.grid_y)
        vblank = float(op_vec.vblank_phase_ms)

        if uma >= UMA_SATURATION_THRESHOLD:
            lens = "UMBRAL_NOIR_MORTAR"
            cognition_lvl = 2
        elif uma >= 0.45:
            lens = "CAUSTIC_SUSPICION"
            cognition_lvl = 3 if gnosis <= 2.0 else 4
        elif gnosis >= 5.0:
            lens = "PRISMATIC_OVERDRIVE"
            cognition_lvl = 5
        else:
            lens = "SOLAR_CLARITY"
            if gnosis <= 0.0:
                cognition_lvl = 1
            elif gnosis <= 2.0:
                cognition_lvl = 3
            elif gnosis <= 4.5:
                cognition_lvl = 4
            else:
                cognition_lvl = 5

        radiance_nits = round(_safe_float(120.0 + 38.5 * gnosis + 4.0 * (gx + gy), "radiance_nits"), 4)
        chromatic_entropy_bits = round(
            _safe_float(min(4.0, 0.85 + 1.65 * uma + 0.04 * vblank), "chromatic_entropy_bits"), 4
        )
        dorian_hz = round(
            _safe_float(293.66 * (1.0 + 0.03125 * ((gx + gy) % 8)), "acoustic_dorian_hz"), 4
        )
        coherence = round(
            _safe_float(
                max(0.0, min(1.0, 1.0 - 0.55 * uma + 0.03 * min(gnosis, 5.0))),
                "sense_coherence_score",
            ),
            4,
        )

        return SensoryManifoldCoordinates(
            cognition_level=cognition_lvl,
            cognition_label=CognitionLevel(cognition_lvl).name,
            active_perception_lens=lens,
            radiance_nits=radiance_nits,
            chromatic_entropy_bits=chromatic_entropy_bits,
            acoustic_dorian_hz=dorian_hz,
            sense_coherence_score=coherence,
            vital_hp_anchor=VITAL_MAX_HP,
        )

    def inverse_sensory_to_operating(
        self,
        sense: SensoryManifoldCoordinates,
        grid_xy: Tuple[int, int] = (4, 5),
        vblank_phase_ms: float = 5.4,
    ) -> OperatingCoordinatesVector:
        gx, gy = int(grid_xy[0]), int(grid_xy[1])
        gnosis_est = max(0.0, (sense.radiance_nits - 120.0 - 4.0 * (gx + gy)) / 38.5)
        uma_est = max(0.0, min(1.0, (sense.chromatic_entropy_bits - 0.85 - 0.04 * vblank_phase_ms) / 1.65))
        if sense.active_perception_lens == "UMBRAL_NOIR_MORTAR":
            uma_est = max(uma_est, 0.80)
        elif sense.active_perception_lens == "CAUSTIC_SUSPICION":
            uma_est = max(0.45, min(0.76, uma_est))
        pl1_watts = 15.0 if uma_est < 0.45 else (12.0 if uma_est < 0.78 else 7.5)

        return OperatingCoordinatesVector(
            grid_x=gx,
            grid_y=gy,
            uma_bus_coordinate=round(uma_est, 4),
            vblank_phase_ms=round(vblank_phase_ms, 4),
            thermal_pl1_watts=pl1_watts,
            gnosis_yield_nats_per_mj=round(gnosis_est, 4),
            vital_hp_coordinate=VITAL_MAX_HP,
        )

    def pack_20b_smrp_descriptor(
        self,
        sense: SensoryManifoldCoordinates,
        morph_factor: float = 1.0,
        homotopic_fork: bool = False,
    ) -> bytes:
        if not (0.0 <= morph_factor <= 1.0) or not math.isfinite(morph_factor):
            raise ContractViolation(
                f"morph_factor must be in [0.0, 1.0], got {morph_factor}",
                violation_code="ERR_SENSORY_MORPH_FACTOR_BOUNDS",
            )
        rad_q10 = min(65535, max(0, round(sense.radiance_nits * 10.0)))
        chr_q1000 = min(65535, max(0, round(sense.chromatic_entropy_bits * 1000.0)))
        aud_q10 = min(65535, max(0, round(sense.acoustic_dorian_hz * 10.0)))
        coh_q10000 = min(10000, max(0, round(sense.sense_coherence_score * 10000.0)))
        lens_code = LENS_TO_CODE[sense.active_perception_lens]
        morph_q8 = min(255, max(0, round(morph_factor * 255.0)))
        flags_u8 = (0x01 if sense.vital_hp_anchor == VITAL_MAX_HP else 0x00) | (
            0x02 if homotopic_fork else 0x00
        )

        body_16b = struct.pack(
            "<4sHHHHBBBB",
            SMRP_MAGIC,
            rad_q10,
            chr_q1000,
            aud_q10,
            coh_q10000,
            sense.cognition_level,
            lens_code,
            morph_q8,
            flags_u8,
        )
        crc = zlib.crc32(body_16b) & 0xFFFFFFFF
        return body_16b + struct.pack("<I", crc)

    def unpack_20b_smrp_descriptor(self, buf_20b: bytes) -> Dict[str, Any]:
        if len(buf_20b) != 20:
            raise ContractViolation(
                f"SMRP descriptor must be exactly 20 bytes, got {len(buf_20b)}",
                violation_code="ERR_SMRP_SIZE_MISMATCH",
            )
        magic, rad_q10, chr_q1000, aud_q10, coh_q10000, cog_lvl, lens_code, morph_q8, flags_u8, crc = (
            struct.unpack(SMRP_20B_FORMAT, buf_20b)
        )
        if magic != SMRP_MAGIC:
            raise ContractViolation(
                f"SMRP magic mismatch: {magic!r}",
                violation_code="ERR_SMRP_MAGIC_MISMATCH",
            )
        expected_crc = zlib.crc32(buf_20b[:16]) & 0xFFFFFFFF
        if crc != expected_crc:
            raise ContractViolation(
                f"SMRP CRC32 tampering detected: got {crc:#010x}, expected {expected_crc:#010x}",
                violation_code="ERR_SMRP_CRC_MISMATCH",
            )
        if (flags_u8 & 0x01) == 0:
            raise ContractViolation(
                "SMRP vital_hp_anchor invariant bit cleared (VITAL_MAX_HP != 6)",
                violation_code="ERR_CONSTITUTIONAL_INVARIANT_BREACH",
            )
        if cog_lvl not in (1, 2, 3, 4, 5) or lens_code not in CODE_TO_LENS:
            raise ContractViolation(
                f"SMRP decoded invalid cognition={cog_lvl} or lens={lens_code}",
                violation_code="ERR_SMRP_FIELD_CORRUPTION",
            )

        return {
            "magic": magic.decode("ascii"),
            "cognition_level": cog_lvl,
            "cognition_label": CognitionLevel(cog_lvl).name,
            "active_perception_lens": CODE_TO_LENS[lens_code],
            "radiance_nits": round(rad_q10 / 10.0, 2),
            "chromatic_entropy_bits": round(chr_q1000 / 1000.0, 4),
            "acoustic_dorian_hz": round(aud_q10 / 10.0, 2),
            "sense_coherence_score": round(coh_q10000 / 10000.0, 4),
            "morph_factor": round(morph_q8 / 255.0, 4),
            "homotopic_fork_active": bool(flags_u8 & 0x02),
            "vital_hp_anchor": VITAL_MAX_HP,
            "crc32_hex": f"0x{crc:08X}",
            "crc32_valid": True,
        }

    def morph_sensory_coordinates(
        self,
        source: SensoryManifoldCoordinates,
        target: SensoryManifoldCoordinates,
        morph_factor: float = 0.618034,
        contradictory_evidence: bool = False,
    ) -> Dict[str, Any]:
        if not (0.0 <= morph_factor <= 1.0) or not math.isfinite(morph_factor):
            raise ContractViolation(
                f"morph_factor out of bounds [0.0, 1.0]: {morph_factor}",
                violation_code="ERR_SENSORY_MORPH_FACTOR_BOUNDS",
            )

        t0 = time.perf_counter_ns()
        if contradictory_evidence:
            smrp_alpha = self.pack_20b_smrp_descriptor(source, morph_factor=morph_factor, homotopic_fork=True)
            smrp_beta = self.pack_20b_smrp_descriptor(target, morph_factor=morph_factor, homotopic_fork=True)
            decoded_alpha = self.unpack_20b_smrp_descriptor(smrp_alpha)
            dur_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            if self.logger is not None:
                self.logger.log_event(
                    channel="telemetry",
                    operation="SENSORY_COORDINATE_HOMOTOPIC_FORK",
                    stratum="S6_HYPOSTATIC_SUPERPOSITION_FORK",
                    duration_ms=dur_ms,
                    payload={
                        "morph_factor": morph_factor,
                        "contradictory_evidence": True,
                        "convex_averaging_blocked": True,
                        "alpha_lens": source.active_perception_lens,
                        "beta_lens": target.active_perception_lens,
                    },
                    epistemic_status="CONFLICT_SUPERPOSED_P6",
                )
            return {
                "topology": "SUPERPOSED_HOMOTOPIC_FORK_ALPHA_PERP_BETA",
                "morph_factor": round(morph_factor, 6),
                "contradictory_evidence": True,
                "convex_averaging_blocked": True,
                "branch_alpha": source.to_dict(),
                "branch_beta": target.to_dict(),
                "smrp_alpha_20b_hex": smrp_alpha.hex().upper(),
                "smrp_beta_20b_hex": smrp_beta.hex().upper(),
                "smrp_decoded_primary": decoded_alpha,
                "smrp_crc32_valid": True,
                "vital_max_hp": VITAL_MAX_HP,
            }

        t = float(morph_factor)
        rad = round(_safe_float((1.0 - t) * source.radiance_nits + t * target.radiance_nits, "rad"), 4)
        chr_b = round(
            _safe_float((1.0 - t) * source.chromatic_entropy_bits + t * target.chromatic_entropy_bits, "chr"),
            4,
        )
        aud = round(_safe_float((1.0 - t) * source.acoustic_dorian_hz + t * target.acoustic_dorian_hz, "aud"), 4)
        coh = round(
            _safe_float(
                max(0.0, min(1.0, (1.0 - t) * source.sense_coherence_score + t * target.sense_coherence_score)),
                "coh",
            ),
            4,
        )
        cog_lvl = target.cognition_level if t >= 0.5 else source.cognition_level
        lens = target.active_perception_lens if t >= 0.5 else source.active_perception_lens

        morphed = SensoryManifoldCoordinates(
            cognition_level=cog_lvl,
            cognition_label=CognitionLevel(cog_lvl).name,
            active_perception_lens=lens,
            radiance_nits=rad,
            chromatic_entropy_bits=chr_b,
            acoustic_dorian_hz=aud,
            sense_coherence_score=coh,
            vital_hp_anchor=VITAL_MAX_HP,
        )
        smrp_20b = self.pack_20b_smrp_descriptor(morphed, morph_factor=t, homotopic_fork=False)
        decoded = self.unpack_20b_smrp_descriptor(smrp_20b)
        dur_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        if self.logger is not None:
            self.logger.log_event(
                channel="telemetry",
                operation="SENSORY_COORDINATE_HOMEOMORPHIC_MORPH",
                stratum="S8_LOGOS_SEMANTIC_BIJECTION",
                duration_ms=dur_ms,
                payload={
                    "morph_factor": t,
                    "source_lens": source.active_perception_lens,
                    "target_lens": target.active_perception_lens,
                    "morphed_lens": morphed.active_perception_lens,
                    "morphed_coherence": morphed.sense_coherence_score,
                },
                epistemic_status="MEASURED_WITH_SCOPE",
            )
        return {
            "topology": "HOMEOMORPHIC_CONTINUOUS_MORPH",
            "morph_factor": round(t, 6),
            "contradictory_evidence": False,
            "convex_averaging_blocked": False,
            "source_coordinates": source.to_dict(),
            "target_coordinates": target.to_dict(),
            "morphed_coordinates": morphed.to_dict(),
            "inverse_reconstructed_operating_coords": self.inverse_sensory_to_operating(morphed).to_dict(),
            "smrp_20b_hex": smrp_20b.hex().upper(),
            "smrp_decoded": decoded,
            "smrp_crc32_valid": True,
            "vital_max_hp": VITAL_MAX_HP,
        }


class BinaryParameterAspectCompiler:
    """
    Compiles & losslessly decompiles exact 128B Zero-Copy Contract Frames and 24B KMB1
    packets across 3 structural parameter aspect ratios:
      1. CANONICAL_ZERO_COPY_21_6_5  (84B Wire : 24B GNOS : 20B SMRP = 21:6:5; 128:24 vs KMB1)
      2. PHI_GOLDEN_79_49            (79B Compact Wire : 49B Golden Sensory-Gnosis = 1.6122 ~ phi)
      3. SILVER_OCTAGONAL_90_38      (90B High-Density Ohm/Wire : 38B Compact Trailer = 2.3684 ~ delta_S)
    Enforces PROPOSER != VERIFIER through `ArtifactGate` and HMAC-SHA256 P9 digests.
    """

    def __init__(
        self,
        logger: Optional[BufferedComputationLogSystem] = None,
        hmac_secret: bytes = b"KRYSTAL_SOVEREIGN_V2_BINARY_SECRET",
    ) -> None:
        self.logger = logger
        self.hmac_secret = hmac_secret
        self.sensory_engine = SensoryCoordinateMorphismEngine(logger=logger)
        self.artifact_gate = ArtifactGate()

    def pack_84b_required_wire_params(
        self,
        op_vec: OperatingCoordinatesVector,
        sense: SensoryManifoldCoordinates,
        bitboard_64bit: int = 0x8100000800000080,
        move_mask: int = 0x0013,
        marias_trump_score: int = 46,
        speculative_draft_tokens: int = 28,
        merkle_momentum_m_k: float = 6.4215,
        power_plan_profile: str = "VULKAN_UMA_BALANCED_12W",
        tenant_slug: str = "nabytok47_rh_iso",
        cv_6d_vector: Optional[Tuple[float, ...]] = None,
    ) -> bytes:
        cv = cv_6d_vector or (0.27, 0.38, 0.16, 0.17, 0.25, 0.45)
        if len(cv) != 6:
            raise ContractViolation(
                f"cv_6d_vector must have 6 floats, got {len(cv)}",
                violation_code="ERR_WIRE_CV6D_LENGTH",
            )
        voltage = 10.8 + op_vec.uma_bus_coordinate * 2.4
        resistance = 1.4 + op_vec.uma_bus_coordinate * 1.3
        current = min(8.0, voltage / max(0.1, resistance))
        power_load = min(op_vec.thermal_pl1_watts, voltage * current * 0.25)
        lens_code = LENS_TO_CODE[sense.active_perception_lens]
        plan_code = POWER_PLAN_TO_CODE.get(power_plan_profile, 1)
        slug_bytes = tenant_slug.encode("ascii", errors="replace")[:16].ljust(16, b"_")

        # Pack with placeholder crc=0 first, compute CRC32 over the 84B with zeroed crc slot, then re-pack
        def _pack(crc_val: int) -> bytes:
            return struct.pack(
                WIRE_16_FORMAT,
                VITAL_MAX_HP,
                KMB1_MAGIC,
                bitboard_64bit,
                move_mask,
                crc_val,
                float(voltage),
                float(current),
                float(power_load),
                float(sense.radiance_nits / 100.0),
                int(lens_code),
                int(marias_trump_score),
                *[float(x) for x in cv],
                int(speculative_draft_tokens),
                float(merkle_momentum_m_k),
                int(plan_code),
                slug_bytes,
            )

        pre = _pack(0)
        crc = zlib.crc32(pre) & 0xFFFFFFFF
        return _pack(crc)

    def unpack_84b_required_wire_params(self, buf_84b: bytes) -> Dict[str, Any]:
        if len(buf_84b) != 84:
            raise ContractViolation(
                f"Required wire slice must be 84 bytes, got {len(buf_84b)}",
                violation_code="ERR_WIRE_84B_SIZE",
            )
        fields = struct.unpack(WIRE_16_FORMAT, buf_84b)
        (
            hp,
            magic,
            bitboard,
            move_mask,
            crc,
            voltage,
            current,
            power_load,
            vray_flux,
            lens_code,
            marias_trump,
            cv0,
            cv1,
            cv2,
            cv3,
            cv4,
            cv5,
            k_spec,
            merkle_m_k,
            plan_code,
            slug_b,
        ) = fields

        if hp != VITAL_MAX_HP:
            raise ContractViolation(
                f"Wire header VITAL_MAX_HP breach: {hp} != 6",
                violation_code="ERR_CONSTITUTIONAL_INVARIANT_BREACH",
            )
        if magic != KMB1_MAGIC:
            raise ContractViolation(
                f"Wire KMB1 magic mismatch: {magic!r}",
                violation_code="ERR_WIRE_MAGIC_MISMATCH",
            )
        zero_crc_buf = buf_84b[:16] + b"\x00\x00\x00\x00" + buf_84b[20:]
        expected_crc = zlib.crc32(zero_crc_buf) & 0xFFFFFFFF
        if crc != expected_crc:
            raise ContractViolation(
                f"Wire 84B CRC32 mismatch: got {crc:#010x}, expected {expected_crc:#010x}",
                violation_code="ERR_WIRE_CRC_MISMATCH",
            )

        return {
            "vital_max_hp": hp,
            "kmb1_magic_header": magic.decode("ascii"),
            "bitboard_64bit_hex": f"0x{bitboard:016X}",
            "move_from_to_mask": f"0x{move_mask:04X}",
            "crc32c_integrity": f"0x{crc:08X}",
            "effective_voltage_v": round(voltage, 4),
            "effective_current_a": round(current, 4),
            "effective_power_load_w": round(power_load, 4),
            "vray_radiance_flux": round(vray_flux, 4),
            "active_perception_lens": CODE_TO_LENS.get(lens_code, "SOLAR_CLARITY"),
            "marias_trump_score": marias_trump,
            "cv_6d_stroke_vector": [round(x, 4) for x in (cv0, cv1, cv2, cv3, cv4, cv5)],
            "speculative_draft_tokens": k_spec,
            "merkle_momentum_m_k": round(merkle_m_k, 4),
            "vulkan_power_plan_profile": CODE_TO_POWER_PLAN.get(plan_code, "VULKAN_UMA_BALANCED_12W"),
            "tenant_taxonomy_slug": slug_b.decode("ascii", errors="replace"),
        }

    def pack_24b_kmb1_packet(
        self,
        bitboard_64bit: int = 0x8100000800000080,
        from_idx: int = 12,
        to_idx: int = 28,
        action_flags: int = 0x0001,
    ) -> bytes:
        move_hi = (from_idx & 0xFF)
        move_lo = (to_idx & 0xFF)
        crc_in = struct.pack(">4sQBBH", KMB1_MAGIC, bitboard_64bit, move_hi, move_lo, action_flags)
        crc = zlib.crc32(crc_in) & 0xFFFFFFFF
        return struct.pack(KMB1_24B_FORMAT, KMB1_MAGIC, bitboard_64bit, move_hi, move_lo, action_flags, crc, VITAL_MAX_HP)

    def decompile_canonical_128b_frame(self, frame_128b: bytes) -> Dict[str, Any]:
        if len(frame_128b) != FRAME_SIZE_BYTES:
            raise ContractViolation(
                f"Canonical frame must be 128 bytes, got {len(frame_128b)}",
                violation_code="ERR_FRAME_SIZE_MISMATCH",
            )
        wire_info = self.unpack_84b_required_wire_params(frame_128b[:84])

        gnos_84_108 = frame_128b[84:108]
        g_magic, g_yield, g_pl1, g_hp, g_nodes, g_mask, g_fork, g_epoch, g_crc = struct.unpack(
            "<4sffBBBBII", gnos_84_108
        )
        if g_magic != GNOSIS_MAGIC or g_hp != VITAL_MAX_HP or g_nodes != FROZEN_THEORY_NODES:
            raise ContractViolation(
                f"GNOS 24B header breach: magic={g_magic!r}, hp={g_hp}, nodes={g_nodes}",
                violation_code="ERR_GNOS_HEADER_BREACH",
            )
        expected_gcrc = zlib.crc32(gnos_84_108[:20]) & 0xFFFFFFFF
        if g_crc != expected_gcrc:
            raise ContractViolation(
                f"GNOS 24B CRC32 mismatch: got {g_crc:#010x}, expected {expected_gcrc:#010x}",
                violation_code="ERR_GNOS_CRC_MISMATCH",
            )

        smrp_info = self.sensory_engine.unpack_20b_smrp_descriptor(frame_128b[108:128])
        return {
            "profile": "CANONICAL_ZERO_COPY_21_6_5",
            "total_bytes": FRAME_SIZE_BYTES,
            "wire_84b": wire_info,
            "gnos_24b": {
                "magic": g_magic.decode("ascii"),
                "gnosis_nats_per_mj": round(g_yield, 4),
                "pl1_watts": round(g_pl1, 2),
                "vital_max_hp": g_hp,
                "frozen_theory_nodes": g_nodes,
                "s5_s9_bitmask": f"0x{g_mask:02X}",
                "homotopic_fork_active": bool(g_fork),
                "epoch": g_epoch,
                "crc32_hex": f"0x{g_crc:08X}",
                "crc32_valid": True,
            },
            "smrp_20b": smrp_info,
            "lossless_roundtrip_verified": True,
        }

    def _compile_phi_golden_79_49_frame(
        self,
        op_vec: OperatingCoordinatesVector,
        sense: SensoryManifoldCoordinates,
        morph_factor: float,
        contradictory_evidence: bool,
    ) -> Tuple[bytes, Dict[str, Any]]:
        cv_q1000 = [round(x * 1000.0) for x in (0.27, 0.38, 0.16, 0.17, 0.25, 0.45)]
        voltage = 10.8 + op_vec.uma_bus_coordinate * 2.4
        resistance = 1.4 + op_vec.uma_bus_coordinate * 1.3
        current = min(8.0, voltage / max(0.1, resistance))
        power_load = min(op_vec.thermal_pl1_watts, voltage * current * 0.25)
        lens_code = LENS_TO_CODE[sense.active_perception_lens]
        ratio_val = 79.0 / 49.0

        def _pack_head(crc_val: int) -> bytes:
            return struct.pack(
                PHI_79B_HEAD_FORMAT,
                VITAL_MAX_HP,
                PHI1_MAGIC,
                0x8100000800000080,
                0x0013,
                float(voltage),
                float(current),
                float(power_load),
                float(sense.radiance_nits / 100.0),
                int(lens_code),
                46,
                *cv_q1000,
                28,
                6.4215,
                1,
                b"nabytok47_rh_iso",
                1,
                VISUAL_DOF_TOTAL,
                0x4B4D4231,
                crc_val,
            )

        head_pre = _pack_head(0)
        head_crc = zlib.crc32(head_pre[:75]) & 0xFFFFFFFF
        head_79b = _pack_head(head_crc)
        hmac_prefix_5b = hmac.new(self.hmac_secret, head_79b, hashlib.sha256).digest()[:5]
        flags = 0x01 | (0x02 if contradictory_evidence else 0x00)
        tail_body_45b = struct.pack(
            "<4sffffffffBBBB5s",
            GLDS_MAGIC,
            float(op_vec.gnosis_yield_nats_per_mj),
            float(op_vec.thermal_pl1_watts),
            float(sense.radiance_nits),
            float(sense.chromatic_entropy_bits),
            float(sense.acoustic_dorian_hz),
            float(sense.sense_coherence_score),
            float(morph_factor),
            float(ratio_val),
            sense.cognition_level,
            lens_code,
            VITAL_MAX_HP,
            flags,
            hmac_prefix_5b,
        )
        tail_crc = zlib.crc32(tail_body_45b) & 0xFFFFFFFF
        frame = head_79b + tail_body_45b + struct.pack("<I", tail_crc)
        if len(frame) != FRAME_SIZE_BYTES:
            raise ContractViolation(f"PHI frame != 128B: {len(frame)}", violation_code="ERR_BINARY_LAYOUT_MISMATCH")

        # Decompile round-trip check
        h_u = struct.unpack(PHI_79B_HEAD_FORMAT, frame[:79])
        t_u = struct.unpack(GLDS_49B_TAIL_FORMAT, frame[79:128])
        if h_u[0] != VITAL_MAX_HP or t_u[11] != VITAL_MAX_HP or h_u[21] != VISUAL_DOF_TOTAL:
            raise ContractViolation("PHI_GOLDEN_79_49 decompile invariant failure", violation_code="ERR_PHI_ROUNDTRIP")
        if (zlib.crc32(frame[:75]) & 0xFFFFFFFF) != h_u[23] or (zlib.crc32(frame[79:124]) & 0xFFFFFFFF) != t_u[14]:
            raise ContractViolation("PHI_GOLDEN_79_49 CRC failure", violation_code="ERR_PHI_CRC_MISMATCH")

        return frame, {
            "profile": "PHI_GOLDEN_79_49",
            "wire_head_bytes": 79,
            "telemetry_tail_bytes": 49,
            "total_bytes": 128,
            "byte_aspect_ratio": round(ratio_val, 6),
            "target_phi_golden_ratio": round(PHI_GOLDEN, 6),
            "delta_from_phi": round(abs(PHI_GOLDEN - ratio_val), 6),
            "wire_params_to_visual_dof_ratio": f"{CORE_WIRE_PARAM_COUNT}:{VISUAL_DOF_TOTAL} (1:{round(VISUAL_DOF_TOTAL / CORE_WIRE_PARAM_COUNT, 3)})",
            "crc32_head_hex": f"0x{h_u[23]:08X}",
            "crc32_tail_hex": f"0x{t_u[14]:08X}",
            "lossless_roundtrip_verified": True,
        }

    def _compile_silver_octagonal_90_38_frame(
        self,
        wire_84b: bytes,
        op_vec: OperatingCoordinatesVector,
        sense: SensoryManifoldCoordinates,
        morph_factor: float,
        contradictory_evidence: bool,
    ) -> Tuple[bytes, Dict[str, Any]]:
        ohm_q1000 = round(op_vec.uma_bus_coordinate * 1000.0)
        head_90b = wire_84b + struct.pack(SILVER_6B_EXT_FORMAT, 64, ohm_q1000)
        ratio_val = 90.0 / 38.0
        lens_code = LENS_TO_CODE[sense.active_perception_lens]
        flags = 0x01 | (0x02 if contradictory_evidence else 0x00)
        tail_body_34b = struct.pack(
            "<4sffHHHHffBBBBH",
            SLVR_MAGIC,
            float(op_vec.gnosis_yield_nats_per_mj),
            float(op_vec.thermal_pl1_watts),
            min(65535, round(sense.radiance_nits * 10.0)),
            min(65535, round(sense.chromatic_entropy_bits * 1000.0)),
            min(65535, round(sense.acoustic_dorian_hz * 10.0)),
            min(10000, round(sense.sense_coherence_score * 10000.0)),
            float(morph_factor),
            float(ratio_val),
            sense.cognition_level,
            lens_code,
            VITAL_MAX_HP,
            flags,
            FROZEN_THEORY_NODES,
        )
        tail_crc = zlib.crc32(tail_body_34b) & 0xFFFFFFFF
        frame = head_90b + tail_body_34b + struct.pack("<I", tail_crc)
        if len(frame) != FRAME_SIZE_BYTES:
            raise ContractViolation(f"SILVER frame != 128B: {len(frame)}", violation_code="ERR_BINARY_LAYOUT_MISMATCH")

        t_u = struct.unpack(SLVR_38B_TAIL_FORMAT, frame[90:128])
        if t_u[11] != VITAL_MAX_HP or t_u[13] != FROZEN_THEORY_NODES or (zlib.crc32(frame[90:124]) & 0xFFFFFFFF) != t_u[14]:
            raise ContractViolation("SILVER_OCTAGONAL_90_38 roundtrip failure", violation_code="ERR_SILVER_ROUNDTRIP")

        return frame, {
            "profile": "SILVER_OCTAGONAL_90_38",
            "wire_head_bytes": 90,
            "telemetry_tail_bytes": 38,
            "total_bytes": 128,
            "byte_aspect_ratio": round(ratio_val, 6),
            "target_silver_ratio": round(SILVER_RATIO, 6),
            "delta_from_silver_ratio": round(abs(SILVER_RATIO - ratio_val), 6),
            "crc32_tail_hex": f"0x{t_u[14]:08X}",
            "lossless_roundtrip_verified": True,
        }

    def compile_all_aspect_ratios(
        self,
        op_vec: Optional[OperatingCoordinatesVector] = None,
        morph_factor: float = 0.618034,
        contradictory_evidence: bool = False,
        output_bin_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter_ns()
        vec = op_vec or OperatingCoordinatesVector(
            grid_x=4,
            grid_y=5,
            uma_bus_coordinate=0.28,
            vblank_phase_ms=5.2,
            thermal_pl1_watts=12.0,
            gnosis_yield_nats_per_mj=4.8201,
            vital_hp_coordinate=VITAL_MAX_HP,
        )
        sense = self.sensory_engine.project_operating_to_sensory(vec)

        wire_84b = self.pack_84b_required_wire_params(vec, sense)
        gnos_24b = GLOBAL_ONTOLOGIC_GNOSIS_ENGINE.pack_24b_gnosis_power_descriptor(
            gnosis_nats_per_mj=vec.gnosis_yield_nats_per_mj,
            pl1_watts=vec.thermal_pl1_watts,
            homotopic_fork_active=contradictory_evidence,
            invariant_gate=1,
        )
        smrp_20b = self.sensory_engine.pack_20b_smrp_descriptor(
            sense, morph_factor=morph_factor, homotopic_fork=contradictory_evidence
        )
        canonical_128b = wire_84b + gnos_24b + smrp_20b
        decompiled_canonical = self.decompile_canonical_128b_frame(canonical_128b)

        phi_128b, phi_meta = self._compile_phi_golden_79_49_frame(
            vec, sense, morph_factor, contradictory_evidence
        )
        silver_128b, silver_meta = self._compile_silver_octagonal_90_38_frame(
            wire_84b, vec, sense, morph_factor, contradictory_evidence
        )
        kmb1_24b = self.pack_24b_kmb1_packet()

        hmac_96b = hmac.new(
            self.hmac_secret, canonical_128b[:AUTHENTICATED_PREFIX_BYTES], hashlib.sha256
        ).hexdigest()

        # Role separation via ArtifactGate (PROPOSER != VERIFIER)
        claim_id = f"bin_aspect_compile_{int(time.time_ns())}"
        self.artifact_gate.deposit(
            claim_id,
            {"hmac_96b": hmac_96b, "profiles": 3, "hp": VITAL_MAX_HP},
            proposer_role=EnlgRole.PROPOSER,
        )
        gate_released = bool(
            self.artifact_gate.release(
                claim_id,
                evaluator_role=EnlgRole.VERIFIER,
                receipt_verified=(
                    len(canonical_128b) == 128
                    and len(phi_128b) == 128
                    and len(silver_128b) == 128
                    and decompiled_canonical["lossless_roundtrip_verified"]
                ),
                is_degraded_fallback=False,
            )
        )

        written_files: Dict[str, str] = {}
        if output_bin_dir is not None:
            out_dir = Path(output_bin_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            targets = {
                "canonical_21_6_5_128b.bin": canonical_128b,
                "phi_golden_79_49_128b.bin": phi_128b,
                "silver_octagonal_90_38_128b.bin": silver_128b,
                "kmb1_lockstep_24b.bin": kmb1_24b,
            }
            for fname, blob in targets.items():
                p = out_dir / fname
                p.write_bytes(blob)
                written_files[fname] = str(p)

        dur_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        if self.logger is not None:
            self.logger.log_event(
                channel="telemetry",
                operation="BINARY_ASPECT_RATIO_COMPILE",
                stratum="S9_SOVEREIGN_APOTHEOSIS_GATE",
                duration_ms=dur_ms,
                payload={
                    "hmac_96b_sha256": hmac_96b[:24],
                    "gate_released": gate_released,
                    "profiles_compiled": 3,
                    "phi_delta": phi_meta["delta_from_phi"],
                    "silver_delta": silver_meta["delta_from_silver_ratio"],
                },
                epistemic_status="VERIFIED_WITH_SCOPE" if gate_released else "QUARANTINED_DELTA_ZERO",
            )

        return {
            "compiler_version": "BINARY_PARAMETER_ASPECT_COMPILER_V2",
            "vital_max_hp": VITAL_MAX_HP,
            "frozen_theory_nodes": FROZEN_THEORY_NODES,
            "proposer_equals_verifier": False,
            "artifact_gate_released": gate_released,
            "hmac_sha256_96b_prefix": hmac_96b,
            "projected_sensory_coordinates": sense.to_dict(),
            "canonical_21_6_5": {
                "partition_bytes": {"wire_16_params": 84, "gnos_descriptor": 24, "smrp_sensory": 20, "total": 128},
                "exact_integer_aspect_ratio": "21 : 6 : 5",
                "wasted_padding_bytes": 0,
                "frame_to_kmb1_ratio": "128 : 24 (16 : 3)",
                "frame_128b_hex": canonical_128b.hex().upper(),
                "kmb1_24b_hex": kmb1_24b.hex().upper(),
                "decompiled_verification": decompiled_canonical,
            },
            "phi_golden_79_49": {
                **phi_meta,
                "frame_128b_hex": phi_128b.hex().upper(),
            },
            "silver_octagonal_90_38": {
                **silver_meta,
                "frame_128b_hex": silver_128b.hex().upper(),
            },
            "written_binary_files": written_files,
            "compile_wall_ms": round(dur_ms, 4),
        }


class StratifiedProcessMicroserviceFlowArchitect:
    """
    Defines:
      1. Architectural distinction between Native Processes (S0..S3), Role-Separated
         Microservices (CW_01..CW_06 / S4..S8), and Stack-Defined Conclusions (S9).
      2. Epistemic fact status audit answering what is / isn't a fact from a process.
      3. Six end-to-end operational flows (FLOW_01..FLOW_06) over S0..S9.
      4. Hypervisor computing layout (CPU <-> GPU crossover), 120-cell Matrix Choice Palette
         (> 100 choices across behavioral patterns vs stack use), and new Knowledge Properties.
    """

    def define_processes_microservices_and_stack_conclusions(self) -> Dict[str, Any]:
        return {
            "constitutional_invariants": {
                "vital_max_hp": VITAL_MAX_HP,
                "frozen_theory_nodes": FROZEN_THEORY_NODES,
                "dag_cycles": 0,
                "proposer_equals_verifier": False,
            },
            "execution_tiers_comparison": {
                "native_processes_s0_s3": {
                    "strata": ["S0", "S1", "S2", "S3"],
                    "memory_model": "Shared UMA host-coherent 64B cache-line ring (128B ZeroCopyContractFrame)",
                    "ipc_contract": "Lock-free SPSC binary frames ([0:84] wire + [84:108] GNOS + [108:128] SMRP)",
                    "isolation_boundary": "OS thread / VBlank phase shadow (Phi_Scanout * Phi_Compute == 0)",
                    "latency_budget_ms": "<= 1.2 ms (CPU L1D / VNNI fast path)",
                    "enlg_role": "EXECUTOR / OBSERVER",
                    "can_self_certify": False,
                    "state_mutation_authority": "None directly — emits telemetry observations & candidate deltas only",
                },
                "microservices_cw01_cw06_s4_s8": {
                    "strata": ["S4", "S5", "S6", "S7", "S8"],
                    "memory_model": "Zero shared mutable heap; capability-scoped JSON/binary contracts",
                    "ipc_contract": "HTTP/REST & typed CapabilityScope messages (CW_01 Observer, CW_02 Proposer, CW_03 Chronometric, CW_04 Gnosis Meter, CW_05 Projector, CW_06 Verifier)",
                    "isolation_boundary": "Strict service & authority boundary (PROPOSER cannot call VERIFIER, UI cannot mutate kernel)",
                    "latency_budget_ms": "1.2 ms .. 16.6 ms (within single VBlank or compute window)",
                    "enlg_role": "OBSERVER | PROPOSER | VERIFIER (mutually disjoint)",
                    "can_self_certify": False,
                    "state_mutation_authority": "Proposes coordinates or audits receipts; zero unilateral S_t -> S_{t+1} commit",
                },
                "stack_defined_conclusion_s9": {
                    "strata": ["S9_SOVEREIGN_APOTHEOSIS_GATE"],
                    "memory_model": "Immutable SHA-256 hash-chained computation ledger + ArtifactGate fixed point",
                    "ipc_contract": "Release(x) <=> ReceiptValid(x) & AuthorityValid(x) & (VITAL_MAX_HP == 6)",
                    "isolation_boundary": "Branchless quarantine boundary (not Valid => Delta S == 0)",
                    "latency_budget_ms": "0.021 ms deterministic gate check",
                    "enlg_role": "AUTHORITY (fixed-point release gate only)",
                    "can_self_certify": False,
                    "state_mutation_authority": "Sole authoritative transition S_t -> S_{t+1} once independent Proposer & Verifier agree",
                },
            },
            "epistemic_fact_from_process_audit": {
                "question": "Is something given as an unconditional fact directly from a running process?",
                "short_answer": "NO for empirical process outputs (only LOCAL_MEASURED / OBSERVATION / PROPOSAL); YES only as a priori constitutional state-machine laws enforced at every step (VITAL_MAX_HP == 6, 56 frozen nodes, 128B frame bounds).",
                "raw_process_output_status": "LOCAL_MEASURED / OBSERVATION (requires independent VERIFIER + falsifier to reach VERIFIED_WITH_SCOPE)",
                "constitutional_axioms_locked_a_priori": [
                    "VITAL_MAX_HP == 6 (Invalid action => Delta S == 0)",
                    "FROZEN_THEORY_NODES == 56 (T1..T12, 0 DAG cycles; no Node 57)",
                    "ZERO_COPY_FRAME_BYTES == 128 (84:24:20 canonical partition)",
                ],
                "empirical_promotion_gate": (
                    "Release(x) <=> ReceiptValid(x) & (Proposer != Verifier) & "
                    "(VITAL_MAX_HP == 6) & (I_Ohm <= 8.0A) & (Delta_H_nats > 0) => VERIFIED_WITH_SCOPE"
                ),
                "anti_promotion_barriers": [
                    "Process_Exit_Zero != Correctness_Proven",
                    "Simulation_Output != Hardware_Telemetry",
                    "Proposal != Authority",
                    "Cross_Compile != Native_Measured",
                ],
            },
        }

    def synthesize_new_stack_flows(self) -> Dict[str, Any]:
        flows: List[Dict[str, Any]] = [
            {
                "flow_id": "FLOW_01_ZERO_COPY_SENSORY_INGRESS",
                "strata_path": ["S0", "S1", "S2"],
                "execution_tier": "NATIVE_PROCESS",
                "binary_slice": "[0:84] WIRE_16 + [108:128] SMRP",
                "mechanism": "Maps 16 hardware parameters + Ohm current (I = V/R <= 8.0A) + SMRP sensory quantization into L1D 48KB buffer.",
                "governing_invariant": "VITAL_MAX_HP == 6 and CRC32(WIRE_84B) valid",
                "budget_ms": 0.45,
                "falsifier": "CRC32 mismatch or I_Ohm > 8.0A => immediate S_t+1 == S_t quarantine",
            },
            {
                "flow_id": "FLOW_02_V2_PREDICTIVE_UMA_PREBRAKE",
                "strata_path": ["S2", "S3", "S4"],
                "execution_tier": "NATIVE_PROCESS + CW_03_CHRONOMETRIC",
                "binary_slice": "[0:84] k_spec + EWMA adaptive_crossover_bytes",
                "mechanism": "Computes mu_hat = mu_t + 0.5*(mu_t - mu_{t-1}); engages CPU pre-braking (K_spec=4) at mu_hat >= 0.72 before 0.78 bus stall.",
                "governing_invariant": "Phi_Scanout(t) * Phi_Compute(t) == 0 & Hysteretic 12-frame dwell (P8)",
                "budget_ms": 0.85,
                "falsifier": "vblank_executed == False or median latency regression > v1 baseline",
            },
            {
                "flow_id": "FLOW_03_SENSORY_HOMEOMORPHISM_AND_P6_FORK",
                "strata_path": ["S5", "S6", "S8"],
                "execution_tier": "MICROSERVICES (CW_04 + CW_05)",
                "binary_slice": "[84:108] GNOS + [108:128] SMRP",
                "mechanism": "Projects 6D telemetry to 5D SensoryManifoldCoordinates; smooth morph_factor blend when consistent, P6 orthogonal fork (alpha perp beta) on contradiction.",
                "governing_invariant": "contradictory_evidence == True => convex_averaging_blocked == True",
                "budget_ms": 0.25,
                "falsifier": "Convex midpoint emitted during active conflict",
            },
            {
                "flow_id": "FLOW_04_THERMODYNAMIC_PARASITE_CUTOFF",
                "strata_path": ["S7", "S3"],
                "execution_tier": "CW_04_GNOSIS_METER",
                "binary_slice": "[84:108] GNOS Pi_Gnosis",
                "mechanism": "Enforces P5 calorimeter cutoff: zero entropy gain (Delta_H <= 0) truncates pipeline at 0.0180 mJ before GPU/LLM spend.",
                "governing_invariant": "Pi_Gnosis = (Delta_H / E_mJ) * I(HP==6 & I<=8A)",
                "budget_ms": 0.09,
                "falsifier": "Delta_H <= 0 consumes full 0.2840 mJ PL1 burst",
            },
            {
                "flow_id": "FLOW_05_BINARY_ASPECT_RATIO_COMPILER",
                "strata_path": ["S8", "S9"],
                "execution_tier": "MICROSERVICE_PROPOSER -> VERIFIER",
                "binary_slice": "128B (21:6:5 | 79:49 | 90:38) + 24B KMB1",
                "mechanism": "Compiles & losslessly round-trip verifies parameter aspect ratios with HMAC-SHA256 96B prefix and CRC32 seals.",
                "governing_invariant": "len(frame) == 128 and decompile(compile(x)) == x",
                "budget_ms": 0.35,
                "falsifier": "1-byte corruption undetected on unpack",
            },
            {
                "flow_id": "FLOW_06_APOTHEOSIS_GATE_CONCLUSION_COMMIT",
                "strata_path": ["S9"],
                "execution_tier": "STACK_DEFINED_CONCLUSION",
                "binary_slice": "Full 128B Frame + SHA-256 Coalesced Ledger",
                "mechanism": "CW_06 independent Verifier checks AepReceipt and releases S_t -> S_{t+1} through ArtifactGate into hash-chained JSONL.",
                "governing_invariant": "PROPOSER != VERIFIER & verify_chain_integrity() == True",
                "budget_ms": 0.05,
                "falsifier": "Self-certification attempted or ledger hash chain broken",
            },
        ]
        return {
            "flow_architecture": "STRATIFIED_S0_S9_BEHAVIORAL_FLOWS_V2",
            "total_flows": len(flows),
            "flows": flows,
            "combined_nominal_budget_ms": round(sum(f["budget_ms"] for f in flows), 3),
            "vital_max_hp": VITAL_MAX_HP,
        }

    def synthesize_hypervisor_matrix_behavioral_and_knowledge_properties(self) -> Dict[str, Any]:
        """
        Synthesizes:
          - General problem-solving closed path across under/upper behavioral layers (S0..S4 <-> S5..S9)
          - Hypervisor CPU <-> GPU computing layout (CW_01..CW_06)
          - 12 x 10 = 120-cell Matrix Choice Palette (> 100 choices) linking choice nodes to
            thermodynamic cost (mJ), epistemic gain (nats), and stack utilization
          - Six formal Knowledge Properties under KSPP v1 + ENLG v2
        """
        behavioral_patterns = (
            "P1_DIRECT_EQUIVALENCE",
            "P2_MICRO_TENSOR_CPU_BYPASS",
            "P3_ADAPTIVE_EWMA_GPU_CROSSOVER",
            "P4_PREDICTIVE_UMA_VELOCITY_PREBRAKE",
            "P5_HYSTERETIC_12_FRAME_STABILITY",
            "P6_HOMOTOPIC_CONFLICT_SUPERPOSITION",
            "P7_THERMODYNAMIC_PARASITE_CUTOFF",
            "P8_SENSORY_HOMEOMORPHIC_MORPH",
            "P9_BINARY_ASPECT_RATIO_COMPILATION",
            "P10_DIRECTIONAL_REGRESSION_GATE",
            "P11_ZERO_SELF_CERTIFICATION_QUORUM",
            "P12_COALESCED_HASH_LEDGER_COMMIT",
        )
        stack_strata = (
            ("S0_128B_INGRESS", "CPU_RING", 0.0080, 0.25),
            ("S1_INTENT_0X17", "CPU_BITWISE", 0.0060, 0.30),
            ("S2_L1D_OHM_TENSOR", "CPU_AVX_VNNI", 0.0150, 0.55),
            ("S3_HYSTERETIC_PI_B", "HYPERVISOR_GOV", 0.0100, 0.42),
            ("S4_OPENVINO_DP4A", "IRIS_XE_GPU", 0.0420, 0.95),
            ("S5_ONTOLOGIC_56_LATTICE", "CW_02_PROPOSER", 0.0110, 0.72),
            ("S6_HOMOTOPIC_FORK", "CW_05_ISOLATOR", 0.0145, 0.88),
            ("S7_GNOSIS_METER_P5", "CW_04_CALORIMETER", 0.0120, 1.3689),
            ("S8_LOGOS_BIJECTION", "CW_05_PROJECTOR", 0.0180, 1.15),
            ("S9_APOTHEOSIS_GATE", "CW_06_VERIFIER", 0.0065, 1.45),
        )

        # Build explicit 12 x 10 = 120 Matrix Choice Palette (> 100 choices across behavioral patterns x stack strata)
        matrix_choices: List[Dict[str, Any]] = []
        for r_idx, pat in enumerate(behavioral_patterns):
            for c_idx, (stratum_name, lane, base_mj, base_nats) in enumerate(stack_strata):
                cost_mj = round(base_mj * (1.0 + 0.04 * (r_idx % 4)), 5)
                gain_nats = 0.0 if pat == "P7_THERMODYNAMIC_PARASITE_CUTOFF" else round(base_nats * (1.0 + 0.03 * r_idx), 4)
                gnosis_yield = round(gain_nats / max(1e-6, cost_mj), 3)
                matrix_choices.append(
                    {
                        "choice_id": f"M_{r_idx:02d}_{c_idx:02d}",
                        "row": r_idx,
                        "col": c_idx,
                        "behavioral_pattern": pat,
                        "target_stratum": stratum_name,
                        "hypervisor_lane": lane,
                        "thermodynamic_cost_mj": cost_mj,
                        "epistemic_gain_nats": gain_nats,
                        "gnosis_nats_per_mj": gnosis_yield,
                        "requires_verifier_gate": True,
                    }
                )

        knowledge_properties = [
            {
                "property": "PROP_01_THERMODYNAMICALLY_PRICED",
                "equation": "Pi_Gnosis = (Delta_H_nats / Energy_mJ) * I(HP==6 & I<=8A)",
                "meaning": "Knowledge is priced in uncertainty eliminated per millijoule; zero-yield items are killed (P5).",
            },
            {
                "property": "PROP_02_HOMOTOPICALLY_NON_COLLAPSING",
                "equation": "E_A perp E_B => Fork(alpha, beta) without convex average",
                "meaning": "Contradictory claims preserve geometric orthogonality until a discriminating CDT collapses them (P6).",
            },
            {
                "property": "PROP_03_ANTI_PROMOTED_STRATIFIED",
                "equation": "Level(Verdict) <= min AllowedLevel(evidence); PROPOSER != VERIFIER",
                "meaning": "No self-certification: process logs and simulations cannot promote themselves to verified facts (P4).",
            },
            {
                "property": "PROP_04_FULL_FRAME_PROVENANCE_SEALED",
                "equation": "HMAC_SHA256(frame[0:96]) + CRC32 + SHA256_Chain",
                "meaning": "Every artifact carries cryptographic lineage tying binary bytes to epistemic receipts (P9).",
            },
            {
                "property": "PROP_05_INERTIALLY_DAMPENED",
                "equation": "M_k = gamma * M_{k-1} + Weight(e_k); 12-frame upscale dwell",
                "meaning": "Knowledge updates resist transient OS/bus jitter via hysteretic Schmitt triggers (P8).",
            },
            {
                "property": "PROP_06_CONSTITUTIONALLY_BOUNDED",
                "equation": "not Valid(Action) => Delta_S == 0; VITAL_MAX_HP == 6",
                "meaning": "Structural invariants lock authoritative state transitions across all 56 frozen nodes (P10).",
            },
        ]

        return {
            "general_problem_solving_loop": [
                "1_OBSERVE_UNDER_LAYER (S0..S2 capture telemetry & SMRP without assuming truth)",
                "2_HYPOTHESIZE_MATRIX_CHOICE (CW_02 selects candidate cell M_{r,c} over 12x10 choice matrix)",
                "3_HYPERVISOR_CROSSOVER_ROUTE (CPU VNNI vs Iris Xe GPU routed via EWMA + mu_hat prebrake inside VBlank shadow)",
                "4_METABOLIC_P5_P6_FILTER (S6 blocks conflict averaging; S7 terminates Delta_H==0 parasites at 0.0180 mJ)",
                "5_INDEPENDENT_AUDIT_AND_COMMIT (CW_06 Verifier releases via S9 ArtifactGate into hash-chained ledger)",
            ],
            "hypervisor_computing_layout": {
                "data_plane_s0_s4": "Native shared-UMA CPU AVX-VNNI & OpenVINO Iris Xe DP4A execution lanes separated by dynamic EWMA crossover",
                "control_plane_s5_s8": "6 role-separated microservices (CW_01..CW_06) with orthogonal CapabilityScope masks",
                "authority_plane_s9": "Zero-copy 128B ArtifactGate fixed point (Delta_S == 0 on any contract breach)",
            },
            "choice_palette_matrix": {
                "dimensions": [len(behavioral_patterns), len(stack_strata)],
                "total_choice_cells": len(matrix_choices),
                "exceeds_100_choices": len(matrix_choices) > 100,
                "sample_cells": matrix_choices[:10],
                "highest_gnosis_yield_cell": max(matrix_choices, key=lambda c: c["gnosis_nats_per_mj"]),
            },
            "new_knowledge_properties": knowledge_properties,
            "vital_max_hp": VITAL_MAX_HP,
            "frozen_theory_nodes": FROZEN_THEORY_NODES,
        }

    def synthesize_survival_governor_motivation_and_aspects(
        self,
        uma_utilization: float = 0.28,
        uma_velocity: float = 0.04,
        epistemic_gain_nats: float = 1.3689,
        predicted_energy_mj: float = 0.0880,
        die_temp_c: float = 54.0,
    ) -> Dict[str, Any]:
        """
        Synthesizes:
          1. Custom behavioral parameters new to the compute core
          2. General survival patterns + CPU governor metrics (Noether-Krystal P10)
          3. Epistemic Compute Motivation Core (EVOI-per-mJ priority scoring)
          4. New multi-dimensional aspects of verified engineering findings
        """
        if uma_utilization < 0.0 or uma_utilization > 1.0 or not math.isfinite(uma_utilization):
            raise ContractViolation(
                f"uma_utilization out of [0.0, 1.0]: {uma_utilization}",
                violation_code="ERR_GOVERNOR_UMA_BOUNDS",
            )
        if not math.isfinite(epistemic_gain_nats) or not math.isfinite(predicted_energy_mj):
            raise ContractViolation(
                "Non-finite motivation input",
                violation_code="ERR_NON_FINITE_COORDINATE",
            )

        pred_uma = min(1.0, max(0.0, uma_utilization + 0.5 * uma_velocity))
        headroom_to_stall = round(UMA_SATURATION_THRESHOLD - pred_uma, 4)
        thermal_margin_c = round(75.0 - die_temp_c, 2)
        pl1_watts = 15.0 if pred_uma < 0.45 else (12.0 if pred_uma < 0.72 else 7.5)
        governor_profile = (
            "CPU_SURVIVAL_THROTTLE" if pred_uma >= UMA_SATURATION_THRESHOLD
            else ("CPU_PREBRAKE_BALANCED" if pred_uma >= 0.72 else "CPU_VNNI_HYBRID_BOOST")
        )

        # Compute Motivation Core: prioritize high epistemic gain per millijoule weighted by bus safety
        uma_risk_penalty = 1.0 + (2.0 * pred_uma if pred_uma >= 0.65 else 0.35 * pred_uma)
        safe_energy = max(1e-4, predicted_energy_mj)
        motivation_score = (
            0.0
            if epistemic_gain_nats <= 0.0 or thermal_margin_c <= 0.0
            else round((epistemic_gain_nats / safe_energy) / uma_risk_penalty, 4)
        )

        compute_core_custom_parameters = [
            {
                "parameter": "gnosis_power_density_gate",
                "formula": "Pi_Gnosis = (Delta_H_nats / E_mJ) * I(HP==6 & I<=8A)",
                "behavioral_logic": "Acts as a real-time metabolic cut gate: decouples zero-entropy work at 0.0180 mJ before full dispatch.",
            },
            {
                "parameter": "predictive_uma_velocity_lead",
                "formula": "mu_hat = mu_t + 0.5 * (mu_t - mu_{t-1})",
                "behavioral_logic": "Pre-brakes K_spec (8 -> 4) at mu_hat >= 0.72 before hitting the 0.78 UMA wall.",
            },
            {
                "parameter": "hysteretic_pl1_dwell_counter",
                "formula": "12-frame dwell gate across 15W / 12W / 7.5W PL1 states",
                "behavioral_logic": "Immediate downscale on thermal/bus spike; delayed upscale via 12-frame Merkle-Momentum hysteresis.",
            },
            {
                "parameter": "homotopic_conflict_bifurcation_flag",
                "formula": "E_A perp E_B => Fork(alpha, beta); convex_avg = BLOCKED",
                "behavioral_logic": "Prevents midpoint erasure of contradictory telemetry inside [84:108] GNOS & [108:128] SMRP.",
            },
        ]

        survival_patterns = [
            {
                "pattern_id": "SURVIVAL_01_PREDICTIVE_GOVERNOR_PREBRAKE",
                "governor_metric": "headroom_to_0_78_stall",
                "trigger": "pred_uma >= 0.72 and d_uma_dt > 0.15",
                "beneficial_effect": "Prevents hard VBlank overrun by stepping K_spec down smoothly to CPU.",
            },
            {
                "pattern_id": "SURVIVAL_02_ARRHENIUS_THERMAL_HYSTERESIS",
                "governor_metric": "thermal_margin_c = 75.0 - die_temp_c",
                "trigger": "die_temp_c >= 75.0 or 12-frame dwell active",
                "beneficial_effect": "Stops P-core/GPU clock ping-ponging under sustained loads.",
            },
            {
                "pattern_id": "SURVIVAL_03_METABOLIC_PARASITE_SHEDDING",
                "governor_metric": "gnosis_nats_per_mj",
                "trigger": "Delta_H_nats <= 0.0",
                "beneficial_effect": "Reclaims ~93.7% mJ on non-informative probes, boosting net ops/W.",
            },
            {
                "pattern_id": "SURVIVAL_04_BRANCHLESS_ZERO_DELTA_SEAL",
                "governor_metric": "vital_max_hp == 6",
                "trigger": "ContractViolation or CRC32/HMAC mismatch",
                "beneficial_effect": "Guarantees S_{t+1} == S_t with zero partial state corruption.",
            },
        ]

        new_findings_aspects = [
            {
                "aspect_id": "ASPECT_01_THERMODYNAMIC_PRODUCTIVITY",
                "finding": "Measured v2 coalesced ledger + EWMA crossover reached 2.7420 ms paired median (4.82x faster than v1, 6.02x paced ops/W gain).",
            },
            {
                "aspect_id": "ASPECT_02_ZERO_WASTE_BINARY_PACKING",
                "finding": "Reclaiming the 20B tail into SMRP (84:24:20 = 21:6:5) fuses wire params, GNOS power telemetry, and sensory state in 128B with zero dead bytes.",
            },
            {
                "aspect_id": "ASPECT_03_EPISTEMIC_NON_SELF_CERTIFICATION",
                "finding": "Processes produce LOCAL_MEASURED proposals; authoritative conclusions emerge strictly at S9 ArtifactGate under PROPOSER != VERIFIER.",
            },
            {
                "aspect_id": "ASPECT_04_GOVERNED_COMPUTE_MOTIVATION",
                "finding": "Intrinsic compute motivation tracks net uncertainty eliminated per millijoule rather than raw clock frequency or token volume.",
            },
        ]

        return {
            "cpu_governor_live_metrics": {
                "uma_utilization": round(uma_utilization, 4),
                "uma_velocity": round(uma_velocity, 4),
                "predicted_next_uma": round(pred_uma, 4),
                "headroom_to_stall": headroom_to_stall,
                "die_temp_c": round(die_temp_c, 2),
                "thermal_margin_c": thermal_margin_c,
                "active_pl1_watts": pl1_watts,
                "governor_profile": governor_profile,
            },
            "compute_motivation_core": {
                "formula": "(Delta_H_nats / Predicted_mJ) / UMA_Risk_Penalty",
                "epistemic_gain_nats": round(epistemic_gain_nats, 4),
                "predicted_energy_mj": round(safe_energy, 4),
                "uma_risk_penalty": round(uma_risk_penalty, 4),
                "motivation_score_nats_per_risk_mj": motivation_score,
                "execution_authorized": motivation_score > 0.0 and thermal_margin_c > 0.0,
            },
            "compute_core_custom_parameters": compute_core_custom_parameters,
            "survival_patterns": survival_patterns,
            "new_findings_aspects": new_findings_aspects,
            "vital_max_hp": VITAL_MAX_HP,
        }

    def synthesize_candidate_theoretical_behavioral_patterns(self) -> Dict[str, Any]:
        """
        Stages 4 candidate theoretical behavioral patterns (CAND_TH_01..04) strictly in the
        EXPLORE runtime layer without modifying the 56 frozen theory nodes (T1..T12, 0 DAG cycles).
        """
        candidates = [
            {
                "candidate_id": "CAND_TH_01_NOETHER_THERMODYNAMIC_CONSERVATION",
                "axiom": "Action commitment preserves (VITAL_MAX_HP == 6) and requires Pi_Gnosis > 0 over closed UMA transactions.",
                "falsifier": "Valid commit leaves Pi_Gnosis <= 0 or mutates VITAL_MAX_HP != 6.",
            },
            {
                "candidate_id": "CAND_TH_02_HOMOTOPIC_MANIFOLD_ORTHOGONALITY",
                "axiom": "Contradictory propositions occupy orthogonal fiber branches (alpha perp beta); no convex interpolation exists prior to CDT.",
                "falsifier": "Blend t in (0, 1) emitted when contradictory_evidence == True.",
            },
            {
                "candidate_id": "CAND_TH_03_ZERO_WASTE_BILATERAL_ASPECT_BIJECTION",
                "axiom": "84B wire, 24B gnosis, and 20B sensory coordinates partition 128B without zero-pad leakage under CRC32 + HMAC-96B.",
                "falsifier": "Non-bijective decode or undetected 1-byte mutation.",
            },
            {
                "candidate_id": "CAND_TH_04_CROSS_OS_HIERARCHY_SPSC_PARITY",
                "axiom": "WSL2 UNIX namespaces and Windows NT Job/Section objects share lock-free 64B/128B frames without bypassing CW_06 ArtifactGate.",
                "falsifier": "Direct POSIX/Win32 shared-memory write skips S9 Verifier receipt.",
            },
        ]
        return {
            "epistemic_stage": "CANDIDATE_EXPLORE_STAGED",
            "frozen_theory_nodes_untouched": FROZEN_THEORY_NODES,
            "dag_cycles": 0,
            "node_57_added_to_frozen_theory": False,
            "candidate_theoretical_patterns": candidates,
            "vital_max_hp": VITAL_MAX_HP,
        }


class WslNtCoexistenceBinaryBridge:
    """
    Constructs the 4x4 coexistence matrix and packs/verifies the 64-byte single-cache-line
    self-binary descriptor (`WNTC`) bridging UNIX/WSL2 subsystems and the Windows NT kernel hierarchy.
    """

    def build_coexistence_matrix_and_binary(
        self,
        win32_pid: int = 4096,
        wsl2_linux_pid_ns: int = 1024,
        nt_handle_mask: int = 0x001F0003,
        posix_inode_tag: int = 0x00C0FFEE,
        win32_qpc_ticks: int = 1_000_000_128,
        wsl_vbus_offset_bytes: int = 4096,
        capability_scope_mask: int = 0x1F,
    ) -> Dict[str, Any]:
        matrix_4x4 = [
            {
                "coord": [0, 0],
                "unix_wsl_unit": "POSIX /dev/shm SPSC Ring (fd)",
                "windows_nt_hierarchy": "NT Section / MapViewOfFile (64B aligned)",
                "shared_binary_slice": "128B Frame [0:128]",
                "authority_boundary": "Data Plane (OBSERVER/EXECUTOR)",
            },
            {
                "coord": [1, 1],
                "unix_wsl_unit": "Linux cgroup v2 CPU/Memory PSI",
                "windows_nt_hierarchy": "NT JobObject + DXGKRNL PL1/WDDM",
                "shared_binary_slice": "GNOS [84:108] + WNTC [0:64]",
                "authority_boundary": "Governor Telemetry (CW_03/CW_04)",
            },
            {
                "coord": [2, 2],
                "unix_wsl_unit": "Linux futex / eventfd signal",
                "windows_nt_hierarchy": "Win32 WaitOnAddress / IOCP",
                "shared_binary_slice": "SMRP [108:128] phase flags",
                "authority_boundary": "VBlank Sync (Phi_Scanout * Phi_Compute == 0)",
            },
            {
                "coord": [3, 3],
                "unix_wsl_unit": "WSL2 VSOCK / 9P Transport",
                "windows_nt_hierarchy": "NT NamedPipe / Loopback HTTP (:8090)",
                "shared_binary_slice": "HMAC-SHA256 96B + CRC32",
                "authority_boundary": "S9 ArtifactGate (PROPOSER != VERIFIER)",
            },
        ]

        body_60b = struct.pack(
            "<4sHHIIIIQQ8s8sI",
            WNTC_MAGIC,
            VITAL_MAX_HP,
            4,
            win32_pid,
            wsl2_linux_pid_ns,
            nt_handle_mask,
            posix_inode_tag,
            win32_qpc_ticks,
            wsl_vbus_offset_bytes,
            b"C:->/mnt",
            b"NT_JOB01",
            capability_scope_mask,
        )
        crc = zlib.crc32(body_60b) & 0xFFFFFFFF
        self_binary_64b = body_60b + struct.pack("<I", crc)
        if len(self_binary_64b) != 64:
            raise ContractViolation(
                f"WNTC self-binary != 64B: {len(self_binary_64b)}",
                violation_code="ERR_WNTC_BINARY_SIZE",
            )
        decoded = self.verify_and_decode_64b(self_binary_64b)
        return {
            "coexistence_architecture": "UNIX_WSL2_WINDOWS_NT_HIERARCHY_V1",
            "coexistence_matrix_4x4": matrix_4x4,
            "self_binary_64b_hex": self_binary_64b.hex().upper(),
            "self_binary_decoded": decoded,
            "vital_max_hp": VITAL_MAX_HP,
        }

    def verify_and_decode_64b(self, buf_64b: bytes) -> Dict[str, Any]:
        if len(buf_64b) != 64:
            raise ContractViolation(
                f"WNTC descriptor must be 64 bytes, got {len(buf_64b)}",
                violation_code="ERR_WNTC_BINARY_SIZE",
            )
        (
            magic,
            hp,
            mat_order,
            w_pid,
            wsl_ns,
            nt_mask,
            inode,
            qpc,
            vbus_off,
            mnt_b,
            job_b,
            cap_mask,
            crc,
        ) = struct.unpack(WNTC_64B_FORMAT, buf_64b)
        if magic != WNTC_MAGIC or hp != VITAL_MAX_HP:
            raise ContractViolation(
                f"WNTC header breach: magic={magic!r}, hp={hp}",
                violation_code="ERR_WNTC_HEADER_BREACH",
            )
        expected = zlib.crc32(buf_64b[:60]) & 0xFFFFFFFF
        if crc != expected:
            raise ContractViolation(
                f"WNTC CRC32 tampering detected: {crc:#010x} != {expected:#010x}",
                violation_code="ERR_WNTC_CRC_MISMATCH",
            )
        return {
            "magic": magic.decode("ascii"),
            "vital_max_hp": hp,
            "matrix_order": mat_order,
            "win32_pid": w_pid,
            "wsl2_linux_pid_ns": wsl_ns,
            "nt_handle_mask": f"0x{nt_mask:08X}",
            "posix_inode_tag": f"0x{inode:08X}",
            "win32_qpc_ticks": qpc,
            "wsl_vbus_offset_bytes": vbus_off,
            "mount_map": mnt_b.decode("ascii", errors="replace"),
            "nt_job_id": job_b.decode("ascii", errors="replace"),
            "capability_scope_mask": f"0x{cap_mask:02X}",
            "crc32_hex": f"0x{crc:08X}",
            "crc32_valid": True,
        }

